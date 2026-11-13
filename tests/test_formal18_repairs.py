"""Local regressions for the recovered run; HTTP traffic uses MockTransport only."""
import csv
import json

import httpx
import pytest

from conftest import MockProvider
from relay.checkpoint import CheckpointStore, compatible_history, config_id
from relay.evaluate import FIELDS
from relay.evaluation_questions import QUESTIONS
from relay.experiment_review import compare, packet
from relay.grounded_output import render_draft
from relay.pipeline import RAGPipeline
from relay.providers.base import ProviderError, RateLimitError
from relay.providers.groq_provider import GroqProvider
from relay.providers.scheduler import RequestScheduler
from relay.verification import verify_answer


PASS = {'decision': 'return', 'unsupported_claims': [], 'reason': 'Supported by the cited passage.'}


def json_response(request, value):
    return httpx.Response(200, request=request, json={
        'choices': [{'message': {'content': json.dumps(value)}, 'finish_reason': 'stop'}],
        'usage': {'prompt_tokens': 20, 'completion_tokens': 8}})


@pytest.mark.parametrize('question_id,architecture', [
    ('H6', 'fast'), ('X1', 'adaptive'), ('M1', 'verified'),
    ('T1', 'adaptive'), ('M1', 'adaptive'), ('M3', 'fast'),
])
@pytest.mark.asyncio
async def test_six_uncited_draft_cases_use_valid_alias_contract(
        monkeypatch, question_id, architecture, config, retriever):
    monkeypatch.setenv('GROQ_API_KEY', 'test-only')
    dispatched = []
    async def handler(request):
        body = json.loads(request.content)
        dispatched.append(body)
        schema = body['response_format']['json_schema']
        if schema['name'] == 'retrieval_plan':
            return json_response(request, {'relevant_papers': [], 'subquestions': [], 'search_queries': []})
        if schema['name'] == 'answer_verification':
            return json_response(request, PASS)
        aliases = schema['schema']['properties']['claims']['items']['properties']['sources']['items']['enum']
        assert aliases and all(a.startswith('S') for a in aliases)
        return json_response(request, {'decision': 'answer', 'claims': [
            {'text': 'A supported fixture claim', 'sources': [aliases[0]]}]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = GroqProvider(config.provider, client=client)
        question = next(q.question for q in QUESTIONS if q.question_id == question_id)
        result = await RAGPipeline(architecture, retriever, provider, config).run(question)
    assert result.status == 'ok'
    assert result.metrics['citation_validation']['valid'] is True
    assert '[S1]' in result.answer
    assert result.metrics['draft_answer'] == result.answer
    assert result.metrics['provider_ttft_ms'] is None  # Buffered strict JSON has no measured TTFT.
    expected_calls = 1 if architecture == 'fast' else 3 if question_id in {'X1', 'T1'} else 2
    assert len(dispatched) == expected_calls


@pytest.mark.parametrize('claim', [
    {'text': 'claim', 'sources': []}, {'text': 'claim', 'sources': ['S99']},
    {'text': '', 'sources': ['S1']},
])
def test_grounded_contract_cannot_invent_citations(claim):
    with pytest.raises(ProviderError):
        render_draft({'decision': 'answer', 'claims': [claim]}, ['S1'])


def test_grounded_contract_preserves_explicit_refusal():
    assert render_draft({'decision': 'refuse', 'claims': []}, ['S1']).startswith('The provided papers')


@pytest.mark.asyncio
@pytest.mark.parametrize('second_verdict,expected', [(PASS, 'ok'), (
    {'decision': 'refuse', 'unsupported_claims': ['still unsupported'], 'reason': 'No support'},
    'revision_verification_failed')])
async def test_h4_refuse_with_claims_gets_one_correction_and_reverification(
        second_verdict, expected, retriever, config):
    refusal = {'decision': 'refuse', 'unsupported_claims': ['no detail is lost'],
               'reason': 'The lossless guarantee is unsupported.'}
    provider = MockProvider(['H-Net compression loses no detail [S1]', refusal,
                             'H-Net uses hierarchical chunking [S1]', second_verdict])
    question = next(q.question for q in QUESTIONS if q.question_id == 'H4')
    from relay.architectures.verified import VerifiedRAG
    from relay.instrumentation import RequestTrace
    result = await VerifiedRAG(retriever, provider, config)._answer_and_verify(
        question, retriever.retrieve('dynamic chunking'), RequestTrace('h4', 'verified'))
    assert result.status == expected
    assert len(provider.calls) == 4
    assert provider.kwargs_seen[2]['citation_aliases'] == ['S1']
    assert 'no detail is lost' in provider.messages_seen[2][1]['content']
    assert result.metrics['revised_draft'] == 'H-Net uses hierarchical chunking [S1]'
    if expected != 'ok':
        assert result.answer.startswith('The provided papers')


@pytest.mark.asyncio
async def test_m3_schema_400_gets_one_strict_retry_and_counts_all_calls(monkeypatch, config, retriever):
    monkeypatch.setenv('GROQ_API_KEY', 'test-only')
    seen = []
    verifier_calls = 0
    async def handler(request):
        nonlocal verifier_calls
        body = json.loads(request.content)
        seen.append(body)
        name = body['response_format']['json_schema']['name']
        if name == 'retrieval_plan':
            return json_response(request, {'relevant_papers': [], 'subquestions': [], 'search_queries': []})
        if name == 'grounded_answer':
            return json_response(request, {'decision': 'answer', 'claims': [
                {'text': 'Moshi supports dialogue', 'sources': ['S1']}]})
        verifier_calls += 1
        if verifier_calls == 1:
            return httpx.Response(400, request=request, json={'error': {
                'code': 'json_validate_failed', 'message': 'missing unsupported_claims',
                'failed_generation': '{"decision":"revise","reason":"long reason"}'}})
        return json_response(request, PASS)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = GroqProvider(config.provider, client=client)
        question = next(q.question for q in QUESTIONS if q.question_id == 'M3')
        result = await RAGPipeline('adaptive', retriever, provider, config).run(question)
    assert result.status == 'ok'
    assert len(seen) == result.metrics['total_model_calls'] == 4
    assert len(provider.scheduler._requests) == 4
    assert result.metrics['model_calls_by_stage']['verification_format_attempt'] == 1
    assert result.metrics['total_input_tokens'] is None  # HTTP 400 usage is unknown, not zero.
    assert [r['max_completion_tokens'] for r in seen[-2:]] == [640, 1024]
    assert all(r['response_format']['json_schema']['strict'] for r in seen)


@pytest.mark.asyncio
async def test_invalid_verifier_fields_never_default_to_acceptance(retriever):
    provider = MockProvider([{'decision': 'return'}, {'decision': 'return'}])
    with pytest.raises(ProviderError, match='schema validation') as failure:
        await verify_answer(provider, 'attention', 'claim [S1]', retriever.retrieve('attention'))
    assert len(provider.calls) == 2
    assert len(failure.value.diagnostics['prior_attempts']) == 1


@pytest.mark.asyncio
async def test_valid_second_format_attempt_keeps_first_usage(retriever):
    from relay.instrumentation import RequestTrace
    provider = MockProvider([{'decision': 'return'}, PASS])
    _, response = await verify_answer(provider, 'attention', 'claim [S1]', retriever.retrieve('attention'))
    trace = RequestTrace('id', 'verified')
    trace.add_call('verification', response)
    assert trace.metrics()['total_model_calls'] == 2
    assert trace.metrics()['total_input_tokens'] == 16
    assert trace.metrics()['total_output_tokens'] == 8


@pytest.mark.asyncio
async def test_verifier_daily_token_429_is_not_a_format_retry(monkeypatch, config, retriever, tmp_path):
    monkeypatch.setenv('GROQ_API_KEY', 'test-only')
    config.provider.scheduler.state_path = tmp_path / 'quota.json'
    requests = []
    async def handler(request):
        requests.append(request)
        return httpx.Response(429, request=request, headers={
            'retry-after': '526', 'x-ratelimit-remaining-tokens': '5838',
            'x-ratelimit-remaining-requests': '924'}, json={'error': {'message':
                'tokens per day (TPD): Limit 200000, Used 199716, Requested 1500'}})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = GroqProvider(config.provider, client=client)
        with pytest.raises(RateLimitError) as error:
            await verify_answer(provider, 'attention', 'claim [S1]', retriever.retrieve('attention'))
        assert error.value.rate_limit_type == 'provider_tpd'
        assert len(requests) == 1
        with pytest.raises(RateLimitError):
            await RequestScheduler(config.provider.scheduler).acquire(1, 1)
    assert len(requests) == 1


def write_rows(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def test_repair_resume_preserves_success_and_failed_raw_history(tmp_path):
    source, output = tmp_path / 'old.csv', tmp_path / 'repair.csv'
    old = {'evaluation_configuration': {'model': 'test', 'rag_logic_sha256': 'old'}}
    new = {'evaluation_configuration': {'model': 'test', 'rag_logic_sha256': 'new'},
           'repair_campaign': True, 'selection_eligible': False}
    source.with_suffix('.manifest.json').write_text(json.dumps(old))
    write_rows(source, [
        {'request_id': 'good', 'question_id': 'M1', 'architecture': 'fast', 'status': 'ok'},
        {'request_id': 'failed', 'question_id': 'H4', 'architecture': 'verified', 'status': 'verification_failed'}])
    original = source.read_bytes()
    store = CheckpointStore(output, FIELDS, new)
    store.import_history(source, repair=True)
    assert len(store.completed()) == 1
    assert store.record({'request_id': 'corrected', 'question_id': 'H4',
                         'architecture': 'verified', 'status': 'ok'})
    restarted = CheckpointStore(output, FIELDS, new)
    restarted.import_history(source, repair=True)
    assert len(restarted.completed()) == 2
    assert len(restarted.rows) == 2
    assert len(restarted.journal.read_text().splitlines()) == 3
    assert source.read_bytes() == original
    assert next(r for r in restarted.rows.values() if r['request_id'] == 'good')[
        'origin_evaluation_config_id'] == config_id(old)
    with pytest.raises(ValueError, match='settings'):
        compatible_history(source, dict(new, evaluation_configuration={'model': 'different'}), repair=True)


def test_repair_history_cannot_be_relabelled_as_fair_comparison(tmp_path):
    path = tmp_path / 'repair.csv'
    manifest = {'evaluation_configuration': {'model': 'test'}, 'repair_campaign': True}
    path.with_suffix('.manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='selection-eligible'):
        compatible_history(path, {'evaluation_configuration': {'model': 'test'}})


def test_packet_is_blinded_separates_failures_and_preserves_grades(tmp_path):
    source, output = tmp_path / 'answers.csv', tmp_path / 'review'
    write_rows(source, [
        {'request_id': 'done', 'question_id': 'T1', 'question': 'attention?', 'architecture': 'fast',
         'status': 'ok', 'answer': 'claim [S1]'},
        {'request_id': 'refused', 'question_id': 'H6', 'question': 'H-Net?', 'architecture': 'fast',
         'status': 'citation_validation_failed', 'answer': 'The provided papers do not establish an answer.'},
        {'request_id': 'quota', 'question_id': 'M3', 'architecture': 'adaptive', 'status': 'rate_limited'}])
    packet([source], output)
    reviews = list(csv.DictReader((output / 'grades.csv').open()))
    assert len(reviews) == 2 and all(r['accuracy'] == '' for r in reviews)
    reviews[0].update(accuracy='0', reviewer='Independent human', rationale='Incorrect')
    with (output / 'grades.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=reviews[0].keys())
        writer.writeheader()
        writer.writerows(reviews)
    packet([source], output)
    assert list(csv.DictReader((output / 'grades.csv').open()))[0]['accuracy'] == '0'
    dossier = json.loads((output / 'blind_answers.json').read_text())
    assert all('status' not in r and 'architecture' not in r for r in dossier)
    assert len(list(csv.DictReader((output / 'infrastructure_failures.csv').open()))) == 1


@pytest.mark.parametrize('repair,missing', [(False, True), (True, False), (False, False)])
def test_selection_requires_complete_unmixed_graded_matrix(tmp_path, repair, missing):
    source, output = tmp_path / 'answers.csv', tmp_path / 'review'
    manifest = {'questions': ['T1', 'M1'], 'architectures': ['fast', 'verified', 'adaptive'],
                'repair_campaign': repair, 'evaluation_configuration': {'model': 'test'}}
    rows = [{'request_id': f'{q}-{a}', 'question_id': q, 'question': q, 'architecture': a,
             'status': 'ok', 'answer': 'claim', 'evaluation_config_id': config_id(manifest)}
            for a in manifest['architectures'] for q in (['T1'] if missing else manifest['questions'])]
    write_rows(source, rows)
    source.with_suffix('.manifest.json').write_text(json.dumps(manifest))
    packet([source], output)
    reviews = list(csv.DictReader((output / 'grades.csv').open()))
    for r in reviews:
        r.update(accuracy='1', citation_grounding='1', reviewer='Independent',
                 paper_pages_checked='Original PDF p.4', rationale='Checked all claims')
    with (output / 'grades.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=reviews[0].keys())
        writer.writeheader()
        writer.writerows(reviews)
    result = compare([source], output / 'grades.csv', tmp_path / 'selection', minimum_questions=1)
    assert result['complete'] == (not repair and not missing)
    assert (result['winner'] is None) == (repair or missing)

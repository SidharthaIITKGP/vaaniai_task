from __future__ import annotations

import csv
import json
from pathlib import Path

import httpx
import pytest

from relay.checkpoint import CheckpointStore, config_id
from relay.evaluate import FIELDS, parser, run
from relay.providers.base import RateLimitError
from relay.providers.groq_provider import GroqProvider, _classify_429
from relay.providers.scheduler import RequestScheduler, _duration_seconds
from relay.config import ProviderConfig, SchedulerConfig


@pytest.mark.parametrize('body, expected', [
    ('Rate limit reached for tokens per day (TPD)', 'provider_tpd'),
    ('Rate limit reached for requests per day (RPD)', 'provider_rpd'),
    ('tokens per minute (TPM)', 'provider_tpm'),
    ('requests per minute (RPM)', 'provider_rpm'),
    ('input tokens per minute (ITPM)', 'provider_itpm'),
    ('output tokens per minute (OTPM)', 'provider_otpm'),
    ('Rate limit exceeded', 'provider_429'),
])
def test_rate_limit_classification_does_not_guess(body, expected):
    assert _classify_429(json.dumps({'error': {'message': body}})) == expected


@pytest.mark.asyncio
async def test_long_cooldown_blocks_later_calls_and_survives_restart(monkeypatch, tmp_path):
    monkeypatch.setenv('GROQ_API_KEY', 'test-only')
    cfg = ProviderConfig()
    cfg.scheduler.state_path = tmp_path / 'quota.json'
    cfg.scheduler.max_provider_cooldown_seconds = 60
    calls = []
    async def handler(request):
        calls.append(request)
        return httpx.Response(429, json={'error': {'message': 'tokens per day (TPD) limit reached'}},
            headers={'retry-after': '1077', 'x-ratelimit-remaining-requests': '900'}, request=request)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = GroqProvider(cfg, client=client)
        with pytest.raises(RateLimitError) as first:
            await provider.generate([{'role': 'user', 'content': 'test'}])
        assert first.value.rate_limit_type == 'provider_tpd'
        assert first.value.response_headers['retry-after'] == '1077'
        assert 'tokens per day' in first.value.error_body
        with pytest.raises(RateLimitError):
            await provider.generate([{'role': 'user', 'content': 'test again'}])
        restarted = RequestScheduler(cfg.scheduler)
        with pytest.raises(RateLimitError):
            await restarted.acquire(10, 10)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_daily_header_exhaustion_stops_without_retry():
    scheduler = RequestScheduler(SchedulerConfig(max_provider_cooldown_seconds=60))
    await scheduler.update_from_headers({'x-ratelimit-remaining-requests': '0',
                                          'x-ratelimit-reset-requests': '1h2m3.5s'})
    with pytest.raises(RateLimitError) as error:
        await scheduler.acquire(10, 10)
    assert error.value.rate_limit_type == 'rpd_exhausted'
    assert scheduler.stop_reason['retry_not_before_epoch'] is not None
    assert _duration_seconds('1h2m3.5s') == 3723.5


def test_checkpoint_retries_preserve_raw_attempts_without_duplicate_evaluations(tmp_path):
    path = tmp_path / 'recovered.csv'
    manifest = {'evaluation_configuration': {'model': 'test', 'top_k': 5}}
    store = CheckpointStore(path, FIELDS, manifest)
    base = {'question_id': 'T1', 'architecture': 'fast', 'execution_order': 1}
    store.record(dict(base, request_id='failed', status='rate_limited'))
    assert not store.completed()
    store.record(dict(base, request_id='completed', status='ok', execution_order=2))
    assert len(store.completed()) == 1
    assert not store.record(dict(base, request_id='duplicate-success', status='ok'))
    assert len(list(csv.DictReader(path.open()))) == 1
    assert len(store.journal.read_text().splitlines()) == 2
    path.unlink()  # Simulate lost derived snapshot; raw attempts still recover.
    restarted = CheckpointStore(path, FIELDS, manifest)
    restarted.snapshot()
    assert len(list(csv.DictReader(path.open()))) == 1
    assert restarted.completed() == {('T1', 'fast', config_id(manifest))}


def test_configuration_changes_are_not_reused(tmp_path):
    path = tmp_path / 'runs.csv'
    store = CheckpointStore(path, FIELDS, {'evaluation_configuration': {'top_k': 5}})
    store.record({'question_id': 'T1', 'architecture': 'fast', 'request_id': 'a', 'status': 'ok'})
    with pytest.raises(ValueError, match='another evaluation configuration'):
        CheckpointStore(path, FIELDS, {'evaluation_configuration': {'top_k': 3}})


@pytest.mark.asyncio
async def test_runner_checkpoints_first_429_then_stops(monkeypatch, tmp_path, config, retriever):
    import relay.evaluate as evaluation
    monkeypatch.setenv('GROQ_API_KEY', 'test-only')
    config.index_override = tmp_path / 'index.jsonl'
    config.index_path.write_text('\n'.join(json.dumps(c.to_dict()) for c in retriever.chunks))
    monkeypatch.setattr(evaluation, 'RelayConfig', lambda: config)
    monkeypatch.setattr(evaluation, 'HybridRetriever', lambda *args: retriever)
    requests = []
    async def handler(request):
        requests.append(request)
        return httpx.Response(429, headers={'retry-after': '160'},
            json={'error': {'message': 'requests per minute (RPM) limit reached'}}, request=request)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    real_provider = evaluation.BudgetedGroqProvider
    monkeypatch.setattr(evaluation, 'BudgetedGroqProvider',
        lambda *args, **kwargs: real_provider(*args, client=client, **kwargs))
    args = parser().parse_args(['--question-id', 'T1', '--architectures', 'fast', 'verified',
        '--output', str(tmp_path / 'runs.csv'), '--quota-state', str(tmp_path / 'quota.json')])
    try:
        assert await run(args) == 2
    finally:
        await client.aclose()
    rows = list(csv.DictReader((tmp_path / 'runs.csv').open()))
    assert len(requests) == 1
    assert len(rows) == 1
    assert rows[0]['status'] == 'rate_limited'
    detail = json.loads(rows[0]['provider_error_json'])
    assert detail['rate_limit_type'] == 'provider_rpm'
    assert detail['response_headers']['retry-after'] == '160'
    assert (tmp_path / 'runs.stop.json').exists()
    # A restart under the same hold must not dispatch a probe or reload embeddings.
    monkeypatch.setattr(evaluation, 'HybridRetriever', lambda *args: pytest.fail('loaded while paused'))
    assert await run(args) == 2


@pytest.mark.asyncio
async def test_remaining_mode_never_loads_provider(monkeypatch, tmp_path, config):
    import relay.evaluate as evaluation
    config.index_override = tmp_path / 'index.jsonl'
    config.index_path.write_text('')
    monkeypatch.setattr(evaluation, 'RelayConfig', lambda: config)
    monkeypatch.setattr(evaluation, 'BudgetedGroqProvider', lambda *args, **kwargs: pytest.fail('API touched'))
    args = parser().parse_args(['--subset', 'formal18', '--remaining',
        '--output', str(tmp_path / 'pending.csv'), '--quota-state', str(tmp_path / 'quota.json')])
    assert await run(args) == 0
    assert not (tmp_path / 'pending.csv').exists()


@pytest.mark.asyncio
async def test_successful_evaluations_are_skipped_on_identical_restart(monkeypatch, tmp_path, config, retriever):
    import relay.evaluate as evaluation
    monkeypatch.setenv('GROQ_API_KEY', 'test-only')
    config.index_override = tmp_path / 'index.jsonl'
    config.index_path.write_text('\n'.join(json.dumps(c.to_dict()) for c in retriever.chunks))
    monkeypatch.setattr(evaluation, 'RelayConfig', lambda: config)
    monkeypatch.setattr(evaluation, 'HybridRetriever', lambda *args: retriever)
    requests = []
    async def handler(request):
        requests.append(request)
        content = {'decision': 'answer', 'claims': [
            {'text': 'Attention uses heads', 'sources': ['S1']}]}
        return httpx.Response(200, json={
            'choices': [{'message': {'content': json.dumps(content)}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 20, 'completion_tokens': 8}}, request=request)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        real_provider = evaluation.BudgetedGroqProvider
        monkeypatch.setattr(evaluation, 'BudgetedGroqProvider',
            lambda *args, **kwargs: real_provider(*args, client=client, **kwargs))
        args = parser().parse_args(['--question-id', 'T1', '--architectures', 'fast',
            '--output', str(tmp_path / 'runs.csv'), '--quota-state', str(tmp_path / 'quota.json')])
        assert await run(args) == 0
        assert await run(args) == 0
    assert len(requests) == 1
    assert len(list(csv.DictReader((tmp_path / 'runs.csv').open()))) == 1
    assert len((tmp_path / 'runs.attempts.jsonl').read_text().splitlines()) == 1


def test_local_queue_failures_do_not_count_as_model_calls():
    from relay.instrumentation import RequestTrace
    trace = RequestTrace('request', 'verified')
    trace.add_failed_call('generation', RateLimitError('not dispatched',
        rate_limit_type='local_queue_deadline', queue_wait_ms=0,
        diagnostics={'dispatched': False}))
    metrics = trace.metrics()
    assert metrics['total_model_calls'] == 0
    assert metrics['total_input_tokens'] == 0
    assert metrics['stage_admission_attempts'] == 1


@pytest.mark.asyncio
async def test_expired_daily_reset_is_not_restored_as_permanent_exhaustion(tmp_path):
    import time
    path = tmp_path / 'quota.json'
    path.write_text(json.dumps({'saved_at': time.time() - 10, 'available_tokens': 100,
        'remaining_daily_requests': 0, 'daily_reset_at': time.time() - 1,
        'stop_reason': {'reason': 'exhausted', 'rate_limit_type': 'rpd_exhausted',
                        'retry_not_before_epoch': time.time() - 1}}))
    scheduler = RequestScheduler(SchedulerConfig(state_path=path))
    reservation = await scheduler.acquire(1, 1)
    assert reservation is not None

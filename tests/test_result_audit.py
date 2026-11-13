import json
from datetime import datetime, timezone, timedelta

import pytest

from relay.architectures.base import REFUSAL
from relay.result_audit import (audit, authored_grades, comparison, cross_paper_coverage,
                                quota_audit, same_version_questions, usage_summary)


def result(**overrides):
    return {'request_id': 'r', 'question_id': 'T1', 'architecture': 'fast', 'status': 'ok',
            'question': 'attention?', 'answer': 'draft', 'evaluation_config_id': 'new',
            'origin_evaluation_config_id': 'old', 'total_model_calls': '1',
            'total_ms': '100', 'queue_wait_ms': '60', 'provider_ttft_ms': '',
            'llm_calls_json': json.dumps([{'stage': 'generation', 'input_tokens': 20,
                'output_tokens': 5, 'dispatched': True, 'latency_ms': 30}]), **overrides}


def test_missing_usage_is_a_lower_bound_not_a_fabricated_recovery():
    row = result(llm_calls_json=json.dumps([
        {'stage': 'generation', 'input_tokens': 2985, 'output_tokens': 235},
        {'stage': 'verification_format_attempt', 'input_tokens': None, 'output_tokens': None,
         'reserved_total_tokens': 1744},
        {'stage': 'verification', 'input_tokens': 1056, 'output_tokens': 327},
        {'stage': 'revision', 'input_tokens': 1055, 'output_tokens': 206},
        {'stage': 'revision_verification', 'input_tokens': 969, 'output_tokens': 104}]))
    usage = usage_summary(row)
    assert usage['known_input_tokens'] == 6065 and usage['known_output_tokens'] == 872
    assert usage['known_total_tokens'] == 6937
    assert usage['unknown_usage_call_count'] == 1
    assert usage['exact_total_tokens'] is None
    assert usage['failed_call_reserved_tokens_estimate'] == 1744


def test_absent_call_log_is_not_proof_of_zero_usage():
    assert usage_summary(result(llm_calls_json='[]'))['exact_total_tokens'] is None
    assert usage_summary(result(llm_calls_json='[]', total_model_calls='0'))['exact_total_tokens'] == 0


def test_only_identical_original_versions_and_accounts_enter_matched_subset():
    rows = [result(request_id=a, architecture=a) for a in ('fast','verified','adaptive')]
    assert same_version_questions(rows) == {'T1'}
    rows[1]['origin_evaluation_config_id'] = ''
    assert same_version_questions(rows) == set()  # Same rewritten current ID does not erase original provenance.
    rows[1]['origin_evaluation_config_id'] = 'old'
    rows[1]['quota_profile'] = 'account2'
    assert same_version_questions(rows) == set()


def test_ok_status_never_supplies_accuracy_and_refusals_are_separate():
    rows = [result(), result(request_id='u', question_id='U1', answer=REFUSAL,
                             status='insufficient_evidence')]
    grades = [{'request_id':'r', 'accuracy':0, 'citation_grounding':0},
              {'request_id':'u', 'accuracy':1, 'citation_grounding':1, 'correct_refusal':1}]
    summary = comparison(rows, grades, scope='test')[0]
    assert summary['factual_answers'] == 1 and summary['factual_accuracy_passes'] == 0
    assert summary['correct_refusals'] == 1 and summary['overall_accuracy_passes'] == 1
    assert summary['processing_mean_ms'] == 40


def test_cross_paper_retrieval_does_not_imply_answer_citation_coverage():
    row = result(question_id='X1', source_papers='1706.03762v7;2507.07955v2',
                 retrieved_chunk_ids_json=json.dumps(['1706.03762-p007-a', '2507.07955-p005-b']),
                 cited_sources_json=json.dumps([{'paper_id':'2507.07955'}]))
    coverage = cross_paper_coverage([row])[0]
    assert coverage['retrieval_coverage_complete'] is True
    assert coverage['citation_coverage_complete'] is False
    assert coverage['missing_cited_papers'] == '1706.03762'


def test_accepted_calls_do_not_certify_provider_rolling_or_daily_limits():
    start = datetime(2026,11,13,tzinfo=timezone.utc)
    rows = []
    for i, offset in enumerate((0,25)):
        call = {'stage':'generation', 'dispatched':True, 'input_tokens':4000, 'output_tokens':500,
                'reserved_total_tokens':5000, 'local_rpm_remaining':23-i,
                'local_tpm_remaining':1400, 'provider_status_code':200,
                'queue_wait_ms':0, 'latency_ms':1000, 'provider_rate_limit_headers':{
                    'x-ratelimit-remaining-tokens':'3000', 'x-ratelimit-remaining-requests':str(999-i),
                    'x-ratelimit-limit-requests':'1000', 'x-ratelimit-limit-tokens':'8000'}}
        rows.append(result(request_id=str(i), quota_profile='account2', execution_order=str(i+1),
            run_started_at=(start+timedelta(seconds=offset)).isoformat(),
            run_completed_at=(start+timedelta(seconds=offset+1)).isoformat(),
            llm_calls_json=json.dumps([call])))
    summary, calls = quota_audit(rows, profile='account2', rpm=30,tpm=8000,safety=.8)
    assert summary['invariant_checks']['every_call_accepted_200'] is True
    assert summary['rolling_60s_max_gross_tokens_reconstructed_upper_bound'] == 9000
    assert summary['rolling_60s_max_gross_tokens_reconstructed_lower_bound'] == 9000
    assert summary['gross_rolling_tokens_exceed_reported_tpm'] is True
    assert summary['provider_quota_compliance_certified'] is False
    assert summary['daily_quota_certified'] is False
    assert len(calls) == 2


def test_authored_review_is_required_even_for_ok_results():
    review = {'refusals':{}, 'answers':{}, 'references':{}, 'reviewer':'Independent reviewer'}
    with pytest.raises(ValueError, match='No authored'):
        authored_grades([result()], review)
    with pytest.raises(ValueError, match='Infrastructure failure'):
        authored_grades([result(status='rate_limited')], review)


def test_changed_snapshot_cannot_reuse_existing_grades(tmp_path):
    source, review = tmp_path/'raw.csv', tmp_path/'review.json'
    source.write_text('changed result')
    review.write_text(json.dumps({'input_sha256':'does-not-match'}))
    with pytest.raises(ValueError, match='reviewed snapshot'):
        audit(source,review,tmp_path/'out')
    assert not (tmp_path/'out').exists()

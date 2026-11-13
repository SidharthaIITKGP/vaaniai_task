"""Offline aggregation of explicitly authored PDF grades and recorded execution evidence.

No provider is initialized. Raw CSVs, journals, logs and quota state are read only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from .architectures.base import REFUSAL
from .checkpoint import TERMINAL_STATUSES, read_rows
from .experiment_review import compare, packet
from .load_lab import write_csv


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value):
    return None if value is None or value == '' else float(value)


def calls(row):
    return json.loads(row.get('llm_calls_json') or '[]')


def authored_grades(rows: list[dict], review: dict) -> tuple[list[dict], list[dict]]:
    """Apply only request-bound, explicitly authored judgments, never pipeline verdicts."""
    grades, details = [], []
    seen = set()
    factual_keys = set()
    for row in rows:
        key = f"{row['question_id']}.{row['architecture']}"
        if key in seen:
            raise ValueError(f'Duplicate evaluation: {key}')
        seen.add(key)
        if row['status'] not in TERMINAL_STATUSES:
            raise ValueError(f'Infrastructure failure must not receive an answer grade: {key}')
        if row['question_id'] in review['refusals']:
            if row['answer'].strip() != REFUSAL:
                raise ValueError(f'Unreviewed refusal wording for {key}')
            judgment = dict(review['refusals'][row['question_id']], classification='correct_refusal', flags=[])
        else:
            factual_keys.add(key)
            if key not in review['answers']:
                raise ValueError(f'No authored independent review: {key}')
            judgment = review['answers'][key]
        if judgment['accuracy'] not in (0, 1) or judgment['citation_grounding'] not in (0, 1):
            raise ValueError(f'Invalid binary grade for {key}')
        reference = review['references'][row['question_id']]
        grade = {'review_id': hashlib.sha256(row['request_id'].encode()).hexdigest()[:16],
                 'request_id': row['request_id'], 'question_id': row['question_id'],
                 'accuracy': judgment['accuracy'], 'citation_grounding': judgment['citation_grounding'],
                 'correct_refusal': judgment.get('correct_refusal', ''), 'reviewer': review['reviewer'],
                 'paper_pages_checked': reference['pages'], 'rationale': judgment['rationale']}
        grades.append(grade)
        details.append({**grade, 'architecture': row['architecture'], 'question': row['question'],
                        'answer': row['answer'], 'classification': judgment['classification'],
                        'flags': judgment['flags'], 'reference_answer': reference['expected'],
                        'cited_sources': json.loads(row.get('cited_sources_json') or '[]')})
    if factual_keys != set(review['answers']):
        raise ValueError('Authored factual reviews and input evaluations do not match exactly')
    return grades, details


def usage_summary(row: dict) -> dict:
    records = calls(row)
    dispatched = [c for c in records if c.get('dispatched') is not False]
    missing = [c for c in dispatched if c.get('input_tokens') is None or c.get('output_tokens') is None]
    known_input = sum(c.get('input_tokens') or 0 for c in dispatched)
    known_output = sum(c.get('output_tokens') or 0 for c in dispatched)
    exact_input = sum(c['input_tokens'] for c in dispatched) if all(
        c.get('input_tokens') is not None for c in dispatched) else None
    exact_output = sum(c['output_tokens'] for c in dispatched) if all(
        c.get('output_tokens') is not None for c in dispatched) else None
    # An absent call log is not evidence of zero usage unless zero calls was recorded.
    if not dispatched and number(row.get('total_model_calls')) != 0:
        exact_input, exact_output = None, None
    return {'request_id': row['request_id'], 'question_id': row['question_id'],
            'architecture': row['architecture'], 'known_input_tokens': known_input,
            'known_output_tokens': known_output, 'known_total_tokens': known_input + known_output,
            'unknown_usage_call_count': len(missing),
            'exact_input_tokens': exact_input, 'exact_output_tokens': exact_output,
            'exact_total_tokens': exact_input + exact_output if exact_input is not None and exact_output is not None else None,
            'failed_call_reserved_tokens_estimate': sum(c.get('reserved_total_tokens') or 0 for c in missing),
            'measurement': 'exact stage usage' if not missing and exact_input is not None else 'measured lower bound; incomplete usage',
            'unknown_stages': ','.join(c['stage'] for c in missing)}


def same_version_questions(rows: list[dict]) -> set[str]:
    groups = defaultdict(list)
    for row in rows:
        groups[row['question_id']].append(row)
    return {q for q, items in groups.items()
            if {r['architecture'] for r in items} == {'fast', 'verified', 'adaptive'}
            and len(items) == 3
            and len({r.get('origin_evaluation_config_id') or r['evaluation_config_id'] for r in items}) == 1
            and len({r.get('quota_profile') or 'default' for r in items}) == 1}


def cross_paper_coverage(rows: list[dict]) -> list[dict]:
    output = []
    for row in rows:
        required = {paper.split('v')[0].strip() for paper in row.get('source_papers', '').split(';') if paper.strip()}
        if len(required) < 2:
            continue
        cited = {s['paper_id'] for s in json.loads(row.get('cited_sources_json') or '[]')}
        retrieved = {chunk.split('-p')[0] for chunk in json.loads(row.get('retrieved_chunk_ids_json') or '[]')}
        output.append({'request_id': row['request_id'], 'question_id': row['question_id'],
            'architecture': row['architecture'], 'required_papers': ';'.join(sorted(required)),
            'retrieved_papers': ';'.join(sorted(retrieved)), 'cited_papers': ';'.join(sorted(cited)),
            'missing_retrieved_papers': ';'.join(sorted(required-retrieved)),
            'missing_cited_papers': ';'.join(sorted(required-cited)),
            'retrieval_coverage_complete': required <= retrieved, 'citation_coverage_complete': required <= cited})
    return output


def comparison(rows: list[dict], grades: list[dict], *, scope: str) -> list[dict]:
    by_id = {g['request_id']: g for g in grades}
    output = []
    for architecture in ('fast', 'verified', 'adaptive'):
        selected = [r for r in rows if r['architecture'] == architecture]
        if not selected:
            continue
        factual = [r for r in selected if not r['question_id'].startswith('U')]
        refusal = [r for r in selected if r['question_id'].startswith('U')]
        def count(field, subset):
            return sum(by_id[r['request_id']].get(field) == 1 for r in subset)
        def metric(field, *, subtract_queue=False):
            values = [number(r.get(field)) for r in selected]
            if any(v is None for v in values):
                return {'mean': None, 'p50': None, 'p95': None}
            if subtract_queue:
                waits = [number(r.get('queue_wait_ms')) for r in selected]
                if any(w is None for w in waits):
                    return {'mean': None, 'p50': None, 'p95': None}
                values = [v-w for v, w in zip(values, waits)]
            return {'mean': float(np.mean(values)), 'p50': float(np.percentile(values, 50)),
                    'p95': float(np.percentile(values, 95))}
        usage = [usage_summary(r) for r in selected]
        row = {'scope': scope, 'architecture': architecture,
               'questions': len(selected), 'factual_answers': len(factual),
               'factual_accuracy_passes': count('accuracy', factual),
               'factual_citation_passes': count('citation_grounding', factual),
               'factual_joint_passes': sum(by_id[r['request_id']]['accuracy'] == 1 and
                                          by_id[r['request_id']]['citation_grounding'] == 1 for r in factual),
               'factual_accuracy_rate': count('accuracy', factual) / len(factual) if factual else None,
               'factual_citation_rate': count('citation_grounding', factual) / len(factual) if factual else None,
               'correct_refusals': count('correct_refusal', refusal), 'refusal_questions': len(refusal),
               'overall_accuracy_passes': count('accuracy', selected),
               'total_model_calls': sum(int(r['total_model_calls']) for r in selected),
               'known_input_tokens': sum(u['known_input_tokens'] for u in usage),
               'known_output_tokens': sum(u['known_output_tokens'] for u in usage),
               'unknown_usage_calls': sum(u['unknown_usage_call_count'] for u in usage),
               'exact_total_tokens': sum(u['exact_total_tokens'] for u in usage)
                    if all(u['exact_total_tokens'] is not None for u in usage) else None,
               'same_version_questions': ','.join(sorted({r['question_id'] for r in selected})),
               'formal_selection_eligible': False}
        for name, field, subtract in [('total', 'total_ms', False), ('queue', 'queue_wait_ms', False),
                                      ('processing', 'total_ms', True)]:
            row.update({f'{name}_{stat}_ms': v for stat, v in metric(field, subtract_queue=subtract).items()})
        row['provider_latency_sum_ms'] = sum(c.get('latency_ms') or 0 for r in selected for c in calls(r))
        row['provider_ttft_observed_count'] = sum(number(r.get('provider_ttft_ms')) is not None for r in selected)
        output.append(row)
    return output


def quota_audit(rows: list[dict], *, profile: str, rpm: int, tpm: int, safety: float) -> tuple[dict, list[dict]]:
    selected = sorted([r for r in rows if (r.get('quota_profile') or 'default') == profile],
                      key=lambda r: int(r['execution_order']))
    records = []
    for row in selected:
        stage_calls = calls(row)
        start = datetime.fromisoformat(row['run_started_at']).timestamp()
        end = datetime.fromisoformat(row['run_completed_at']).timestamp()
        durations = [(c.get('queue_wait_ms') or 0) + (c.get('latency_ms') or 0) for c in stage_calls]
        prefix = 0.0
        for index, call in enumerate(stage_calls):
            if call.get('dispatched') is False:
                prefix += durations[index]
                continue
            headers = call.get('provider_rate_limit_headers') or {}
            lower = start + (prefix + (call.get('queue_wait_ms') or 0)) / 1000
            upper = end - (sum(durations[index+1:]) + (call.get('latency_ms') or 0)) / 1000
            records.append({'request_id': row['request_id'], 'question_id': row['question_id'],
                'architecture': row['architecture'], 'execution_order': int(row['execution_order']),
                'stage': call['stage'], 'quota_profile': profile,
                'dispatch_lower_epoch': lower, 'dispatch_upper_epoch': upper,
                'dispatch_timing_measurement': 'reconstructed bounds, not directly logged stage timestamps',
                'input_tokens': call.get('input_tokens'), 'output_tokens': call.get('output_tokens'),
                'reserved_total_tokens': call.get('reserved_total_tokens'),
                'local_rpm_remaining': call.get('local_rpm_remaining'),
                'local_tpm_remaining': call.get('local_tpm_remaining'),
                'queue_wait_ms': call.get('queue_wait_ms'), 'provider_latency_ms': call.get('latency_ms'),
                'provider_status_code': call.get('provider_status_code'),
                'remaining_rpd': number(headers.get('x-ratelimit-remaining-requests')),
                'remaining_tpm': number(headers.get('x-ratelimit-remaining-tokens')),
                'reported_rpd_limit': number(headers.get('x-ratelimit-limit-requests')),
                'reported_tpm_limit': number(headers.get('x-ratelimit-limit-tokens')),
                'provider_request_id': headers.get('x-request-id'),
                'scheduler_wait_reason': call.get('scheduler_wait_reason'),
                'retry_after': headers.get('retry-after')})
            prefix += durations[index]
    if not records:
        return {'profile': profile, 'pipeline_rows': 0, 'model_calls': 0}, []
    endpoints = [c['dispatch_lower_epoch'] for c in records] + [c['dispatch_upper_epoch'] for c in records]
    def window(start):
        return [c for c in records if c['dispatch_upper_epoch'] >= start and c['dispatch_lower_epoch'] < start + 60]
    max_requests = max(len(window(t)) for t in endpoints)
    known = all(c['input_tokens'] is not None and c['output_tokens'] is not None for c in records)
    max_tokens = max(sum(c['input_tokens'] + c['output_tokens'] for c in window(t)) for t in endpoints) if known else None
    guaranteed_tokens = max(sum(c['input_tokens'] + c['output_tokens'] for c in records
        if c['dispatch_lower_epoch'] >= t and c['dispatch_upper_epoch'] < t+60)
        for t in endpoints) if known else None
    inputs = sum(c['input_tokens'] or 0 for c in records)
    outputs = sum(c['output_tokens'] or 0 for c in records)
    span = (datetime.fromisoformat(selected[-1]['run_completed_at']) -
            datetime.fromisoformat(selected[0]['run_started_at'])).total_seconds()
    token_budget = int(tpm * safety)
    invariant_checks = {
        'every_call_accepted_200': all(c['provider_status_code'] == 200 for c in records),
        'local_reservations_within_bucket_capacity': all(c['reserved_total_tokens'] is not None and
            0 <= c['reserved_total_tokens'] <= token_budget for c in records),
        'nonnegative_local_tpm_at_admission': all(c['local_tpm_remaining'] is not None and
            c['local_tpm_remaining'] >= 0 for c in records),
        'local_rpm_counter_within_budget': all(c['local_rpm_remaining'] is not None and
            0 <= c['local_rpm_remaining'] < int(rpm * safety) for c in records),
        'no_negative_provider_remaining_headers': all(c['remaining_tpm'] is not None and c['remaining_tpm'] >= 0
            and c['remaining_rpd'] is not None and c['remaining_rpd'] >= 0 for c in records),
        'observed_usage_no_more_than_reservation': known and all(
            c['input_tokens'] + c['output_tokens'] <= c['reserved_total_tokens'] for c in records),
        'max_reconstructed_rolling_rpm_within_local_limit': max_requests <= int(rpm * safety),
        'coherent_timing_bounds': all(c['dispatch_upper_epoch'] + .001 >= c['dispatch_lower_epoch'] for c in records),
    }
    summary = {'profile': profile, 'pipeline_rows': len(selected), 'model_calls': len(records),
        'observed_input_tokens': inputs, 'observed_output_tokens': outputs,
        'observed_total_tokens': inputs + outputs if known else None,
        'span_seconds': span, 'gross_average_tokens_per_minute': (inputs+outputs)/span*60,
        'configured_rpm': rpm, 'configured_tpm': tpm, 'safety_margin': safety,
        'local_rpm_budget': int(rpm*safety), 'local_token_bucket_capacity': token_budget,
        'local_token_refill_per_second': token_budget/60,
        'initial_burst_plus_refill_envelope_tokens': token_budget + span*token_budget/60,
        'rolling_60s_max_calls_reconstructed_upper_bound': max_requests,
        'rolling_60s_max_gross_tokens_reconstructed_upper_bound': max_tokens,
        'rolling_60s_max_gross_tokens_reconstructed_lower_bound': guaranteed_tokens,
        'gross_rolling_tokens_exceed_reported_tpm': max_tokens > tpm if max_tokens is not None else None,
        'invariant_checks': invariant_checks,
        'minimum_remaining_tpm_header': min(c['remaining_tpm'] for c in records if c['remaining_tpm'] is not None),
        'last_remaining_rpd_header': records[-1]['remaining_rpd'],
        'tpd_remaining_measured': None,
        'cached_token_counts_measured': None,
        'daily_quota_certified': False,
        'provider_quota_compliance_certified': False,
        'verdict': 'All observed calls accepted and recorded bucket admissions valid; strict rolling TPM and full-day TPD compliance cannot be certified.',
        'limitations': [
            'The scheduler is a continuously refilling bucket, not a strict rolling 60-second token window.',
            'Gross token counts include potentially cached inputs; cached-token exemptions were not exported.',
            'Groq documents TPM/header meanings and cache exemptions but not the precise enforcement window algorithm.',
            'Only these observed requests are known; no account-specific TPD remaining or full organization usage ledger was exported.',
            'A separate profile and user attestation do not cryptographically establish distinct Groq organizations.'],
        'documentation': 'https://console.groq.com/docs/rate-limits'}
    return summary, records


def audit(source: Path, review_path: Path, output: Path, logs: Path | None = None,
          quota_state: Path | None = None) -> dict:
    review = json.loads(review_path.read_text())
    if digest(source) != review['input_sha256']:
        raise ValueError('Input differs from independently reviewed snapshot; do not reuse these grades')
    root = Path(__file__).resolve().parent.parent
    companions = [source.with_suffix('.manifest.json'), source.with_suffix('.attempts.jsonl')]
    preserved = [source, review_path, *[p for p in companions if p.exists()],
                 *sorted((root / 'Research Papers').glob('*.pdf'))]
    preserved.extend(p for p in (logs, quota_state) if p is not None and p.exists())
    preserved.extend(p for p in source.parent.glob('architecture_formal18*') if p.is_file())
    preserved.extend((root / 'data').glob('groq_quota_state*.json'))
    original = {str(p): digest(p) for p in preserved}
    rows = read_rows(source)
    grades, details = authored_grades(rows, review)
    output.mkdir(parents=True, exist_ok=True)
    packet([source], output / 'review_packet')
    write_csv(output / 'grades.csv', grades)
    (output / 'grade_details.json').write_text(json.dumps(details, ensure_ascii=False, indent=2))
    write_csv(output / 'factual_answer_grades.csv', [
        {**{k:v for k,v in d.items() if k not in {'flags','cited_sources'}},
         'flags_json': json.dumps(d['flags']), 'cited_sources_json': json.dumps(d['cited_sources'], ensure_ascii=False)}
        for d in details if not d['question_id'].startswith('U')])
    (output / 'reference_checklist.json').write_text(json.dumps(review['references'], ensure_ascii=False, indent=2))
    (output / 'rubric.json').write_text(json.dumps({k: review[k] for k in ('reviewer','method','rubric')}, indent=2))
    usages = [usage_summary(r) for r in rows]
    write_csv(output / 'usage_recovery.csv', usages)
    coverage = cross_paper_coverage(rows)
    write_csv(output / 'cross_paper_coverage.csv', coverage)
    annotated = []
    by_grade, by_usage = {g['request_id']: g for g in grades}, {u['request_id']: u for u in usages}
    by_detail = {g['request_id']: g for g in details}
    for row in rows:
        grade, usage = by_grade[row['request_id']], by_usage[row['request_id']]
        annotated.append({**row, **{f'audit_{k}': v for k,v in grade.items() if k != 'request_id'},
            **{f'audit_{k}': v for k,v in usage.items() if k not in {'request_id','question_id','architecture'}},
            'audit_flags_json': json.dumps(by_detail[row['request_id']]['flags']),
            'audit_origin_config': row.get('origin_evaluation_config_id') or row['evaluation_config_id']})
    write_csv(output / 'graded_results.csv', annotated)
    full = comparison(rows, grades, scope='mixed-version repair snapshot; descriptive only')
    matched_ids = same_version_questions(rows)
    matched = comparison([r for r in rows if r['question_id'] in matched_ids], grades,
                         scope='same-version and same-profile matched subset; below formal minimum')
    write_csv(output / 'descriptive_comparison.csv', full)
    write_csv(output / 'matched_comparison.csv', matched)
    manifest = json.loads(source.with_suffix('.manifest.json').read_text())
    quota, stage_rows = quota_audit(rows, profile='account2', rpm=manifest['rpm'],
                                  tpm=manifest['tpm'], safety=manifest['safety_margin'])
    write_csv(output / 'account2_call_audit.csv', stage_rows)
    write_csv(output / 'account2_row_audit.csv', [
        {'request_id': r['request_id'], 'question_id': r['question_id'], 'architecture': r['architecture'],
         'execution_order': r['execution_order'], 'run_started_at': r['run_started_at'],
         'run_completed_at': r['run_completed_at'], 'total_model_calls': r['total_model_calls'],
         'input_tokens': r['total_input_tokens'], 'output_tokens': r['total_output_tokens'],
         'total_ms': r['total_ms'], 'queue_wait_ms': r['queue_wait_ms'],
         'processing_ms': float(r['total_ms'])-float(r['queue_wait_ms']),
         'all_stages_accepted_200': all(c['provider_status_code']==200 for c in calls(r))}
        for r in rows if r.get('quota_profile')=='account2'])
    (output / 'quota_audit.json').write_text(json.dumps(quota, indent=2))
    h4 = next(r for r in rows if r['question_id'] == 'H4' and r['architecture'] == 'verified')
    h4_usage = by_usage[h4['request_id']]
    matches = []
    for path in [p for p in (logs, source.with_suffix('.attempts.jsonl')) if p and p.exists()]:
        for line_number, line in enumerate(path.open(), 1):
            if h4['request_id'] not in line:
                continue
            event = json.loads(line)
            record = event.get('row', event)
            records = record.get('llm_calls') or json.loads(record.get('llm_calls_json') or '[]')
            matches.append({'path': str(path), 'line': line_number,
                'request_id': h4['request_id'], 'stage_usage': [
                    {k:c.get(k) for k in ('stage','input_tokens','output_tokens','provider_status_code')}
                    for c in records]})
    h4_recovery = {**h4_usage, 'raw_csv_totals': {k:h4[k] for k in ('total_input_tokens','total_output_tokens')},
        'evidence_matches': matches, 'recovery': 'Partial exact recovery only; one HTTP 400 call remains unmetered.',
        'actual_total_tokens': h4_usage['exact_total_tokens'],
        'reservation_based_total_estimate': h4_usage['known_total_tokens'] + h4_usage['failed_call_reserved_tokens_estimate'],
        'reservation_estimate_is_actual_usage': False,
        'reason': 'CSV, append-only journal and request trace all lack usage for verification_format_attempt. '
                  'Refilling rate-limit headers and failed text cannot recover exact billed input, hidden reasoning or output tokens.'}
    (output / 'h4_usage_recovery.json').write_text(json.dumps(h4_recovery, indent=2))
    selection = compare([source], output / 'grades.csv', output / 'guarded_selection')
    recommendation = {'winner': None, 'formal_selection_eligible': False,
        'reason': 'Selective repair mixes 43 original-version successes with 11 corrected runs. '
                  'Only ten same-version/same-account questions cover all three candidates; corrected-version common coverage is zero.',
        'matched_question_ids': sorted(matched_ids), 'minimum_questions_required': 15,
        'descriptive_best_factual_accuracy': max(full,key=lambda r:r['factual_accuracy_rate'])['architecture'],
        'descriptive_most_fully_correct_and_grounded': max(full,key=lambda r:r['factual_joint_passes'])['architecture'],
        'recommendation': 'Keep all three candidates. Run the same 18 questions for all three under one frozen corrected configuration, '
                          'then independently grade before choosing. Do not selectively repair wrong answers in that fair experiment.',
        'guarded_selection': selection}
    (output / 'recommendation.json').write_text(json.dumps(recommendation, indent=2))
    invariant_ok = all(digest(Path(p)) == sha for p, sha in original.items())
    integrity = {'raw_files_unchanged': invariant_ok, 'sha256': original,
                 'measurements': 'Real recorded execution only; reconstructed timing bounds and reservation estimates explicitly labeled.',
                 'no_live_model_calls': True}
    (output / 'integrity.json').write_text(json.dumps(integrity, indent=2))
    if not invariant_ok:
        raise ValueError('A preserved input changed during audit; investigate concurrent writers')
    write_report(output, rows, full, matched, quota, h4_recovery, details, recommendation)
    return recommendation


def write_report(output, rows, full, matched, quota, h4, details, recommendation):
    def table(items):
        result = ['| Architecture | Factual accuracy | Citation support | Correct + grounded | Correct refusals |',
                  '|---|---:|---:|---:|---:|']
        for r in items:
            n = r['factual_answers']
            result.append(f"| {r['architecture']} | {r['factual_accuracy_passes']}/{n} | "
                f"{r['factual_citation_passes']}/{n} | {r['factual_joint_passes']}/{n} | "
                f"{r['correct_refusals']}/{r['refusal_questions']} |")
        return '\n'.join(result)
    factual = [d for d in details if not d['question_id'].startswith('U')]
    lines = ['# Formal18 repaired-run independent audit', '',
        'All 54 final pipeline runs are present: 39 factual answers and 15 refusals. '
        'The raw results are unchanged; no live model calls were made. '
        'Grades are an explicitly authored model-assisted PDF review, not blinded human judgments '
        'and not pipeline/verifier scores. See `rubric.json`, `grade_details.json` and `grades.csv`.', '',
        '## Full repaired snapshot: descriptive comparison only', '', table(full), '',
        'Accuracy means a fully correct and complete requested explanation, with no material false or unestablished extension. '
        'Partial answers receive 0 under the binary rubric. Citation support is checked against the exact retrieved excerpts; '
        'being true elsewhere in the PDF does not repair a weak citation. Refusals are shown separately to avoid inflating factual accuracy.', '',
        '## Fairness and recommendation', '', recommendation['reason'], '',
        'Matched question IDs: ' + ', '.join(recommendation['matched_question_ids']) + '.', '', table(matched), '',
        'All candidates tie on fully correct-and-grounded answers in this limited matched subset. '
        'It was selected by provenance availability after repairs and is not a representative replacement for the planned experiment. '
        'No winning architecture is established. Adaptive has the highest descriptive factual accuracy in the mixed snapshot; '
        'Fast has the most factual answers passing both correctness and citation support. Neither observation removes the version/repair confound. '
        'The existing selection guard remains incomplete with winner=null.', '',
        '## Errors, citation weaknesses and omissions', '']
    for d in factual:
        if d['accuracy'] == 0 or d['citation_grounding'] == 0:
            lines.extend([f"- **{d['question_id']} / {d['architecture']}** "
                          f"(accuracy {d['accuracy']}, citations {d['citation_grounding']}): "
                          f"{d['rationale']} Reference: {d['paper_pages_checked']}."])
    lines += ['', 'Verified X1 retrieved Transformer evidence but did not cite it: all final cited passages are H-Net. '
        'Fast and Adaptive X1 cite both required papers. Retrieval coverage and answer citation coverage are separate checks.', '',
        '## Account2 quota audit', '',
        f"Ten rows contain {quota['model_calls']} real model calls and "
        f"{quota['observed_input_tokens']:,} input + {quota['observed_output_tokens']:,} output = "
        f"{quota['observed_total_tokens']:,} gross tokens over {quota['span_seconds']:.3f} seconds. "
        f"All calls returned 200; the final RPD header is {quota['last_remaining_rpd_header']:.0f}. "
        'The original account state was not cleared or reused for account2.', '',
        f"Recorded local bucket capacity is {quota['local_token_bucket_capacity']:,} with "
        f"{quota['local_token_refill_per_second']:.3f} tokens/second refill. All recorded reservations fit, "
        'actual per-call totals fit their reservations, and admission counters are nonnegative. '
        f"The initial burst plus refill envelope is {quota['initial_burst_plus_refill_envelope_tokens']:.1f} tokens, "
        'consistent with observed aggregate use.', '',
        f"**Strict minute compliance is not certified.** Reconstructed serial-call timing intervals give a peak of "
        f"{quota['rolling_60s_max_calls_reconstructed_upper_bound']} calls and "
        f"{quota['rolling_60s_max_gross_tokens_reconstructed_upper_bound']:,} gross tokens in 60 seconds, "
        f"above the {quota['configured_tpm']:,} TPM header and the local 6,400-token refill amount. "
        'A bucket permits an initial burst plus refill and therefore does not enforce a strict rolling token window. '
        'This is an audit concern, not proof of a provider violation: cached-token counts were not retained and precise server window semantics are unknown.', '',
        '[Groq documents](https://console.groq.com/docs/rate-limits) organization-level limits, '
        'TPM and RPD header meanings, and cached-token exemptions. These headers do not expose remaining TPD. '
        'Consequently the full-day organization quota cannot be certified from this export. '
        'Separate profile names and user confirmation are the available evidence of the account switch; no authenticated organization identity was recorded. '
        'Before a strict-quota certification, retain cache-adjusted usage and exact dispatch timestamps, obtain account-specific limits, '
        'and add a conservative rolling reservation ledger without changing answer behavior.', '',
        '## H4 Verified token recovery', '',
        f"Request `{h4['request_id']}` contains five calls. Four calls have exact usage: "
        f"{h4['known_input_tokens']:,} input and {h4['known_output_tokens']:,} output "
        f"({h4['known_total_tokens']:,} measured tokens). The HTTP 400 formatting attempt has no usage in the CSV, "
        'journal or request trace. Exact total recovery is unavailable offline.', '',
        f"Its failed reservation was {h4['failed_call_reserved_tokens_estimate']:,} tokens. Adding it gives "
        f"{h4['reservation_based_total_estimate']:,}, a reservation-based estimate, **not actual usage or a proven upper bound**. "
        'Raw totals stay blank. `usage_recovery.csv` exposes known totals, unknown calls and exact totals separately; '
        '`h4_usage_recovery.json` records matching log locations. Provider request ID '
        '`req_01m7b1305me3pvb5xj4ftb7fs2` is the unresolved call for a future provider-side usage export.', '',
        '## Speed and resource comparison', '',
        '| Architecture | Mean total s | Mean queue s | Mean processing s | Model calls | Exact tokens |',
        '|---|---:|---:|---:|---:|---:|']
    for r in full:
        exact = str(r['exact_total_tokens']) if r['exact_total_tokens'] is not None else f"unknown; >= {r['known_input_tokens']+r['known_output_tokens']}"
        lines.append(f"| {r['architecture']} | {r['total_mean_ms']/1000:.3f} | "
                     f"{r['queue_mean_ms']/1000:.3f} | {r['processing_mean_ms']/1000:.3f} | "
                     f"{r['total_model_calls']} | {exact} |")
    lines += ['', 'Processing is measured total latency minus measured queue wait; it includes retrieval and checks. '
        'It is not solely provider execution time. Quantiles and provider latency sums are in the comparison CSVs. '
        'Reported tokens/calls here cover final selected attempts only, not all historical failed attempts. '
        'Provider TTFT is unavailable for buffered strict-JSON runs and is not invented. Account, version, order and warmup confound '
        'end-to-end latency across the mixed run; the matched subset is still descriptive and below the selection minimum.', '',
        'Reproduce offline with:', '', '```bash',
        'python -m relay.result_audit --input results/architecture_formal18_repair_v2.csv \\',
        '  --review audits/formal18_repair_v2_review.json \\',
        '  --logs logs/relay.jsonl --quota-state data/groq_quota_state.account2.json \\',
        '  --output results/formal18_repair_v2_audit', '```', '']
    (output / 'REPORT.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--logs', type=Path)
    parser.add_argument('--quota-state', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.input, args.review, args.output, args.logs, args.quota_state), indent=2))


if __name__ == '__main__':
    main()

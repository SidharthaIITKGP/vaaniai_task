"""Export the exact request schema and tables from The_Relay_Assignment_Brief.pdf.

Unreviewed answers have a blank answer_grade and grade_status=not_measured.
Fabricating a grade to fill the prescribed enum would be less truthful.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from .load_lab import write_csv


REQUIRED_COLUMNS = ('request_id', 'scenario', 'source', 'start_time', 'first_token_ms',
                    'total_ms', 'status', 'retries', 'tokens_in', 'tokens_out',
                    'retrieval_setting', 'answer_grade')
EXTRA_COLUMNS = ('grade_status', 'original_status', 'input_file', 'queue_wait_ms',
                 'processing_ms', 'queue_length', 'estimated_cost_usd', 'time_basis',
                 'pipeline_or_service', 'provider_ttft_ms', 'retry_after', 'http_status')


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def number(value):
    return float(value) if value not in ('', None, 'not measured') else None


def export_requests(load_directory: Path) -> list[dict]:
    records = []
    def append(row, extra):
        record = {key: '' for key in (*REQUIRED_COLUMNS, *EXTRA_COLUMNS)}
        record.update(row)
        record.update(extra)
        record['grade_status'] = 'reviewed' if record['answer_grade'] else 'not_measured'
        records.append(record)
    probe = Path('groq_probe.csv')
    for row in read_csv(probe):
        append({key: row.get(key, '') for key in REQUIRED_COLUMNS}, {
            'original_status': row['status'], 'input_file': str(probe),
            'queue_wait_ms': row.get('queue_ms'), 'provider_ttft_ms': row.get('first_token_ms'),
            'retry_after': row.get('retry_after'), 'http_status': row.get('http_status'),
            'time_basis': 'measured_wall_clock', 'pipeline_or_service': 'standalone_service'})
    real_file = Path('results/final_bounded_T1.csv')
    grade_file = Path('results/final_T1_review/grades.csv')
    grades = {row['request_id']: row for row in read_csv(grade_file)} if grade_file.exists() else {}
    if real_file.exists():
        for row in read_csv(real_file):
            grade = grades.get(row['request_id'], {})
            final_grade = ''
            if grade.get('accuracy') == '1' and grade.get('citation_grounding') == '1':
                final_grade = 'correct'
            elif grade.get('accuracy') == '0':
                final_grade = 'refused_wrongly' if 'do not establish' in row['answer'] else 'wrong'
            total = number(row['total_ms'])
            wait = number(row['queue_wait_ms'])
            status = ('ok' if row['status'] in {'ok', 'insufficient_evidence', 'citation_validation_failed',
                'verification_failed', 'revision_verification_failed'} else 'error')
            append({'request_id': row['request_id'], 'scenario': 'quality_smoke_T1', 'source': 'real',
                'start_time': row['run_started_at'], 'first_token_ms': row['user_visible_ttft_ms'],
                'total_ms': row['total_ms'], 'status': status, 'retries': 0,
                'tokens_in': row['total_input_tokens'], 'tokens_out': row['total_output_tokens'],
                'retrieval_setting': f"{row['architecture']}:500/5:no_rerank", 'answer_grade': final_grade}, {
                'original_status': row['status'], 'input_file': str(real_file),
                'queue_wait_ms': wait, 'processing_ms': total - wait if total is not None and wait is not None else '',
                'pipeline_or_service': 'rag_pipeline', 'time_basis': 'measured_wall_clock',
                'provider_ttft_ms': row['provider_ttft_ms']})
    anchor = datetime(2026, 11, 13, 0, 0, tzinfo=timezone.utc)
    for path in sorted(load_directory.glob('*_retries*.csv')):
        for row in read_csv(path):
            ok = row['status'] == 'ok'
            status = ('ok' if ok else 'timeout' if row['status'] == 'admission_timeout'
                      else 'rate_limited')
            provider_ttft = number(row['provider_ttft_ms'])
            wait = number(row['queue_wait_ms']) or 0
            pipeline = row['architecture'] != 'standalone_llm'
            first_token = (number(row['user_visible_ttft_ms']) if pipeline else
                           wait + provider_ttft if provider_ttft is not None else None)
            append({'request_id': f'{path.stem}:{row["request_id"]}',
                'scenario': path.stem, 'source': 'stand-in',
                'start_time': (anchor + timedelta(seconds=float(row['arrival_s']))).isoformat(),
                'first_token_ms': first_token, 'total_ms': row['latency_ms'], 'status': status,
                'retries': row['retries'], 'tokens_in': row['input_tokens'],
                'tokens_out': row['output_tokens'], 'retrieval_setting': 'not applicable',
                'answer_grade': ''}, {'original_status': row['status'], 'input_file': str(path),
                'queue_wait_ms': row['queue_wait_ms'], 'processing_ms': row['processing_ms'],
                'queue_length': row['queue_length_at_arrival'],
                'estimated_cost_usd': row['estimated_cost_usd'], 'time_basis': 'virtual_time',
                'pipeline_or_service': 'rag_bundle_capacity' if pipeline else 'standalone_service',
                'provider_ttft_ms': row['provider_ttft_ms']})
    return records


def percentile(rows, key, q):
    values = [number(r[key]) for r in rows]
    measured = [value for value in values if value is not None]
    return round(float(np.percentile(measured, q)), 2) if measured else 'not measured'


def tables(records):
    groups = defaultdict(list)
    for row in records:
        groups[(row['scenario'], row['source'])].append(row)
    latency, outcomes = [], []
    for (scenario, source), rows in groups.items():
        n = len(rows)
        success = [r for r in rows if r['status'] == 'ok']
        start = [datetime.fromisoformat(r['start_time']) for r in rows]
        ends = [s + timedelta(milliseconds=number(r['total_ms']) or 0) for s, r in zip(start, rows)]
        duration = max((max(ends) - min(start)).total_seconds(), 0)
        arrival_span = (max(start) - min(start)).total_seconds()
        a = {'Scenario': scenario, 'Real or stand-in': source, 'Requests sent': n,
             'Duration (s)': round(duration, 3),
             'Offered rate (per min)': round((n - 1) * 60 / arrival_span, 2) if arrival_span else 'instant burst'}
        for field, prefix in (('first_token_ms', 'First token'), ('total_ms', 'Total time')):
            for q in (50, 95, 99):
                a[f'{prefix} p{q} (ms)'] = percentile(rows, field, q)
        latency.append(a)
        token_values = [number(r['tokens_in']) + number(r['tokens_out']) for r in success
                        if number(r['tokens_in']) is not None and number(r['tokens_out']) is not None]
        queues = [number(r['queue_length']) for r in rows if number(r['queue_length']) is not None]
        costs = [number(r['estimated_cost_usd']) for r in rows]
        outcomes.append({'Scenario': scenario, 'Real or stand-in': source,
            'Succeeded %': f'{len(success) * 100 / n:.1f}',
            'Rate-limited %': f'{sum(r["status"] == "rate_limited" for r in rows) * 100 / n:.1f}',
            'Timed out or given up %': f'{sum(r["status"] == "timeout" for r in rows) * 100 / n:.1f}',
            'Retries per request': round(sum(number(r['retries']) or 0 for r in rows) / n, 3),
            'Peak queue length': max(queues) if queues else 'not measured',
            'Tokens per answer': round(sum(token_values) / len(success), 2)
                if success and len(token_values) == len(success) else 'not measured',
            'Cost per 1000 answers (USD estimated)': round(sum(costs) * 1000 / len(success), 6)
                if success and all(v is not None for v in costs) else 'not measured'})
    return latency, outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--load-directory', default='results/final_load_lab_complete')
    parser.add_argument('--output', default='results/submission_pdf_aligned')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = export_requests(Path(args.load_directory))
    write_csv(output / 'results.csv', records)
    latency, outcomes = tables(records)
    write_csv(output / 'table2a.csv', latency)
    write_csv(output / 'table2b.csv', outcomes)
    limits = [
        {'Limit': 'Requests per minute', 'Value you found': 30,
         'How you found it': 'user account probe; 429s in groq_probe.csv', 'Confidence': 'medium'},
        {'Limit': 'Tokens per minute', 'Value you found': 8000,
         'How you found it': 'real x-ratelimit-limit-tokens header', 'Confidence': 'high'},
        {'Limit': 'Requests per day', 'Value you found': 1000,
         'How you found it': 'real x-ratelimit-limit-requests header', 'Confidence': 'high'},
        {'Limit': 'Tokens per day', 'Value you found': 'not measured', 'How you found it': 'not measured', 'Confidence': 'low'},
        {'Limit': 'Requests in flight at once', 'Value you found': 'not measured', 'How you found it': 'not measured', 'Confidence': 'low'},
        {'Limit': 'What it returns when a limit is hit', 'Value you found': 'HTTP 429; Retry-After',
         'How you found it': 'raw probe http_status/retry_after', 'Confidence': 'high'},
        {'Limit': 'Time until it recovers', 'Value you found': 'not measured',
         'How you found it': 'Retry-After observed; actual recovery not verified', 'Confidence': 'low'},
    ]
    write_csv(output / 'table1.csv', limits)
    retrieval = [{'Setting': f'{size} whitespace tokens; top-K {k}; re-rank no',
        'Questions asked': 'not measured', 'Correct and cited %': 'not measured', 'Wrong %': 'not measured',
        'Correctly refused %': 'not measured', 'Tokens per answer': 'not measured',
        'First token p50 / p95 (ms)': 'not measured', 'Total time p50 / p95 (ms)': 'not measured'}
        for size, k in ((300, 3), (500, 5), (800, 7))]
    write_csv(output / 'table3.csv', retrieval)
    (output / 'schema_notes.json').write_text(json.dumps({
        'assignment_pdf': 'The_Relay_Assignment_Brief.pdf', 'complete': False,
        'required_columns': REQUIRED_COLUMNS, 'additional_diagnostics': EXTRA_COLUMNS,
        'blank_answer_grades': 'Not measured; no simulated answer-quality grades invented',
        'stand_in_scope': 'Virtual service simulation; pipeline-under-HTTP-load remains outstanding',
        'first_token': 'Arrival to visible token; RAG buffered until validation. Provider TTFT separately stored.',
        'table_percentiles': 'Available request measurements; missing values excluded; total includes failed requests.',
        'success': 'Transport/application completion; quality is independently recorded in answer_grade.',
        'simulated_start_time': 'Declared virtual anchor, not measured execution wall time.'}, indent=2))
    print(f'Exported {len(records)} request rows and required tables to {output}')


if __name__ == '__main__':
    main()

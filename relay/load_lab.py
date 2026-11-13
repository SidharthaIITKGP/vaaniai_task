"""Empirical service calibration and virtual-time load experiments; never calls Groq.

The simulator models service timing and capacity, not answer quality. HTTP 429
observations calibrate capacity diagnostics, not a random independent failure rate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, deque
from pathlib import Path
from typing import Any

import numpy as np


DAILY_RESPONSE = (
    "I can’t answer from the papers right now because today’s model request quota is exhausted. "
    "No answer has been generated. Please try again after the provider’s quota resets."
)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def calibrate(probe: Path, rag_files: list[Path]) -> dict:
    records: list[dict] = []
    bundles: dict[str, list[list[dict]]] = {}
    provenance = []
    counts = Counter()
    for path in [probe, *rag_files]:
        if not path.exists():
            continue
        provenance.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        with path.open(newline='', encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            if path == probe:
                counts[row.get('status', 'unknown')] += 1
                if row.get('status') != 'ok':
                    continue
                try:
                    records.append({'stage': 'probe', 'ttft_ms': float(row['first_token_ms']),
                        'latency_ms': float(row['total_ms']) - float(row.get('queue_ms') or 0),
                        'input_tokens': int(row['tokens_in']), 'output_tokens': int(row['tokens_out'])})
                except (KeyError, ValueError):
                    continue
            else:
                calls = json.loads(row.get('llm_calls_json') or '[]')
                complete = []
                for call in calls:
                    if any(call.get(key) is None for key in ('latency_ms', 'input_tokens', 'output_tokens')):
                        continue
                    record = {'stage': call['stage'], 'latency_ms': call['latency_ms'],
                              'ttft_ms': call.get('provider_ttft_ms'),
                              'input_tokens': call['input_tokens'], 'output_tokens': call['output_tokens']}
                    records.append(record)
                    complete.append(record)
                if (complete and len(complete) == len(calls)
                        and row.get('status') == 'ok'):
                    bundles.setdefault(row['architecture'], []).append(complete)
    if not records:
        raise ValueError('No usable provider observations')
    return {'measurement_type': 'calibration_from_real_logs', 'sources': provenance,
            'records': records, 'bundles': bundles, 'probe_outcomes': dict(counts),
            'limits': {'rpm': 30, 'tpm': 8000, 'rpd': 1000},
            'limits_basis': 'user account measurements; not live quota verification',
            'limitations': ['Empirical bootstrap; no validated extrapolation to unseen token lengths.',
                'Historical RAG revisions are mixed; service timing only.',
                'Capacity 429s are modeled through quotas, not sampled as independent failures.',
                'Unknown failure latencies and non-streamed TTFT remain missing.']}


class EmpiricalService:
    """Reusable service model for virtual time and the standalone HTTP server."""
    def __init__(self, calibration: dict, seed: int = 42):
        self.calibration = calibration
        self.rng = random.Random(seed)

    def sample(self, stage: str | None = None) -> dict:
        records = self.calibration['records']
        eligible = [r for r in records if r['stage'] == stage] or records
        return dict(self.rng.choice(eligible))

    def bundle(self, architecture: str | None) -> list[dict]:
        if architecture is None:
            return [self.sample('probe')]
        values = self.calibration['bundles'].get(architecture, [])
        if not values:
            raise ValueError(f'No complete real call bundles for {architecture}')
        return self.rng.choice(values)


def simulate(calibration: dict, scenario: str, *, seed: int = 42,
             architecture: str | None = None, count: int = 120,
             retries: int = 0, queue_limit: int = 8, queue_deadline: float = 10,
             input_rate: float | None = None, output_rate: float | None = None) -> list[dict]:
    """Deterministic virtual-time FIFO load simulation with bounded admission.

    Token admission uses the same token-bucket/RPM-window policy as the local
    scheduler. Each sampled bundle is reserved atomically, conservatively; this
    simulator is not an execution of the answer-generating RAG pipeline.
    """
    service = EmpiricalService(calibration, seed)
    rpm = 15 if scenario == 'halved_rpm' else 30
    rpd = 0 if scenario == 'daily_exhausted' else 1000
    arrivals = [index * (3 if scenario in {'steady', 'halved_rpm'} else .1)
                for index in range(count)]
    if scenario == 'sustained_overload':
        arrivals = [index * .5 for index in range(count * 5)]
    if scenario == 'burst':
        arrivals = [0.0] * count
    if scenario == 'daily_exhausted':
        arrivals = [15 * 3600.0] * count
    tokens = 6400.0
    updated = arrivals[0] if arrivals else 0.0
    dispatch_floor = updated
    request_times: deque[float] = deque()
    pending_finishes: deque[float] = deque()
    daily_used = 0
    rows = []
    for index, arrival in enumerate(arrivals):
        bundle = service.bundle(architecture)
        n_calls = len(bundle)
        inputs = sum(c['input_tokens'] for c in bundle)
        outputs = sum(c['output_tokens'] for c in bundle)
        total_tokens = inputs + outputs
        processing = sum(c['latency_ms'] for c in bundle) / 1000
        while pending_finishes and pending_finishes[0] <= arrival:
            pending_finishes.popleft()
        queue_length = len(pending_finishes)
        status = 'ok'
        wait = 0.0
        dispatched = None
        retry_count = 0
        if daily_used + n_calls > rpd:
            status = 'daily_exhausted'
        elif queue_length >= queue_limit:
            status = 'queue_full'
        elif total_tokens > 6400:
            status = 'request_too_large'
        else:
            candidate = max(arrival, dispatch_floor)
            tokens = min(6400.0, tokens + max(0, candidate - updated) * 6400 / 60)
            updated = candidate
            while request_times and request_times[0] <= candidate - 60:
                request_times.popleft()
            token_wait = max(0, total_tokens - tokens) / (6400 / 60)
            rpm_wait = 0.0
            if len(request_times) + n_calls > int(rpm * .8):
                remove = len(request_times) + n_calls - int(rpm * .8)
                rpm_wait = max(0, list(request_times)[remove - 1] + 60 - candidate)
            quota_wait = max(token_wait, rpm_wait)
            # Retry policy extends an admission deadline by bounded Retry-After waits.
            wait = candidate + quota_wait - arrival
            while wait > queue_deadline * (retry_count + 1) and retry_count < retries:
                retry_count += 1
            if wait > queue_deadline * (retry_count + 1):
                status = 'admission_timeout'
                wait = queue_deadline * (retry_count + 1)
            else:
                dispatched = candidate + quota_wait
                tokens = min(6400.0, tokens + quota_wait * 6400 / 60) - total_tokens
                updated = dispatched
                dispatch_floor = dispatched
                daily_used += n_calls
                request_times.extend([dispatched] * n_calls)
                pending_finishes.append(dispatched + processing)
        completed = status == 'ok'
        cost = (((inputs * input_rate + outputs * output_rate) / 1e6 if completed else 0.0)
                if input_rate is not None and output_rate is not None else None)
        rows.append({'source': 'simulated', 'scenario': scenario, 'request_id': f'{scenario}-{index}',
            'seed': seed, 'architecture': architecture or 'standalone_llm',
            'arrival_s': arrival, 'dispatch_s': dispatched, 'status': status,
            'queue_length_at_arrival': queue_length, 'queue_wait_ms': wait * 1000,
            'processing_ms': processing * 1000 if completed else None,
            'latency_ms': (wait + processing) * 1000 if completed else wait * 1000,
            'provider_ttft_ms': bundle[0]['ttft_ms'] if completed else None,
            'user_visible_ttft_ms': (wait + processing) * 1000 if completed else None,
            'retries': retry_count, 'retry_policy': 'bounded admission retries; no daily retry',
            'input_tokens': inputs if completed else 0, 'output_tokens': outputs if completed else 0,
            'model_calls': n_calls if completed else 0, 'estimated_cost_usd': cost,
            'rate_limit_rpm': rpm, 'rate_limit_tpm': 8000,
            'response': DAILY_RESPONSE if status == 'daily_exhausted' else ''})
    return rows


def summarize(rows: list[dict]) -> dict:
    success = [r for r in rows if r['status'] == 'ok']
    result = {'source': 'simulated', 'scenario': rows[0]['scenario'],
        'architecture': rows[0]['architecture'], 'requests': len(rows), 'completed': len(success),
        'outcomes_json': json.dumps(dict(Counter(r['status'] for r in rows))),
        'retry_total': sum(r['retries'] for r in rows),
        'max_queue_length': max(r['queue_length_at_arrival'] for r in rows),
        'input_tokens': sum(r['input_tokens'] for r in rows),
        'output_tokens': sum(r['output_tokens'] for r in rows),
        'estimated_cost_usd': (sum(r['estimated_cost_usd'] for r in rows)
            if all(r['estimated_cost_usd'] is not None for r in rows) else None)}
    for key in ('latency_ms', 'queue_wait_ms', 'processing_ms'):
        for percentile in (50, 95, 99):
            result[f'{key}_p{percentile}'] = (float(np.percentile([r[key] for r in success], percentile))
                                           if success else None)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/load_lab')
    parser.add_argument('--architecture', choices=('fast', 'verified', 'adaptive'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--count', type=int, default=120)
    parser.add_argument('--input-usd-per-million', type=float)
    parser.add_argument('--output-usd-per-million', type=float)
    args = parser.parse_args()
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    calibration = calibrate(Path('groq_probe.csv'), sorted(Path('results').glob('*smoke*.csv')))
    (directory / 'calibration.json').write_text(json.dumps(calibration, indent=2))
    summaries = []
    slow = []
    for scenario in ('steady', 'burst', 'sustained_overload', 'halved_rpm', 'daily_exhausted'):
        for retries in (0, 2):
            rows = simulate(calibration, scenario, seed=args.seed, architecture=args.architecture,
                count=args.count, retries=retries, input_rate=args.input_usd_per_million,
                output_rate=args.output_usd_per_million)
            write_csv(directory / f'{scenario}_retries{retries}.csv', rows)
            summary = summarize(rows)
            summary['max_admission_retries'] = retries
            summaries.append(summary)
            slow.extend(sorted(rows, key=lambda r: r['latency_ms'], reverse=True)[:3])
    write_csv(directory / 'results.csv', summaries)
    write_csv(directory / 'slow_requests.csv', slow)
    (directory / 'quota_3pm.json').write_text(json.dumps({
        'source': 'simulated_policy_response', 'local_time': '15:00 Asia/Kolkata',
        'http_status': 503, 'error_code': 'daily_quota_exhausted',
        'retry_after_seconds': None, 'model_calls': 0, 'response': DAILY_RESPONSE}, indent=2))
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()

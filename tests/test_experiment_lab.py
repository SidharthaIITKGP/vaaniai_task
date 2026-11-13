import csv
from pathlib import Path

from relay.experiment_review import compare, packet
from relay.load_lab import simulate, summarize
from relay.verification import VerifierOutput
from pydantic import ValidationError
import pytest


def calibration():
    record = {'stage': 'probe', 'input_tokens': 84, 'output_tokens': 60,
              'ttft_ms': 500, 'latency_ms': 800}
    return {'records': [record], 'bundles': {'fast': [[record]]}}


def test_load_simulation_reproducible_and_daily_fail_closed():
    assert simulate(calibration(), 'burst', count=12) == simulate(calibration(), 'burst', count=12)
    daily = simulate(calibration(), 'daily_exhausted', count=12, retries=2)
    assert all(r['status'] == 'daily_exhausted' for r in daily)
    assert sum(r['model_calls'] for r in daily) == 0
    assert sum(r['retries'] for r in daily) == 0
    assert summarize(daily)['latency_ms_p95'] is None


def test_ungraded_ok_status_never_selects_winner(tmp_path):
    path = tmp_path / 'runs.csv'
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['request_id', 'question_id', 'question',
             'answer', 'architecture', 'status'])
        writer.writeheader()
        writer.writerow({'request_id': 'r', 'question_id': 'T1', 'question': 'test',
                         'answer': 'unsupported', 'architecture': 'fast', 'status': 'ok'})
    packet([path], tmp_path / 'review')
    selection = compare([path], tmp_path / 'review/grades.csv', tmp_path / 'comparison')
    assert selection['winner'] is None
    assert selection['complete'] is False


def test_verifier_missing_fields_is_a_local_error():
    with pytest.raises(ValidationError):
        VerifierOutput.model_validate({'decision': 'return'})


def test_simulator_steady_usage_and_cost_are_explicit():
    rows = simulate(calibration(), 'steady', count=5)
    assert sum(r['input_tokens'] for r in rows) == 420
    assert all(r['estimated_cost_usd'] is None for r in rows)
    assert all(r['latency_ms'] >= r['processing_ms'] for r in rows)


def test_integration_rejects_incomplete_experiments(tmp_path):
    from relay.finalize import deployment
    path = tmp_path / 'selection.json'
    path.write_text('{"complete":false,"winner":null}')
    with pytest.raises(ValueError, match='independently graded'):
        deployment(path, path, tmp_path / 'variants.json')

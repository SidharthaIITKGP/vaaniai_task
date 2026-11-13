import csv
import json
from datetime import datetime, timezone

import httpx
import pytest

from relay.evaluate import history_hold, parser, quota_state_path, run


def test_account_state_and_lock_paths_are_separate_and_stable(tmp_path):
    base = tmp_path / 'quota.json'
    assert quota_state_path(base) == base
    assert quota_state_path(base, 'account2') == tmp_path / 'quota.account2.json'
    assert quota_state_path(base, 'account2').with_suffix('.lock') != base.with_suffix('.lock')
    with pytest.raises(ValueError, match='Quota profile'):
        quota_state_path(base, '../account2')


def test_historical_daily_holds_follow_account_not_completed_answers():
    failure = {'question_id': 'T1', 'architecture': 'verified',
               'run_completed_at': datetime.now(timezone.utc).isoformat(),
               'provider_error_json': json.dumps({'status_code': 429,
                   'rate_limit_type': 'provider_tpd', 'retry_after_seconds': 1077})}
    assert history_hold([failure]) is not None  # Untagged legacy rows belong to default.
    assert history_hold([failure], 'account2') is None
    new_failure = dict(failure, quota_profile='account2')
    assert history_hold([new_failure], 'account2') is not None
    assert history_hold([new_failure]) is None


@pytest.mark.asyncio
async def test_account_switch_preserves_old_hold_and_resumes_only_failed(
        monkeypatch, tmp_path, config, retriever):
    import relay.evaluate as evaluation
    monkeypatch.setenv('GROQ_API_KEY', 'old-test-only')
    config.index_override = tmp_path / 'index.jsonl'
    config.index_path.write_text('\n'.join(json.dumps(c.to_dict()) for c in retriever.chunks))
    monkeypatch.setattr(evaluation, 'RelayConfig', lambda: config)
    monkeypatch.setattr(evaluation, 'HybridRetriever', lambda *args: retriever)
    requests = []
    async def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(429, request=request, headers={'retry-after': '1077'},
                json={'error': {'message': 'tokens per day (TPD) limit reached'}})
        assert request.headers['authorization'] == 'Bearer new-test-only'
        content = {'decision': 'answer', 'claims': [
            {'text': 'Attention uses heads', 'sources': ['S1']}]}
        return httpx.Response(200, request=request, json={
            'choices': [{'message': {'content': json.dumps(content)}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 20, 'completion_tokens': 8}})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        real_provider = evaluation.BudgetedGroqProvider
        monkeypatch.setattr(evaluation, 'BudgetedGroqProvider',
            lambda *args, **kwargs: real_provider(*args, client=client, **kwargs))
        command = ['--question-id', 'T1', '--architectures', 'fast',
                   '--output', str(tmp_path / 'runs.csv'), '--quota-state', str(tmp_path / 'quota.json')]
        assert await run(parser().parse_args(command)) == 2
        monkeypatch.setenv('GROQ_API_KEY', 'new-test-only')
        # A changed key by itself does not remove an account's quota hold.
        assert await run(parser().parse_args(command)) == 2
        assert len(requests) == 1
        old_state = (tmp_path / 'quota.json').read_bytes()
        new_account = parser().parse_args([*command, '--quota-profile', 'account2'])
        assert await run(new_account) == 0
        assert await run(new_account) == 0
    assert len(requests) == 2
    assert (tmp_path / 'quota.json').read_bytes() == old_state
    assert (tmp_path / 'quota.account2.json').exists()
    rows = list(csv.DictReader((tmp_path / 'runs.csv').open()))
    assert len(rows) == 1 and rows[0]['status'] == 'ok'
    assert rows[0]['quota_profile'] == 'account2'
    events = [json.loads(line)['row'] for line in (tmp_path / 'runs.attempts.jsonl').read_text().splitlines()]
    assert [r['quota_profile'] for r in events] == ['default', 'account2']


@pytest.mark.asyncio
async def test_new_account_daily_exhaustion_still_stops_and_persists(monkeypatch, tmp_path, config, retriever):
    import relay.evaluate as evaluation
    monkeypatch.setenv('GROQ_API_KEY', 'test-only')
    config.index_override = tmp_path / 'index.jsonl'
    config.index_path.write_text('\n'.join(json.dumps(c.to_dict()) for c in retriever.chunks))
    monkeypatch.setattr(evaluation, 'RelayConfig', lambda: config)
    monkeypatch.setattr(evaluation, 'HybridRetriever', lambda *args: retriever)
    requests = []
    async def handler(request):
        requests.append(request)
        return httpx.Response(429, request=request, headers={'retry-after': '526'},
            json={'error': {'message': 'tokens per day (TPD) limit reached'}})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        real_provider = evaluation.BudgetedGroqProvider
        monkeypatch.setattr(evaluation, 'BudgetedGroqProvider',
            lambda *args, **kwargs: real_provider(*args, client=client, **kwargs))
        args = parser().parse_args(['--question-id', 'T1', '--architectures', 'fast',
            '--output', str(tmp_path / 'runs.csv'), '--quota-state', str(tmp_path / 'quota.json'),
            '--quota-profile', 'account2'])
        assert await run(args) == 2
        assert await run(args) == 2
    assert len(requests) == 1
    assert not (tmp_path / 'quota.json').exists()
    state = json.loads((tmp_path / 'quota.account2.json').read_text())
    assert state['stop_reason']['rate_limit_type'] == 'provider_tpd'

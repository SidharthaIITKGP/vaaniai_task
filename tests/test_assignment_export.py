from relay.assignment_export import REQUIRED_COLUMNS, tables
from relay.conversation_export import export_visible
import json


def test_table_percentages_and_missing_metrics():
    rows = [{'scenario': 'steady', 'source': 'stand-in',
             'start_time': '2026-11-13T00:00:00+00:00', 'first_token_ms': None,
             'total_ms': 0, 'status': 'rate_limited', 'retries': 0,
             'tokens_in': 0, 'tokens_out': 0, 'queue_length': 0,
             'estimated_cost_usd': 0}]
    latency, outcomes = tables(rows)
    assert latency[0]['First token p50 (ms)'] == 'not measured'
    assert outcomes[0]['Succeeded %'] == '0.0'
    assert outcomes[0]['Rate-limited %'] == '100.0'
    assert outcomes[0]['Tokens per answer'] == 'not measured'
    assert REQUIRED_COLUMNS[-1] == 'answer_grade'


def test_visible_history_omits_private_reasoning_and_keeps_real_messages(tmp_path):
    session = tmp_path / 'session.jsonl'
    def message(role, channel, text):
        return {'type': 'response_item', 'payload': {'type': 'message', 'role': role,
            'channel': channel, 'content': [{'type': 'output_text', 'text': text}]}}
    session.write_text('\n'.join(json.dumps(r) for r in [message('user', None, 'hello'),
        message('assistant', 'analysis', 'private'), message('assistant', 'final', 'reply')]))
    output = tmp_path / 'history.json'
    assert export_visible(session, output) == 2
    exported = json.loads(output.read_text())
    assert [r['text'] for r in exported['messages']] == ['hello', 'reply']

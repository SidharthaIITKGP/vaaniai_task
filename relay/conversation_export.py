"""Export visible user/assistant conversation messages for the assignment appendix."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def export_visible(session: Path, output: Path) -> int:
    messages = []
    with session.open(encoding='utf-8') as handle:
        for line in handle:
            record = json.loads(line)
            payload = record.get('payload', {})
            if record.get('type') != 'response_item' or payload.get('type') != 'message':
                continue
            if payload.get('role') not in {'user', 'assistant'}:
                continue
            if payload.get('channel') not in {None, 'final', 'commentary'}:
                continue
            text = '\n'.join(block.get('text', '') for block in payload.get('content', [])
                             if block.get('type') in {'input_text', 'output_text'})
            if not text:
                continue
            messages.append({'timestamp': record.get('timestamp'), 'role': payload['role'],
                'channel': payload.get('channel'),
                'text': re.sub(r'gsk_[A-Za-z0-9_-]{20,}', '[API KEY REDACTED]', text)})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'export_scope': 'Visible messages in original order; no rewriting. '
        'System/developer instructions, private reasoning and tool traces excluded; keys redacted.',
        'messages': messages}, indent=2, ensure_ascii=False), encoding='utf-8')
    return len(messages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True)
    parser.add_argument('--output', default='results/submission_pdf_aligned/conversation_history.json')
    args = parser.parse_args()
    print(f'Exported {export_visible(Path(args.session), Path(args.output))} visible messages')


if __name__ == '__main__':
    main()

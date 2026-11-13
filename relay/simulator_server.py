"""Standalone timing/capacity stand-in. Responses are labeled simulated refusals."""
from __future__ import annotations

import argparse
import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .load_lab import EmpiricalService


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', default='results/load_lab/calibration.json')
    parser.add_argument('--port', type=int, default=8081)
    parser.add_argument('--rpm', type=int, default=30)
    parser.add_argument('--tpm', type=int, default=8000)
    parser.add_argument('--rpd', type=int, default=1000)
    args = parser.parse_args()
    service = EmpiricalService(json.loads(Path(args.calibration).read_text()))
    lock = threading.Lock()
    requests = deque()
    daily = [0]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers.get('Content-Length', '0'))))
            now = time.monotonic()
            with lock:
                sample = service.sample(payload.get('stage', 'probe'))
                while requests and now - requests[0][0] >= 60:
                    requests.popleft()
                tokens = sample['input_tokens'] + sample['output_tokens']
                exhausted = daily[0] >= args.rpd
                limited = len(requests) >= args.rpm or sum(r[1] for r in requests) + tokens > args.tpm
                if not exhausted and not limited:
                    requests.append((now, tokens))
                    daily[0] += 1
            if exhausted or limited:
                self.send_response(429)
                if not exhausted:
                    self.send_header('Retry-After', str(max(1, int(60 - (now - requests[0][0])))))
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'source': 'simulated', 'error': {
                    'code': 'daily_quota_exhausted' if exhausted else 'rate_limit_exceeded'}}).encode())
                return
            text = '[SIMULATED] The provided papers do not establish an answer to this question.'
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream' if payload.get('stream') else 'application/json')
            self.end_headers()
            if payload.get('stream'):
                ttft = min(sample['ttft_ms'] or sample['latency_ms'], sample['latency_ms']) / 1000
                time.sleep(ttft)
                self.wfile.write(('data: ' + json.dumps({'choices': [{'delta': {'content': text}}]}) + '\n\n').encode())
                self.wfile.flush()
                time.sleep(max(0, sample['latency_ms'] / 1000 - ttft))
                self.wfile.write(('data: ' + json.dumps({'choices': [], 'usage': {
                    'prompt_tokens': sample['input_tokens'], 'completion_tokens': sample['output_tokens']}})
                    + '\n\ndata: [DONE]\n\n').encode())
            else:
                time.sleep(sample['latency_ms'] / 1000)
                self.wfile.write(json.dumps({'source': 'simulated',
                    'choices': [{'message': {'content': text}}], 'usage': {
                    'prompt_tokens': sample['input_tokens'], 'completion_tokens': sample['output_tokens']}}).encode())

    print(f'Simulated timing service on http://127.0.0.1:{args.port}; no answer-quality claims')
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()

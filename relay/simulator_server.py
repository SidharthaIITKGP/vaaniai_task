"""Standalone timing/capacity stand-in. Responses are labeled simulated refusals."""
from __future__ import annotations

import argparse
import json
import threading
import time
import random
import re
from dataclasses import dataclass
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .load_lab import EmpiricalService
from .providers.scheduler import estimate_input_tokens


@dataclass
class SimulationPolicy:
    rpm: int = 30
    tpm: int = 8000
    rpd: int = 1000
    time_scale: float = 1.0
    seed: int = 42
    failure_every: int = 0
    fail_first: int = 0
    slow_every: int = 11
    capacity: int = 4


def fixture_content(payload):
    """Protocol fixture, never a claim of simulated answer correctness."""
    schema = (payload.get('response_format') or {}).get('json_schema') or {}
    if schema.get('name') == 'intent_classification':
        return json.dumps({'intent':'paper_question','complexity':'simple','relevant_papers':[]})
    if schema.get('name') == 'answer_verification':
        return json.dumps({'decision':'refuse','unsupported_claims':[],
                           'reason':'Stand-in does not perform semantic grading.'})
    if schema.get('name') == 'grounded_answer':
        aliases = schema['schema']['properties']['claims']['items']['properties']['sources']['items']['enum']
        evidence = next((m['content'].split('EVIDENCE:\n',1)[1] for m in payload['messages']
                         if 'EVIDENCE:\n' in m['content']), '')
        # Quote an actual retrieved excerpt for syntactic pipeline execution only.
        body = evidence.split('\n',1)[-1].split('\n\n---')[0].strip()[:200]
        body = re.sub(r'\[S\d+\]', '', body)
        return json.dumps({'decision':'answer','claims':[{'text':body or 'Simulation fixture',
                                                         'sources':aliases[:1]}]})
    return '[SIMULATED protocol fixture; answer quality not measured]'


def create_server(calibration: dict, policy: SimulationPolicy, port: int = 0):
    """Existing empirical timing model, real loopback HTTP, strict rolling reservations."""
    if policy.time_scale <= 0:
        raise ValueError('time_scale must be positive')
    service = EmpiricalService(calibration, policy.seed)
    guard = threading.Lock()
    state = {'window':deque(), 'daily':0, 'active':0, 'ordinal':0, 'events':[]}
    clock = lambda: time.monotonic()/policy.time_scale

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers.get('Content-Length','0'))))
            started = clock()
            stage = payload.get('simulation_stage','generation')
            request_id = payload.get('simulation_request_id','unassigned')
            inputs = estimate_input_tokens(payload.get('messages',[])) + len(json.dumps(payload.get('response_format',{})))//4
            cap = int(payload.get('max_completion_tokens',640))
            content = fixture_content(payload)
            outputs = min(cap, max(1,(len(content)+3)//4))
            status, retry, error = 200, None, ''
            with guard:
                while state['window'] and state['window'][0]['at'] <= started-60:
                    state['window'].popleft()
                used = sum(x['tokens'] for x in state['window'])
                if state['daily'] >= policy.rpd:
                    status,error = 429,'requests per day (RPD) quota exhausted'
                elif inputs+cap > policy.tpm:
                    status,error = 429,'tokens per minute (TPM): request too large'
                elif len(state['window']) >= policy.rpm or used+inputs+cap > policy.tpm:
                    status,error = 429,'requests per minute (RPM)' if len(state['window'])>=policy.rpm else 'tokens per minute (TPM)'
                    retry = max(.1, 60-(started-state['window'][0]['at'])) if state['window'] else 60
                elif state['active'] >= policy.capacity:
                    status,error,retry = 503,'stand-in in-flight capacity exceeded',.5
                if status == 200:
                    state['daily'] += 1
                    state['active'] += 1
                    state['ordinal'] += 1
                    ordinal = state['ordinal']
                    reservation = {'at':started,'tokens':inputs+cap}
                    state['window'].append(reservation)
                    sample = service.sample(stage)
                    active = state['active']
                    failed = ordinal <= policy.fail_first or (policy.failure_every and ordinal%policy.failure_every==0)
                    if failed:
                        status,error = 500,'injected transient model failure (synthetic assumption)'
                        outputs = 0
                    latency = float(sample['latency_ms'])*(1+.35*(active-1)**2)
                    if policy.slow_every and ordinal%policy.slow_every==0:
                        latency *= 5
                    ttft = min(float(sample.get('ttft_ms') or latency*.5),latency)
                else:
                    sample, latency, ttft, outputs, reservation = None,0,0,0,None
            event = {'request_id':request_id,'stage':stage,'source':'stand-in',
                     'started_virtual_s':started,'status':status,'retry_after':retry,
                     'tokens_in':inputs if reservation else 0,'tokens_out':outputs,
                     'reserved_tokens':inputs+cap if reservation else 0,
                     'sample':sample, 'configured_service_ms':latency,'client_disconnected':False}
            try:
                if reservation:
                    time.sleep(ttft/1000*policy.time_scale)
                if status != 200:
                    if reservation:
                        time.sleep((latency-ttft)/1000*policy.time_scale)
                    self.send_response(status)
                    self.send_header('Content-Type','application/json')
                    if retry is not None:self.send_header('Retry-After',str(retry))
                    self.end_headers()
                    self.wfile.write(json.dumps({'source':'stand-in','error':{'message':error},
                        'usage':{'prompt_tokens':inputs if reservation else 0,'completion_tokens':outputs}}).encode())
                elif payload.get('stream'):
                    self.send_response(200)
                    self.send_header('Content-Type','text/event-stream')
                    self.end_headers()
                    parts = [content[i:i+24] for i in range(0,len(content),24)]
                    for part in parts:
                        self.wfile.write(('data: '+json.dumps({'choices':[{'delta':{'content':part}}]})+'\n\n').encode())
                        self.wfile.flush()
                        time.sleep((latency-ttft)/1000*policy.time_scale/max(1,len(parts)))
                    self.wfile.write(('data: '+json.dumps({'choices':[],'usage':{
                        'prompt_tokens':inputs,'completion_tokens':outputs}})+'\n\ndata: [DONE]\n\n').encode())
                    self.wfile.flush()
                else:
                    time.sleep((latency-ttft)/1000*policy.time_scale)
                    self.send_response(200)
                    self.send_header('Content-Type','application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'source':'stand-in','choices':[{'message':{'content':content},
                        'finish_reason':'stop'}],'usage':{'prompt_tokens':inputs,'completion_tokens':outputs}}).encode())
            except (BrokenPipeError,ConnectionResetError):
                event['client_disconnected'] = True
            finally:
                with guard:
                    if reservation:
                        reservation['tokens'] = inputs+outputs
                        state['active'] -= 1
                    event['completed_virtual_s'] = clock()
                    state['events'].append(event)

    server = ThreadingHTTPServer(('127.0.0.1',port),Handler)
    server.daemon_threads = True
    server.simulation_state = state
    server.simulation_policy = policy
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', default='results/load_lab/calibration.json')
    parser.add_argument('--port', type=int, default=8081)
    parser.add_argument('--rpm', type=int, default=30)
    parser.add_argument('--tpm', type=int, default=8000)
    parser.add_argument('--rpd', type=int, default=1000)
    parser.add_argument('--pipeline-fixtures',action='store_true')
    parser.add_argument('--time-scale',type=float,default=1)
    args = parser.parse_args()
    if args.pipeline_fixtures:
        server = create_server(json.loads(Path(args.calibration).read_text()),
            SimulationPolicy(rpm=args.rpm,tpm=args.tpm,rpd=args.rpd,time_scale=args.time_scale),args.port)
        print(f'Pipeline stand-in on http://127.0.0.1:{args.port}; fixtures are not quality measurements',flush=True)
        server.serve_forever()
        return
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

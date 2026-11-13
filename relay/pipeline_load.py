"""Execute the real Fast RAG pipeline against loopback HTTP; never contacts Groq."""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import threading
import time
from collections import Counter
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import numpy as np

from .config import RelayConfig
from .evaluate import FORMAL_18
from .evaluation_questions import QUESTIONS
from .ingestion import load_chunks
from .instrumentation import JsonlLogger
from .load_lab import calibrate, write_csv, DAILY_RESPONSE
from .pipeline import RAGPipeline
from .providers.stand_in import SIM_REQUEST, StandInProvider
from .retrieval import HybridRetriever
from .simulator_server import SimulationPolicy, create_server


SCENARIOS = ('steady','burst','sustained_overload','halved_rpm','daily_exhausted')
ANCHOR = datetime(2026,11,14,9,30,tzinfo=timezone.utc)  # Declared virtual 3 PM IST, not wall time.


async def scenario_run(name, calibration, retriever, config, directory, *, count=30, scale=.03, retries=1,
                       fail_first=0, tpm=8000):
    policy = SimulationPolicy(rpm=15 if name=='halved_rpm' else 30,tpm=tpm,
        rpd=0 if name=='daily_exhausted' else 1000,time_scale=scale,
        failure_every=13 if name=='sustained_overload' else 0,fail_first=fail_first)
    server = create_server(calibration,policy)
    thread = threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    endpoint = f'http://127.0.0.1:{server.server_address[1]}/openai/v1/chat/completions'
    queue = asyncio.Queue(maxsize=8)
    rows, details = [],[]
    start = time.perf_counter()
    n = count*3 if name=='sustained_overload' else count
    interval = 65 if name in {'steady','halved_rpm'} else 3
    arrivals = [0 if name in {'burst','daily_exhausted','retry_probe'} else i*interval for i in range(n)]
    questions = [q for q in QUESTIONS if q.question_id in FORMAL_18]
    prefix = f'{name}_retries{retries}'

    async with httpx.AsyncClient(timeout=15,trust_env=False) as client:
        provider = StandInProvider(endpoint,client,time_scale=scale,retries=retries)

        def record(index,arrival,status,original,response='',result=None,wait=0,peak=0):
            elapsed = max(0,(time.perf_counter()-start)/scale-arrival)*1000
            rid = f'{prefix}:{index}'
            metrics = result.metrics if result else {}
            first = metrics.get('user_visible_ttft_ms')
            first = min(elapsed, wait+first/scale) if first is not None else None
            row = {'request_id':rid,'scenario':prefix,'source':'stand-in',
                'start_time':(ANCHOR+timedelta(seconds=arrival)).isoformat(),
                'first_token_ms':first,'total_ms':elapsed,'status':status,'retries':0,
                'tokens_in':0,'tokens_out':0,'retrieval_setting':'fast:500/5:no_rerank:16000char_budget',
                'answer_grade':'','queue_wait_ms':wait,'processing_ms':max(0,elapsed-wait),
                'queue_length':peak,'original_status':original,'response':response,
                'question_id':questions[index%len(questions)].question_id,
                'pipeline_executed':result is not None,'pipeline_request_id':result.request_id if result else '',
                'time_basis':'accelerated_wall_clock','time_scale':scale,
                'token_measurement':'simulated character-based token estimate',
                'estimated_cost_usd':0.0,'model_calls':0,'retry_wait_ms':0.0,
                'simulation_fixture_quality':'not measured'}
            rows.append(row)
            details.append({'request_id':rid,'metrics_wall_ms':metrics,'response':response,
                            'answer_quality_measured':False})

        async def offer(index,arrival):
            await asyncio.sleep(max(0,start+arrival*scale-time.perf_counter()))
            if policy.rpd==0:
                record(index,arrival,'rate_limited','daily_exhausted',DAILY_RESPONSE)
                return
            try:
                queue.put_nowait((index,arrival,queue.qsize()+1))
            except asyncio.QueueFull:
                record(index,arrival,'rate_limited','queue_full','The assistant is busy. Please try again later.',peak=8)

        async def worker():
            while True:
                item = await queue.get()
                if item is None:
                    queue.task_done()
                    return
                index,arrival,peak = item
                wait = max(0,(time.perf_counter()-start)/scale-arrival)*1000
                try:
                    if wait>=10000:
                        record(index,arrival,'timeout','queue_deadline',wait=wait,peak=peak)
                        continue
                    question = questions[index%len(questions)].question
                    if index%6==5:
                        question = 'How does it handle sequences?'  # Exercises actual structured routing call.
                    if name=='retry_probe':
                        question = next(q.question for q in QUESTIONS if q.question_id=='T5')
                    token = SIM_REQUEST.set(f'{prefix}:{index}')
                    try:
                        cfg = replace(config,total_request_deadline_seconds=(10-wait/1000)*scale)
                        result = await RAGPipeline('fast',retriever,provider,cfg,
                            logger=JsonlLogger(directory/f'{prefix}.traces.jsonl')).run(question)
                    finally:
                        SIM_REQUEST.reset(token)
                    status = ('ok' if result.status in {'ok','insufficient_evidence','conversation','out_of_scope'} else
                              'rate_limited' if result.status in {'rate_limited','capacity_unavailable'} else
                              'timeout' if result.status=='capacity_timeout' else 'error')
                    record(index,arrival,status,result.status,result.answer,result,wait,peak)
                finally:
                    queue.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(4)]
        try:
            await asyncio.gather(*(offer(i,a) for i,a in enumerate(arrivals)))
            await queue.join()
        finally:
            for _ in workers:await queue.put(None)
            await asyncio.gather(*workers)
        # Account for model work that finishes after a user deadline/cancel.
        for _ in range(500):
            if not server.simulation_state['active']:break
            await asyncio.sleep(.01)
        if server.simulation_state['active']:
            raise RuntimeError('Stand-in calls still in flight; refusing incomplete usage accounting')
        for row in rows:
            events = [e for e in server.simulation_state['events'] if e['request_id']==row['request_id']]
            attempts = [e for e in provider.attempts if e['request_id']==row['request_id']]
            row['tokens_in'] = sum(e['tokens_in'] for e in events)
            row['tokens_out'] = sum(e['tokens_out'] for e in events)
            row['model_calls'] = len(events)
            row['retries'] = sum(e['retry']>0 for e in attempts)
            row['retry_wait_ms'] = sum(e['virtual_ms'] for e in provider.retry_waits if e['request_id']==row['request_id'])
            row['estimated_cost_usd'] = (row['tokens_in']*.075+row['tokens_out']*.30)/1e6
        write_csv(directory/f'{prefix}.attempts.csv',server.simulation_state['events'])
        (directory/f'{prefix}.details.json').write_text(json.dumps(details,ensure_ascii=False,indent=2))
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)
    rows.sort(key=lambda r:int(r['request_id'].rsplit(':',1)[1]))
    write_csv(directory/f'{prefix}.csv',rows)
    return rows


def summarize(rows):
    out = {'scenario':rows[0]['scenario'],'source':'stand-in','requests':len(rows),
        'outcomes_json':json.dumps(dict(Counter(r['status'] for r in rows))),
        'pipeline_executions':sum(str(r['pipeline_executed']).lower()=='true' for r in rows),
        'model_calls':sum(int(r['model_calls']) for r in rows),
        'tokens_in':sum(int(r['tokens_in']) for r in rows),'tokens_out':sum(int(r['tokens_out']) for r in rows),
        'retry_count':sum(int(r['retries']) for r in rows),
        'peak_queue_length':max(int(r['queue_length']) for r in rows),
        'failed_percent':100*sum(r['status']!='ok' for r in rows)/len(rows),
        'estimated_paid_equivalent_usd':sum(float(r['estimated_cost_usd']) for r in rows)}
    for key in ('first_token_ms','total_ms','queue_wait_ms','retry_wait_ms'):
        vals = [float(r[key]) for r in rows if r[key] not in (None,'')]
        for q in (50,95,99):out[f'{key}_p{q}'] = float(np.percentile(vals,q)) if vals else None
    return out


async def streaming_probe(output, scale=.02):
    calibration = json.loads((output/'calibration.json').read_text())
    server = create_server(calibration,SimulationPolicy(time_scale=scale,slow_every=0))
    thread = threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    first=None;pieces=[];started=time.perf_counter()
    try:
        async with httpx.AsyncClient(trust_env=False) as client:
            async with client.stream('POST',f'http://127.0.0.1:{server.server_address[1]}/openai/v1/chat/completions',
                json={'messages':[{'role':'user','content':'Protocol stream probe'}],'stream':True}) as response:
                async for line in response.aiter_lines():
                    if not line.startswith('data: ') or line=='data: [DONE]':continue
                    event=json.loads(line[6:])
                    for choice in event.get('choices',[]):
                        content=choice.get('delta',{}).get('content')
                        if content:
                            if first is None:first=(time.perf_counter()-started)*1000/scale
                            pieces.append(content)
        result={'source':'stand-in','chunk_count':len(pieces),'first_token_ms':first,
            'total_ms':(time.perf_counter()-started)*1000/scale,'time_scale':scale,
            'quality_measured':False,'purpose':'Separate protocol capability check; pipeline answers remain buffered.'}
        (output/'stream_probe.json').write_text(json.dumps(result,indent=2))
        return result
    finally:
        server.shutdown();server.server_close();thread.join(timeout=2)


async def suite(output, *, count=24, scale=.02):
    output.mkdir(parents=True,exist_ok=True)
    calibration_path = output/'calibration.json'
    if not calibration_path.exists():
        calibration = calibrate(Path('groq_probe.csv'),[Path('results/architecture_formal18_recovered.csv')])
        calibration['limitations'].extend([
            'Only original-account logs used; account2 and mixed repair outputs excluded.',
            'Server uses current prompt-sized character-token estimates, not sampled probe token sizes.',
            'Slow-every-11 and overload-failure-every-13 are explicit synthetic assumptions, not measured failure probabilities.',
            'Accelerated wall time includes CPU/transport overhead divided by time_scale. Not production latency.'])
        calibration_path.write_text(json.dumps(calibration,indent=2))
    calibration = json.loads(calibration_path.read_text())
    config = RelayConfig(evidence_character_budget=16000)
    config.intent_router.mode = 'hybrid'
    config.provider.scheduler.enabled = False
    manifest = {'architecture':'fast','retrieval':'500_5','provisional':True,'count':count,
        'time_scale':scale,'calibration_sha256':hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
        'index_sha256':hashlib.sha256(config.index_path.read_bytes()).hexdigest(),
        'evidence_character_budget':16000,'scenarios':SCENARIOS,'retries':[0,1],
        'source':'stand-in','quality_evaluated':False,'wall_time_scaled':True}
    mp = output/'manifest.json'
    if mp.exists() and json.loads(mp.read_text())!=json.loads(json.dumps(manifest)):
        raise ValueError('Load configuration changed; choose a new output directory')
    mp.write_text(json.dumps(manifest,indent=2))
    retriever = HybridRetriever(load_chunks(config.index_path),config.retrieval)
    retriever.retrieve('scaled dot product attention')  # Explicit warmup, outside measurements.
    summaries, all_rows = [],[]
    for name in SCENARIOS:
        for retries in (0,1):
            p = output/f'{name}_retries{retries}.csv'
            if p.exists():
                with p.open(newline='') as f:rows=list(csv.DictReader(f))
            else:
                print(f'Executing actual pipeline: {name}, retries={retries}',flush=True)
                rows = await scenario_run(name,calibration,retriever,config,output,count=count,scale=scale,retries=retries)
            summaries.append(summarize(rows));all_rows.extend(rows)
    probe = []
    for retries in (0,1):
        p = output/f'retry_probe_retries{retries}.csv'
        if p.exists():
            with p.open(newline='') as f:rs=list(csv.DictReader(f))
        else:
            rs = await scenario_run('retry_probe',calibration,retriever,config,output,
                count=1,scale=scale,retries=retries,fail_first=1,tpm=6500)
        probe.extend(rs)
    write_csv(output/'retry_case.csv',probe)
    write_csv(output/'summary.csv',summaries)
    write_csv(output/'slow_requests.csv',sorted(all_rows,key=lambda r:float(r['total_ms']),reverse=True)[:10])
    (output/'quota_3pm.json').write_text(json.dumps({'source':'stand-in','local_time':'15:00 Asia/Kolkata',
        'http_status':503,'model_calls':0,'retry_after_seconds':None,'response':DAILY_RESPONSE},indent=2))
    if not (output/'stream_probe.json').exists():await streaming_probe(output,scale)
    return summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('results/assignment_finish/pipeline_load'))
    p.add_argument('--count',type=int,default=24)
    p.add_argument('--time-scale',type=float,default=.02)
    p.add_argument('--stream-probe',action='store_true',help='Only check HTTP streaming capability locally')
    args=p.parse_args()
    if args.stream_probe:print(json.dumps(asyncio.run(streaming_probe(args.output,args.time_scale)),indent=2))
    else:print(json.dumps(asyncio.run(suite(args.output,count=args.count,scale=args.time_scale)),indent=2))


if __name__=='__main__':main()

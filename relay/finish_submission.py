"""Rebuild an assignment-format snapshot from real records and executed pipeline loads.

No inference API is called. Unmeasured quality stays blank; historical and current
experiments are separated. Extra-account records are disclosed and excluded.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

from .assignment_export import REQUIRED_COLUMNS, number, tables, percentile
from .checkpoint import read_rows
from .load_lab import write_csv, DAILY_RESPONSE
from .retrieval_campaign import SETTINGS, prepare, select


def historical_real():
    """Only original-account, original-version Fast rows; grades joined by request ID."""
    rows = read_rows(Path('results/architecture_formal18_recovered.csv'))
    grades = {r['request_id']:r for r in read_rows(Path('results/formal18_repair_v2_audit/grades.csv'))}
    false_refusals = {'784eaa81-e61c-407a-a27e-1603be533374','cfec57be-f0a4-4ad0-b21e-8ad2b8a1dc77'}
    output = []
    for row in rows:
        if row['architecture']!='fast' or row.get('quota_profile') not in ('',None,'default'):
            continue
        grade = grades.get(row['request_id'],{})
        answer_grade = ('refused_correctly' if grade.get('correct_refusal')=='1' else
            'correct' if grade.get('accuracy')=='1' and grade.get('citation_grounding')=='1' else
            'wrong' if grade else 'refused_wrongly' if row['request_id'] in false_refusals else '')
        output.append({'request_id':row['request_id'],'scenario':'historical_fast_quality_18',
            'source':'real','start_time':row['run_started_at'],'first_token_ms':row['user_visible_ttft_ms'],
            'total_ms':row['total_ms'],'status':'ok','retries':0,
            'tokens_in':row['total_input_tokens'],'tokens_out':row['total_output_tokens'],
            'retrieval_setting':'historical_fast:500/5:original_code:no_common_budget',
            'answer_grade':answer_grade,'queue_wait_ms':row['queue_wait_ms'],
            'processing_ms':float(row['total_ms'])-float(row['queue_wait_ms']),
            'queue_length':None,'estimated_cost_usd':None,'quality_scope':'historical only; incompatible with new candidates',
            'question_id':row['question_id'],'original_status':row['status'],
            'input_file':'results/architecture_formal18_recovered.csv','time_basis':'measured_wall_clock'})
    return output


def candidate_rows(campaign):
    output = []
    grades = {r['request_id']:r for r in read_rows(campaign/'review/grades.csv')}
    for size,k in SETTINGS:
        path = campaign/f'real_{size}_{k}.csv'
        for r in read_rows(path):
            if r.get('quota_profile') not in ('',None,'default'):
                raise ValueError('Extra-account rows cannot enter the compliant retrieval campaign')
            g = grades.get(r['request_id'],{})
            infrastructure = r['status'] not in {'ok','insufficient_evidence','citation_validation_failed',
                'verification_failed','revision_failed','revision_verification_failed','conversation','out_of_scope'}
            grade = '' if infrastructure else ('refused_correctly' if g.get('correct_refusal')=='1' else
                'correct' if g.get('accuracy')=='1' and g.get('citation_grounding')=='1' else
                'refused_wrongly' if g.get('accuracy')=='0' and 'do not establish' in r['answer'] else
                'wrong' if g.get('accuracy') in {'0','1'} and g.get('citation_grounding') in {'0','1'} else '')
            status = ('rate_limited' if r['status'] in {'rate_limited','capacity_unavailable'} else
                'timeout' if r['status']=='capacity_timeout' else 'error' if infrastructure else 'ok')
            total, wait = number(r['total_ms']),number(r['queue_wait_ms'])
            output.append({'request_id':r['request_id'],'scenario':f'retrieval_{size}_{k}',
                'source':'real','start_time':r['run_started_at'],'first_token_ms':r['user_visible_ttft_ms'],
                'total_ms':r['total_ms'],'status':status,'retries':0,'tokens_in':r['total_input_tokens'],
                'tokens_out':r['total_output_tokens'],'retrieval_setting':f'fast:{size}/{k}:no_rerank:16000char_budget',
                'answer_grade':grade,'queue_wait_ms':wait,'processing_ms':total-wait if total is not None and wait is not None else None,
                'queue_length':None,'estimated_cost_usd':None,'question_id':r['question_id'],
                'original_status':r['status'],'input_file':str(path),'time_basis':'measured_wall_clock',
                'quality_scope':'current retrieval candidate'})
    return output


def retrieval_table(records):
    table = []
    for size,k in SETTINGS:
        rows = [r for r in records if r['scenario']==f'retrieval_{size}_{k}']
        graded = [r for r in rows if r['answer_grade']]
        complete = len(graded)==len(rows)==18
        success = [r for r in rows if r['status']=='ok']
        tokens = [number(r['tokens_in'])+number(r['tokens_out']) for r in success
                  if number(r['tokens_in']) is not None and number(r['tokens_out']) is not None]
        table.append({'Setting':f'{size}/{k}; no rerank; shared 16000-char budget',
            'Questions asked':len(rows),'Correct and cited %':f"{100*sum(r['answer_grade']=='correct' for r in graded)/len(graded):.1f}" if complete else 'not measured',
            'Wrong %':f"{100*sum(r['answer_grade'] in {'wrong','refused_wrongly'} for r in graded)/len(graded):.1f}" if complete else 'not measured',
            'Correctly refused %':f"{100*sum(r['answer_grade']=='refused_correctly' for r in graded)/len(graded):.1f}" if complete else 'not measured',
            'Tokens per answer':round(sum(tokens)/len(tokens),1) if tokens and len(tokens)==len(success) else 'not measured',
            'First token p50 / p95 (ms)':f"{percentile(rows,'first_token_ms',50)} / {percentile(rows,'first_token_ms',95)}",
            'Total time p50 / p95 (ms)':f"{percentile(rows,'total_ms',50)} / {percentile(rows,'total_ms',95)}"})
    return table


def compact_table(headers,rows):
    escape = lambda s:html.escape(str(s))
    return '<table><tr>'+''.join('<th>'+escape(h)+'</th>' for h in headers)+'</tr>'+''.join(
        '<tr>'+''.join('<td>'+escape(c)+'</td>' for c in row)+'</tr>' for row in rows)+'</table>'


def build(output,load,campaign,personal=None):
    output.mkdir(parents=True,exist_ok=True)
    prepare(campaign)
    selection = select(campaign)
    records = historical_real()+candidate_rows(campaign)
    for name in ('steady','burst','sustained_overload','halved_rpm','daily_exhausted'):
        for retries in (0,1):
            records.extend(read_rows(load/f'{name}_retries{retries}.csv'))
    records.extend(read_rows(load/'retry_case.csv'))
    probe = read_rows(Path('groq_probe.csv'))
    for r in probe:
        records.append({**{k:r.get(k,'') for k in REQUIRED_COLUMNS},'scenario':'historical_service_calibration',
            'source':'real','answer_grade':'','queue_wait_ms':r.get('queue_ms'),
            'processing_ms':None,'queue_length':None,'estimated_cost_usd':None,
            'input_file':'groq_probe.csv','time_basis':'measured_wall_clock',
            'quality_scope':'standalone timing probe; not paper QA'})
    if not all(any(r['scenario']==f'{s}_retries1' for r in records) for s in
               ('steady','burst','sustained_overload','halved_rpm','daily_exhausted')):
        raise ValueError('All five pipeline load scenarios must complete before report generation')
    if len({r['request_id'] for r in records})!=len(records):
        raise ValueError('Duplicate raw request IDs; no aggregation allowed')
    write_csv(output/'results.csv',[{k:r.get(k,'') for k in REQUIRED_COLUMNS} for r in records])
    # The mandated minimal schema omits queue/cost/limit fields; keep a joinable raw sidecar.
    (output/'request_details.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    write_csv(output/'results_detailed.csv',[{k:r.get(k,'') for k in sorted(set().union(*(r.keys() for r in records)))} for r in records])
    normalized = [{**{k:None for k in ('queue_length','estimated_cost_usd')},**r} for r in records]
    lat,outcomes = tables(normalized)
    write_csv(output/'table2a.csv',lat);write_csv(output/'table2b.csv',outcomes)
    rt = retrieval_table(records);write_csv(output/'table3.csv',rt)
    limits = [
        {'Limit':'RPM','Value':'30','Evidence':'Original-account configured limit/probe; not separately isolated','Confidence':'medium'},
        {'Limit':'TPM','Value':'8000','Evidence':'Original-account response header','Confidence':'high'},
        {'Limit':'RPD','Value':'1000','Evidence':'Original-account response header','Confidence':'high'},
        {'Limit':'TPD','Value':'200000','Evidence':'Original-account preserved TPD 429 error body','Confidence':'high'},
        {'Limit':'In flight','Value':'not measured','Evidence':'Simulator capacity 4 is an assumption','Confidence':'low'},
        {'Limit':'Limit response','Value':'HTTP 429; Retry-After','Evidence':'Original-account raw 429 bodies/headers','Confidence':'high'},
        {'Limit':'Recovery','Value':'not measured','Evidence':'Retry-After lower bound observed; full recovery unverified','Confidence':'low'}]
    write_csv(output/'table1.csv',limits)
    excluded = [r for r in read_rows(Path('results/architecture_formal18_repair_v2.csv')) if r.get('quota_profile')=='account2']
    write_csv(output/'excluded_extra_account_rows.csv',[{k:r.get(k,'') for k in
        ('request_id','question_id','architecture','quota_profile','run_started_at')} for r in excluded])
    q = json.loads((load/'quota_3pm.json').read_text())
    (output/'quota_3pm.json').write_text(json.dumps(q,indent=2))
    slow = max((r for r in records if r['source']=='stand-in' and r.get('pipeline_executed') in (True,'True')),
               key=lambda r:float(r['total_ms']))
    split = {'request_id':slow['request_id'],'scenario':slow['scenario'],'total_ms':float(slow['total_ms']),
        'queue_ms':float(slow['queue_wait_ms']),'retry_backoff_ms':float(slow['retry_wait_ms']),
        'working_and_transport_ms':float(slow['total_ms'])-float(slow['queue_wait_ms'])-float(slow['retry_wait_ms']),
        'first_token_ms':number(slow['first_token_ms']),
        'before_or_after_first_token':'All work precedes first visible token because answers are buffered until validation.'}
    (output/'slow_request_breakdown.json').write_text(json.dumps(split,indent=2))
    person = json.loads(personal.read_text()) if personal and personal.exists() else {}
    blockers = ['54 real retrieval evaluations and independent grades remain outstanding; no live inference authorized.'] if not selection.get('complete') else []
    if not person: blockers.append('User-specific website feedback and reflection not supplied; visible decisions are summarized without impersonation.')
    status = {'complete':not blockers,'architecture':'fast','architecture_basis':'User-selected provisional baseline',
        'retrieval_winner':selection.get('winner'),'blockers':blockers,
        'extra_account_rows_excluded':len(excluded),'simulator_answer_quality':'not measured',
        'raw_columns_exact':list(REQUIRED_COLUMNS),'missing_answer_grade':
        'Blank for timing probes and simulator fixtures; no invented enum value or quality grade.',
        'table_reproduction':'results.csv joined to request_details.jsonl for fields absent from prescribed minimal schema.',
        'simulated_time':'Accelerated wall clock; CPU and HTTP overhead also scaled. Not production latency.',
        'costs':'Estimated paid-tier equivalent using dated 2026-11-13 price snapshot; not billing or local CPU cost.'}
    (output/'status.json').write_text(json.dumps(status,indent=2))
    report(output,lat,outcomes,rt,limits,records,split,status,person)
    return status


def report(output,lat,outcomes,rt,limits,records,split,status,person):
    import pymupdf
    escape = html.escape
    nm = lambda v: 'NM' if str(v)=='not measured' else str(v)
    chosen = [r for r in lat if r['Scenario'].endswith('_retries1')]
    outcome = [r for r in outcomes if r['Scenario'].endswith('_retries1')]
    t1 = compact_table(['Limit','Found','Evidence','Confidence'],[[r[k] for k in ('Limit','Value','Evidence','Confidence')] for r in limits])
    t2a = compact_table(['Scenario','Source','N','Duration s','Offered/min','TTFT p50/p95/p99 ms','Total p50/p95/p99 ms'],[
        [r['Scenario'].replace('_retries1',''),'stand-in',r['Requests sent'],r['Duration (s)'],r['Offered rate (per min)'],
         '/'.join(nm(r[f'First token p{p} (ms)']) for p in (50,95,99)),
         '/'.join(nm(r[f'Total time p{p} (ms)']) for p in (50,95,99))] for r in chosen])
    t2b = compact_table(['Scenario','OK %','429 %','Timeout %','Retry/req','Q peak','Tokens/answer','USD/1000 est.'],[
        [r['Scenario'].replace('_retries1',''),*[nm(r[k]) for k in ('Succeeded %','Rate-limited %','Timed out or given up %',
            'Retries per request','Peak queue length','Tokens per answer','Cost per 1000 answers (USD estimated)')]] for r in outcome])
    t3 = compact_table(['Pieces/K','N','Correct+cited %','Wrong %','Refused %','Tokens','TTFT 50/95','Total 50/95'],[
        [r['Setting'].split(';')[0],r['Questions asked'],*[nm(r[k]) for k in ('Correct and cited %','Wrong %','Correctly refused %',
          'Tokens per answer','First token p50 / p95 (ms)','Total time p50 / p95 (ms)')]] for r in rt])
    hist = [r for r in records if r['scenario']=='historical_fast_quality_18']
    hcorrect = sum(r['answer_grade']=='correct' for r in hist)
    hrefuse = sum(r['answer_grade']=='refused_correctly' for r in hist)
    ht = compact_table(['Historical Fast quality (original account/code)','Questions','Correct+cited','Correct refusals','Mean tokens'],[
        ['Diagnostic; not a new retrieval candidate',len(hist),f'{hcorrect}/13',f'{hrefuse}/5',
         round(sum(float(r['tokens_in'])+float(r['tokens_out']) for r in hist)/len(hist),1)]])
    pending = 'Retrieval winner is pending real measurements; no final setting selected.' if not status['retrieval_winner'] else f"Selected retrieval: {status['retrieval_winner']}."
    first = f'''<h1>The Relay — reproducible submission snapshot</h1>
    <p><b>{'COMPLETE' if status['complete'] else 'FINAL FORMAT; REAL RETRIEVAL EVIDENCE/PERSONAL INPUT PENDING'}</b>. NM = not measured. Historical, simulated and estimated measurements are separated.</p>
    <h2>1. What I built and why.</h2><p>Fast RAG is the user-selected provisional baseline: hybrid intent routing, shared BM25+dense retrieval, evidence gate, one grounded generation and strict citation validation. A common 16,000-character evidence budget uses contiguous query-focused excerpts; 800/7 otherwise exceeds the free-tier request budget. The main tradeoff is buffering until validation: visible TTFT equals safe answer-release time. Fast validates source identity, not semantic entailment.</p>
    <h2>2. The real service and my stand-in.</h2><p>Original-account Groq probe and stage logs calibrate empirical service timing; queue waits are excluded. The loopback HTTP stand-in executes through the actual pipeline. It streams protocol fixtures, enforces rolling RPM/TPM and daily admission, and slows with concurrent calls. Slow-every-11 and overload failure-every-13 are declared synthetic assumptions. Fixture answers and character-estimated tokens are not real answer-quality measurements. Accelerated clock factor 0.02 also scales CPU/HTTP overhead; these are capacity experiments, not production latency forecasts.</p>
    {t1}<h2>3. Results.</h2><p>Table 2a/2b: stand-in, actual pipeline, one bounded retry; TTFT quantiles use released answers, total quantiles include failures. Tail estimates have small sample sizes. Estimated USD uses the dated price snapshot, not actual billing. All retry-off runs and real probe rows are in the raw exports.</p>
    {t2a}{t2b}{ht}
    <h2>4. Where the time goes.</h2><p>Slow request {escape(split['request_id'])}: total {split['total_ms']:.1f} ms = queue {split['queue_ms']:.1f} + backoff {split['retry_backoff_ms']:.1f} + working/HTTP {split['working_and_transport_ms']:.1f}. Join request_details.jsonl and the per-call stand-in log by request ID. All delay precedes first visible output, or timeout prevents any answer.</p>
    <h2>5. When a limit is hit.</h2><p>At minute limits, only short Retry-After values are retried; longer waits fail visibly. The waiting queue holds eight requests; overflow is turned away and a ten-second virtual deadline bounds waiting plus work. Daily exhaustion returns 503 without inference or retry. No answer is weakened to satisfy capacity. The original scheduler's bucket did not prove strict rolling TPM; new real campaign commands deliberately pace evaluations sixty seconds apart and preserve the original account hold.</p>'''
    retry = [r for r in records if r['scenario'].startswith('retry_probe')]
    tr = compact_table(['Controlled retry stress','Total ms','HTTP attempts','Final status','Tokens'],[
        [r['scenario'],round(float(r['total_ms']),1),r['model_calls'],r['status'],int(r['tokens_in'])+int(r['tokens_out'])] for r in retry])
    second = f'''<h2>Retrieval: quality versus cost.</h2><p>Same eighteen questions, Fast, model/routing/overlap and evidence budget for 300/3, 500/5, 800/7; no reranking. Real requests require explicit opt-in. Historical 500/5 has different generation behavior and is excluded from this comparison. {escape(pending)} Near quota, preserve this configuration and refuse/defer rather than silently change evidence. Piece sizes are whitespace-token approximations.</p>{t3}
    <h2>How this library will mislead a naive retriever.</h2><p>(1) “Why scale attention by sqrt(d_k)?”: broken PDF math/table extraction can hide the equation. (2) “How does cosine routing pick boundaries?”: high-level H-Net ablations rank above the actual inverse-cosine equation on PDF p.6. (3) “Compare Transformer and H-Net tokenization”: retrieval or final citations may cover only one paper; required-paper retrieval and citation coverage must be checked separately.</p>
    <h2>6. Retries.</h2><p>At most one transient retry with bounded backoff; never daily exhaustion, unknown consumption without a deadline, or a long quota cooldown. The paired synthetic stress below injects an initial failure; retrying can spend another HTTP attempt and wait only to hit the same token budget. Its values are measured locally, not observed Groq failure rates.</p>{tr}
    <h2>7. What the person sees.</h2><p>At 3 PM, daily admission is closed. HTTP 503, zero model calls, no invented reset time: “{escape(DAILY_RESPONSE)}”</p>
    <h2>8. What I'm unsure about, what I left out, what I learned.</h2><p>Least certain: prompt-length extrapolation and accelerated CPU effects; if wrong, simulator tail predictions fail. Deliberately omitted: distributed quota coordination and a learned reranker. A new lesson is that a full token bucket plus refill permits a burst above a strict rolling-minute total; nonnegative counters and HTTP 200 alone do not certify compliance. The ten account2 rows violate the assignment's no-extra-account rule and are excluded and disclosed. Exact H4 failed-call usage remains unknown rather than estimated as measured.</p>
    <h2>Your feedback on Vaani.</h2><p>{escape(person.get('website_feedback','User feedback not supplied. Assistant observations from the documented site review: concrete live-call examples and prompt-diff review explain the product well; the word harness is unclear; show a short playable example beside the hero. These are assistant observations, not the user’s personal testimony.'))}</p>
    <h2>How I worked with the assistant.</h2><p>The unedited visible transcript (when available) accompanies the submission; private reasoning and credentials are excluded. Three influential user instructions were: “Do not select the final architecture ... before evaluating” (prevent unsupported selection); “Never fabricate answer grades or missing experiments” (keep missing measurements visible); and “Use Fast RAG as the provisional baseline” with no automatic Groq calls (reduce scope and respect quota). The user supplied the test questions, demanded independent PDF grading, and overruled further architecture selection in favor of Fast provisionally. These observed choices are distinctly the user's. The assistant made a serious mistake by helping switch accounts before noticing the brief's prohibition; those results remain preserved but excluded. {escape(person.get('reflection','User confirmation of personal lessons, a mistake they themselves caught, and their own website opinions is still pending; this account does not invent those experiences.'))}</p>
    <p>Reproduce: python -m relay.pipeline_load; python -m relay.retrieval_campaign remaining; python -m relay.finish_submission. Exact 13-column results.csv uses blank grades for non-QA fixtures/probes. Join request_details.jsonl for queue/cost fields absent from the prescribed schema. All full tables, source traces, exclusions and incomplete-status reasons accompany this two-page snapshot.</p>'''
    css = 'body{font-family:Arial,sans-serif;font-size:7.6pt;line-height:1.12}h1{font-size:12pt;margin:0 0 5px}h2{font-size:8.6pt;margin:6px 0 2px}p{margin:2px 0 4px}table{border-collapse:collapse;width:100%;font-size:6.4pt;margin:4px 0}td,th{border:0.4px solid #aaa;padding:2px;text-align:left}th{background:#eef2f6}'
    doc = pymupdf.open()
    scales=[]
    for text in (first,second):
        page=doc.new_page(width=595,height=842)
        spare,scale=page.insert_htmlbox(pymupdf.Rect(30,28,565,814),text,css=css,scale_low=.80)
        if spare<0:raise RuntimeError('Two-page report overflow; reduce prose rather than omit required sections')
        scales.append(scale)
    doc.save(output/'report_two_pages.pdf');doc.close()
    (output/'report.html').write_text('<style>'+css+'</style>'+first+'<hr>'+second)
    (output/'report_layout.json').write_text(json.dumps({'pages':2,'scales':scales},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('results/assignment_finish/submission'))
    p.add_argument('--load-directory',type=Path,default=Path('results/assignment_finish/pipeline_load'))
    p.add_argument('--campaign',type=Path,default=Path('results/assignment_finish/retrieval'))
    p.add_argument('--personal-input',type=Path)
    args=p.parse_args()
    print(json.dumps(build(args.output,args.load_directory,args.campaign,args.personal_input),indent=2))


if __name__=='__main__':main()

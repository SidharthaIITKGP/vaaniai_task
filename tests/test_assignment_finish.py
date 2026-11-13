import asyncio
import csv
import json

import httpx
import pytest

from relay.architectures.base import REFUSAL
from relay.config import RelayConfig
from relay.finish_submission import retrieval_table
from relay.pipeline_load import scenario_run, streaming_probe, summarize
from relay.providers.stand_in import StandInProvider
from relay.retrieval_campaign import prepare, select
from relay.verification import evidence_window, format_evidence


def calibration():
    return {'records':[{'stage':'generation','input_tokens':84,'output_tokens':60,
                        'ttft_ms':50,'latency_ms':100}], 'bundles':{}}


def test_budgeted_evidence_is_verbatim_bounded_and_keeps_aliases(retriever):
    text='padding '*100+'important cosine boundary target'
    excerpt=evidence_window(text,'cosine boundary',60)
    assert excerpt in text and 'cosine boundary' in excerpt and len(excerpt)<=60
    passages=retriever.retrieve('attention')
    rendered=format_evidence(passages,character_budget=12,question='attention')
    assert all(f'[S{i}]' in rendered for i in range(1,len(passages)+1))
    assert all(p.chunk.paper_id in rendered for p in passages)


def test_stand_in_rejects_remote_endpoints():
    with pytest.raises(ValueError,match='loopback'):
        StandInProvider('https://api.groq.com/openai/v1/chat/completions',None)


@pytest.mark.asyncio
async def test_daily_closed_admission_has_no_pipeline_or_model_calls(tmp_path,retriever,config,monkeypatch):
    async def forbidden(*args,**kwargs):pytest.fail('Model called under exhausted daily quota')
    monkeypatch.setattr(StandInProvider,'_post',forbidden)
    rows=await scenario_run('daily_exhausted',calibration(),retriever,config,tmp_path,count=3,scale=.1)
    assert all(r['model_calls']==0 and r['retries']==0 and not r['pipeline_executed'] for r in rows)
    assert all(r['status']=='rate_limited' and r['answer_grade']=='' for r in rows)


@pytest.mark.asyncio
async def test_loopback_executes_actual_retrieval_and_citation_validation(tmp_path,retriever,config,monkeypatch):
    from relay.providers.groq_provider import GroqProvider
    monkeypatch.setattr(GroqProvider,'__init__',lambda *a,**kw:pytest.fail('Groq initialization forbidden'))
    rows=await scenario_run('burst',calibration(),retriever,config,tmp_path,count=1,scale=.1)
    assert rows[0]['pipeline_executed'] is True
    assert rows[0]['status']=='ok' and rows[0]['answer_grade']==''
    details=json.loads((tmp_path/'burst_retries1.details.json').read_text())[0]
    assert details['metrics_wall_ms']['retrieved_chunk_ids']
    assert details['metrics_wall_ms']['citation_validation']['valid']
    assert rows[0]['model_calls']==1


@pytest.mark.asyncio
async def test_stand_in_streams_multiple_protocol_chunks(tmp_path):
    (tmp_path/'calibration.json').write_text(json.dumps(calibration()))
    result=await streaming_probe(tmp_path,scale=.2)
    assert result['chunk_count']>1
    assert 0<result['first_token_ms']<=result['total_ms']
    assert result['quality_measured'] is False


def test_retrieval_campaign_has_same_questions_and_no_premature_winner(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    source=tmp_path/'results/final_retrieval_lab18';source.mkdir(parents=True)
    for n in (300,500,800):(source/f'chunks_{n}.jsonl').write_text('{}\n')
    campaign=tmp_path/'campaign'
    manifest=prepare(campaign)
    assert len(manifest['questions'])==18
    assert len(list(csv.DictReader((campaign/'plan.csv').open())))==54
    assert select(campaign)['winner'] is None
    assert not (campaign/'deployment.json').exists()


def test_pending_retrieval_quality_is_not_filled_with_zero_or_simulated_accuracy():
    rows=retrieval_table([])
    assert len(rows)==3
    assert all(r['Questions asked']==0 and r['Correct and cited %']=='not measured' for r in rows)


def test_resume_summary_interprets_false_strings_correctly():
    row={'scenario':'test','pipeline_executed':'False','model_calls':'0','tokens_in':'0','tokens_out':'0',
         'retries':'0','queue_length':'0','status':'rate_limited','estimated_cost_usd':'0',
         'first_token_ms':'','total_ms':'1','queue_wait_ms':'0','retry_wait_ms':'0'}
    assert summarize([row])['pipeline_executions']==0

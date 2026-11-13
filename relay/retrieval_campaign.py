"""Resumable three-setting Fast RAG experiment. Real inference requires explicit flags."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import time
from pathlib import Path

from .checkpoint import read_rows
from .evaluate import FORMAL_18, account_lock, parser as evaluation_parser, run as evaluate
from .evaluation_questions import QUESTIONS
from .experiment_review import compare, packet
from .load_lab import write_csv


def diagnostics(output):
    """Execute all three real retrievers locally; never infer answer grades."""
    from .config import RelayConfig
    from .ingestion import load_chunks
    from .retrieval import HybridRetriever, SentenceTransformerEmbedder
    from .architectures.fast import FastRAG
    from .instrumentation import RequestTrace
    from .verification import format_evidence
    manifest=prepare(output)
    cfg=RelayConfig(evidence_character_budget=16000)
    embedder=SentenceTransformerEmbedder(cfg.retrieval.dense_model)
    rows=[]
    for variant in manifest['variants']:
        cfg.retrieval.top_k=variant['top_k']
        retriever=HybridRetriever(load_chunks(Path(variant['index'])),cfg.retrieval,embedder)
        architecture=FastRAG(retriever,None,cfg)
        for q in QUESTIONS:
            if q.question_id not in FORMAL_18:continue
            trace=RequestTrace('local-diagnostic','fast')
            started=time.perf_counter()
            evidence,sufficient=architecture.retrieve(q.question,trace)
            elapsed=(time.perf_counter()-started)*1000
            rendered=format_evidence(evidence,character_budget=16000,question=q.question)
            raw=sum(len(e.chunk.text) for e in evidence)
            rows.append({'source':'local_retrieval_only','configuration':variant['configuration'],
                'question_id':q.question_id,'question':q.question,'retrieval_ms':elapsed,
                'retrieved_chunks':len(evidence),'source_characters':raw,'rendered_characters':len(rendered),
                'evidence_shortened':raw>16000,'retrieved_papers':';'.join(sorted({e.chunk.paper_id for e in evidence})),
                'estimated_evidence_tokens_chars_div4':(len(rendered)+3)//4,
                'answer_correctness':'not measured','citation_support':'not measured',
                'answer_ttft_ms':'not measured','answer_total_ms':'not measured'})
    write_csv(output/'local_diagnostics.csv',rows)
    return rows


SETTINGS = ((300,3),(500,5),(800,7))


def prepare(output: Path):
    output.mkdir(parents=True,exist_ok=True)
    variants = []
    plan = []
    for size,k in SETTINGS:
        source = Path(f'results/final_retrieval_lab18/chunks_{size}.jsonl')
        index = output/f'chunks_{size}.jsonl'
        sha = hashlib.sha256(source.read_bytes()).hexdigest()
        if index.exists() and hashlib.sha256(index.read_bytes()).hexdigest()!=sha:
            raise ValueError('Variant index changed; use a new campaign directory')
        if not index.exists():shutil.copyfile(source,index)
        variants.append({'configuration':f'{size}_{k}','index':str(index),'top_k':k,'index_sha256':sha,
                         'chunk_size_whitespace_tokens':size,'overlap':80,'evidence_character_budget':16000})
        for q in QUESTIONS:
            if q.question_id in FORMAL_18:
                plan.append({'configuration':f'{size}_{k}','question_id':q.question_id,'question':q.question,
                    'architecture':'fast','planned_source':'real','quality_measured':False})
    manifest = {'baseline':'fast','baseline_selection':'explicit user choice; provisional',
        'retrieval_winner':None,'questions':sorted(FORMAL_18),'variants':variants,
        'evidence_character_budget':16000,'routing':'hybrid','model':'openai/gpt-oss-20b',
        'quota_profile':'default','extra_accounts_permitted':False,
        'grading':'independent original-PDF review required; status is not a grade'}
    p = output/'campaign.json'
    if p.exists() and json.loads(p.read_text())!=manifest:
        raise ValueError('Campaign settings differ; preserve this campaign and choose a new directory')
    p.write_text(json.dumps(manifest,indent=2))
    (output/'variants.json').write_text(json.dumps(variants,indent=2))
    write_csv(output/'plan.csv',plan)
    return manifest


def evaluation_args(output,variant,*,remaining=False):
    args = ['--subset','formal18','--architectures','fast','--workers','1','--intent-router','hybrid',
        '--index',variant['index'],'--top-k',str(variant['top_k']),
        '--configuration',variant['configuration'],'--evidence-character-budget','16000',
        '--output',str(output/f"real_{variant['configuration']}.csv"),
        '--quota-profile','default','--quota-state','data/groq_quota_state.json',
        '--max-cooldown','60','--max-model-calls','40','--wall-seconds','3600',
        '--delay','60']
    if remaining:args.append('--remaining')
    return evaluation_parser().parse_args(args)


async def execute(output,remaining=False):
    manifest = prepare(output)
    for variant in manifest['variants']:
        print('Retrieval setting:',variant['configuration'],flush=True)
        code = await evaluate(evaluation_args(output,variant,remaining=remaining))
        if code and not remaining:
            print('Campaign checkpointed and paused. Repeat the same command after quota recovery.',flush=True)
            return code
    return 0


def make_packet(output):
    manifest = prepare(output)
    paths = [output/f"real_{v['configuration']}.csv" for v in manifest['variants']]
    existing = [p for p in paths if p.exists()]
    if existing:
        packet(existing,output/'review')
    else:
        print('No real outputs yet. No grades or completed evaluations were fabricated.')


def select(output):
    manifest = prepare(output)
    paths = [output/f"real_{v['configuration']}.csv" for v in manifest['variants']]
    if not all(p.exists() for p in paths) or not (output/'review/grades.csv').exists():
        decision = {'winner':None,'complete':False,'reason':'All three real 18-question settings and independent grades are required.'}
    else:
        decision = compare(paths,output/'review/grades.csv',output/'selection')
    (output/'retrieval_selection.json').write_text(json.dumps(decision,indent=2))
    if decision.get('complete'):
        variant = next(v for v in manifest['variants'] if v['configuration']==decision['winner'])
        (output/'deployment.json').write_text(json.dumps({'architecture':'fast','index':variant['index'],
            'top_k':variant['top_k'],'evidence_character_budget':16000,
            'source':'User-selected Fast baseline; independently graded retrieval selection',
            'retrieval_selection':str(output/'retrieval_selection.json')},indent=2))
    print(json.dumps(decision,indent=2))
    return decision


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','diagnostics','remaining','run','packet','select'))
    p.add_argument('--output',type=Path,default=Path('results/assignment_finish/retrieval'))
    p.add_argument('--real',action='store_true')
    p.add_argument('--original-account',action='store_true',help='Confirm .env uses your original assignment account; no quota bypass')
    args = p.parse_args()
    if args.action=='prepare':prepare(args.output)
    elif args.action=='diagnostics':print(f'Executed {len(diagnostics(args.output))} local retrieval diagnostics; answer quality remains unmeasured.')
    elif args.action=='packet':make_packet(args.output)
    elif args.action=='select':select(args.output)
    elif args.action=='remaining':raise SystemExit(asyncio.run(execute(args.output,True)))
    else:
        if not (args.real and args.original_account):
            p.error('Real requests require --real --original-account. Restore the original account key first.')
        with account_lock(Path('data/groq_quota_state.lock')):
            raise SystemExit(asyncio.run(execute(args.output)))


if __name__=='__main__':main()

"""Prepare isolated chunk variants and measure retrieval diagnostics, never answer accuracy."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .config import RelayConfig
from .evaluate import FORMAL_18, passage_record
from .evaluation_questions import QUESTIONS
from .ingestion import ingest
from .load_lab import write_csv
from .retrieval import HybridRetriever, SentenceTransformerEmbedder, comparison_paper_ids, merge_passages_with_coverage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/retrieval_lab')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    embedder = SentenceTransformerEmbedder(RelayConfig().retrieval.dense_model)
    summary = []
    for size, top_k in ((300, 3), (500, 5), (800, 7)):
        cfg = RelayConfig(chunk_tokens=size, chunk_overlap=80)
        cfg.index_override = output / f'chunks_{size}.jsonl'
        cfg.retrieval.top_k = top_k
        report = ingest(cfg)
        if len(report.verified) != 3 or not all(report.verified.values()):
            raise RuntimeError('All three papers must be verified before retrieval experiments')
        retriever = HybridRetriever(report.chunks, cfg.retrieval, embedder)
        records = []
        for question in QUESTIONS:
            if question.question_id not in FORMAL_18:
                continue
            started = time.perf_counter()
            groups = [retriever.retrieve(question.question)]
            required = comparison_paper_ids(question.question)
            for paper in required:
                groups.append(retriever.retrieve(question.question,
                    top_k=max(1, top_k // len(required)), paper_ids={paper}))
            result = merge_passages_with_coverage(groups, top_k, required)
            records.append({'source': 'local_retrieval_diagnostic', 'configuration': f'{size}_{top_k}',
                'question_id': question.question_id, 'chunk_size_whitespace_tokens': size,
                'top_k': top_k, 'retrieval_ms': (time.perf_counter() - started) * 1000,
                'required_papers': ';'.join(required),
                'required_coverage': all(p in {r.chunk.paper_id for r in result} for p in required),
                'passage_characters': sum(len(r.chunk.text) for r in result),
                'passages_json': json.dumps([passage_record(r) for r in result], ensure_ascii=False),
                'answer_accuracy': None, 'note': 'Coverage is diagnostic, not factual correctness.'})
        write_csv(output / f'retrieval_{size}_{top_k}.csv', records)
        summary.append({'configuration': f'{size}_{top_k}', 'index': str(cfg.index_path),
            'chunk_count': len(report.chunks), 'top_k': top_k,
            'questions': len(records), 'quality_evaluated': False})
    (output / 'variants.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

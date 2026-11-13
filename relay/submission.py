"""Build an honest progress report and results table; never invent missing experiments."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from .load_lab import write_csv


def main():
    import pymupdf
    root = Path('results')
    out = root / 'submission'
    out.mkdir(parents=True, exist_ok=True)
    audit = []
    for path in sorted(root.glob('*.csv')):
        with path.open(newline='', encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
        if not rows or 'architecture' not in rows[0]:
            continue
        for architecture in ('fast', 'verified', 'adaptive'):
            selected = [r for r in rows if r['architecture'] == architecture]
            if not selected:
                continue
            audit.append({'source': 'historical_real', 'file': str(path),
                'architecture': architecture, 'questions': len({r['question_id'] for r in selected}),
                'rows': len(selected), 'outcomes': json.dumps(dict(Counter(r['status'] for r in selected))),
                'independent_accuracy': None, 'selection_eligible': False})
    write_csv(out / 'historical_audit.csv', audit)
    with (root / 'final_load_lab_complete/results.csv').open(newline='') as handle:
        simulated = list(csv.DictReader(handle))
    write_csv(out / 'results.csv', [{'experiment': 'architecture_quality', 'source': 'missing',
        'candidate': '', 'status': '15+ matched independently graded questions not completed',
        'p50_ms': None, 'p95_ms': None, 'p99_ms': None, 'accuracy': None, 'estimated_cost_usd': None},
        {'experiment': 'retrieval_quality', 'source': 'missing', 'candidate': '',
         'status': 'Requires architecture winner and independently graded variant outputs',
         'p50_ms': None, 'p95_ms': None, 'p99_ms': None, 'accuracy': None, 'estimated_cost_usd': None},
        *[{'experiment': row['scenario'], 'source': 'simulated', 'candidate': 'standalone_llm',
            'status': row['outcomes_json'], 'p50_ms': row['latency_ms_p50'],
            'p95_ms': row['latency_ms_p95'], 'p99_ms': row['latency_ms_p99'],
            'accuracy': None, 'estimated_cost_usd': row['estimated_cost_usd']}
           for row in simulated if row['max_admission_retries'] == '0']])
    pages = [
        ('The Relay: experiment status and correctness',
         'Status: PARTIAL SUBMISSION. No final architecture or retrieval configuration has been selected. '
         'The assignment PDF is now available. This progress report does not yet include all required '
         'tables and personal reflection; see docs/ASSIGNMENT_COMPLIANCE.md.\n\n'
         'Bounded current-code check: three real T1 runs consumed 11,280 tokens. Independent PDF-page-4 '
         'inspection supports Verified and Adaptive answers; Fast refused because it emitted no valid alias. '
         'This single-question check is insufficient for architecture selection.\n\n'
         'Implementation: three shared-retrieval pipelines remain intact. Fast generates once and validates '
         'citation aliases; Verified checks semantic support and permits one correction; Adaptive adds bounded '
         'planning for complex questions. Answers remain buffered until required checks pass. The local '
         'verifier now validates all structured fields with Pydantic.\n\n'
         'Historical evidence: the original 90-row run covered 30 questions but suffered 58 rate limits. '
         'Later six-question runs exercise only 18 question/architecture pairs each. These changing-code runs '
         'are diagnostic evidence, not a controlled final architecture experiment. A successful request status '
         'does not establish factual accuracy or citation entailment.\n\n'
         'Architecture protocol: run the same 18 questions (including unsupported paper-related requests) '
         'through every candidate with identical model, retrieval, router and scheduler settings. Blind review '
         'packets omit architecture and require accuracy, citation support, correct-refusal grades, checked '
         'PDF pages and a rationale. No winner is emitted until all candidates cover the same 15+ questions '
         'and all reviews are complete. Rank quality first, then reliability, tokens and latency.\n\n'
         'Retrieval protocol: isolated 300/3, 500/5 and 800/7 chunk-size/top-K variants preserve original '
         'files and use the same embedding model. Local retrieval timing, paper coverage and passage sizes '
         'are measured; these diagnostics cannot substitute for answer grading. Chunk sizes currently use '
         'whitespace tokens, not the provider tokenizer. The selected architecture must be evaluated on all '
         'three variants before integration.\n\n'
         'Outstanding: complete controlled architecture answers and independent grades; then repeat for '
         'retrieval variants. Until those experiments finish, defaults are developmental settings.'),
        ('The Relay: capacity, reproducibility and remaining work',
         'Calibration: empirical provider timing samples come from groq_probe.csv and recorded RAG call '
         'timings. The probe has 120 requests: 52 successes and 68 capacity 429s. Rate limits are modeled '
         'explicitly; 429 observations are not injected as an independent 56.7% random failure probability. '
         'Raw-source hashes are stored in calibration.json. Unmeasured timings remain missing.\n\n'
         'Local virtual-time experiments: five scenarios run automatically at seed 42: steady traffic, burst, '
         'sustained overload, half RPM, and exhausted daily quota. Each scenario compares zero versus two '
         'bounded admission retries. Detailed CSVs record queue wait, service time, tokens, outcomes, queue '
         'length and retry counts. Costs use public list prices: $0.075 input and $0.30 output per million '
         'tokens, checked in Groq model documentation on 2026-11-13; actual free-tier billing is unmeasured. '
         'These are service-capacity simulations, not generated RAG answers or answer-accuracy results.\n\n'
         'Limits: 30 RPM, 8000 TPM and 1000 daily requests are account-observed baseline limits. Safety '
         'margin is 80%. The simulator uses empirical token/latency bootstrap samples; latency scaling to '
         'unobserved prompt lengths and provider day-reset times have not been validated. Pipeline bundles '
         'are reserved atomically in optional architecture simulations; this is conservative and differs '
         'from stage-by-stage production scheduling.\n\n'
         'At 3 PM with exhausted daily quota: HTTP 503, code daily_quota_exhausted, zero model calls, '
         'no daily retry, and no invented reset time. Exact user text is saved in quota_3pm.json.\n\n'
         'Reproduce local measurements: python -m relay.load_lab --output results/final_load_lab_v2. '
         'Prepare retrieval variants: HF_HUB_OFFLINE=1 .venv/bin/python -m relay.retrieval_lab '
         '--output results/final_retrieval_lab. Start standalone simulated service with '
         'python -m relay.simulator_server --calibration results/final_load_lab/calibration.json.\n\n'
         'Remaining integration is intentionally gated on measured quality. No final-system load result '
         'or final architecture choice is claimed. The supplied assignment PDF has been inspected; '
         'PDF-aligned request rows and tables are produced by relay.assignment_export. '
         'See docs/EXPERIMENTS.md for exact next commands.')]
    document = pymupdf.open()
    for title, body in pages:
        page = document.new_page(width=595, height=842)
        page.insert_text((42, 48), title, fontsize=15)
        remaining = page.insert_textbox(pymupdf.Rect(42, 76, 553, 798), body,
                                       fontsize=10, lineheight=1.25)
        if remaining < 0:
            raise RuntimeError('Report overflow; reduce text before publishing')
    document.save(out / 'report_two_pages.pdf')
    document.close()
    (out / 'status.json').write_text(json.dumps({'complete': False, 'architecture_winner': None,
        'retrieval_winner': None, 'assignment_pdf_found': Path('The_Relay_Assignment_Brief.pdf').exists(),
        'report_type': 'two-page progress report; not a completed assignment'}, indent=2))
    print(f'Wrote truthful partial-submission artifacts to {out}')


if __name__ == '__main__':
    main()

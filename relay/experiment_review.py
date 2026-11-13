"""Create blind review packets, aggregate independent grades, and select only complete experiments."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .evaluation_questions import QUESTIONS
from .load_lab import write_csv


def packet(inputs: list[Path], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in inputs:
        with path.open(newline='', encoding='utf-8') as handle:
            for row in csv.DictReader(handle):
                row['_input'] = str(path)
                rows.append(row)
    template = []
    dossier = []
    for row in rows:
        request_id = row['request_id']
        review_id = hashlib.sha256(request_id.encode()).hexdigest()[:16]
        template.append({'review_id': review_id, 'request_id': request_id,
            'question_id': row['question_id'], 'accuracy': '', 'citation_grounding': '',
            'correct_refusal': '', 'reviewer': '', 'paper_pages_checked': '', 'rationale': ''})
        dossier.append({'review_id': review_id, 'question_id': row['question_id'],
            'question': row['question'], 'answer': row['answer'],
            'citations': json.loads(row.get('cited_sources_json') or '[]'),
            'instruction': 'Check the original PDFs. Do not grade from status or self-verification. '
                'Score accuracy and citation_grounding 0 or 1. For unanswerable questions score '
                'correct_refusal 0 or 1; otherwise leave it blank. Record pages and specific errors.'})
    write_csv(output / 'grades.csv', template)
    (output / 'blind_answers.json').write_text(json.dumps(dossier, ensure_ascii=False, indent=2))


def compare(inputs: list[Path], grades: Path, output: Path, minimum_questions: int = 15) -> dict:
    with grades.open(newline='', encoding='utf-8') as handle:
        grade_rows = list(csv.DictReader(handle))
    by_request = {row['request_id']: row for row in grade_rows}
    grouped = defaultdict(list)
    for path in inputs:
        with path.open(newline='', encoding='utf-8') as handle:
            for row in csv.DictReader(handle):
                group = row.get('configuration') or row['architecture']
                grouped[group].append(row)
    summaries = []
    manifests = []
    for path in inputs:
        manifest_path = path.with_suffix('.manifest.json')
        if manifest_path.exists():
            manifests.append(json.loads(manifest_path.read_text()))
    complete = len(manifests) == len(inputs)
    invariant_keys = ('model', 'routing', 'verification_evidence', 'revision',
                      'seed', 'mode', 'workers', 'rpm', 'tpm', 'safety_margin', 'source_sha256')
    if manifests:
        complete &= all(all(m.get(k) == manifests[0].get(k) for k in invariant_keys)
                        for m in manifests)
        if len({m.get('configuration') for m in manifests}) == 1:
            complete &= all(m.get('index_sha256') == manifests[0].get('index_sha256')
                            and m.get('top_k') == manifests[0].get('top_k') for m in manifests)
    question_sets = []
    for group, rows in grouped.items():
        # Latest successful retry is still a distinct recorded attempt; fail duplicates to avoid cherry-picking.
        ids = [r['question_id'] for r in rows]
        question_sets.append(set(ids))
        eligible = len(ids) == len(set(ids)) and len(ids) >= minimum_questions
        reviewed = []
        for row in rows:
            grade = by_request.get(row['request_id'], {})
            if not (grade.get('reviewer') and grade.get('paper_pages_checked') and grade.get('rationale')
                    and grade.get('accuracy') in {'0', '1'} and grade.get('citation_grounding') in {'0', '1'}):
                eligible = False
                continue
            if row['question_id'].startswith('U') and grade.get('correct_refusal') not in {'0', '1'}:
                eligible = False
                continue
            reviewed.append((row, grade))
        n = len(reviewed)
        complete &= eligible
        def mean_metric(key):
            values = [float(r[key]) for r in rows if r.get(key) not in {None, ''}]
            return float(np.mean(values)) if len(values) == len(rows) and values else None
        summaries.append({'source': 'real_with_independent_grades', 'candidate': group,
            'question_count': len(ids), 'reviewed_count': n, 'selection_eligible': eligible,
            'accuracy': sum(int(g['accuracy']) for _, g in reviewed) / n if n else None,
            'citation_grounding': sum(int(g['citation_grounding']) for _, g in reviewed) / n if n else None,
            'correct_refusals': sum(g.get('correct_refusal') == '1' for _, g in reviewed),
            'refusal_questions': sum(r['question_id'].startswith('U') for r in rows),
            'provider_failures': sum(r['status'] in {'provider_error', 'rate_limited', 'runner_error',
                'capacity_unavailable', 'capacity_timeout'} for r in rows),
            'mean_total_ms': mean_metric('total_ms'), 'mean_queue_wait_ms': mean_metric('queue_wait_ms'),
            'mean_model_calls': mean_metric('total_model_calls'),
            'mean_input_tokens': mean_metric('total_input_tokens'),
            'mean_output_tokens': mean_metric('total_output_tokens')})
    complete &= bool(grouped) and all(q == question_sets[0] for q in question_sets)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'comparison.csv', summaries)
    winner = None
    if complete:
        winner = max(summaries, key=lambda r: (r['accuracy'], r['citation_grounding'],
            r['correct_refusals'], -r['provider_failures'],
            -(r['mean_input_tokens'] or float('inf')), -(r['mean_total_ms'] or float('inf'))))['candidate']
    selection = {'winner': winner, 'complete': bool(complete),
        'policy': 'lexicographic accuracy, citation grounding, correct refusal, reliability, input tokens, latency',
        'reason': 'Complete independent reviews and equal question coverage required',
        'inputs': [str(p) for p in inputs], 'grades': str(grades)}
    (output / 'selection.json').write_text(json.dumps(selection, indent=2))
    return selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('packet', 'compare'))
    parser.add_argument('--input', nargs='+', required=True)
    parser.add_argument('--grades')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    paths = [Path(p) for p in args.input]
    if args.action == 'packet':
        packet(paths, Path(args.output))
    else:
        print(json.dumps(compare(paths, Path(args.grades), Path(args.output)), indent=2))


if __name__ == '__main__':
    main()

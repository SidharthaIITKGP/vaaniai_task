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
from .checkpoint import TERMINAL_STATUSES, config_id


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
    infrastructure = []
    existing = {}
    grades_path = output / 'grades.csv'
    if grades_path.exists():
        with grades_path.open(newline='', encoding='utf-8') as handle:
            existing = {r['request_id']: r for r in csv.DictReader(handle)}
    seen = {}
    for row in rows:
        request_id = row['request_id']
        content = {k: v for k, v in row.items() if k != '_input'}
        if request_id in seen:
            if seen[request_id] != content:
                raise ValueError(f'Conflicting results for request {request_id}')
            continue
        seen[request_id] = content
        if row.get('status') not in TERMINAL_STATUSES:
            infrastructure.append({'request_id': row['request_id'], 'question_id': row['question_id'],
                'architecture': row['architecture'], 'status': row['status'],
                'error': row.get('error'), 'grading': 'not applicable: infrastructure failure'})
            continue
        review_id = hashlib.sha256(request_id.encode()).hexdigest()[:16]
        blank = {'review_id': review_id, 'request_id': request_id,
            'question_id': row['question_id'], 'accuracy': '', 'citation_grounding': '',
            'correct_refusal': '', 'reviewer': '', 'paper_pages_checked': '', 'rationale': ''}
        blank.update(existing.get(request_id, {}))
        template.append(blank)
        dossier.append({'review_id': review_id, 'question_id': row['question_id'],
            'question': row['question'], 'answer': row['answer'],
            'required_papers': row.get('source_papers', ''),
            'citations': json.loads(row.get('cited_sources_json') or '[]'),
            'instruction': 'Check the original PDFs. Do not grade from status or self-verification. '
                'Score accuracy and citation_grounding 0 or 1. For unanswerable questions score '
                'correct_refusal 0 or 1; otherwise leave it blank. Record pages and specific errors.'})
    # Never erase grades from an earlier snapshot when a reviewer refreshes the packet.
    template.extend(r for key, r in existing.items() if key not in {t['request_id'] for t in template})
    write_csv(grades_path, template)
    write_csv(output / 'infrastructure_failures.csv', infrastructure)
    (output / 'blind_answers.json').write_text(json.dumps(dossier, ensure_ascii=False, indent=2))
    write_csv(output / 'blind_answers.csv', [
        {**{k: v for k, v in answer.items() if k != 'citations'},
         'citations_json': json.dumps(answer['citations'], ensure_ascii=False)} for answer in dossier])
    root = Path(__file__).resolve().parent.parent
    papers = [{'path': str(p.relative_to(root)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
              for p in sorted((root / 'Research Papers').glob('*.pdf'))]
    metadata = {'inputs': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                           for p in inputs], 'completed_outcomes': len(dossier),
                'infrastructure_failures': len(infrastructure), 'papers': papers,
                'grading': 'Independent review required; no grades inferred from pipeline status.'}
    (output / 'packet_manifest.json').write_text(json.dumps(metadata, indent=2))
    (output / 'README.md').write_text(
        '# Independent answer review\n\n'
        'Review `blind_answers.json` against the original PDFs named in `packet_manifest.json`. '
        'It omits architecture, pipeline status, and verifier verdict. Do not treat cited excerpts '
        'alone as an answer key. Check numerical values, qualifications, all question parts, and '
        'whether each cited page supports its attached claims. Check every required paper in comparisons.\n\n'
        'Enter grades in `grades.csv`: accuracy=1 only for a fully correct, complete answer; '
        'citation_grounding=1 only when all factual claims have supporting citations. '
        'A justified refusal on an unanswerable question can receive 1 for both; an inappropriate '
        'refusal on an answerable question receives 0 for both. For U questions, also enter '
        'correct_refusal=0 or 1; leave that cell blank for other questions. '
        'Record your reviewer name, original PDF page numbers, and a specific rationale. '
        'Partial correctness receives accuracy=0 under this binary rubric; explain the correct parts.\n\n'
        'Infrastructure failures are excluded from answer grades and remain reliability outcomes '
        'in `infrastructure_failures.csv`. Packet refreshes preserve existing request-keyed grades. '
        'No grade has been generated automatically.\n')


def compare(inputs: list[Path], grades: Path, output: Path, minimum_questions: int = 15) -> dict:
    with grades.open(newline='', encoding='utf-8') as handle:
        grade_rows = list(csv.DictReader(handle))
    by_request = {row['request_id']: row for row in grade_rows}
    if len(by_request) != len(grade_rows):
        raise ValueError('Duplicate request IDs in grades; resolve conflicting reviews explicitly')
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
        complete &= not any(m.get('repair_campaign') or m.get('selection_eligible') is False for m in manifests)
        complete &= all(all(m.get(k) == manifests[0].get(k) for k in invariant_keys)
                        for m in manifests)
        if len({m.get('configuration') for m in manifests}) == 1:
            complete &= all(m.get('index_sha256') == manifests[0].get('index_sha256')
                            and m.get('top_k') == manifests[0].get('top_k') for m in manifests)
    # An equal but truncated subset must not masquerade as a completed formal experiment.
    for path in inputs:
        companion = path.with_suffix('.manifest.json')
        if not companion.exists():
            continue
        manifest = json.loads(companion.read_text())
        with path.open(newline='', encoding='utf-8') as handle:
            source_rows = list(csv.DictReader(handle))
        planned = {(q, a) for q in manifest.get('questions', [])
                   for a in manifest.get('architectures', [])}
        if planned:
            complete &= {(r['question_id'], r['architecture']) for r in source_rows} == planned
        if 'evaluation_configuration' in manifest:
            complete &= all(r.get('evaluation_config_id') == config_id(manifest) for r in source_rows)
            if not manifest.get('repair_campaign'):
                complete &= all(not r.get('origin_evaluation_config_id') or
                                r['origin_evaluation_config_id'] == config_id(manifest) for r in source_rows)
    provenance_eligible = bool(complete)
    question_sets = []
    for group, rows in grouped.items():
        # Latest successful retry is still a distinct recorded attempt; fail duplicates to avoid cherry-picking.
        ids = [r['question_id'] for r in rows]
        question_sets.append(set(ids))
        eligible = provenance_eligible and len(ids) == len(set(ids)) and len(ids) >= minimum_questions
        reviewed = []
        for row in rows:
            if row.get('status') not in TERMINAL_STATUSES:
                eligible = False
                continue
            grade = by_request.get(row['request_id'], {})
            if grade.get('question_id') != row['question_id']:
                eligible = False
                continue
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
        'reason': 'Complete independent reviews, full planned coverage, and consistent non-repair configuration required',
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

"""Integrate measured winners only after both independent comparisons are complete."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def deployment(architecture_file: Path, retrieval_file: Path, variants_file: Path) -> dict:
    architecture = json.loads(architecture_file.read_text())
    retrieval = json.loads(retrieval_file.read_text())
    if not architecture.get('complete') or not retrieval.get('complete'):
        raise ValueError('Both experiments must be independently graded before integration')
    if architecture.get('winner') not in {'fast', 'verified', 'adaptive'}:
        raise ValueError('Invalid measured architecture winner')
    variants = json.loads(variants_file.read_text())
    variant = next((v for v in variants if v['configuration'] == retrieval['winner']), None)
    if variant is None or not Path(variant['index']).exists():
        raise ValueError('Measured retrieval index is missing')
    return {'architecture': architecture['winner'], 'index': variant['index'],
            'top_k': variant['top_k'], 'source': 'independently_graded_experiments',
            'architecture_selection': str(architecture_file),
            'retrieval_selection': str(retrieval_file)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--architecture-selection', required=True)
    parser.add_argument('--retrieval-selection', required=True)
    parser.add_argument('--variants', required=True)
    parser.add_argument('--output', default='results/deployment.json')
    args = parser.parse_args()
    config = deployment(Path(args.architecture_selection), Path(args.retrieval_selection), Path(args.variants))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2))


if __name__ == '__main__':
    main()

"""Append-only raw attempts and an atomic, unique-per-evaluation CSV view."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path


TERMINAL_STATUSES = frozenset({
    'ok', 'insufficient_evidence', 'conversation', 'out_of_scope',
    'citation_validation_failed', 'verification_failed', 'revision_failed',
    'revision_verification_failed',
})


def config_id(manifest: dict) -> str:
    semantic = manifest.get('evaluation_configuration', manifest)
    return hashlib.sha256(json.dumps(semantic, sort_keys=True).encode()).hexdigest()


def evaluation_key(row: dict, identity: str) -> tuple[str, str, str]:
    return row['question_id'], row['architecture'], identity


def read_rows(path: Path) -> list[dict]:
    if not path.exists() or not path.stat().st_size:
        return []
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def compatible_history(path: Path, manifest: dict, *, repair: bool = False) -> dict:
    companion = path.with_suffix('.manifest.json')
    if not companion.exists():
        raise ValueError(f'Missing configuration manifest for {path}')
    old = json.loads(companion.read_text())
    if (old.get('repair_campaign') or old.get('selection_eligible') is False) and not manifest.get('repair_campaign'):
        raise ValueError('Repair history cannot become a selection-eligible experiment')
    if 'evaluation_configuration' in old:
        if config_id(old) != config_id(manifest):
            if not repair:
                raise ValueError(f'Evaluation configuration differs: {path}. Use a separately labeled --repair-from campaign or a fresh full experiment.')
            old_config = old['evaluation_configuration']
            new_config = manifest['evaluation_configuration']
            allowed = {'rag_logic_sha256', 'structured_grounding', 'verifier_format_retries', 'verifier_retry_output_tokens'}
            changed = [key for key in old_config.keys() | new_config.keys()
                       if old_config.get(key) != new_config.get(key) and key not in allowed]
            if changed:
                raise ValueError(f'Repair cannot change experiment settings: {changed}')
    else:
        # Legacy source hash mixed prompt code with infrastructure. Preserve it as
        # provenance and validate every declared experimental setting explicitly.
        keys = ('model', 'top_k', 'index_sha256', 'routing', 'verification_evidence',
                'revision', 'configuration', 'seed', 'mode', 'workers', 'rpm', 'tpm', 'safety_margin')
        changed = [key for key in keys if old.get(key) != manifest.get(key)]
        if changed:
            raise ValueError(f'Legacy experiment settings differ ({", ".join(changed)}): {path}')
    return old


class CheckpointStore:
    def __init__(self, path: Path, fields: tuple[str, ...], manifest: dict):
        self.path, self.fields = path, fields
        self.manifest = manifest
        self.identity = config_id(manifest)
        self.terminal_statuses = (frozenset({'ok', 'insufficient_evidence', 'conversation', 'out_of_scope'})
                                 if manifest.get('repair_campaign') else TERMINAL_STATUSES)
        self.journal = path.with_suffix('.attempts.jsonl')
        self.rows: dict[tuple, dict] = {}
        self.attempt_ids: set[str] = set()
        self.max_execution_order = 0
        if path.exists() and path.stat().st_size and not self.journal.exists():
            raise ValueError('Historical CSV is immutable. Use --resume-from HISTORY --output NEW.csv')
        if self.journal.exists():
            with self.journal.open(encoding='utf-8') as handle:
                lines = handle.readlines()
            for index, line in enumerate(lines):
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    # A crash can leave only the last append incomplete. Never
                    # truncate raw data; refuse until an explicit recovery is made.
                    raise ValueError(f'Incomplete checkpoint journal line {index + 1}: {self.journal}')
                if event['evaluation_config_id'] != self.identity:
                    raise ValueError('Checkpoint belongs to another evaluation configuration')
                self._accept(event['row'])

    def _accept(self, row: dict) -> None:
        self.rows[evaluation_key(row, self.identity)] = row
        self.attempt_ids.add(row['request_id'])
        self.max_execution_order = max(self.max_execution_order, int(row.get('execution_order') or 0))

    def completed(self) -> set[tuple[str, str, str]]:
        return {key for key, row in self.rows.items() if row['status'] in self.terminal_statuses}

    def record(self, row: dict, *, input_file: str | None = None) -> bool:
        if row['request_id'] in self.attempt_ids:
            return False
        key = evaluation_key(row, self.identity)
        if key in self.completed():
            return False
        row = dict(row, evaluation_config_id=self.identity)
        if input_file:
            row['input_file'] = input_file
        self.path.parent.mkdir(parents=True, exist_ok=True)
        event = {'evaluation_config_id': self.identity, 'row': row,
                 'execution_source_sha256': self.manifest.get('source_sha256'),
                 'historical_input': input_file}
        with self.journal.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        self._accept(row)
        self.snapshot()
        return True

    def snapshot(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w', newline='', encoding='utf-8',
                                         dir=self.path.parent, delete=False) as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(self.rows.values())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, self.path)

    def import_history(self, path: Path, *, repair: bool = False) -> None:
        old = compatible_history(path, self.manifest, repair=repair)
        for row in read_rows(path):
            row['origin_evaluation_config_id'] = row.get('origin_evaluation_config_id') or config_id(old)
            self.record(row, input_file=str(path))

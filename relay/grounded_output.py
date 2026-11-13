"""One-call structured draft contract; citations remain request-local and validated."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from .providers.base import ProviderError


REFUSAL_TEXT = 'The provided papers do not establish an answer to this question.'
GROUNDING_CONTRACT = (
    'Return the grounded_answer JSON schema. For an answer, use concise claims; put exact '
    'source aliases in sources, not citation brackets in text. Every claim needs sources. '
    'Use decision refuse and claims [] only when the question cannot be answered from evidence. '
    'Prefer a short supported answer to embellishment. Never invent a source or claim.')


class GroundedClaim(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    text: str
    sources: list[str]


class GroundedDraft(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    decision: Literal['answer', 'refuse']
    claims: list[GroundedClaim]


def draft_schema(aliases: list[str]) -> dict:
    return {'type': 'object', 'properties': {
        'decision': {'type': 'string', 'enum': ['answer', 'refuse']},
        'claims': {'type': 'array', 'items': {'type': 'object', 'properties': {
            'text': {'type': 'string'},
            'sources': {'type': 'array', 'items': {'type': 'string', 'enum': aliases}},
        }, 'required': ['text', 'sources'], 'additionalProperties': False}},
    }, 'required': ['decision', 'claims'], 'additionalProperties': False}


def render_draft(data: dict, aliases: list[str]) -> str:
    try:
        draft = GroundedDraft.model_validate(data)
        if draft.decision == 'refuse':
            if draft.claims:
                raise ValueError('A refusal must have no factual claims')
            return REFUSAL_TEXT
        if not draft.claims:
            raise ValueError('An answer must contain supported claims')
        rendered = []
        for claim in draft.claims:
            if not claim.text.strip() or not claim.sources or set(claim.sources) - set(aliases):
                raise ValueError('Each claim requires nonempty text and valid source aliases')
            rendered.append(claim.text.strip() + ' ' + ''.join(
                f'[{source}]' for source in dict.fromkeys(claim.sources)))
        return '\n\n'.join(rendered)
    except ValueError as exc:
        raise ProviderError(f'Invalid grounded draft: {exc}') from exc

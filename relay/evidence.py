from __future__ import annotations

from dataclasses import dataclass

from .retrieval import tokenize
from .schemas import RetrievedPassage


@dataclass(slots=True)
class EvidenceDecision:
    sufficient_for_generation: bool
    reason: str


def assess_evidence(question: str, passages: list[RetrievedPassage]) -> EvidenceDecision:
    """A conservative gate, not a claim that the question is answerable."""
    if not passages:
        return EvidenceDecision(False, "retrieval returned no passages")
    query_terms = {t for t in tokenize(question) if len(t) > 2}
    evidence_terms = set().union(*(set(tokenize(p.chunk.text)) for p in passages))
    overlap = query_terms & evidence_terms
    if query_terms and not overlap:
        return EvidenceDecision(False, "no meaningful lexical overlap with the retrieved collection")
    return EvidenceDecision(True, "retrieved evidence is available; the model must still refuse unsupported claims")

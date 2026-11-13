from __future__ import annotations

import re
from dataclasses import dataclass, field

from .schemas import RetrievedPassage


BRACKET_RE = re.compile(r"\[([^\[\]\n]+)\]")
ALIAS_RE = re.compile(r"S[1-9]\d*")


@dataclass(slots=True)
class CitationValidation:
    valid: bool
    cited_ids: list[str] = field(default_factory=list)
    invalid_ids: list[str] = field(default_factory=list)
    reason: str = ""
    cited_aliases: list[str] = field(default_factory=list)
    alias_mapping: dict[str, str] = field(default_factory=dict)


def build_alias_mapping(evidence: list[RetrievedPassage]) -> dict[str, str]:
    """Stable request-local aliases in final evidence order."""
    return {f"S{index}": passage.chunk.chunk_id for index, passage in enumerate(evidence, 1)}


def validate_citations(answer: str, evidence: list[RetrievedPassage],
                       alias_mapping: dict[str, str] | None = None,
                       refusal_text: str | None = None) -> CitationValidation:
    mapping = alias_mapping or build_alias_mapping(evidence)
    if refusal_text is not None and answer.strip() == refusal_text:
        return CitationValidation(True, reason="explicit refusal; citations not required",
                                  alias_mapping=mapping)
    raw = [value.strip() for value in BRACKET_RE.findall(answer)]
    aliases = list(dict.fromkeys(raw))
    invalid = [value for value in aliases if not ALIAS_RE.fullmatch(value) or value not in mapping]
    cited_ids = [mapping[value] for value in aliases if value in mapping and ALIAS_RE.fullmatch(value)]
    available = {passage.chunk.chunk_id for passage in evidence}
    unresolved = [chunk_id for chunk_id in cited_ids if chunk_id not in available]
    invalid.extend(item for item in unresolved if item not in invalid)
    valid = bool(aliases) and not invalid
    if valid:
        reason = "ok; aliases resolve to retrieved chunks (semantic support not implied)"
    elif not aliases:
        reason = "no citation aliases found"
    else:
        reason = "nonexistent, malformed, or ambiguous citation alias"
    return CitationValidation(valid, cited_ids, invalid, reason, aliases, mapping)


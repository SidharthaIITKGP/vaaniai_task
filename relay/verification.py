from __future__ import annotations

from typing import Any

from .citation_validator import build_alias_mapping
from .providers.base import ModelProvider, ProviderError
from .schemas import ProviderResponse, RetrievedPassage, VerificationResult


def format_evidence(passages: list[RetrievedPassage],
                    aliases: dict[str, str] | None = None) -> str:
    aliases = aliases or build_alias_mapping(passages)
    chunk_to_alias = {chunk_id: alias for alias, chunk_id in aliases.items()}
    blocks = []
    for item in passages:
        c = item.chunk
        blocks.append(
            f"CITATION_ALIAS: {chunk_to_alias[c.chunk_id]}\nCHUNK_ID: {c.chunk_id}"
            f"\nPAPER: {c.paper_title}\nARXIV: {c.paper_id}{c.arxiv_version}"
            f"\nPDF_PAGE: {c.page_number}\nSECTION: {c.section or 'unknown'}\nPASSAGE:\n{c.text}"
        )
    return "\n\n---\n\n".join(blocks)


async def verify_answer(provider: ModelProvider, question: str, answer: str,
                        evidence: list[RetrievedPassage],
                        aliases: dict[str, str] | None = None) -> tuple[VerificationResult, ProviderResponse]:
    aliases = aliases or build_alias_mapping(evidence)
    messages = [
        {"role": "system", "content": (
            "You are an independent evidence verifier. Check every factual claim against the supplied passages. "
            "A citation label alone is not support. Return JSON only with: supported (boolean), "
            "unsupported_claims (array of strings), invalid_citations (array of strings), "
            "decision ('return', 'refuse', or 'revise'), and reason (string). Be conservative."
        )},
        {"role": "user", "content": f"QUESTION:\n{question}\n\nDRAFT:\n{answer}\n\nEVIDENCE:\n{format_evidence(evidence, aliases)}"},
    ]
    schema = {
        "type": "object",
        "properties": {
            "supported": {"type": "boolean"},
            "unsupported_claims": {"type": "array", "items": {"type": "string"}},
            "invalid_citations": {"type": "array", "items": {"type": "string"}},
            "decision": {"type": "string", "enum": ["return", "refuse", "revise"]},
            "reason": {"type": "string"},
        },
        "required": ["supported", "unsupported_claims", "invalid_citations", "decision", "reason"],
        "additionalProperties": False,
    }
    data, response = await provider.generate_json(
        messages, json_schema=schema, schema_name="answer_verification", strict=True,
        max_output_tokens=600, reasoning_effort="low")
    try:
        decision = str(data.get("decision", "refuse")).lower()
        if decision not in {"return", "refuse", "revise"}:
            decision = "refuse"
        unsupported = data.get("unsupported_claims", [])
        invalid = data.get("invalid_citations", [])
        if not isinstance(unsupported, list) or not isinstance(invalid, list):
            raise TypeError("claim and citation fields must be arrays")
        result = VerificationResult(
            supported=data.get("supported") is True,
            unsupported_claims=[str(x) for x in unsupported],
            invalid_citations=[str(x) for x in invalid],
            decision=decision, reason=str(data.get("reason", "")),
        )
    except (TypeError, ValueError) as exc:
        raise ProviderError(f"Malformed verifier response: {exc}") from exc
    if not result.supported or result.unsupported_claims or result.invalid_citations:
        result.decision = "refuse" if result.decision == "return" else result.decision
    return result, response


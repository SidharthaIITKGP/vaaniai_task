from __future__ import annotations

from typing import Any
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from .citation_validator import build_alias_mapping
from .providers.base import ModelProvider, ProviderError
from .schemas import ProviderResponse, RetrievedPassage, VerificationResult


class VerifierOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    decision: Literal["return", "refuse", "revise"]
    unsupported_claims: list[str]
    reason: str


def format_evidence(passages: list[RetrievedPassage],
                    aliases: dict[str, str] | None = None) -> str:
    aliases = aliases or build_alias_mapping(passages)
    chunk_to_alias = {chunk_id: alias for alias, chunk_id in aliases.items()}
    blocks = []
    for item in passages:
        c = item.chunk
        blocks.append(
            f"[{chunk_to_alias[c.chunk_id]}] {c.paper_title}; arXiv {c.paper_id}{c.arxiv_version}; "
            f"PDF p.{c.page_number}; {c.section or 'section unknown'}\n{c.text}"
        )
    return "\n\n---\n\n".join(blocks)


async def verify_answer(provider: ModelProvider, question: str, answer: str,
                        evidence: list[RetrievedPassage],
                        aliases: dict[str, str] | None = None, *,
                        request_id: str | None = None,
                        architecture: str | None = None,
                        stage: str = "verification") -> tuple[VerificationResult, ProviderResponse]:
    aliases = aliases or build_alias_mapping(evidence)
    messages = [
        {"role": "system", "content": (
            "Verify every factual claim against only the supplied cited passages. A citation alone is not proof. "
            "Return all three schema fields. Use return only when every claim is supported; otherwise use revise "
            "when removing/correcting claims can fix the draft, or refuse. Keep the claim list and reason concise."
        )},
        {"role": "user", "content": f"QUESTION:\n{question}\n\nDRAFT:\n{answer}\n\nEVIDENCE:\n{format_evidence(evidence, aliases)}"},
    ]
    schema = {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": ["return", "refuse", "revise"]},
            "unsupported_claims": {"type": "array", "items": {"type": "string"}},
            "reason": {"type": "string"},
        },
        "required": ["decision", "unsupported_claims", "reason"],
        "additionalProperties": False,
    }
    data, response = await provider.generate_json(
        messages, json_schema=schema, schema_name="answer_verification", strict=True,
        max_output_tokens=640, reasoning_effort="low", stage=stage, request_id=request_id,
        architecture=architecture)
    try:
        parsed = VerifierOutput.model_validate(data)
        decision = parsed.decision
        unsupported = parsed.unsupported_claims
        supported = decision == "return" and not unsupported
        result = VerificationResult(
            supported=supported,
            unsupported_claims=[str(x) for x in unsupported],
            invalid_citations=[],
            decision=decision, reason=str(data.get("reason", "")),
        )
    except (TypeError, ValueError, ValidationError) as exc:
        raise ProviderError(f"Malformed verifier response: {exc}") from exc
    if not result.supported or result.unsupported_claims:
        result.decision = "refuse" if result.decision == "return" else result.decision
    return result, response


from __future__ import annotations

from typing import Any
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from .citation_validator import build_alias_mapping
from .providers.base import ModelProvider, ProviderError, RateLimitError, TruncatedResponseError
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
            "Match each citation alias to its passage before judging support. Accurate paraphrases count as support; "
            "verbatim wording is not required. Return all three schema fields, including unsupported_claims: [] "
            "when empty. Use return only when every claim is supported. Use revise for fixable overclaims or "
            "unsupported numbers/guarantees; use refuse only when evidence cannot support an answer. "
            "List only unsupported claim fragments; do not repeat the whole draft. Keep reason under 40 words."
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
    prior_attempts = []
    for attempt in range(2):
        response = None
        try:
            data, response = await provider.generate_json(
                messages, json_schema=schema, schema_name='answer_verification', strict=True,
                max_output_tokens=640 if attempt == 0 else 1024,
                reasoning_effort='low', stage=stage, request_id=request_id, architecture=architecture)
            parsed = VerifierOutput.model_validate(data)
            response.prior_attempts = prior_attempts
            break
        except (ProviderError, ValidationError) as exc:
            if isinstance(exc, ValidationError):
                error = ProviderError('Verifier JSON failed local schema validation', diagnostics={
                    'dispatched': True,
                    'usage_input_tokens': response.usage.input_tokens if response else None,
                    'usage_output_tokens': response.usage.output_tokens if response else None,
                    'provider_latency_ms': response.latency_ms if response else None},
                    status_code=response.provider_status_code if response else None,
                    queue_wait_ms=response.queue_wait_ms if response else None,
                    response_headers=response.rate_limit_headers if response else None)
            else:
                error = exc
            text = f'{error} {error.error_body or ""}'.lower()
            format_failure = (isinstance(exc, (ValidationError, TruncatedResponseError))
                or 'json_validate_failed' in text or 'invalid json' in text
                or 'malformed verifier' in text)
            if attempt or isinstance(exc, RateLimitError) or not format_failure:
                error.diagnostics['prior_attempts'] = prior_attempts
                raise error
            prior_attempts.append(error.details())
            messages = [*messages, {'role': 'system', 'content': (
                'The previous attempt failed JSON formatting; no verdict was accepted. Recheck the evidence '
                'and use this shape: {"decision":"revise",'
                '"unsupported_claims":[],"reason":"brief justification"}. '
                'Choose return, revise, or refuse from the evidence. Include every field. '
                'Unsupported claims must be an array of short strings.') }]
    try:
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


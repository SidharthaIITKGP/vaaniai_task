from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    paper_id: str
    paper_title: str
    arxiv_version: str
    filename: str
    page_number: int
    section: str | None
    text: str
    token_start: int = 0
    token_end: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RetrievedPassage:
    chunk: Chunk
    rank: int
    rrf_score: float
    bm25_score: float | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    dense_rank: int | None = None


@dataclass(slots=True)
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(slots=True)
class ProviderResponse:
    text: str
    usage: Usage = field(default_factory=Usage)
    latency_ms: float | None = None
    ttft_ms: float | None = None
    model: str | None = None
    queue_wait_ms: float | None = None
    provider_status_code: int | None = None
    rate_limit_type: str | None = None
    retry_after_seconds: float | None = None
    rate_limit_headers: dict[str, str] = field(default_factory=dict)
    estimated_input_tokens: int | None = None
    reserved_output_tokens: int | None = None
    reserved_total_tokens: int | None = None
    local_rpm_remaining: int | None = None
    local_tpm_remaining: float | None = None
    scheduler_wait_reason: str | None = None
    pipeline_request_id: str | None = None
    architecture: str | None = None
    stage: str | None = None
    prior_attempts: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] | None = None


@dataclass(slots=True)
class VerificationResult:
    supported: bool
    unsupported_claims: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    decision: Literal["return", "refuse", "revise"] = "refuse"
    reason: str = ""


@dataclass(slots=True)
class PipelineResult:
    request_id: str
    architecture: str
    status: str
    answer: str
    citations: list[str] = field(default_factory=list)
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    route: str | None = None
    route_reason: str | None = None
    verification: VerificationResult | None = None
    error: str | None = None


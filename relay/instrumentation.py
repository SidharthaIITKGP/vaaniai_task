from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schemas import ProviderResponse
from .providers.base import ProviderError


def ms(seconds: float) -> float:
    return round(seconds * 1000, 3)


@dataclass(slots=True)
class CallMetric:
    stage: str
    pipeline_request_id: str | None
    architecture: str | None
    latency_ms: float | None
    provider_ttft_ms: float | None
    input_tokens: int | None
    output_tokens: int | None
    model: str | None
    queue_wait_ms: float | None
    provider_status_code: int | None
    rate_limit_type: str | None
    retry_after_seconds: float | None
    estimated_input_tokens: int | None
    reserved_output_tokens: int | None
    reserved_total_tokens: int | None
    local_rpm_remaining: int | None
    local_tpm_remaining: float | None
    scheduler_wait_reason: str | None
    provider_rate_limit_headers: dict[str, str]
    dispatched: bool | None = None


@dataclass(slots=True)
class RequestTrace:
    request_id: str
    architecture: str
    arrival_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    query_processing_ms: float | None = None
    retrieval_ms: float | None = None
    evidence_decision_ms: float | None = None
    planning_ms: float | None = None
    answer_generation_ms: float | None = None
    verification_ms: float | None = None
    revision_ms: float | None = None
    total_latency_ms: float | None = None
    user_visible_ttft_ms: float | None = None
    routing_mode: str = "hybrid"
    detected_intent: str | None = None
    routing_method: str | None = None
    routing_reason: str | None = None
    routing_complexity: str | None = None
    routing_relevant_papers: list[str] = field(default_factory=list)
    routing_latency_ms: float | None = None
    routing_tokens_in: int | None = None
    routing_tokens_out: int | None = None
    routing_model_calls: int = 0
    routing_error: str | None = None
    routing_schema_error: str | None = None
    routing_fallback_used: bool = False
    llm_calls: list[CallMetric] = field(default_factory=list)
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    route: str | None = None
    route_reason: str | None = None
    citation_validation: dict[str, Any] | None = None
    citation_alias_mapping: dict[str, str] = field(default_factory=dict)
    citation_error_reason: str | None = None
    verification_outcome: dict[str, Any] | None = None
    verification_executed: bool = False
    verification_supported: bool | None = None
    pipeline_failure_stage: str | None = None
    final_status: str | None = None
    provider_error: dict[str, Any] | None = None
    draft_answer: str | None = None
    revised_draft: str | None = None
    evidence_character_budget: int | None = None
    evidence_source_characters: int | None = None
    evidence_rendered_characters: int | None = None
    evidence_shortened: bool = False

    def add_call(self, stage: str, response: ProviderResponse) -> None:
        for previous in response.prior_attempts:
            self.add_failed_call(f'{stage}_format_attempt', _previous_error(previous))
        self.llm_calls.append(CallMetric(stage, response.pipeline_request_id,
                                         response.architecture,
                                         response.latency_ms, response.ttft_ms,
                                         response.usage.input_tokens, response.usage.output_tokens,
                                         response.model, response.queue_wait_ms,
                                         response.provider_status_code, response.rate_limit_type,
                                         response.retry_after_seconds,
                                         response.estimated_input_tokens,
                                         response.reserved_output_tokens,
                                         response.reserved_total_tokens,
                                         response.local_rpm_remaining,
                                         response.local_tpm_remaining,
                                         response.scheduler_wait_reason,
                                         response.rate_limit_headers, True))

    def add_failed_call(self, stage: str, error: Any | None = None) -> None:
        details = getattr(error, 'diagnostics', {})
        for previous in details.get('prior_attempts', []):
            self.add_failed_call(f'{stage}_format_attempt', _previous_error(previous))
        self.llm_calls.append(CallMetric(stage, self.request_id, self.architecture,
                                         details.get('provider_latency_ms'), None,
                                         details.get('usage_input_tokens'), details.get('usage_output_tokens'), None,
                                         getattr(error, "queue_wait_ms", None),
                                         getattr(error, "status_code", None),
                                         getattr(error, "rate_limit_type", None),
                                         _float_or_none(getattr(error, "retry_after", None)),
                                         details.get('estimated_input_tokens'),
                                         details.get('reserved_output_tokens'),
                                         details.get('reserved_total_tokens'),
                                         details.get('local_rpm_remaining'),
                                         details.get('local_tpm_remaining'),
                                         details.get('scheduler_wait_reason'),
                                         getattr(error, 'response_headers', {}),
                                         details.get('dispatched', False if
                                             (getattr(error, 'rate_limit_type', '') or '').startswith('local_')
                                             else True if getattr(error, 'status_code', None) else None)))

    def metrics(self) -> dict[str, Any]:
        values = asdict(self)
        counted = [call for call in self.llm_calls if call.dispatched is not False]
        values['stage_admission_attempts'] = len(self.llm_calls)
        values["total_model_calls"] = len(counted)
        values['routing_model_calls'] = sum(c.stage == 'intent_routing' for c in counted)
        values["architecture_model_calls"] = sum(c.stage != 'intent_routing' for c in counted)
        values['confirmed_provider_dispatches'] = sum(c.dispatched is True for c in self.llm_calls)
        inputs = [c.input_tokens for c in counted if c.input_tokens is not None]
        outputs = [c.output_tokens for c in counted if c.output_tokens is not None]
        values["total_input_tokens"] = (
            0 if not counted else
            sum(inputs) if len(inputs) == len(counted) else None
        )
        values["total_output_tokens"] = (
            0 if not counted else
            sum(outputs) if len(outputs) == len(counted) else None
        )
        provider_ttfts = [c.provider_ttft_ms for c in self.llm_calls if c.provider_ttft_ms is not None]
        values["provider_ttft_ms"] = provider_ttfts[0] if provider_ttfts else None
        waits = [c.queue_wait_ms for c in self.llm_calls if c.queue_wait_ms is not None]
        values["queue_wait_ms"] = sum(waits) if waits else (0 if not self.llm_calls else None)
        values["provider_status_code"] = next(
            (c.provider_status_code for c in reversed(self.llm_calls) if c.provider_status_code is not None), None)
        values["rate_limit_type"] = next(
            (c.rate_limit_type for c in reversed(self.llm_calls) if c.rate_limit_type), None)
        values["retry_after_seconds"] = next(
            (c.retry_after_seconds for c in reversed(self.llm_calls) if c.retry_after_seconds is not None), None)
        stages: dict[str, int] = {}
        for call in counted:
            stages[call.stage] = stages.get(call.stage, 0) + 1
        values["model_calls_by_stage"] = stages
        values["total_ms"] = self.total_latency_ms
        return values


class JsonlLogger:
    def __init__(self, path: Path):
        self.path = path

    def write(self, trace: RequestTrace) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trace.metrics(), ensure_ascii=False) + "\n")


class Timer:
    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        self.elapsed_ms = ms(time.perf_counter() - self.start)


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _previous_error(details: dict[str, Any]) -> ProviderError:
    return ProviderError(details.get('message', 'Structured output formatting failure'),
        status_code=details.get('status_code'), retry_after=details.get('retry_after'),
        rate_limit_type=details.get('rate_limit_type'), queue_wait_ms=details.get('queue_wait_ms'),
        response_headers=details.get('response_headers'), error_body=details.get('error_body'),
        diagnostics=details.get('diagnostics'))


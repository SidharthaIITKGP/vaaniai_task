from __future__ import annotations

import time
import uuid
from collections.abc import Callable

from .architectures import AdaptiveRAG, FastRAG, VerifiedRAG
from .config import RelayConfig
from .instrumentation import JsonlLogger, RequestTrace, ms
from .intent_router import IntentRouter
from .providers.base import ModelProvider, ProviderError, RateLimitError
from .retrieval import HybridRetriever
from .schemas import PipelineResult


class RAGPipeline:
    strategies = {"fast": FastRAG, "verified": VerifiedRAG, "adaptive": AdaptiveRAG}

    def __init__(self, architecture: str, retriever: HybridRetriever | Callable[[], HybridRetriever],
                 provider: ModelProvider, config: RelayConfig | None = None,
                 logger: JsonlLogger | None = None, router: IntentRouter | None = None):
        if architecture not in self.strategies:
            raise ValueError(f"architecture must be one of: {', '.join(self.strategies)}")
        self.config = config or RelayConfig()
        self.architecture = architecture
        self._retriever = retriever
        self.provider = provider
        self.router = router or IntentRouter(provider, self.config.intent_router)
        self.logger = logger or JsonlLogger(self.config.log_path)

    def _strategy(self):
        retriever = self._retriever() if callable(self._retriever) else self._retriever
        return self.strategies[self.architecture](retriever, self.provider, self.config)

    async def run(self, question: str) -> PipelineResult:
        if not question or not question.strip():
            raise ValueError("question cannot be empty")
        trace = RequestTrace(str(uuid.uuid4()), self.architecture,
                             routing_mode=self.config.intent_router.mode)
        started = time.perf_counter()
        try:
            routing = await self.router.route(question)
            trace.detected_intent = routing.intent
            trace.routing_method = routing.method
            trace.routing_reason = routing.reason
            trace.routing_complexity = routing.complexity
            trace.routing_relevant_papers = routing.relevant_papers
            trace.routing_latency_ms = routing.latency_ms
            trace.routing_model_calls = routing.model_calls
            trace.routing_error = routing.error
            trace.routing_schema_error = routing.schema_error
            trace.routing_fallback_used = routing.fallback_used
            if routing.model_calls == 0:
                trace.routing_tokens_in = 0
                trace.routing_tokens_out = 0
            if routing.provider_response is not None:
                trace.add_call("intent_routing", routing.provider_response)
                trace.routing_tokens_in = routing.provider_response.usage.input_tokens
                trace.routing_tokens_out = routing.provider_response.usage.output_tokens
            elif routing.model_calls:
                trace.add_failed_call("intent_routing", routing.provider_error)
            if routing.intent != "paper_question":
                status = "conversation" if routing.intent == "conversation" else "out_of_scope"
                trace.final_status = status
                result = PipelineResult(trace.request_id, self.architecture, status,
                                        routing.response or "", metrics=trace.metrics())
            else:
                result = await self._strategy().run(question, trace)
        except (RateLimitError, ProviderError) as exc:
            status = "rate_limited" if isinstance(exc, RateLimitError) else "provider_error"
            trace.provider_error = {"type": type(exc).__name__, "message": str(exc),
                                    "status_code": exc.status_code, "retry_after": exc.retry_after,
                                    "rate_limit_type": exc.rate_limit_type,
                                    "queue_wait_ms": exc.queue_wait_ms}
            if trace.pipeline_failure_stage:
                trace.add_failed_call(trace.pipeline_failure_stage, exc)
            trace.final_status = status
            result = PipelineResult(trace.request_id, self.architecture, status, "",
                                    retrieved_chunk_ids=trace.retrieved_chunk_ids,
                                    metrics=trace.metrics(), error=str(exc),
                                    route=trace.route, route_reason=trace.route_reason)
        trace.total_latency_ms = ms(time.perf_counter() - started)
        if result.answer:
            trace.user_visible_ttft_ms = trace.total_latency_ms
        result.metrics = trace.metrics()
        self.logger.write(trace)
        return result

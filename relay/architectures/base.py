from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Any

from ..citation_validator import build_alias_mapping, validate_citations
from ..config import RelayConfig
from ..evidence import assess_evidence
from ..instrumentation import RequestTrace, Timer
from ..providers.base import ModelProvider
from ..retrieval import (HybridRetriever, comparison_paper_ids,
                         merge_passages_with_coverage)
from ..schemas import PipelineResult, ProviderResponse, RetrievedPassage
from ..verification import format_evidence


REFUSAL = "The provided papers do not establish an answer to this question."


class BaseArchitecture(ABC):
    name: str

    def __init__(self, retriever: HybridRetriever, provider: ModelProvider, config: RelayConfig):
        self.retriever, self.provider, self.config = retriever, provider, config

    def process_query(self, question: str) -> str:
        return re.sub(r"\s+", " ", question).strip()

    def retrieve(self, question: str, trace: RequestTrace) -> tuple[list[RetrievedPassage], bool]:
        with Timer() as timer:
            required = comparison_paper_ids(question)
            if len(required) >= 2:
                per_paper = max(1, self.config.retrieval.top_k // len(required))
                groups = [self.retriever.retrieve(question)]
                groups.extend(self.retriever.retrieve(
                    question, top_k=per_paper, paper_ids={paper_id}) for paper_id in required)
                evidence = merge_passages_with_coverage(
                    groups, self.config.retrieval.top_k, required)
            else:
                evidence = self.retriever.retrieve(question)
        trace.retrieval_ms = timer.elapsed_ms
        trace.retrieved_chunk_ids = [p.chunk.chunk_id for p in evidence]
        with Timer() as timer:
            decision = assess_evidence(question, evidence)
        trace.evidence_decision_ms = timer.elapsed_ms
        return evidence, decision.sufficient_for_generation

    async def generate(self, question: str, evidence: list[RetrievedPassage],
                       trace: RequestTrace, stage: str = "generation") -> tuple[str, ProviderResponse]:
        aliases = build_alias_mapping(evidence)
        trace.citation_alias_mapping = aliases
        messages = [
            {"role": "system", "content": (
                "Answer only from the supplied evidence. Cite every factual claim using only the short aliases "
                "shown at the start of each passage, formatted exactly like [S1] or [S1][S2]. Never put multiple aliases "
                "inside one bracket and never cite CHUNK_ID, page, section, or SOURCE_ID text. "
                f"If the evidence does not establish the answer, respond exactly: {REFUSAL}"
            )},
            {"role": "user", "content": f"QUESTION:\n{question}\n\nEVIDENCE:\n{format_evidence(evidence, aliases)}"},
        ]
        trace.pipeline_failure_stage = stage
        response = await self.provider.generate_streaming(
            messages, stage=stage, request_id=trace.request_id, architecture=self.name)
        trace.add_call(stage, response)
        trace.pipeline_failure_stage = None
        return response.text.strip(), response

    def result(self, trace: RequestTrace, status: str, answer: str,
               citations: list[str] | None = None, **kwargs: Any) -> PipelineResult:
        trace.final_status = status
        if trace.pipeline_failure_stage is None:
            trace.pipeline_failure_stage = {
                "insufficient_evidence": "evidence_decision",
                "citation_validation_failed": "citation_validation",
                "verification_failed": "verification",
                "revision_failed": "revision",
                "revision_verification_failed": "revision_verification",
            }.get(status)
        return PipelineResult(trace.request_id, self.name, status, answer,
                              citations or [], trace.retrieved_chunk_ids,
                              metrics=trace.metrics(), route=trace.route,
                              route_reason=trace.route_reason, **kwargs)

    @abstractmethod
    async def run(self, question: str, trace: RequestTrace) -> PipelineResult: ...

    def prepare(self, question: str, trace: RequestTrace) -> str:
        with Timer() as timer:
            processed = self.process_query(question)
        trace.query_processing_ms = timer.elapsed_ms
        return processed

    def deterministic_check(self, answer: str, evidence: list[RetrievedPassage], trace: RequestTrace):
        aliases = trace.citation_alias_mapping or build_alias_mapping(evidence)
        check = validate_citations(answer, evidence, aliases, refusal_text=REFUSAL)
        trace.citation_validation = asdict(check)
        trace.citation_error_reason = None if check.valid else check.reason
        return check


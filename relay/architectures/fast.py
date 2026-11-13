from __future__ import annotations

from ..instrumentation import RequestTrace, Timer
from ..schemas import PipelineResult
from .base import BaseArchitecture, REFUSAL


class FastRAG(BaseArchitecture):
    name = "fast"

    async def run(self, question: str, trace: RequestTrace) -> PipelineResult:
        query = self.prepare(question, trace)
        evidence, sufficient = self.retrieve(query, trace)
        if not sufficient:
            return self.result(trace, "insufficient_evidence", REFUSAL)
        with Timer() as timer:
            answer, _ = await self.generate(query, evidence, trace)
        trace.answer_generation_ms = timer.elapsed_ms
        check = self.deterministic_check(answer, evidence, trace)
        if answer == REFUSAL:
            return self.result(trace, "insufficient_evidence", answer)
        if not check.valid:
            return self.result(trace, "citation_validation_failed", REFUSAL)
        return self.result(trace, "ok", answer, check.cited_ids)


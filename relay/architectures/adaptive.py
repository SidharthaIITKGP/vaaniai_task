from __future__ import annotations

import re

from ..instrumentation import RequestTrace, Timer
from ..retrieval import comparison_paper_ids, merge_passages_with_coverage
from ..schemas import PipelineResult
from .base import REFUSAL
from .verified import VerifiedRAG


class AdaptiveRAG(VerifiedRAG):
    name = "adaptive"

    def route(self, question: str) -> tuple[str, str]:
        lower = question.lower()
        indicators = []
        if any(word in lower for word in ("compare", "contrast", "across papers", "differences", "versus", " vs ")):
            indicators.append("cross-paper comparison")
        if len(re.findall(r"\?", question)) > 1 or any(x in lower for x in (" and why", " and how", "first,", "second,")):
            indicators.append("multi-part question")
        if len(question.split()) >= 35:
            indicators.append("long question")
        return ("complex", ", ".join(indicators)) if indicators else ("simple", "no configured complexity indicator matched")

    async def _plan(self, question: str, trace: RequestTrace) -> list[str]:
        messages = [{"role": "system", "content": (
            "Plan retrieval over only these papers: 1706.03762v7, 2410.00037v1, 2507.07955v2. "
            "Return JSON with relevant_papers (array), subquestions (array), and search_queries (array). "
            "Use at most four concise search queries. Do not answer the question."
        )}, {"role": "user", "content": question}]
        schema = {
            "type": "object",
            "properties": {
                "relevant_papers": {"type": "array", "items": {"type": "string"}},
                "subquestions": {"type": "array", "items": {"type": "string"}},
                "search_queries": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["relevant_papers", "subquestions", "search_queries"],
            "additionalProperties": False,
        }
        with Timer() as timer:
            trace.pipeline_failure_stage = "planning"
            data, response = await self.provider.generate_json(
                messages, json_schema=schema, schema_name="retrieval_plan", strict=True,
                reasoning_effort="low", stage="planning",
                request_id=trace.request_id, architecture=self.name)
        trace.planning_ms = timer.elapsed_ms
        trace.add_call("planning", response)
        trace.pipeline_failure_stage = None
        queries = [str(x).strip() for x in data.get("search_queries", []) if str(x).strip()]
        return queries[:self.config.max_retrieval_expansions]

    async def run(self, question: str, trace: RequestTrace) -> PipelineResult:
        query = self.prepare(question, trace)
        trace.route, trace.route_reason = self.route(query)
        initial, sufficient = self.retrieve(query, trace)
        if trace.route == "simple":
            if not sufficient:
                return self.result(trace, "insufficient_evidence", REFUSAL)
            return await self._answer_and_verify(query, initial, trace)
        queries = await self._plan(query, trace)
        groups = [initial]
        with Timer() as timer:
            groups.extend(self.retriever.retrieve(item) for item in queries)
            evidence = merge_passages_with_coverage(
                groups, self.config.retrieval.top_k, comparison_paper_ids(query))
        trace.retrieval_ms = (trace.retrieval_ms or 0) + timer.elapsed_ms
        trace.retrieved_chunk_ids = [p.chunk.chunk_id for p in evidence]
        if not evidence:
            return self.result(trace, "insufficient_evidence", REFUSAL)
        return await self._answer_and_verify(query, evidence, trace)

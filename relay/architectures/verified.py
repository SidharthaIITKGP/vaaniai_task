from __future__ import annotations

from dataclasses import asdict

from ..instrumentation import RequestTrace, Timer
from ..schemas import PipelineResult, RetrievedPassage, VerificationResult
from ..verification import format_evidence, verify_answer
from .base import BaseArchitecture, REFUSAL


class VerifiedRAG(BaseArchitecture):
    name = "verified"

    def _verification_evidence(self, evidence: list[RetrievedPassage],
                               cited_ids: list[str]) -> list[RetrievedPassage]:
        if self.config.verification_evidence_mode == "cited":
            cited = set(cited_ids)
            return [passage for passage in evidence if passage.chunk.chunk_id in cited]
        return evidence

    async def _answer_and_verify(self, query: str, evidence: list[RetrievedPassage],
                                 trace: RequestTrace) -> PipelineResult:
        with Timer() as timer:
            answer, _ = await self.generate(query, evidence, trace)
        trace.answer_generation_ms = timer.elapsed_ms
        check = self.deterministic_check(answer, evidence, trace)
        if answer == REFUSAL:
            return self.result(trace, "insufficient_evidence", answer)
        if not check.valid:
            return self.result(trace, "citation_validation_failed", REFUSAL)
        with Timer() as timer:
            trace.pipeline_failure_stage = "verification"
            verification, response = await verify_answer(
                self.provider, query, answer,
                self._verification_evidence(evidence, check.cited_ids),
                trace.citation_alias_mapping,
                request_id=trace.request_id, architecture=self.name)
        trace.verification_ms = timer.elapsed_ms
        trace.add_call("verification", response)
        trace.pipeline_failure_stage = None
        trace.verification_outcome = asdict(verification)
        trace.verification_executed = True
        trace.verification_supported = verification.supported
        if verification.supported and verification.decision == "return":
            return self.result(trace, "ok", answer, check.cited_ids, verification=verification)
        if self.config.enable_revision and self.config.max_revisions > 0 and verification.decision == "revise":
            return await self._revise_once(query, answer, evidence, trace, verification)
        return self.result(trace, "verification_failed", REFUSAL, verification=verification)

    async def _revise_once(self, query: str, draft: str, evidence: list[RetrievedPassage],
                           trace: RequestTrace, feedback: VerificationResult) -> PipelineResult:
        cited_ids = (trace.citation_validation or {}).get("cited_ids", [])
        revision_evidence = self._verification_evidence(evidence, cited_ids)
        with Timer() as timer:
            messages = [{"role": "system", "content": (
                "Revise using only the evidence and feedback. Cite only aliases like [S1] or [S1][S2]. "
                "Remove every unsupported claim. Return only the revised answer."
            )}, {"role": "user", "content": (
                f"Question: {query}\nDraft: {draft}\nFeedback: {feedback.reason}"
                f"\nUnsupported claims: {feedback.unsupported_claims}"
                f"\n\nEVIDENCE:\n{format_evidence(revision_evidence, trace.citation_alias_mapping)}"
            )}]
            trace.pipeline_failure_stage = "revision"
            response = await self.provider.generate_streaming(
                messages, stage="revision", request_id=trace.request_id, architecture=self.name)
            trace.add_call("revision", response)
            revised = response.text.strip()
            trace.pipeline_failure_stage = None
        trace.revision_ms = timer.elapsed_ms
        check = self.deterministic_check(revised, evidence, trace)
        if not check.valid:
            return self.result(trace, "revision_failed", REFUSAL)
        with Timer() as timer:
            trace.pipeline_failure_stage = "revision_verification"
            verified, response = await verify_answer(
                self.provider, query, revised,
                self._verification_evidence(evidence, check.cited_ids),
                trace.citation_alias_mapping,
                request_id=trace.request_id, architecture=self.name,
                stage="revision_verification")
        trace.verification_ms = (trace.verification_ms or 0) + timer.elapsed_ms
        trace.add_call("revision_verification", response)
        trace.pipeline_failure_stage = None
        trace.verification_outcome = asdict(verified)
        trace.verification_executed = True
        trace.verification_supported = verified.supported
        if verified.supported and verified.decision == "return":
            return self.result(trace, "ok", revised, check.cited_ids, verification=verified)
        return self.result(trace, "revision_verification_failed", REFUSAL, verification=verified)

    async def run(self, question: str, trace: RequestTrace) -> PipelineResult:
        query = self.prepare(question, trace)
        evidence, sufficient = self.retrieve(query, trace)
        if not sufficient:
            return self.result(trace, "insufficient_evidence", REFUSAL)
        return await self._answer_and_verify(query, evidence, trace)


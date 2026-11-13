import pytest
import asyncio

from relay.pipeline import RAGPipeline
from relay.providers.base import ProviderError, RateLimitError

from conftest import MockProvider


OK_VERIFY = {"unsupported_claims": [],
             "decision": "return", "reason": "all claims supported"}
FAIL_VERIFY = {"unsupported_claims": ["unsupported"],
               "decision": "refuse", "reason": "claim lacks support"}


@pytest.mark.asyncio
async def test_fast_single_call(retriever, config):
    provider = MockProvider(["Attention has heads [S1]"])
    result = await RAGPipeline("fast", retriever, provider, config).run("What is multi head attention?")
    assert result.status == "ok"
    assert provider.calls == ["generate_streaming"]
    assert result.metrics["total_model_calls"] == 1
    assert result.metrics["total_input_tokens"] == 10


@pytest.mark.asyncio
async def test_unsupported_question_refuses_before_model(retriever, config):
    provider = MockProvider([])
    result = await RAGPipeline("fast", retriever, provider, config).run("xylophone zebrafish quasar?")
    assert result.status == "insufficient_evidence"
    assert provider.calls == []
    assert result.metrics["total_model_calls"] == 0


@pytest.mark.asyncio
async def test_verified_two_calls(retriever, config):
    provider = MockProvider(["Moshi enables dialogue [S1]", OK_VERIFY])
    result = await RAGPipeline("verified", retriever, provider, config).run("How does Moshi support dialogue?")
    assert result.status == "ok"
    assert provider.calls == ["generate_streaming", "json"]
    assert result.metrics["total_model_calls"] == 2
    assert result.metrics["architecture_model_calls"] == 2
    assert result.metrics["verification_executed"] is True
    assert result.metrics["verification_supported"] is True
    verification_prompt = provider.messages_seen[1][1]["content"]
    assert "[S1]" in verification_prompt
    assert "[S2]" not in verification_prompt
    assert provider.kwargs_seen[1]["max_output_tokens"] == 640


@pytest.mark.asyncio
async def test_verification_failure_refuses(retriever, config):
    provider = MockProvider(["Unsupported claim [S1]", FAIL_VERIFY])
    result = await RAGPipeline("verified", retriever, provider, config).run("How does Moshi support dialogue?")
    assert result.status == "verification_failed"
    assert "do not establish" in result.answer
    assert result.metrics["verification_executed"] is True
    assert result.metrics["verification_supported"] is False


@pytest.mark.asyncio
async def test_adaptive_simple_and_complex_routes(retriever, config):
    simple = MockProvider(["Dialogue [S1]", OK_VERIFY])
    result = await RAGPipeline("adaptive", retriever, simple, config).run("How does Moshi support dialogue?")
    assert result.route == "simple" and len(simple.calls) == 2
    assert result.metrics["architecture_model_calls"] == 2

    plan = {"relevant_papers": ["1706.03762", "2507.07955"], "subquestions": [],
            "search_queries": ["transformer attention", "dynamic chunking"]}
    complex_provider = MockProvider([plan, "Comparison [S1][S2]", OK_VERIFY])
    result = await RAGPipeline("adaptive", retriever, complex_provider, config).run(
        "Compare Moshi and Transformer sequence modeling")
    assert result.route == "complex"
    assert complex_provider.calls == ["json", "generate_streaming", "json"]
    assert result.metrics["architecture_model_calls"] == 3


@pytest.mark.asyncio
async def test_valid_alias_to_unrelated_passage_is_rejected_semantically(retriever, config):
    provider = MockProvider(["Moshi is a translation-only model [S2]", FAIL_VERIFY])
    result = await RAGPipeline("verified", retriever, provider, config).run("How does Moshi support dialogue?")
    assert result.metrics["citation_validation"]["valid"] is True
    assert result.status == "verification_failed"


@pytest.mark.asyncio
async def test_revision_is_bounded(retriever, config):
    config.enable_revision = True
    config.max_revisions = 1
    revise = dict(FAIL_VERIFY, decision="revise")
    provider = MockProvider(["Draft [S1]", revise, "Revised [S1]", FAIL_VERIFY])
    result = await RAGPipeline("verified", retriever, provider, config).run("Explain Moshi dialogue")
    assert result.status == "revision_verification_failed"
    assert provider.calls == ["generate_streaming", "json", "generate_streaming", "json"]


@pytest.mark.asyncio
async def test_default_revision_corrects_once_and_reverifies(retriever, config):
    revise = dict(FAIL_VERIFY, decision="revise")
    provider = MockProvider(["Draft [S1]", revise, "Corrected [S1]", OK_VERIFY])
    result = await RAGPipeline("verified", retriever, provider, config).run("Explain Moshi dialogue")
    assert result.status == "ok"
    assert result.answer == "Corrected [S1]"
    assert provider.calls == ["generate_streaming", "json", "generate_streaming", "json"]
    assert result.metrics["model_calls_by_stage"] == {
        "generation": 1, "verification": 1, "revision": 1, "revision_verification": 1}


@pytest.mark.asyncio
async def test_fast_cross_paper_path_contains_each_named_paper(retriever, config):
    provider = MockProvider(["Comparison [S1][S2]"])
    result = await RAGPipeline("fast", retriever, provider, config).run(
        "Compare the original Transformer and H-Net input approaches")
    retrieved_papers = {chunk.paper_id for chunk in retriever.chunks
                        if chunk.chunk_id in result.retrieved_chunk_ids}
    assert result.status == "ok"
    assert {"1706.03762", "2507.07955"}.issubset(retrieved_papers)


@pytest.mark.asyncio
@pytest.mark.parametrize("error,status", [(ProviderError("down"), "provider_error"),
                                           (RateLimitError("slow", status_code=429, retry_after="2"), "rate_limited")])
async def test_provider_errors_are_distinct(retriever, config, error, status):
    result = await RAGPipeline("fast", retriever, MockProvider([error]), config).run("Explain attention")
    assert result.status == status and result.error


@pytest.mark.asyncio
async def test_total_interactive_deadline_returns_capacity_status(retriever, config):
    class SlowProvider(MockProvider):
        async def generate_streaming(self, messages, **kwargs):
            await asyncio.sleep(0.05)
            return await super().generate_streaming(messages, **kwargs)
    config.total_request_deadline_seconds = 0.01
    result = await RAGPipeline("fast", retriever, SlowProvider(["answer [S1]"]), config).run(
        "Explain attention")
    assert result.status == "capacity_timeout"
    assert "deadline" in result.error.lower()

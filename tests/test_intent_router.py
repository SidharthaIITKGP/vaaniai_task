from __future__ import annotations

import pytest

from relay.intent_router import IntentRouter
from relay.pipeline import RAGPipeline
from relay.providers.base import ProviderError

from conftest import MockProvider


def classification(intent, *, retrieval, reason="test", complexity="simple"):
    return {"intent": intent, "complexity": complexity, "relevant_papers": []}


@pytest.mark.asyncio
@pytest.mark.parametrize("question", ["Hey", "Hello!", "Good morning", "How are you?"])
async def test_obvious_greetings_are_rule_routed(question, config):
    config.intent_router.mode = "hybrid"
    provider = MockProvider([])
    decision = await IntentRouter(provider, config.intent_router).route(question)
    assert decision.intent == "conversation"
    assert decision.method == "rules" and decision.model_calls == 0
    assert decision.response


@pytest.mark.asyncio
@pytest.mark.parametrize("question", ["Thanks", "Thank you!", "Got it"])
async def test_acknowledgements_are_rule_routed(question, config):
    decision = await IntentRouter(MockProvider([]), config.intent_router).route(question)
    assert decision.intent == "conversation" and "welcome" in decision.response.lower()


@pytest.mark.asyncio
async def test_capability_question_is_deterministic(config):
    decision = await IntentRouter(MockProvider([]), config.intent_router).route("What can you do?")
    assert decision.intent == "conversation"
    assert "three research papers" in decision.response


@pytest.mark.asyncio
async def test_mixed_greeting_technical_question_uses_llm(config):
    config.intent_router.mode = "hybrid"
    provider = MockProvider([classification("paper_question", retrieval=True)])
    decision = await IntentRouter(provider, config.intent_router).route("Hey, explain multi-head attention")
    assert decision.intent == "paper_question" and decision.method == "rules"
    assert provider.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("question", [
    "Explain Moshi's full-duplex architecture",
    "How do learned boundaries affect hierarchical sequence processing?",
])
async def test_explicit_and_implicit_paper_questions(question, config):
    config.intent_router.mode = "hybrid"
    provider = MockProvider([classification("paper_question", retrieval=True)])
    decision = await IntentRouter(provider, config.intent_router).route(question)
    assert decision.intent == "paper_question"


@pytest.mark.asyncio
async def test_out_of_scope_has_no_unrestricted_answer(config):
    config.intent_router.mode = "hybrid"
    provider = MockProvider([classification("out_of_scope", retrieval=False)])
    decision = await IntentRouter(provider, config.intent_router).route("Who won the 2022 FIFA World Cup?")
    assert decision.intent == "out_of_scope"
    assert "only answer factual questions" in decision.response
    assert "Argentina" not in decision.response


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [
    {"intent": "anything", "complexity": "simple", "relevant_papers": []},
    {"intent": "paper_question", "complexity": "invalid", "relevant_papers": []},
])
async def test_invalid_classifier_falls_back_to_rag(bad, config):
    config.intent_router.mode = "llm"
    decision = await IntentRouter(MockProvider([bad]), config.intent_router).route("ambiguous question")
    assert decision.intent == "paper_question"
    assert decision.error and decision.model_calls == 1


@pytest.mark.asyncio
async def test_routing_provider_failure_falls_back_to_rag(config):
    config.intent_router.mode = "hybrid"
    decision = await IntentRouter(MockProvider([ProviderError("router unavailable")]), config.intent_router).route(
        "Explain an unfamiliar technical mechanism")
    assert decision.intent == "paper_question" and "ProviderError" in decision.error


@pytest.mark.asyncio
async def test_truncated_json_failure_is_logged_as_fallback(config):
    config.intent_router.mode = "llm"
    error = ProviderError("Groq returned invalid JSON: truncated document")
    decision = await IntentRouter(MockProvider([error]), config.intent_router).route("ambiguous request")
    assert decision.intent == "paper_question"
    assert decision.fallback_used and decision.schema_error


@pytest.mark.asyncio
@pytest.mark.parametrize("question", [
    "What is Inner Monologue?",
    "How does H-Net's routing module identify chunk boundaries?",
    "Explain H-Net's encoder and decoder.",
    "Why does Mimi use split residual vector quantization?",
    "How many RTX 4090 GPUs are required to serve 100 concurrent Moshi users?",
])
async def test_known_paper_topics_never_get_rejected_by_router(question, config):
    config.intent_router.mode = "hybrid"
    decision = await IntentRouter(MockProvider([]), config.intent_router).route(question)
    assert decision.intent == "paper_question"
    assert decision.method == "rules"


@pytest.mark.asyncio
@pytest.mark.parametrize("architecture", ["fast", "verified", "adaptive"])
async def test_identical_direct_routing_across_architectures(architecture, config):
    config.intent_router.mode = "hybrid"
    provider = MockProvider([])
    built = 0
    def forbidden_retriever():
        nonlocal built
        built += 1
        raise AssertionError("retriever must not be constructed")
    result = await RAGPipeline(architecture, forbidden_retriever, provider, config).run("Hey")
    assert result.status == "conversation"
    assert result.metrics["detected_intent"] == "conversation"
    assert result.metrics["routing_method"] == "rules"
    assert result.metrics["routing_model_calls"] == 0
    assert result.metrics["architecture_model_calls"] == 0
    assert result.metrics["total_model_calls"] == 0
    assert result.metrics["retrieval_ms"] is None
    assert built == 0 and provider.calls == []


@pytest.mark.asyncio
async def test_router_and_architecture_call_counts(retriever, config):
    config.intent_router.mode = "llm"
    provider = MockProvider([
        classification("paper_question", retrieval=True),
        "Multi-head attention uses several heads [S1]",
    ])
    result = await RAGPipeline("fast", retriever, provider, config).run("Explain multi-head attention")
    assert result.status == "ok"
    assert result.metrics["routing_model_calls"] == 1
    assert result.metrics["architecture_model_calls"] == 1
    assert result.metrics["total_model_calls"] == 2
    assert result.metrics["routing_tokens_in"] == 8
    assert result.metrics["total_input_tokens"] == 18


@pytest.mark.asyncio
async def test_out_of_scope_pipeline_skips_retrieval(config):
    config.intent_router.mode = "llm"
    provider = MockProvider([classification("out_of_scope", retrieval=False)])
    def forbidden_retriever():
        raise AssertionError("out-of-scope requests must not retrieve")
    result = await RAGPipeline("verified", forbidden_retriever, provider, config).run(
        "What is the capital of France?")
    assert result.status == "out_of_scope"
    assert result.metrics["routing_model_calls"] == 1
    assert result.metrics["architecture_model_calls"] == 0
    assert result.metrics["retrieval_ms"] is None

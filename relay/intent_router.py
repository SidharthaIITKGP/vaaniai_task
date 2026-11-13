from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .config import IntentRouterConfig
from .instrumentation import ms
from .providers.base import ModelProvider, ProviderError
from .schemas import ProviderResponse


Intent = Literal["conversation", "paper_question", "out_of_scope"]
PaperID = Literal["1706.03762v7", "2410.00037v1", "2507.07955v2"]

CONVERSATION_RESPONSES = {
    "greeting": "Hey! What would you like to know about the research papers?",
    "thanks": "You’re welcome! Feel free to ask about any of the three research papers.",
    "capability": (
        "I can answer questions about the three research papers: Attention Is All You Need, Moshi, and Dynamic Chunking, "
        "with traceable citations from those papers."
    ),
}
SCOPE_REFUSAL = (
    "I can only answer factual questions using the three provided papers: Attention Is All You Need, "
    "Moshi, and Dynamic Chunking."
)


class IntentClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")
    intent: Intent
    complexity: Literal["simple", "complex"]
    relevant_papers: list[PaperID] = Field(max_length=3)


INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": ["conversation", "paper_question", "out_of_scope"]},
        "complexity": {"type": "string", "enum": ["simple", "complex"]},
        "relevant_papers": {
            "type": "array",
            "items": {"type": "string", "enum": ["1706.03762v7", "2410.00037v1", "2507.07955v2"]},
        },
    },
    "required": ["intent", "complexity", "relevant_papers"],
    "additionalProperties": False,
}


@dataclass(slots=True)
class RoutingDecision:
    intent: Intent
    method: Literal["rules", "llm"]
    response: str | None = None
    complexity: str = "simple"
    relevant_papers: list[str] = field(default_factory=list)
    reason: str = ""
    latency_ms: float | None = None
    model_calls: int = 0
    provider_response: ProviderResponse | None = None
    error: str | None = None
    schema_error: str | None = None
    fallback_used: bool = False
    provider_error: ProviderError | None = None


def _normalized(text: str) -> str:
    return re.sub(r"[^a-z0-9' ]+", " ", text.lower()).strip()


def _conversation_rule(text: str) -> tuple[str, str] | None:
    value = _normalized(text)
    # Full-message matching prevents "Hey, explain multi-head attention" from bypassing RAG.
    greetings = {
        "hi", "hey", "hello", "hiya", "good morning", "good afternoon", "good evening",
        "how are you", "how are you doing",
    }
    thanks = {"thanks", "thank you", "thanks a lot", "got it", "okay thanks", "ok thanks"}
    capabilities = {
        "what can you do", "what do you do", "how can you help", "what can i ask you",
        "which papers can you answer about",
    }
    if value in greetings:
        return "greeting", CONVERSATION_RESPONSES["greeting"]
    if value in thanks:
        return "thanks", CONVERSATION_RESPONSES["thanks"]
    if value in capabilities:
        return "capability", CONVERSATION_RESPONSES["capability"]
    return None


PAPER_TERMS = re.compile(
    r"\b(transformer|attention|multi[- ]?head|self[- ]?attention|scaled dot|positional encoding|moshi|"
    r"mimi|helium|inner monologue|full[- ]?duplex|speech[- ]?text|rvq|asr|tts|dynamic chunking|h[- ]?net|"
    r"mamba[- ]?2|bpe|ratio loss|chunk boundar(?:y|ies)|smoothing module|sequence model|hierarchical|arxiv|"
    r"encoder|decoder|token|embedding|bleu|wmt)\b",
    re.IGNORECASE,
)
OUT_OF_SCOPE_PATTERNS = re.compile(
    r"\b(capital of|world cup|president of|prime minister|weather|stock price|recipe|movie|"
    r"not included|outside (?:the )?library|another paper)\b",
    re.IGNORECASE,
)


class IntentRouter:
    def __init__(self, provider: ModelProvider, config: IntentRouterConfig | None = None):
        self.provider = provider
        self.config = config or IntentRouterConfig()
        if self.config.mode not in {"rules", "llm", "hybrid"}:
            raise ValueError("intent router mode must be rules, llm, or hybrid")

    def _rules_only(self, question: str) -> RoutingDecision:
        conversational = _conversation_rule(question)
        if conversational:
            reason, response = conversational
            return RoutingDecision("conversation", "rules", response=response, reason=reason)
        if OUT_OF_SCOPE_PATTERNS.search(question):
            return RoutingDecision("out_of_scope", "rules", response=SCOPE_REFUSAL,
                                   reason="obvious general-knowledge request")
        # Conservative default: grounding, not rules, decides paper answerability.
        reason = "paper-domain term matched" if PAPER_TERMS.search(question) else "ambiguous; conservative RAG fallback"
        return RoutingDecision("paper_question", "rules", reason=reason)

    async def route(self, question: str, *, request_id: str | None = None,
                    architecture: str | None = None) -> RoutingDecision:
        started = time.perf_counter()
        if self.config.mode != "llm":
            conversational = _conversation_rule(question)
            if conversational:
                reason, response = conversational
                decision = RoutingDecision("conversation", "rules", response=response, reason=reason)
                decision.latency_ms = ms(time.perf_counter() - started)
                return decision
            if PAPER_TERMS.search(question):
                decision = RoutingDecision(
                    "paper_question", "rules", reason="explicit paper-domain concept matched")
                decision.latency_ms = ms(time.perf_counter() - started)
                return decision
            if OUT_OF_SCOPE_PATTERNS.search(question):
                decision = RoutingDecision(
                    "out_of_scope", "rules", response=SCOPE_REFUSAL,
                    reason="obvious unrelated general-knowledge request")
                decision.latency_ms = ms(time.perf_counter() - started)
                return decision
            if self.config.mode == "rules":
                decision = self._rules_only(question)
                decision.latency_ms = ms(time.perf_counter() - started)
                return decision
        try:
            data, response = await self.provider.generate_json([
                {"role": "system", "content": (
                    "Classify only. paper_question means related to Transformer, Moshi/Mimi/Helium, H-Net/Dynamic "
                    "Chunking, or their technical concepts—even if the requested fact may be absent. "
                    "out_of_scope means clearly unrelated. Never answer the question."
                )},
                {"role": "user", "content": question.strip()},
            ], max_output_tokens=self.config.max_output_tokens,
               reasoning_effort=self.config.reasoning_effort,
               json_schema=INTENT_SCHEMA, schema_name="intent_classification", strict=True,
               stage="routing", request_id=request_id, architecture=architecture)
            parsed = IntentClassification.model_validate(data)
            direct = None
            if parsed.intent == "conversation":
                matched = _conversation_rule(question)
                direct = matched[1] if matched else CONVERSATION_RESPONSES["capability"]
            elif parsed.intent == "out_of_scope":
                direct = SCOPE_REFUSAL
            decision = RoutingDecision(
                parsed.intent, "llm", response=direct, complexity=parsed.complexity,
                relevant_papers=list(parsed.relevant_papers), reason="strict structured classifier",
                model_calls=1, provider_response=response,
            )
        except ValidationError as exc:
            decision = RoutingDecision(
                "paper_question", "llm", reason="classifier failed; conservative RAG fallback",
                model_calls=1, error=f"{type(exc).__name__}: {exc}", schema_error=str(exc),
                fallback_used=True,
            )
        except (ProviderError, ValueError, TypeError) as exc:
            decision = RoutingDecision(
                "paper_question", "llm", reason="classifier failed; conservative RAG fallback",
                model_calls=1, error=f"{type(exc).__name__}: {exc}",
                schema_error=str(exc) if "JSON" in str(exc) or "schema" in str(exc).lower() else None,
                fallback_used=True,
                provider_error=exc if isinstance(exc, ProviderError) else None,
            )
        decision.latency_ms = ms(time.perf_counter() - started)
        return decision

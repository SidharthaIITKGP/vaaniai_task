from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
import pytest

from relay.config import RelayConfig, RetrievalConfig
from relay.providers.base import ModelProvider
from relay.retrieval import HybridRetriever
from relay.schemas import Chunk, ProviderResponse, Usage


class FakeEmbedder:
    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        rows = []
        for text in texts:
            vector = np.zeros(12)
            for word in text.lower().split():
                vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % len(vector)] += 1
            rows.append(vector)
        return np.asarray(rows)


class MockProvider(ModelProvider):
    def __init__(self, responses: list[Any]):
        self.responses = list(responses)
        self.calls: list[str] = []
        self.messages_seen: list[list[dict[str, str]]] = []
        self.kwargs_seen: list[dict[str, Any]] = []

    async def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> ProviderResponse:
        self.calls.append("generate")
        self.messages_seen.append(messages)
        self.kwargs_seen.append(kwargs)
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return ProviderResponse(str(value), Usage(10, 5), latency_ms=2.0, model="mock")

    async def generate_streaming(self, messages: list[dict[str, str]], **kwargs: Any) -> ProviderResponse:
        self.calls.append("generate_streaming")
        self.messages_seen.append(messages)
        self.kwargs_seen.append(kwargs)
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return ProviderResponse(str(value), Usage(10, 5), latency_ms=2.0, ttft_ms=0.5, model="mock")

    async def stream(self, messages: list[dict[str, str]], **kwargs: Any) -> AsyncIterator[str]:
        self.calls.append("stream")
        self.messages_seen.append(messages)
        self.kwargs_seen.append(kwargs)
        yield str(self.responses.pop(0))

    async def generate_json(self, messages: list[dict[str, str]], **kwargs: Any):
        self.calls.append("json")
        self.messages_seen.append(messages)
        self.kwargs_seen.append(kwargs)
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value, ProviderResponse("{}", Usage(8, 4), latency_ms=3.0, model="mock")


@pytest.fixture
def chunks():
    return [
        Chunk("1706.03762-p001-aaaaaaaaaaaaaaaa", "1706.03762", "Attention Is All You Need", "v7", "a.pdf", 1, "Introduction", "multi head attention transformer uses attention heads"),
        Chunk("2410.00037-p002-bbbbbbbbbbbbbbbb", "2410.00037", "Moshi", "v1", "m.pdf", 2, "Model", "Moshi supports real time spoken dialogue speech model"),
        Chunk("2507.07955-p003-cccccccccccccccc", "2507.07955", "Dynamic Chunking", "v2", "d.pdf", 3, "Method", "dynamic chunking hierarchical sequence boundaries chunks"),
        Chunk("1706.03762-p004-dddddddddddddddd", "1706.03762", "Attention Is All You Need", "v7", "a.pdf", 4, "Method", "scaled dot product query key value attention"),
    ]


@pytest.fixture
def retriever(chunks):
    return HybridRetriever(chunks, RetrievalConfig(top_k=3, candidate_k=4), FakeEmbedder())


@pytest.fixture
def config(tmp_path):
    value = RelayConfig(root=tmp_path)
    # Existing architecture unit tests isolate architecture call counts.
    value.intent_router.mode = "rules"
    return value

import json

import httpx
import pytest

from relay.config import ProviderConfig
from relay.providers.base import RateLimitError
from relay.providers.groq_provider import GroqProvider


def test_groq_loads_key_from_explicit_env_file(monkeypatch, tmp_path):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("GROQ_API_KEY=file-test-key\n", encoding="utf-8")
    provider = GroqProvider(env_path=env_file)
    assert provider.api_key == "file-test-key"


@pytest.mark.asyncio
async def test_groq_rate_limit(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-only")
    async def handler(request):
        return httpx.Response(429, headers={"retry-after": "3"}, request=request)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = GroqProvider(ProviderConfig(), client=client)
    with pytest.raises(RateLimitError) as caught:
        await provider.generate([{"role": "user", "content": "test"}])
    assert caught.value.retry_after == "3"
    assert caught.value.rate_limit_type == "provider_429"
    await client.aclose()


@pytest.mark.asyncio
async def test_groq_usage_and_latency(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-only")
    async def handler(request):
        return httpx.Response(200, json={"model": "mock", "choices": [{"message": {"content": "ok"}}],
                                         "usage": {"prompt_tokens": 4, "completion_tokens": 2}}, request=request)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    response = await GroqProvider(client=client).generate([{"role": "user", "content": "test"}])
    assert response.usage.input_tokens == 4 and response.latency_ms is not None
    await client.aclose()


@pytest.mark.asyncio
async def test_buffered_stream_records_ttft_and_usage(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-only")
    body = "\n\n".join((
        'data: {"model":"mock","choices":[{"delta":{"content":"hel"}}]}',
        'data: {"choices":[{"delta":{"content":"lo"}}]}',
        'data: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":2}}',
        "data: [DONE]",
    ))
    async def handler(request):
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"}, request=request)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    response = await GroqProvider(client=client).generate_streaming([{"role": "user", "content": "test"}])
    assert response.text == "hello"
    assert response.ttft_ms is not None
    assert response.usage.input_tokens == 7 and response.usage.output_tokens == 2
    await client.aclose()


@pytest.mark.asyncio
async def test_strict_json_schema_payload(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-only")
    captured = {}
    async def handler(request):
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={
            "model": "mock", "choices": [{"message": {"content": '{"intent":"paper_question"}'}}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 3},
        }, request=request)
    schema = {"type": "object", "properties": {"intent": {"type": "string"}},
              "required": ["intent"], "additionalProperties": False}
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    data, _ = await GroqProvider(client=client).generate_json(
        [{"role": "user", "content": "test"}], json_schema=schema,
        schema_name="intent", strict=True, max_output_tokens=300)
    assert data["intent"] == "paper_question"
    response_format = captured["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"]["additionalProperties"] is False
    assert captured["stream"] is False and captured["max_completion_tokens"] == 300
    await client.aclose()

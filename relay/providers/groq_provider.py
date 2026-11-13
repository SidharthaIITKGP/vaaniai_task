from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from ..config import ProviderConfig
from ..schemas import ProviderResponse, Usage
from .base import ModelProvider, ProviderError, RateLimitError
from .scheduler import RequestScheduler, Reservation, estimate_tokens


class GroqProvider(ModelProvider):
    endpoint = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, config: ProviderConfig | None = None, *, env_path: Path | None = None,
                 client: httpx.AsyncClient | None = None):
        # An explicit path makes CLI behavior independent of the launch directory.
        # Library users retain the conventional current-working-directory fallback.
        load_dotenv(env_path or Path.cwd() / ".env", override=False)
        self.config = config or ProviderConfig()
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ProviderError("GROQ_API_KEY is not set. Export it or add it to .env.")
        self._client = client
        self.scheduler = RequestScheduler(self.config.scheduler) if self.config.scheduler.enabled else None

    def _payload(self, messages: list[dict[str, str]], *, stream: bool,
                 json_mode: bool = False, **overrides: Any) -> dict[str, Any]:
        json_schema = overrides.pop("json_schema", None)
        schema_name = overrides.pop("schema_name", "structured_response")
        strict = overrides.pop("strict", True)
        payload = {
            "model": overrides.pop("model", self.config.model), "messages": messages,
            "stream": stream,
            "reasoning_effort": overrides.pop("reasoning_effort", self.config.reasoning_effort),
            "reasoning_format": "hidden",
            "max_completion_tokens": overrides.pop("max_output_tokens", self.config.max_output_tokens),
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}
        if json_mode:
            payload["response_format"] = (
                {"type": "json_schema", "json_schema": {
                    "name": schema_name, "strict": strict, "schema": json_schema,
                }} if json_schema else {"type": "json_object"}
            )
        payload.update(overrides)
        return payload

    async def _reserve(self, payload: dict[str, Any]) -> Reservation | None:
        if self.scheduler is None:
            return None
        return await self.scheduler.acquire(estimate_tokens(
            payload.get("messages", []), int(payload.get("max_completion_tokens", 0))))

    @staticmethod
    def _rate_headers(response: httpx.Response) -> dict[str, str]:
        names = (
            "retry-after", "x-ratelimit-limit-requests", "x-ratelimit-remaining-requests",
            "x-ratelimit-reset-requests", "x-ratelimit-limit-tokens",
            "x-ratelimit-remaining-tokens", "x-ratelimit-reset-tokens",
        )
        return {name: response.headers[name] for name in names if name in response.headers}

    @staticmethod
    def _retry_seconds(value: str | None) -> float | None:
        try:
            return float(value) if value is not None else None
        except ValueError:
            return None

    async def _observe_response(self, response: httpx.Response) -> dict[str, str]:
        headers = self._rate_headers(response)
        if self.scheduler:
            await self.scheduler.update_from_headers(headers)
            if response.status_code == 429:
                await self.scheduler.note_429(self._retry_seconds(headers.get("retry-after")))
        return headers

    async def _post(self, payload: dict[str, Any], reservation: Reservation | None) -> tuple[httpx.Response, dict[str, str]]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        if self._client:
            response = await self._client.post(self.endpoint, headers=headers, json=payload)
        else:
            async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
                response = await client.post(self.endpoint, headers=headers, json=payload)
        rate_headers = await self._observe_response(response)
        if response.status_code == 429:
            retry = rate_headers.get("retry-after")
            raise RateLimitError("Groq rate limit exceeded", status_code=429,
                                 retry_after=retry, rate_limit_type=_classify_429(response.text),
                                 queue_wait_ms=reservation.queue_wait_ms if reservation else None)
        if response.status_code >= 400:
            raise ProviderError(f"Groq HTTP {response.status_code}: {response.text[:500]}",
                                status_code=response.status_code,
                                queue_wait_ms=reservation.queue_wait_ms if reservation else None)
        return response, rate_headers

    async def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> ProviderResponse:
        payload = self._payload(messages, stream=False, **kwargs)
        reservation = await self._reserve(payload)
        started = time.perf_counter()
        try:
            response, rate_headers = await self._post(payload, reservation)
        except httpx.TimeoutException as exc:
            raise ProviderError(f"Groq request timed out: {exc}") from exc
        try:
            data = response.json()
            text = data["choices"][0]["message"].get("content", "")
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"Malformed Groq response: {exc}") from exc
        usage = data.get("usage") or {}
        if self.scheduler and reservation:
            await self.scheduler.reconcile(reservation, _usage_total(usage))
        return ProviderResponse(
            text=text,
            usage=Usage(usage.get("prompt_tokens"), usage.get("completion_tokens")),
            latency_ms=(time.perf_counter() - started) * 1000,
            model=data.get("model", self.config.model),
            queue_wait_ms=reservation.queue_wait_ms if reservation else None,
            provider_status_code=response.status_code, rate_limit_headers=rate_headers, raw=data,
        )

    async def stream(self, messages: list[dict[str, str]], **kwargs: Any) -> AsyncIterator[str]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        client = self._client or httpx.AsyncClient(timeout=self.config.timeout_seconds)
        owns = self._client is None
        payload = self._payload(messages, stream=True, **kwargs)
        reservation = await self._reserve(payload)
        usage: dict[str, Any] = {}
        try:
            async with client.stream("POST", self.endpoint, headers=headers,
                                     json=payload) as response:
                rate_headers = await self._observe_response(response)
                if response.status_code == 429:
                    body = (await response.aread()).decode(errors="replace")
                    raise RateLimitError("Groq rate limit exceeded", status_code=429,
                                         retry_after=rate_headers.get("retry-after"),
                                         rate_limit_type=_classify_429(body),
                                         queue_wait_ms=reservation.queue_wait_ms if reservation else None)
                if response.status_code >= 400:
                    body = (await response.aread()).decode(errors="replace")
                    raise ProviderError(f"Groq HTTP {response.status_code}: {body[:500]}",
                                        status_code=response.status_code)
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if not raw or raw == "[DONE]":
                        continue
                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        raise ProviderError(f"Malformed Groq stream event: {exc}") from exc
                    usage.update(event.get("usage") or {})
                    for choice in event.get("choices", []):
                        content = (choice.get("delta") or {}).get("content")
                        if content:
                            yield content
        except httpx.TimeoutException as exc:
            raise ProviderError(f"Groq stream timed out: {exc}") from exc
        finally:
            if self.scheduler and reservation:
                await self.scheduler.reconcile(reservation, _usage_total(usage))
            if owns:
                await client.aclose()

    async def generate_streaming(self, messages: list[dict[str, str]], **kwargs: Any) -> ProviderResponse:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        client = self._client or httpx.AsyncClient(timeout=self.config.timeout_seconds)
        owns = self._client is None
        first_token: float | None = None
        chunks: list[str] = []
        usage: dict[str, Any] = {}
        model: str | None = self.config.model
        payload = self._payload(messages, stream=True, **kwargs)
        reservation = await self._reserve(payload)
        started = time.perf_counter()
        rate_headers: dict[str, str] = {}
        try:
            async with client.stream("POST", self.endpoint, headers=headers,
                                     json=payload) as response:
                rate_headers = await self._observe_response(response)
                if response.status_code == 429:
                    body = (await response.aread()).decode(errors="replace")
                    raise RateLimitError("Groq rate limit exceeded", status_code=429,
                                         retry_after=rate_headers.get("retry-after"),
                                         rate_limit_type=_classify_429(body),
                                         queue_wait_ms=reservation.queue_wait_ms if reservation else None)
                if response.status_code >= 400:
                    body = (await response.aread()).decode(errors="replace")
                    raise ProviderError(f"Groq HTTP {response.status_code}: {body[:500]}",
                                        status_code=response.status_code)
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if not raw or raw == "[DONE]":
                        continue
                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        raise ProviderError(f"Malformed Groq stream event: {exc}") from exc
                    model = event.get("model", model)
                    usage.update(event.get("usage") or {})
                    for choice in event.get("choices", []):
                        content = (choice.get("delta") or {}).get("content")
                        if content:
                            if first_token is None:
                                first_token = time.perf_counter()
                            chunks.append(content)
        except httpx.TimeoutException as exc:
            raise ProviderError(f"Groq stream timed out: {exc}") from exc
        finally:
            if self.scheduler and reservation:
                await self.scheduler.reconcile(reservation, _usage_total(usage))
            if owns:
                await client.aclose()
        return ProviderResponse(
            text="".join(chunks),
            usage=Usage(usage.get("prompt_tokens"), usage.get("completion_tokens")),
            latency_ms=(time.perf_counter() - started) * 1000,
            ttft_ms=(first_token - started) * 1000 if first_token is not None else None,
            model=model, queue_wait_ms=reservation.queue_wait_ms if reservation else None,
            provider_status_code=200, rate_limit_headers=rate_headers,
        )

    async def generate_json(self, messages: list[dict[str, str]], **kwargs: Any) -> tuple[dict[str, Any], ProviderResponse]:
        payload = self._payload(messages, stream=False, json_mode=True, **kwargs)
        reservation = await self._reserve(payload)
        started = time.perf_counter()
        response, rate_headers = await self._post(payload, reservation)
        try:
            data = response.json()
            text = data["choices"][0]["message"].get("content", "")
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"Malformed Groq response: {exc}") from exc
        usage = data.get("usage") or {}
        if self.scheduler and reservation:
            await self.scheduler.reconcile(reservation, _usage_total(usage))
        result = ProviderResponse(text=text, usage=Usage(usage.get("prompt_tokens"), usage.get("completion_tokens")),
                                  latency_ms=(time.perf_counter() - started) * 1000,
                                  model=data.get("model", self.config.model),
                                  queue_wait_ms=reservation.queue_wait_ms if reservation else None,
                                  provider_status_code=response.status_code,
                                  rate_limit_headers=rate_headers, raw=data)
        try:
            return json.loads(text), result
        except json.JSONDecodeError as exc:
            raise ProviderError(f"Groq returned invalid JSON: {exc}") from exc


def _usage_total(usage: dict[str, Any]) -> int | None:
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if prompt is None and completion is None:
        return None
    return int(prompt or 0) + int(completion or 0)


def _classify_429(body: str) -> str:
    lower = body.lower()
    if "tokens per minute" in lower or "tpm" in lower:
        return "provider_tpm"
    if "requests per minute" in lower or "rpm" in lower:
        return "provider_rpm"
    if "requests per day" in lower or "rpd" in lower or "daily" in lower:
        return "provider_rpd"
    return "provider_429"

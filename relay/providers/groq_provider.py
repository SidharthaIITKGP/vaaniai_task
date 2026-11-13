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
from .base import ModelProvider, ProviderError, RateLimitError, TruncatedResponseError
from .scheduler import RequestScheduler, Reservation, estimate_input_tokens


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
        deadline = payload.pop("_queue_deadline_seconds", None)
        if self.scheduler is None:
            return None
        return await self.scheduler.acquire(
            estimate_input_tokens(payload.get("messages", [])),
            int(payload.get("max_completion_tokens", 0)),
            deadline_seconds=deadline,
        )

    def _context(self, overrides: dict[str, Any]) -> dict[str, Any]:
        stage = str(overrides.pop("stage", "unknown"))
        context = {
            "stage": stage,
            "pipeline_request_id": overrides.pop("request_id", None),
            "architecture": overrides.pop("architecture", None),
            "queue_deadline_seconds": overrides.pop("queue_deadline_seconds", None),
        }
        if "max_output_tokens" not in overrides and stage in self.config.stage_output_tokens:
            overrides["max_output_tokens"] = self.config.stage_output_tokens[stage]
        return context

    @staticmethod
    def _reservation_metadata(reservation: Reservation | None, context: dict[str, Any]) -> dict[str, Any]:
        return {
            "queue_wait_ms": reservation.queue_wait_ms if reservation else None,
            "estimated_input_tokens": reservation.estimated_input_tokens if reservation else None,
            "reserved_output_tokens": reservation.reserved_output_tokens if reservation else None,
            "reserved_total_tokens": reservation.reserved_tokens if reservation else None,
            "local_rpm_remaining": reservation.local_rpm_remaining if reservation else None,
            "local_tpm_remaining": reservation.local_tpm_remaining if reservation else None,
            "scheduler_wait_reason": reservation.wait_reason if reservation else None,
            "pipeline_request_id": context.get("pipeline_request_id"),
            "architecture": context.get("architecture"),
            "stage": context.get("stage"),
        }

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
            if self.scheduler:
                await self.scheduler.refund_rejected(reservation)
            retry = rate_headers.get("retry-after")
            raise RateLimitError("Groq rate limit exceeded", status_code=429,
                                 retry_after=retry, rate_limit_type=_classify_429(response.text),
                                 queue_wait_ms=reservation.queue_wait_ms if reservation else None)
        if response.status_code >= 400:
            if self.scheduler and response.status_code < 500:
                await self.scheduler.refund_rejected(reservation)
            raise ProviderError(f"Groq HTTP {response.status_code}: {response.text[:500]}",
                                status_code=response.status_code,
                                queue_wait_ms=reservation.queue_wait_ms if reservation else None)
        return response, rate_headers

    async def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> ProviderResponse:
        context = self._context(kwargs)
        payload = self._payload(messages, stream=False, **kwargs)
        payload["_queue_deadline_seconds"] = context["queue_deadline_seconds"]
        reservation = await self._reserve(payload)
        started = time.perf_counter()
        try:
            response, rate_headers = await self._post(payload, reservation)
        except httpx.TimeoutException as exc:
            raise ProviderError(
                f"Groq request timed out with uncertain token consumption: {exc}",
                queue_wait_ms=reservation.queue_wait_ms if reservation else None) from exc
        try:
            data = response.json()
            text = data["choices"][0]["message"].get("content", "")
            finish_reason = data["choices"][0].get("finish_reason")
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"Malformed Groq response: {exc}") from exc
        usage = data.get("usage") or {}
        if self.scheduler and reservation:
            await self.scheduler.reconcile(reservation, _usage_total(usage))
        if finish_reason == "length":
            raise TruncatedResponseError(
                "Groq response was truncated at max_completion_tokens",
                status_code=response.status_code,
                queue_wait_ms=reservation.queue_wait_ms if reservation else None)
        result = ProviderResponse(
            text=text,
            usage=Usage(usage.get("prompt_tokens"), usage.get("completion_tokens")),
            latency_ms=(time.perf_counter() - started) * 1000,
            model=data.get("model", self.config.model),
            provider_status_code=response.status_code, rate_limit_headers=rate_headers, raw=data,
            **self._reservation_metadata(reservation, context),
        )
        return result

    async def stream(self, messages: list[dict[str, str]], **kwargs: Any) -> AsyncIterator[str]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        client = self._client or httpx.AsyncClient(timeout=self.config.timeout_seconds)
        owns = self._client is None
        context = self._context(kwargs)
        payload = self._payload(messages, stream=True, **kwargs)
        payload["_queue_deadline_seconds"] = context["queue_deadline_seconds"]
        reservation = await self._reserve(payload)
        usage: dict[str, Any] = {}
        try:
            async with client.stream("POST", self.endpoint, headers=headers,
                                     json=payload) as response:
                rate_headers = await self._observe_response(response)
                if response.status_code == 429:
                    if self.scheduler:
                        await self.scheduler.refund_rejected(reservation)
                    body = (await response.aread()).decode(errors="replace")
                    raise RateLimitError("Groq rate limit exceeded", status_code=429,
                                         retry_after=rate_headers.get("retry-after"),
                                         rate_limit_type=_classify_429(body),
                                         queue_wait_ms=reservation.queue_wait_ms if reservation else None)
                if response.status_code >= 400:
                    if self.scheduler and response.status_code < 500:
                        await self.scheduler.refund_rejected(reservation)
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
        finish_reason: str | None = None
        context = self._context(kwargs)
        payload = self._payload(messages, stream=True, **kwargs)
        payload["_queue_deadline_seconds"] = context["queue_deadline_seconds"]
        reservation = await self._reserve(payload)
        started = time.perf_counter()
        rate_headers: dict[str, str] = {}
        try:
            async with client.stream("POST", self.endpoint, headers=headers,
                                     json=payload) as response:
                rate_headers = await self._observe_response(response)
                if response.status_code == 429:
                    if self.scheduler:
                        await self.scheduler.refund_rejected(reservation)
                    body = (await response.aread()).decode(errors="replace")
                    raise RateLimitError("Groq rate limit exceeded", status_code=429,
                                         retry_after=rate_headers.get("retry-after"),
                                         rate_limit_type=_classify_429(body),
                                         queue_wait_ms=reservation.queue_wait_ms if reservation else None)
                if response.status_code >= 400:
                    if self.scheduler and response.status_code < 500:
                        await self.scheduler.refund_rejected(reservation)
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
                        finish_reason = choice.get("finish_reason") or finish_reason
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
        result = ProviderResponse(
            text="".join(chunks),
            usage=Usage(usage.get("prompt_tokens"), usage.get("completion_tokens")),
            latency_ms=(time.perf_counter() - started) * 1000,
            ttft_ms=(first_token - started) * 1000 if first_token is not None else None,
            model=model,
            provider_status_code=200, rate_limit_headers=rate_headers,
            **self._reservation_metadata(reservation, context),
        )
        if finish_reason == "length":
            raise TruncatedResponseError(
                "Groq response was truncated at max_completion_tokens",
                status_code=200, queue_wait_ms=reservation.queue_wait_ms if reservation else None)
        return result

    async def generate_json(self, messages: list[dict[str, str]], **kwargs: Any) -> tuple[dict[str, Any], ProviderResponse]:
        context = self._context(kwargs)
        payload = self._payload(messages, stream=False, json_mode=True, **kwargs)
        payload["_queue_deadline_seconds"] = context["queue_deadline_seconds"]
        reservation = await self._reserve(payload)
        started = time.perf_counter()
        response, rate_headers = await self._post(payload, reservation)
        try:
            data = response.json()
            text = data["choices"][0]["message"].get("content", "")
            finish_reason = data["choices"][0].get("finish_reason")
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"Malformed Groq response: {exc}") from exc
        usage = data.get("usage") or {}
        if self.scheduler and reservation:
            await self.scheduler.reconcile(reservation, _usage_total(usage))
        result = ProviderResponse(text=text, usage=Usage(usage.get("prompt_tokens"), usage.get("completion_tokens")),
                                  latency_ms=(time.perf_counter() - started) * 1000,
                                  model=data.get("model", self.config.model),
                                  provider_status_code=response.status_code,
                                  rate_limit_headers=rate_headers, raw=data,
                                  **self._reservation_metadata(reservation, context))
        if finish_reason == "length":
            raise TruncatedResponseError(
                "Groq structured response was truncated at max_completion_tokens",
                status_code=response.status_code,
                queue_wait_ms=reservation.queue_wait_ms if reservation else None)
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

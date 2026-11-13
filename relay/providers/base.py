from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from ..schemas import ProviderResponse


class ProviderError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None,
                 retry_after: str | None = None, rate_limit_type: str | None = None,
                 queue_wait_ms: float | None = None,
                 response_headers: dict[str, str] | None = None,
                 error_body: str | None = None, diagnostics: dict[str, Any] | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after
        self.rate_limit_type = rate_limit_type
        self.queue_wait_ms = queue_wait_ms
        self.response_headers = response_headers or {}
        self.error_body = error_body
        self.diagnostics = diagnostics or {}

    def details(self) -> dict[str, Any]:
        return {'type': type(self).__name__, 'message': str(self),
                'status_code': self.status_code, 'retry_after': self.retry_after,
                'retry_after_seconds': retry_seconds(self.retry_after),
                'rate_limit_type': self.rate_limit_type, 'queue_wait_ms': self.queue_wait_ms,
                'response_headers': self.response_headers, 'error_body': self.error_body,
                'diagnostics': self.diagnostics}


class RateLimitError(ProviderError):
    pass


class TruncatedResponseError(ProviderError):
    pass


def retry_seconds(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            return max(0.0, (date - datetime.now(timezone.utc)).total_seconds())
        except (ValueError, TypeError, OverflowError):
            return None


class ModelProvider(ABC):
    @abstractmethod
    async def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> ProviderResponse: ...

    @abstractmethod
    async def generate_streaming(self, messages: list[dict[str, str]], **kwargs: Any) -> ProviderResponse:
        """Buffer a streamed response and return timing/usage without exposing a draft."""
        ...

    @abstractmethod
    async def stream(self, messages: list[dict[str, str]], **kwargs: Any) -> AsyncIterator[str]: ...

    @abstractmethod
    async def generate_json(self, messages: list[dict[str, str]], **kwargs: Any) -> tuple[dict[str, Any], ProviderResponse]: ...

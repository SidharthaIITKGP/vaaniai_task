from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from ..schemas import ProviderResponse


class ProviderError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None,
                 retry_after: str | None = None, rate_limit_type: str | None = None,
                 queue_wait_ms: float | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after
        self.rate_limit_type = rate_limit_type
        self.queue_wait_ms = queue_wait_ms


class RateLimitError(ProviderError):
    pass


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

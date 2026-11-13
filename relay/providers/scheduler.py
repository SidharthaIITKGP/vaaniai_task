from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from dataclasses import dataclass

from ..config import SchedulerConfig
from .base import RateLimitError


@dataclass(slots=True)
class Reservation:
    reservation_id: str
    timestamp: float
    reserved_tokens: int
    queue_wait_ms: float


class RequestScheduler:
    """Shared sliding-window RPM/TPM admission control for every provider call."""

    def __init__(self, config: SchedulerConfig):
        self.config = config
        self._events: deque[Reservation] = deque()
        self._lock = asyncio.Lock()
        self._pending = 0
        self._pending_lock = asyncio.Lock()
        self._blocked_until = 0.0
        self.remaining_daily_requests: int | None = None
        self.remaining_provider_tokens: int | None = None
        self._provider_token_reset_at = 0.0

    @property
    def request_budget(self) -> int:
        return max(1, int(self.config.rpm_limit * self.config.safety_margin))

    @property
    def token_budget(self) -> int:
        return max(1, int(self.config.tpm_limit * self.config.safety_margin))

    def _prune(self, now: float) -> None:
        while self._events and now - self._events[0].timestamp >= 60.0:
            self._events.popleft()

    async def acquire(self, estimated_tokens: int) -> Reservation:
        if estimated_tokens > self.token_budget:
            raise RateLimitError(
                f"Estimated request size {estimated_tokens} exceeds scheduler TPM budget {self.token_budget}",
                rate_limit_type="local_tpm_request_too_large", queue_wait_ms=0.0,
            )
        queued = time.perf_counter()
        async with self._pending_lock:
            if self._pending >= self.config.max_queue_size:
                raise RateLimitError("Scheduler queue is full", rate_limit_type="local_queue_full",
                                     queue_wait_ms=(time.perf_counter() - queued) * 1000)
            self._pending += 1
        try:
            async with asyncio.timeout(self.config.request_deadline_seconds):
                while True:
                    async with self._lock:
                        now = time.monotonic()
                        self._prune(now)
                        if self.remaining_daily_requests is not None and self.remaining_daily_requests <= 0:
                            raise RateLimitError("Groq daily request quota is exhausted",
                                                 rate_limit_type="rpd_exhausted",
                                                 queue_wait_ms=(time.perf_counter() - queued) * 1000)
                        used_tokens = sum(event.reserved_tokens for event in self._events)
                        rpm_ok = len(self._events) < self.request_budget
                        tpm_ok = used_tokens + estimated_tokens <= self.token_budget
                        block_wait = max(0.0, self._blocked_until - now)
                        provider_wait = 0.0
                        if (self.remaining_provider_tokens is not None
                                and self.remaining_provider_tokens < estimated_tokens
                                and self._provider_token_reset_at > now):
                            provider_wait = self._provider_token_reset_at - now
                        if self._provider_token_reset_at and now >= self._provider_token_reset_at:
                            self.remaining_provider_tokens = None
                            self._provider_token_reset_at = 0.0
                        if rpm_ok and tpm_ok and block_wait <= 0 and provider_wait <= 0:
                            reservation = Reservation(
                                str(uuid.uuid4()), now, estimated_tokens,
                                (time.perf_counter() - queued) * 1000,
                            )
                            self._events.append(reservation)
                            return reservation
                        waits = [block_wait] if block_wait > 0 else []
                        if provider_wait > 0:
                            waits.append(provider_wait)
                        if self._events and (not rpm_ok or not tpm_ok):
                            waits.append(max(0.05, 60.0 - (now - self._events[0].timestamp)))
                        wait_seconds = min(waits) if waits else 0.1
                    await asyncio.sleep(wait_seconds)
        except TimeoutError as exc:
            raise RateLimitError("Scheduler request deadline exceeded",
                                 rate_limit_type="local_queue_deadline",
                                 queue_wait_ms=(time.perf_counter() - queued) * 1000) from exc
        finally:
            async with self._pending_lock:
                self._pending -= 1

    async def reconcile(self, reservation: Reservation, actual_tokens: int | None) -> None:
        if actual_tokens is None:
            return
        async with self._lock:
            for event in self._events:
                if event.reservation_id == reservation.reservation_id:
                    event.reserved_tokens = max(0, actual_tokens)
                    break

    async def update_from_headers(self, headers: dict[str, str]) -> None:
        remaining = headers.get("x-ratelimit-remaining-requests")
        if remaining is not None:
            try:
                self.remaining_daily_requests = int(remaining)
            except ValueError:
                pass
        tokens = headers.get("x-ratelimit-remaining-tokens")
        if tokens is not None:
            try:
                self.remaining_provider_tokens = int(tokens)
            except ValueError:
                pass
        reset = _duration_seconds(headers.get("x-ratelimit-reset-tokens"))
        if reset is not None:
            self._provider_token_reset_at = time.monotonic() + reset

    async def note_429(self, retry_after_seconds: float | None) -> None:
        if retry_after_seconds is None:
            return
        async with self._lock:
            self._blocked_until = max(self._blocked_until, time.monotonic() + retry_after_seconds)


def estimate_tokens(messages: list[dict[str, str]], output_capacity: int) -> int:
    # Common dependency-free approximation for English/technical prompts; actual usage is reconciled.
    input_estimate = sum(max(1, (len(message.get("content", "")) + 3) // 4) for message in messages)
    return input_estimate + max(0, output_capacity)


def _duration_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        if value.endswith("ms"):
            return float(value[:-2]) / 1000
        if value.endswith("s") and "m" not in value:
            return float(value[:-1])
        if "m" in value and value.endswith("s"):
            minutes, seconds = value[:-1].split("m", 1)
            return float(minutes) * 60 + float(seconds or 0)
        return float(value)
    except ValueError:
        return None

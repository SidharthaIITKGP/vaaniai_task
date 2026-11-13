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
    estimated_input_tokens: int
    reserved_output_tokens: int
    reserved_tokens: int
    queue_wait_ms: float
    local_rpm_remaining: int
    local_tpm_remaining: float
    wait_reason: str | None = None
    reconciled: bool = False


class RequestScheduler:
    """Shared RPM window plus continuously-refilling TPM token bucket."""

    def __init__(self, config: SchedulerConfig):
        self.config = config
        self._requests: deque[float] = deque()
        self._condition = asyncio.Condition()
        self._pending = 0
        self._pending_lock = asyncio.Lock()
        self._waiters: deque[str] = deque()
        self._blocked_until = 0.0
        self._available_tokens = float(self.token_budget)
        self._last_refill = time.monotonic()
        self.remaining_daily_requests: int | None = None
        self.remaining_provider_tokens: int | None = None
        self._provider_token_reset_at = 0.0

    @property
    def request_budget(self) -> int:
        return max(1, int(self.config.rpm_limit * self.config.safety_margin))

    @property
    def token_budget(self) -> int:
        return max(1, int(self.config.tpm_limit * self.config.safety_margin))

    @property
    def refill_tokens_per_second(self) -> float:
        return self.token_budget / 60.0

    def _refresh(self, now: float) -> None:
        elapsed = max(0.0, now - self._last_refill)
        self._available_tokens = min(
            float(self.token_budget),
            self._available_tokens + elapsed * self.refill_tokens_per_second,
        )
        self._last_refill = now
        while self._requests and now - self._requests[0] >= 60.0:
            self._requests.popleft()

    def snapshot(self) -> dict[str, float | int | None]:
        now = time.monotonic()
        self._refresh(now)
        return {
            "local_rpm_remaining": max(0, self.request_budget - len(self._requests)),
            "local_tpm_remaining": max(0.0, self._available_tokens),
            "pending": self._pending,
            "provider_daily_remaining": self.remaining_daily_requests,
            "provider_tpm_remaining": self.remaining_provider_tokens,
        }

    async def acquire(self, estimated_input_tokens: int, reserved_output_tokens: int,
                      *, deadline_seconds: float | None = None) -> Reservation:
        estimated_tokens = estimated_input_tokens + reserved_output_tokens
        if estimated_tokens > self.token_budget:
            raise RateLimitError(
                f"Estimated request size {estimated_tokens} exceeds scheduler TPM budget {self.token_budget}",
                rate_limit_type="local_tpm_request_too_large", queue_wait_ms=0.0,
            )
        queued = time.perf_counter()
        ticket = str(uuid.uuid4())
        deadline = deadline_seconds or self.config.request_deadline_seconds
        encountered: set[str] = set()
        async with self._pending_lock:
            if self._pending >= self.config.max_queue_size:
                raise RateLimitError("Scheduler queue is full", rate_limit_type="local_queue_full",
                                     queue_wait_ms=(time.perf_counter() - queued) * 1000)
            self._pending += 1
        async with self._condition:
            self._waiters.append(ticket)
        try:
            async with asyncio.timeout(deadline):
                while True:
                    async with self._condition:
                        now = time.monotonic()
                        self._refresh(now)
                        is_head = bool(self._waiters) and self._waiters[0] == ticket
                        if self.remaining_daily_requests is not None and self.remaining_daily_requests <= 0:
                            raise RateLimitError("Groq daily request quota is exhausted",
                                                 rate_limit_type="rpd_exhausted",
                                                 queue_wait_ms=(time.perf_counter() - queued) * 1000)
                        rpm_ok = len(self._requests) < self.request_budget
                        tpm_ok = self._available_tokens >= estimated_tokens
                        block_wait = max(0.0, self._blocked_until - now)
                        provider_wait = 0.0
                        if (self.remaining_provider_tokens is not None
                                and self.remaining_provider_tokens < estimated_tokens
                                and self._provider_token_reset_at > now):
                            provider_wait = self._provider_token_reset_at - now
                        if self._provider_token_reset_at and now >= self._provider_token_reset_at:
                            self.remaining_provider_tokens = None
                            self._provider_token_reset_at = 0.0
                        if is_head and rpm_ok and tpm_ok and block_wait <= 0 and provider_wait <= 0:
                            self._available_tokens -= estimated_tokens
                            self._requests.append(now)
                            self._waiters.popleft()
                            self._condition.notify_all()
                            reservation = Reservation(
                                str(uuid.uuid4()), now, estimated_input_tokens,
                                reserved_output_tokens, estimated_tokens,
                                (time.perf_counter() - queued) * 1000,
                                max(0, self.request_budget - len(self._requests)),
                                max(0.0, self._available_tokens),
                                ",".join(sorted(encountered)) or None,
                            )
                            return reservation
                        waits: list[float] = []
                        if not is_head:
                            encountered.add("fair_queue")
                            waits.append(0.5)
                        if not rpm_ok and self._requests:
                            encountered.add("rpm")
                            waits.append(max(0.001, 60.0 - (now - self._requests[0])))
                        if not tpm_ok:
                            encountered.add("tpm")
                            deficit = estimated_tokens - self._available_tokens
                            waits.append(max(0.001, deficit / self.refill_tokens_per_second))
                        if block_wait > 0:
                            encountered.add("provider_429")
                            waits.append(block_wait)
                        if provider_wait > 0:
                            encountered.add("provider_tpm")
                            waits.append(provider_wait)
                        wait_seconds = max(waits) if waits else 0.05
                        try:
                            await asyncio.wait_for(self._condition.wait(), timeout=wait_seconds)
                        except TimeoutError:
                            pass
        except TimeoutError as exc:
            raise RateLimitError("Scheduler request deadline exceeded",
                                 rate_limit_type="local_queue_deadline",
                                 queue_wait_ms=(time.perf_counter() - queued) * 1000) from exc
        finally:
            async with self._condition:
                if ticket in self._waiters:
                    self._waiters.remove(ticket)
                    self._condition.notify_all()
            async with self._pending_lock:
                self._pending -= 1

    async def reconcile(self, reservation: Reservation, actual_tokens: int | None) -> None:
        if actual_tokens is None or reservation.reconciled:
            return
        async with self._condition:
            now = time.monotonic()
            self._refresh(now)
            difference = reservation.reserved_tokens - max(0, actual_tokens)
            self._available_tokens = min(float(self.token_budget), self._available_tokens + difference)
            reservation.reconciled = True
            self._condition.notify_all()

    async def refund_rejected(self, reservation: Reservation | None) -> None:
        """Refund tokens only when the provider definitively rejected before inference."""
        if reservation is None or reservation.reconciled:
            return
        async with self._condition:
            self._refresh(time.monotonic())
            self._available_tokens = min(
                float(self.token_budget), self._available_tokens + reservation.reserved_tokens)
            reservation.reconciled = True
            self._condition.notify_all()

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
        async with self._condition:
            self._condition.notify_all()

    async def note_429(self, retry_after_seconds: float | None) -> None:
        if retry_after_seconds is None:
            return
        async with self._condition:
            self._blocked_until = max(self._blocked_until, time.monotonic() + retry_after_seconds)
            self._condition.notify_all()


def estimate_input_tokens(messages: list[dict[str, str]]) -> int:
    # Calibrated from the recorded technical prompts: ~4 chars/token plus chat framing overhead.
    content = sum(max(1, (len(message.get("content", "")) + 3) // 4) for message in messages)
    return content + 12 * len(messages) + 8


def estimate_tokens(messages: list[dict[str, str]], output_capacity: int) -> int:
    return estimate_input_tokens(messages) + max(0, output_capacity)


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

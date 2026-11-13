from __future__ import annotations

import pytest

from relay.config import SchedulerConfig
from relay.providers.base import RateLimitError
from relay.providers.scheduler import RequestScheduler


@pytest.mark.asyncio
async def test_tpm_scheduler_delays_until_window_resets(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("relay.providers.scheduler.time.monotonic", lambda: clock[0])
    async def advance(seconds):
        clock[0] += seconds
    monkeypatch.setattr("relay.providers.scheduler.asyncio.sleep", advance)
    scheduler = RequestScheduler(SchedulerConfig(
        rpm_limit=30, tpm_limit=100, safety_margin=1.0,
        max_queue_size=2, request_deadline_seconds=2))
    await scheduler.acquire(60)
    await scheduler.acquire(60)
    assert clock[0] >= 160.0


@pytest.mark.asyncio
async def test_scheduler_reconciles_estimate_with_actual_usage():
    scheduler = RequestScheduler(SchedulerConfig(
        rpm_limit=30, tpm_limit=100, safety_margin=1.0,
        max_queue_size=2, request_deadline_seconds=2))
    reservation = await scheduler.acquire(80)
    await scheduler.reconcile(reservation, 20)
    second = await scheduler.acquire(70)
    assert second.reserved_tokens == 70


@pytest.mark.asyncio
async def test_scheduler_rejects_when_bounded_queue_is_full():
    scheduler = RequestScheduler(SchedulerConfig(max_queue_size=1))
    scheduler._pending = 1
    with pytest.raises(RateLimitError) as caught:
        await scheduler.acquire(10)
    assert caught.value.rate_limit_type == "local_queue_full"

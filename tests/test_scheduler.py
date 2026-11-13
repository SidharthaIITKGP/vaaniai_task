from __future__ import annotations

import pytest

from relay.config import SchedulerConfig
from relay.providers.base import RateLimitError
from relay.providers.scheduler import RequestScheduler


@pytest.mark.asyncio
async def test_tpm_scheduler_wakes_when_deficit_refills():
    import time
    scheduler = RequestScheduler(SchedulerConfig(
        rpm_limit=30, tpm_limit=60000, safety_margin=1.0,
        max_queue_size=2, request_deadline_seconds=2))
    await scheduler.acquire(59950, 0)
    started = time.perf_counter()
    second = await scheduler.acquire(100, 0)
    elapsed = time.perf_counter() - started
    assert 0.035 <= elapsed < 0.5
    assert second.wait_reason == "tpm"


@pytest.mark.asyncio
async def test_scheduler_reconciles_estimate_with_actual_usage():
    scheduler = RequestScheduler(SchedulerConfig(
        rpm_limit=30, tpm_limit=100, safety_margin=1.0,
        max_queue_size=2, request_deadline_seconds=2))
    reservation = await scheduler.acquire(40, 40)
    await scheduler.reconcile(reservation, 20)
    second = await scheduler.acquire(35, 35)
    assert second.reserved_tokens == 70


@pytest.mark.asyncio
async def test_scheduler_rejects_when_bounded_queue_is_full():
    scheduler = RequestScheduler(SchedulerConfig(max_queue_size=1))
    scheduler._pending = 1
    with pytest.raises(RateLimitError) as caught:
        await scheduler.acquire(10, 0)
    assert caught.value.rate_limit_type == "local_queue_full"


@pytest.mark.asyncio
async def test_rpm_wait_wakes_at_oldest_request_expiry():
    import time
    scheduler = RequestScheduler(SchedulerConfig(
        rpm_limit=1, tpm_limit=60000, safety_margin=1.0,
        request_deadline_seconds=1))
    scheduler._requests.append(time.monotonic() - 59.95)
    started = time.perf_counter()
    reservation = await scheduler.acquire(1, 0)
    assert 0.03 <= time.perf_counter() - started < 0.5
    assert reservation.wait_reason == "rpm"


@pytest.mark.asyncio
async def test_concurrent_reservations_and_no_double_reconcile():
    import asyncio
    scheduler = RequestScheduler(SchedulerConfig(
        rpm_limit=30, tpm_limit=1000, safety_margin=1.0))
    first, second = await asyncio.gather(
        scheduler.acquire(100, 0), scheduler.acquire(100, 0))
    assert scheduler.snapshot()["local_tpm_remaining"] <= 801
    await asyncio.gather(scheduler.reconcile(first, 50), scheduler.reconcile(second, 50))
    after = scheduler.snapshot()["local_tpm_remaining"]
    await scheduler.reconcile(first, 0)
    assert abs(scheduler.snapshot()["local_tpm_remaining"] - after) < 1


@pytest.mark.asyncio
async def test_interactive_admission_deadline():
    scheduler = RequestScheduler(SchedulerConfig(
        rpm_limit=30, tpm_limit=100, safety_margin=1.0,
        request_deadline_seconds=1))
    await scheduler.acquire(100, 0)
    with pytest.raises(RateLimitError) as caught:
        await scheduler.acquire(100, 0, deadline_seconds=0.02)
    assert caught.value.rate_limit_type == "local_queue_deadline"


@pytest.mark.asyncio
async def test_daily_quota_exhaustion_is_not_queued():
    scheduler = RequestScheduler(SchedulerConfig())
    scheduler.remaining_daily_requests = 0
    with pytest.raises(RateLimitError) as caught:
        await scheduler.acquire(1, 0)
    assert caught.value.rate_limit_type == "rpd_exhausted"


@pytest.mark.asyncio
async def test_fifo_prevents_large_request_starvation():
    import asyncio
    scheduler = RequestScheduler(SchedulerConfig(
        rpm_limit=30, tpm_limit=60000, safety_margin=1.0,
        request_deadline_seconds=1))
    await scheduler.acquire(59950, 0)
    completed = []
    async def reserve(name, tokens):
        await scheduler.acquire(tokens, 0)
        completed.append(name)
    large = asyncio.create_task(reserve("large", 200))
    await asyncio.sleep(0)
    small = asyncio.create_task(reserve("small", 10))
    await asyncio.gather(large, small)
    assert completed == ["large", "small"]

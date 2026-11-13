from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from pathlib import Path


OLD_CAPS = {"routing": 384, "planning": 600, "generation": 1200,
            "verification": 600, "revision": 1200, "revision_verification": 600}
NEW_CAPS = {"routing": 256, "planning": 384, "generation": 640,
            "verification": 384, "revision": 640, "revision_verification": 384}


def load_calls(path: Path) -> list[tuple[dict, bool]]:
    calls: list[tuple[dict, bool]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row_calls = json.loads(row.get("llm_calls_json") or "[]")
            for index, call in enumerate(row_calls):
                if call.get("input_tokens") is not None and call.get("output_tokens") is not None:
                    calls.append((call, index == len(row_calls) - 1))
    return calls


def simulate_fixed_window(calls: list[tuple[dict, bool]], budget: int,
                          runner_delay: float) -> float:
    now = 0.0
    wait = 0.0
    events: deque[list[float]] = deque()
    for call, pipeline_end in calls:
        cap = OLD_CAPS.get(call["stage"], 1200)
        reserved = int(call["input_tokens"]) + cap
        while True:
            while events and now - events[0][0] >= 60:
                events.popleft()
            if sum(event[1] for event in events) + reserved <= budget:
                break
            delta = max(0.0, events[0][0] + 60 - now)
            now += delta
            wait += delta
        events.append([now, float(reserved)])
        now += float(call.get("latency_ms") or 0) / 1000
        events[-1][1] = float(call["input_tokens"] + call["output_tokens"])
        if pipeline_end:
            now += runner_delay
    return wait


def simulate_token_bucket(calls: list[tuple[dict, bool]], budget: int,
                          runner_delay: float) -> float:
    now = 0.0
    last_refill = 0.0
    available = float(budget)
    refill_rate = budget / 60
    wait = 0.0
    for call, pipeline_end in calls:
        available = min(budget, available + (now - last_refill) * refill_rate)
        last_refill = now
        cap = NEW_CAPS.get(call["stage"], 640)
        reserved = int(call["input_tokens"]) + cap
        if available < reserved:
            delta = (reserved - available) / refill_rate
            now += delta
            wait += delta
            available = min(budget, available + delta * refill_rate)
            last_refill = now
        available -= reserved
        now += float(call.get("latency_ms") or 0) / 1000
        available = min(budget, available + (now - last_refill) * refill_rate)
        last_refill = now
        actual = int(call["input_tokens"]) + int(call["output_tokens"])
        available = min(budget, available + reserved - actual)
        if pipeline_end:
            now += runner_delay
    return wait


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic scheduler simulation from an existing CSV")
    parser.add_argument("--csv", default="results/smoke_fixed.csv")
    parser.add_argument("--tpm", type=int, default=8000)
    parser.add_argument("--safety-margin", type=float, default=0.8)
    parser.add_argument("--runner-delay", type=float, default=1.0)
    args = parser.parse_args()
    calls = load_calls(Path(args.csv))
    budget = int(args.tpm * args.safety_margin)
    old = simulate_fixed_window(calls, budget, args.runner_delay)
    new = simulate_token_bucket(calls, budget, args.runner_delay)
    actual = sum((call.get("queue_wait_ms") or 0) for call, _ in calls) / 1000
    print(f"Calls: {len(calls)}; effective TPM budget: {budget}")
    print(f"Recorded old queue wait: {actual:.3f}s")
    print(f"Simulated old fixed-window wait: {old:.3f}s")
    print(f"Simulated new token-bucket wait: {new:.3f}s")
    print("Simulation reuses observed token usage/provider durations; it is not a real-provider measurement.")


if __name__ == "__main__":
    main()

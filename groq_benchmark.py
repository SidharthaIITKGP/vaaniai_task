"""Small, reproducible Groq streaming/concurrency probe for Vaani Relay.

Run examples:
    pip install httpx
    export GROQ_API_KEY='...'
    python groq_benchmark.py --scenario steady --n 5 --concurrency 1 --spacing 4
    python groq_benchmark.py --scenario burst_5 --n 5 --concurrency 5 --spacing 0

This version permits a burst of up to 100 real requests. Respect the provider limits; rate-limit responses are expected.
"""

import argparse
import asyncio
import csv
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().with_name(".env"))

API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-20b"
FIELDS = [
    "request_id", "scenario", "source", "start_time", "first_token_ms",
    "total_ms", "status", "retries", "tokens_in", "tokens_out",
    "retrieval_setting", "answer_grade", "queue_ms", "http_status",
    "retry_after", "limit_rpd", "remaining_rpd", "limit_tpm",
    "remaining_tpm", "response_chars", "error_message",
]


def ms(seconds):
    return round(seconds * 1000, 2)


async def request_one(client, sem, args, index):
    # Arrival time and queue time are distinct from provider/service time.
    await asyncio.sleep(index * args.spacing)
    arrival = time.perf_counter()
    arrival_utc = datetime.now(timezone.utc).isoformat()
    async with sem:
        begin = time.perf_counter()
        row = {key: "" for key in FIELDS}
        row.update({
            "request_id": str(uuid.uuid4()),
            "scenario": args.scenario,
            "source": "real",
            "start_time": arrival_utc,
            "retries": 0,
            "retrieval_setting": "none_pre_rag",
            "answer_grade": "",  # Must be graded separately, never fabricated.
            "queue_ms": ms(begin - arrival),
        })
        headers = {
            "Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": (
                "Briefly explain the difference between latency and throughput "
                "in two sentences."
            )}],
            "stream": True,
            "stream_options": {"include_usage": True},
            "reasoning_effort": "low",
            "reasoning_format": "hidden",
            "max_completion_tokens": 160,
        }
        chunks = []
        saw_done = False
        try:
            async with client.stream("POST", API_URL, headers=headers, json=payload) as resp:
                row["http_status"] = resp.status_code
                h = resp.headers
                row.update({
                    "retry_after": h.get("retry-after", ""),
                    "limit_rpd": h.get("x-ratelimit-limit-requests", ""),
                    "remaining_rpd": h.get("x-ratelimit-remaining-requests", ""),
                    "limit_tpm": h.get("x-ratelimit-limit-tokens", ""),
                    "remaining_tpm": h.get("x-ratelimit-remaining-tokens", ""),
                })
                if resp.status_code != 200:
                    body = (await resp.aread()).decode("utf-8", errors="replace")
                    row["error_message"] = body[:300]
                    row["status"] = "rate_limited" if resp.status_code == 429 else "error"
                else:
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        raw = line.removeprefix("data:").strip()
                        if raw == "[DONE]":
                            saw_done = True
                            break
                        if not raw:
                            continue
                        event = json.loads(raw)
                        usage = event.get("usage") or {}
                        if "prompt_tokens" in usage:
                            row["tokens_in"] = usage["prompt_tokens"]
                        if "completion_tokens" in usage:
                            row["tokens_out"] = usage["completion_tokens"]
                        for choice in event.get("choices", []):
                            content = (choice.get("delta") or {}).get("content")
                            if content:
                                if row["first_token_ms"] == "":
                                    # Visible content, not hidden reasoning or just HTTP headers.
                                    row["first_token_ms"] = ms(time.perf_counter() - arrival)
                                chunks.append(content)
                    row["status"] = "ok" if saw_done else "error"
                    if not saw_done:
                        row["error_message"] = "stream ended without [DONE]"
        except httpx.TimeoutException as exc:
            row["status"] = "timeout"
            row["error_message"] = str(exc)[:300]
        except (httpx.HTTPError, ValueError) as exc:
            row["status"] = "error"
            row["error_message"] = str(exc)[:300]
        finally:
            row["total_ms"] = ms(time.perf_counter() - arrival)
            row["response_chars"] = sum(len(x) for x in chunks)
        print(
            f"{index + 1:>2}  {row['status']:>12}  "
            f"TTFT={str(row['first_token_ms']):>8} ms  "
            f"total={row['total_ms']:>8} ms  "
            f"tokens={row['tokens_in']}/{row['tokens_out']}  "
            f"HTTP={row['http_status']}"
        )
        return row


async def main(args):
    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit("Set GROQ_API_KEY in your environment first.")
    sem = asyncio.Semaphore(args.concurrency)
    timeout = httpx.Timeout(45, connect=10)
    async with httpx.AsyncClient(timeout=timeout, limits=httpx.Limits(max_connections=args.concurrency, max_keepalive_connections=min(args.concurrency, 20))) as client:
        rows = await asyncio.gather(*(
            request_one(client, sem, args, i) for i in range(args.n)
        ))

    path = Path(args.csv)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} request-level measurements to {path}")
    print("Note: an empty token count means not reported; it is NOT zero.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Groq real-service latency probe")
    parser.add_argument("--scenario", default="steady")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--spacing", type=float, default=4.0,
                        help="Seconds between arrivals. Use 0 for a simultaneous burst.")
    parser.add_argument("--csv", default="groq_probe.csv")
    args = parser.parse_args()
    if not 1 <= args.n <= 100 or not 1 <= args.concurrency <= args.n:
        parser.error("Choose 1 <= concurrency <= n <= 100")
    if args.spacing < 0:
        parser.error("--spacing must not be negative")
    asyncio.run(main(args))

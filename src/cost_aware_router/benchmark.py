from __future__ import annotations

import argparse
import asyncio
import csv
import statistics
import time
import uuid
from pathlib import Path

import httpx
import pandas as pd

from .models import GenerateRequest, RoutePolicy


TOPICS = [
    "distributed systems cache consistency routing scheduler throughput latency",
    "machine learning inference prefix attention kv cache batching prefill decode",
    "operating systems queue semaphore thread process memory scheduling fairness",
    "database transactions isolation index query optimizer storage buffer cache",
]


def repeated_prefix_workload(total: int, prefix_tokens: int, suffix_tokens: int) -> list[str]:
    prompts: list[str] = []
    for i in range(total):
        topic = TOPICS[i % len(TOPICS)]
        base = (topic.split() * ((prefix_tokens // len(topic.split())) + 1))[:prefix_tokens]
        suffix = [f"req{i}", "variant", str(i % 17)] * ((suffix_tokens // 3) + 1)
        prompts.append(" ".join(base + suffix[:suffix_tokens]))
    return prompts


async def run_one(client: httpx.AsyncClient, router_url: str, prompt: str, max_tokens: int) -> dict[str, object]:
    req = GenerateRequest(prompt=prompt, max_tokens=max_tokens, request_id=str(uuid.uuid4()))
    wall_start = time.perf_counter()
    resp = await client.post(f"{router_url.rstrip('/')}/generate", json=req.model_dump())
    resp.raise_for_status()
    row = resp.json()
    row["client_wall_ms"] = (time.perf_counter() - wall_start) * 1000
    return row


async def run_benchmark(args: argparse.Namespace) -> list[dict[str, object]]:
    prompts = repeated_prefix_workload(args.requests, args.prefix_tokens, args.suffix_tokens)
    limits = httpx.Limits(max_connections=args.concurrency * 2, max_keepalive_connections=args.concurrency)
    async with httpx.AsyncClient(timeout=args.timeout, limits=limits) as client:
        semaphore = asyncio.Semaphore(args.concurrency)

        async def guarded(prompt: str) -> dict[str, object]:
            async with semaphore:
                return await run_one(client, args.router_url, prompt, args.max_tokens)

        return await asyncio.gather(*(guarded(prompt) for prompt in prompts))


def summarize(rows: list[dict[str, object]]) -> dict[str, float | int]:
    ttft = [float(r["ttft_ms"]) for r in rows]
    latency = [float(r["latency_ms"]) for r in rows]
    wall = [float(r["client_wall_ms"]) for r in rows]
    hits = [float(r["cache_hit_tokens"]) for r in rows]
    prompt_tokens = [float(r["prompt_tokens"]) for r in rows]
    workers = [str(r["worker_id"]) for r in rows]
    counts = {worker: workers.count(worker) for worker in sorted(set(workers))}
    min_count = max(min(counts.values()), 1) if counts else 1
    return {
        "requests": len(rows),
        "ttft_mean_ms": statistics.fmean(ttft),
        "ttft_p50_ms": statistics.median(ttft),
        "ttft_p95_ms": pd.Series(ttft).quantile(0.95),
        "latency_mean_ms": statistics.fmean(latency),
        "latency_p95_ms": pd.Series(latency).quantile(0.95),
        "client_wall_mean_ms": statistics.fmean(wall),
        "cache_hit_tokens_mean": statistics.fmean(hits),
        "cache_hit_rate": sum(hits) / max(sum(prompt_tokens), 1.0),
        "partial_match_rate": sum(1 for r in rows if r["partial_match"]) / max(len(rows), 1),
        "imbalance_ratio": max(counts.values(), default=0) / min_count,
    }


def write_outputs(rows: list[dict[str, object]], output_dir: Path, label: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"{label}_raw.csv"
    summary_path = output_dir / f"{label}_summary.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with raw_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=80)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--prefix-tokens", type=int, default=80)
    parser.add_argument("--suffix-tokens", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--label", default="benchmark")
    args = parser.parse_args()

    rows = asyncio.run(run_benchmark(args))
    write_outputs(rows, args.output_dir, args.label)
    summary = summarize(rows)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

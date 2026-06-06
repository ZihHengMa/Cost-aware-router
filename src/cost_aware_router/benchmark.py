from __future__ import annotations

import argparse
import asyncio
import csv
import random
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


def prefix_words(group_id: int, prefix_tokens: int) -> list[str]:
    topic = TOPICS[group_id % len(TOPICS)].split()
    group_marker = [f"tenant_{group_id}", f"document_{group_id}", "shared_context"]
    source = group_marker + topic
    return (source * ((prefix_tokens // len(source)) + 1))[:prefix_tokens]


def suffix_words(index: int, suffix_tokens: int) -> list[str]:
    suffix = [f"req{index}", "variant", str(index % 17)] * ((suffix_tokens // 3) + 1)
    return suffix[:suffix_tokens]


def repeated_prefix_workload(
    total: int,
    prefix_tokens: int,
    suffix_tokens: int,
    *,
    prefix_groups: int,
    start_index: int = 0,
    prompt_id_offset: int = 0,
) -> list[str]:
    prompts: list[str] = []
    groups = max(1, prefix_groups)
    for offset in range(total):
        sequence_index = start_index + offset
        prompt_id = prompt_id_offset + sequence_index
        group_id = sequence_index % groups
        prompts.append(" ".join(prefix_words(group_id, prefix_tokens) + suffix_words(prompt_id, suffix_tokens)))
    return prompts


def realistic_workload(
    total: int,
    prefix_tokens: int,
    suffix_tokens: int,
    *,
    prefix_groups: int,
    hot_prefix_groups: int,
    hot_share: float,
    burst_size: int,
    seed: int,
    start_index: int = 0,
    prompt_id_offset: int = 0,
) -> list[str]:
    prompts: list[str] = []
    rng = random.Random(seed + start_index)
    groups = max(1, prefix_groups)
    hot_groups = list(range(max(1, min(hot_prefix_groups, groups))))
    cold_groups = list(range(len(hot_groups), groups))
    hot_probability = max(0.0, min(hot_share, 1.0))
    burst = max(1, burst_size)

    while len(prompts) < total:
        use_hot = not cold_groups or rng.random() < hot_probability
        group_id = rng.choice(hot_groups if use_hot else cold_groups)
        for _ in range(burst):
            if len(prompts) >= total:
                break
            sequence_index = start_index + len(prompts)
            prompt_id = prompt_id_offset + sequence_index
            prompts.append(" ".join(prefix_words(group_id, prefix_tokens) + suffix_words(prompt_id, suffix_tokens)))
    return prompts


def build_workload(total: int, args: argparse.Namespace, *, start_index: int = 0) -> list[str]:
    if args.workload == "realistic":
        return realistic_workload(
            total,
            args.prefix_tokens,
            args.suffix_tokens,
            prefix_groups=args.prefix_groups,
            hot_prefix_groups=args.hot_prefix_groups,
            hot_share=args.hot_share,
            burst_size=args.burst_size,
            seed=args.seed,
            start_index=start_index,
            prompt_id_offset=args.prompt_id_offset,
        )
    return repeated_prefix_workload(
        total,
        args.prefix_tokens,
        args.suffix_tokens,
        prefix_groups=args.prefix_groups,
        start_index=start_index,
        prompt_id_offset=args.prompt_id_offset,
    )


async def run_one(client: httpx.AsyncClient, router_url: str, prompt: str, max_tokens: int) -> dict[str, object]:
    req = GenerateRequest(prompt=prompt, max_tokens=max_tokens, request_id=str(uuid.uuid4()))
    wall_start = time.perf_counter()
    resp = await client.post(f"{router_url.rstrip('/')}/generate", json=req.model_dump())
    resp.raise_for_status()
    row = resp.json()
    row["client_wall_ms"] = (time.perf_counter() - wall_start) * 1000
    return row


async def run_prompts(
    client: httpx.AsyncClient,
    router_url: str,
    prompts: list[str],
    *,
    concurrency: int,
    max_tokens: int,
) -> list[dict[str, object]]:
    semaphore = asyncio.Semaphore(concurrency)

    async def guarded(prompt: str) -> dict[str, object]:
        async with semaphore:
            return await run_one(client, router_url, prompt, max_tokens)

    return await asyncio.gather(*(guarded(prompt) for prompt in prompts))


async def run_benchmark(args: argparse.Namespace) -> list[dict[str, object]]:
    warmup_prompts = build_workload(args.warmup_requests, args)
    prompts = build_workload(args.requests, args, start_index=args.warmup_requests)
    limits = httpx.Limits(max_connections=args.concurrency * 2, max_keepalive_connections=args.concurrency)
    async with httpx.AsyncClient(timeout=args.timeout, limits=limits) as client:
        if warmup_prompts:
            print(f"warming cache with {len(warmup_prompts)} requests...")
            await run_prompts(
                client,
                args.router_url,
                warmup_prompts,
                concurrency=args.warmup_concurrency,
                max_tokens=args.warmup_max_tokens,
            )
        return await run_prompts(
            client,
            args.router_url,
            prompts,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
        )


def summarize(rows: list[dict[str, object]], *, worker_count: int = 2) -> dict[str, float | int]:
    ttft = [float(r["ttft_ms"]) for r in rows]
    latency = [float(r["latency_ms"]) for r in rows]
    wall = [float(r["client_wall_ms"]) for r in rows]
    hits = [float(r["cache_hit_tokens"]) for r in rows]
    prompt_tokens = [float(r["prompt_tokens"]) for r in rows]
    workers = [str(r["worker_id"]) for r in rows]
    expected_workers = [f"worker-{idx}" for idx in range(max(1, worker_count))]
    observed_workers = sorted(set(workers))
    worker_ids = expected_workers + [worker for worker in observed_workers if worker not in expected_workers]
    counts = {worker: workers.count(worker) for worker in worker_ids}
    min_count = max(min(counts.values()), 1) if counts else 1
    summary: dict[str, float | int] = {
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
        "max_worker_share": max(counts.values(), default=0) / max(len(rows), 1),
        "min_worker_share": min(counts.values(), default=0) / max(len(rows), 1),
    }
    for worker_id, count in counts.items():
        summary[f"{worker_id}_requests"] = count
    return summary


def write_outputs(rows: list[dict[str, object]], output_dir: Path, label: str, *, worker_count: int = 2) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"{label}_raw.csv"
    summary_path = output_dir / f"{label}_summary.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with raw_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows, worker_count=worker_count)
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
    parser.add_argument("--prefix-groups", type=int, default=len(TOPICS))
    parser.add_argument("--workload", choices=["repeated_prefix", "realistic"], default="repeated_prefix")
    parser.add_argument("--hot-prefix-groups", type=int, default=1)
    parser.add_argument("--hot-share", type=float, default=0.80)
    parser.add_argument("--burst-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt-id-offset", type=int, default=0)
    parser.add_argument("--suffix-tokens", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--warmup-requests", type=int, default=0)
    parser.add_argument("--warmup-concurrency", type=int, default=1)
    parser.add_argument("--warmup-max-tokens", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--label", default="benchmark")
    parser.add_argument("--worker-count", type=int, default=2)
    args = parser.parse_args()

    rows = asyncio.run(run_benchmark(args))
    write_outputs(rows, args.output_dir, args.label, worker_count=args.worker_count)
    summary = summarize(rows, worker_count=args.worker_count)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CostAwareParams:
    queue_weight: float
    cache_hit_bonus: float
    locality_queue_slack: int
    locality_threshold: float

    @property
    def label(self) -> str:
        return (
            f"ca_q{safe_number(self.queue_weight)}"
            f"_b{safe_number(self.cache_hit_bonus)}"
            f"_s{self.locality_queue_slack}"
            f"_t{safe_number(self.locality_threshold)}"
        )


def safe_number(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def parse_list(raw: str, cast):
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def prompt_id_offset(label: str, seed: int) -> int:
    # Keep the same prefix distribution per seed, but avoid exact prompt reuse
    # across candidates by changing only request suffix IDs.
    return seed * 10_000_000 + sum((idx + 1) * ord(ch) for idx, ch in enumerate(label)) * 1_000


def http_ready(url: str, timeout_s: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            return 200 <= resp.status < 500
    except Exception:
        return False


def wait_http(url: str, name: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if 200 <= resp.status < 500:
                    print(f"{name} is ready: {url}", flush=True)
                    return
        except Exception as exc:
            last_error = exc
        time.sleep(1)
    message = f"Timed out waiting for {name}: {url}"
    if last_error is not None:
        message += f"\nLast error: {last_error}"
    raise RuntimeError(message)


def stop_process(proc: subprocess.Popen[str] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def read_summary(path: Path) -> dict[str, float | str]:
    with path.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    out: dict[str, float | str] = {}
    for key, value in row.items():
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            out[key] = value
    return out


def router_command(
    args: argparse.Namespace,
    *,
    policy: str,
    metadata_db: Path,
    params: CostAwareParams | None = None,
) -> list[str]:
    command = [
        args.python,
        "-m",
        "cost_aware_router.router",
        "--host",
        args.router_host,
        "--port",
        str(args.router_port),
        "--policy",
        policy,
        "--metadata-db",
        str(metadata_db),
    ]
    if params is not None:
        command += [
            "--queue-weight",
            str(params.queue_weight),
            "--cache-hit-bonus",
            str(params.cache_hit_bonus),
            "--locality-queue-slack",
            str(params.locality_queue_slack),
            "--locality-threshold",
            str(params.locality_threshold),
        ]
    for worker in args.worker:
        command += ["--worker", worker]
    return command


def benchmark_command(args: argparse.Namespace, *, label: str, seed: int) -> list[str]:
    return [
        args.python,
        "-m",
        "cost_aware_router.benchmark",
        "--router-url",
        args.router_url,
        "--label",
        label,
        "--workload",
        args.workload,
        "--requests",
        str(args.requests),
        "--concurrency",
        str(args.concurrency),
        "--prefix-tokens",
        str(args.prefix_tokens),
        "--prefix-groups",
        str(args.prefix_groups),
        "--hot-prefix-groups",
        str(args.hot_prefix_groups),
        "--hot-share",
        str(args.hot_share),
        "--burst-size",
        str(args.burst_size),
        "--suffix-tokens",
        str(args.suffix_tokens),
        "--max-tokens",
        str(args.max_tokens),
        "--warmup-requests",
        str(args.warmup_requests),
        "--warmup-concurrency",
        str(args.warmup_concurrency),
        "--warmup-max-tokens",
        str(args.warmup_max_tokens),
        "--timeout",
        str(args.timeout),
        "--output-dir",
        str(args.output_dir),
        "--seed",
        str(seed),
        "--prompt-id-offset",
        str(prompt_id_offset(label, seed)),
        "--worker-count",
        str(len(args.worker)),
    ]


def run_policy_once(
    args: argparse.Namespace,
    *,
    label: str,
    seed: int,
    policy: str,
    metadata_db: Path,
    params: CostAwareParams | None = None,
) -> dict[str, float | str]:
    summary_path = args.output_dir / f"{label}_summary.csv"
    if args.resume and summary_path.exists():
        print(f"Skipping existing result: {summary_path}", flush=True)
        return read_summary(summary_path)

    if metadata_db.exists() and args.reset_metadata:
        metadata_db.unlink()

    router_log = args.output_dir / f"{label}_router.log"
    bench_log = args.output_dir / f"{label}_benchmark.log"
    proc: subprocess.Popen[str] | None = None

    try:
        with router_log.open("w") as router_handle:
            proc = subprocess.Popen(
                router_command(args, policy=policy, metadata_db=metadata_db, params=params),
                stdout=router_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        wait_http(f"{args.router_url}/state", f"router {label}", args.router_start_timeout)

        print(f"Running benchmark {label}", flush=True)
        with bench_log.open("w") as bench_handle:
            subprocess.run(
                benchmark_command(args, label=label, seed=seed),
                stdout=bench_handle,
                stderr=subprocess.STDOUT,
                check=True,
                text=True,
            )
    except Exception:
        if router_log.exists():
            print(f"\nRouter log tail for {label}:", file=sys.stderr)
            print("\n".join(router_log.read_text(errors="replace").splitlines()[-80:]), file=sys.stderr)
        if bench_log.exists():
            print(f"\nBenchmark log tail for {label}:", file=sys.stderr)
            print("\n".join(bench_log.read_text(errors="replace").splitlines()[-80:]), file=sys.stderr)
        raise
    finally:
        stop_process(proc)

    return read_summary(summary_path)


def build_grid(args: argparse.Namespace) -> list[CostAwareParams]:
    return [
        CostAwareParams(queue_weight, cache_hit_bonus, locality_queue_slack, locality_threshold)
        for queue_weight, cache_hit_bonus, locality_queue_slack, locality_threshold in itertools.product(
            parse_list(args.queue_weights, float),
            parse_list(args.cache_hit_bonuses, float),
            parse_list(args.locality_queue_slacks, int),
            parse_list(args.locality_thresholds, float),
        )
    ]


def write_parameters(args: argparse.Namespace, grid: list[CostAwareParams]) -> None:
    path = args.output_dir / "tuning_parameters.txt"
    path.write_text(
        "\n".join(
            [
                f"seeds={args.seeds}",
                f"baseline_policies={args.baseline_policies}",
                f"candidates={len(grid)}",
                f"queue_weights={args.queue_weights}",
                f"cache_hit_bonuses={args.cache_hit_bonuses}",
                f"locality_queue_slacks={args.locality_queue_slacks}",
                f"locality_thresholds={args.locality_thresholds}",
                f"workload={args.workload}",
                f"requests={args.requests}",
                f"concurrency={args.concurrency}",
                f"prefix_tokens={args.prefix_tokens}",
                f"prefix_groups={args.prefix_groups}",
                f"hot_prefix_groups={args.hot_prefix_groups}",
                f"hot_share={args.hot_share}",
                f"burst_size={args.burst_size}",
                f"suffix_tokens={args.suffix_tokens}",
                f"max_tokens={args.max_tokens}",
                f"warmup_requests={args.warmup_requests}",
                f"warmup_concurrency={args.warmup_concurrency}",
                "",
            ]
        )
    )


def rank_candidates(
    args: argparse.Namespace,
    *,
    baselines: dict[int, dict[str, dict[str, float | str]]],
    candidates: dict[str, dict[int, dict[str, float | str]]],
    params_by_label: dict[str, CostAwareParams],
) -> list[dict[str, float | str]]:
    metric = args.metric
    rows: list[dict[str, float | str]] = []
    baseline_policies = parse_list(args.baseline_policies, str)
    seeds = parse_list(args.seeds, int)

    for label, by_seed in candidates.items():
        params = params_by_label[label]
        metric_values: list[float] = []
        ttft_mean_values: list[float] = []
        latency_p95_values: list[float] = []
        cache_hit_values: list[float] = []
        imbalance_values: list[float] = []
        deltas_vs_best: list[float] = []
        wins_vs_best = 0
        pairwise_wins = 0
        pairwise_total = 0

        for seed in seeds:
            candidate_summary = by_seed[seed]
            candidate_metric = float(candidate_summary[metric])
            metric_values.append(candidate_metric)
            ttft_mean_values.append(float(candidate_summary["ttft_mean_ms"]))
            latency_p95_values.append(float(candidate_summary["latency_p95_ms"]))
            cache_hit_values.append(float(candidate_summary["cache_hit_rate"]))
            imbalance_values.append(float(candidate_summary["imbalance_ratio"]))

            baseline_metrics = [float(baselines[seed][policy][metric]) for policy in baseline_policies]
            best_baseline_metric = min(baseline_metrics)
            deltas_vs_best.append(candidate_metric - best_baseline_metric)
            if candidate_metric <= best_baseline_metric:
                wins_vs_best += 1
            for policy in baseline_policies:
                pairwise_total += 1
                if candidate_metric <= float(baselines[seed][policy][metric]):
                    pairwise_wins += 1

        rows.append(
            {
                "label": label,
                "queue_weight": params.queue_weight,
                "cache_hit_bonus": params.cache_hit_bonus,
                "locality_queue_slack": params.locality_queue_slack,
                "locality_threshold": params.locality_threshold,
                "metric": metric,
                "seeds": len(seeds),
                "wins_vs_best_baseline": wins_vs_best,
                "win_rate_vs_best_baseline": wins_vs_best / max(len(seeds), 1),
                "pairwise_baseline_wins": pairwise_wins,
                "pairwise_baseline_win_rate": pairwise_wins / max(pairwise_total, 1),
                f"mean_{metric}": sum(metric_values) / len(metric_values),
                f"mean_delta_{metric}_vs_best_baseline": sum(deltas_vs_best) / len(deltas_vs_best),
                "mean_ttft_mean_ms": sum(ttft_mean_values) / len(ttft_mean_values),
                "mean_latency_p95_ms": sum(latency_p95_values) / len(latency_p95_values),
                "mean_cache_hit_rate": sum(cache_hit_values) / len(cache_hit_values),
                "mean_imbalance_ratio": sum(imbalance_values) / len(imbalance_values),
            }
        )

    rows.sort(
        key=lambda row: (
            -float(row["win_rate_vs_best_baseline"]),
            -float(row["pairwise_baseline_win_rate"]),
            float(row[f"mean_delta_{metric}_vs_best_baseline"]),
            float(row[f"mean_{metric}"]),
            float(row["mean_imbalance_ratio"]),
        )
    )
    return rows


def write_rankings(args: argparse.Namespace, rankings: list[dict[str, float | str]]) -> None:
    report_path = args.output_dir / "cost_aware_tuning_report.csv"
    if not rankings:
        raise RuntimeError("No cost-aware candidates were ranked")

    with report_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rankings[0].keys()))
        writer.writeheader()
        writer.writerows(rankings)

    best = rankings[0]
    command_path = args.output_dir / "best_cost_aware_command.sh"
    command_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "./scripts/run_router.sh cost_aware \\",
                f"  --queue-weight {best['queue_weight']} \\",
                f"  --cache-hit-bonus {best['cache_hit_bonus']} \\",
                f"  --locality-queue-slack {int(float(best['locality_queue_slack']))} \\",
                f"  --locality-threshold {best['locality_threshold']}",
                "",
            ]
        )
    )
    command_path.chmod(0o755)

    print(f"\nWrote {report_path}", flush=True)
    print(f"Wrote {command_path}", flush=True)
    print("\nTop candidates:", flush=True)
    metric = args.metric
    for row in rankings[: min(args.print_top, len(rankings))]:
        print(
            "  {label}: q={queue_weight:g} bonus={cache_hit_bonus:g} slack={locality_queue_slack} "
            "threshold={locality_threshold:g} win_best={win_rate_vs_best_baseline:.2f} "
            "pairwise={pairwise_baseline_win_rate:.2f} mean_{metric}={mean_metric:.2f} "
            "delta={delta:.2f} imbalance={imbalance:.2f}".format(
                label=row["label"],
                queue_weight=float(row["queue_weight"]),
                cache_hit_bonus=float(row["cache_hit_bonus"]),
                locality_queue_slack=int(float(row["locality_queue_slack"])),
                locality_threshold=float(row["locality_threshold"]),
                win_rate_vs_best_baseline=float(row["win_rate_vs_best_baseline"]),
                pairwise_baseline_win_rate=float(row["pairwise_baseline_win_rate"]),
                metric=metric,
                mean_metric=float(row[f"mean_{metric}"]),
                delta=float(row[f"mean_delta_{metric}_vs_best_baseline"]),
                imbalance=float(row["mean_imbalance_ratio"]),
            ),
            flush=True,
        )


def main() -> None:
    default_python = str(Path(".venv/bin/python")) if Path(".venv/bin/python").exists() else sys.executable
    parser = argparse.ArgumentParser(description="Grid-search cost-aware router parameters.")
    parser.add_argument("--python", default=default_python)
    parser.add_argument("--worker", action="append")
    parser.add_argument("--router-host", default="127.0.0.1")
    parser.add_argument("--router-port", type=int, default=8000)
    parser.add_argument("--router-start-timeout", type=float, default=60)
    parser.add_argument("--output-dir", type=Path, default=Path("results/cost_aware_tuning"))
    parser.add_argument("--metadata-dir", type=Path, default=Path("data/cost_aware_tuning"))
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--baseline-policies", default="rr,least_queue,cache_aware")
    parser.add_argument("--metric", default="ttft_p95_ms")
    parser.add_argument("--queue-weights", default="32,64,128,192")
    parser.add_argument("--cache-hit-bonuses", default="0,0.5,1")
    parser.add_argument("--locality-queue-slacks", default="1")
    parser.add_argument("--locality-thresholds", default="0.9")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--reset-metadata", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-top", type=int, default=10)

    parser.add_argument("--workload", choices=["repeated_prefix", "realistic"], default="realistic")
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--prefix-tokens", type=int, default=512)
    parser.add_argument("--prefix-groups", type=int, default=20)
    parser.add_argument("--hot-prefix-groups", type=int, default=2)
    parser.add_argument("--hot-share", type=float, default=0.85)
    parser.add_argument("--burst-size", type=int, default=4)
    parser.add_argument("--suffix-tokens", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--warmup-requests", type=int, default=40)
    parser.add_argument("--warmup-concurrency", type=int, default=2)
    parser.add_argument("--warmup-max-tokens", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()

    if args.worker is None:
        args.worker = ["http://127.0.0.1:8100", "http://127.0.0.1:8101"]

    args.router_url = f"http://{args.router_host}:{args.router_port}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.metadata_dir.mkdir(parents=True, exist_ok=True)

    grid = build_grid(args)
    seeds = parse_list(args.seeds, int)
    baseline_policies = parse_list(args.baseline_policies, str)
    write_parameters(args, grid)

    print(f"Cost-aware candidates: {len(grid)}", flush=True)
    print(f"Seeds: {seeds}", flush=True)
    print(f"Output: {args.output_dir}", flush=True)

    if args.dry_run:
        for params in grid:
            print(params.label, params, flush=True)
        return

    if http_ready(f"{args.router_url}/state"):
        raise SystemExit(f"A router is already running at {args.router_url}; stop it first or choose --router-port.")
    for idx, worker in enumerate(args.worker):
        wait_http(f"{worker.rstrip('/')}/state", f"worker-{idx} adapter", 10)

    baselines: dict[int, dict[str, dict[str, float | str]]] = {}
    candidates: dict[str, dict[int, dict[str, float | str]]] = {params.label: {} for params in grid}
    params_by_label = {params.label: params for params in grid}

    for seed in seeds:
        baselines[seed] = {}
        if args.skip_baselines:
            for policy in baseline_policies:
                label = f"{policy}_seed{seed}"
                baselines[seed][policy] = read_summary(args.output_dir / f"{label}_summary.csv")
        else:
            for policy in baseline_policies:
                actual_policy = "round_robin" if policy == "rr" else policy
                label = f"{policy}_seed{seed}"
                metadata_db = args.metadata_dir / f"{label}.sqlite"
                baselines[seed][policy] = run_policy_once(
                    args,
                    label=label,
                    seed=seed,
                    policy=actual_policy,
                    metadata_db=metadata_db,
                )

        for params in grid:
            label = f"{params.label}_seed{seed}"
            metadata_db = args.metadata_dir / f"{label}.sqlite"
            candidates[params.label][seed] = run_policy_once(
                args,
                label=label,
                seed=seed,
                policy="cost_aware",
                metadata_db=metadata_db,
                params=params,
            )

    rankings = rank_candidates(args, baselines=baselines, candidates=candidates, params_by_label=params_by_label)
    write_rankings(args, rankings)


if __name__ == "__main__":
    main()

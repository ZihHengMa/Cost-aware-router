from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/summary.png"))
    args = parser.parse_args()

    frames = []
    for path in sorted(args.results_dir.glob("*_summary.csv")):
        df = pd.read_csv(path)
        df["policy"] = path.name.removesuffix("_summary.csv")
        frames.append(df)
    if not frames:
        raise SystemExit(f"No *_summary.csv files found in {args.results_dir}")

    data = pd.concat(frames, ignore_index=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    data.plot.bar(x="policy", y="ttft_mean_ms", ax=axes[0], legend=False, color="#2f6f73")
    axes[0].set_title("Mean TTFT")
    axes[0].set_ylabel("ms")
    data.plot.bar(x="policy", y="cache_hit_rate", ax=axes[1], legend=False, color="#8f5d2c")
    axes[1].set_title("Cache hit rate")
    axes[1].set_ylim(0, 1)
    data.plot.bar(x="policy", y="imbalance_ratio", ax=axes[2], legend=False, color="#555a7a")
    axes[2].set_title("Load imbalance")
    axes[2].set_ylabel("max/min requests")
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(data.to_string(index=False))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

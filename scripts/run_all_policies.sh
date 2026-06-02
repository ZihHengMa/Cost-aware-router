#!/usr/bin/env bash
set -euo pipefail

for policy in round_robin least_queue cache_aware cost_aware prefill_scratch; do
  echo "Run router separately with policy=${policy}, then press enter."
  read -r
  python -m cost_aware_router.benchmark --label "${policy}" --requests 80 --concurrency 8
done

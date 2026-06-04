#!/usr/bin/env bash
set -euo pipefail

for policy in round_robin least_queue cache_aware cost_aware; do
  echo "Run router separately with policy=${policy}, then press enter."
  read -r
  python -m cost_aware_router.benchmark --label "${policy}" --requests 80 --concurrency 8
done

echo "Run prefill_scratch separately after restarting vLLM with VLLM_ENABLE_PREFIX_CACHING=0."

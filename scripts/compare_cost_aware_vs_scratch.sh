#!/usr/bin/env bash
set -euo pipefail

REQUESTS="${REQUESTS:-80}"
CONCURRENCY="${CONCURRENCY:-8}"
MAX_TOKENS="${MAX_TOKENS:-32}"

echo "1) Start vLLM with prefix caching enabled, adapters enabled, and router policy cost_aware."
echo "   Then press enter to run the cost-aware benchmark."
read -r
python -m cost_aware_router.benchmark \
  --label cost_aware \
  --requests "${REQUESTS}" \
  --concurrency "${CONCURRENCY}" \
  --max-tokens "${MAX_TOKENS}"

echo "2) Restart vLLM with VLLM_ENABLE_PREFIX_CACHING=0, adapters with ADAPTER_ENABLE_PREFIX_CACHE=0,"
echo "   and router policy prefill_scratch. Then press enter to run the scratch baseline."
read -r
python -m cost_aware_router.benchmark \
  --label prefill_scratch \
  --requests "${REQUESTS}" \
  --concurrency "${CONCURRENCY}" \
  --max-tokens "${MAX_TOKENS}"

python -m cost_aware_router.analyze --results-dir results --output results/cost_aware_vs_scratch.png

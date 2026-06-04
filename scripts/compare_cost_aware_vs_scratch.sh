#!/usr/bin/env bash
set -euo pipefail

REQUESTS="${REQUESTS:-120}"
CONCURRENCY="${CONCURRENCY:-8}"
PREFIX_TOKENS="${PREFIX_TOKENS:-2048}"
PREFIX_GROUPS="${PREFIX_GROUPS:-2}"
SUFFIX_TOKENS="${SUFFIX_TOKENS:-8}"
MAX_TOKENS="${MAX_TOKENS:-1}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-32}"
WARMUP_CONCURRENCY="${WARMUP_CONCURRENCY:-1}"
TIMEOUT="${TIMEOUT:-900}"
OUTPUT_DIR="${OUTPUT_DIR:-results/prefill_sensitive}"

echo "1) Start vLLM with prefix caching enabled, adapters enabled, and router policy cost_aware."
echo "   Then press enter to run the cost-aware benchmark."
read -r
python -m cost_aware_router.benchmark \
  --label qwen25_cost_aware_prefill_sensitive \
  --requests "${REQUESTS}" \
  --concurrency "${CONCURRENCY}" \
  --prefix-tokens "${PREFIX_TOKENS}" \
  --prefix-groups "${PREFIX_GROUPS}" \
  --suffix-tokens "${SUFFIX_TOKENS}" \
  --max-tokens "${MAX_TOKENS}" \
  --warmup-requests "${WARMUP_REQUESTS}" \
  --warmup-concurrency "${WARMUP_CONCURRENCY}" \
  --timeout "${TIMEOUT}" \
  --output-dir "${OUTPUT_DIR}"

echo "2) Restart vLLM with VLLM_ENABLE_PREFIX_CACHING=0, adapters with ADAPTER_ENABLE_PREFIX_CACHE=0,"
echo "   and router policy prefill_scratch. Then press enter to run the scratch baseline."
read -r
python -m cost_aware_router.benchmark \
  --label qwen25_prefill_scratch_prefill_sensitive \
  --requests "${REQUESTS}" \
  --concurrency "${CONCURRENCY}" \
  --prefix-tokens "${PREFIX_TOKENS}" \
  --prefix-groups "${PREFIX_GROUPS}" \
  --suffix-tokens "${SUFFIX_TOKENS}" \
  --max-tokens "${MAX_TOKENS}" \
  --warmup-requests "${WARMUP_REQUESTS}" \
  --warmup-concurrency "${WARMUP_CONCURRENCY}" \
  --timeout "${TIMEOUT}" \
  --output-dir "${OUTPUT_DIR}"

python -m cost_aware_router.analyze --results-dir "${OUTPUT_DIR}" --output "${OUTPUT_DIR}/cost_aware_vs_scratch.png"

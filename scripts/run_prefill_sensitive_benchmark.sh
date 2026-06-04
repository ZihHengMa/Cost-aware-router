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

run_policy() {
  local label="$1"
  python -m cost_aware_router.benchmark \
    --label "${label}" \
    --requests "${REQUESTS}" \
    --concurrency "${CONCURRENCY}" \
    --prefix-tokens "${PREFIX_TOKENS}" \
    --prefix-groups "${PREFIX_GROUPS}" \
    --suffix-tokens "${SUFFIX_TOKENS}" \
    --max-tokens "${MAX_TOKENS}" \
    --warmup-requests "${WARMUP_REQUESTS}" \
    --warmup-concurrency "${WARMUP_CONCURRENCY}" \
    --warmup-max-tokens 1 \
    --timeout "${TIMEOUT}" \
    --output-dir "${OUTPUT_DIR}"
}

echo "Use this after starting vLLM servers/adapters and the router for the desired policy."
echo "Current workload: prefix=${PREFIX_TOKENS}, groups=${PREFIX_GROUPS}, warmup=${WARMUP_REQUESTS}, max_tokens=${MAX_TOKENS}."
echo "Press enter to start benchmark label: ${1:-qwen25_cost_aware_prefill_sensitive}"
read -r

run_policy "${1:-qwen25_cost_aware_prefill_sensitive}"

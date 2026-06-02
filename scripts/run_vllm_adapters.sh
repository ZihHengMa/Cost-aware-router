#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${SERVED_MODEL_NAME:-Qwen2.5-7B-Instruct}"
ADAPTER_ENABLE_PREFIX_CACHE="${ADAPTER_ENABLE_PREFIX_CACHE:-1}"

ADAPTER_CACHE_ARGS=()
if [ "${ADAPTER_ENABLE_PREFIX_CACHE}" != "1" ]; then
  ADAPTER_CACHE_ARGS+=(--disable-prefix-cache)
fi

python -m cost_aware_router.worker \
  --mode vllm \
  --worker-id worker-0 \
  --port 8100 \
  --backend-url http://127.0.0.1:8200 \
  --model "${MODEL_NAME}" \
  "${ADAPTER_CACHE_ARGS[@]}" &

python -m cost_aware_router.worker \
  --mode vllm \
  --worker-id worker-1 \
  --port 8101 \
  --backend-url http://127.0.0.1:8201 \
  --model "${MODEL_NAME}" \
  "${ADAPTER_CACHE_ARGS[@]}" &

wait

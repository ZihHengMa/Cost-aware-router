#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-${MODEL_PATH:-}}"
if [ -z "${MODEL_PATH}" ]; then
  echo "Usage: $0 MODEL_PATH" >&2
  echo "Example: $0 /mnt/data1/llm_team/Qwen2.5-7B-Instruct" >&2
  exit 1
fi

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "${MODEL_PATH}")}"
HOST="${VLLM_HOST:-127.0.0.1}"
PORT_0="${VLLM_PORT_0:-8200}"
PORT_1="${VLLM_PORT_1:-8201}"
GPU_0="${GPU_0:-0}"
GPU_1="${GPU_1:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-1}"

if ! command -v vllm >/dev/null 2>&1; then
  echo "vllm command not found. Install vLLM in this environment first, for example: pip install vllm" >&2
  exit 1
fi

export LD_LIBRARY_PATH="/usr/local/cuda-12.6/targets/x86_64-linux/lib:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

PREFIX_CACHE_ARGS=()
if [ "${VLLM_ENABLE_PREFIX_CACHING}" = "1" ]; then
  PREFIX_CACHE_ARGS+=(--enable-prefix-caching)
else
  PREFIX_CACHE_ARGS+=(--no-enable-prefix-caching)
fi

CUDA_VISIBLE_DEVICES="${GPU_0}" vllm serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT_0}" \
  "${PREFIX_CACHE_ARGS[@]}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  ${EXTRA_VLLM_ARGS} &

CUDA_VISIBLE_DEVICES="${GPU_1}" vllm serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT_1}" \
  "${PREFIX_CACHE_ARGS[@]}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  ${EXTRA_VLLM_ARGS} &

wait

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [ -d ".venv/bin" ]; then
  export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
fi

MODEL_PATH="${1:-${MODEL_PATH:-}}"
if [ -z "${MODEL_PATH}" ]; then
  echo "Usage: $0 MODEL_PATH" >&2
  echo "Example: COST_AWARE_ROUTER_ARGS='--queue-weight 128 --cache-hit-bonus 0 --locality-queue-slack 1 --locality-threshold 0.9' $0 /mnt/data1/llm_team/Qwen2.5-7B-Instruct" >&2
  exit 1
fi

PYTHON="${PYTHON:-python}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "${MODEL_PATH}")}"
export SERVED_MODEL_NAME

POLICIES="${POLICIES:-round_robin least_queue cache_aware cost_aware}"
SEEDS="${SEEDS:-0,1,2,3}"
OUTPUT_DIR="${OUTPUT_DIR:-results/full_restart_tuned}"
METADATA_DIR="${METADATA_DIR:-data/full_restart_tuned}"
PLOT_OUTPUT="${PLOT_OUTPUT:-${OUTPUT_DIR}/full_restart_summary.png}"
RESUME="${RESUME:-0}"
SKIP_ANALYZE="${SKIP_ANALYZE:-0}"
KILL_EXISTING="${KILL_EXISTING:-1}"

COST_AWARE_ROUTER_ARGS="${COST_AWARE_ROUTER_ARGS:-${TUNED_COST_AWARE_ARGS:-}}"
if [ -z "${COST_AWARE_ROUTER_ARGS}" ]; then
  echo "COST_AWARE_ROUTER_ARGS is empty; cost_aware will use default router settings." >&2
fi

ROUTER_HOST="${ROUTER_HOST:-127.0.0.1}"
ROUTER_PORT="${ROUTER_PORT:-8000}"
ROUTER_URL="${ROUTER_URL:-http://${ROUTER_HOST}:${ROUTER_PORT}}"
WORKER_0="${WORKER_0:-http://127.0.0.1:8100}"
WORKER_1="${WORKER_1:-http://127.0.0.1:8101}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT_0="${VLLM_PORT_0:-8200}"
VLLM_PORT_1="${VLLM_PORT_1:-8201}"
VLLM_URL_0="http://${VLLM_HOST}:${VLLM_PORT_0}"
VLLM_URL_1="http://${VLLM_HOST}:${VLLM_PORT_1}"

VLLM_START_TIMEOUT="${VLLM_START_TIMEOUT:-900}"
ADAPTER_START_TIMEOUT="${ADAPTER_START_TIMEOUT:-90}"
ROUTER_START_TIMEOUT="${ROUTER_START_TIMEOUT:-60}"
SHUTDOWN_WAIT="${SHUTDOWN_WAIT:-5}"

WORKLOAD="${WORKLOAD:-realistic}"
REQUESTS="${REQUESTS:-300}"
CONCURRENCY="${CONCURRENCY:-16}"
PREFIX_TOKENS="${PREFIX_TOKENS:-512}"
PREFIX_GROUPS="${PREFIX_GROUPS:-20}"
HOT_PREFIX_GROUPS="${HOT_PREFIX_GROUPS:-2}"
HOT_SHARE="${HOT_SHARE:-0.85}"
BURST_SIZE="${BURST_SIZE:-4}"
SUFFIX_TOKENS="${SUFFIX_TOKENS:-64}"
MAX_TOKENS="${MAX_TOKENS:-64}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-40}"
WARMUP_CONCURRENCY="${WARMUP_CONCURRENCY:-2}"
WARMUP_MAX_TOKENS="${WARMUP_MAX_TOKENS:-1}"
TIMEOUT="${TIMEOUT:-900}"

VLLM_PID=""
ADAPTER_PID=""
ROUTER_PID=""

mkdir -p "${OUTPUT_DIR}" "${METADATA_DIR}"

split_csv() {
  local raw="$1"
  echo "${raw//,/ }"
}

http_ready() {
  local url="$1"
  "${PYTHON}" - "$url" <<'PY'
import sys
import urllib.request

try:
    with urllib.request.urlopen(sys.argv[1], timeout=2) as resp:
        raise SystemExit(0 if 200 <= resp.status < 500 else 1)
except Exception:
    raise SystemExit(1)
PY
}

wait_http() {
  local url="$1"
  local name="$2"
  local timeout_s="$3"
  "${PYTHON}" - "$url" "$name" "$timeout_s" <<'PY'
import sys
import time
import urllib.request

url = sys.argv[1]
name = sys.argv[2]
timeout_s = float(sys.argv[3])
deadline = time.monotonic() + timeout_s
last_error = None

while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            if 200 <= resp.status < 500:
                print(f"{name} is ready: {url}", flush=True)
                raise SystemExit(0)
    except Exception as exc:
        last_error = exc
    time.sleep(2)

print(f"Timed out waiting for {name}: {url}", file=sys.stderr)
if last_error is not None:
    print(f"Last error: {last_error}", file=sys.stderr)
raise SystemExit(1)
PY
}

stop_group() {
  local pid="$1"
  if [ -z "${pid}" ]; then
    return
  fi
  if kill -0 "${pid}" >/dev/null 2>&1; then
    kill -TERM "-${pid}" >/dev/null 2>&1 || kill -TERM "${pid}" >/dev/null 2>&1 || true
    sleep "${SHUTDOWN_WAIT}"
  fi
  if kill -0 "${pid}" >/dev/null 2>&1; then
    kill -KILL "-${pid}" >/dev/null 2>&1 || kill -KILL "${pid}" >/dev/null 2>&1 || true
  fi
  wait "${pid}" >/dev/null 2>&1 || true
}

kill_port() {
  local port="$1"
  if [ "${KILL_EXISTING}" != "1" ]; then
    return
  fi
  if command -v fuser >/dev/null 2>&1; then
    fuser -k -TERM "${port}/tcp" >/dev/null 2>&1 || true
    sleep 1
    fuser -k -KILL "${port}/tcp" >/dev/null 2>&1 || true
  elif command -v lsof >/dev/null 2>&1; then
    local pids
    pids="$(lsof -ti "tcp:${port}" || true)"
    if [ -n "${pids}" ]; then
      kill -TERM ${pids} >/dev/null 2>&1 || true
      sleep 1
      kill -KILL ${pids} >/dev/null 2>&1 || true
    fi
  fi
}

stop_all() {
  stop_group "${ROUTER_PID}"
  ROUTER_PID=""
  stop_group "${ADAPTER_PID}"
  ADAPTER_PID=""
  stop_group "${VLLM_PID}"
  VLLM_PID=""

  kill_port "${ROUTER_PORT}"
  kill_port 8100
  kill_port 8101
  kill_port "${VLLM_PORT_0}"
  kill_port "${VLLM_PORT_1}"
}

cleanup() {
  stop_all
}
trap cleanup EXIT INT TERM

start_vllm() {
  local label="$1"
  local log_path="${OUTPUT_DIR}/${label}_vllm.log"

  echo "Starting vLLM for ${label}"
  setsid bash "${SCRIPT_DIR}/run_vllm_servers.sh" "${MODEL_PATH}" > "${log_path}" 2>&1 &
  VLLM_PID="$!"

  if ! wait_http "${VLLM_URL_0}/v1/models" "vLLM worker-0" "${VLLM_START_TIMEOUT}"; then
    echo "vLLM log tail (${log_path}):" >&2
    tail -n 120 "${log_path}" >&2 || true
    return 1
  fi
  if ! wait_http "${VLLM_URL_1}/v1/models" "vLLM worker-1" "${VLLM_START_TIMEOUT}"; then
    echo "vLLM log tail (${log_path}):" >&2
    tail -n 120 "${log_path}" >&2 || true
    return 1
  fi
}

start_adapters() {
  local label="$1"
  local log_path="${OUTPUT_DIR}/${label}_adapters.log"

  echo "Starting adapters for ${label}"
  setsid bash "${SCRIPT_DIR}/run_vllm_adapters.sh" > "${log_path}" 2>&1 &
  ADAPTER_PID="$!"

  if ! wait_http "${WORKER_0}/state" "adapter worker-0" "${ADAPTER_START_TIMEOUT}"; then
    echo "Adapter log tail (${log_path}):" >&2
    tail -n 120 "${log_path}" >&2 || true
    return 1
  fi
  if ! wait_http "${WORKER_1}/state" "adapter worker-1" "${ADAPTER_START_TIMEOUT}"; then
    echo "Adapter log tail (${log_path}):" >&2
    tail -n 120 "${log_path}" >&2 || true
    return 1
  fi
}

policy_label() {
  case "$1" in
    round_robin) echo "rr" ;;
    cost_aware) echo "cost_aware_tuned" ;;
    *) echo "$1" ;;
  esac
}

start_router() {
  local policy="$1"
  local label="$2"
  local metadata_db="${METADATA_DIR}/${label}.sqlite"
  local log_path="${OUTPUT_DIR}/${label}_router.log"
  local -a cost_args=()

  if [ "${policy}" = "cost_aware" ] && [ -n "${COST_AWARE_ROUTER_ARGS}" ]; then
    read -r -a cost_args <<< "${COST_AWARE_ROUTER_ARGS}"
  fi

  rm -f "${metadata_db}"
  echo "Starting router policy=${policy} label=${label}"
  setsid "${PYTHON}" -m cost_aware_router.router \
    --host "${ROUTER_HOST}" \
    --port "${ROUTER_PORT}" \
    --policy "${policy}" \
    --metadata-db "${metadata_db}" \
    "${cost_args[@]}" \
    --worker "${WORKER_0}" \
    --worker "${WORKER_1}" \
    > "${log_path}" 2>&1 &
  ROUTER_PID="$!"

  if ! wait_http "${ROUTER_URL}/state" "router ${label}" "${ROUTER_START_TIMEOUT}"; then
    echo "Router log tail (${log_path}):" >&2
    tail -n 120 "${log_path}" >&2 || true
    return 1
  fi
}

run_benchmark() {
  local label="$1"
  local seed="$2"
  local log_path="${OUTPUT_DIR}/${label}_benchmark.log"

  echo "Running benchmark label=${label} seed=${seed}"
  "${PYTHON}" -m cost_aware_router.benchmark \
    --router-url "${ROUTER_URL}" \
    --label "${label}" \
    --workload "${WORKLOAD}" \
    --requests "${REQUESTS}" \
    --concurrency "${CONCURRENCY}" \
    --prefix-tokens "${PREFIX_TOKENS}" \
    --prefix-groups "${PREFIX_GROUPS}" \
    --hot-prefix-groups "${HOT_PREFIX_GROUPS}" \
    --hot-share "${HOT_SHARE}" \
    --burst-size "${BURST_SIZE}" \
    --suffix-tokens "${SUFFIX_TOKENS}" \
    --max-tokens "${MAX_TOKENS}" \
    --warmup-requests "${WARMUP_REQUESTS}" \
    --warmup-concurrency "${WARMUP_CONCURRENCY}" \
    --warmup-max-tokens "${WARMUP_MAX_TOKENS}" \
    --timeout "${TIMEOUT}" \
    --seed "${seed}" \
    --output-dir "${OUTPUT_DIR}" \
    --worker-count 2 \
    > "${log_path}" 2>&1
}

cat > "${OUTPUT_DIR}/parameters.txt" <<EOF
model_path=${MODEL_PATH}
served_model_name=${SERVED_MODEL_NAME}
policies=${POLICIES}
seeds=${SEEDS}
cost_aware_router_args=${COST_AWARE_ROUTER_ARGS}
workload=${WORKLOAD}
requests=${REQUESTS}
concurrency=${CONCURRENCY}
prefix_tokens=${PREFIX_TOKENS}
prefix_groups=${PREFIX_GROUPS}
hot_prefix_groups=${HOT_PREFIX_GROUPS}
hot_share=${HOT_SHARE}
burst_size=${BURST_SIZE}
suffix_tokens=${SUFFIX_TOKENS}
max_tokens=${MAX_TOKENS}
warmup_requests=${WARMUP_REQUESTS}
warmup_concurrency=${WARMUP_CONCURRENCY}
timeout=${TIMEOUT}
EOF

if [ "${KILL_EXISTING}" = "1" ]; then
  stop_all
fi

for seed in $(split_csv "${SEEDS}"); do
  for policy in ${POLICIES}; do
    base_label="$(policy_label "${policy}")"
    label="${base_label}_seed${seed}"
    summary_path="${OUTPUT_DIR}/${label}_summary.csv"

    if [ "${RESUME}" = "1" ] && [ -f "${summary_path}" ]; then
      echo "Skipping existing summary: ${summary_path}"
      continue
    fi

    echo
    echo "=== Full restart run: policy=${policy} seed=${seed} label=${label} ==="
    stop_all
    start_vllm "${label}"
    start_adapters "${label}"
    start_router "${policy}" "${label}"
    run_benchmark "${label}" "${seed}"
    stop_all
  done
done

if [ "${SKIP_ANALYZE}" != "1" ]; then
  "${PYTHON}" -m cost_aware_router.analyze \
    --results-dir "${OUTPUT_DIR}" \
    --output "${PLOT_OUTPUT}"
fi

echo "Full-restart validation complete: ${OUTPUT_DIR}"

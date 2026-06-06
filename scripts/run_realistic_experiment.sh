#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [ -z "${PYTHON:-}" ]; then
  if [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
  else
    PYTHON="python"
  fi
fi

WORKER_0="${WORKER_0:-http://127.0.0.1:8100}"
WORKER_1="${WORKER_1:-http://127.0.0.1:8101}"
ROUTER_HOST="${ROUTER_HOST:-127.0.0.1}"
ROUTER_PORT="${ROUTER_PORT:-8000}"
ROUTER_URL="${ROUTER_URL:-http://${ROUTER_HOST}:${ROUTER_PORT}}"
ROUTER_START_TIMEOUT="${ROUTER_START_TIMEOUT:-60}"

OUTPUT_DIR="${OUTPUT_DIR:-results/realcase_auto}"
METADATA_DIR="${METADATA_DIR:-data/experiment_metadata}"
PLOT_OUTPUT="${PLOT_OUTPUT:-${OUTPUT_DIR}/real_case_summary.png}"
POLICIES="${POLICIES:-round_robin least_queue cache_aware cost_aware}"

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
SEED="${SEED:-0}"
WORKER_COUNT="${WORKER_COUNT:-2}"

ROUTER_ARGS="${ROUTER_ARGS:-}"
COST_AWARE_ROUTER_ARGS="${COST_AWARE_ROUTER_ARGS:-}"
RESET_METADATA="${RESET_METADATA:-1}"
SKIP_ANALYZE="${SKIP_ANALYZE:-0}"

ROUTER_PID=""

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
                print(f"{name} is ready: {url}")
                raise SystemExit(0)
    except Exception as exc:
        last_error = exc
    time.sleep(1)

print(f"Timed out waiting for {name}: {url}", file=sys.stderr)
if last_error is not None:
    print(f"Last error: {last_error}", file=sys.stderr)
raise SystemExit(1)
PY
}

stop_router() {
  if [ -n "${ROUTER_PID}" ] && kill -0 "${ROUTER_PID}" >/dev/null 2>&1; then
    kill "${ROUTER_PID}" >/dev/null 2>&1 || true
    wait "${ROUTER_PID}" >/dev/null 2>&1 || true
  fi
  ROUTER_PID=""
}

cleanup() {
  stop_router
}
trap cleanup EXIT INT TERM

policy_label() {
  case "$1" in
    round_robin) echo "rr" ;;
    *) echo "$1" ;;
  esac
}

start_router() {
  local policy="$1"
  local label="$2"
  local metadata_db="${METADATA_DIR}/${label}.sqlite"
  local log_path="${OUTPUT_DIR}/${label}_router.log"
  local -a common_args=()
  local -a cost_args=()

  if [ -n "${ROUTER_ARGS}" ]; then
    read -r -a common_args <<< "${ROUTER_ARGS}"
  fi
  if [ "${policy}" = "cost_aware" ] && [ -n "${COST_AWARE_ROUTER_ARGS}" ]; then
    read -r -a cost_args <<< "${COST_AWARE_ROUTER_ARGS}"
  fi

  if [ "${RESET_METADATA}" = "1" ]; then
    rm -f "${metadata_db}"
  fi

  echo "Starting router policy=${policy} metadata=${metadata_db}"
  "${PYTHON}" -m cost_aware_router.router \
    --host "${ROUTER_HOST}" \
    --port "${ROUTER_PORT}" \
    --policy "${policy}" \
    --metadata-db "${metadata_db}" \
    "${common_args[@]}" \
    "${cost_args[@]}" \
    --worker "${WORKER_0}" \
    --worker "${WORKER_1}" \
    > "${log_path}" 2>&1 &
  ROUTER_PID="$!"

  if ! wait_http "${ROUTER_URL}/state" "router ${policy}" "${ROUTER_START_TIMEOUT}"; then
    echo "Router log (${log_path}):" >&2
    tail -n 80 "${log_path}" >&2 || true
    return 1
  fi
}

run_benchmark() {
  local policy="$1"
  local label="$2"

  echo "Running benchmark label=${label}"
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
    --output-dir "${OUTPUT_DIR}" \
    --seed "${SEED}" \
    --worker-count "${WORKER_COUNT}"

  echo "Finished policy=${policy}"
}

mkdir -p "${OUTPUT_DIR}" "${METADATA_DIR}"

if http_ready "${ROUTER_URL}/state"; then
  echo "A router is already running at ${ROUTER_URL}." >&2
  echo "Stop it first, or set ROUTER_PORT to a free port." >&2
  exit 1
fi

wait_http "${WORKER_0}/state" "worker-0 adapter" 10
wait_http "${WORKER_1}/state" "worker-1 adapter" 10

cat > "${OUTPUT_DIR}/parameters.txt" <<EOF
  --workload ${WORKLOAD} \\
  --requests ${REQUESTS} \\
  --concurrency ${CONCURRENCY} \\
  --prefix-tokens ${PREFIX_TOKENS} \\
  --prefix-groups ${PREFIX_GROUPS} \\
  --hot-prefix-groups ${HOT_PREFIX_GROUPS} \\
  --hot-share ${HOT_SHARE} \\
  --burst-size ${BURST_SIZE} \\
  --suffix-tokens ${SUFFIX_TOKENS} \\
  --max-tokens ${MAX_TOKENS} \\
  --warmup-requests ${WARMUP_REQUESTS} \\
  --warmup-concurrency ${WARMUP_CONCURRENCY} \\
  --timeout ${TIMEOUT} \\
  --seed ${SEED} \\
  --output-dir ${OUTPUT_DIR}
EOF

for policy in ${POLICIES}; do
  label="$(policy_label "${policy}")"
  start_router "${policy}" "${label}"
  run_benchmark "${policy}" "${label}"
  stop_router
done

if [ "${SKIP_ANALYZE}" != "1" ]; then
  "${PYTHON}" -m cost_aware_router.analyze \
    --results-dir "${OUTPUT_DIR}" \
    --output "${PLOT_OUTPUT}"
fi

echo "Experiment complete: ${OUTPUT_DIR}"

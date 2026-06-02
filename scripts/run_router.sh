#!/usr/bin/env bash
set -euo pipefail

POLICY="${1:-cost_aware}"
LMCACHE_FLAG="${2:-}"

python -m cost_aware_router.router \
  --port 8000 \
  --policy "${POLICY}" \
  ${LMCACHE_FLAG} \
  --worker http://127.0.0.1:8100 \
  --worker http://127.0.0.1:8101

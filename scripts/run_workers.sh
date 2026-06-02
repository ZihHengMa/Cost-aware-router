#!/usr/bin/env bash
set -euo pipefail

python -m cost_aware_router.worker --worker-id worker-0 --port 8100 &
python -m cost_aware_router.worker --worker-id worker-1 --port 8101 &
wait

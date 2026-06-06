#!/usr/bin/env bash
set -euo pipefail
./scripts/run_router.sh cost_aware \
  --queue-weight 64.0 \
  --cache-hit-bonus 1.0 \
  --locality-queue-slack 1 \
  --locality-threshold 0.9

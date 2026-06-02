#!/usr/bin/env bash
set -euo pipefail

curl -s http://127.0.0.1:8200/v1/models
printf '\n'
curl -s http://127.0.0.1:8201/v1/models
printf '\n'

# Cost-aware Routing for vLLM Prefix Caching

This final project compares routing policies for two vLLM-style workers behind one Python/FastAPI router.
It focuses on workloads with repeated prompt prefixes, where routing can trade off queueing delay against KV-prefix cache reuse.

## Features

- 2 real vLLM workers, one per GPU
- 1 FastAPI router
- token-based prefix hashing
- round-robin baseline
- least-queue baseline
- prefill-from-scratch baseline
- cache-aware routing baseline
- cost-aware routing
- optional LMCache-style shared backend mode
- partial prefix matching
- repeated-prefix benchmark workload
- TTFT, latency, cache-hit, and load-imbalance evaluation

This project is intended to run with real vLLM OpenAI-compatible servers using `/mnt/data1/llm_team/Qwen2.5-7B-Instruct`.

## Architecture

```text
benchmark client
      |
      v
FastAPI router :8000
  |      |
  v      v
adapter-0 :8100     adapter-1 :8101
      |                    |
      v                    v
vLLM GPU-0 :8200      vLLM GPU-1 :8201
```

The router also maintains a SQLite request metadata table at `data/router_metadata.sqlite` by default.
Each completed request is inserted into `request_metadata` with its prompt, prompt hash, prefix hash, selected worker, routing policy, estimated prefix hit, queue depth, TTFT, latency, and cache-hit result.

Each request prompt is split into deterministic routing tokens. The router computes prefix-hash candidates against its cache view while vLLM serves the real model:

- `round_robin`: alternate workers.
- `least_queue`: choose the worker with the smallest observed queue.
- `cache_aware`: choose the worker with the longest prefix-cache match.
- `cost_aware`: minimize `queue_weight * queue_depth + prefill_weight * uncached_tokens - cache_hit_bonus * hit_tokens`.
- `prefill_scratch`: choose the least-queued worker but force every request to pay full prefill with zero prefix-cache reuse.

Partial prefix matching is counted when the worker can reuse some, but not all, prompt tokens.
`--lmcache-shared` makes the router model a shared cache backend, similar to the high-level effect of LMCache.

## Setup

```bash
cd /mnt/data1/ponymichael/distributedsystem
python -m venv .venv
source .venv/bin/activate
pip install -e .
chmod +x scripts/*.sh
```

Install the vLLM stack for this machine:

```bash
source .venv/bin/activate
uv pip install --python .venv/bin/python --reinstall \
  'https://github.com/vllm-project/vllm/releases/download/v0.10.0/vllm-0.10.0+cu126-cp38-abi3-manylinux1_x86_64.whl' \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  --index-strategy unsafe-best-match

uv pip install --python .venv/bin/python --reinstall \
  'transformers==4.53.2' 'tokenizers>=0.21.1,<0.22' \
  'huggingface-hub>=0.33.0,<1.0' 'numpy>=2.0,<2.3'
```

## Run One Experiment

Terminal 1, start two vLLM servers:

```bash
source .venv/bin/activate
VLLM_ENABLE_PREFIX_CACHING=1 ./scripts/run_vllm_servers.sh
```

Terminal 2, start the router-compatible adapters:

```bash
source .venv/bin/activate
ADAPTER_ENABLE_PREFIX_CACHE=1 ./scripts/run_vllm_adapters.sh
```

Terminal 3, start the router:

```bash
source .venv/bin/activate
./scripts/run_router.sh cost_aware
```

Terminal 4, run the benchmark:

```bash
source .venv/bin/activate
python -m cost_aware_router.benchmark --label qwen25_cost_aware --requests 80 --concurrency 8 --max-tokens 32
```

Results are written to:

- `results/qwen25_cost_aware_raw.csv`
- `results/qwen25_cost_aware_summary.csv`

Router request metadata is stored in:

- `data/router_metadata.sqlite`

Inspect recent requests:

```bash
curl 'http://127.0.0.1:8000/requests?limit=10'
sqlite3 data/router_metadata.sqlite 'select request_id, worker_id, route_policy, prompt_tokens, cache_hit_tokens, ttft_ms from request_metadata order by created_at desc limit 10;'
```

## Compare Policies

Keep the two vLLM servers and adapters running, then restart the router for each policy:

```bash
./scripts/run_router.sh round_robin
python -m cost_aware_router.benchmark --label qwen25_round_robin --requests 80 --concurrency 8 --max-tokens 32

./scripts/run_router.sh least_queue
python -m cost_aware_router.benchmark --label qwen25_least_queue --requests 80 --concurrency 8 --max-tokens 32

./scripts/run_router.sh cache_aware
python -m cost_aware_router.benchmark --label qwen25_cache_aware --requests 80 --concurrency 8 --max-tokens 32

./scripts/run_router.sh cost_aware
python -m cost_aware_router.benchmark --label qwen25_cost_aware --requests 80 --concurrency 8 --max-tokens 32

./scripts/run_router.sh prefill_scratch
python -m cost_aware_router.benchmark --label qwen25_prefill_scratch --requests 80 --concurrency 8 --max-tokens 32
```

Then plot summaries:

```bash
python -m cost_aware_router.analyze --results-dir results
```

## Scenario Where Cost-aware Should Win

Cost-aware routing is most useful when cache locality and load balance are both important.
The target situation is a repeated-prefix workload with long prompts, short generation, and enough concurrency to create queue pressure.

Example workload:

```text
Many requests share a long document/system prefix:
  "Here is a 1,000-token paper/legal contract/code file..."

Each request has a short different suffix:
  "Question 1..."
  "Question 2..."
  "Question 3..."
```

This creates a scheduling tradeoff:

- `round_robin` balances traffic, but often sends a repeated prefix to a worker without the best KV cache.
- `least_queue` avoids busy workers, but may discard prefix-cache locality.
- `cache_aware` maximizes prefix reuse, but can overload the worker with the hottest cached prefix.
- `cost_aware` can choose a partial cache hit on a less busy worker when that has lower total cost than waiting behind a long queue.

Use a workload where prefill dominates decode:

```bash
python -m cost_aware_router.benchmark \
  --label cost_aware \
  --requests 160 \
  --concurrency 16 \
  --prefix-tokens 512 \
  --suffix-tokens 8 \
  --max-tokens 8
```

Run the same benchmark for all policies:

```bash
./scripts/run_router.sh round_robin
python -m cost_aware_router.benchmark --label rr_win_case --requests 160 --concurrency 16 --prefix-tokens 512 --suffix-tokens 8 --max-tokens 8

./scripts/run_router.sh least_queue
python -m cost_aware_router.benchmark --label least_queue_win_case --requests 160 --concurrency 16 --prefix-tokens 512 --suffix-tokens 8 --max-tokens 8

./scripts/run_router.sh cache_aware
python -m cost_aware_router.benchmark --label cache_aware_win_case --requests 160 --concurrency 16 --prefix-tokens 512 --suffix-tokens 8 --max-tokens 8

./scripts/run_router.sh cost_aware
python -m cost_aware_router.benchmark --label cost_aware_win_case --requests 160 --concurrency 16 --prefix-tokens 512 --suffix-tokens 8 --max-tokens 8
```

For the cleanest comparison, restart the router between policies.
For real vLLM experiments, also restart the vLLM workers if you want each policy to start from the same cold-cache condition.
Use separate metadata DB files with `--metadata-db` if you want per-policy request tables.

Then plot:

```bash
python -m cost_aware_router.analyze --results-dir results --output results/win_case_summary.png
```

For the report, the expected pattern is:

```text
round_robin:
  balanced load, but weaker cache-hit rate

least_queue:
  low queue depth, but lower cache reuse

cache_aware:
  high cache-hit rate, but high load imbalance and worse tail latency

cost_aware:
  good cache reuse without overloading one worker
  best or near-best TTFT p95 / latency p95
```

Use `request_metadata` to support the explanation:

```bash
sqlite3 data/router_metadata.sqlite '
select
  route_policy,
  worker_id,
  count(*) as requests,
  avg(estimated_prefix_hit_tokens) as avg_est_hit,
  avg(cache_hit_tokens) as avg_hit,
  avg(ttft_ms) as avg_ttft,
  avg(latency_ms) as avg_latency
from request_metadata
group by route_policy, worker_id
order by route_policy, worker_id;
'
```

If the `sqlite3` command is not installed, use Python:

```bash
python - <<'PY'
import sqlite3

conn = sqlite3.connect("data/router_metadata.sqlite")
for row in conn.execute("""
select
  route_policy,
  worker_id,
  count(*) as requests,
  avg(estimated_prefix_hit_tokens) as avg_est_hit,
  avg(cache_hit_tokens) as avg_hit,
  avg(ttft_ms) as avg_ttft,
  avg(latency_ms) as avg_latency
from request_metadata
group by route_policy, worker_id
order by route_policy, worker_id
"""):
    print(row)
PY
```

If cost-aware still does not win, the likely reason is that the cost weights do not match the actual vLLM timing on the machine.
In that case, present the result as calibration evidence: cost-aware routing is a framework, but it needs the queue cost and prefill cost weights to reflect the serving environment.

## LMCache Shared Backend Experiment

Run the router with the shared-cache flag:

```bash
./scripts/run_router.sh cost_aware --lmcache-shared
python -m cost_aware_router.benchmark --label cost_aware_lmcache
```

In the report, treat this as a model of an LMCache-style shared backend: cached prefixes are no longer purely local to one worker, reducing the penalty of routing a repeated prefix to a different worker.

## Cost-aware vs Prefill-from-scratch Baseline

Use this comparison when you want to show the benefit of your cost-aware infrastructure against a baseline where every request must prefill from scratch.

Run the two experiments separately:

Cost-aware with prefix caching:

```bash
VLLM_ENABLE_PREFIX_CACHING=1 ./scripts/run_vllm_servers.sh
ADAPTER_ENABLE_PREFIX_CACHE=1 ./scripts/run_vllm_adapters.sh
./scripts/run_router.sh cost_aware
python -m cost_aware_router.benchmark --label qwen25_cost_aware --requests 80 --concurrency 8 --max-tokens 32
```

Scratch baseline with prefix caching disabled:

```bash
VLLM_ENABLE_PREFIX_CACHING=0 ./scripts/run_vllm_servers.sh
ADAPTER_ENABLE_PREFIX_CACHE=0 ./scripts/run_vllm_adapters.sh
./scripts/run_router.sh prefill_scratch
python -m cost_aware_router.benchmark --label qwen25_prefill_scratch --requests 80 --concurrency 8 --max-tokens 32
```

Then compare:

```bash
python -m cost_aware_router.analyze --results-dir results --output results/qwen25_cost_aware_vs_scratch.png
```

In `request_metadata`, the scratch baseline should show `estimated_prefix_hit_tokens = 0`, `cache_hit_tokens = 0`, and `estimated_prefill_tokens = prompt_tokens`.

## vLLM on Two GPUs

The project runs two real vLLM instances using `/mnt/data1/llm_team/Qwen2.5-7B-Instruct`, one process per GPU.
The router still talks to worker adapters on ports `8100` and `8101`; each adapter forwards requests to one vLLM OpenAI-compatible server and measures actual streaming TTFT.
Use the setup commands above before starting the servers.

Terminal 1, start vLLM on GPU 0 and GPU 1:

```bash
source .venv/bin/activate
./scripts/run_vllm_servers.sh
```

This launches:

- GPU 0: `http://127.0.0.1:8200/v1`
- GPU 1: `http://127.0.0.1:8201/v1`

Terminal 2, after both vLLM servers finish loading:

```bash
source .venv/bin/activate
./scripts/check_vllm_servers.sh
./scripts/run_vllm_adapters.sh
```

Terminal 3, start the router:

```bash
source .venv/bin/activate
./scripts/run_router.sh cost_aware
```

Terminal 4, run the benchmark:

```bash
source .venv/bin/activate
python -m cost_aware_router.benchmark --label qwen25_7b_cost_aware --requests 80 --concurrency 8 --max-tokens 32
```

Useful overrides:

```bash
GPU_0=2 GPU_1=3 ./scripts/run_vllm_servers.sh
MAX_MODEL_LEN=4096 GPU_MEMORY_UTILIZATION=0.85 ./scripts/run_vllm_servers.sh
EXTRA_VLLM_ARGS="--trust-remote-code" ./scripts/run_vllm_servers.sh
```

`EXTRA_VLLM_ARGS` is passed directly to your installed `vllm serve`.
Check supported flags with:

```bash
vllm serve --help=all
```

If vLLM raises `ImportError: libcudart.so.13: cannot open shared object file`, the installed vLLM wheel is the wrong CUDA variant for this driver.
Reinstall the CUDA 12.6 wheel above. Do not fix this by adding CUDA 13 to `LD_LIBRARY_PATH`; that leads to `CUDA driver version is insufficient for CUDA runtime version` on this machine.

Sanity check:

```bash
vllm --version
python -c 'import torch, vllm, vllm._C; print(torch.__version__, torch.version.cuda, vllm.__version__)'
python -c 'import numpy, numba; print(numpy.__version__, numba.__version__)'
python -c 'from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained("/mnt/data1/llm_team/Qwen2.5-7B-Instruct"); print(type(t), hasattr(t, "all_special_tokens_extended"))'
```

## Suggested Report Outline

1. Problem: LLM serving uses expensive prompt prefill; prefix cache reuse lowers TTFT.
2. Baselines: round-robin, least-queue, cache-aware, and prefill-from-scratch.
3. Method: token-based prefix hashing and partial prefix matching.
4. Cost model: balance queue delay against uncached prefill tokens and cache-hit benefit.
5. LMCache discussion: shared cache lowers locality pressure but introduces backend overhead in real deployments.
6. Evaluation: repeated-prefix workload, TTFT, total latency, cache-hit rate, imbalance ratio.
7. Analysis: cache-aware can overload a hot worker; least-queue can miss cache locality; cost-aware should sit between them.

## Notes on Real vLLM Metrics

The adapter measures TTFT from the streamed `/v1/completions` response, so TTFT and total latency are real wall-clock measurements.
Cache-hit tokens are the router-side prefix-cache estimate used for policy comparison; vLLM's internal prefix-cache hit counters are not exposed through the OpenAI response.

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
- `cost_aware`: preserve near-best prefix locality unless the queue gap is large, then minimize `queue_weight * queue_depth + prefill_weight * uncached_tokens - cache_hit_bonus * hit_tokens`.
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
VLLM_ENABLE_PREFIX_CACHING=1 ./scripts/run_vllm_servers.sh /mnt/data1/llm_team/Qwen2.5-7B-Instruct
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
```

Run `prefill_scratch` separately after restarting vLLM with prefix caching disabled:

```bash
VLLM_ENABLE_PREFIX_CACHING=0 ./scripts/run_vllm_servers.sh /mnt/data1/llm_team/Qwen2.5-7B-Instruct
ADAPTER_ENABLE_PREFIX_CACHE=0 ./scripts/run_vllm_adapters.sh
./scripts/run_router.sh prefill_scratch
python -m cost_aware_router.benchmark --label qwen25_prefill_scratch --requests 80 --concurrency 8 --max-tokens 32
```

If `prefill_scratch` is run while vLLM was started with `VLLM_ENABLE_PREFIX_CACHING=1`, it is not a true scratch baseline because vLLM may still reuse its internal prefix cache.

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

The default cost-aware router is locality-guarded:

```text
queue_weight = 8.0
prefill_weight = 1.0
cache_hit_bonus = 2.0
locality_threshold = 0.90
locality_queue_slack = 2
```

This means cost-aware will not give up a near-best prefix match for only a tiny queue advantage.
It only trades cache locality for queue relief when the better-cache worker is meaningfully more loaded.

Use a workload where prefill dominates decode:

```bash
python -m cost_aware_router.benchmark \
  --label cost_aware \
  --requests 120 \
  --concurrency 8 \
  --prefix-tokens 2048 \
  --prefix-groups 2 \
  --suffix-tokens 8 \
  --max-tokens 1 \
  --warmup-requests 32 \
  --warmup-concurrency 1 \
  --timeout 900
```

The warmup phase is not included in the reported CSV summary.
It exists to put the repeated long prefixes into the vLLM prefix cache before measuring TTFT.
This is important because a cold-cache run mostly measures first-touch prefill, where every policy looks similar.

Run the same benchmark for all policies:

```bash
./scripts/run_router.sh round_robin
python -m cost_aware_router.benchmark --label rr_prefill_sensitive --requests 120 --concurrency 8 --prefix-tokens 2048 --prefix-groups 2 --suffix-tokens 8 --max-tokens 1 --warmup-requests 32 --warmup-concurrency 1 --timeout 900 --output-dir results/prefill_sensitive

./scripts/run_router.sh least_queue
python -m cost_aware_router.benchmark --label least_queue_prefill_sensitive --requests 120 --concurrency 8 --prefix-tokens 2048 --prefix-groups 2 --suffix-tokens 8 --max-tokens 1 --warmup-requests 32 --warmup-concurrency 1 --timeout 900 --output-dir results/prefill_sensitive

./scripts/run_router.sh cache_aware
python -m cost_aware_router.benchmark --label cache_aware_prefill_sensitive --requests 120 --concurrency 8 --prefix-tokens 2048 --prefix-groups 2 --suffix-tokens 8 --max-tokens 1 --warmup-requests 32 --warmup-concurrency 1 --timeout 900 --output-dir results/prefill_sensitive

./scripts/run_router.sh cost_aware
python -m cost_aware_router.benchmark --label cost_aware_prefill_sensitive --requests 120 --concurrency 8 --prefix-tokens 2048 --prefix-groups 2 --suffix-tokens 8 --max-tokens 1 --warmup-requests 32 --warmup-concurrency 1 --timeout 900 --output-dir results/prefill_sensitive
```

You can make cost-aware more cache-preserving with:

```bash
./scripts/run_router.sh cost_aware --queue-weight 4 --cache-hit-bonus 3 --locality-threshold 0.95 --locality-queue-slack 1
```

You can make it more load-balancing with:

```bash
./scripts/run_router.sh cost_aware --queue-weight 12 --cache-hit-bonus 1.5 --locality-queue-slack 3
```

For more realistic mixed traffic, start by reducing or removing `cache_hit_bonus`.
`prefill_weight * uncached_tokens` already gives a benefit to cache hits, so a large extra cache bonus can make cost-aware too sticky to one hot worker.
If you are willing to give up about 512 cached-prefix tokens once a worker is 4 active requests busier than another worker, use a queue weight around `512 / 4 = 128`:

```bash
./scripts/run_router.sh cost_aware --queue-weight 128 --cache-hit-bonus 0 --locality-queue-slack 1
```

To stress a more realistic hot/cold request mix and make load imbalance visible:

```bash
python -m cost_aware_router.benchmark \
  --label realistic_cost_aware \
  --workload realistic \
  --requests 300 \
  --concurrency 16 \
  --prefix-tokens 512 \
  --prefix-groups 20 \
  --hot-prefix-groups 2 \
  --hot-share 0.85 \
  --burst-size 4 \
  --suffix-tokens 64 \
  --max-tokens 64 \
  --warmup-requests 40 \
  --warmup-concurrency 2 \
  --timeout 900 \
  --output-dir results/realcase
```

To run the same realistic experiment automatically across all router policies, start vLLM and the worker adapters first, but do not start the router manually:

```bash
./scripts/run_realistic_experiment.sh
```

The script starts and stops one router per policy, writes per-policy metadata DBs under `data/experiment_metadata`, writes benchmark outputs under `results/realcase_auto`, and generates `results/realcase_auto/real_case_summary.png`.
You can override the workload or policy list with environment variables:

```bash
OUTPUT_DIR=results/realcase_seed1 SEED=1 ./scripts/run_realistic_experiment.sh
POLICIES="least_queue cache_aware cost_aware" ./scripts/run_realistic_experiment.sh
COST_AWARE_ROUTER_ARGS="--queue-weight 128 --cache-hit-bonus 0 --locality-queue-slack 1" ./scripts/run_realistic_experiment.sh
```

To automatically search for a stronger default cost-aware setting, use the tuner:

```bash
./scripts/tune_cost_aware_params.sh
```

The tuner starts one router at a time, runs baseline policies and cost-aware candidates across multiple seeds, then ranks candidates by `ttft_p95_ms`.
It writes:

```text
results/cost_aware_tuning/cost_aware_tuning_report.csv
results/cost_aware_tuning/best_cost_aware_command.sh
```

For a quicker first pass:

```bash
./scripts/tune_cost_aware_params.sh \
  --seeds 0,1 \
  --queue-weights 64,128 \
  --cache-hit-bonuses 0,0.5 \
  --locality-queue-slacks 1 \
  --locality-thresholds 0.9
```

For a wider search:

```bash
./scripts/tune_cost_aware_params.sh \
  --queue-weights 32,64,96,128,192,256 \
  --cache-hit-bonuses 0,0.25,0.5,1 \
  --locality-queue-slacks 1,2 \
  --locality-thresholds 0.85,0.9,0.95
```

Use the tuner to choose a candidate, then validate that candidate with full vLLM restarts in separate runs before reporting it as the final setting.
The tuner changes request suffix IDs between candidates to avoid exact-prompt cache leakage, while keeping the same prefix distribution for each seed.

To validate a tuned cost-aware setting with a full restart before every policy run:

```bash
COST_AWARE_ROUTER_ARGS="--queue-weight 128 --cache-hit-bonus 0 --locality-queue-slack 1 --locality-threshold 0.9" \
  ./scripts/validate_tuned_full_restart.sh /mnt/data1/llm_team/Qwen2.5-7B-Instruct
```

This script runs `round_robin`, `least_queue`, `cache_aware`, and tuned `cost_aware` over the same seeds, but restarts vLLM, the worker adapters, and the router before every policy/seed run.
It writes results to:

```text
results/full_restart_tuned/
data/full_restart_tuned/
```

You can shrink or change the run:

```bash
SEEDS=0,1 POLICIES="least_queue cache_aware cost_aware" \
  COST_AWARE_ROUTER_ARGS="--queue-weight 128 --cache-hit-bonus 0 --locality-queue-slack 1 --locality-threshold 0.9" \
  ./scripts/validate_tuned_full_restart.sh /mnt/data1/llm_team/Qwen2.5-7B-Instruct
```

By default the full-restart script stops existing processes on ports `8000`, `8100`, `8101`, `8200`, and `8201`.
Use it when those ports belong to this experiment.

For the cleanest comparison, restart the router between policies.
For real vLLM experiments, also restart the vLLM workers if you want each policy to start from the same cold-cache condition.
Use separate metadata DB files with `--metadata-db` if you want per-policy request tables.
If you do not restart the worker adapters between policies, exact prompt hits from a previous policy run can leak into the next result and make `cache_aware` or `round_robin` look better than they should.

Then plot:

```bash
python -m cost_aware_router.analyze --results-dir results/prefill_sensitive --output results/prefill_sensitive/win_case_summary.png
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

If cost-aware still does not win, tune `queue_weight`, `cache_hit_bonus`, `locality_threshold`, and `locality_queue_slack`.
For your current repeated-prefix experiment, start with the cache-preserving command above because it prevents cost-aware from choosing too many partial-prefix routes.

## LMCache Shared Backend Experiment

Run the router with the shared-cache flag:

```bash
./scripts/run_router.sh cost_aware --lmcache-shared
python -m cost_aware_router.benchmark --label cost_aware_lmcache
```

In the report, treat this as a model of an LMCache-style shared backend: cached prefixes are no longer purely local to one worker, reducing the penalty of routing a repeated prefix to a different worker.

## Cost-aware vs Prefill-from-scratch Baseline

Use this comparison when you want to show the benefit of your cost-aware infrastructure against a baseline where every request must prefill from scratch.
This experiment intentionally makes prefill dominate TTFT:

```text
long repeated prefix: 2048 routing tokens
few prefix groups: 2
short suffix: 8 routing tokens
short generation: 1 token
warmup: 32 requests, excluded from measured results
```

Run the two experiments separately:

Cost-aware with prefix caching:

```bash
VLLM_ENABLE_PREFIX_CACHING=1 ./scripts/run_vllm_servers.sh /mnt/data1/llm_team/Qwen2.5-7B-Instruct
ADAPTER_ENABLE_PREFIX_CACHE=1 ./scripts/run_vllm_adapters.sh
./scripts/run_router.sh cost_aware --queue-weight 4 --cache-hit-bonus 3 --locality-threshold 0.95 --locality-queue-slack 1
python -m cost_aware_router.benchmark \
  --label qwen25_cost_aware_prefill_sensitive \
  --requests 120 \
  --concurrency 8 \
  --prefix-tokens 2048 \
  --prefix-groups 2 \
  --suffix-tokens 8 \
  --max-tokens 1 \
  --warmup-requests 32 \
  --warmup-concurrency 1 \
  --timeout 900 \
  --output-dir results/prefill_sensitive
```

Scratch baseline with prefix caching disabled:

```bash
VLLM_ENABLE_PREFIX_CACHING=0 ./scripts/run_vllm_servers.sh /mnt/data1/llm_team/Qwen2.5-7B-Instruct
ADAPTER_ENABLE_PREFIX_CACHE=0 ./scripts/run_vllm_adapters.sh
./scripts/run_router.sh prefill_scratch
python -m cost_aware_router.benchmark \
  --label qwen25_prefill_scratch_prefill_sensitive \
  --requests 120 \
  --concurrency 8 \
  --prefix-tokens 2048 \
  --prefix-groups 2 \
  --suffix-tokens 8 \
  --max-tokens 1 \
  --warmup-requests 32 \
  --warmup-concurrency 1 \
  --timeout 900 \
  --output-dir results/prefill_sensitive
```

Then compare:

```bash
python -m cost_aware_router.analyze --results-dir results/prefill_sensitive --output results/prefill_sensitive/qwen25_cost_aware_vs_scratch.png
```

In `request_metadata`, the scratch baseline should show `estimated_prefix_hit_tokens = 0`, `cache_hit_tokens = 0`, and `estimated_prefill_tokens = prompt_tokens`.

## vLLM on Two GPUs

The project runs two real vLLM instances using the model path passed to `run_vllm_servers.sh`, one process per GPU.
The router still talks to worker adapters on ports `8100` and `8101`; each adapter forwards requests to one vLLM OpenAI-compatible server and measures actual streaming TTFT.
Use the setup commands above before starting the servers.

Terminal 1, start vLLM on GPU 0 and GPU 1:

```bash
source .venv/bin/activate
./scripts/run_vllm_servers.sh /mnt/data1/llm_team/Qwen2.5-7B-Instruct
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
GPU_0=2 GPU_1=3 ./scripts/run_vllm_servers.sh /mnt/data1/llm_team/Qwen2.5-7B-Instruct
MAX_MODEL_LEN=4096 GPU_MEMORY_UTILIZATION=0.85 ./scripts/run_vllm_servers.sh /mnt/data1/llm_team/Qwen2.5-7B-Instruct
EXTRA_VLLM_ARGS="--trust-remote-code" ./scripts/run_vllm_servers.sh /mnt/data1/llm_team/Qwen2.5-7B-Instruct
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

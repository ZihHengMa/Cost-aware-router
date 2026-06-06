from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException

from .cache import PrefixCache, prefix_hash, tokenize
from .models import GenerateRequest, GenerateResponse, RoutePolicy, WorkerState
from .policies import RouterPolicyEngine, WorkerView
from .storage import RequestMetadataStore


class CostAwareRouter:
    def __init__(
        self,
        worker_urls: list[str],
        policy: RoutePolicy,
        *,
        lmcache_shared: bool = False,
        metadata_db: str | Path = "data/router_metadata.sqlite",
        queue_weight: float = 8.0,
        prefill_weight: float = 1.0,
        cache_hit_bonus: float = 2.0,
        locality_threshold: float = 0.90,
        locality_queue_slack: int = 2,
        timeout_s: float = 300.0,
    ) -> None:
        shared_cache = PrefixCache() if lmcache_shared else None
        self.workers = [
            WorkerView(
                worker_id=f"worker-{idx}",
                url=url.rstrip("/"),
                cache=shared_cache if shared_cache else PrefixCache(),
            )
            for idx, url in enumerate(worker_urls)
        ]
        self.engine = RouterPolicyEngine(
            self.workers,
            policy,
            queue_weight=queue_weight,
            prefill_weight=prefill_weight,
            cache_hit_bonus=cache_hit_bonus,
            locality_threshold=locality_threshold,
            locality_queue_slack=locality_queue_slack,
        )
        self.policy = policy
        self.lmcache_shared = lmcache_shared
        self.disable_prefix_cache = policy == RoutePolicy.PREFILL_SCRATCH
        self.client = httpx.AsyncClient(timeout=timeout_s)
        self.metadata_store = RequestMetadataStore(metadata_db)

    async def close(self) -> None:
        await self.client.aclose()
        self.metadata_store.close()

    async def refresh_worker_state(self) -> None:
        async def fetch(worker: WorkerView) -> None:
            try:
                resp = await self.client.get(f"{worker.url}/state")
                resp.raise_for_status()
                state = WorkerState.model_validate(resp.json())
                worker.queue_depth = state.queue_depth
            except Exception:
                worker.queue_depth = 10_000

        await asyncio.gather(*(fetch(worker) for worker in self.workers))

    async def generate(self, req: GenerateRequest) -> GenerateResponse:
        await self.refresh_worker_state()
        candidate = self.engine.choose(req.prompt)
        worker = next(w for w in self.workers if w.worker_id == candidate.worker_id)
        worker.routed_requests += 1
        tokens = tokenize(req.prompt)

        routed_req = req.model_copy(
            update={
                "request_id": req.request_id or str(uuid.uuid4()),
                "metadata": {
                    **req.metadata,
                    "route_policy": self.policy.value,
                    "route_cost": candidate.estimated_cost,
                    "lmcache_hit_tokens": candidate.longest_prefix_hit if self.lmcache_shared and not self.disable_prefix_cache else 0,
                    "disable_prefix_cache": self.disable_prefix_cache,
                },
            }
        )
        try:
            resp = await self.client.post(f"{candidate.url}/generate", json=routed_req.model_dump())
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"worker request failed: {exc}") from exc

        if not self.disable_prefix_cache:
            worker.cache.insert_prefixes(tokens)
        result = GenerateResponse.model_validate(resp.json()).model_copy(
            update={
                "route_policy": self.policy.value,
                "route_cost": candidate.estimated_cost,
            }
        )
        self.metadata_store.insert(
            {
                "request_id": result.request_id,
                "worker_id": result.worker_id,
                "worker_url": candidate.url,
                "route_policy": self.policy.value,
                "route_cost": candidate.estimated_cost,
                "queue_depth_at_route": candidate.queue_depth,
                "queue_depth_at_start": result.queue_depth_at_start,
                "prompt": req.prompt,
                "prompt_sha256": hashlib.sha256(req.prompt.encode("utf-8")).hexdigest(),
                "full_prefix_hash": prefix_hash(tokens),
                "prompt_tokens": result.prompt_tokens,
                "max_tokens": req.max_tokens,
                "generated_tokens": result.generated_tokens,
                "estimated_prefix_hit_tokens": candidate.longest_prefix_hit,
                "estimated_prefill_tokens": candidate.estimated_prefill_tokens,
                "cache_hit_tokens": result.cache_hit_tokens,
                "exact_prefix_hit": int(candidate.exact_prefix_hit),
                "partial_match": int(result.partial_match),
                "ttft_ms": result.ttft_ms,
                "latency_ms": result.latency_ms,
                "lmcache_shared": int(self.lmcache_shared),
                "metadata_json": json.dumps(req.metadata, sort_keys=True),
            }
        )
        return result

    def router_state(self) -> dict[str, object]:
        total = sum(w.routed_requests for w in self.workers) or 1
        counts = {w.worker_id: w.routed_requests for w in self.workers}
        shares = {wid: count / total for wid, count in counts.items()}
        return {
            "policy": self.policy.value,
            "lmcache_shared": self.lmcache_shared,
            "routed_requests": counts,
            "traffic_share": shares,
            "imbalance_ratio": max(counts.values(), default=0) / max(min(counts.values(), default=1), 1),
            "cost_model": {
                "queue_weight": self.engine.queue_weight,
                "prefill_weight": self.engine.prefill_weight,
                "cache_hit_bonus": self.engine.cache_hit_bonus,
                "locality_threshold": self.engine.locality_threshold,
                "locality_queue_slack": self.engine.locality_queue_slack,
            },
            "metadata": self.metadata_store.stats(),
        }


def build_app(router: CostAwareRouter) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await router.close()

    app = FastAPI(title="Cost-aware vLLM Router", lifespan=lifespan)

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(req: GenerateRequest) -> GenerateResponse:
        return await router.generate(req)

    @app.get("/state")
    async def state() -> dict[str, object]:
        await router.refresh_worker_state()
        return router.router_state()

    @app.get("/requests")
    async def requests(limit: int = 50) -> list[dict[str, object]]:
        return router.metadata_store.latest(limit=max(1, min(limit, 500)))

    @app.get("/requests/{request_id}")
    async def request(request_id: str) -> dict[str, object]:
        row = router.metadata_store.get(request_id)
        if row is None:
            raise HTTPException(status_code=404, detail="request not found")
        return row

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--worker", action="append", required=True, help="Worker base URL. Pass twice.")
    parser.add_argument("--policy", choices=[p.value for p in RoutePolicy], default=RoutePolicy.COST_AWARE.value)
    parser.add_argument("--lmcache-shared", action="store_true")
    parser.add_argument("--metadata-db", default="data/router_metadata.sqlite")
    parser.add_argument("--queue-weight", type=float, default=8.0)
    parser.add_argument("--prefill-weight", type=float, default=1.0)
    parser.add_argument("--cache-hit-bonus", type=float, default=2.0)
    parser.add_argument("--locality-threshold", type=float, default=0.90)
    parser.add_argument("--locality-queue-slack", type=int, default=2)
    args = parser.parse_args()

    if len(args.worker) != 2:
        raise SystemExit("This project is scoped to exactly 2 workers. Pass --worker twice.")
    router = CostAwareRouter(
        args.worker,
        RoutePolicy(args.policy),
        lmcache_shared=args.lmcache_shared,
        metadata_db=args.metadata_db,
        queue_weight=args.queue_weight,
        prefill_weight=args.prefill_weight,
        cache_hit_bonus=args.cache_hit_bonus,
        locality_threshold=args.locality_threshold,
        locality_queue_slack=args.locality_queue_slack,
    )
    uvicorn.run(build_app(router), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

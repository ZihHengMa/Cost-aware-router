from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from typing import Protocol

import httpx
import uvicorn
from fastapi import FastAPI

from .cache import PrefixCache, tokenize
from .models import GenerateRequest, GenerateResponse, WorkerState


class Worker(Protocol):
    worker_id: str

    async def state(self) -> WorkerState: ...

    async def generate(self, req: GenerateRequest, route_policy: str, route_cost: float | None) -> GenerateResponse: ...


class VllmProxyWorker:
    def __init__(
        self,
        worker_id: str,
        *,
        backend_url: str,
        model: str,
        min_prefix_tokens: int = 8,
        enable_prefix_cache: bool = True,
        timeout_s: float = 300.0,
    ) -> None:
        self.worker_id = worker_id
        self.backend_url = backend_url.rstrip("/")
        self.model = model
        self.enable_prefix_cache = enable_prefix_cache
        self.cache = PrefixCache(min_prefix_tokens=min_prefix_tokens)
        self.active_requests = 0
        self.lock = asyncio.Lock()
        self.client = httpx.AsyncClient(timeout=timeout_s)

    async def state(self) -> WorkerState:
        return WorkerState(
            worker_id=self.worker_id,
            queue_depth=self.active_requests,
            active_requests=self.active_requests,
            cached_prefixes=len(self.cache.stored_prefixes),
            cached_tokens=self.cache.cached_tokens,
        )

    async def generate(self, req: GenerateRequest, route_policy: str, route_cost: float | None) -> GenerateResponse:
        request_id = req.request_id or str(uuid.uuid4())
        tokens = tokenize(req.prompt)
        async with self.lock:
            queue_depth_at_start = self.active_requests
            self.active_requests += 1

        no_cache = not self.enable_prefix_cache or bool(req.metadata.get("disable_prefix_cache", False))
        local_hit_tokens = 0 if no_cache else self.cache.longest_match(tokens)
        shared_hit_tokens = 0 if no_cache else int(req.metadata.get("lmcache_hit_tokens", 0))
        hit_tokens = max(local_hit_tokens, shared_hit_tokens)
        partial_match = 0 < hit_tokens < len(tokens)
        start = time.perf_counter()
        first_token_at: float | None = None
        generated_text: list[str] = []
        generated_tokens = req.max_tokens

        payload = {
            "model": self.model,
            "prompt": req.prompt,
            "max_tokens": req.max_tokens,
            "temperature": req.metadata.get("temperature", 0),
            "stream": True,
        }

        try:
            async with self.client.stream("POST", f"{self.backend_url}/v1/completions", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line.removeprefix("data: ").strip()
                    if data == "[DONE]":
                        break
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    chunk = json.loads(data)
                    choice = chunk.get("choices", [{}])[0]
                    generated_text.append(choice.get("text", ""))
                    usage = chunk.get("usage")
                    if usage:
                        generated_tokens = int(usage.get("completion_tokens", generated_tokens))

            end = time.perf_counter()
            if not no_cache:
                self.cache.insert_prefixes(tokens)
            return GenerateResponse(
                request_id=request_id,
                worker_id=self.worker_id,
                text="".join(generated_text),
                ttft_ms=((first_token_at or end) - start) * 1000,
                latency_ms=(end - start) * 1000,
                cache_hit_tokens=hit_tokens,
                prompt_tokens=len(tokens),
                generated_tokens=generated_tokens,
                queue_depth_at_start=queue_depth_at_start,
                route_policy=route_policy,
                route_cost=route_cost,
                partial_match=partial_match,
            )
        finally:
            async with self.lock:
                self.active_requests -= 1


def build_app(worker: Worker) -> FastAPI:
    app = FastAPI(title=f"Cost-aware Worker Adapter {worker.worker_id}")

    @app.get("/state", response_model=WorkerState)
    async def state() -> WorkerState:
        return await worker.state()

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(req: GenerateRequest) -> GenerateResponse:
        route_policy = str(req.metadata.get("route_policy", "direct"))
        route_cost = req.metadata.get("route_cost")
        return await worker.generate(req, route_policy, route_cost)

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["vllm"], default="vllm")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--disable-prefix-cache", action="store_true")
    parser.add_argument("--backend-url", help="Base URL for a vLLM OpenAI-compatible server.")
    parser.add_argument("--model", default="/mnt/data1/llm_team/Qwen2.5-7B-Instruct")
    args = parser.parse_args()

    if not args.backend_url:
        raise SystemExit("--backend-url is required")
    worker: Worker = VllmProxyWorker(
        args.worker_id,
        backend_url=args.backend_url,
        model=args.model,
        enable_prefix_cache=not args.disable_prefix_cache,
    )
    uvicorn.run(build_app(worker), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

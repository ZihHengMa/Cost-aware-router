from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class RoutePolicy(StrEnum):
    ROUND_ROBIN = "round_robin"
    LEAST_QUEUE = "least_queue"
    CACHE_AWARE = "cache_aware"
    COST_AWARE = "cost_aware"
    PREFILL_SCRATCH = "prefill_scratch"


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=64, ge=1, le=4096)
    request_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GenerateResponse(BaseModel):
    request_id: str
    worker_id: str
    text: str
    ttft_ms: float
    latency_ms: float
    cache_hit_tokens: int
    prompt_tokens: int
    generated_tokens: int
    queue_depth_at_start: int
    route_policy: str
    route_cost: float | None = None
    partial_match: bool = False


class WorkerState(BaseModel):
    worker_id: str
    queue_depth: int
    active_requests: int
    cached_prefixes: int
    cached_tokens: int


class WorkerCandidate(BaseModel):
    worker_id: str
    url: str
    queue_depth: int
    longest_prefix_hit: int
    exact_prefix_hit: bool
    estimated_prefill_tokens: int
    estimated_cost: float

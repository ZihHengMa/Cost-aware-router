from __future__ import annotations

from dataclasses import dataclass

from .cache import PrefixCache, tokenize
from .models import RoutePolicy, WorkerCandidate


@dataclass
class WorkerView:
    worker_id: str
    url: str
    queue_depth: int = 0
    cache: PrefixCache | None = None
    routed_requests: int = 0


class RouterPolicyEngine:
    def __init__(
        self,
        workers: list[WorkerView],
        policy: RoutePolicy,
        *,
        queue_weight: float = 8.0,
        prefill_weight: float = 1.0,
        cache_hit_bonus: float = 2.0,
        locality_threshold: float = 0.90,
        locality_queue_slack: int = 2,
    ) -> None:
        self.workers = workers
        self.policy = policy
        self.queue_weight = queue_weight
        self.prefill_weight = prefill_weight
        self.cache_hit_bonus = cache_hit_bonus
        self.locality_threshold = locality_threshold
        self.locality_queue_slack = locality_queue_slack
        self._rr_index = 0

    def choose(self, prompt: str) -> WorkerCandidate:
        candidates = self.candidates(prompt)
        if self.policy == RoutePolicy.ROUND_ROBIN:
            worker = self.workers[self._rr_index % len(self.workers)]
            self._rr_index += 1
            return next(c for c in candidates if c.worker_id == worker.worker_id)
        if self.policy == RoutePolicy.LEAST_QUEUE:
            return min(candidates, key=lambda c: (c.queue_depth, c.worker_id))
        if self.policy == RoutePolicy.PREFILL_SCRATCH:
            return min(candidates, key=lambda c: (c.queue_depth, c.worker_id))
        if self.policy == RoutePolicy.CACHE_AWARE:
            return min(candidates, key=lambda c: (-c.longest_prefix_hit, c.queue_depth, c.worker_id))
        if self.policy == RoutePolicy.COST_AWARE:
            return self._choose_cost_aware(candidates)
        return min(candidates, key=lambda c: (c.estimated_cost, c.queue_depth, c.worker_id))

    def _choose_cost_aware(self, candidates: list[WorkerCandidate]) -> WorkerCandidate:
        max_hit = max(c.longest_prefix_hit for c in candidates)
        min_queue = min(c.queue_depth for c in candidates)

        if max_hit > 0:
            min_good_hit = max(1, int(max_hit * self.locality_threshold))
            locality_candidates = [
                c
                for c in candidates
                if c.longest_prefix_hit >= min_good_hit
                and c.queue_depth <= min_queue + self.locality_queue_slack
            ]
            if locality_candidates:
                return min(
                    locality_candidates,
                    key=lambda c: (c.estimated_cost, c.queue_depth, -c.longest_prefix_hit, c.worker_id),
                )

        return min(candidates, key=lambda c: (c.estimated_cost, c.queue_depth, -c.longest_prefix_hit, c.worker_id))

    def candidates(self, prompt: str) -> list[WorkerCandidate]:
        tokens = tokenize(prompt)
        out: list[WorkerCandidate] = []
        for worker in self.workers:
            cache = worker.cache or PrefixCache()
            hit = 0 if self.policy == RoutePolicy.PREFILL_SCRATCH else cache.longest_match(tokens)
            prefill_tokens = max(len(tokens) - hit, 0)
            cost = (
                self.queue_weight * worker.queue_depth
                + self.prefill_weight * prefill_tokens
                - self.cache_hit_bonus * hit
            )
            out.append(
                WorkerCandidate(
                    worker_id=worker.worker_id,
                    url=worker.url,
                    queue_depth=worker.queue_depth,
                    longest_prefix_hit=hit,
                    exact_prefix_hit=False if self.policy == RoutePolicy.PREFILL_SCRATCH else cache.exact_match(tokens),
                    estimated_prefill_tokens=prefill_tokens,
                    estimated_cost=cost,
                )
            )
        return out

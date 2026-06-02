from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


def tokenize(text: str) -> list[str]:
    """A deterministic toy tokenizer for experiments without model weights."""
    return text.strip().split()


def prefix_hash(tokens: list[str], length: int | None = None) -> str:
    selected = tokens if length is None else tokens[:length]
    joined = "\x1f".join(selected).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


@dataclass
class PrefixCache:
    min_prefix_tokens: int = 8
    stored_prefixes: set[tuple[str, int]] = field(default_factory=set)

    def longest_match(self, tokens: list[str]) -> int:
        upper = len(tokens)
        for length in range(upper, self.min_prefix_tokens - 1, -1):
            if (prefix_hash(tokens, length), length) in self.stored_prefixes:
                return length
        return 0

    def exact_match(self, tokens: list[str]) -> bool:
        return (prefix_hash(tokens), len(tokens)) in self.stored_prefixes

    def insert_prefixes(self, tokens: list[str], step: int = 8) -> None:
        if len(tokens) < self.min_prefix_tokens:
            return
        for length in range(self.min_prefix_tokens, len(tokens) + 1, step):
            self.stored_prefixes.add((prefix_hash(tokens, length), length))
        self.stored_prefixes.add((prefix_hash(tokens), len(tokens)))

    @property
    def cached_tokens(self) -> int:
        return sum(length for _, length in self.stored_prefixes)

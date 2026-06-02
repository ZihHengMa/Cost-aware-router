from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RequestMetadataStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        self._init_schema()

    def close(self) -> None:
        with self.lock:
            self.conn.close()

    def _init_schema(self) -> None:
        with self.lock:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS request_metadata (
                    request_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    worker_id TEXT NOT NULL,
                    worker_url TEXT NOT NULL,
                    route_policy TEXT NOT NULL,
                    route_cost REAL,
                    queue_depth_at_route INTEGER NOT NULL,
                    queue_depth_at_start INTEGER NOT NULL,
                    prompt TEXT NOT NULL,
                    prompt_sha256 TEXT NOT NULL,
                    full_prefix_hash TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    max_tokens INTEGER NOT NULL,
                    generated_tokens INTEGER NOT NULL,
                    estimated_prefix_hit_tokens INTEGER NOT NULL,
                    estimated_prefill_tokens INTEGER NOT NULL,
                    cache_hit_tokens INTEGER NOT NULL,
                    exact_prefix_hit INTEGER NOT NULL,
                    partial_match INTEGER NOT NULL,
                    ttft_ms REAL NOT NULL,
                    latency_ms REAL NOT NULL,
                    lmcache_shared INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_request_metadata_created_at
                    ON request_metadata(created_at);
                CREATE INDEX IF NOT EXISTS idx_request_metadata_worker_id
                    ON request_metadata(worker_id);
                CREATE INDEX IF NOT EXISTS idx_request_metadata_full_prefix_hash
                    ON request_metadata(full_prefix_hash);
                """
            )

    def insert(self, row: dict[str, Any]) -> None:
        columns = [
            "request_id",
            "created_at",
            "worker_id",
            "worker_url",
            "route_policy",
            "route_cost",
            "queue_depth_at_route",
            "queue_depth_at_start",
            "prompt",
            "prompt_sha256",
            "full_prefix_hash",
            "prompt_tokens",
            "max_tokens",
            "generated_tokens",
            "estimated_prefix_hit_tokens",
            "estimated_prefill_tokens",
            "cache_hit_tokens",
            "exact_prefix_hit",
            "partial_match",
            "ttft_ms",
            "latency_ms",
            "lmcache_shared",
            "metadata_json",
        ]
        payload = {**row, "created_at": row.get("created_at") or utc_now()}
        values = [payload[column] for column in columns]
        placeholders = ", ".join("?" for _ in columns)
        column_sql = ", ".join(columns)
        update_sql = ", ".join(f"{column}=excluded.{column}" for column in columns if column != "request_id")
        with self.lock:
            self.conn.execute(
                f"""
                INSERT INTO request_metadata ({column_sql})
                VALUES ({placeholders})
                ON CONFLICT(request_id) DO UPDATE SET {update_sql}
                """,
                values,
            )
            self.conn.commit()

    def latest(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute(
                """
                SELECT * FROM request_metadata
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [row_to_dict(row) for row in rows]

    def get(self, request_id: str) -> dict[str, Any] | None:
        with self.lock:
            row = self.conn.execute(
                "SELECT * FROM request_metadata WHERE request_id = ?",
                (request_id,),
            ).fetchone()
        return row_to_dict(row) if row else None

    def stats(self) -> dict[str, Any]:
        with self.lock:
            total = self.conn.execute("SELECT COUNT(*) AS count FROM request_metadata").fetchone()["count"]
            workers = self.conn.execute(
                """
                SELECT worker_id, COUNT(*) AS requests,
                       AVG(ttft_ms) AS avg_ttft_ms,
                       AVG(latency_ms) AS avg_latency_ms,
                       AVG(cache_hit_tokens) AS avg_cache_hit_tokens
                FROM request_metadata
                GROUP BY worker_id
                ORDER BY worker_id
                """
            ).fetchall()
        return {
            "db_path": str(self.db_path),
            "total_requests": total,
            "workers": [row_to_dict(row) for row in workers],
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    out = dict(row)
    out["exact_prefix_hit"] = bool(out["exact_prefix_hit"])
    out["partial_match"] = bool(out["partial_match"])
    out["lmcache_shared"] = bool(out["lmcache_shared"])
    out["metadata"] = json.loads(out.pop("metadata_json"))
    return out

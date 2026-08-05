"""Query storage with lightweight persistence and size guardrails (M2-E6)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class QueryRecord:
    query_id: str
    text: str
    intent: str
    issued_at: float
    meta: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    user_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryRecord":
        return cls(
            query_id=data["query_id"],
            text=data.get("text", ""),
            intent=data.get("intent", ""),
            issued_at=data.get("issued_at", time.time()),
            meta=data.get("meta", {}) or {},
            dependencies=data.get("dependencies", []) or [],
            user_id=data.get("user_id"),
        )


class QueryStore:
    """JSONL-backed query store with in-memory index."""

    def __init__(self, path: Optional[Path] = None, size_limit_bytes: int = 10_000):
        self.path = Path(path) if path else None
        self.size_limit_bytes = size_limit_bytes
        self._records: Dict[str, QueryRecord] = {}
        if self.path and self.path.exists():
            self._load()

    def _load(self) -> None:
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            rec = QueryRecord.from_dict(data)
            self._records[rec.query_id] = rec

    def add_query(self, record: QueryRecord) -> None:
        payload = record.to_dict()
        size = len(json.dumps(payload).encode("utf-8"))
        if size > self.size_limit_bytes:
            raise ValueError(f"query {record.query_id} exceeds size limit {self.size_limit_bytes}B")
        self._records[record.query_id] = record
        if self.path:
            self._append_line(payload)

    def upsert(
        self,
        query_id: str,
        text: str,
        intent: str,
        meta: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None,
        user_id: Optional[str] = None,
    ) -> QueryRecord:
        rec = QueryRecord(
            query_id=query_id,
            text=text,
            intent=intent,
            issued_at=time.time(),
            meta=meta or {},
            dependencies=dependencies or [],
            user_id=user_id,
        )
        self.add_query(rec)
        return rec

    def get(self, query_id: str) -> Optional[QueryRecord]:
        return self._records.get(query_id)

    def list_by_intent(self, intent: str) -> List[QueryRecord]:
        return [r for r in self._records.values() if r.intent == intent]

    def stats(self) -> Dict[str, Any]:
        sizes = [len(json.dumps(r.to_dict()).encode("utf-8")) for r in self._records.values()]
        return {
            "count": len(self._records),
            "max_size_bytes": max(sizes) if sizes else 0,
            "mean_size_bytes": sum(sizes) / len(sizes) if sizes else 0,
        }

    def _append_line(self, payload: Dict[str, Any]) -> None:
        line = json.dumps(payload)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


__all__ = ["QueryStore", "QueryRecord"]

"""Result reification storage with freshness tracking (M2-E6)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


@dataclass
class ResultRecord:
    result_id: str
    query_id: str
    results: List[Any]
    result_timestamp: float
    freshness: Dict[str, Any] = field(default_factory=dict)
    dependent_facts: List[str] = field(default_factory=list)
    invalidation_rules: List[str] = field(default_factory=list)
    status: str = "active"  # active | stale

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResultRecord":
        return cls(
            result_id=data["result_id"],
            query_id=data.get("query_id", ""),
            results=data.get("results", []) or [],
            result_timestamp=data.get("result_timestamp", time.time()),
            freshness=data.get("freshness", {}) or {},
            dependent_facts=data.get("dependent_facts", []) or [],
            invalidation_rules=data.get("invalidation_rules", []) or [],
            status=data.get("status", "active"),
        )


class ResultStore:
    """JSONL-backed result store with freshness propagation."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self._records: Dict[str, ResultRecord] = {}
        if self.path and self.path.exists():
            self._load()

    def _load(self) -> None:
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = ResultRecord.from_dict(json.loads(line))
            self._records[rec.result_id] = rec

    def add_result(self, record: ResultRecord) -> None:
        self._records[record.result_id] = record
        if self.path:
            self._append_line(record.to_dict())

    def upsert(
        self,
        result_id: str,
        query_id: str,
        results: List[Any],
        freshness: Optional[Dict[str, Any]] = None,
        dependent_facts: Optional[List[str]] = None,
        invalidation_rules: Optional[List[str]] = None,
    ) -> ResultRecord:
        rec = ResultRecord(
            result_id=result_id,
            query_id=query_id,
            results=results,
            result_timestamp=time.time(),
            freshness=freshness or {},
            dependent_facts=dependent_facts or [],
            invalidation_rules=invalidation_rules or [],
        )
        self.add_result(rec)
        return rec

    def get(self, result_id: str) -> Optional[ResultRecord]:
        return self._records.get(result_id)

    def mark_stale_by_facts(self, touched_facts: Set[str]) -> List[ResultRecord]:
        return self._mark(lambda r: touched_facts.intersection(set(r.dependent_facts)))

    def mark_stale_by_rules(self, fired_rules: Set[str]) -> List[ResultRecord]:
        return self._mark(lambda r: fired_rules.intersection(set(r.invalidation_rules)))

    def active_results(self, query_id: Optional[str] = None) -> List[ResultRecord]:
        records = [r for r in self._records.values() if r.status == "active"]
        if query_id:
            records = [r for r in records if r.query_id == query_id]
        return records

    def _mark(self, predicate) -> List[ResultRecord]:
        changed: List[ResultRecord] = []
        for r in self._records.values():
            if r.status == "active" and predicate(r):
                r.status = "stale"
                changed.append(r)
        if changed and self.path:
            # rewrite file to persist status changes
            self._rewrite()
        return changed

    def _append_line(self, payload: Dict[str, Any]) -> None:
        line = json.dumps(payload)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _rewrite(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            for rec in self._records.values():
                f.write(json.dumps(rec.to_dict()) + "\n")


__all__ = ["ResultStore", "ResultRecord"]

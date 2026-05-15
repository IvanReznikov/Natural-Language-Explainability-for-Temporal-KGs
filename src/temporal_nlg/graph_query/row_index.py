"""Row-centric retrieval index over processed_graph.jsonl.

Implements the MultiVector / parent-document retriever pattern from graph_retrieval.md §2.3 / §3.3:
- Each row in processed_graph.jsonl is a parent document (query text + domain + tags).
- Search is done over row query text (TF-IDF or dense embeddings).
- Results carry linked node_uids, edge_uids, tag_uids so the caller can fetch
  the subgraph around the matched rows.

This provides a standard-RAG fallback when structural graph retrieval fails to
find matching edges for a user question.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _iter_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8-sig") as fh:
        for line in fh:
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


@dataclass(frozen=True)
class RowDocument:
    """A single row from processed_graph.jsonl."""

    row_id: str
    source_id: str
    query: str
    domain: str
    tags: List[str]
    node_uids: List[str]
    edge_uids: List[str]
    tag_uids: List[str]


@dataclass
class ScoredRow:
    row: RowDocument
    score: float


class RowRetrievalIndex:
    """TF-IDF index over row query texts for standard-RAG fallback.

    Build once at pipeline init; reuse across queries.
    """

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self._rows: List[RowDocument] = []
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._matrix = None
        self._build()

    @property
    def available(self) -> bool:
        return len(self._rows) > 0 and self._vectorizer is not None

    def _build(self) -> None:
        pg_path = self.output_dir / "processed_graph.jsonl"
        if not pg_path.exists():
            return

        rows: List[RowDocument] = []
        texts: List[str] = []
        for obj in _iter_jsonl(pg_path):
            query = str(obj.get("query") or "").strip()
            if not query:
                continue
            row = RowDocument(
                row_id=str(obj.get("id") or ""),
                source_id=str(obj.get("source_id") or ""),
                query=query,
                domain=str(obj.get("domain") or ""),
                tags=[str(t) for t in (obj.get("tags") or [])],
                node_uids=[str(u) for u in (obj.get("node_uids") or [])],
                edge_uids=[str(u) for u in (obj.get("edge_uids") or [])],
                tag_uids=[str(u) for u in (obj.get("tag_uids") or [])],
            )
            rows.append(row)
            # Combine query + tags for richer TF-IDF representation
            tag_text = " ".join(row.tags)
            texts.append(f"{query} {tag_text} {row.domain}")

        if not rows:
            return

        self._rows = rows
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
        )
        self._matrix = self._vectorizer.fit_transform(texts)

    def search(
        self,
        query: str,
        top_k: int = 10,
        domain_filter: Optional[str] = None,
    ) -> List[ScoredRow]:
        """Search rows by TF-IDF similarity to the user question."""

        if not self.available or self._vectorizer is None or self._matrix is None:
            return []

        vec = self._vectorizer.transform([query or ""])
        scores = cosine_similarity(vec, self._matrix)[0]

        candidates: List[ScoredRow] = []
        for idx, score in enumerate(scores):
            if score <= 0.0:
                continue
            row = self._rows[idx]
            if domain_filter and row.domain != domain_filter:
                continue
            candidates.append(ScoredRow(row=row, score=float(score)))

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[: max(1, int(top_k))]

    def collect_linked_uids(
        self,
        rows: Sequence[ScoredRow],
    ) -> Dict[str, List[str]]:
        """Gather unique node/edge/tag UIDs referenced by a set of rows."""

        node_uids: List[str] = []
        edge_uids: List[str] = []
        tag_uids: List[str] = []
        seen_n: set = set()
        seen_e: set = set()
        seen_t: set = set()
        for sr in rows:
            for uid in sr.row.node_uids:
                if uid not in seen_n:
                    seen_n.add(uid)
                    node_uids.append(uid)
            for uid in sr.row.edge_uids:
                if uid not in seen_e:
                    seen_e.add(uid)
                    edge_uids.append(uid)
            for uid in sr.row.tag_uids:
                if uid not in seen_t:
                    seen_t.add(uid)
                    tag_uids.append(uid)
        return {
            "node_uids": node_uids,
            "edge_uids": edge_uids,
            "tag_uids": tag_uids,
        }

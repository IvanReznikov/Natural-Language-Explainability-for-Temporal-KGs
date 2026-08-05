from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# BM25 is preferred for sparse retrieval over TF-IDF: better frequency saturation + IDF normalization.
# If rank_bm25 is not installed, we fall back to TF-IDF transparently.
try:
    from rank_bm25 import BM25Okapi as _BM25

    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False

from temporal_nlg.graph_query.index import GraphEdge, TemporalGraphIndex


def _norm(text: str) -> str:
    return str(text or "").strip().lower()


def _is_truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ScoredEdge:
    edge: GraphEdge
    score: float


class EdgeSemanticIndex:
    """Semantic retrieval index over graph edges.

    Backends:
    - qwen_server: HTTP embedding endpoint from a dedicated Qwen process
    - qwen_local: SentenceTransformer local model (Qwen embedding model)
    - tfidf: lexical fallback
    """

    def __init__(
        self,
        index: TemporalGraphIndex,
        embed_model_name: Optional[str] = None,
        embed_server_url: Optional[str] = None,
    ):
        self.index = index
        # Explicit model name takes priority over env var
        self._embed_model_name = (
            embed_model_name or os.getenv("LOCAL_EMBEDDING_MODEL_NAME") or "qwen_embedding"
        )
        self.strict_no_fallback = _is_truthy(os.getenv("GRAPH_STRICT_NO_FALLBACK"))
        self.backend = (os.getenv("GRAPH_RETRIEVAL_BACKEND") or "").strip().lower()
        if not self.backend:
            if self.strict_no_fallback:
                raise RuntimeError(
                    "GRAPH_RETRIEVAL_BACKEND must be explicitly set when GRAPH_STRICT_NO_FALLBACK=1"
                )
            # Prefer Qwen local embedding if model name is set; otherwise TF-IDF.
            self.backend = "qwen_local" if os.getenv("LOCAL_EMBEDDING_MODEL_NAME") else "tfidf"

        low_mem_mode = str(os.getenv("GRAPH_LOW_MEM_MODE", "0") or "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        disable_edge_dense = str(
            os.getenv("GRAPH_DISABLE_EDGE_DENSE", "1" if low_mem_mode else "0") or "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        if disable_edge_dense and self.backend in {"qwen_server", "qwen_local"}:
            if self.strict_no_fallback:
                raise RuntimeError(
                    "GRAPH_DISABLE_EDGE_DENSE=1 conflicts with strict retrieval backend mode. "
                    "Disable strict mode or use a non-dense backend explicitly."
                )
            self.backend = "tfidf"

        self.qwen_server_url = (
            embed_server_url or os.getenv("QWEN_EMBED_URL") or os.getenv("QWEN_SERVER_URL") or ""
        ).rstrip("/")
        if self.backend == "qwen_server" and not self.qwen_server_url:
            if self.strict_no_fallback:
                raise RuntimeError(
                    "QWEN_EMBED_URL (or QWEN_SERVER_URL) is required when GRAPH_RETRIEVAL_BACKEND=qwen_server in strict mode"
                )
            self.backend = "tfidf"

        self._dense_model = None
        self._edge_uids: List[str] = []
        self._edge_texts: List[str] = []
        self._dense_matrix: Optional[np.ndarray] = None
        self._server_embed_mode: Optional[str] = None
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._matrix = None
        # BM25 sparse index (preferred over TF-IDF when rank_bm25 is installed)
        self._bm25: Optional[object] = None
        self._build()

    @staticmethod
    def _model_slug(model_name: Optional[str]) -> str:
        raw = (model_name or "default").strip().lower()
        slug = "".join(ch if ch.isalnum() else "_" for ch in raw)
        while "__" in slug:
            slug = slug.replace("__", "_")
        return slug.strip("_") or "default"

    def _cache_paths(self) -> Tuple[Path, Path]:
        slug = self._model_slug(self._embed_model_name)
        cache_dir = self.index.output_dir / "embeddings"
        cache_dir.mkdir(parents=True, exist_ok=True)
        npy_path = cache_dir / f"edge_embeddings_{self.backend}_{slug}.npy"
        uids_path = cache_dir / f"edge_embeddings_{self.backend}_{slug}.uids.json"
        return npy_path, uids_path

    def _load_cached_edge_embeddings(self) -> Optional[np.ndarray]:
        if self.backend not in {"qwen_local", "qwen_server"}:
            return None
        npy_path, uids_path = self._cache_paths()
        if not npy_path.exists() or not uids_path.exists():
            return None
        try:
            cached_uids = json.loads(uids_path.read_text(encoding="utf-8"))
            if not isinstance(cached_uids, list):
                return None
            if [str(v) for v in cached_uids] != self._edge_uids:
                return None
            arr = np.load(npy_path)
            if arr.ndim != 2 or arr.shape[0] != len(self._edge_uids):
                return None
            return arr
        except Exception:
            return None

    def _save_cached_edge_embeddings(self, arr: np.ndarray) -> None:
        if self.backend not in {"qwen_local", "qwen_server"}:
            return
        try:
            npy_path, uids_path = self._cache_paths()
            np.save(npy_path, arr)
            uids_path.write_text(json.dumps(self._edge_uids, ensure_ascii=False), encoding="utf-8")
        except Exception:
            return

    def _edge_text(self, edge: GraphEdge) -> str:
        """Return a natural-language sentence for the edge — must stay in sync with
        precompute_graph_embeddings._edge_text() so cached .npy vectors remain valid."""
        source = self.index.node_label(edge.source_uid)
        target = self.index.node_label(edge.target_uid)
        src_cat = self.index.node_category(edge.source_uid)
        tgt_cat = self.index.node_category(edge.target_uid)
        relation = (edge.relation or "related_to").replace("_", " ")
        start = (edge.start or "").split("-")[0]
        end = (edge.end or "").split("-")[0]
        if start and end and start != end:
            temporal = f" from {start} to {end}"
        elif start:
            temporal = f" from {start}"
        elif end:
            temporal = f" until {end}"
        else:
            temporal = ""
        return f"{source} [{src_cat}] {relation} {target} [{tgt_cat}]{temporal}"

    def _build(self) -> None:
        edges = list(self.index.edge_by_uid.values())
        if not edges:
            self._edge_uids = []
            return

        self._edge_uids = [edge.edge_uid for edge in edges]
        self._edge_texts = [self._edge_text(edge) for edge in edges]

        cached = self._load_cached_edge_embeddings()
        if cached is not None:
            self._dense_matrix = cached
            return

        if self.backend == "qwen_server":
            self._dense_matrix = self._embed_with_server(self._edge_texts)
            if self._dense_matrix is not None:
                self._save_cached_edge_embeddings(self._dense_matrix)
            else:
                if self.strict_no_fallback:
                    raise RuntimeError(
                        "qwen_server embedding backend failed while GRAPH_STRICT_NO_FALLBACK=1"
                    )
                self.backend = "tfidf"

        if self.backend == "qwen_local":
            self._dense_model = self._load_local_embedding_model()
            if self._dense_model is not None:
                self._dense_matrix = self._embed_with_local(self._edge_texts, is_query=False)
            if self._dense_matrix is not None:
                self._save_cached_edge_embeddings(self._dense_matrix)
            else:
                if self.strict_no_fallback:
                    raise RuntimeError(
                        "qwen_local embedding backend failed while GRAPH_STRICT_NO_FALLBACK=1"
                    )
                self.backend = "tfidf"

        # Always build sparse index for hybrid search fallback
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self._matrix = self._vectorizer.fit_transform(self._edge_texts)
        if _BM25_AVAILABLE:
            # Tokenise with the same vocabulary as TF-IDF for consistency
            tokenised = [text.lower().split() for text in self._edge_texts]
            self._bm25 = _BM25(tokenised)

    def _load_local_embedding_model(self):
        model_name = self._embed_model_name
        if not model_name:
            return None
        try:
            from sentence_transformers import SentenceTransformer

            return SentenceTransformer(model_name)
        except Exception:
            return None

    def _embed_with_local(self, texts: Sequence[str], is_query: bool) -> Optional[np.ndarray]:
        if self._dense_model is None:
            return None
        try:
            if is_query:
                arr = self._dense_model.encode(
                    list(texts),
                    prompt_name="query",
                    normalize_embeddings=True,
                )
            else:
                arr = self._dense_model.encode(
                    list(texts),
                    normalize_embeddings=True,
                )
            return np.asarray(arr, dtype=float)
        except Exception:
            return None

    def _embed_with_server(
        self, texts: Sequence[str], is_query: bool = False
    ) -> Optional[np.ndarray]:
        if not self.qwen_server_url:
            return None

        def _post(url: str, payload_obj: dict) -> dict:
            req = urllib.request.Request(
                url=url,
                data=json.dumps(payload_obj).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode("utf-8"))

        def _resolve_mode() -> Optional[str]:
            if self._server_embed_mode:
                return self._server_embed_mode

            model_name = self._embed_model_name
            probe = ["probe"]
            try:
                body = _post(
                    f"{self.qwen_server_url}/v1/embeddings",
                    {"model": model_name, "input": probe},
                )
                data = body.get("data")
                if (
                    isinstance(data, list)
                    and len(data) == 1
                    and isinstance(data[0], dict)
                    and isinstance(data[0].get("embedding"), list)
                ):
                    self._server_embed_mode = "v1_embeddings"
                    return self._server_embed_mode
            except urllib.error.HTTPError as exc:
                if exc.code not in {404, 405}:
                    return None
            except Exception:
                return None

            try:
                body = _post(
                    f"{self.qwen_server_url}/embed",
                    {"texts": probe, "is_query": False},
                )
                vectors = body.get("vectors")
                if isinstance(vectors, list) and len(vectors) == 1:
                    self._server_embed_mode = "embed"
                    return self._server_embed_mode
            except Exception:
                return None

            return None

        mode = _resolve_mode()
        if mode is None:
            return None

        try:
            if mode == "v1_embeddings":
                model_name = self._embed_model_name
                body = _post(
                    f"{self.qwen_server_url}/v1/embeddings",
                    {"model": model_name, "input": list(texts)},
                )
                data = body.get("data")
                if not isinstance(data, list):
                    return None
                vectors = [item.get("embedding") for item in data if isinstance(item, dict)]
                if len(vectors) != len(texts) or any(not isinstance(v, list) for v in vectors):
                    return None
                return np.asarray(vectors, dtype=float)

            body = _post(
                f"{self.qwen_server_url}/embed",
                {"texts": list(texts), "is_query": bool(is_query)},
            )
            vectors = body.get("vectors")
            if not isinstance(vectors, list):
                return None
            return np.asarray(vectors, dtype=float)
        except Exception:
            return None

    def search(
        self,
        query: str,
        top_k: int = 40,
        entity_uids: Optional[Sequence[str]] = None,
        relation_hint: Optional[str] = None,
        year: Optional[int] = None,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        query_type: str = "point_in_time",
    ) -> List[ScoredEdge]:
        if not self._edge_uids:
            return []

        scores = np.zeros(len(self._edge_uids), dtype=float)
        dense_applied = False

        SPARSE_WEIGHTS = {
            "point_in_time": 0.15,
            "state_at_time": 0.15,
            "ordering": 0.25,
            "first_occurrence": 0.20,
            "summary": 0.10,
        }
        sparse_w = SPARSE_WEIGHTS.get(query_type, 0.30)
        dense_w = 1.0 - sparse_w

        if self.backend in {"qwen_local", "qwen_server"} and self._dense_matrix is not None:
            if self.backend == "qwen_server":
                qvec = self._embed_with_server([query or ""], is_query=True)
            else:
                qvec = self._embed_with_local([query or ""], is_query=True)

            if qvec is not None and len(qvec) == 1:
                dense_scores = cosine_similarity(qvec, self._dense_matrix)[0]
                scores += dense_scores * dense_w
                dense_applied = True

        # Compute sparse (lexical) scores: BM25 if available, else TF-IDF
        if self._bm25 is not None:
            try:
                tokens = (query or "").lower().split()
                bm25_scores = np.array(self._bm25.get_scores(tokens), dtype=float)
                # Normalise BM25 to [0, 1] range before blending
                bm25_max = bm25_scores.max()
                if bm25_max > 0:
                    bm25_scores = bm25_scores / bm25_max
                w = sparse_w if dense_applied else 1.0
                scores += bm25_scores * w
            except Exception:
                pass
        elif self._vectorizer is not None and self._matrix is not None:
            try:
                vec = self._vectorizer.transform([query or ""])
                lexical_scores = cosine_similarity(vec, self._matrix)[0]
                w = sparse_w if dense_applied else 1.0
                scores += lexical_scores * w
            except Exception:
                pass

        if not np.any(scores):
            return []

        allow_uids = set(entity_uids or [])
        relation_hint_norm = _norm(relation_hint or "")

        out: List[ScoredEdge] = []
        for idx, score in enumerate(scores):
            edge_uid = self._edge_uids[idx]
            edge = self.index.edge_by_uid[edge_uid]

            if allow_uids:
                if edge.source_uid not in allow_uids and edge.target_uid not in allow_uids:
                    continue

            if relation_hint_norm:
                if relation_hint_norm not in _norm(edge.relation):
                    continue

            if year is not None and not edge.overlaps_year(year):
                continue

            if start_year is not None:
                e_start = edge.start_year
                e_end = edge.end_year
                if e_start is not None and e_end is not None and max(e_start, e_end) < start_year:
                    continue
                if e_start is None and e_end is not None and e_end < start_year:
                    continue

            if end_year is not None:
                e_start = edge.start_year
                e_end = edge.end_year
                if e_start is not None and e_end is not None and min(e_start, e_end) > end_year:
                    continue
                if e_end is None and e_start is not None and e_start > end_year:
                    continue

            out.append(ScoredEdge(edge=edge, score=float(score)))

        out.sort(key=lambda item: (item.score, item.edge.support_count), reverse=True)
        return out[: max(1, int(top_k))]

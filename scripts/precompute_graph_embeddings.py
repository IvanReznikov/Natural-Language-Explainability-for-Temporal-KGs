#!/usr/bin/env python3
"""Precompute and persist graph embeddings.

Outputs are written under:
  <graph-output-dir>/embeddings/

Primary files:
- node_normalized_label_embeddings_<backend>_<model>.npy
- node_normalized_label_embeddings_<backend>_<model>.meta.jsonl
- edge_relation_embeddings_<backend>_<model>.npy
- edge_relation_embeddings_<backend>_<model>.meta.jsonl
- tag_normalized_tag_embeddings_<backend>_<model>.npy
- tag_normalized_tag_embeddings_<backend>_<model>.meta.jsonl

Compatibility/retrieval files:
- node_embeddings_<model>.npy
- node_embeddings_<model>.meta.jsonl
- edge_embeddings_<model>.npy
- edge_embeddings_<model>.meta.jsonl

Compatibility cache for graph retrieval (`EdgeSemanticIndex`):
- edge_embeddings_qwen_local_<model>.npy
- edge_embeddings_qwen_local_<model>.uids.json
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from temporal_nlg.graph_query.index import TemporalGraphIndex
from temporal_nlg.models import QwenEmbeddingModel


def _slug(value: str) -> str:
    text = (value or "default").strip().lower()
    out = "".join(ch if ch.isalnum() else "_" for ch in text)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "default"


def _batched(items: Sequence[str], batch_size: int) -> Iterable[List[str]]:
    size = max(1, int(batch_size))
    for i in range(0, len(items), size):
        yield list(items[i : i + size])


def _render_progress(done: int, total: int, label: str) -> None:
    """Kept for compatibility; tqdm is used in _embed_texts_generic."""


def _cleanup_duplicate_chunk_files(checkpoint_dir: Path) -> Tuple[int, int]:
    """Remove Drive-conflict chunk files like 'chunk_000001 (1).npy'.

    If the canonical chunk file is missing, promote the first duplicate to canonical
    to preserve resume data; remove remaining duplicates.
    """
    removed = 0
    promoted = 0
    for path in sorted(checkpoint_dir.glob("chunk_*.npy")):
        name = path.name
        if " (" not in name or not name.endswith(").npy"):
            continue
        base, suffix = name.rsplit(" (", 1)
        count_text = suffix[: -len(").npy")]
        if not base.startswith("chunk_") or not count_text.isdigit():
            continue

        canonical = checkpoint_dir / f"{base}.npy"
        try:
            if canonical.exists():
                path.unlink()
                removed += 1
            else:
                path.replace(canonical)
                promoted += 1
        except OSError:
            continue
    return removed, promoted


def _load_checkpoint_chunk(
    path: Path,
    expected_rows: Optional[int] = None,
    retries: int = 2,
) -> Tuple[Optional[np.ndarray], Optional[Exception]]:
    """Load a checkpoint chunk with retries for flaky synced filesystems.

    Returns (array, None) on success, otherwise (None, exception).
    """
    last_exc: Optional[Exception] = None
    for attempt in range(max(0, int(retries)) + 1):
        try:
            arr = np.asarray(np.load(path), dtype=np.float32)
            if arr.ndim != 2:
                raise ValueError(f"expected 2D array, got shape={arr.shape}")
            if expected_rows is not None and arr.shape[0] != int(expected_rows):
                raise ValueError(
                    f"row count mismatch: expected {expected_rows}, got {arr.shape[0]}"
                )
            return arr, None
        except (OSError, ValueError, EOFError) as exc:
            last_exc = exc
            if attempt < int(retries):
                time.sleep(0.4 * (attempt + 1))
                continue
            return None, last_exc
    return None, last_exc


def _edge_text(index: TemporalGraphIndex, edge_uid: str) -> str:
    """Return a natural-language sentence for the edge, used as the embedding corpus text.

    Sentence form is ~10x better aligned to user questions than keyword concatenation in
    the Qwen embedding space because the model was trained on sentence similarity.
    Category annotations help the model distinguish person/org/event lookups.

    Example:
        'Bill Clinton [person] served as US President [event] from 1993 to 2001'
    """
    edge = index.edge_by_uid[edge_uid]
    source = index.node_label(edge.source_uid)
    target = index.node_label(edge.target_uid)
    src_cat = index.node_category(edge.source_uid)
    tgt_cat = index.node_category(edge.target_uid)
    relation = (edge.relation or "related_to").replace("_", " ")
    start = (edge.start or "").split("-")[0]  # year only for readability
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


def _node_text(index: TemporalGraphIndex, node_uid: str) -> str:
    """Return a natural-language token for the node used as its embedding corpus text."""
    label = index.node_label(node_uid)
    category = index.node_category_by_uid.get(node_uid, "unknown")
    return f"{label} [{category}]"


def _embed_texts(model: QwenEmbeddingModel, texts: Sequence[str], batch_size: int) -> np.ndarray:
    parts: List[np.ndarray] = []
    for chunk in _batched(texts, batch_size):
        vecs = model.embed_documents(chunk)
        parts.append(np.asarray(vecs, dtype=float))
    if not parts:
        return np.zeros((0, 0), dtype=float)
    return np.vstack(parts)


def _embed_texts_generic(
    embed_fn: Callable[[Sequence[str]], np.ndarray],
    texts: Sequence[str],
    batch_size: int,
    progress_label: str = "Embedding",
    checkpoint_dir: Optional[Path] = None,
    parallelism: int = 1,
) -> np.ndarray:
    total = len(texts)
    if total == 0:
        return np.zeros((0, 0), dtype=np.float32)

    chunks = list(_batched(texts, batch_size))
    chunk_count = len(chunks)
    results: List[Optional[np.ndarray]] = [None] * chunk_count
    done = 0

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        removed_dups, promoted_dups = _cleanup_duplicate_chunk_files(checkpoint_dir)
        if removed_dups or promoted_dups:
            print(
                f"Checkpoint cleanup ({checkpoint_dir.name}): "
                f"removed={removed_dups}, promoted={promoted_dups}"
            )

    pending: List[Tuple[int, List[str]]] = []
    for idx, chunk in enumerate(chunks):
        if checkpoint_dir is not None:
            chunk_path = checkpoint_dir / f"chunk_{idx:06d}.npy"
            if chunk_path.exists():
                arr, load_err = _load_checkpoint_chunk(chunk_path, expected_rows=len(chunk))
                if arr is not None:
                    results[idx] = arr
                    done += len(chunk)
                    continue
                try:
                    chunk_path.unlink()
                except OSError:
                    pass
                err_name = type(load_err).__name__ if load_err is not None else "UnknownError"
                print(f"WARN removed unreadable checkpoint chunk: {chunk_path.name} ({err_name})")
        pending.append((idx, chunk))

    bar = tqdm(
        total=total,
        initial=done,
        desc=progress_label,
        unit="item",
        dynamic_ncols=True,
        leave=True,
    )

    if pending and int(parallelism) > 1:
        with ThreadPoolExecutor(max_workers=int(parallelism)) as pool:
            future_map = {
                pool.submit(embed_fn, chunk): (idx, len(chunk))
                for idx, chunk in pending
            }
            for future in as_completed(future_map):
                idx, chunk_len = future_map[future]
                arr = np.asarray(future.result(), dtype=np.float32)
                if arr.ndim != 2 or arr.shape[0] != chunk_len:
                    bar.close()
                    raise RuntimeError(
                        f"Invalid embedding chunk shape for chunk {idx}: got {arr.shape}, expected row count {chunk_len}."
                    )
                results[idx] = arr
                if checkpoint_dir is not None:
                    _atomic_save_npy(checkpoint_dir / f"chunk_{idx:06d}.npy", arr)
                bar.update(chunk_len)
    else:
        for idx, chunk in pending:
            arr = np.asarray(embed_fn(chunk), dtype=np.float32)
            if arr.ndim != 2 or arr.shape[0] != len(chunk):
                bar.close()
                raise RuntimeError(
                    f"Invalid embedding chunk shape for chunk {idx}: got {arr.shape}, expected row count {len(chunk)}."
                )
            results[idx] = arr
            if checkpoint_dir is not None:
                _atomic_save_npy(checkpoint_dir / f"chunk_{idx:06d}.npy", arr)
            bar.update(len(chunk))

    bar.close()

    if any(part is None for part in results):
        missing = [i for i, part in enumerate(results) if part is None]
        raise RuntimeError(f"Missing embedding chunks after processing: {missing[:10]}")

    parts = [part for part in results if part is not None]
    if not parts:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack(parts).astype(np.float32)


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                yield value


def _post_json(
    url: str,
    payload: dict,
    timeout_sec: int = 180,
    retries: int = 3,
    retry_backoff_sec: float = 2.0,
) -> dict:
    req = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    attempt = 0
    max_attempts = max(1, int(retries))
    while True:
        attempt += 1
        try:
            with urllib.request.urlopen(req, timeout=max(1, int(timeout_sec))) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in {429, 500, 502, 503, 504} and attempt < max_attempts:
                time.sleep(max(0.0, float(retry_backoff_sec)) * attempt)
                continue
            raise
        except Exception:
            if attempt < max_attempts:
                time.sleep(max(0.0, float(retry_backoff_sec)) * attempt)
                continue
            raise


def _resolve_server_embed_mode(
    url: str,
    model_name: str = "",
    timeout_sec: int = 180,
    retries: int = 3,
    retry_backoff_sec: float = 2.0,
) -> str:
    base = url.rstrip("/")
    probe = ["probe"]

    effective_model = (model_name or "Qwen/Qwen3-Embedding-0.6B").strip()
    try:
        body = _post_json(
            f"{base}/v1/embeddings",
            {"model": effective_model, "input": probe},
            timeout_sec=timeout_sec,
            retries=retries,
            retry_backoff_sec=retry_backoff_sec,
        )
        data = body.get("data")
        if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict) and isinstance(data[0].get("embedding"), list):
            return "v1_embeddings"
    except urllib.error.HTTPError as exc:
        if exc.code not in {404, 405}:
            raise

    body = _post_json(
        f"{base}/embed",
        {"texts": probe, "is_query": False},
        timeout_sec=timeout_sec,
        retries=retries,
        retry_backoff_sec=retry_backoff_sec,
    )
    vectors = body.get("vectors")
    if isinstance(vectors, list) and len(vectors) == 1:
        return "embed"
    raise RuntimeError("Could not resolve server embedding endpoint: /embed or /v1/embeddings")


def _embed_with_server(
    url: str,
    texts: Sequence[str],
    model_name: str = "",
    mode: str = "",
    timeout_sec: int = 180,
    retries: int = 3,
    retry_backoff_sec: float = 2.0,
) -> np.ndarray:
    base = url.rstrip("/")
    effective_mode = mode or _resolve_server_embed_mode(
        url,
        model_name=model_name,
        timeout_sec=timeout_sec,
        retries=retries,
        retry_backoff_sec=retry_backoff_sec,
    )

    if effective_mode == "embed":
        body = _post_json(
            f"{base}/embed",
            {"texts": list(texts), "is_query": False},
            timeout_sec=timeout_sec,
            retries=retries,
            retry_backoff_sec=retry_backoff_sec,
        )
        vectors = body.get("vectors")
        if not isinstance(vectors, list):
            raise RuntimeError("Embedding server returned invalid /embed response: missing vectors list")
        return np.asarray(vectors, dtype=float)

    if effective_mode == "v1_embeddings":
        effective_model = (model_name or "Qwen/Qwen3-Embedding-0.6B").strip()
        body = _post_json(
            f"{base}/v1/embeddings",
            {"model": effective_model, "input": list(texts)},
            timeout_sec=timeout_sec,
            retries=retries,
            retry_backoff_sec=retry_backoff_sec,
        )
        data = body.get("data")
        if not isinstance(data, list):
            raise RuntimeError("Embedding server returned invalid response: missing data list")
        vectors = [item.get("embedding") for item in data if isinstance(item, dict)]
        if len(vectors) != len(texts) or any(not isinstance(v, list) for v in vectors):
            raise RuntimeError("Embedding server returned invalid response: malformed embeddings")
        return np.asarray(vectors, dtype=float)

    raise RuntimeError(f"Unsupported embedding mode: {effective_mode}")


def _safe_unique(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in values:
        value = str(raw or "").strip()
        if not value:
            continue
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _write_meta_jsonl(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp.npy")
    np.save(temp_path, arr)
    temp_path.replace(path)


def _meta_prefix_matches(existing_rows: Sequence[dict], target_rows: Sequence[dict], keys: Sequence[str]) -> bool:
    if len(existing_rows) > len(target_rows):
        return False
    for idx, row in enumerate(existing_rows):
        tgt = target_rows[idx]
        for key in keys:
            if row.get(key) != tgt.get(key):
                return False
    return True


def _embed_with_optional_append(
    *,
    append_only: bool,
    embed_fn: Callable[[Sequence[str]], np.ndarray],
    texts: Sequence[str],
    meta_rows: Sequence[dict],
    meta_keys: Sequence[str],
    npy_path: Path,
    meta_path: Path,
    batch_size: int,
    progress_label: str,
    checkpoint_dir: Path,
    parallelism: int,
) -> np.ndarray:
    if not append_only:
        matrix = _embed_texts_generic(
            embed_fn,
            texts,
            batch_size,
            progress_label=progress_label,
            checkpoint_dir=checkpoint_dir,
            parallelism=parallelism,
        )
        _atomic_save_npy(npy_path, matrix)
        _write_meta_jsonl(meta_path, meta_rows)
        return matrix

    meta_exists = meta_path.exists()
    npy_exists = npy_path.exists()

    if meta_exists != npy_exists or not meta_exists:
        raise RuntimeError(
            f"append-only requested, but base embedding artifacts are missing or inconsistent for {npy_path.name}. "
            "Run a full embedding build once (without --append-only), then retry incremental append."
        )

    existing_rows = list(_iter_jsonl(meta_path))

    def _key_of(row: dict) -> Tuple[Optional[str], ...]:
        return tuple(row.get(k) for k in meta_keys)

    target_keys: List[Tuple[Optional[str], ...]] = []
    target_key_to_text: Dict[Tuple[Optional[str], ...], str] = {}
    target_key_to_row: Dict[Tuple[Optional[str], ...], dict] = {}
    for idx, row in enumerate(meta_rows):
        key = _key_of(row)
        if key in target_key_to_text:
            raise RuntimeError(
                f"append-only requested, but duplicate metadata key {key} found in target rows for {meta_path.name}."
            )
        target_keys.append(key)
        target_key_to_text[key] = texts[idx]
        target_key_to_row[key] = row

    existing_keys = [_key_of(row) for row in existing_rows]
    existing_key_set = set(existing_keys)

    missing_from_target = [key for key in existing_keys if key not in target_key_to_text]
    if missing_from_target:
        raise RuntimeError(
            f"append-only requested, but existing metadata keys are missing from current artifact set for {meta_path.name}. "
            "This indicates deletions/reordering that cannot be handled by append-only mode."
        )

    existing_matrix = np.asarray(np.load(npy_path), dtype=np.float32)
    if existing_matrix.ndim != 2 or existing_matrix.shape[0] != len(existing_rows):
        raise RuntimeError(
            f"append-only requested, but existing matrix shape {existing_matrix.shape} is incompatible "
            f"with metadata rows {len(existing_rows)} for {npy_path.name}."
        )

    append_keys = [key for key in target_keys if key not in existing_key_set]
    append_count = len(append_keys)
    if append_count <= 0:
        return existing_matrix

    tail_texts = [target_key_to_text[key] for key in append_keys]
    tail_rows = [target_key_to_row[key] for key in append_keys]
    append_checkpoint_dir = checkpoint_dir / f"append_from_{len(existing_rows):09d}"
    tail_matrix = _embed_texts_generic(
        embed_fn,
        tail_texts,
        batch_size,
        progress_label=f"{progress_label} (append)",
        checkpoint_dir=append_checkpoint_dir,
        parallelism=parallelism,
    )
    combined = np.vstack([existing_matrix, tail_matrix]).astype(np.float32)
    _atomic_save_npy(npy_path, combined)
    _write_meta_jsonl(meta_path, list(existing_rows) + tail_rows)
    return combined


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-output-dir", type=str, required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--qwen-server-url", type=str, default="")
    ap.add_argument(
        "--embedding-model-name",
        type=str,
        default="",
        help="Optional explicit model name for filename slugs.",
    )
    ap.add_argument(
        "--server-embedding-model-name",
        type=str,
        default="",
        help="OpenAI-compatible embedding model name for /v1/embeddings servers.",
    )
    ap.add_argument(
        "--parallelism",
        type=int,
        default=0,
        help="Number of parallel embedding requests. 0 = auto (4 for server, 1 for local).",
    )
    ap.add_argument(
        "--request-timeout-sec",
        type=int,
        default=240,
        help="HTTP timeout per embedding request in seconds.",
    )
    ap.add_argument(
        "--request-retries",
        type=int,
        default=4,
        help="Retries per failed embedding request.",
    )
    ap.add_argument(
        "--retry-backoff-sec",
        type=float,
        default=3.0,
        help="Linear backoff base in seconds between retries.",
    )
    ap.add_argument(
        "--core-only",
        action="store_true",
        help="Only generate core field embeddings (normalized_label/relation/normalized_tag).",
    )
    ap.add_argument(
        "--append-only",
        action="store_true",
        help="Append embeddings only for new tail records when existing metadata is a prefix; fallback to full recompute otherwise.",
    )
    args = ap.parse_args()

    output_dir = Path(args.graph_output_dir)
    index = TemporalGraphIndex(output_dir)

    qwen_server_url = (args.qwen_server_url or "").strip()
    embed_backend = "qwen_server" if qwen_server_url else "qwen_local"

    model: Optional[QwenEmbeddingModel] = None
    model_name = (args.embedding_model_name or "").strip()
    if embed_backend == "qwen_local":
        model = QwenEmbeddingModel()
        if not model.available:
            raise SystemExit(
                "Qwen embedding model is unavailable. Check LOCAL_EMBEDDING_MODEL_NAME and dependencies."
            )
        model_name = model_name or model.model_name
        embed_fn = lambda texts: np.asarray(model.embed_documents(list(texts)), dtype=float)
    else:
        server_model_name = (args.server_embedding_model_name or "").strip()
        model_name = model_name or server_model_name or "qwen_server"
        server_mode = _resolve_server_embed_mode(
            qwen_server_url,
            model_name=server_model_name,
            timeout_sec=args.request_timeout_sec,
            retries=args.request_retries,
            retry_backoff_sec=args.retry_backoff_sec,
        )
        embed_fn = lambda texts: _embed_with_server(
            qwen_server_url,
            texts,
            model_name=server_model_name,
            mode=server_mode,
            timeout_sec=args.request_timeout_sec,
            retries=args.request_retries,
            retry_backoff_sec=args.retry_backoff_sec,
        )

    model_slug = _slug(model_name)
    emb_dir = output_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = emb_dir / f"checkpoints_{embed_backend}_{model_slug}"

    if int(args.parallelism) > 0:
        parallelism = int(args.parallelism)
    else:
        parallelism = 2 if embed_backend == "qwen_server" else 1

    nodes_path = output_dir / "nodes.jsonl"
    edges_path = output_dir / "edges.jsonl"
    tags_path = output_dir / "tags.jsonl"

    node_rows = list(_iter_jsonl(nodes_path))
    tag_rows = list(_iter_jsonl(tags_path))
    edge_rows = list(_iter_jsonl(edges_path))

    node_labels = _safe_unique([str(row.get("normalized_label") or "") for row in node_rows])
    tag_values = _safe_unique([str(row.get("normalized_tag") or "") for row in tag_rows])
    edge_relations = _safe_unique([str(row.get("relation") or "") for row in edge_rows])

    node_label_npy = emb_dir / f"node_normalized_label_embeddings_{embed_backend}_{model_slug}.npy"
    node_label_meta = emb_dir / f"node_normalized_label_embeddings_{embed_backend}_{model_slug}.meta.jsonl"
    edge_relation_npy = emb_dir / f"edge_relation_embeddings_{embed_backend}_{model_slug}.npy"
    edge_relation_meta = emb_dir / f"edge_relation_embeddings_{embed_backend}_{model_slug}.meta.jsonl"
    tag_npy = emb_dir / f"tag_normalized_tag_embeddings_{embed_backend}_{model_slug}.npy"
    tag_meta = emb_dir / f"tag_normalized_tag_embeddings_{embed_backend}_{model_slug}.meta.jsonl"

    node_label_rows = [{"normalized_label": label} for label in node_labels]
    edge_relation_rows = [{"relation": rel} for rel in edge_relations]
    tag_rows_meta = [{"normalized_tag": tag} for tag in tag_values]

    node_label_matrix = _embed_with_optional_append(
        append_only=args.append_only,
        embed_fn=embed_fn,
        texts=node_labels,
        meta_rows=node_label_rows,
        meta_keys=["normalized_label"],
        npy_path=node_label_npy,
        meta_path=node_label_meta,
        batch_size=args.batch_size,
        progress_label="Embedding node labels",
        checkpoint_dir=checkpoint_root / "node_normalized_label",
        parallelism=parallelism,
    )
    edge_relation_matrix = _embed_with_optional_append(
        append_only=args.append_only,
        embed_fn=embed_fn,
        texts=edge_relations,
        meta_rows=edge_relation_rows,
        meta_keys=["relation"],
        npy_path=edge_relation_npy,
        meta_path=edge_relation_meta,
        batch_size=args.batch_size,
        progress_label="Embedding edge relations",
        checkpoint_dir=checkpoint_root / "edge_relation",
        parallelism=parallelism,
    )
    tag_matrix = _embed_with_optional_append(
        append_only=args.append_only,
        embed_fn=embed_fn,
        texts=tag_values,
        meta_rows=tag_rows_meta,
        meta_keys=["normalized_tag"],
        npy_path=tag_npy,
        meta_path=tag_meta,
        batch_size=args.batch_size,
        progress_label="Embedding tag values",
        checkpoint_dir=checkpoint_root / "tag_normalized_tag",
        parallelism=parallelism,
    )

    node_npy: Optional[Path] = None
    edge_npy: Optional[Path] = None
    compat_edge_npy: Optional[Path] = None
    node_matrix: Optional[np.ndarray] = None
    edge_matrix: Optional[np.ndarray] = None

    if not args.core_only:
        node_uids = sorted(index.node_label_by_uid.keys())
        node_texts = [_node_text(index, uid) for uid in node_uids]
        edge_uids = sorted(index.edge_by_uid.keys())
        edge_texts = [_edge_text(index, uid) for uid in edge_uids]

        node_npy = emb_dir / f"node_embeddings_{model_slug}.npy"
        node_meta = emb_dir / f"node_embeddings_{model_slug}.meta.jsonl"
        edge_npy = emb_dir / f"edge_embeddings_{model_slug}.npy"
        edge_meta = emb_dir / f"edge_embeddings_{model_slug}.meta.jsonl"

        node_meta_rows = [
            {
                "node_uid": uid,
                "label": index.node_label(uid),
                "category": index.node_category_by_uid.get(uid, "unknown"),
            }
            for uid in node_uids
        ]
        edge_meta_rows = [
            {
                "edge_uid": uid,
                "source_uid": index.edge_by_uid[uid].source_uid,
                "target_uid": index.edge_by_uid[uid].target_uid,
                "relation": index.edge_by_uid[uid].relation,
            }
            for uid in edge_uids
        ]

        node_matrix = _embed_with_optional_append(
            append_only=args.append_only,
            embed_fn=embed_fn,
            texts=node_texts,
            meta_rows=node_meta_rows,
            meta_keys=["node_uid"],
            npy_path=node_npy,
            meta_path=node_meta,
            batch_size=args.batch_size,
            progress_label="Embedding retrieval nodes",
            checkpoint_dir=checkpoint_root / "retrieval_nodes",
            parallelism=parallelism,
        )

        edge_matrix = _embed_with_optional_append(
            append_only=args.append_only,
            embed_fn=embed_fn,
            texts=edge_texts,
            meta_rows=edge_meta_rows,
            meta_keys=["edge_uid", "source_uid", "target_uid", "relation"],
            npy_path=edge_npy,
            meta_path=edge_meta,
            batch_size=args.batch_size,
            progress_label="Embedding retrieval edges",
            checkpoint_dir=checkpoint_root / "retrieval_edges",
            parallelism=parallelism,
        )

        # Compatibility cache consumed directly by EdgeSemanticIndex.
        # Write both qwen_local and qwen_server prefixes so the pipeline works
        # regardless of which GRAPH_RETRIEVAL_BACKEND env var is set at query time.
        for compat_prefix in ("qwen_local", "qwen_server"):
            compat_edge_npy = emb_dir / f"edge_embeddings_{compat_prefix}_{model_slug}.npy"
            compat_edge_uids_path = emb_dir / f"edge_embeddings_{compat_prefix}_{model_slug}.uids.json"
            _atomic_save_npy(compat_edge_npy, edge_matrix)
            compat_edge_uids_path.write_text(json.dumps(edge_uids, ensure_ascii=False), encoding="utf-8")
        compat_edge_npy = emb_dir / f"edge_embeddings_qwen_server_{model_slug}.npy"

    print(f"Backend: {embed_backend}")
    print(f"Model: {model_name}")
    if embed_backend == "qwen_server":
        print(f"Server endpoint mode: {server_mode}")
    print(f"Parallelism: {parallelism}")
    print(f"Checkpoint dir: {checkpoint_root}")
    print(f"Wrote node normalized_label embeddings: {node_label_npy} ({node_label_matrix.shape})")
    print(f"Wrote edge relation embeddings: {edge_relation_npy} ({edge_relation_matrix.shape})")
    print(f"Wrote tag normalized_tag embeddings: {tag_npy} ({tag_matrix.shape})")
    if args.core_only:
        print("Core-only mode: skipped retrieval node/edge embedding artifacts.")
    else:
        assert node_npy is not None and edge_npy is not None and compat_edge_npy is not None
        assert node_matrix is not None and edge_matrix is not None
        print(f"Wrote node embeddings: {node_npy} ({node_matrix.shape})")
        print(f"Wrote edge embeddings: {edge_npy} ({edge_matrix.shape})")
        print(f"Wrote retrieval cache: {compat_edge_npy}")


if __name__ == "__main__":
    main()

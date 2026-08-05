#!/usr/bin/env python3
"""Run M3-E5 with graph LCEL retrieval + GPT answer generation across models.

This runner is intended to mirror the Qwen graph flow while letting you swap GPT
models with one CLI flag.

Example:
  python experiments/m3_e5_benchmark/run_m3_e5_lcel_openai_models.py \
      --models gpt-4o,gpt-4.1,gpt-5 \
      --eval-set data/jsonls/temporal_evaluation_set_v2.jsonl \
      --graph-dir data/jsonls/temporal_graph_output_v3 \
      --output-dir output/m3_e5_gpt_lcel
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List
import urllib.request

from langchain_openai import ChatOpenAI


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def _load_dotenv_if_present(repo_root: Path) -> None:
    env_path = repo_root / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if (len(value) >= 2) and ((value[0] == value[-1]) and value[0] in {'"', "'"}):
            value = value[1:-1]
        if key not in os.environ:
            os.environ[key] = value


def _normalize(text: str) -> str:
    t = str(text or "").strip().lower()
    t = t.replace("’", "'")
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _score_prediction(prediction: str, gold: str) -> Dict[str, float]:
    p = _normalize(prediction)
    g = _normalize(gold)
    if not g:
        return {"exact": 0.0, "contains": 0.0}
    exact = 1.0 if p == g else 0.0
    contains = 1.0 if (g and g in p) or exact else 0.0
    return {"exact": exact, "contains": contains}


def _clean_answer(raw: str) -> str:
    text = str(raw or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    text = re.sub(
        r"^(the answer is[:\s]+|answer[:\s]+|final answer[:\s]+|output[:\s]+)",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines:
        text = lines[0]
    yn_m = re.match(r"^(yes|no)\b", text, flags=re.IGNORECASE)
    if yn_m:
        return yn_m.group(1).capitalize()
    text = text.strip(" \t\n\r\"'`.,;:()[]{}")
    date_m = re.match(r"^(\d{4})[-/\s]\d{1,2}[-/\s]\d{1,2}$", text.strip())
    if date_m:
        text = date_m.group(1)
    if len(text.split()) > 10:
        m = re.match(r"^(.*?[.!?])", text)
        if m:
            text = m.group(1).rstrip(".!?").strip()
        else:
            text = " ".join(text.split()[:8])
    return text


def _slug(model_name: str) -> str:
    out = re.sub(r"[^a-zA-Z0-9._-]+", "_", model_name.strip())
    out = re.sub(r"_+", "_", out).strip("_")
    return out or "model"


def _write_jsonl(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _is_truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _serialize_path(path: Path, repo_root: Path) -> str:
    p = path.resolve()
    root = repo_root.resolve()
    try:
        return str(p.relative_to(root)).replace("\\", "/")
    except Exception:
        return p.as_posix()


def _check_url_ok(url: str, timeout_sec: int = 8) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=max(1, int(timeout_sec))) as resp:
            return 200 <= int(getattr(resp, "status", 0) or 0) < 300
    except Exception:
        return False


def _aggregate(predictions: List[dict]) -> dict:
    exact_scores = [float((p.get("scores") or {}).get("exact", 0.0)) for p in predictions]
    contains_scores = [float((p.get("scores") or {}).get("contains", 0.0)) for p in predictions]
    latencies = [float(p.get("latency_sec") or 0.0) for p in predictions]

    by_difficulty: Dict[str, Dict] = defaultdict(lambda: {"n": 0, "exact": [], "contains": []})
    by_domain: Dict[str, Dict] = defaultdict(lambda: {"n": 0, "exact": [], "contains": []})

    for p, e, c in zip(predictions, exact_scores, contains_scores):
        diff = str(p.get("difficulty") or "unknown")
        domain = str(p.get("domain") or "unknown")
        by_difficulty[diff]["n"] += 1
        by_difficulty[diff]["exact"].append(e)
        by_difficulty[diff]["contains"].append(c)
        by_domain[domain]["n"] += 1
        by_domain[domain]["exact"].append(e)
        by_domain[domain]["contains"].append(c)

    def _pack(d: Dict[str, Dict]) -> Dict[str, Dict]:
        return {
            k: {
                "n": v["n"],
                "exact": _mean(v["exact"]),
                "contains": _mean(v["contains"]),
            }
            for k, v in d.items()
        }

    return {
        "n": len(predictions),
        "exact": _mean(exact_scores),
        "contains": _mean(contains_scores),
        "latency_sec_mean": _mean(latencies),
        "by_difficulty": _pack(by_difficulty),
        "by_domain": _pack(by_domain),
    }


def _build_openai_llm(model_name: str, max_tokens: int):
    # gpt-5/o4 families can reject temperature controls on some endpoints.
    if model_name.startswith("gpt-5") or model_name.startswith("o4-"):
        # Reasoning-family models often need a larger completion budget to emit
        # a final short answer token after internal reasoning.
        return ChatOpenAI(model=model_name, max_tokens=max(max_tokens, 256))
    return ChatOpenAI(model=model_name, temperature=0.0, max_tokens=max_tokens)


def _extract_llm_text(content) -> str:
    parts: List[str] = []

    def _collect(value) -> None:
        if value is None:
            return
        if isinstance(value, str):
            text = value.strip()
            if text:
                parts.append(text)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                _collect(item)
            return
        if isinstance(value, dict):
            for key in ("text", "output_text", "content", "message", "answer"):
                if key in value:
                    _collect(value.get(key))
            return

        # Some providers return content blocks as objects instead of dicts.
        for attr in ("text", "output_text", "content", "message"):
            if hasattr(value, attr):
                attr_value = getattr(value, attr)
                # Avoid deprecated message.text() calls; consume only non-callable fields.
                if callable(attr_value):
                    continue
                _collect(attr_value)

        if hasattr(value, "model_dump") and callable(getattr(value, "model_dump")):
            try:
                _collect(value.model_dump())
                return
            except Exception:  # noqa: BLE001
                pass
        if hasattr(value, "dict") and callable(getattr(value, "dict")):
            try:
                _collect(value.dict())
                return
            except Exception:  # noqa: BLE001
                pass

    _collect(content)
    return "\n".join(parts).strip()


def _extract_from_ai_message(msg) -> str:
    text = _extract_llm_text(getattr(msg, "content", msg))
    if text:
        return text

    msg_text = getattr(msg, "text", None)
    if msg_text is not None and not callable(msg_text):
        text = _extract_llm_text(msg_text)
        if text:
            return text

    # Provider-specific fallback fields.
    ak = getattr(msg, "additional_kwargs", {}) or {}
    if isinstance(ak, dict):
        for key in ("text", "output_text", "content"):
            if key in ak:
                t = _extract_llm_text(ak.get(key))
                if t:
                    return t
        t = _extract_llm_text(ak)
        if t:
            return t

    rm = getattr(msg, "response_metadata", {}) or {}
    if isinstance(rm, dict):
        for key in ("text", "output_text", "content"):
            if key in rm:
                t = _extract_llm_text(rm.get(key))
                if t:
                    return t
        t = _extract_llm_text(rm)
        if t:
            return t

    # Last resort: inspect the entire message object for structured text fields.
    return _extract_llm_text(msg)


def _is_transient_openai_error(exc: Exception) -> bool:
    name = type(exc).__name__
    if name in {
        "InternalServerError",
        "RateLimitError",
        "APIConnectionError",
        "APITimeoutError",
        "ServiceUnavailableError",
    }:
        return True
    text = str(exc).lower()
    return (
        "error code: 500" in text
        or "server had an error" in text
        or "temporarily unavailable" in text
        or "timeout" in text
    )


def _invoke_openai(llm, prompt: str) -> str:
    max_attempts = int(os.getenv("OPENAI_TRANSIENT_RETRIES", "5") or 5)
    max_attempts = max(1, max_attempts)

    for attempt in range(1, max_attempts + 1):
        try:
            response = llm.invoke(prompt)
            return _extract_from_ai_message(response)
        except Exception as exc:  # noqa: BLE001
            if attempt >= max_attempts or not _is_transient_openai_error(exc):
                raise
            sleep_s = min(8.0, 0.75 * (2 ** (attempt - 1)))
            print(
                f"[WARN] transient OpenAI error ({type(exc).__name__}) "
                f"attempt {attempt}/{max_attempts}; retrying in {sleep_s:.2f}s"
            )
            time.sleep(sleep_s)

    raise RuntimeError("Unreachable retry branch in _invoke_openai")


def _build_graph_prompt(question: str, answer_text: str, confidence, evidence_json: str) -> str:
    return (
        "You are a precise temporal QA assistant.\n"
        "You are evaluated with exact string match against a short gold answer.\n\n"
        "Rules:\n"
        "- Return exactly one final answer span and nothing else.\n"
        "- Yes/No questions: output exactly Yes or No.\n"
        "- Year questions: output a 4-digit year only.\n"
        "- Entity questions: output only the entity/name phrase.\n"
        "- Do not output explanations.\n\n"
        f"Question:\n{question}\n\n"
        f"Retrieved graph answer:\n{answer_text}\n\n"
        f"Confidence:\n{confidence}\n\n"
        f"Evidence:\n{evidence_json}"
    )


def _build_pure_prompt(question: str) -> str:
    return (
        "You are a precise temporal QA assistant.\n"
        "You are evaluated with exact string match against a short gold answer.\n\n"
        "Rules:\n"
        "- Return exactly one final answer span and nothing else.\n"
        "- Yes/No questions: output exactly Yes or No.\n"
        "- Year questions: output a 4-digit year only.\n"
        "- Entity questions: output only the entity/name phrase.\n"
        "- Do not output explanations.\n\n"
        f"Question:\n{question}"
    )


def _build_pure_retry_prompt(question: str) -> str:
    return (
        "Return ONLY the final answer span and nothing else.\n"
        "Allowed outputs are short: a name, a 4-digit year, Yes, or No.\n"
        "No explanation.\n\n"
        f"Question: {question}"
    )


def _build_pure_force_answer_prompt(question: str) -> str:
    return (
        "Return exactly one short final answer token.\n"
        "Allowed outputs: a name, a 4-digit year, Yes, No, or Unknown.\n"
        "Do not leave output empty. No explanation.\n\n"
        f"Question: {question}"
    )


def _invoke_pure_until_nonempty(llm, question: str) -> tuple[str, int]:
    retries = int(os.getenv("OPENAI_EMPTY_ANSWER_RETRIES", "3") or 3)
    retries = max(0, retries)
    total_attempts = 1 + retries

    last_raw = ""
    for attempt in range(1, total_attempts + 1):
        if attempt == 1:
            prompt = _build_pure_prompt(question)
        elif attempt == 2:
            prompt = _build_pure_retry_prompt(question)
        else:
            prompt = _build_pure_force_answer_prompt(question)

        last_raw = _invoke_openai(llm, prompt)
        if _clean_answer(str(last_raw)):
            return last_raw, attempt

        if attempt < total_attempts:
            print(f"[WARN] empty answer from model on attempt {attempt}/{total_attempts}; retrying")

    return last_raw, total_attempts


def main() -> int:
    parser = argparse.ArgumentParser(description="Run LCEL graph QA with GPT models")
    parser.add_argument("--models", required=True, help="Comma-separated GPT model names")
    parser.add_argument("--eval-set", default="data/jsonls/temporal_evaluation_set_v2.jsonl")
    parser.add_argument("--graph-dir", default="data/jsonls/temporal_graph_output_v3")
    parser.add_argument("--output-dir", default="output/m3_e5_gpt_lcel")
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument(
        "--mode",
        choices=["graph_lcel", "pure_llm"],
        default="graph_lcel",
        help="graph_lcel: graph retrieval + LLM answer; pure_llm: question-only LLM baseline",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Skip model if completed summary exists"
    )
    parser.add_argument(
        "--allow-fallbacks",
        action="store_true",
        help="Allow fallback behavior (disabled by default for experiment integrity)",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to next question when a GPT call fails or returns empty after retries",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    _load_dotenv_if_present(repo_root)
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. Put it in environment or .env")

    strict_no_fallback = not args.allow_fallbacks
    continue_on_error = args.continue_on_error or _is_truthy(
        os.getenv("OPENAI_CONTINUE_ON_ERROR", "1")
    )
    if args.mode == "graph_lcel":
        os.environ["GRAPH_STRICT_NO_FALLBACK"] = "1" if strict_no_fallback else "0"
        os.environ.setdefault("GRAPH_ENABLE_PARAPHRASE", "0")
        os.environ.setdefault("GRAPH_RETRIEVAL_BACKEND", "qwen_server")

        if strict_no_fallback:
            backend = (os.getenv("GRAPH_RETRIEVAL_BACKEND") or "").strip().lower()
            if backend == "qwen_server":
                embed_url = (os.getenv("QWEN_EMBED_URL") or "").strip().rstrip("/")
                if not embed_url:
                    raise SystemExit(
                        "Strict mode requires QWEN_EMBED_URL when GRAPH_RETRIEVAL_BACKEND=qwen_server"
                    )
                # Fail fast if embedding server is not healthy instead of silently falling back.
                if not _check_url_ok(f"{embed_url}/health", timeout_sec=8):
                    raise SystemExit(
                        f"Strict mode: embedding server is not healthy at {embed_url}/health"
                    )
    else:
        # Pure mode does not use graph retrieval, embeddings, or retrieval backend fallbacks.
        os.environ["GRAPH_STRICT_NO_FALLBACK"] = "0"

    eval_path = (repo_root / args.eval_set).resolve()
    graph_dir = (repo_root / args.graph_dir).resolve()
    out_root = (repo_root / args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    rows = list(_iter_jsonl(eval_path))
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        raise SystemExit("No models provided.")

    # Imports here so script remains import-safe in minimal environments.
    sys.path.insert(0, str(repo_root / "src"))
    from temporal_nlg.graph_query.lcel import TemporalGraphLCELPipeline

    matrix: Dict[str, dict] = {}

    for model_name in models:
        run_id = f"lcel__{_slug(model_name)}"
        run_dir = out_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        summary_path = run_dir / "summary.json"
        predictions_path = run_dir / "predictions.jsonl"
        debug_path = run_dir / "debug_log.jsonl"

        if args.resume and summary_path.exists():
            try:
                old_summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if old_summary.get("status") == "completed":
                    print(f"[SKIP] {run_id} already completed")
                    matrix[run_id] = old_summary
                    continue
            except Exception:
                pass

        print(f"[RUN ] {run_id} ({len(rows)} questions)")

        # Configure internal LCEL reasoning to OpenAI so stage selection is not
        # tied to local Qwen server behavior.
        os.environ["GRAPH_INTERNAL_LLM_BACKEND"] = "openai"
        os.environ["GRAPH_INTERNAL_LLM_MODEL"] = model_name
        os.environ.setdefault("GRAPH_PLANNER_BACKEND", "heuristic")
        os.environ.setdefault("GRAPH_ENABLE_PARAPHRASE", "0")

        pipeline = TemporalGraphLCELPipeline(graph_dir) if args.mode == "graph_lcel" else None
        llm = _build_openai_llm(model_name=model_name, max_tokens=args.max_tokens)

        predictions: List[dict] = []
        predictions_path.write_text("", encoding="utf-8")
        debug_path.write_text("", encoding="utf-8")

        running_summary = {
            "run_id": run_id,
            "family": "gpt_lcel",
            "model": model_name,
            "dataset": _serialize_path(eval_path, repo_root),
            "graph_dir": _serialize_path(graph_dir, repo_root),
            "n_questions": 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "metrics": {"n": 0, "exact": 0.0, "contains": 0.0, "latency_sec_mean": 0.0},
        }
        summary_path.write_text(
            json.dumps(running_summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        for idx, item in enumerate(rows, start=1):
            question = str(item.get("question") or item.get("query") or "").strip()
            gold = str(item.get("answer") or item.get("gold_answer") or "").strip()
            difficulty = str(item.get("difficulty") or "unknown")
            domain = str(item.get("domain") or item.get("category_type") or "unknown")

            if not question:
                continue

            t0 = time.perf_counter()
            error = None
            prediction = ""
            graph_answer_text = ""
            graph_conf = None
            evidence = []
            attempts_used = 1
            llm_raw = ""

            try:
                if args.mode == "graph_lcel":
                    graph_ctx = (
                        pipeline.invoke(question, use_llm=False) if pipeline is not None else {}
                    )
                    graph_answer_text = str(graph_ctx.get("answer_text") or "")
                    graph_conf = graph_ctx.get("confidence")
                    evidence = list(graph_ctx.get("evidence") or [])

                    llm_raw = _invoke_openai(
                        llm,
                        _build_graph_prompt(
                            question,
                            graph_answer_text,
                            graph_conf,
                            json.dumps(evidence[:8], ensure_ascii=False),
                        ),
                    )
                else:
                    llm_raw, attempts_used = _invoke_pure_until_nonempty(llm, question)
                prediction = _clean_answer(str(llm_raw))
                if strict_no_fallback and not prediction:
                    raise RuntimeError(
                        "Empty answer returned by model after "
                        f"{attempts_used} attempt(s); "
                        f"idx={idx}; model={model_name}; question={question[:140]!r}; "
                        f"raw_preview={str(llm_raw)[:180]!r}"
                    )
            except Exception as exc:  # noqa: BLE001
                if strict_no_fallback and not continue_on_error:
                    raise
                error = f"{type(exc).__name__}: {exc}"

            latency = time.perf_counter() - t0
            scores = (
                _score_prediction(prediction, gold)
                if not error
                else {"exact": 0.0, "contains": 0.0}
            )

            predictions.append(
                {
                    "idx": idx,
                    "question": question,
                    "gold_answer": gold,
                    "difficulty": difficulty,
                    "domain": domain,
                    "system": run_id,
                    "model": model_name,
                    "prediction": prediction,
                    "scores": scores,
                    "latency_sec": latency,
                    "error": error,
                    "graph_answer_text": graph_answer_text,
                    "graph_confidence": graph_conf,
                }
            )

            _append_jsonl(predictions_path, predictions[-1])

            _append_jsonl(
                debug_path,
                {
                    "idx": idx,
                    "question": question,
                    "gold": gold,
                    "prediction": prediction,
                    "graph_answer_text": graph_answer_text,
                    "graph_answer": graph_answer_text,
                    "graph_evidence_count": len(evidence),
                    "attempts_used": attempts_used,
                    "llm_raw_preview": str(llm_raw)[:280],
                    "error": error,
                },
            )

            if idx % 25 == 0:
                m = _aggregate(predictions)
                running_summary.update(
                    {
                        "n_questions": len(predictions),
                        "status": "running",
                        "metrics": m,
                    }
                )
                summary_path.write_text(
                    json.dumps(running_summary, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                print(f"  {idx}/{len(rows)} exact={m['exact']:.3f} contains={m['contains']:.3f}")

        metrics = _aggregate(predictions)

        summary = {
            "run_id": run_id,
            "family": "gpt_lcel",
            "mode": args.mode,
            "model": model_name,
            "dataset": _serialize_path(eval_path, repo_root),
            "graph_dir": _serialize_path(graph_dir, repo_root),
            "n_questions": len(predictions),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "completed",
            "metrics": metrics,
        }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        matrix[run_id] = summary
        print(f"  files: {predictions_path.name}, {debug_path.name}, {summary_path.name}")
        print(f"  exact={metrics['exact']:.3f} contains={metrics['contains']:.3f}")

    (out_root / "MATRIX.json").write_text(
        json.dumps(matrix, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {(out_root / 'MATRIX.json').as_posix()} ({len(matrix)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

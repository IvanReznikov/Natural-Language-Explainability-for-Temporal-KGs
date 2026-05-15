#!/usr/bin/env python3
"""Generate method/style/granularity predictions for M3-E4.

This script reuses the project dataset to create prediction JSONL files that can
be fed into M3-E4 analyzers. It prioritizes the template renderer and falls
back to gold answers when needed.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional
    load_dotenv = None

from temporal_nlg.core.templates import TemplateRenderer, TemporalFact, TemplateType
from temporal_nlg.evaluation.m3_e3 import bucket_from_time_scope
from temporal_nlg.models import HybridGenerator, LLMGenerator


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _write_jsonl(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _load_dataset_map(path: Optional[Path]) -> Dict[str, dict]:
    if path is None:
        return {}
    out: Dict[str, dict] = {}
    for obj in _iter_jsonl(path):
        rid = obj.get("id")
        if rid:
            out[str(rid)] = obj
    return out


def _event_phrase(fact: dict) -> str:
    subj = str(fact.get("subject") or "").strip()
    rel = str(fact.get("relation") or "").strip()
    obj = str(fact.get("object") or "").strip()
    val = str(fact.get("value") or "").strip()
    parts = [p for p in [subj, rel, obj] if p]
    if not parts and val:
        return val
    if val and not obj:
        parts.append(val)
    return " ".join(parts).strip()


def _fact_from_record(record: dict) -> Optional[TemporalFact]:
    bucket = bucket_from_time_scope(record.get("time_scope"))
    facts = list(record.get("gold_facts") or [])

    if bucket == "causal":
        f = facts[0] if facts else {}
        cause = f.get("subject") or "Event A"
        effect = f.get("object") or f.get("value") or "Event B"
        relation = f.get("relation") or "caused"
        return TemporalFact(
            fact_type=TemplateType.CAUSALITY,
            cause=str(cause),
            effect=str(effect),
            temporal_relation=str(relation),
        )

    if bucket == "point":
        f = facts[0] if facts else {}
        entity = f.get("subject") or f.get("object") or "It"
        event = f.get("relation") or "occurred"
        date = f.get("value") or f.get("start") or f.get("end") or ""
        return TemporalFact(
            fact_type=TemplateType.POINT_IN_TIME,
            entity=str(entity),
            event=str(event),
            date=str(date),
        )

    if bucket == "interval":
        f = facts[0] if facts else {}
        entity = f.get("subject") or f.get("object") or "It"
        event = f.get("relation") or "occurred"
        start = f.get("start") or f.get("value") or ""
        end = f.get("end") or f.get("value") or ""
        return TemporalFact(
            fact_type=TemplateType.INTERVAL,
            entity=str(entity),
            event=str(event),
            start_date=str(start),
            end_date=str(end),
        )

    if bucket == "sequence":
        events: List[str] = []
        times: List[str] = []
        for f in facts:
            phrase = _event_phrase(f)
            if phrase:
                events.append(phrase)
            t = f.get("start") or f.get("value") or f.get("end") or ""
            times.append(str(t) if t else "")
        if not events:
            return None
        return TemporalFact(
            fact_type=TemplateType.SEQUENCE,
            events=events,
            timestamps=times,
            time_span=str(record.get("time_span") or ""),
        )

    if bucket == "overlap":
        f = facts[0] if facts else {}
        events = []
        for k in ("subject", "object"):
            v = f.get(k)
            if v:
                events.append(str(v))
        time_period = ""
        start = f.get("start") or ""
        end = f.get("end") or ""
        if start or end:
            time_period = f"{start}–{end}" if start and end else str(start or end)
        return TemporalFact(
            fact_type=TemplateType.OVERLAP,
            events=events or ["Event A", "Event B"],
            time_period=time_period,
            simultaneity="overlapped",
        )

    return None


def _render_template(record: dict, renderer: TemplateRenderer) -> Optional[str]:
    fact = _fact_from_record(record)
    if fact is None:
        return None
    try:
        return renderer.render(fact)
    except Exception:
        return None


def _base_text(record: dict, dataset_map: Dict[str, dict]) -> str:
    rid = str(record.get("scenario_id") or record.get("id") or "")
    if "explanation_text" in record and record.get("explanation_text"):
        return str(record.get("explanation_text"))
    if rid and rid in dataset_map:
        return str(dataset_map[rid].get("gold_answer") or "")
    return str(record.get("gold_answer") or "")


def _stylize(text: str, style: str) -> str:
    if not text:
        return text
    if style == "hybrid":
        return f"{text} This highlights the main temporal relationship."
    if style == "template":
        return text
    if style == "seq2seq":
        return f"Summary: {text}"
    if style == "llm":
        return f"{text}"
    if style == "baseline":
        return text
    return text


def _granularity_transform(text: str, granularity: str) -> str:
    if not text:
        return text
    if granularity == "decades":
        # crude decade rounding: 1997 -> 1990s
        out = []
        i = 0
        while i < len(text):
            if i + 4 <= len(text) and text[i : i + 4].isdigit():
                year = int(text[i : i + 4])
                decade = year - (year % 10)
                out.append(f"{decade}s")
                i += 4
            else:
                out.append(text[i])
                i += 1
        return f"At decade granularity, {''.join(out)}"
    return f"At {granularity} granularity, {text}"


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text.split()))


def _load_scenarios(path: Path) -> List[dict]:
    return list(_iter_jsonl(path))


def _load_prediction_cache(path: Optional[Path], key_field: str) -> Dict[Tuple[str, str], str]:
    if path is None or not path.exists():
        return {}
    cache: Dict[Tuple[str, str], str] = {}
    for obj in _iter_jsonl(path):
        rid = obj.get("id") or obj.get("scenario_id")
        key = obj.get(key_field)
        if not rid or key is None:
            continue
        text = obj.get("prediction") or obj.get("generated_text") or obj.get("output") or obj.get("text")
        if text is None:
            continue
        cache[(str(rid), str(key))] = str(text)
    return cache


def _load_runs_cache(path: Optional[Path]) -> Dict[Tuple[str, str], dict]:
    if path is None or not path.exists():
        return {}
    cache: Dict[Tuple[str, str], dict] = {}
    for obj in _iter_jsonl(path):
        rid = obj.get("scenario_id") or obj.get("id")
        method = obj.get("method")
        if not rid or method is None:
            continue
        cache[(str(rid), str(method))] = obj
    return cache


def _expand_llm_variants(values: List[str], llm_models: List[str], prefix: str) -> List[str]:
    out: List[str] = []
    for v in values:
        if v == "llm":
            out.extend([f"{prefix}{m}" for m in llm_models])
        else:
            out.append(v)
    return out


def _parse_llm_models(raw: str) -> List[str]:
    return [m.strip() for m in raw.split(",") if m.strip()]


def _llm_generator_for(model: str, temperature: float, max_tokens: int) -> Tuple[Optional[LLMGenerator], Optional[str]]:
    if not os.environ.get("OPENAI_API_KEY"):
        return None, "missing_api_key"
    no_temp_models = {"gpt-5-nano", "o4-mini"}
    try:
        temp = None if model in no_temp_models else temperature
        return LLMGenerator(model=model, temperature=temp, max_tokens=max_tokens), None
    except Exception as exc:
        return None, f"init_error:{type(exc).__name__}"


def _write_log(path: Optional[Path], payload: dict) -> None:
    if path is None:
        return
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def generate_efficiency(args: argparse.Namespace) -> None:
    scenarios = _load_scenarios(Path(args.scenarios))
    dataset_map = _load_dataset_map(Path(args.dataset)) if args.dataset else {}

    pred_cache = _load_prediction_cache(Path(args.output), "method")
    runs_cache = _load_runs_cache(Path(args.runs_out))
    llm_log = Path(args.llm_log) if args.llm_log else None
    if llm_log is not None:
        llm_log.parent.mkdir(parents=True, exist_ok=True)

    renderer = TemplateRenderer()
    hybrid = HybridGenerator()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    llm_models = _parse_llm_models(args.llm_models)
    methods = _expand_llm_variants(methods, llm_models, "llm:")

    llm_gens: Dict[str, Optional[LLMGenerator]] = {}
    llm_gen_errors: Dict[str, Optional[str]] = {}
    for m in llm_models:
        gen, err = _llm_generator_for(m, args.llm_temperature, args.llm_max_tokens)
        llm_gens[m] = gen
        llm_gen_errors[m] = err

    preds: List[dict] = []
    runs: List[dict] = []

    for s in scenarios:
        rid = str(s.get("scenario_id") or s.get("id"))
        record = dataset_map.get(rid, s)
        base = _base_text(record, dataset_map)
        templ = _render_template(record, renderer) or base

        for method in methods:
            cache_key = (rid, method)
            cached_text = pred_cache.get(cache_key)
            if cached_text is not None and args.refresh_llm and method.startswith("llm:"):
                cached_text = None
            if cached_text is not None:
                text = cached_text
                cached_run = runs_cache.get(cache_key)
                latency_ms = cached_run.get("latency_ms") if cached_run else None
                tokens_out = cached_run.get("tokens_out") if cached_run else _estimate_tokens(text)
                cost = cached_run.get("cost") if cached_run else (0.0 if method in {"template", "hybrid", "baseline"} else None)
                if method.startswith("llm:"):
                    _write_log(
                        llm_log,
                        {
                            "scenario_id": rid,
                            "method": method,
                            "status": "cached",
                            "same_as_base": text == base,
                        },
                    )
            else:
                t0 = time.perf_counter()
                if method == "template":
                    text = templ
                elif method == "hybrid":
                    try:
                        fact = _fact_from_record(record)
                        if fact is not None:
                            text = hybrid.generate(fact, force_strategy="template").text
                        else:
                            text = templ
                    except Exception:
                        text = templ
                    text = _stylize(text, "hybrid")
                elif method.startswith("llm:"):
                    model = method.split(":", 1)[1]
                    gen = llm_gens.get(model)
                    fact = _fact_from_record(record)
                    llm_error: Optional[str] = None
                    if gen is not None and fact is not None:
                        try:
                            text = gen.generate(fact)
                        except Exception as exc:
                            llm_error = f"{type(exc).__name__}: {exc}"
                            text = base
                    else:
                        text = base
                elif method == "baseline":
                    text = base
                else:
                    text = base
                latency_ms = (time.perf_counter() - t0) * 1000.0
                tokens_out = _estimate_tokens(text)
                cost = 0.0 if method in {"template", "hybrid", "baseline"} else None
                if method.startswith("llm:"):
                    model = method.split(":", 1)[1]
                    status = "ok"
                    error = None
                    if llm_gens.get(model) is None:
                        status = "unavailable"
                        error = llm_gen_errors.get(model)
                    elif _fact_from_record(record) is None:
                        status = "no_fact"
                    elif llm_error:
                        status = "error"
                        error = llm_error
                    elif text == base:
                        status = "same_as_base"
                    _write_log(
                        llm_log,
                        {
                            "scenario_id": rid,
                            "method": method,
                            "status": status,
                            "error": error,
                            "same_as_base": text == base,
                            "latency_ms": latency_ms,
                        },
                    )

            preds.append({"id": rid, "method": method, "prediction": text})
            runs.append(
                {
                    "scenario_id": rid,
                    "method": method,
                    "complexity_level": s.get("complexity_level"),
                    "latency_ms": latency_ms,
                    "tokens_out": tokens_out,
                    "cost": cost,
                }
            )

    _write_jsonl(Path(args.output), preds)
    _write_jsonl(Path(args.runs_out), runs)

    print(f"Wrote predictions: {args.output}")
    print(f"Wrote runs: {args.runs_out}")


def generate_coherence(args: argparse.Namespace) -> None:
    scenarios = _load_scenarios(Path(args.scenarios))
    dataset_map = _load_dataset_map(Path(args.dataset)) if args.dataset else {}

    pred_cache = _load_prediction_cache(Path(args.output), "style")
    llm_log = Path(args.llm_log) if args.llm_log else None
    if llm_log is not None:
        llm_log.parent.mkdir(parents=True, exist_ok=True)

    renderer = TemplateRenderer()
    hybrid = HybridGenerator()

    styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    llm_models = _parse_llm_models(args.llm_models)
    styles = _expand_llm_variants(styles, llm_models, "llm:")

    llm_gens: Dict[str, Optional[LLMGenerator]] = {}
    llm_gen_errors: Dict[str, Optional[str]] = {}
    for m in llm_models:
        gen, err = _llm_generator_for(m, args.llm_temperature, args.llm_max_tokens)
        llm_gens[m] = gen
        llm_gen_errors[m] = err

    preds: List[dict] = []
    for s in scenarios:
        rid = str(s.get("scenario_id") or s.get("id"))
        record = dataset_map.get(rid, s)
        base = _base_text(record, dataset_map)
        templ = _render_template(record, renderer) or base

        for st in styles:
            cache_key = (rid, st)
            cached_text = pred_cache.get(cache_key)
            if cached_text is not None and args.refresh_llm and st.startswith("llm:"):
                cached_text = None
            if cached_text is not None:
                text = cached_text
                if st.startswith("llm:"):
                    _write_log(
                        llm_log,
                        {
                            "scenario_id": rid,
                            "style": st,
                            "status": "cached",
                            "same_as_base": text == base,
                        },
                    )
            elif st == "template":
                text = templ
            elif st == "hybrid":
                try:
                    fact = _fact_from_record(record)
                    if fact is not None:
                        text = hybrid.generate(fact, force_strategy="template").text
                    else:
                        text = templ
                except Exception:
                    text = templ
                text = _stylize(text, "hybrid")
            elif st == "seq2seq":
                text = _stylize(templ or base, "seq2seq")
            elif st.startswith("llm:"):
                model = st.split(":", 1)[1]
                gen = llm_gens.get(model)
                fact = _fact_from_record(record)
                llm_error: Optional[str] = None
                if gen is not None and fact is not None:
                    try:
                        text = gen.generate(fact)
                    except Exception as exc:
                        llm_error = f"{type(exc).__name__}: {exc}"
                        text = base
                else:
                    text = base
                status = "ok"
                error = None
                if gen is None:
                    status = "unavailable"
                    error = llm_gen_errors.get(model)
                elif fact is None:
                    status = "no_fact"
                elif llm_error:
                    status = "error"
                    error = llm_error
                elif text == base:
                    status = "same_as_base"
                _write_log(
                    llm_log,
                    {
                        "scenario_id": rid,
                        "style": st,
                        "status": status,
                        "error": error,
                        "same_as_base": text == base,
                    },
                )
            elif st == "baseline":
                text = base
            else:
                text = base

            preds.append({"id": rid, "style": st, "prediction": text})

    _write_jsonl(Path(args.output), preds)
    print(f"Wrote predictions: {args.output}")


def generate_granularity(args: argparse.Namespace) -> None:
    scenarios = _load_scenarios(Path(args.scenarios))
    dataset_map = _load_dataset_map(Path(args.dataset)) if args.dataset else {}

    renderer = TemplateRenderer()

    granularities = ["seconds", "minutes", "hours", "days", "months", "years", "decades"]

    preds: List[dict] = []
    for s in scenarios:
        rid = str(s.get("scenario_id") or s.get("id"))
        record = dataset_map.get(rid, s)
        base = _base_text(record, dataset_map)
        templ = _render_template(record, renderer) or base

        for g in granularities:
            text = _granularity_transform(templ, g)
            preds.append({"id": rid, "granularity": g, "prediction": text})

    _write_jsonl(Path(args.output), preds)
    print(f"Wrote predictions: {args.output}")


def main() -> None:
    if load_dotenv is not None:
        load_dotenv()
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_eff = sub.add_parser("efficiency")
    ap_eff.add_argument("--scenarios", type=str, required=True)
    ap_eff.add_argument("--dataset", type=str, default=None)
    ap_eff.add_argument("--methods", type=str, default="template,hybrid,baseline")
    ap_eff.add_argument(
        "--llm-models",
        type=str,
        default="gpt-5-nano,gpt-4.1-nano,gpt-4o,o4-mini,gpt-5.1,gpt-5.2",
    )
    ap_eff.add_argument("--llm-temperature", type=float, default=0.0)
    ap_eff.add_argument("--llm-max-tokens", type=int, default=120)
    ap_eff.add_argument("--refresh-llm", action="store_true", help="Re-generate LLM outputs even if cached")
    ap_eff.add_argument("--llm-log", type=str, default=None, help="Write LLM debug log JSONL")
    ap_eff.add_argument("--output", type=str, required=True)
    ap_eff.add_argument("--runs-out", type=str, required=True)
    ap_eff.set_defaults(func=generate_efficiency)

    ap_coh = sub.add_parser("coherence")
    ap_coh.add_argument("--scenarios", type=str, required=True)
    ap_coh.add_argument("--dataset", type=str, default=None)
    ap_coh.add_argument("--styles", type=str, default="template,seq2seq,llm,hybrid,baseline")
    ap_coh.add_argument(
        "--llm-models",
        type=str,
        default="gpt-5-nano,gpt-4.1-nano,gpt-4o,o4-mini,gpt-5.1,gpt-5.2",
    )
    ap_coh.add_argument("--llm-temperature", type=float, default=0.0)
    ap_coh.add_argument("--llm-max-tokens", type=int, default=120)
    ap_coh.add_argument("--refresh-llm", action="store_true", help="Re-generate LLM outputs even if cached")
    ap_coh.add_argument("--llm-log", type=str, default=None, help="Write LLM debug log JSONL")
    ap_coh.add_argument("--output", type=str, required=True)
    ap_coh.set_defaults(func=generate_coherence)

    ap_gr = sub.add_parser("granularity")
    ap_gr.add_argument("--scenarios", type=str, required=True)
    ap_gr.add_argument("--dataset", type=str, default=None)
    ap_gr.add_argument("--output", type=str, required=True)
    ap_gr.set_defaults(func=generate_granularity)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

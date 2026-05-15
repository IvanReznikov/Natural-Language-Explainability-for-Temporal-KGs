#!/usr/bin/env python3
"""M3-E4c: Cross-Explanation Coherence."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from temporal_nlg.evaluation.m3_e4 import (
    CoherenceScenario,
    CoherenceExplanation,
    CoherenceScore,
    aggregate_coherence,
)


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


def _load_style_predictions(path: Optional[Path]) -> Dict[Tuple[str, str], str]:
    if path is None:
        return {}
    preds: Dict[Tuple[str, str], str] = {}
    for obj in _iter_jsonl(path):
        rid = obj.get("id") or obj.get("scenario_id")
        style = obj.get("style")
        if not rid or not style:
            continue
        text = obj.get("prediction") or obj.get("generated_text") or obj.get("output") or obj.get("text")
        if text is None:
            continue
        preds[(str(rid), str(style))] = str(text)
    return preds


def _parse_styles(raw: Optional[str]) -> List[str]:
    if not raw:
        return ["template", "seq2seq", "llm", "hybrid", "baseline"]
    return [s.strip() for s in raw.split(",") if s.strip()]


def export_scenarios(args: argparse.Namespace) -> None:
    dataset = Path(args.dataset)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(_iter_jsonl(dataset))
    rng = random.Random(args.seed)

    if args.n_scenarios <= 0:
        raise SystemExit("--n-scenarios must be > 0")

    sample = rows if len(rows) <= args.n_scenarios else rng.sample(rows, args.n_scenarios)

    scenarios: List[dict] = []
    explanations: List[dict] = []
    styles = _parse_styles(args.styles)
    preds = _load_style_predictions(Path(args.predictions)) if args.predictions else {}
    if preds:
        pred_styles = sorted({style for (_, style) in preds.keys()})
        if args.styles:
            for st in pred_styles:
                if st not in styles:
                    styles.append(st)
        else:
            styles = pred_styles

    for idx, r in enumerate(sample):
        rid = str(r.get("id") or f"row_{idx}")
        scenarios.append(
            CoherenceScenario(
                scenario_id=rid,
                domain=str(r.get("domain") or "unknown"),
                query=str(r.get("query") or r.get("question") or ""),
                gold_facts=r.get("gold_facts"),
            ).model_dump()
        )

        base_text = str(r.get("gold_answer") or "")
        for st in styles:
            text = preds.get((rid, st)) or base_text
            explanations.append(
                CoherenceExplanation(
                    scenario_id=rid,
                    style=st,
                    text=text,
                ).model_dump()
            )

    scenarios_path = out_dir / "m3_e4c_scenarios.jsonl"
    expl_path = out_dir / "m3_e4c_explanations.jsonl"
    _write_jsonl(scenarios_path, scenarios)
    _write_jsonl(expl_path, explanations)

    template_path = out_dir / "m3_e4c_scores_template.csv"
    with template_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "scenario_id",
                "style",
                "semantic_consistency",
                "narrative_consistency",
                "logical_consistency",
            ],
        )
        w.writeheader()
        for s in scenarios:
            for st in styles:
                w.writerow(
                    {
                        "scenario_id": s["scenario_id"],
                        "style": st,
                        "semantic_consistency": "",
                        "narrative_consistency": "",
                        "logical_consistency": "",
                    }
                )

    print(f"Wrote scenarios: {scenarios_path}")
    print(f"Wrote explanations: {expl_path}")
    print(f"Wrote scores template: {template_path}")


def _read_scores(path: Path) -> List[CoherenceScore]:
    out: List[CoherenceScore] = []
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    out.append(CoherenceScore(**row))
                except Exception:
                    continue
        return out

    for obj in _iter_jsonl(path):
        try:
            out.append(CoherenceScore(**obj))
        except Exception:
            continue
    return out


def _extract_years(text: str) -> List[int]:
    years: List[int] = []
    if not text:
        return years
    tokens = "".join(ch if ch.isdigit() else " " for ch in text).split()
    for t in tokens:
        if len(t) == 4 and t.isdigit():
            y = int(t)
            if 1000 <= y <= 2999:
                years.append(y)
    return years


def _year_sequence(text: str) -> List[int]:
    return _extract_years(text)


def _lcs_len(a: List[int], b: List[int]) -> int:
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        prev = 0
        for j in range(1, len(b) + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[-1]


def _temporal_contradiction(a: str, b: str) -> bool:
    years_a = _extract_years(a)
    years_b = _extract_years(b)
    if not years_a or not years_b:
        return False
    min_a, max_a = min(years_a), max(years_a)
    min_b, max_b = min(years_b), max(years_b)
    # If ranges do not overlap at all, treat as a contradiction signal.
    return max_a < min_b or max_b < min_a


def _score_from_explanations(path: Path, embed_model: str) -> List[CoherenceScore]:
    by_scenario: Dict[str, List[dict]] = {}
    for obj in _iter_jsonl(path):
        sid = obj.get("scenario_id") or obj.get("id")
        if not sid:
            continue
        by_scenario.setdefault(str(sid), []).append(obj)

    out: List[CoherenceScore] = []
    for sid, items in sorted(by_scenario.items()):
        texts = [str(it.get("text") or "") for it in items]
        styles = [str(it.get("style") or "") for it in items]
        if len(texts) < 2:
            continue

        # Semantic consistency via sentence-transformers cosine similarity (fallback to TF-IDF).
        sims = None
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(embed_model)
            embeddings = model.encode(texts, normalize_embeddings=True)
            sims = cosine_similarity(embeddings)
        except Exception:
            vectorizer = TfidfVectorizer(min_df=1)
            tfidf = vectorizer.fit_transform(texts)
            sims = cosine_similarity(tfidf)

        sim_vals: List[float] = []
        narrative_vals: List[float] = []
        logical_vals: List[float] = []

        per_style: Dict[str, Dict[str, List[float]]] = {}

        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                sim = float(sims[i, j])
                sim_vals.append(sim)

                seq_i = _year_sequence(texts[i])
                seq_j = _year_sequence(texts[j])
                lcs = _lcs_len(seq_i, seq_j)
                denom = max(len(seq_i), len(seq_j), 1)
                narrative = lcs / denom
                narrative_vals.append(narrative)

                logical = 0.0 if _temporal_contradiction(texts[i], texts[j]) else 1.0
                logical_vals.append(logical)

                style_i = styles[i] or "unknown"
                style_j = styles[j] or "unknown"
                per_style.setdefault(style_i, {"semantic": [], "narrative": [], "logical": []})
                per_style.setdefault(style_j, {"semantic": [], "narrative": [], "logical": []})
                per_style[style_i]["semantic"].append(sim)
                per_style[style_i]["narrative"].append(narrative)
                per_style[style_i]["logical"].append(logical)
                per_style[style_j]["semantic"].append(sim)
                per_style[style_j]["narrative"].append(narrative)
                per_style[style_j]["logical"].append(logical)

        semantic = sum(sim_vals) / len(sim_vals) if sim_vals else None
        narrative = sum(narrative_vals) / len(narrative_vals) if narrative_vals else None
        logical = sum(logical_vals) / len(logical_vals) if logical_vals else None

        out.append(
            CoherenceScore(
                scenario_id=sid,
                style=None,
                semantic_consistency=semantic,
                narrative_consistency=narrative,
                logical_consistency=logical,
            )
        )

        for style, vals in sorted(per_style.items()):
            out.append(
                CoherenceScore(
                    scenario_id=sid,
                    style=style,
                    semantic_consistency=sum(vals["semantic"]) / len(vals["semantic"]) if vals["semantic"] else None,
                    narrative_consistency=sum(vals["narrative"]) / len(vals["narrative"]) if vals["narrative"] else None,
                    logical_consistency=sum(vals["logical"]) / len(vals["logical"]) if vals["logical"] else None,
                )
            )

    return out


def analyze(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scores: List[CoherenceScore] = []
    if args.scores:
        scores = _read_scores(Path(args.scores))

    if args.explanations:
        scores.extend(_score_from_explanations(Path(args.explanations), args.embed_model))

    summary = aggregate_coherence(scores)

    (out_dir / "m3_e4c_coherence.summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Wrote summary: {out_dir / 'm3_e4c_coherence.summary.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_exp = sub.add_parser("export")
    ap_exp.add_argument("--dataset", type=str, required=True)
    ap_exp.add_argument("--predictions", type=str, default=None, help="JSONL with {id, style, prediction}")
    ap_exp.add_argument("--styles", type=str, default=None, help="Comma-separated style list")
    ap_exp.add_argument("--output-dir", type=str, required=True)
    ap_exp.add_argument("--n-scenarios", type=int, default=100)
    ap_exp.add_argument("--seed", type=int, default=13)
    ap_exp.set_defaults(func=export_scenarios)

    ap_an = sub.add_parser("analyze")
    ap_an.add_argument("--scores", type=str, default=None, help="CSV or JSONL scores file")
    ap_an.add_argument("--explanations", type=str, default=None, help="m3_e4c_explanations.jsonl")
    ap_an.add_argument("--embed-model", type=str, default="all-MiniLM-L6-v2")
    ap_an.add_argument("--output-dir", type=str, required=True)
    ap_an.set_defaults(func=analyze)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

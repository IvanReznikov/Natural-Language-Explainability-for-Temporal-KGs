#!/usr/bin/env python3
"""
M1-E2 Debug: Works with BOTH M1-E1 and M1-E2 results
Usage: python m1_e2_debug.py [result_dir]
Compatible with your existing m1_e1_evaluation_debug.py format
"""

import json
import csv
import sys
from pathlib import Path


def analyze_results(results_dir: str):
    """Universal analyzer for M1-E1/M1-E2 results"""
    path = Path(results_dir)

    # Try M1-E2 first, fallback to M1-E1
    json_files = list(path.glob("m1_e*_report.json")) + list(path.glob("m1_e1_report.json"))
    if not json_files:
        print(f"❌ No report JSON in {path}")
        return

    with open(json_files[0]) as f:
        data = json.load(f)

    renders = data.get("render_details", [])
    print(f"🎯 Analyzing {len(renders)} renders from {json_files[0].name}")

    # Your exact M1-E1 debug output format
    print("\n" + "=" * 100)
    print("DETAILED RENDER ANALYSIS")
    print("=" * 100)

    if not renders:
        print("❌ No render_details found")
        return

    # Worst/Best (exact M1-E1 format)
    renders_sorted = sorted(renders, key=lambda x: x.get("flesch_score", 0))

    print("\n❌ WORST 10 RENDERS")
    print("-" * 100)
    print(f"{'Type':<20} {'Template':<25} {'Flesch':<8} {'Words':<6} {'Output'}")
    for r in renders_sorted[:10]:
        typ, tid = r.get("template_type", "?"), r.get("template_id", "?")
        out, f = r.get("rendered_output", ""), r.get("flesch_score", 0)
        print(f"{typ:<20} {tid:<25} {f:<8.1f} {len(out.split()):<6} {out[:40]}")

    print("\nBEST 10 RENDERS")
    print("-" * 100)
    for r in renders_sorted[-10:]:
        typ, tid = r.get("template_type", "?"), r.get("template_id", "?")
        out, f = r.get("rendered_output", ""), r.get("flesch_score", 0)
        print(f"{typ:<20} {tid:<25} {f:<8.1f} {len(out.split()):<6} {out[:40]}")

    # Save CSV (M1-E1 compatible)
    csv_path = path / "render_details.csv"
    with open(csv_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "template", "flesch", "words", "success", "output"])
        for r in renders:
            writer.writerow(
                [
                    r.get("template_type", ""),
                    r.get("template_id", ""),
                    r.get("flesch_score", 0),
                    len(r.get("rendered_output", "").split()),
                    r.get("success", False),
                    r.get("rendered_output", ""),
                ]
            )
    print(f"\n📁 CSV: {csv_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python m1_e2_debug.py output/m1_e2_llm_nlg/20251214_XXXX")
        sys.exit(1)
    analyze_results(sys.argv[1])

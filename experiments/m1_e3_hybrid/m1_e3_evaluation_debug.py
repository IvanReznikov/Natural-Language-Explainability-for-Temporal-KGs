#!/usr/bin/env python3
"""
M1-E3 Debug: Analyze hybrid evaluation results
Usage: python m1_e3_debug.py [result_dir]
Works with M1-E3 report format
"""

import json
import csv
import sys
from pathlib import Path
from typing import List, Dict, Any


def analyze_results(results_dir: str):
    """Analyze M1-E3 hybrid evaluation results"""
    path = Path(results_dir)

    # Look for M1-E3 report
    json_files = list(path.glob("m1_e3_report.json"))
    if not json_files:
        print(f"❌ No m1_e3_report.json in {path}")
        return

    with open(json_files[0]) as f:
        report = json.load(f)

    print(f"\n🎯 M1-E3 Hybrid Evaluation Results")
    print("=" * 100)

    # Summary metrics
    metrics = report.get("metrics", {})
    print("\n📊 SUMMARY METRICS")
    print("-" * 100)
    print(
        f"Flesch Score:       {metrics.get('flesch_mean', 0):.1f} ± {metrics.get('flesch_std', 0):.1f}"
    )
    print(f"Success Rate:       {metrics.get('success_rate', 0)*100:.1f}%")
    print(f"Latency p50:        {metrics.get('latency_p50', 0):.0f}ms")
    print(f"Latency p95:        {metrics.get('latency_p95', 0):.0f}ms")
    print(f"Latency p99:        {metrics.get('latency_p99', 0):.0f}ms")
    print(f"Accuracy:           {metrics.get('accuracy', 0)*100:.1f}%")
    print(f"Type Coverage:      {metrics.get('type_coverage', 0)*100:.1f}%")
    print(f"Template Hit Rate:  {metrics.get('template_hit_rate', 0)*100:.1f}%")
    print(f"Polishing Rate:     {metrics.get('polishing_rate', 0)*100:.1f}%")
    print(f"Hallucination Rate: {metrics.get('hallucination_rate', 0)*100:.1f}%")

    # Check success criteria
    print("\nSUCCESS CRITERIA")
    print("-" * 100)
    checks = {
        "Flesch > 70": metrics.get("flesch_mean", 0) > 70,
        "Success > 98%": metrics.get("success_rate", 0) > 0.98,
        "Latency p95 < 200ms": metrics.get("latency_p95", 0) < 200,
        "Accuracy > 99%": metrics.get("accuracy", 0) > 0.99,
        "Type Coverage > 95%": metrics.get("type_coverage", 0) > 0.95,
    }
    for check, result in checks.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{check:<30} {status}")

    all_pass = all(checks.values())
    print(f"\n{'='*100}")
    print(f"OVERALL: {'✓ ALL CRITERIA PASS' if all_pass else '✗ SOME CRITERIA FAIL'}")
    print(f"{'='*100}\n")

    # Load render details from CSV if available
    csv_path = path / "render_details.csv"
    if csv_path.exists():
        print(f"\n📋 RENDER DETAILS ANALYSIS")
        print("-" * 100)

        renders = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            renders = list(reader)

        print(f"Total renders: {len(renders)}")

        # Convert string numbers back to float
        for r in renders:
            try:
                r["flesch"] = float(r.get("flesch", 0))
                r["words"] = int(r.get("words", 0))
                r["latency_ms"] = float(r.get("latency_ms", 0))
            except ValueError:
                pass

        # Sort by Flesch score
        renders_sorted = sorted(renders, key=lambda x: x.get("flesch", 0))

        # Worst renders
        print("\n❌ WORST 10 RENDERS (Lowest Flesch)")
        print("-" * 100)
        print(
            f"{'Type':<15} {'Template':<25} {'Source':<12} {'Flesch':<8} {'Words':<6} {'Output':<40}"
        )
        print("-" * 100)
        for r in renders_sorted[:10]:
            typ = r.get("type", "?")
            template = r.get("template_id", "?")
            source = r.get("source", "?")
            flesch = float(r.get("flesch", 0))
            words = int(r.get("words", 0))
            output = r.get("output", "")[:40]
            print(f"{typ:<15} {template:<25} {source:<12} {flesch:<8.1f} {words:<6} {output}")

        # Best renders
        print("\nBEST 10 RENDERS (Highest Flesch)")
        print("-" * 100)
        print(
            f"{'Type':<15} {'Template':<25} {'Source':<12} {'Flesch':<8} {'Words':<6} {'Output':<40}"
        )
        print("-" * 100)
        for r in renders_sorted[-10:]:
            typ = r.get("type", "?")
            template = r.get("template_id", "?")
            source = r.get("source", "?")
            flesch = float(r.get("flesch", 0))
            words = int(r.get("words", 0))
            output = r.get("output", "")[:40]
            print(f"{typ:<15} {template:<25} {source:<12} {flesch:<8.1f} {words:<6} {output}")

        # Per-type breakdown
        print("\n📊 PER-TYPE BREAKDOWN")
        print("-" * 100)
        types = {}
        for r in renders:
            typ = r.get("type", "unknown")
            if typ not in types:
                types[typ] = {"flesch": [], "words": [], "source": {}, "success": 0}
            types[typ]["flesch"].append(float(r.get("flesch", 0)))
            types[typ]["words"].append(int(r.get("words", 0)))
            source = r.get("source", "unknown")
            types[typ]["source"][source] = types[typ]["source"].get(source, 0) + 1

        for typ in sorted(types.keys()):
            data = types[typ]
            avg_flesch = sum(data["flesch"]) / len(data["flesch"])
            avg_words = sum(data["words"]) / len(data["words"])
            source_str = ", ".join([f"{s}:{c}" for s, c in data["source"].items()])
            print(
                f"{typ:<20} Flesch: {avg_flesch:6.1f}  Words: {avg_words:5.1f}  Sources: {source_str}"
            )

        # Source breakdown (template vs polished)
        print("\n🔀 SOURCE BREAKDOWN")
        print("-" * 100)
        sources = {}
        for r in renders:
            source = r.get("source", "unknown")
            if source not in sources:
                sources[source] = {"count": 0, "flesch": []}
            sources[source]["count"] += 1
            sources[source]["flesch"].append(float(r.get("flesch", 0)))

        for source in sorted(sources.keys()):
            data = sources[source]
            avg_flesch = sum(data["flesch"]) / len(data["flesch"])
            pct = data["count"] / len(renders) * 100
            print(
                f"{source:<15} {data['count']:4} renders ({pct:5.1f}%)  Avg Flesch: {avg_flesch:6.1f}"
            )

    print(f"\n{'='*100}")
    print(f"Timestamp: {report.get('timestamp', 'unknown')}")
    print(f"Model: {report.get('model', 'unknown')}")
    print(f"Examples per type: {report.get('examples_per_type', 'unknown')}")
    print(f"Duration: {report.get('duration_seconds', 0):.1f} seconds")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python m1_e3_debug.py output/m1_e3_hybrid/20251215_XXXX")
        sys.exit(1)
    analyze_results(sys.argv[1])

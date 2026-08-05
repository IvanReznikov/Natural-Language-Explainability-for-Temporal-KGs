#!/usr/bin/env python3
"""
M1-E4 Debug: Analyze accuracy evaluation results
Usage: python m1_e4_debug.py [result_dir]
Works with M1-E4 report format
"""

import json
import csv
import sys
from pathlib import Path
from typing import List, Dict, Any


def analyze_results(results_dir: str):
    """Analyze M1-E4 accuracy evaluation results"""
    path = Path(results_dir)

    # Look for M1-E4 report
    json_files = list(path.glob("m1_e4_report.json"))
    if not json_files:
        print(f"❌ No m1_e4_report.json in {path}")
        return

    with open(json_files[0]) as f:
        report = json.load(f)

    print(f"\n🎯 M1-E4 ACCURACY VALIDATION RESULTS")
    print("=" * 100)

    # Summary metrics
    metrics = report.get("metrics", {})
    print("\n📊 SUMMARY METRICS")
    print("-" * 100)
    print(f"Overall Accuracy:      {metrics.get('overall_accuracy', 0)*100:.2f}%")
    print(f"Date Preservation:     {metrics.get('date_preservation_mean', 0)*100:.2f}%")
    print(f"Entity Preservation:   {metrics.get('entity_preservation_mean', 0)*100:.2f}%")
    print(f"Relation Preservation: {metrics.get('relation_preservation_mean', 0)*100:.2f}%")
    print(f"Entity F1 Score:       {metrics.get('entity_f1', 0):.4f}")
    print(f"Hallucination Rate:    {metrics.get('hallucination_rate', 0)*100:.2f}%")

    # Check success criteria
    print("\nSUCCESS CRITERIA")
    print("-" * 100)
    checks = {
        "Overall Accuracy > 99%": metrics.get("overall_accuracy", 0) > 0.99,
        "Entity F1 > 0.95": metrics.get("entity_f1", 0) > 0.95,
        "Hallucination Rate < 1%": metrics.get("hallucination_rate", 0) < 0.01,
        "Template > Polished": metrics.get("template_accuracy", 0)
        > metrics.get("polished_accuracy", 0),
    }
    for check, result in checks.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{check:<35} {status}")

    all_pass = all(checks.values())
    print(f"\n{'='*100}")
    print(f"OVERALL: {'✓ ALL CRITERIA PASS' if all_pass else '✗ SOME CRITERIA FAIL'}")
    print(f"{'='*100}\n")

    # Source comparison
    print("\n🔀 SOURCE COMPARISON")
    print("-" * 100)
    print(f"Template Accuracy: {metrics.get('template_accuracy', 0)*100:6.2f}%")
    print(f"Polished Accuracy: {metrics.get('polished_accuracy', 0)*100:6.2f}%")
    print(f"Gap (Template - Polished): {metrics.get('accuracy_gap', 0)*100:+6.2f}%")

    # Type breakdown
    print("\n📊 TYPE-SPECIFIC ACCURACY")
    print("-" * 100)
    type_accs = metrics.get("type_accuracies", {})
    for fact_type in sorted(type_accs.keys()):
        acc = type_accs[fact_type]
        status = "✓" if acc > 0.98 else "✗"
        print(f"{status} {fact_type:15}: {acc*100:6.2f}%")

    # Load detailed results if available
    csv_path = path / "accuracy_details.csv"
    if csv_path.exists():
        print(f"\n📋 DETAILED ACCURACY ANALYSIS")
        print("-" * 100)

        results = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            results = list(reader)

        print(f"Total evaluations: {len(results)}")

        # Convert to float
        for r in results:
            try:
                r["overall_accuracy"] = float(r.get("overall_accuracy", 0))
                r["date"] = float(r.get("date", 0))
                r["entity"] = float(r.get("entity", 0))
                r["relation"] = float(r.get("relation", 0))
            except ValueError:
                pass

        # Sort by accuracy
        results_sorted = sorted(results, key=lambda x: x.get("overall_accuracy", 0))

        # Worst cases
        print("\n❌ WORST 10 ACCURACY SCORES")
        print("-" * 100)
        print(
            f"{'Type':<15} {'Source':<10} {'Overall':<10} {'Date':<8} {'Entity':<8} {'Relation':<10}"
        )
        print("-" * 100)
        for r in results_sorted[:10]:
            typ = r.get("type", "?")
            source = r.get("source", "?")
            overall = float(r.get("overall_accuracy", 0))
            date = float(r.get("date", 0))
            entity = float(r.get("entity", 0))
            relation = float(r.get("relation", 0))
            print(
                f"{typ:<15} {source:<10} {overall:<10.4f} {date:<8.4f} {entity:<8.4f} {relation:<10.4f}"
            )

        # Best cases
        print("\nBEST 10 ACCURACY SCORES")
        print("-" * 100)
        print(
            f"{'Type':<15} {'Source':<10} {'Overall':<10} {'Date':<8} {'Entity':<8} {'Relation':<10}"
        )
        print("-" * 100)
        for r in results_sorted[-10:]:
            typ = r.get("type", "?")
            source = r.get("source", "?")
            overall = float(r.get("overall_accuracy", 0))
            date = float(r.get("date", 0))
            entity = float(r.get("entity", 0))
            relation = float(r.get("relation", 0))
            print(
                f"{typ:<15} {source:<10} {overall:<10.4f} {date:<8.4f} {entity:<8.4f} {relation:<10.4f}"
            )

        # Hallucination analysis
        halluc_count = sum(1 for r in results if r.get("hallucinated", "").lower() == "true")
        print(f"\n🚨 HALLUCINATION ANALYSIS")
        print("-" * 100)
        print(f"Hallucinated: {halluc_count}/{len(results)} ({halluc_count/len(results)*100:.1f}%)")

        halluc_types = {}
        for r in results:
            if r.get("hallucinated", "").lower() == "true":
                h_type = r.get("hallucination_type", "unknown")
                halluc_types[h_type] = halluc_types.get(h_type, 0) + 1

        if halluc_types:
            print("\nHallucination types:")
            for h_type in sorted(halluc_types.keys()):
                count = halluc_types[h_type]
                print(f"  {h_type}: {count}")

    print(f"\n{'='*100}")
    print(f"Timestamp: {report.get('timestamp', 'unknown')}")
    print(f"Model: {report.get('model', 'unknown')}")
    print(f"Duration: {report.get('duration_seconds', 0):.1f} seconds")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python m1_e4_debug.py output/m1_e4_accuracy/20251215_XXXX")
        sys.exit(1)
    analyze_results(sys.argv[1])

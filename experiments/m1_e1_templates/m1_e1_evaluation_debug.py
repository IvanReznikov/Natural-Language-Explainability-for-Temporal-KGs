"""
M1-E1 Evaluation Debug Tool

Shows actual rendered outputs to diagnose why Flesch scores are low.
This works with the render_details field from m1_e1_evaluation_v2.py
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Any


def analyze_renders(results_dir: str) -> None:
    """Load and analyze renders from evaluation results."""
    results_path = Path(results_dir)
    json_file = results_path / "m1_e1_report.json"

    if not json_file.exists():
        print(f"No results file found at {json_file}")
        return

    with open(json_file) as f:
        data = json.load(f)

    # Extract all renders
    renders = data.get("render_details", [])

    print("\n" + "=" * 100)
    print("DETAILED RENDER ANALYSIS")
    print("=" * 100 + "\n")

    if not renders:
        print("❌ No `render_details` found in report JSON.")
        print("Make sure m1_e1_evaluation_v2.py is saving render_details.")
        print()
        return

    print(f"✓ Found {len(renders)} render records to analyze\n")

    # Sort by Flesch score
    renders_sorted = sorted(renders, key=lambda x: x.get("flesch_score", 0.0))

    # Show worst renders (low Flesch)
    print("❌ WORST 10 RENDERS (Lowest Flesch Scores)")
    print("-" * 100)
    print(f"{'Type':<20} {'Template':<25} {'Flesch':<8} " f"{'Words':<6} {'Output':<40}")
    print("-" * 100)

    for i, render in enumerate(renders_sorted[:10], 1):
        template_type = render.get("template_type", "?")
        template_id = render.get("template_id", "?")
        output_text = render.get("rendered_output", "")
        flesch = render.get("flesch_score", 0.0)
        word_count = len(output_text.split())
        output_preview = output_text[:37] + "..." if len(output_text) > 40 else output_text

        print(
            f"{template_type:<20} {template_id:<25} "
            f"{flesch:<8.1f} {word_count:<6} {output_preview:<40}"
        )

    print("\n")

    # Show best renders (high Flesch)
    print("BEST 10 RENDERS (Highest Flesch Scores)")
    print("-" * 100)
    print(f"{'Type':<20} {'Template':<25} {'Flesch':<8} " f"{'Words':<6} {'Output':<40}")
    print("-" * 100)

    for i, render in enumerate(renders_sorted[-10:], 1):
        template_type = render.get("template_type", "?")
        template_id = render.get("template_id", "?")
        output_text = render.get("rendered_output", "")
        flesch = render.get("flesch_score", 0.0)
        word_count = len(output_text.split())
        output_preview = output_text[:37] + "..." if len(output_text) > 40 else output_text

        print(
            f"{template_type:<20} {template_id:<25} "
            f"{flesch:<8.1f} {word_count:<6} {output_preview:<40}"
        )

    print("\n")

    # Show statistics by type
    print("📊 STATISTICS BY TYPE")
    print("-" * 100)

    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for render in renders:
        template_type = render.get("template_type", "unknown")
        by_type.setdefault(template_type, []).append(render)

    for template_type in sorted(by_type.keys()):
        renders_of_type = by_type[template_type]
        flesch_scores = [
            r.get("flesch_score", 0.0) for r in renders_of_type if r.get("success", False)
        ]
        word_counts = [
            len(r.get("rendered_output", "").split())
            for r in renders_of_type
            if r.get("success", False)
        ]

        if not flesch_scores:
            print(f"\n{template_type}: NO SUCCESSFUL RENDERS")
            continue

        avg_flesch = sum(flesch_scores) / len(flesch_scores)
        avg_words = sum(word_counts) / len(word_counts) if word_counts else 0.0
        min_flesch = min(flesch_scores)
        max_flesch = max(flesch_scores)
        success_count = len([r for r in renders_of_type if r.get("success", False)])
        fail_count = len([r for r in renders_of_type if not r.get("success", False)])

        print(f"\n{template_type}:")
        print(f"  Renders: {success_count} successful, {fail_count} failed")
        print(f"  Flesch: avg={avg_flesch:.1f}, min={min_flesch:.1f}, " f"max={max_flesch:.1f}")
        print(f"  Avg words per render: {avg_words:.1f}")
        print("  Sample outputs:")

        for render in renders_of_type[:2]:
            if render.get("success"):
                output = render.get("rendered_output", "")
                flesch = render.get("flesch_score", 0.0)
                print(f'    "{output[:60]}..." (Flesch: {flesch:.1f})')

    print("\n")

    # Save detailed CSV
    csv_path = results_path / "render_details.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "template_type",
                "template_id",
                "flesch_score",
                "word_count",
                "success",
                "rendered_output",
            ]
        )
        for render in renders_sorted:
            writer.writerow(
                [
                    render.get("template_type", ""),
                    render.get("template_id", ""),
                    render.get("flesch_score", 0.0),
                    len(render.get("rendered_output", "").split()),
                    render.get("success", False),
                    render.get("rendered_output", ""),
                ]
            )

    print(f"📁 Detailed CSV saved to: {csv_path}\n")

    # Key insight
    print("=" * 100)
    print("🔍 KEY INSIGHT: CONTENT vs TEMPLATE")
    print("=" * 100)

    # Compare top and bottom
    successful = [r for r in renders if r.get("success", False)]
    if len(successful) < 10:
        print("\nNot enough successful renders to analyze.\n")
        return

    worst_subset = sorted(successful, key=lambda x: x.get("flesch_score", 0.0))[:100]
    best_subset = sorted(successful, key=lambda x: x.get("flesch_score", 0.0))[-100:]

    worst_words = (
        sum(len(r.get("rendered_output", "").split()) for r in worst_subset) / len(worst_subset)
        if worst_subset
        else 0
    )
    best_words = (
        sum(len(r.get("rendered_output", "").split()) for r in best_subset) / len(best_subset)
        if best_subset
        else 0
    )

    worst_flesch = sum(r.get("flesch_score", 0.0) for r in worst_subset) / len(worst_subset)
    best_flesch = sum(r.get("flesch_score", 0.0) for r in best_subset) / len(best_subset)

    print(f"""
The difference between LOW and HIGH Flesch renders is CONTENT LENGTH:

  Worst 100 renders: avg {worst_words:.1f} words, avg {worst_flesch:.1f} Flesch
  Best 100 renders:  avg {best_words:.1f} words, avg {best_flesch:.1f} Flesch
  
  Difference: {worst_words - best_words:.1f} words = {best_flesch - worst_flesch:.1f} Flesch points

INTERPRETATION:

If word count difference is large (>2 words):
  → Problem: Fact values are too verbose
  → Solution: Shorten values in loaders.py
  
If word count is similar but Flesch differs:
  → Problem: Template/syllable structure
  → Solution: Redesign templates
""")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python m1_e1_evaluation_debug.py <results_dir>")
        print()
        print("Example:")
        print("  python m1_e1_evaluation_debug.py " "output/m1_e1_templates/20251213_232903")
        sys.exit(1)

    results_dir_arg = sys.argv[1]
    analyze_renders(results_dir_arg)

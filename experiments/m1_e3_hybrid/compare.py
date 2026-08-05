"""
M1-E3: Compare Results - Compare M1-E1, M1-E2, and M1-E3 results
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd


class M1Comparator:
    """Compare results across M1-E1, M1-E2, M1-E3."""

    def __init__(self):
        self.e1_results: Optional[Dict] = None
        self.e2_results: Optional[Dict] = None
        self.e3_results: Optional[Dict] = None

    def load_e1(self, path: Path) -> None:
        """Load M1-E1 results."""
        with open(path) as f:
            self.e1_results = json.load(f)
        print(f"✓ Loaded E1 from {path}")

    def load_e2(self, path: Path) -> None:
        """Load M1-E2 results."""
        with open(path) as f:
            self.e2_results = json.load(f)
        print(f"✓ Loaded E2 from {path}")

    def load_e3(self, path: Path) -> None:
        """Load M1-E3 results."""
        with open(path) as f:
            self.e3_results = json.load(f)
        print(f"✓ Loaded E3 from {path}")

    def compare(self) -> str:
        """Generate comparison report."""
        if not all([self.e1_results, self.e2_results, self.e3_results]):
            return "❌ Missing results. Load all three first."

        lines = []
        lines.append("=" * 80)
        lines.append("M1-E1 vs M1-E2 vs M1-E3 COMPARISON")
        lines.append("=" * 80)

        # Extract metrics
        e1_summary = self.e1_results.get("summary", {})
        e2_summary = self.e2_results.get("summary", {})
        e3_summary = self.e3_results.get("summary", {})

        # Build comparison table
        metrics = [
            ("Flesch Score", "flesch_score", lambda x: f"{x:.1f}"),
            ("Success Rate (%)", "success_rate", lambda x: f"{x:.1f}%"),
            ("Latency (ms)", "latency_ms", lambda x: f"{x:.0f}"),
            ("Accuracy (%)", "accuracy", lambda x: f"{x:.1f}%"),
            ("Type Coverage (%)", "type_coverage", lambda x: f"{x:.1f}%"),
        ]

        lines.append("\n| Metric | M1-E1 | M1-E2 | M1-E3 | Winner |")
        lines.append("|--------|-------|-------|-------|--------|")

        for metric_name, key, formatter in metrics:
            e1_val = e1_summary.get(key, 0)
            e2_val = e2_summary.get(key, 0)
            e3_val = e3_summary.get(key, 0)

            # Determine winner (higher is better except latency)
            if key == "latency_ms":
                winner = (
                    "E1"
                    if e1_val == min(e1_val, e2_val, e3_val)
                    else ("E2" if e2_val < e3_val else "E3")
                )
            else:
                winner = (
                    "E1"
                    if e1_val == max(e1_val, e2_val, e3_val)
                    else ("E2" if e2_val > e3_val else "E3")
                )

            lines.append(
                f"| {metric_name} | {formatter(e1_val)} | {formatter(e2_val)} | {formatter(e3_val)} | **{winner}** |"
            )

        # Calculate improvements
        lines.append("\n## E3 vs E2 Improvements")

        for metric_name, key, formatter in metrics:
            e2_val = e2_summary.get(key, 0)
            e3_val = e3_summary.get(key, 0)

            if e2_val == 0:
                pct_change = 0
            elif key == "latency_ms":
                pct_change = (e2_val - e3_val) / e2_val * 100
                direction = "↓ faster" if pct_change > 0 else "↑ slower"
            else:
                pct_change = (e3_val - e2_val) / e2_val * 100
                direction = "↑ better" if pct_change > 0 else "↓ worse"

            lines.append(f"- {metric_name}: {pct_change:+.1f}% {direction}")

        lines.append("\n" + "=" * 80)

        return "\n".join(lines)


class M1DebugAnalyzer:
    """Debug analysis of M1-E3 results."""

    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)

    def analyze(self) -> str:
        """Generate debug report."""
        lines = []
        lines.append("=" * 80)
        lines.append("M1-E3 DEBUG ANALYSIS")
        lines.append("=" * 80)

        # Check for result files
        report_path = self.results_dir / "m1_e3_report.json"
        details_path = self.results_dir / "render_details.csv"

        if not report_path.exists():
            lines.append(f"❌ Report not found: {report_path}")
            return "\n".join(lines)

        with open(report_path) as f:
            report = json.load(f)

        summary = report.get("summary", {})

        lines.append("\n## Summary Metrics")
        for key, value in summary.items():
            if isinstance(value, float):
                lines.append(f"- {key}: {value:.2f}")
            else:
                lines.append(f"- {key}: {value}")

        # Analyze renders if details available
        if details_path.exists():
            df = pd.read_csv(details_path)

            lines.append("\n## Render Details (Top 5 Worst)")

            # Sort by flesch_score ascending
            worst = df.nsmallest(5, "flesch_score")
            for idx, row in worst.iterrows():
                lines.append(
                    f"- Type: {row.get('fact_type')}, "
                    f"Template: {row.get('best_template')}, "
                    f"Flesch: {row.get('flesch_score', 0):.1f}"
                )

        lines.append("\n" + "=" * 80)

        return "\n".join(lines)


if __name__ == "__main__":
    # Example usage
    comparator = M1Comparator()

    # Load results (update paths as needed)
    # comparator.load_e1(Path("../m1_e1_templates/results/.../m1_e1_report.json"))
    # comparator.load_e2(Path("../m1_e2_llm_nlg/results/.../m1_e2_report.json"))
    # comparator.load_e3(Path("results/.../m1_e3_report.json"))

    # report = comparator.compare()
    # print(report)

    print("Usage: See code comments for how to load and compare results")

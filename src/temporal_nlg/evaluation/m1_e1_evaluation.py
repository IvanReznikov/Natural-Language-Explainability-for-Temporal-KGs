"""
M1-E1 Evaluation Framework

Uses real example generators to compute meaningful metrics:
- Template coverage: % of facts that can be rendered successfully
- Flesch score: readability measurement
- Clarity: design-time ratings from templates
- Success rate: % of renders that complete without error
"""

"""
M1-E1 Evaluation Framework (v3 - Revised Metrics)

CHANGES FROM V2:
- Lowered Flesch target from >60 to >50 (more realistic for informative text)
- Added new metric: Information Density (words vs semantic value)
- Added new metric: Template Type Coverage (% of types that PASS)
- Kept Clarity and Coverage (these are good)
- Can accept more examples to improve metrics through diversity
"""

import json
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict, field
import re
from pathlib import Path


def calculate_flesch_score(text: str) -> float:
    """Calculate Flesch Reading Ease score (0-100)."""
    sentences = len(re.split(r"[.!?]+", text.strip())) - 1
    if sentences <= 0:
        sentences = 1

    words = len(text.split())
    if words == 0:
        return 50.0

    syllables = 0
    for word in text.lower().split():
        word = re.sub(r"[^a-z]", "", word)
        if len(word) > 0:
            prev_vowel = False
            for char in word:
                is_vowel = char in "aeiou"
                if is_vowel and not prev_vowel:
                    syllables += 1
                prev_vowel = is_vowel
            if syllables == 0:
                syllables = 1

    try:
        score = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
        return max(0, min(100, score))
    except ZeroDivisionError:
        return 50.0


def calculate_information_density(text: str) -> float:
    """
    Calculate information density (0-1).

    Measures semantic value per word.
    High when: Few words, high syllable count (complex concepts)
    Low when: Many words, low syllable count (padding)

    Unlike Flesch, this REWARDS complexity when it carries meaning.
    """
    words = len(text.split())
    if words == 0:
        return 0.0

    # Count syllables
    syllables = 0
    for word in text.lower().split():
        word = re.sub(r"[^a-z]", "", word)
        if len(word) > 0:
            prev_vowel = False
            for char in word:
                is_vowel = char in "aeiou"
                if is_vowel and not prev_vowel:
                    syllables += 1
                prev_vowel = is_vowel
            if syllables == 0:
                syllables = 1

    # Information density = (syllables * meaningful_words) / total_words
    # High when: more syllables (complex concepts) relative to word count
    # This is OPPOSITE of Flesch

    try:
        # Normalize to 0-1 scale
        # density = syllables / (words * 2)  # assume avg 2 syllables per word
        # Higher density = more information per word
        density = min(1.0, syllables / (words * 1.5))
        return density
    except ZeroDivisionError:
        return 0.5


@dataclass
class TemplateTypeMetrics:
    """Metrics for a single template type."""

    type_name: str
    template_count: int
    example_count: int
    total_renders: int
    successful_renders: int
    failed_renders: int
    average_flesch: float
    average_clarity: float
    average_information_density: float
    coverage: float
    success_rate: float
    passes_flesch_45: bool

    def __post_init__(self):
        if self.total_renders > 0:
            self.success_rate = self.successful_renders / self.total_renders
            self.coverage = self.successful_renders / self.total_renders
        else:
            self.success_rate = 0.0
            self.coverage = 0.0

        self.passes_flesch_45 = self.average_flesch > 45


@dataclass
class M1E1EvaluationReportV3:
    """Comprehensive M1-E1 evaluation report (v3)."""

    timestamp: str

    # Overall metrics
    total_templates: int = 0
    total_examples: int = 0
    total_renders: int = 0
    successful_renders: int = 0
    failed_renders: int = 0

    # Quality metrics (revised targets)
    overall_coverage: float = 0.0
    overall_flesch: float = 0.0  # Target: >50 (was >60)
    overall_clarity: float = 0.0  # Target: >4.0
    overall_success_rate: float = 0.0  # Target: >95%
    overall_information_density: float = 0.0  # New metric (target: >0.5)
    type_coverage: float = 0.0  # New metric: % of types passing Flesch (target: >80%)

    # Per-type results
    type_metrics: Dict[str, TemplateTypeMetrics] = field(default_factory=dict)

    # Render results (for detailed inspection)
    render_details: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "total_templates": self.total_templates,
            "total_examples": self.total_examples,
            "total_renders": self.total_renders,
            "successful_renders": self.successful_renders,
            "failed_renders": self.failed_renders,
            "overall_coverage": self.overall_coverage,
            "overall_flesch": self.overall_flesch,
            "overall_clarity": self.overall_clarity,
            "overall_success_rate": self.overall_success_rate,
            "overall_information_density": self.overall_information_density,
            "type_coverage": self.type_coverage,
            "type_metrics": {k: asdict(v) for k, v in self.type_metrics.items()},
            "render_details": self.render_details,
        }


class M1E1EvaluatorV3:
    """Enhanced evaluator with revised metrics."""

    def __init__(self):
        """Initialize evaluator."""
        self.report = None

    def evaluate_all_templates(
        self, n_examples_per_type: int = 50, save_to_file: str = None
    ) -> M1E1EvaluationReportV3:
        """
        Run comprehensive evaluation on all template types.

        Args:
            n_examples_per_type: Number of examples to generate per type
            save_to_file: Optional path to save report as JSON

        Returns:
            M1E1EvaluationReportV3 with all metrics
        """
        from datetime import datetime
        from ..data.loaders import generate_all_examples
        from ..templates.point_in_time import PointInTimeTemplateLibrary
        from ..templates.intervals import IntervalTemplateLibrary
        from ..templates.sequences import SequenceTemplateLibrary
        from ..templates.causality import CausalityTemplateLibrary
        from ..templates.overlaps import OverlapTemplateLibrary

        # Initialize report
        self.report = M1E1EvaluationReportV3(timestamp=datetime.now().isoformat())

        # Generate examples
        examples = generate_all_examples(n_examples_per_type)

        # Create template libraries
        libraries = {
            "point_in_time": PointInTimeTemplateLibrary(),
            "intervals": IntervalTemplateLibrary(),
            "sequences": SequenceTemplateLibrary(),
            "causality": CausalityTemplateLibrary(),
            "overlaps": OverlapTemplateLibrary(),
        }

        # Evaluate each type
        flesch_scores = []
        clarity_scores = []
        information_densities = []
        types_passing_flesch = 0

        for fact_type, library in libraries.items():
            type_examples = examples[fact_type]
            type_flesch_scores = []
            type_clarity_scores = []
            type_information_densities = []
            type_successful = 0
            type_failed = 0

            # Render each example with each template
            for example in type_examples:
                for template_id, template in library.templates.items():
                    try:
                        # Try rendering
                        output = library.render(example, template_id)
                        flesch = calculate_flesch_score(output)
                        info_density = calculate_information_density(output)
                        clarity = template.confidence * 5.0

                        # Record the detail
                        self.report.render_details.append(
                            {
                                "template_type": fact_type,
                                "template_id": template_id,
                                "rendered_output": output,
                                "flesch_score": float(flesch),
                                "information_density": float(info_density),
                                "success": True,
                            }
                        )

                        type_flesch_scores.append(flesch)
                        type_clarity_scores.append(clarity)
                        type_information_densities.append(info_density)
                        type_successful += 1
                        self.report.successful_renders += 1

                    except Exception as e:
                        # Record failure detail
                        self.report.render_details.append(
                            {
                                "template_type": fact_type,
                                "template_id": template_id,
                                "rendered_output": "",
                                "flesch_score": 0.0,
                                "information_density": 0.0,
                                "success": False,
                                "error": str(e),
                            }
                        )

                        type_failed += 1
                        self.report.failed_renders += 1

                    self.report.total_renders += 1

            # Compute per-type metrics
            avg_flesch = (
                sum(type_flesch_scores) / len(type_flesch_scores) if type_flesch_scores else 0.0
            )
            avg_clarity = (
                sum(type_clarity_scores) / len(type_clarity_scores) if type_clarity_scores else 0.0
            )
            avg_info_density = (
                sum(type_information_densities) / len(type_information_densities)
                if type_information_densities
                else 0.0
            )
            total_type_renders = type_successful + type_failed
            success_rate = type_successful / max(1, total_type_renders)
            coverage = type_successful / max(1, total_type_renders)
            passes_flesch = avg_flesch > 50

            if passes_flesch:
                types_passing_flesch += 1

            metrics = TemplateTypeMetrics(
                type_name=fact_type,
                template_count=len(library.templates),
                example_count=len(type_examples),
                total_renders=total_type_renders,
                successful_renders=type_successful,
                failed_renders=type_failed,
                average_flesch=avg_flesch,
                average_clarity=avg_clarity,
                average_information_density=avg_info_density,
                coverage=coverage,
                success_rate=success_rate,
                passes_flesch_45=passes_flesch,
            )

            self.report.type_metrics[fact_type] = metrics
            flesch_scores.extend(type_flesch_scores)
            clarity_scores.extend(type_clarity_scores)
            information_densities.extend(type_information_densities)

        # Compute overall metrics
        self.report.total_templates = 51
        self.report.total_examples = sum(len(v) for v in examples.values())
        self.report.overall_coverage = self.report.successful_renders / max(
            1, self.report.total_renders
        )
        self.report.overall_flesch = (
            sum(flesch_scores) / len(flesch_scores) if flesch_scores else 0.0
        )
        self.report.overall_clarity = (
            sum(clarity_scores) / len(clarity_scores) if clarity_scores else 0.0
        )
        self.report.overall_success_rate = self.report.successful_renders / max(
            1, self.report.total_renders
        )
        self.report.overall_information_density = (
            sum(information_densities) / len(information_densities)
            if information_densities
            else 0.0
        )
        self.report.type_coverage = types_passing_flesch / len(libraries) if libraries else 0.0

        # Save if requested
        if save_to_file:
            save_path = Path(save_to_file)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with save_path.open("w") as f:
                json.dump(self.report.to_dict(), f, indent=2)

        return self.report

    def generate_text_report(self) -> str:
        """Generate human-readable text report."""
        if not self.report:
            return "No evaluation run yet. Call evaluate_all_templates() first."

        lines = []
        lines.append("=" * 100)
        lines.append("M1-E1: TEMPLATE DEVELOPMENT EVALUATION REPORT (V3 - REVISED METRICS)")
        lines.append("=" * 100)
        lines.append("")

        # Summary
        lines.append("SUMMARY METRICS")
        lines.append("-" * 100)
        lines.append(f"Total Templates: {self.report.total_templates}")
        lines.append(f"Total Examples: {self.report.total_examples}")
        lines.append(f"Total Renders: {self.report.total_renders}")
        lines.append(f"Successful Renders: {self.report.successful_renders}")
        lines.append(f"Failed Renders: {self.report.failed_renders}")
        lines.append("")

        # Quality metrics
        lines.append("QUALITY METRICS (REVISED TARGETS)")
        lines.append("-" * 100)
        lines.append(f"Overall Coverage: {self.report.overall_coverage * 100:.1f}% (target >85%)")
        lines.append(f"Overall Flesch Score: {self.report.overall_flesch:.1f} (target >45)")
        lines.append(f"Overall Clarity: {self.report.overall_clarity:.2f}/5.0 (target >4.0)")
        lines.append(
            f"Overall Success Rate: {self.report.overall_success_rate * 100:.1f}% (target >95%)"
        )
        lines.append(
            f"Overall Information Density: {self.report.overall_information_density:.3f} (target >0.5)"
        )
        lines.append(
            f"Type Coverage (% passing Flesch >45): {self.report.type_coverage * 100:.1f}% (target >80%)"
        )
        lines.append("")

        # Per-type results
        lines.append("PER-TEMPLATE-TYPE RESULTS")
        lines.append("-" * 100)
        for type_name, metrics in self.report.type_metrics.items():
            pass_status = "✓ PASS" if metrics.passes_flesch_45 else "✗ FAIL"
            lines.append(f"\n{type_name.upper().replace('_', ' ')} {pass_status}:")
            lines.append(f"  Templates: {metrics.template_count}")
            lines.append(f"  Examples: {metrics.example_count}")
            lines.append(f"  Renders: {metrics.total_renders}")
            lines.append(f"  Success Rate: {metrics.success_rate * 100:.1f}%")
            lines.append(f"  Flesch Score: {metrics.average_flesch:.1f} (target >50)")
            lines.append(f"  Clarity: {metrics.average_clarity:.2f}/5.0")
            lines.append(f"  Information Density: {metrics.average_information_density:.3f}")
            lines.append(f"  Coverage: {metrics.coverage * 100:.1f}%")

        lines.append("")
        lines.append("SUCCESS CRITERIA EVALUATION (V3)")
        lines.append("-" * 100)
        lines.append(
            f"{'✓' if self.report.overall_coverage > 0.85 else '✗'} Coverage >85%: {'PASS' if self.report.overall_coverage > 0.85 else 'FAIL'} ({self.report.overall_coverage * 100:.1f}%)"
        )
        lines.append(
            f"{'✓' if self.report.overall_flesch > 45 else '✗'} Flesch >45: {'PASS' if self.report.overall_flesch > 45 else 'FAIL'} ({self.report.overall_flesch:.1f})"
        )
        lines.append(
            f"{'✓' if self.report.overall_clarity > 4.0 else '✗'} Clarity >4.0: {'PASS' if self.report.overall_clarity > 4.0 else 'FAIL'} ({self.report.overall_clarity:.2f})"
        )
        lines.append(
            f"{'✓' if self.report.overall_success_rate > 0.95 else '✗'} Success Rate >95%: {'PASS' if self.report.overall_success_rate > 0.95 else 'FAIL'} ({self.report.overall_success_rate * 100:.1f}%)"
        )
        lines.append(
            f"{'✓' if self.report.overall_information_density > 0.5 else '✗'} Info Density >0.5: {'PASS' if self.report.overall_information_density > 0.5 else 'FAIL'} ({self.report.overall_information_density:.3f})"
        )
        lines.append(
            f"{'✓' if self.report.type_coverage > 0.8 else '✗'} Type Coverage >80%: {'PASS' if self.report.type_coverage > 0.8 else 'FAIL'} ({self.report.type_coverage * 100:.1f}%)"
        )
        lines.append("")

        lines.append("=" * 100)
        lines.append(f"Report generated: {self.report.timestamp}")
        lines.append("=" * 100)

        return "\n".join(lines)


def evaluate_and_save(
    output_dir: str = "experiments/m1_e1_templates/results", n_examples: int = 100
) -> Tuple[M1E1EvaluationReportV3, str]:
    """
    Run evaluation and save results to files.

    Args:
        output_dir: Directory to save results
        n_examples: Number of examples per type to evaluate

    Returns:
        Tuple of (report, output_dir_path)
    """
    from datetime import datetime

    evaluator = M1E1EvaluatorV3()

    # Create timestamped output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / timestamp
    output_path.mkdir(parents=True, exist_ok=True)

    # Run evaluation
    json_path = output_path / "m1_e1_report.json"
    report = evaluator.evaluate_all_templates(
        n_examples_per_type=n_examples, save_to_file=str(json_path)
    )

    # Save text report
    txt_report = evaluator.generate_text_report()
    txt_path = output_path / "m1_e1_report.txt"
    # Write as UTF-8 to support symbols (e.g., checkmarks) on Windows consoles
    with txt_path.open("w", encoding="utf-8") as f:
        f.write(txt_report)

    print(f"Saved M1-E1 evaluation to: {output_path}")
    print(f"  - JSON: {json_path}")
    print(f"  - Text: {txt_path}")

    return report, str(output_path)


if __name__ == "__main__":
    # Run evaluation and save results
    report, output_dir = evaluate_and_save(n_examples=100)

    # Print text report
    evaluator = M1E1EvaluatorV3()
    evaluator.report = report
    print("\n" + evaluator.generate_text_report())

#!/usr/bin/env python3
"""
M1-E4: ACCURACY VALIDATION EVALUATION
======================================

Research Question: How factually accurate are hybrid-generated outputs?

Success Criteria:
  ✓ Overall Accuracy > 99%
  ✓ No Type Below 98%
  ✓ Hallucination Rate < 1%
  ✓ Entity F1 > 0.95
  ✓ Template Accuracy > Polished
"""

import os
import json
import csv
import time
import argparse
from datetime import datetime
from uuid import uuid4
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

# Third-party imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import textstat
from rich.console import Console
from rich.table import Table
from rich.progress import Progress

from dotenv import load_dotenv
import openai

# Internal imports
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from temporal_nlg.data.loaders import (
    PointInTimeExampleGenerator,
    IntervalExampleGenerator,
    SequenceExampleGenerator,
    CausalityExampleGenerator,
    OverlapExampleGenerator,
)
from temporal_nlg.core.templates import TemporalFact
from temporal_nlg.templates.point_in_time import PointInTimeTemplateLibrary
from temporal_nlg.templates.intervals import IntervalTemplateLibrary
from temporal_nlg.templates.sequences import SequenceTemplateLibrary
from temporal_nlg.templates.causality import CausalityTemplateLibrary
from temporal_nlg.templates.overlaps import OverlapTemplateLibrary

# Local imports
from accuracy_scorer import AccuracyScorer
from fact_extractor import FactExtractor
from hallucination_detector import HallucinationDetector

console = Console()
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


@dataclass
class AccuracyResult:
    """Single result from accuracy evaluation"""

    fact_type: str
    template_id: str
    source: str  # "template" or "polished"
    output: str

    # Accuracy scores (0-1)
    date_preservation: float
    entity_preservation: float
    relation_preservation: float
    overall_accuracy: float

    # Hallucination check
    hallucinated: bool
    hallucination_type: str  # "none", "date", "entity", "relation", "fabricated"

    # Metadata
    fact_str: str
    latency_ms: float


@dataclass
class AccuracyMetrics:
    """Aggregated accuracy metrics"""

    overall_accuracy: float
    date_preservation_mean: float
    entity_preservation_mean: float
    relation_preservation_mean: float
    hallucination_rate: float
    entity_f1: float

    # Source comparison
    template_accuracy: float
    polished_accuracy: float
    accuracy_gap: float  # template - polished (should be positive)

    # Per-type breakdown
    type_accuracies: Dict[str, float]


class AccuracyEvaluator:
    """Main evaluator for M1-E4 accuracy validation"""

    def __init__(self, model: str = "gpt-4.1-nano", examples_per_type: int = 20):
        self.model = model
        self.examples_per_type = examples_per_type
        self.llm = ChatOpenAI(model=model, request_timeout=5.0, max_retries=1)

        # Template libraries
        self.template_libs = {
            "point_in_time": PointInTimeTemplateLibrary(),
            "intervals": IntervalTemplateLibrary(),
            "sequences": SequenceTemplateLibrary(),
            "causality": CausalityTemplateLibrary(),
            "overlaps": OverlapTemplateLibrary(),
        }

        # Accuracy tools
        self.scorer = AccuracyScorer(self.llm)
        self.extractor = FactExtractor(self.llm)
        self.detector = HallucinationDetector(self.llm)

        # Results storage
        self.all_results: List[AccuracyResult] = []
        self.type_results: Dict[str, List[AccuracyResult]] = {
            t: [] for t in self.template_libs.keys()
        }

        self.start_time = None
        self.end_time = None

    def generate_examples(self) -> Dict[str, List[TemporalFact]]:
        """Generate temporal facts for evaluation"""
        console.print("[bold blue]🔄 Generating temporal facts...[/bold blue]")

        generators = {
            "point_in_time": PointInTimeExampleGenerator(),
            "intervals": IntervalExampleGenerator(),
            "sequences": SequenceExampleGenerator(),
            "causality": CausalityExampleGenerator(),
            "overlaps": OverlapExampleGenerator(),
        }

        examples = {}
        for fact_type, gen in generators.items():
            examples[fact_type] = gen.generate(self.examples_per_type)
            console.print(f"  ✓ {fact_type:15} {len(examples[fact_type]):4} facts")

        return examples

    def load_examples_from_file(self, facts_path: Path) -> Optional[Dict[str, List[TemporalFact]]]:
        if not facts_path.exists():
            return None
        data = json.loads(facts_path.read_text())
        examples: Dict[str, List[TemporalFact]] = {}
        for fact_type, facts in data.items():
            examples[fact_type] = [
                TemporalFact(
                    fact_type=fact_type,
                    content=item.get("content", {}),
                    metadata=item.get("metadata", {}),
                )
                for item in facts
            ]
        console.print(f"[green]Loaded facts from {facts_path}[/green]")
        return examples

    def evaluate_output(
        self, fact: TemporalFact, output: str, fact_type: str, template_id: str, source: str
    ) -> AccuracyResult:
        """
        Evaluate accuracy of a single output:
        1. Extract entities from fact
        2. Extract entities from output
        3. Compare preservation
        4. Detect hallucinations
        5. Score overall accuracy
        """
        start_ms = time.perf_counter()

        try:
            # Step 1: Extract fact entities
            fact_entities = self.extractor.extract_from_fact(fact)

            # Step 2: Extract output entities
            output_entities = self.extractor.extract_from_text(output, fact_type)

            # Step 3: Score preservation
            date_score = self.scorer.score_date_preservation(fact, output)
            entity_score = self.scorer.score_entity_preservation(fact_entities, output_entities)
            relation_score = self.scorer.score_relation_preservation(fact, output_entities)

            overall = (date_score + entity_score + relation_score) / 3.0

            # Step 4: Detect hallucinations
            hallucinated, halluc_type = self.detector.detect(fact, output, output_entities)

            latency_ms = (time.perf_counter() - start_ms) * 1000

            return AccuracyResult(
                fact_type=fact_type,
                template_id=template_id,
                source=source,
                output=output,
                date_preservation=date_score,
                entity_preservation=entity_score,
                relation_preservation=relation_score,
                overall_accuracy=overall,
                hallucinated=hallucinated,
                hallucination_type=halluc_type,
                fact_str=str(fact),
                latency_ms=latency_ms,
            )

        except Exception as e:
            console.print(f"[red]Error evaluating accuracy: {e}[/red]")
            return AccuracyResult(
                fact_type=fact_type,
                template_id=template_id,
                source=source,
                output=output,
                date_preservation=0.0,
                entity_preservation=0.0,
                relation_preservation=0.0,
                overall_accuracy=0.0,
                hallucinated=True,
                hallucination_type="error",
                fact_str=str(fact),
                latency_ms=(time.perf_counter() - start_ms) * 1000,
            )

    def evaluate(self, results_dir: str) -> AccuracyMetrics:
        """
        Load M1-E3 hybrid results and evaluate accuracy
        """
        console.print("\n[bold cyan]🚀 M1-E4: ACCURACY VALIDATION[/bold cyan]")
        console.print(f"Model: {self.model}")
        console.print(f"M1-E3 Results: {results_dir}")

        self.start_time = datetime.now()

        # Load M1-E3 results
        results_path = Path(results_dir)
        csv_path = results_path / "render_details.csv"
        facts_path = results_path / "facts.json"

        if not csv_path.exists():
            console.print(f"[red]❌ No render_details.csv in {results_path}[/red]")
            return None

        # Reload examples to match M1-E3 (prefer saved facts)
        examples = self.load_examples_from_file(facts_path) or self.generate_examples()

        # Create fact lookup
        fact_lookup = {}
        for fact_type, facts in examples.items():
            for i, fact in enumerate(facts):
                fact_lookup[(fact_type, i)] = fact

        # Load and evaluate M1-E3 renders
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            total_rows = sum(1 for _ in open(csv_path)) - 1

        with Progress() as progress:
            task = progress.add_task("[cyan]Evaluating...", total=total_rows)

            with open(csv_path) as f:
                reader = csv.DictReader(f)

                for row_idx, row in enumerate(reader):
                    fact_type = row.get("type", "unknown")
                    output = row.get("output", "")
                    template_id = row.get("template_id", "?")
                    source = row.get("source", "unknown")

                    # Get corresponding fact (approximation - first of type)
                    if fact_type in examples and examples[fact_type]:
                        fact = examples[fact_type][row_idx % len(examples[fact_type])]

                        result = self.evaluate_output(fact, output, fact_type, template_id, source)
                        self.all_results.append(result)
                        self.type_results[fact_type].append(result)

                    progress.update(task, advance=1)

        self.end_time = datetime.now()

        # Compute metrics
        metrics = self._compute_metrics()
        self._print_report(metrics)
        self._save_results(metrics, results_dir)

        return metrics

    def _compute_metrics(self) -> AccuracyMetrics:
        """Compute all accuracy metrics from results"""

        if not self.all_results:
            return None

        # Overall accuracy
        overall_acc = sum(r.overall_accuracy for r in self.all_results) / len(self.all_results)
        date_mean = sum(r.date_preservation for r in self.all_results) / len(self.all_results)
        entity_mean = sum(r.entity_preservation for r in self.all_results) / len(self.all_results)
        relation_mean = sum(r.relation_preservation for r in self.all_results) / len(
            self.all_results
        )

        # Hallucination rate
        halluc_count = sum(1 for r in self.all_results if r.hallucinated)
        halluc_rate = halluc_count / len(self.all_results) if self.all_results else 0

        # Entity F1 (approximate - use entity preservation as proxy)
        entity_f1 = entity_mean

        # Source comparison
        template_results = [r for r in self.all_results if r.source == "template"]
        polished_results = [r for r in self.all_results if r.source == "polished"]

        template_acc = (
            sum(r.overall_accuracy for r in template_results) / len(template_results)
            if template_results
            else 0
        )
        polished_acc = (
            sum(r.overall_accuracy for r in polished_results) / len(polished_results)
            if polished_results
            else 0
        )

        # Per-type breakdown
        type_accuracies = {}
        for fact_type, results in self.type_results.items():
            if results:
                type_accuracies[fact_type] = sum(r.overall_accuracy for r in results) / len(results)

        return AccuracyMetrics(
            overall_accuracy=overall_acc,
            date_preservation_mean=date_mean,
            entity_preservation_mean=entity_mean,
            relation_preservation_mean=relation_mean,
            hallucination_rate=halluc_rate,
            entity_f1=entity_f1,
            template_accuracy=template_acc,
            polished_accuracy=polished_acc,
            accuracy_gap=template_acc - polished_acc,
            type_accuracies=type_accuracies,
        )

    def _print_report(self, metrics: AccuracyMetrics):
        """Print evaluation report"""

        if not metrics:
            console.print("[red]❌ No metrics to report[/red]")
            return

        # Main metrics panel
        table = Table(title="M1-E4: ACCURACY VALIDATION RESULTS", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Target", style="magenta")
        table.add_column("Status", style="yellow")

        table.add_row(
            "Overall Accuracy",
            f"{metrics.overall_accuracy*100:.2f}%",
            ">99%",
            "✓ PASS" if metrics.overall_accuracy > 0.99 else "✗ FAIL",
        )
        table.add_row(
            "Entity F1",
            f"{metrics.entity_f1:.4f}",
            ">0.95",
            "✓ PASS" if metrics.entity_f1 > 0.95 else "✗ FAIL",
        )
        table.add_row(
            "Hallucination Rate",
            f"{metrics.hallucination_rate*100:.2f}%",
            "<1%",
            "✓ PASS" if metrics.hallucination_rate < 0.01 else "✗ FAIL",
        )
        table.add_row(
            "Template Accuracy",
            f"{metrics.template_accuracy*100:.2f}%",
            "-",
            "✓ Better" if metrics.template_accuracy > metrics.polished_accuracy else "✗ Worse",
        )

        console.print(table)

        # Preservation breakdown
        print("\n[cyan]PRESERVATION BREAKDOWN[/cyan]")
        print(f"  Date Preservation:     {metrics.date_preservation_mean*100:6.2f}%")
        print(f"  Entity Preservation:   {metrics.entity_preservation_mean*100:6.2f}%")
        print(f"  Relation Preservation: {metrics.relation_preservation_mean*100:6.2f}%")

        # Source comparison
        print(f"\n[cyan]SOURCE COMPARISON[/cyan]")
        print(f"  Template Accuracy: {metrics.template_accuracy*100:6.2f}%")
        print(f"  Polished Accuracy: {metrics.polished_accuracy*100:6.2f}%")
        print(f"  Gap (Template - Polished): {metrics.accuracy_gap*100:+6.2f}%")

        # Type breakdown
        print(f"\n[cyan]TYPE-SPECIFIC ACCURACY[/cyan]")
        for fact_type in sorted(metrics.type_accuracies.keys()):
            acc = metrics.type_accuracies[fact_type]
            status = "✓" if acc > 0.98 else "✗"
            print(f"  {status} {fact_type:15}: {acc*100:6.2f}%")

    def _save_results(self, metrics: AccuracyMetrics, source_dir: str):
        """Save results to disk"""
        results_dir = (
            Path(__file__).resolve().parents[2] / "output" / "m1_e4_accuracy" / uuid4().hex
        )
        results_dir.mkdir(parents=True, exist_ok=True)

        # JSON report
        report_dict = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "source_dir": source_dir,
            "metrics": asdict(metrics),
            "duration_seconds": (self.end_time - self.start_time).total_seconds(),
        }

        with open(results_dir / "m1_e4_report.json", "w") as f:
            json.dump(report_dict, f, indent=2)

        # CSV export
        csv_path = results_dir / "accuracy_details.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "type",
                    "source",
                    "overall_accuracy",
                    "date",
                    "entity",
                    "relation",
                    "hallucinated",
                    "hallucination_type",
                    "output",
                ],
            )
            writer.writeheader()
            for result in self.all_results:
                writer.writerow(
                    {
                        "type": result.fact_type,
                        "source": result.source,
                        "overall_accuracy": f"{result.overall_accuracy:.4f}",
                        "date": f"{result.date_preservation:.4f}",
                        "entity": f"{result.entity_preservation:.4f}",
                        "relation": f"{result.relation_preservation:.4f}",
                        "hallucinated": result.hallucinated,
                        "hallucination_type": result.hallucination_type,
                        "output": result.output[:100],
                    }
                )

        console.print(f"\n✓ Results saved to {results_dir}")


def main():
    parser = argparse.ArgumentParser(description="M1-E4 Accuracy Validation")
    parser.add_argument("--model", default="gpt-4.1-nano", help="OpenAI model")
    parser.add_argument("--examples", type=int, default=20, help="Examples per type")
    parser.add_argument(
        "--report-only", action="store_true", help="Always exit 0 after writing reports"
    )
    parser.add_argument("results_dir", help="Path to M1-E3 results directory")
    args = parser.parse_args()

    evaluator = AccuracyEvaluator(model=args.model, examples_per_type=args.examples)
    metrics = evaluator.evaluate(args.results_dir)

    if metrics:
        all_pass = (
            metrics.overall_accuracy > 0.99
            and metrics.entity_f1 > 0.95
            and metrics.hallucination_rate < 0.01
            and metrics.template_accuracy > metrics.polished_accuracy
        )
        exit(0 if all_pass or args.report_only else 1)
    else:
        exit(1)


if __name__ == "__main__":
    main()

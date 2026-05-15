#!/usr/bin/env python3
"""
M1-E3: HYBRID LLM+TEMPLATE SELECTOR EVALUATION
===============================================

Research Question: Can intelligent routing achieve Flesch 75+ with 99% accuracy and <200ms latency?

Success Criteria:
  ✓ Flesch > 70
  ✓ Success Rate > 98%
  ✓ Latency p95 < 200ms
  ✓ Accuracy > 99%
  ✓ Type Coverage > 95%
"""

import csv
import time
import argparse
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from uuid import uuid4
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import csv

# Third-party imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from pydantic import BaseModel, Field
import textstat
from rich.console import Console
from rich.panel import Panel
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
import temporal_nlg.templates.sequences as sequences_module
import temporal_nlg.templates.causality as causality_module
import temporal_nlg.templates.overlaps as overlaps_module
from temporal_nlg.tms.belief_store import BeliefStore, Belief
from temporal_nlg.tms.justification import JustificationBuilder
from temporal_nlg.explain.justified_render import JustifiedRenderer

console = Console()
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

@dataclass
class HybridResult:
    """Single result from hybrid NLG pipeline"""
    fact_type: str
    template_id: str
    source: str  # "template" or "llm" or "polished"
    output: str
    flesch: float
    words: int
    confidence: float
    latency_ms: float
    hallucination_detected: bool = False
    render_details: str = ""
    justification: str = ""

@dataclass
class HybridMetrics:
    """Aggregated metrics for hybrid experiment"""
    flesch_mean: float
    flesch_std: float
    success_rate: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    accuracy: float
    type_coverage: float
    template_hit_rate: float
    polishing_rate: float
    hallucination_rate: float

class HybridNLGEvaluator:
    """Main evaluator for M1-E3 hybrid approach"""
    
    def __init__(
        self,
        model: str = "gpt-4.1-nano",
        examples_per_type: int = 100,
        enable_polish: bool = True,
        anchor_snippet_count: int = 1,
        enable_justification: bool = True,
    ):
        self.model = model
        self.examples_per_type = examples_per_type
        self.enable_polish = enable_polish
        self.anchor_snippet_count = max(1, anchor_snippet_count)
        self.enable_justification = enable_justification
        # Lower timeout and retries to keep p95 latency controlled
        self.llm = ChatOpenAI(model=model, request_timeout=1.0, max_retries=0)
        # Adjust anchor density before loading libraries
        for mod in (sequences_module, causality_module, overlaps_module):
            if hasattr(mod, "set_anchor_snippet_count"):
                mod.set_anchor_snippet_count(self.anchor_snippet_count)
        
        # Template libraries
        self.template_libs = {
            "point_in_time": PointInTimeTemplateLibrary(),
            "intervals": IntervalTemplateLibrary(),
            "sequences": SequenceTemplateLibrary(),
            "causality": CausalityTemplateLibrary(),
            "overlaps": OverlapTemplateLibrary(),
        }

        # Justification plumbing
        self.belief_store = BeliefStore()
        self.justifier = JustifiedRenderer(self.belief_store, JustificationBuilder())
        self._last_justification = ""
        
        # Polishing control: default on; gated by overlap/thresholds below
        self.polish_flesch_threshold = 35.0
        self.polish_allowed_types = {"point_in_time", "intervals"}
        self.polish_extra_types = {"sequences", "causality", "overlaps"}
        self.polish_flesch_threshold_low = 45.0

        # Results storage
        self.all_results: List[HybridResult] = []
        self.type_results: Dict[str, List[HybridResult]] = {
            t: [] for t in self.template_libs.keys()
        }
        
        # Timing
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
    
    def hybrid_generate(self, fact: TemporalFact, fact_type: str) -> HybridResult:
        """
        Hybrid generation pipeline:
        1. Get all applicable templates
        2. Render all templates
        3. Select best by Flesch + confidence
        4. Polish if needed
        5. Detect hallucinations
        """
        start_ms = time.perf_counter()
        
        try:
            # Step 1: Get all templates (NO is_applicable check)
            lib = self.template_libs[fact_type]
            applicable_templates = list(lib.templates.items())
            
            if not applicable_templates:
                return HybridResult(
                    fact_type=fact_type,
                    template_id="none",
                    source="error",
                    output="",
                    flesch=0.0,
                    words=0,
                    confidence=0.0,
                    latency_ms=0,
                    hallucination_detected=False,
                )
            
            # Step 2: Render all applicable templates
            renders = {}
            fact_tokens = self._fact_tokens(getattr(fact, "content", {}) or {})
            fact_years = self._years_in_text(str(getattr(fact, "content", {})))
            for template_id, template in applicable_templates:
                try:
                    output = template.render(fact)
                    if output and len(output.strip()) > 5:
                        flesch = textstat.flesch_reading_ease(output)
                        words = len(output.split())
                        output_tokens = self._normalize_tokens(output)
                        output_years = self._years_in_text(output)
                        token_overlap = bool(fact_tokens and (output_tokens & fact_tokens))
                        year_overlap = bool(fact_years and (output_years & fact_years))
                        renders[template_id] = {
                            "output": output,
                            "flesch": max(0, min(100, flesch)),
                            "words": words,
                            "confidence": self._compute_confidence(output, flesch),
                            "token_overlap": token_overlap,
                            "year_overlap": year_overlap,
                            "score": 0.0,
                        }
                except Exception:
                    pass
            
            if not renders:
                return HybridResult(
                    fact_type=fact_type,
                    template_id="none",
                    source="error",
                    output="",
                    flesch=0.0,
                    words=0,
                    confidence=0.0,
                    latency_ms=0,
                )
            
            # Step 3: Select best template with token/year-aware scoring
            for k, v in renders.items():
                bonus = 0.0
                if v.get("token_overlap"):
                    bonus += 20.0
                if v.get("year_overlap"):
                    bonus += 5.0
                v["score"] = v["flesch"] * 0.8 + v["confidence"] * 100 * 0.2 + bonus

            sorted_templates = sorted(renders.keys(), key=lambda k: renders[k]["score"], reverse=True)
            best_template_id = sorted_templates[0]
            
            best_render = renders[best_template_id]
            result_source = "template"
            result_output = best_render["output"]
            result_flesch = best_render["flesch"]
            result_confidence = best_render["confidence"]

            # Remove anchors if we decide to polish (anchors already help template-only path)
            anchorless_output = result_output.split(" Key facts:", 1)[0].strip()
            
            # Step 4: Optional polish (disabled by default to preserve accuracy)
            token_and_year = best_render.get("token_overlap") and best_render.get("year_overlap")
            should_polish = (
                self.enable_polish
                and (
                    (
                        fact_type in self.polish_allowed_types
                        and result_flesch < self.polish_flesch_threshold
                    )
                    or (
                        fact_type in self.polish_extra_types
                        and token_and_year
                        and result_flesch < self.polish_flesch_threshold_low
                    )
                )
            )

            if should_polish:
                polished = self._polish_output(anchorless_output or result_output, fact)
                if polished:
                    polished_flesch = textstat.flesch_reading_ease(polished)
                    # Only accept polish if it improves score AND keeps tokens/years
                    if polished_flesch > result_flesch and not self._detect_hallucination(polished, fact):
                        result_output = polished
                        result_flesch = max(0, min(100, polished_flesch))
                        result_source = "polished"
            
            # Step 5: Validate for hallucinations
            hallucinated = self._detect_hallucination(result_output, fact)

            # Fallback: if hallucinated, try next-best template that preserves tokens/years
            if hallucinated:
                for alt_id in sorted_templates[1:]:
                    alt_render = renders[alt_id]
                    alt_output = alt_render["output"]
                    if not self._detect_hallucination(alt_output, fact):
                        result_output = alt_output
                        result_flesch = alt_render["flesch"]
                        result_confidence = alt_render["confidence"]
                        result_source = "template"
                        best_template_id = alt_id
                        hallucinated = False
                        break
            
            latency_ms = (time.perf_counter() - start_ms) * 1000
            
            return HybridResult(
                fact_type=fact_type,
                template_id=best_template_id,
                source=result_source,
                output=result_output,
                flesch=result_flesch,
                words=len(result_output.split()),
                confidence=result_confidence,
                latency_ms=latency_ms,
                hallucination_detected=hallucinated,
                render_details=self._render_details_with_justification(best_template_id, renders, result_output, fact),
                justification=self._last_justification,
            )
        
        except Exception as e:
            console.print(f"[red]Error processing fact: {e}[/red]")
            return HybridResult(
                fact_type=fact_type,
                template_id="error",
                source="error",
                output="",
                flesch=0.0,
                words=0,
                confidence=0.0,
                latency_ms=(time.perf_counter() - start_ms) * 1000,
            )
    
    def _compute_confidence(self, output: str, flesch: float) -> float:
        """Compute confidence score (0-1)"""
        # Penalize very long outputs more heavily (likely template failure)
        length_score = 1.0 if len(output.split()) < 25 else 0.5
        flesch_score = min(1.0, flesch / 75)
        return (length_score * 0.4 + flesch_score * 0.6)
    
    def _polish_output(self, output: str, fact: TemporalFact) -> Optional[str]:
        """Optional LLM polishing for low-Flesch outputs"""
        try:
            # NEW PROMPT: Optimized for Flesch score (simplicity), NOT "vocabulary"
            prompt = ChatPromptTemplate.from_template(
                """You are a readability expert. Rewrite the text to maximize Flesch Reading Ease score (>80).

Original: {output}
Fact: {fact_str}

Guidelines:
1. Use short sentences (under 15 words).
2. Use simple, common words (1-2 syllables).
3. Active voice only ("X did Y", not "Y was done by X").
4. NO complex clauses, semicolons, or academic jargon.
5. Keep dates/numbers accurate but concise.

Rewritten:"""
            )
            
            chain = prompt | self.llm
            response = chain.invoke({
                "output": output,
                "fact_str": str(fact)
            })
            
            polished = response.content.strip()
            # Basic sanity check
            if len(polished.split()) >= 3 and polished != output:
                return polished
            return None
        except Exception:
            # If LLM times out or fails, fallback to template (0 latency penalty)
            return None
    
    def _detect_hallucination(self, output: str, fact: TemporalFact) -> bool:
        """Require years (if present) and at least 2 fact tokens (or all if fewer)."""
        try:
            content = getattr(fact, "content", {}) or {}
            fact_tokens = self._fact_tokens(content)
            fact_years = self._years_in_text(str(content))

            output_str = output.lower()
            output_tokens = self._normalize_tokens(output_str)
            output_years = self._years_in_text(output_str)

            if fact_years:
                if not (output_years & fact_years):
                    return True

            if fact_tokens:
                required = min(2, len(fact_tokens))
                if len(output_tokens & fact_tokens) < required:
                    return True

            return False
        except Exception:
            return False

    def _years_in_text(self, text: str):
        import re
        return set(re.findall(r"\b(1\d{3}|2\d{3})\b", text))

    def _fact_tokens(self, content: Dict[str, Any]):
        import re
        tokens = set()

        def add(val: Any):
            if isinstance(val, str):
                for w in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", val.lower()):
                    if len(w) >= 4:
                        tokens.add(w)
            elif isinstance(val, (list, tuple)):
                for item in val:
                    add(item)
            elif isinstance(val, dict):
                for v in val.values():
                    add(v)
            else:
                for w in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", str(val).lower()):
                    if len(w) >= 4:
                        tokens.add(w)

        add(content)
        return tokens

    def _normalize_tokens(self, text: str):
        import re
        stopwords = {
            "the", "and", "for", "with", "from", "this", "that", "then", "when",
            "was", "were", "are", "is", "at", "of", "in", "to", "by", "on",
        }
        tokens = set()
        for w in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", text.lower()):
            if len(w) >= 4 and w not in stopwords:
                tokens.add(w)
        return tokens

    def _render_details_with_justification(self, template_id: str, renders: Dict[str, Dict[str, Any]], output: str, fact: TemporalFact) -> str:
        if not self.enable_justification:
            return str(renders)
        belief_id = f"b_{template_id}"
        belief = Belief(belief_id=belief_id, payload={"fact_type": fact.fact_type, "content": getattr(fact, "content", {})})
        self.belief_store.add_belief(belief)
        justified = self.justifier.render_with_justification(belief_id, output)
        self._last_justification = justified.get("justification", "")
        renders_copy = dict(renders)
        renders_copy[template_id] = {**renders_copy.get(template_id, {}), "justification": justified.get("justification", "")}
        return str(renders_copy)
    
    def evaluate(self) -> HybridMetrics:
        """Run full evaluation"""
        console.print("\n[bold cyan]🚀 M1-E3: HYBRID LLM+TEMPLATE SELECTOR[/bold cyan]")
        console.print(f"Model: {self.model}")
        console.print(f"Examples per type: {self.examples_per_type}")
        
        self.start_time = datetime.now()
        examples = self.generate_examples()
        
        # Evaluate with progress bar
        total_facts = sum(len(f) for f in examples.values())
        with Progress() as progress:
            task = progress.add_task("[cyan]Processing...", total=total_facts)
            
            for fact_type, facts in examples.items():
                for fact in facts:
                    result = self.hybrid_generate(fact, fact_type)
                    self.all_results.append(result)
                    self.type_results[fact_type].append(result)
                    progress.update(task, advance=1)
        
        self.end_time = datetime.now()
        
        # Compute metrics
        metrics = self._compute_metrics()
        self._print_report(metrics)
        self._save_results(metrics, examples)
        
        return metrics
    
    def _compute_metrics(self) -> HybridMetrics:
        """Compute all metrics from results"""
        
        # Filter valid results
        valid_results = [
            r for r in self.all_results
            if r.output and r.words > 0
        ]
        
        if not valid_results:
            return HybridMetrics(
                flesch_mean=0, flesch_std=0, success_rate=0,
                latency_p50=0, latency_p95=0, latency_p99=0,
                accuracy=0, type_coverage=0, template_hit_rate=0,
                polishing_rate=0, hallucination_rate=0
            )
        
        # Flesch metrics
        flesch_scores = [r.flesch for r in valid_results]
        flesch_mean = sum(flesch_scores) / len(flesch_scores)
        flesch_std = (sum((f - flesch_mean) ** 2 for f in flesch_scores) / len(flesch_scores)) ** 0.5
        
        # Success metrics
        success_rate = len(valid_results) / len(self.all_results) if self.all_results else 0
        
        # Latency metrics (in ms)
        latencies = sorted([r.latency_ms for r in valid_results])
        latency_p50 = latencies[len(latencies) // 2] if latencies else 0
        latency_p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0
        latency_p99 = latencies[int(len(latencies) * 0.99)] if latencies else 0
        
        # Accuracy (rough estimate)
        non_hallucinated = sum(1 for r in valid_results if not r.hallucination_detected)
        accuracy = non_hallucinated / len(valid_results) if valid_results else 0
        
        # Type coverage (% types with mean Flesch > 70)
        type_flesch = {}
        for fact_type, results in self.type_results.items():
            valid = [r for r in results if r.output and r.words > 0]
            if valid:
                type_flesch[fact_type] = sum(r.flesch for r in valid) / len(valid)
        
        type_coverage = sum(1 for f in type_flesch.values() if f > 70) / len(type_flesch) if type_flesch else 0
        
        # Routing metrics
        template_source = sum(1 for r in valid_results if r.source == "template")
        template_hit_rate = template_source / len(valid_results) if valid_results else 0
        
        polished_count = sum(1 for r in valid_results if r.source == "polished")
        polishing_rate = polished_count / len(valid_results) if valid_results else 0
        
        hallucination_rate = sum(1 for r in valid_results if r.hallucination_detected) / len(valid_results) if valid_results else 0
        
        return HybridMetrics(
            flesch_mean=flesch_mean,
            flesch_std=flesch_std,
            success_rate=success_rate,
            latency_p50=latency_p50,
            latency_p95=latency_p95,
            latency_p99=latency_p99,
            accuracy=accuracy,
            type_coverage=type_coverage,
            template_hit_rate=template_hit_rate,
            polishing_rate=polishing_rate,
            hallucination_rate=hallucination_rate,
        )
    
    def _print_report(self, metrics: HybridMetrics):
        """Print evaluation report"""
        
        # Main metrics panel
        table = Table(title="M1-E3: HYBRID RESULTS", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Target", style="magenta")
        table.add_column("Status", style="yellow")
        
        table.add_row(
            "Flesch Score",
            f"{metrics.flesch_mean:.1f}±{metrics.flesch_std:.1f}",
            ">70",
            "✓ PASS" if metrics.flesch_mean > 70 else "✗ FAIL"
        )
        table.add_row(
            "Success Rate",
            f"{metrics.success_rate*100:.1f}%",
            ">98%",
            "✓ PASS" if metrics.success_rate > 0.98 else "✗ FAIL"
        )
        table.add_row(
            "Latency p95",
            f"{metrics.latency_p95:.0f}ms",
            "<200ms",
            "✓ PASS" if metrics.latency_p95 < 200 else "✗ FAIL"
        )
        table.add_row(
            "Accuracy",
            f"{metrics.accuracy*100:.1f}%",
            ">99%",
            "✓ PASS" if metrics.accuracy > 0.99 else "✗ FAIL"
        )
        table.add_row(
            "Type Coverage",
            f"{metrics.type_coverage*100:.1f}%",
            ">95%",
            "✓ PASS" if metrics.type_coverage > 0.95 else "✗ FAIL"
        )
        
        console.print(table)
        
        # Routing metrics
        print("\n[cyan]ROUTING METRICS[/cyan]")
        print(f"  Template Hit Rate: {metrics.template_hit_rate*100:.1f}% (target: >95%)")
        print(f"  Polishing Rate: {metrics.polishing_rate*100:.1f}% (target: <30%)")
        print(f"  Hallucination Rate: {metrics.hallucination_rate*100:.1f}% (target: <1%)")
        
        # Latency breakdown
        print(f"\n[cyan]LATENCY BREAKDOWN[/cyan]")
        print(f"  p50: {metrics.latency_p50:.0f}ms")
        print(f"  p95: {metrics.latency_p95:.0f}ms")
        print(f"  p99: {metrics.latency_p99:.0f}ms")
        
        # Type breakdown
        print(f"\n[cyan]TYPE BREAKDOWN[/cyan]")
        for fact_type, results in self.type_results.items():
            valid = [r for r in results if r.output and r.words > 0]
            if valid:
                flesch_avg = sum(r.flesch for r in valid) / len(valid)
                success = len(valid) / len(results)
                print(f"  {fact_type:15}: Flesch {flesch_avg:6.1f} Success {success*100:5.1f}%")
    
    def _serialize_examples(self, examples: Dict[str, List[TemporalFact]]):
        serialized = {}
        for fact_type, facts in examples.items():
            serialized[fact_type] = [
                {
                    "fact_type": fact_type,
                    "content": getattr(fact, "content", {}),
                    "metadata": getattr(fact, "metadata", {}),
                }
                for fact in facts
            ]
        return serialized

    def _save_results(self, metrics: HybridMetrics, examples: Dict[str, List[TemporalFact]]):
        """Save results to disk"""
        results_dir = Path(__file__).resolve().parents[2] / "output" / "m1_e3_hybrid" / uuid4().hex
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # JSON report
        report_dict = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "examples_per_type": self.examples_per_type,
            "metrics": asdict(metrics),
            "duration_seconds": (self.end_time - self.start_time).total_seconds(),
        }
        
        with open(results_dir / "m1_e3_report.json", "w") as f:
            json.dump(report_dict, f, indent=2)

        # Save facts used
        facts_path = results_dir / "facts.json"
        with open(facts_path, "w") as f:
            json.dump(self._serialize_examples(examples), f, indent=2)
        
        # CSV export
        csv_path = results_dir / "render_details.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["type", "template_id", "source", "flesch", "words", "confidence", "latency_ms", "hallucination", "output"]
            )
            writer.writeheader()
            for result in self.all_results:
                writer.writerow({
                    "type": result.fact_type,
                    "template_id": result.template_id,
                    "source": result.source,
                    "flesch": f"{result.flesch:.1f}",
                    "words": result.words,
                    "confidence": f"{result.confidence:.2f}",
                    "latency_ms": f"{result.latency_ms:.0f}",
                    "hallucination": result.hallucination_detected,
                    "output": result.output[:100],
                })
        
        console.print(f"\n✓ Results saved to {results_dir}")

def main():
    parser = argparse.ArgumentParser(description="M1-E3 Hybrid LLM+Template Selector")
    parser.add_argument("--model", default="gpt-4.1-nano", help="OpenAI model")
    parser.add_argument("--examples", type=int, default=100, help="Examples per type")
    parser.add_argument("--polish", action=argparse.BooleanOptionalAction, default=True, help="Enable polishing")
    parser.add_argument(
        "--anchor-snippets",
        type=int,
        default=1,
        help="Anchor hints per template (e.g., 1 or 2).",
    )
    parser.add_argument("--no-justification", dest="enable_justification", action="store_false", help="Disable justification plumbing")
    args = parser.parse_args()
    
    evaluator = HybridNLGEvaluator(
        model=args.model,
        examples_per_type=args.examples,
        enable_polish=args.polish,
        anchor_snippet_count=args.anchor_snippets,
        enable_justification=args.enable_justification,
    )
    metrics = evaluator.evaluate()
    
    # Return exit code based on success
    all_pass = (
        metrics.flesch_mean > 70
        and metrics.success_rate > 0.98
        and metrics.latency_p95 < 200
        and metrics.accuracy > 99  # Slightly relaxed check here, but target remains high
    )
    
    exit(0 if all_pass else 1)

if __name__ == "__main__":
    main()

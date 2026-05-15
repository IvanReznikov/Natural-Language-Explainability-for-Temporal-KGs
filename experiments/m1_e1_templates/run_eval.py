#!/usr/bin/env python3
# experiments/m1_e1_templates/run_eval.py
"""
M1-E1 Evaluation Runner

Runs comprehensive evaluation on all template types using real examples.
Saves JSON and text reports to timestamped results directory.

Usage:
  export PYTHONPATH=src
  python experiments/m1_e1_templates/run_eval.py [--examples N] [--output DIR]
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from temporal_nlg.evaluation.m1_e1_evaluation import (
    evaluate_and_save,
    M1E1EvaluatorV3
)


def main():
    parser = argparse.ArgumentParser(
        description="Run M1-E1 template evaluation"
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=100,
        help="Number of examples per template type (default: 100)"
    )
    parser.add_argument(
        "--output",
        default="output/m1_e1_templates",
        help="Output directory for results (default: output/m1_e1_templates)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed results"
    )
    
    args = parser.parse_args()
    
    print("=" * 100)
    print("M1-E1: TEMPLATE DEVELOPMENT EVALUATION")
    print("=" * 100)
    print(f"Generating {args.examples} examples per template type...")
    print(f"Output directory: {args.output}")
    print("")
    
    # Run evaluation
    report, output_dir = evaluate_and_save(
        output_dir=args.output,
        n_examples=args.examples
    )
    
    # Print report
    evaluator = M1E1EvaluatorV3()
    evaluator.report = report
    report_text = evaluator.generate_text_report()
    
    print("")
    print(report_text)
    
    # Print sample renders if verbose
    if args.verbose:
        print("")
        print("=" * 100)
        print("SAMPLE RENDERS (first 10)")
        print("=" * 100)
        for i, result in enumerate(report.render_results[:10]):
            print(f"\n[{i+1}] {result.template_id} ({result.fact_type})")
            if result.success:
                print(f"    Flesch: {result.flesch_score:.1f}")
                print(f"    Output: {result.output[:100]}...")
            else:
                print(f"    ERROR: {result.error}")
    
    print("")
    print(f"Full results saved to: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
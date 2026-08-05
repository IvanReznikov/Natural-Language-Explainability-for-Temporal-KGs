#!/usr/bin/env python3
"""
M1-E3 (Path): Path-to-narrative quick evaluation.
Generates short narratives for sample graph paths and reports Flesch scores.
"""

import json
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from temporal_nlg.path_narratives import PathNarrativeRenderer, PathExample, PathStep


def load_or_create_samples(path: Path):
    if path.exists():
        return PathNarrativeRenderer.load_examples(path)

    # Minimal synthetic samples if no file provided
    samples = [
        PathExample(
            path_id="sample_1",
            steps=[
                PathStep(subject="Discovery", relation="led to", obj="Invention", time="1901"),
                PathStep(subject="Invention", relation="enabled", obj="Deployment", time="1903"),
                PathStep(subject="Deployment", relation="caused", obj="Adoption", time="1904"),
            ],
        ),
        PathExample(
            path_id="sample_2",
            steps=[
                PathStep(subject="Policy", relation="changed", obj="Market", time="2010"),
                PathStep(subject="Market", relation="raised", obj="Prices", time="2011"),
            ],
        ),
    ]
    path.write_text(
        json.dumps(
            [
                {
                    "id": s.path_id,
                    "steps": [step.__dict__ for step in s.steps],
                }
                for s in samples
            ],
            indent=2,
        )
    )
    return samples


def main():
    parser = argparse.ArgumentParser(description="Path-to-narrative evaluation")
    parser.add_argument(
        "--input", default="experiments/m1_e3_path/sample_paths.json", help="Path JSON file"
    )
    parser.add_argument("--output", default="output/m1_e3_path", help="Output directory")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = load_or_create_samples(input_path)

    renderer = PathNarrativeRenderer()
    report = renderer.render_batch(examples)

    results = {
        "input": str(input_path),
        "avg_flesch": report.avg_flesch,
        "flesch_scores": report.flesch_scores,
        "rendered": report.rendered,
    }

    out_file = output_dir / "path_narrative_report.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"Saved path narrative report to {out_file}")
    print(f"Average Flesch: {report.avg_flesch:.1f}")


if __name__ == "__main__":
    main()

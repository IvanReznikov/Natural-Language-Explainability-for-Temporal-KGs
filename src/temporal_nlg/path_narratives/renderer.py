"""
Path-to-narrative rendering utilities.
Generates short, high-Flesch narratives from temporal reasoning paths.
"""

from dataclasses import dataclass
from typing import List, Dict
from pathlib import Path
import json
import textstat


@dataclass
class PathStep:
    subject: str
    relation: str
    obj: str
    time: str = ""

    def to_text(self) -> str:
        parts = [self.subject, self.relation, self.obj]
        if self.time:
            parts.append(f"({self.time})")
        return " ".join(p for p in parts if p).strip()


@dataclass
class PathExample:
    path_id: str
    steps: List[PathStep]


@dataclass
class PathNarrativeReport:
    rendered: List[str]
    flesch_scores: List[float]
    avg_flesch: float


class PathNarrativeRenderer:
    """Render graph paths to compact narratives."""

    def __init__(self):
        self.openers = [
            "Here is the chain:",
            "The path is:",
            "It went this way:",
        ]

    def render_path(self, example: PathExample) -> str:
        # Compress each step to a short clause
        clauses = [step.to_text() for step in example.steps]
        body = " then ".join(clauses)
        opener = self.openers[hash(example.path_id) % len(self.openers)]
        return f"{opener} {body}."

    def render_batch(self, examples: List[PathExample]) -> PathNarrativeReport:
        rendered = [self.render_path(ex) for ex in examples]
        flesch_scores = [max(0.0, min(100.0, textstat.flesch_reading_ease(r))) for r in rendered]
        avg_flesch = sum(flesch_scores) / len(flesch_scores) if flesch_scores else 0.0
        return PathNarrativeReport(
            rendered=rendered, flesch_scores=flesch_scores, avg_flesch=avg_flesch
        )

    @staticmethod
    def load_examples(path: Path) -> List[PathExample]:
        examples: List[PathExample] = []
        data = json.loads(Path(path).read_text())
        for item in data:
            steps = [PathStep(**s) for s in item.get("steps", [])]
            examples.append(PathExample(path_id=item.get("id", ""), steps=steps))
        return examples

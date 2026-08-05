"""Configuration for the MeTTa/MORK integration bridge (M4).

All settings are read from environment variables so the same bridge works in
tests, examples, and the benchmark harness. Defaults resolve to the repo's
canonical precomputed graph so a fresh checkout works out of the box.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


def _repo_root() -> Path:
    # src/temporal_nlg_metta/config.py -> repo root is 3 parents up.
    return Path(__file__).resolve().parents[2]


@dataclass
class MettaConfig:
    """Resolved configuration for :class:`TemporalBridge`."""

    graph_dir: Path
    use_llm: bool = False
    # M1 NLG generation strategy routing
    nlg_model: str = "gpt-4.1-nano"
    polish_threshold: float = 0.7
    # M2 trace sampling: 1.0 records every rule firing.
    trace_sampling_rate: float = 1.0
    # Narrative rendering style/domain for path explanations.
    narrative_style: str = "neutral"
    narrative_domain: str = "general"
    # MORK CLI integration (subprocess-based adapter).
    mork_binary: Optional[str] = None
    mork_timeout_s: float = 30.0
    # Extra include paths passed to the MeTTa runner for ``import!``.
    include_paths: List[str] = field(default_factory=list)

    @classmethod
    def from_env(cls, repo_root: Optional[Path] = None) -> "MettaConfig":
        root = repo_root or _repo_root()

        default_graph = root / "data" / "jsonls" / "temporal_graph_output_v3"
        graph_dir = Path(os.getenv("METTA_GRAPH_DIR") or default_graph)

        mork_binary = os.getenv("MORK_BINARY") or os.getenv("MORK_PATH")

        include_paths: List[str] = []
        raw_includes = os.getenv("METTA_INCLUDE_PATHS") or ""
        if raw_includes:
            include_paths = [p.strip() for p in raw_includes.split(os.pathsep) if p.strip()]

        def _env_bool(name: str, default: bool) -> bool:
            val = os.getenv(name)
            if val is None:
                return default
            return val.strip().lower() in {"1", "true", "yes", "on"}

        return cls(
            graph_dir=graph_dir,
            use_llm=_env_bool("METTA_USE_LLM", False),
            nlg_model=os.getenv("METTA_NLG_MODEL", "gpt-4.1-nano"),
            polish_threshold=float(os.getenv("METTA_POLISH_THRESHOLD", "0.7")),
            trace_sampling_rate=float(os.getenv("METTA_TRACE_SAMPLING_RATE", "1.0")),
            narrative_style=os.getenv("METTA_NARRATIVE_STYLE", "neutral"),
            narrative_domain=os.getenv("METTA_NARRATIVE_DOMAIN", "general"),
            mork_binary=mork_binary,
            mork_timeout_s=float(os.getenv("MORK_TIMEOUT_S", "30.0")),
            include_paths=include_paths,
        )


__all__ = ["MettaConfig"]

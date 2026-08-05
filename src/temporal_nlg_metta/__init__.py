"""Temporal-NLG <-> MeTTa / MORK integration (Milestone 4).

This package connects the temporal explanation system (milestones M1-M3) to the
MeTTa language and the MORK kernel so that temporal inference paths can be
explained in natural language, in real time, from inside MeTTa programs.

The core dependency ``hyperon`` (the canonical MeTTa interpreter) is optional:
install it with ``pip install 'temporal-nlg[metta]'``. Everything except the
runner construction and operation registration works without it, which keeps the
package importable and testable in minimal environments.

Quick start::

    from temporal_nlg_metta import TemporalBridge, make_metta_runner

    bridge = TemporalBridge()
    metta = make_metta_runner(bridge=bridge)        # registers all ops
    print(metta.run('!(graph-answer "What caused the Model T price drop?")'))

See ``docs/TECH_DOCUMENTATION_M4.md`` for the full operation reference and
``examples/milestone4/`` for runnable programs.
"""

from __future__ import annotations

from .bridge import TemporalBridge
from .config import MettaConfig

__all__ = [
    "TemporalBridge",
    "MettaConfig",
    "available_tokens",
    "hyperon_available",
    "make_metta_runner",
    "run_metta",
    "run_metta_file",
    "MORKRunner",
    "MORKHttpRunner",
    "mork_http_available",
]

# These pull in ``hyperon`` lazily; safe to re-export at import time because the
# underlying functions import hyperon only when called.
from .atoms import available_tokens  # noqa: E402
from .runner import (  # noqa: E402
    MORKHttpRunner,
    MORKRunner,
    hyperon_available,
    make_metta_runner,
    mork_http_available,
    run_metta,
    run_metta_file,
)

__version__ = "0.1.0"

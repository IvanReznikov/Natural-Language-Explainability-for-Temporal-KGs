"""Compatibility shim for legacy baseline test filename.

Prefer `tests/test_baseline.py` as the canonical test module.
"""

from tests.test_baseline import *  # noqa: F401,F403

"""Tests for the MORK HTTP atomspace adapter (requires a live MORK server).

These exercise ``MORKHttpRunner`` against a real MORK HTTP server (default
``http://127.0.0.1:8000``, override with ``MORK_SERVER_URL``). Every test runs
under a unique namespace prefix and clears it afterwards, so the shared
atomspace is left as it was found. The module skips cleanly when no server is
reachable.
"""

from __future__ import annotations

import time
import uuid

import pytest

from temporal_nlg_metta import MORKHttpRunner, mork_http_available

pytestmark = pytest.mark.skipif(
    not mork_http_available(),
    reason="MORK HTTP server not reachable (set MORK_SERVER_URL; default http://127.0.0.1:8000)",
)


def _wait_export(
    runner: MORKHttpRunner, pattern: str, template: str, timeout_s: float = 5.0
) -> str:
    """Poll an export until it returns data (uploads/transforms commit async)."""
    deadline = time.monotonic() + timeout_s
    out = ""
    while time.monotonic() < deadline:
        out = runner.export(pattern, template)
        if out.strip():
            return out
        time.sleep(0.1)
    return out


@pytest.fixture
def runner() -> MORKHttpRunner:
    return MORKHttpRunner(timeout_s=5.0)


@pytest.fixture
def prefix(runner: MORKHttpRunner):
    """A unique atomspace namespace for the test; cleared on teardown."""
    ns = f"m4test_{uuid.uuid4().hex[:8]}"
    yield ns
    runner.clear(f"({ns} $s $r $t $y)")
    runner.clear(f"({ns}out $s $t)")


def test_server_reports_available(runner: MORKHttpRunner):
    assert runner.available is True


def test_upload_export_round_trip(runner: MORKHttpRunner, prefix: str):
    pattern = f"({prefix} $s $r $t $y)"
    runner.upload(pattern, pattern, f"({prefix} Ford launched ModelT 1908)")
    out = _wait_export(runner, pattern, "($s $r $t $y)")
    assert "(Ford launched ModelT 1908)" in out


def test_export_filters_by_source(runner: MORKHttpRunner, prefix: str):
    pattern = f"({prefix} $s $r $t $y)"
    data = (
        f"({prefix} Ford introduced AssemblyLine 1913)\n({prefix} US ended GoldConvertibility 1971)"
    )
    runner.upload(pattern, pattern, data)
    out = _wait_export(runner, f"({prefix} Ford $r $t $y)", "($r $t $y)")
    assert "(introduced AssemblyLine 1913)" in out
    assert "GoldConvertibility" not in out


def test_transform_materializes_join(runner: MORKHttpRunner, prefix: str):
    pattern = f"({prefix} $s $r $t $y)"
    runner.upload(pattern, pattern, f"({prefix} AssemblyLine enabled PriceDrop 1913)")
    _wait_export(runner, pattern, "($s $r $t $y)")
    runner.transform(f"(transform (, ({prefix} $s $r $t $y)) (, ({prefix}out $s $t)))")
    out = _wait_export(runner, f"({prefix}out $s $t)", "($s $t)")
    assert "(AssemblyLine PriceDrop)" in out


def test_clear_removes_atoms(runner: MORKHttpRunner, prefix: str):
    pattern = f"({prefix} $s $r $t $y)"
    runner.upload(pattern, pattern, f"({prefix} X caused Y 2000)")
    _wait_export(runner, pattern, "($s $r $t $y)")
    runner.clear(pattern)
    deadline = time.monotonic() + 5.0
    out = "pending"
    while time.monotonic() < deadline:
        out = runner.export(pattern, "($s $r $t $y)")
        if not out.strip():
            break
        time.sleep(0.1)
    assert not out.strip()

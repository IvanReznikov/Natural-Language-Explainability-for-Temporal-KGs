"""Unit tests for runner construction and the MORK adapter (no hyperon required).

The ``make_metta_runner`` path that needs hyperon is covered in
``test_integration.py``; here we test the behavior that holds without it.
"""

from __future__ import annotations

import pytest

from temporal_nlg_metta.runner import (
    MORKRunner,
    hyperon_available,
    make_metta_runner,
    mork_available,
)


def test_hyperon_available_returns_bool():
    assert isinstance(hyperon_available(), bool)


def test_make_metta_runner_raises_without_hyperon():
    if hyperon_available():
        pytest.skip("hyperon is installed; the no-hyperon path is not exercised here")
    with pytest.raises(RuntimeError, match="hyperon"):
        make_metta_runner()


def test_mork_runner_constructs_without_binary():
    # An explicit, nonexistent binary -> not available, but constructable.
    runner = MORKRunner(binary="/definitely/not/installed/mork")
    assert runner.available is False


def test_mork_runner_run_raises_when_unavailable(tmp_path):
    runner = MORKRunner(binary="/definitely/not/installed/mork")
    f = tmp_path / "p.metta"
    f.write_text("!(+ 1 2)")
    with pytest.raises(RuntimeError, match="MORK binary not found"):
        runner.run_file(f)


def test_mork_runner_run_file_raises_on_missing_file(tmp_path):
    runner = MORKRunner(binary="/definitely/not/installed/mork")
    with pytest.raises(FileNotFoundError):
        runner.run_file(tmp_path / "absent.metta")


def test_mork_available_returns_bool():
    assert isinstance(mork_available(), bool)


def test_mork_runner_default_args_use_file_placeholder():
    runner = MORKRunner(binary="/x/mork")
    # Default invocation is `mork <file>` (no subcommand).
    cmd = runner._build_command("prog.metta")
    assert cmd == ["/x/mork", "prog.metta"]


def test_mork_runner_custom_args_with_placeholder():
    runner = MORKRunner(binary="/x/mork", args=["run", "{file}"])
    cmd = runner._build_command("prog.metta")
    assert cmd == ["/x/mork", "run", "prog.metta"]


def test_mork_runner_custom_args_without_placeholder_appends_file():
    runner = MORKRunner(binary="/x/mork", args=["repl", "--eval"])
    cmd = runner._build_command("prog.metta")
    assert cmd == ["/x/mork", "repl", "--eval", "prog.metta"]


# ── MORKHttpRunner pure helpers (no server required) ─────────────────────────


def test_mork_http_encode_expr_encodes_specials():
    from temporal_nlg_metta.runner import MORKHttpRunner

    assert MORKHttpRunner._encode_expr("(edge $e caused)") == "(edge%20%24e%20caused)"
    # Parentheses and dots are preserved; spaces and `$` are percent-encoded.
    encoded = MORKHttpRunner._encode_expr("(a.b $x)")
    assert "(a.b" in encoded and "%24x" in encoded


def test_mork_http_runner_url_resolution(monkeypatch):
    from temporal_nlg_metta.runner import MORKHttpRunner

    monkeypatch.delenv("MORK_SERVER_URL", raising=False)
    assert MORKHttpRunner().url == "http://127.0.0.1:8000"
    assert MORKHttpRunner(url="http://example:9000/").url == "http://example:9000"
    monkeypatch.setenv("MORK_SERVER_URL", "http://env:7000")
    assert MORKHttpRunner().url == "http://env:7000"


def test_mork_http_available_returns_bool():
    from temporal_nlg_metta.runner import mork_http_available

    assert isinstance(mork_http_available(url="http://127.0.0.1:1"), bool)

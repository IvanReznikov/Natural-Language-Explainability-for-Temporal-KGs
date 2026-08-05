"""MeTTa runner construction and the MORK kernel adapter (M4).

This module provides two ways to evaluate MeTTa programs against the temporal
bridge:

* :func:`make_metta_runner` / :func:`run_metta` / :func:`run_metta_file` — the
  canonical path using the ``hyperon`` interpreter. These are the primary
  integration surface today.
* :class:`MORKRunner` — an adapter that evaluates the *same* ``.metta`` programs
  against the MORK kernel by shelling out to its CLI. MORK currently has no
  Python FFI, so a subprocess seam is the only way to reach it. The adapter is
  optional and degrades to a clear error when the ``mork`` binary is absent.

All paths share a common requirement: the temporal grounded operations must be
registered before any ``.metta`` program that calls them is executed.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .bridge import TemporalBridge


def _import_hyperon():
    try:
        import hyperon  # type: ignore

        return hyperon
    except Exception:
        return None


def hyperon_available() -> bool:
    """True when the ``hyperon`` interpreter package is importable."""
    return _import_hyperon() is not None


def make_metta_runner(
    bridge: Optional[TemporalBridge] = None,
    *,
    working_dir: Optional[Union[str, Path]] = None,
    include_paths: Optional[List[str]] = None,
    register: bool = True,
):
    """Construct a headless ``MeTTa`` runner with temporal ops registered.

    The runner is built via ``Environment.custom_env(config_dir=None)`` so it
    does not touch the user's config directory and runs fully isolated — the
    recommended headless construction in the hyperon test suite.
    """
    hyperon = _import_hyperon()
    if hyperon is None:
        raise RuntimeError(
            "The 'hyperon' package is required to build a MeTTa runner. "
            "Install it with: pip install 'temporal-nlg[metta]'"
        )
    from hyperon import MeTTa, Environment  # type: ignore

    env = Environment.custom_env(
        working_dir=str(working_dir or os.getcwd()),
        config_dir=None,
        include_paths=list(include_paths or []),
    )
    metta = MeTTa(env_builder=env)

    if register:
        from .atoms import register_with

        b = bridge or TemporalBridge()
        register_with(metta, b)
    return metta


def run_metta(
    program: str,
    *,
    bridge: Optional[TemporalBridge] = None,
    metta: Any = None,
    flat: bool = False,
):
    """Run a MeTTa program string and return the interpreter result.

    If ``metta`` is omitted a fresh headless runner is constructed with the
    bridge's operations registered onto it.
    """
    if metta is None:
        metta = make_metta_runner(bridge=bridge)
    return metta.run(program, flat=flat)


def run_metta_file(
    path: Union[str, Path],
    *,
    bridge: Optional[TemporalBridge] = None,
    metta: Any = None,
    flat: bool = False,
):
    """Read and run a ``.metta`` file. Returns the interpreter result."""
    program = Path(path).read_text(encoding="utf-8")
    return run_metta(program, bridge=bridge, metta=metta, flat=flat)


# ----------------------------------------------------------------------
# MORK kernel adapter
# ----------------------------------------------------------------------


class MORKRunner:
    """Evaluate ``.metta`` programs against the MORK kernel via its CLI.

    MORK (the MeTTa Optimal Reduction Kernel) is a Rust implementation of the
    MeTTa evaluator with no Python FFI today. This adapter shells out to the
    ``mork`` CLI (built via ``cargo build --release`` in ``/kernel``) so the
    *same* ``.metta`` files developed against ``hyperon`` can be exercised on
    MORK once the binary is on PATH.

    The MORK CLI argument surface is still evolving (clap-derived subcommands),
    so the exact invocation is configurable via ``args`` / ``MORK_ARGS``: the
    placeholder ``{file}`` in the args list is replaced with the ``.metta`` file
    path. The default ``["{file}"]`` matches a plain ``mork <file>`` invocation.

    Note: MORK does not host Python grounded operations, so any program that
    calls temporal-* / nlg-* / tms-* / graph-* tokens requires the operations to
    be defined in MeTTa itself (see ``metta/temporal_nlg.metta`` for the
    higher-level wrappers that are kernel-portable). Programs that only use the
    pure-MeTTa wrapper layer run unchanged on both interpreters.
    """

    #: Placeholder substituted with the .metta file path in ``args``.
    FILE_PLACEHOLDER = "{file}"

    def __init__(
        self,
        binary: Optional[str] = None,
        *,
        timeout_s: float = 30.0,
        args: Optional[List[str]] = None,
    ):
        self.binary = binary or shutil.which("mork") or shutil.which("mork.exe")
        self.timeout_s = timeout_s
        # Default to a plain `mork <file>` invocation; allow override via env.
        default_args = ["{file}"]
        env_args = os.getenv("MORK_ARGS")
        self.args = [a.strip() for a in env_args.split()] if env_args else (args or default_args)

    @property
    def available(self) -> bool:
        return self.binary is not None and Path(self.binary).exists()

    def _build_command(self, file_path: str) -> List[str]:
        if self.FILE_PLACEHOLDER not in self.args:
            # No placeholder: append the file path at the end.
            return [self.binary, *self.args, file_path]
        return [self.binary] + [a.replace(self.FILE_PLACEHOLDER, file_path) for a in self.args]

    def run_program(self, program: str) -> str:
        """Run a MeTTa program string through the MORK CLI."""
        import tempfile

        with tempfile.NamedTemporaryFile(
            "w", suffix=".metta", delete=False, encoding="utf-8"
        ) as handle:
            handle.write(program)
            tmp_path = handle.name
        try:
            return self.run_file(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def run_file(self, path: Union[str, Path]) -> str:
        """Run a ``.metta`` file through the MORK CLI and return stdout."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        if not self.available:
            raise RuntimeError(
                "MORK binary not found. Build it with `cargo build --release` in the "
                "MORK /kernel directory, or set MORK_BINARY."
            )
        command = self._build_command(str(path))
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=self.timeout_s,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"MORK exited with code {completed.returncode}.\n"
                f"command: {' '.join(command)}\n"
                f"stderr:\n{completed.stderr.strip()}"
            )
        return completed.stdout


def mork_available(config=None) -> bool:
    """True when the MORK CLI binary is configured/discoverable."""
    bridge_cfg = config
    if bridge_cfg is None:
        return MORKRunner().available
    return MORKRunner(binary=bridge_cfg.mork_binary, timeout_s=bridge_cfg.mork_timeout_s).available


def mork_http_available(url: Optional[str] = None) -> bool:
    """True when the MORK HTTP atomspace server is reachable."""
    try:
        runner = MORKHttpRunner(url=url, timeout_s=3.0)
        return runner.available
    except Exception:
        return False


# ----------------------------------------------------------------------
# MORK HTTP atomspace adapter
# ----------------------------------------------------------------------


class MORKHttpRunner:
    """Evaluate temporal operations against the MORK HTTP atomspace server.

    MORK exposes a RESTful atomspace with upload, export, transform, explore,
    and count operations over versioned S-expression spaces. This adapter wraps
    the HTTP endpoints (documented in the MORK server's API) so temporal facts
    can be loaded, queried, and joined through the same REST surface that the
    MORK kernel provides.

    Unlike ``hyperon`` (which hosts Python grounded ops as MeTTa callbacks),
    MORK is a pure-Rust atomspace — temporal operations must be encoded as
    S-expression patterns uploaded to its space. The adapter uses the
    following verified endpoints:

    * **POST /upload/<pattern>/<template>/** — upload S-expressions
    * **GET  /export/<pattern>/<template>/** — query atoms matching pattern
    * **POST /transform/** — join-match pattern pairs to materialise new atoms
    * **GET  /status/<expr>/**   — server status / poll command result
    * **GET  /count/<expr>/**    — count atoms under an expression
    * **GET  /explore/<expr>/<token>/** — traverse the atom space tree

    URL path segments containing S-expressions are percent-encoded
    (spaces → %20, ``$`` → %24, ``#`` → %23). The adapter handles this
    automatically via :meth:`_encode_expr`.
    """

    DEFAULT_URL = "http://127.0.0.1:8000"

    def __init__(
        self,
        url: Optional[str] = None,
        *,
        timeout_s: float = 30.0,
    ):
        import urllib.request

        self.url = (url or os.getenv("MORK_SERVER_URL") or self.DEFAULT_URL).rstrip("/")
        self.timeout_s = timeout_s

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        try:
            status = self.status("-")
            return isinstance(status, dict)
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Core atomspace operations
    # ------------------------------------------------------------------

    def upload(self, pattern: str, template: str, data: str) -> str:
        """Upload S-expression *data* into the space matched by *pattern*.

        The upload is acknowledged synchronously but the atomspace commit is
        asynchronous — call :meth:`poll_upload` after a batch to confirm all
        data has landed before querying.
        """
        path = f"/upload/{self._encode_expr(pattern)}/{self._encode_expr(template)}/"
        result = self._request("POST", path, body=data, content_type="text/plain")
        self._last_upload_status_path = None
        return result

    def poll_idle(self, timeout_s: float = 5.0, interval_s: float = 0.05) -> bool:
        """Block until the server reports idle (no pending async work).

        Returns True if the server became idle within *timeout_s* seconds.
        Use after batch uploads or transforms before querying the atomspace.
        """
        import time

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            st = self.status("-")
            if isinstance(st, dict) and st.get("status") == "pathClear":
                return True
            time.sleep(interval_s)
        return False

    def export(self, pattern: str, template: str) -> str:
        """Query atoms matching *pattern* and project them via *template*."""
        path = f"/export/{self._encode_expr(pattern)}/{self._encode_expr(template)}/"
        return self._request("GET", path)

    def transform(self, transforms: str) -> str:
        """Join-match pattern pairs: body is ``(transform …)`` S-expression."""
        return self._request("POST", "/transform/", body=transforms, content_type="text/plain")

    def status(self, expr: str = "-") -> dict | str:
        """Server status or poll last command result."""
        resp = self._request("GET", f"/status/{self._encode_expr(expr)}/")
        try:
            return json.loads(resp)
        except json.JSONDecodeError:
            return resp

    def count(self, expr: str = "-") -> int:
        """Count atoms matching *expr* under the root."""
        self._request("GET", f"/count/{self._encode_expr(expr)}/")
        # The count is asynchronous; poll status for the result.
        st = self.status("-")
        if isinstance(st, dict) and "count" in st:
            return int(st["count"])
        return -1

    def explore(self, expr: str, token: str) -> str:
        """Traverse the atom space tree from *expr* along *token*."""
        return self._request(
            "GET", f"/explore/{self._encode_expr(expr)}/{self._encode_expr(token)}/"
        )

    def copy(self, src: str, dst: str) -> str:
        """Copy atoms from one subspace to another."""
        return self._request("GET", f"/copy/{self._encode_expr(src)}/{self._encode_expr(dst)}/")

    def clear(self, expr: str) -> str:
        """Clear atoms matching *expr*."""
        return self._request("GET", f"/clear/{self._encode_expr(expr)}/")

    # ------------------------------------------------------------------
    # Temporal helper: upload edges / query evidence
    # ------------------------------------------------------------------

    def upload_temporal_edge(
        self,
        edge_uid: str,
        source: str,
        relation: str,
        target: str,
        year: str,
    ) -> str:
        """Upload one temporal graph edge as a MORK S-expression.

        The template preserves the ``edge`` prefix so the data is queryable
        via ``export('(edge $e $s $r $t $y)', '($e $s $r $t $y)')``.
        """
        data = f"(edge {edge_uid} {source} {relation} {target} {year})"
        return self.upload("(edge $e $s $r $t $y)", "(edge $e $s $r $t $y)", data)

    def query_edges_by_source(self, source: str) -> str:
        """Export all edges whose source matches *source*."""
        pattern = f"(edge $e {source} $r $t $y)"
        template = "($e $r $t $y)"
        return self.export(pattern, template)

    # ------------------------------------------------------------------
    # Internal: HTTP + URL encoding
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_expr(expr: str) -> str:
        """Percent-encode an S-expression for use in a URL path segment.

        Preserves parentheses and dots; encodes spaces, ``$``, ``#``, and other
        characters that are special in URLs or in MORK's pattern syntax.
        """
        import urllib.parse

        # Only encode the special characters MORK's URL parser is sensitive to.
        out = []
        for ch in expr:
            if ch in " $#%&+,":
                out.append(urllib.parse.quote(ch))
            else:
                out.append(ch)
        return "".join(out)

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> str:
        import urllib.request
        import urllib.error

        full_url = self.url + path
        data = body.encode("utf-8") if body is not None else None
        req = urllib.request.Request(full_url, data=data, method=method)
        if content_type and body is not None:
            req.add_header("Content-Type", content_type)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return resp.read().decode("utf-8").strip()
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"MORK HTTP {method} {path} returned {exc.code}: {body_text}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"MORK HTTP {method} {path} connection failed: {exc.reason}"
            ) from exc


__all__ = [
    "hyperon_available",
    "make_metta_runner",
    "run_metta",
    "run_metta_file",
    "MORKRunner",
    "mork_available",
    "MORKHttpRunner",
    "mork_http_available",
]

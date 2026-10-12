"""Lean 4 kernel adapter.

The constraint is the ``lean`` binary on this machine. This module does not
reimplement a type theory and does not edit a snippet after Lean rejects it.
A snippet is accepted only when that binary exits 0 on the text it was given.
No lakefile and no mathlib are required; snippets are checked against Init alone.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_LEAN_BIN = "/Users/fulkanjou/.elan/bin/lean"
_NO_TOOLCHAIN = "no default toolchain configured"

# ``sorry`` / ``admit`` as tokens. ``sorrow`` and ``admission`` do not count.
_FORBIDDEN_TOKEN = re.compile(r"\b(?:sorry|admit)\b")
_BLOCK_COMMENT = re.compile(r"/-.*?-/", re.DOTALL)
_AXIOM_DECL = re.compile(
    r"^[ \t]*(?:@\[[^\]]*\][ \t]*)*"
    r"(?:(?:private|protected|noncomputable|unsafe|scoped|partial)[ \t]+)*"
    r"axiom\b"
)
# (lean binary, ELAN_TOOLCHAIN already in the environment) -> env for children.
_ENVS: dict[tuple[str, str], dict[str, str]] = {}


@dataclass(frozen=True)
class CheckResult:
    """What ``lean`` decided about one snippet.

    ``accepted`` is true only for a clean successful check: exit code 0, and
    no ``sorry``, ``admit``, or fresh ``axiom``. ``has_sorry`` records the
    sorry/admit rejection separately from other failures.
    """

    accepted: bool
    has_sorry: bool
    returncode: int
    message: str


def check_snippet(source: str, *, timeout_s: float = 60.0) -> CheckResult:
    """Run `lean` on `source`. Accepted only if exit code is 0 AND the snippet does not use sorry, admit, or a new axiom.
    A warning that a declaration uses sorry means rejected. Timeout or missing binary means rejected, not accepted.
    """
    has_token = _FORBIDDEN_TOKEN.search(source) is not None
    declares_axiom = _declares_axiom(source)
    lean_bin = os.environ.get("LEAN_BIN") or _DEFAULT_LEAN_BIN
    if not _is_executable(lean_bin):
        return CheckResult(
            accepted=False,
            has_sorry=has_token,
            returncode=-1,
            message=f"lean binary not found: {lean_bin}",
        )

    returncode, message = _check_source(lean_bin, source, timeout_s)
    has_sorry = has_token or _output_blocks(message)
    accepted = returncode == 0 and not has_sorry and not declares_axiom
    return CheckResult(
        accepted=accepted,
        has_sorry=has_sorry,
        returncode=returncode,
        message=message,
    )


def check_many(
    sources: list[str],
    *,
    workers: int,
    timeout_s: float = 60.0,
    on_result: object = None,
) -> list[CheckResult]:
    """Typecheck ``sources`` with up to ``workers`` Lean processes.

    Results line up with ``sources``. Each worker is one process and sets
    ``LEAN_NUM_THREADS`` to 1, so a machine with many cores spends them on
    separate files rather than on one compiler. A worker that raises is a
    rejection, not an acceptance. ``on_result``, if given, is called as
    ``on_result(index, result)`` as each file finishes.
    """
    if workers < 1:
        raise ValueError("workers must be positive")
    if not sources:
        return []
    if workers == 1 or len(sources) == 1:
        results = [check_snippet(source, timeout_s=timeout_s) for source in sources]
        if on_result is not None:
            for index, result in enumerate(results):
                on_result(index, result)
        return results

    results: list[CheckResult | None] = [None] * len(sources)
    with ProcessPoolExecutor(
        max_workers=min(workers, len(sources)),
        initializer=_lean_worker_init,
    ) as pool:
        futures = [
            pool.submit(_check_indexed, index, source, timeout_s)
            for index, source in enumerate(sources)
        ]
        for future in as_completed(futures):
            index, result = future.result()
            results[index] = result
            if on_result is not None:
                on_result(index, result)
    return [result if result is not None else _worker_failed() for result in results]


def _lean_worker_init() -> None:
    """One core per Lean process. The pool already supplies the processes."""
    os.environ["LEAN_NUM_THREADS"] = "1"


def _check_indexed(index: int, source: str, timeout_s: float) -> tuple[int, CheckResult]:
    try:
        return index, check_snippet(source, timeout_s=timeout_s)
    except Exception as exc:
        return index, CheckResult(
            accepted=False,
            has_sorry=False,
            returncode=-1,
            message=f"worker failed: {exc}",
        )


def _worker_failed() -> CheckResult:
    return CheckResult(
        accepted=False,
        has_sorry=False,
        returncode=-1,
        message="worker returned no result",
    )


def _is_executable(path: str) -> bool:
    return os.path.isfile(path) and os.access(path, os.X_OK)


def _output_blocks(message: str) -> bool:
    """True when Lean's combined output mentions sorry or admit."""
    return "sorry" in message or "admit" in message


def _declares_axiom(source: str) -> bool:
    """True when a code line declares an ``axiom``."""
    code = _BLOCK_COMMENT.sub("", source)
    for line in code.splitlines():
        visible = line.split("--", 1)[0]
        if _AXIOM_DECL.match(visible):
            return True
    return False


def _check_source(lean_bin: str, source: str, timeout_s: float) -> tuple[int, str]:
    """Run Lean on ``source`` exactly as given."""
    env = _environment(lean_bin)
    with tempfile.TemporaryDirectory(prefix="bitter-lean-") as directory:
        path = Path(directory) / "snippet.lean"
        return _run_lean(lean_bin, path, source, timeout_s, env)


def _run_lean(
    lean_bin: str,
    path: Path,
    source: str,
    timeout_s: float,
    env: dict[str, str],
) -> tuple[int, str]:
    path.write_text(source, encoding="utf-8")
    try:
        proc = subprocess.run(
            [lean_bin, path.name],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
            cwd=str(path.parent),
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired as exc:
        message = _combined(exc.stdout, exc.stderr)
        if not message:
            message = f"lean timed out after {timeout_s} seconds"
        return -1, message
    except OSError as exc:
        return -1, f"lean failed to start: {exc}"
    return proc.returncode, _combined(proc.stdout, proc.stderr)


def _combined(stdout: object, stderr: object) -> str:
    return _as_text(stdout) + _as_text(stderr)


def _as_text(data: object) -> str:
    if data is None:
        return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return str(data)


def _environment(lean_bin: str) -> dict[str, str]:
    """Environment for the elan shim.

    ``elan`` with no default toolchain exits before typechecking. When that
    happens, pin an installed toolchain through ``ELAN_TOOLCHAIN`` without
    changing the user's elan default. A toolchain the caller already set is
    left alone. A real ``lean`` binary ignores the variable.
    """
    preset = os.environ.get("ELAN_TOOLCHAIN", "")
    key = (lean_bin, preset)
    cached = _ENVS.get(key)
    if cached is not None:
        return cached.copy()

    env = os.environ.copy()
    if not preset:
        probe_text, probe_code = _probe_version(lean_bin, env)
        if probe_code != 0 and _NO_TOOLCHAIN in probe_text:
            chosen = _discover_toolchain(lean_bin)
            if chosen:
                env["ELAN_TOOLCHAIN"] = chosen
    _ENVS[key] = env
    return env.copy()


def _probe_version(lean_bin: str, env: dict[str, str]) -> tuple[str, int]:
    try:
        proc = subprocess.run(
            [lean_bin, "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        detail = getattr(exc, "stderr", None)
        return _as_text(detail) or str(exc), 1
    return _combined(proc.stdout, proc.stderr), proc.returncode


def _discover_toolchain(lean_bin: str) -> str | None:
    roots = [
        Path(lean_bin).resolve().parent.parent / "toolchains",
        Path.home() / ".elan" / "toolchains",
    ]
    names: list[str] = []
    for root in roots:
        if not root.is_dir():
            continue
        for entry in root.iterdir():
            if (entry / "bin" / "lean").is_file():
                names.append(_elan_toolchain_name(entry.name))
        if names:
            break
    lean4 = [name for name in names if "lean4" in name]
    return _prefer_toolchain(lean4 or names)


def _elan_toolchain_name(dirname: str) -> str:
    """Turn ``leanprover--lean4---v4.15.0`` into ``leanprover/lean4:v4.15.0``."""
    if "---" not in dirname:
        return dirname
    channel, version = dirname.split("---", 1)
    return f"{channel.replace('--', '/')}:{version}"


def _prefer_toolchain(names: list[str]) -> str | None:
    if not names:
        return None
    stable = [name for name in names if "-rc" not in name.split(":")[-1]]
    pool = stable or list(names)

    def version_key(name: str) -> tuple[int, ...]:
        raw = name.split(":")[-1]
        if raw.startswith("v"):
            raw = raw[1:]
        raw = raw.split("-", 1)[0]
        parts: list[int] = []
        for piece in raw.split("."):
            parts.append(int(piece) if piece.isdigit() else 0)
        return tuple(parts)

    return max(pool, key=version_key)

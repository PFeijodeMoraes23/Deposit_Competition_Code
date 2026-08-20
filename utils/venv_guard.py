"""Helpers to enforce execution inside the project-local .venv."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _venv_python(project_dir: Path) -> Path:
    """Interpreter for the project venv.

    ``OPEN_FINANCE_VENV`` points at a venv root kept outside the repo, so the
    environment does not have to live under a cloud-synced working tree. When
    it is unset the layout is the in-repo ``.venv``.
    """
    override = os.environ.get("OPEN_FINANCE_VENV")
    if override:
        return Path(override) / "Scripts" / "python.exe"
    return project_dir / ".venv" / "Scripts" / "python.exe"


def ensure_project_venv(script_file: str | Path, quiet: bool = False) -> None:
    """
    Re-exec the current script with project .venv Python if needed.

    Safe no-op when already inside the target venv.
    """
    if os.environ.get("OPEN_FINANCE_VENV_ENFORCED") == "1":
        return

    script_path = Path(script_file).resolve()
    if script_path.is_dir():
        candidate_dirs = [script_path]
    else:
        candidate_dirs = [script_path.parent, script_path.parent.parent]

    project_dir = None
    if not os.environ.get("OPEN_FINANCE_VENV"):
        for cand in candidate_dirs:
            if (cand / ".venv").exists():
                project_dir = cand
                break
    if project_dir is None:
        project_dir = script_path.parent if script_path.is_file() else script_path

    target_py = _venv_python(project_dir)

    if not target_py.exists():
        if not quiet:
            print(
                f"[VENV] WARNING: no project interpreter at {target_py}; running under "
                f"{sys.executable} instead. Set OPEN_FINANCE_VENV to the venv root.",
                file=sys.stderr,
            )
        return

    current = Path(sys.executable).resolve()
    target = target_py.resolve()
    if current == target:
        return

    env = os.environ.copy()
    env["OPEN_FINANCE_VENV_ENFORCED"] = "1"

    if not quiet:
        print(f"[VENV] Re-launching with project venv: {target}")

    cmd = [str(target), str(script_path), *sys.argv[1:]]
    result = subprocess.run(cmd, env=env)
    raise SystemExit(result.returncode)

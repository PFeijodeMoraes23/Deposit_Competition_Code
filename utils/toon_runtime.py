"""Runtime helpers to consume TOON context in pipeline scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from utils.toon_parser import (
    _safe_read_text,
    get_script_config,
    load_default_toon_context,
)


def load_runtime_context(script_dir: str | Path, quiet: bool = True) -> dict[str, Any]:
    """
    Load TOON context with priority:
      1) TOON_CONTEXT_PATH JSON file
      2) utils/gemini_*.html parser output
    """
    ctx_path = os.environ.get("TOON_CONTEXT_PATH", "").strip()
    if ctx_path:
        p = Path(ctx_path)
        if p.exists():
            import json

            try:
                raw = _safe_read_text(p)
                obj = json.loads(raw)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

    utils_dir = Path(script_dir) / "utils"
    return load_default_toon_context(utils_dir, quiet=quiet)


def resolve_script_paths(
    script_name: str,
    defaults: dict[str, str],
    script_dir: str | Path,
) -> dict[str, str]:
    """
    Resolve path settings for a script from TOON context.

    Expected TOON schema:
      {
        "<script_name>": {
          "paths": {
            "some_key": "C:/..."
          }
        }
      }

    Backward-compatible fallback:
      direct keys under script config are also accepted.
    """
    ctx = load_runtime_context(script_dir, quiet=True)
    cfg = get_script_config(ctx, script_name)
    if not isinstance(cfg, dict):
        return defaults

    resolved = dict(defaults)

    path_cfg = cfg.get("paths", {}) if isinstance(cfg.get("paths", {}), dict) else {}

    for key, default_value in defaults.items():
        candidate = path_cfg.get(key, cfg.get(key))
        if isinstance(candidate, str) and candidate.strip():
            resolved[key] = str(Path(candidate).expanduser())
        else:
            resolved[key] = default_value

    return resolved

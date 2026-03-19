"""
toon_parser.py
==============
Reusable TOON parser utilities for extracting structured runtime configuration
and markdown artifacts from Gemini/HTML chat exports.

Design goals
------------
* Non-breaking: if no parseable TOON payload exists, return an empty context.
* Portable: pure-Python fallback parsing without hard dependency on bs4.
* Practical: support JSON and simple key-value style blocks.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover
    BeautifulSoup = None


@dataclass
class HtmlExtraction:
    """Container for extracted text from an HTML source file."""

    source: Path
    title: str
    text: str
    code_blocks: list[str]


TOON_FENCE_RE = re.compile(
    r"```(?:toon|json|yaml|yml|toml)?\s*\n(.*?)\n```",
    flags=re.IGNORECASE | re.DOTALL,
)
TOON_MARKER_RE = re.compile(
    r"<!--\s*TOON(?::[^>]*)?\s*-->(.*?)<!--\s*/TOON\s*-->",
    flags=re.IGNORECASE | re.DOTALL,
)
JSON_OBJECT_RE = re.compile(r"\{[\s\S]*?\}")

# Canonical context sections consumed by local scripts.
CANONICAL_SECTIONS = {
    "run_data_pipeline": "run_data_pipeline",
    "pipeline": "run_data_pipeline",
    "estimation_1_sleep": "estimation_1_sleep",
    "estimation": "estimation_1_sleep",
    "export_comments": "export_comments",
    "export": "export_comments",
    "markdown_artifacts": "markdown_artifacts",
}


def _safe_read_text(path: Path) -> str:
    """Read text with conservative fallback encodings."""
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def _extract_from_html(path: Path) -> HtmlExtraction:
    raw = _safe_read_text(path)

    if BeautifulSoup is None:
        title = ""
        text = raw
        code_blocks = TOON_FENCE_RE.findall(raw)
        return HtmlExtraction(source=path, title=title, text=text, code_blocks=code_blocks)

    soup = BeautifulSoup(raw, "lxml")
    title = (soup.title.get_text(strip=True) if soup.title else "")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text("\n", strip=True)
    code_blocks = [tag.get_text("\n", strip=True) for tag in soup.find_all(["pre", "code"])]

    return HtmlExtraction(source=path, title=title, text=text, code_blocks=code_blocks)


def _extract_candidates(extraction: HtmlExtraction) -> list[str]:
    """Collect candidate payload blocks that may contain TOON data."""
    cands: list[str] = []

    combined = "\n\n".join([extraction.text, *extraction.code_blocks])

    cands.extend(TOON_MARKER_RE.findall(combined))
    cands.extend(TOON_FENCE_RE.findall(combined))

    for block in extraction.code_blocks:
        block_s = block.strip()
        if not block_s:
            continue
        if block_s.startswith("{") and block_s.endswith("}"):
            cands.append(block_s)

    # Last-resort JSON candidate scan.
    for match in JSON_OBJECT_RE.findall(combined):
        if "toon" in match.lower() or any(k in match for k in CANONICAL_SECTIONS):
            cands.append(match)

    # De-duplicate while preserving order.
    unique: list[str] = []
    seen: set[str] = set()
    for cand in cands:
        cand_s = cand.strip()
        if cand_s and cand_s not in seen:
            unique.append(cand_s)
            seen.add(cand_s)
    return unique


def _parse_simple_kv(text: str) -> dict[str, Any]:
    """
    Parse minimal key-value syntax:
      key: value
      nested.key: value
    """
    out: dict[str, Any] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        try:
            parsed_val = ast.literal_eval(value)
        except Exception:
            parsed_val = value

        cursor = out
        parts = [p for p in key.split(".") if p]
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                cursor[part] = {}
            cursor = cursor[part]
        cursor[parts[-1]] = parsed_val

    return out


def _parse_candidate(candidate: str) -> dict[str, Any] | None:
    for parser in (json.loads, ast.literal_eval):
        try:
            obj = parser(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    kv = _parse_simple_kv(candidate)
    return kv if kv else None


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    for k, v in extra.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _canonicalize_context(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize aliases into canonical section names."""
    out: dict[str, Any] = {}

    for key, value in raw.items():
        key_l = str(key).strip().lower()
        if key_l in CANONICAL_SECTIONS and isinstance(value, dict):
            canon = CANONICAL_SECTIONS[key_l]
            out.setdefault(canon, {})
            _deep_merge(out[canon], value)
        else:
            out[key] = value

    # Common top-level keys that may arrive flattened.
    if "brain_dir" in raw:
        out.setdefault("export_comments", {})
        out["export_comments"]["brain_dir"] = raw["brain_dir"]

    for artifact_key in ("estimation_summary", "se_comparison"):
        if artifact_key in raw:
            out.setdefault("markdown_artifacts", {})
            out["markdown_artifacts"][artifact_key] = raw[artifact_key]

    return out


def load_toon_context_from_html(paths: Iterable[str | Path]) -> dict[str, Any]:
    """Load and merge TOON context from one or many HTML files."""
    merged: dict[str, Any] = {}

    for path_in in paths:
        path = Path(path_in)
        if not path.exists() or path.stat().st_size == 0:
            continue

        extraction = _extract_from_html(path)
        for cand in _extract_candidates(extraction):
            parsed = _parse_candidate(cand)
            if not parsed:
                continue
            _deep_merge(merged, _canonicalize_context(parsed))

    return merged


def load_default_toon_context(utils_dir: str | Path, quiet: bool = True) -> dict[str, Any]:
    """
    Convenience loader for local Gemini export files in the utils folder.
    """
    base = Path(utils_dir)
    html_paths = [
        base / "gemini_gems_data.html",
        base / "gemini_scheduled_actions_data.html",
    ]
    ctx = load_toon_context_from_html(html_paths)

    if not quiet and not ctx:
        print("[TOON] No parseable TOON payload found in Gemini HTML exports.")

    return ctx


def get_script_config(context: dict[str, Any], script_name: str) -> dict[str, Any]:
    """Fetch canonical script-specific config section."""
    if not context:
        return {}
    return context.get(script_name, {}) if isinstance(context.get(script_name, {}), dict) else {}


def read_artifact_text(spec: Any, base_dir: str | Path | None = None) -> str:
    """
    Read an artifact spec that can be:
      * inline text
      * {'text': '...'}
      * {'path': 'relative/or/absolute/path'}
    """
    if spec is None:
        return ""

    if isinstance(spec, str):
        candidate = Path(spec)
        if candidate.exists():
            return _safe_read_text(candidate)
        if base_dir is not None:
            joined = Path(base_dir) / spec
            if joined.exists():
                return _safe_read_text(joined)
        return spec

    if isinstance(spec, dict):
        if "text" in spec and isinstance(spec["text"], str):
            return spec["text"]
        if "path" in spec and isinstance(spec["path"], str):
            return read_artifact_text(spec["path"], base_dir=base_dir)

    return ""


def load_markdown_artifacts(
    context: dict[str, Any],
    default_brain_dir: str | Path,
    summary_filename: str = "estimation_summary.md",
    se_filename: str = "se_comparison.md",
) -> tuple[str, str, dict[str, str]]:
    """
    Resolve summary/se markdown text from TOON context first, then fallback files.

    Returns
    -------
    (summary_text, se_text, metadata)
    """
    export_cfg = get_script_config(context, "export_comments")
    md_cfg = context.get("markdown_artifacts", {}) if isinstance(context.get("markdown_artifacts", {}), dict) else {}

    brain_dir = Path(export_cfg.get("brain_dir", default_brain_dir))

    summary_text = read_artifact_text(md_cfg.get("estimation_summary"), base_dir=brain_dir)
    se_text = read_artifact_text(md_cfg.get("se_comparison"), base_dir=brain_dir)

    # Fallback to local files expected by existing workflow.
    summary_path = brain_dir / summary_filename
    se_path = brain_dir / se_filename

    if not summary_text and summary_path.exists():
        summary_text = _safe_read_text(summary_path)
    if not se_text and se_path.exists():
        se_text = _safe_read_text(se_path)

    meta = {
        "brain_dir": str(brain_dir),
        "summary_path": str(summary_path),
        "se_path": str(se_path),
    }
    return summary_text, se_text, meta


def dump_context_json(context: dict[str, Any], out_path: str | Path) -> Path:
    """Save normalized context to JSON for debugging/interoperability."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(context, ensure_ascii=True, indent=2), encoding="utf-8")
    return out

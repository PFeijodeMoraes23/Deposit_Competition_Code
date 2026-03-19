"""CLI helper for TOON context extraction from Gemini HTML exports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _THIS_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None

if ensure_project_venv is not None:
    ensure_project_venv(_PROJECT_DIR, quiet=True)

from utils.toon_parser import dump_context_json, load_default_toon_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse Gemini HTML files into TOON JSON context.")
    parser.add_argument(
        "--utils-dir",
        default=str(Path(__file__).resolve().parent),
        help="Path to folder containing gemini_gems_data.html and gemini_scheduled_actions_data.html.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output JSON path (defaults to <utils-dir>/toon_context.json).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress parser status messages.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    utils_dir = Path(args.utils_dir)
    context = load_default_toon_context(utils_dir, quiet=args.quiet)

    out_path = Path(args.out) if args.out else (utils_dir / "toon_context.json")
    dump_context_json(context, out_path)

    if not args.quiet:
        status = "empty" if not context else f"{len(context)} top-level sections"
        print(f"[TOON] Wrote {status} to: {out_path}")


if __name__ == "__main__":
    main()

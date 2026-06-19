"""audit/audit_3_path_refs.py — Phase 3: map how each project references paths.

Scans code under ``Open-Finance/Code/*`` and ``Open-Finance/Drafts/`` for:
  (1) the presence of a central path module / override layer (a file named
      paths.* / config.*, or an ``import paths`` / ``resolve_script_paths`` use
      like the Egan repo's), and
  (2) hardcoded path literals (OneDrive, Open[- ]Finance, /BCB/, "raw",
      "processed", ``../..``, drive letters, os.path.join / Path( / here::here).

Emits ``path_refs.csv``.  The point is to see which projects already centralize
their paths versus hardcode them, and which datasets are referenced from more
than one project (the cross-project consumers).

Read-only.
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import paths  # noqa: E402

LOG = logging.getLogger("audit.pathrefs")

CODE_EXTS = {".py", ".r", ".jl", ".do", ".m", ".sh"}
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__",
             ".ipynb_checkpoints", ".Rproj.user"}

LITERAL_RE = re.compile(
    r"(OneDrive|Open[- ]Finance|[\\/]BCB[\\/]|[\"']raw[\"']|[\"']processed[\"']"
    r"|\.\.[\\/]\.\.|[A-Za-z]:[\\/]"
    r"|os\.path\.join|Path\(|here::here|file\.path|read_parquet|read_csv|readRDS)",
    re.IGNORECASE,
)
MODULE_RE = re.compile(
    r"(import\s+paths|from\s+\S+\s+import\s+paths|resolve_script_paths"
    r"|source\([^)]*paths)", re.IGNORECASE)
MODULE_FILE_RE = re.compile(r"^(paths|config|globals|settings)\.(py|r|jl)$",
                            re.IGNORECASE)

DATASET_HINTS = ["ESTBAN", "IF_DATA", "IF Data", "COSIF", "SGS", "ANATEL",
                 "CadUnico", "PIX", "IBGE", "Inclusion", "Tarifas", "FGC",
                 "WorldBank", "Findex", "SCR", "STR", "RAIS", "INSS", "CVM",
                 "EDGAR", "SEC", "Accounts"]


def project_of(scan_root: Path, full: Path) -> str:
    """Group loose ``Code/*`` scripts under "Code"; subprojects as "Code/<sub>"."""
    dirs = full.relative_to(scan_root).parts[:-1]   # drop filename
    if not dirs:
        return "(root)"
    if dirs[0] == "Code":
        return "/".join(dirs[:2]) if len(dirs) >= 2 else "Code"
    return dirs[0]


def infer_dataset(line: str) -> str:
    low = line.lower()
    return "|".join(d for d in DATASET_HINTS if d.lower() in low)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Map path references across projects (read-only).")
    ap.add_argument("--root", type=Path, default=paths.OPEN_FINANCE,
                    help="Code scan root (default: Open-Finance).")
    ap.add_argument("--out", type=Path,
                    default=paths.OPEN_FINANCE.parent / "_audit_output")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    scan_root = args.root.resolve()
    targets = [scan_root / "Code", scan_root / "Drafts"]

    out_rows: list[tuple] = []
    module_projects: set[str] = set()
    for base in targets:
        if not base.exists():
            LOG.warning("missing: %s", base)
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for name in filenames:
                full = Path(dirpath) / name
                if full.suffix.lower() not in CODE_EXTS:
                    continue
                proj = project_of(scan_root, full)
                if MODULE_FILE_RE.match(name):
                    out_rows.append((proj, str(full), 0, "module-file", name, ""))
                    module_projects.add(proj)
                try:
                    text = full.read_text(encoding="utf-8", errors="replace")
                except OSError as e:
                    LOG.warning("read failed: %s (%s)", full, e)
                    continue
                for lineno, line in enumerate(text.splitlines(), 1):
                    if MODULE_RE.search(line):
                        out_rows.append((proj, str(full), lineno, "module",
                                         line.strip()[:200], ""))
                        module_projects.add(proj)
                    elif LITERAL_RE.search(line):
                        out_rows.append((proj, str(full), lineno, "literal",
                                         line.strip()[:200], infer_dataset(line)))

    out_csv = args.out / "path_refs.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["project", "file", "line", "kind", "snippet",
                    "inferred_dataset"])
        w.writerows(out_rows)
    LOG.info("Wrote %s (%d rows). Projects with a path module/override: %s",
             out_csv, len(out_rows),
             ", ".join(sorted(module_projects)) or "(none)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

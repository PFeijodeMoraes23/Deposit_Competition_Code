#!/usr/bin/env python
"""
sync_identification_figs.py
===========================
Refresh the figure COPIES that identification_notes.md renders from, then (optionally) rebuild
the PDF.

Why this exists. The phi battery (run_phi_diagnostics.py) regenerates its figures into
    processed/ESTIMATION_OUTPUT/DIAG_PHI_SEPARATION/
but identification_notes.md references
    identification_figs/<name>
i.e. COPIES sitting next to the note in Drafts/Deposit Competition. Re-rendering the PDF without
refreshing those copies silently produces a document with NEW text and OLD figures — the failure
mode this script exists to prevent.

It is deliberately conservative:
  * only files the .md ACTUALLY references are copied (never a blanket sync), so an unreferenced
    experiment in DIAG_PHI_SEPARATION cannot leak into the note;
  * a copy happens only when the source is NEWER than the destination;
  * a referenced figure with no source is reported, not silently skipped — that means the .md
    points at something the battery no longer produces.

Usage:
  python sync_identification_figs.py              # sync only, report what changed
  python sync_identification_figs.py --render     # sync, then rebuild the PDF
  python sync_identification_figs.py --dry-run
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import shutil
import subprocess
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_ROOT = Path(__file__).resolve().parents[2]
DRAFTS = _ROOT / "Drafts" / "Deposit Competition"
NOTES = DRAFTS / "identification_notes.md"
FIGS_DST = DRAFTS / "identification_figs"
FIGS_SRC = (_ROOT / "BCB" / "Egan_et_al_2025_Rep" / "processed" / "ESTIMATION_OUTPUT"
            / "DIAG_PHI_SEPARATION")


def referenced_figures(md: Path) -> list[str]:
    """Figure names the note actually renders (identification_figs/<name>)."""
    txt = md.read_text(encoding="utf-8", errors="replace")
    return sorted(set(re.findall(r"identification_figs/([A-Za-z0-9_.\-]+)", txt)))


def sync(dry: bool = False) -> tuple[int, list[str]]:
    if not NOTES.exists():
        raise SystemExit(f"notes not found: {NOTES}")
    FIGS_DST.mkdir(parents=True, exist_ok=True)

    names = referenced_figures(NOTES)
    print(f"{len(names)} figure(s) referenced by {NOTES.name}")
    copied, missing, current = 0, [], 0

    for n in names:
        src, dst = FIGS_SRC / n, FIGS_DST / n
        if not src.exists():
            missing.append(n)
            continue
        if dst.exists() and src.stat().st_mtime <= dst.stat().st_mtime:
            current += 1
            continue
        age = dt.datetime.fromtimestamp(src.stat().st_mtime)
        print(f"  {'would copy' if dry else 'copied'}  {n}   (source {age:%m-%d %H:%M})")
        if not dry:
            shutil.copy2(src, dst)
        copied += 1

    print(f"\n  {copied} refreshed, {current} already current, {len(missing)} missing")
    if missing:
        print("  !! referenced but NOT produced by the battery — the note points at a figure "
              "that no longer exists:")
        for n in missing:
            print(f"       {n}")
    return copied, missing


def render() -> int:
    builder = DRAFTS / "build_notes_pdf.py"
    if not builder.exists():
        print(f"!! {builder.name} not found — cannot render"); return 1
    print(f"\nrendering {NOTES.name} -> PDF")
    r = subprocess.run([sys.executable, str(builder), NOTES.name], cwd=str(DRAFTS))
    if r.returncode == 0:
        pdf = NOTES.with_suffix(".pdf")
        if pdf.exists():
            print(f"  OK -> {pdf.name} ({pdf.stat().st_size/1e6:.2f} MB, "
                  f"{dt.datetime.fromtimestamp(pdf.stat().st_mtime):%H:%M:%S})")
    else:
        print(f"  !! build_notes_pdf.py exited {r.returncode}")
    return r.returncode


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true", help="rebuild the PDF after syncing")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    _, missing = sync(dry=a.dry_run)
    rc = 0
    if a.render and not a.dry_run:
        rc = render()
    sys.exit(rc)

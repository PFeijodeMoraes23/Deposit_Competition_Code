"""audit/audit_report.py — Phase 4: roll the audit CSVs into SUMMARY.md.

Reads inventory.csv / duplicates_exact.csv / collisions_name.csv / path_refs.csv
from the audit output dir and writes a human-readable ``SUMMARY.md``: sized tree
by project, duplicate totals + reclaimable bytes, top name collisions, a
per-project path-hygiene table, and the candidate cross-project shared datasets
(future ``Open-Finance/shared/``).

Read-only.
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import paths  # noqa: E402

LOG = logging.getLogger("audit.report")


def _read(p: Path) -> list[dict]:
    if not p.exists():
        LOG.warning("missing (skipped): %s", p)
        return []
    with open(p, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _mb(n: float) -> str:
    return f"{n / 1e6:,.1f} MB"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Summarize the Open Finance audit.")
    ap.add_argument("--out", type=Path,
                    default=paths.OPEN_FINANCE.parent / "_audit_output")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    inv = _read(args.out / "inventory.csv")
    dup = _read(args.out / "duplicates_exact.csv")
    coll = _read(args.out / "collisions_name.csv")
    refs = _read(args.out / "path_refs.csv")

    size_by_proj: dict[str, int] = defaultdict(int)
    count_by_proj: dict[str, int] = defaultdict(int)
    total = 0
    for r in inv:
        sz = int(r["size_bytes"])
        total += sz
        size_by_proj[r["project"]] += sz
        count_by_proj[r["project"]] += 1

    reclaim = sum(int(r["reclaimable_bytes"]) for r in dup)
    cross_dup = [r for r in dup if r["span"] == "cross-project"]

    module_projects = {r["project"] for r in refs
                       if r["kind"] in ("module", "module-file")}
    literal_projects = {r["project"] for r in refs if r["kind"] == "literal"}
    all_projects = module_projects | literal_projects

    ds_projects: dict[str, set[str]] = defaultdict(set)
    for r in refs:
        for d in (r.get("inferred_dataset") or "").split("|"):
            if d:
                ds_projects[d].add(r["project"])
    shared = {d: ps for d, ps in ds_projects.items() if len(ps) > 1}

    L: list[str] = []
    A = L.append
    A("# Open Finance folder audit — SUMMARY\n")
    A(f"Scan root: `{paths.OPEN_FINANCE.parent}`\n")
    materialized = sum(1 for r in inv if r["materialized"] == "True")
    A(f"- Data files inventoried: **{len(inv)}** ({materialized} materialized locally)")
    A(f"- Total data size (materialized stats): **{_mb(total)}**")
    A(f"- Exact-duplicate groups: **{len(dup)}** → reclaimable **{_mb(reclaim)}** "
      f"({len(cross_dup)} span projects)")
    A(f"- Name-collision groups (version drift): **{len(coll)}**\n")

    A("## Size by project\n")
    A("| project | files | size |")
    A("|---|---:|---:|")
    for p, sz in sorted(size_by_proj.items(), key=lambda kv: -kv[1])[:25]:
        A(f"| {p} | {count_by_proj[p]} | {_mb(sz)} |")
    A("")

    A("## Largest exact-duplicate groups\n")
    A(f"_Paths shown relative to_ `{paths.OPEN_FINANCE.parent}`\n")
    prefix = str(paths.OPEN_FINANCE.parent) + os.sep
    A("| reclaimable | copies | span | paths (relative) |")
    A("|---:|---:|---|---|")
    for r in dup[:15]:
        rel = r["paths"].replace(prefix, "")
        A(f"| {_mb(int(r['reclaimable_bytes']))} | {r['n_copies']} | {r['span']} "
          f"| {rel[:400]} |")
    A("")

    A("## Top name collisions (same name, different/unknown content)\n")
    A("| filename | locations | span | projects |")
    A("|---|---:|---|---|")
    for r in coll[:20]:
        A(f"| {r['filename']} | {r['n_locations']} | {r['span']} | {r['projects']} |")
    A("")

    A("## Path hygiene by project\n")
    A("| project | has path module/override | hardcoded literals |")
    A("|---|:--:|:--:|")
    for p in sorted(all_projects):
        A(f"| {p} | {'yes' if p in module_projects else 'NO'} "
          f"| {'yes' if p in literal_projects else '-'} |")
    A("")

    A("## Candidate cross-project shared datasets (future `Open-Finance/shared/`)\n")
    if shared:
        A("| dataset | consuming projects |")
        A("|---|---|")
        for d, ps in sorted(shared.items(), key=lambda kv: -len(kv[1])):
            A(f"| {d} | {', '.join(sorted(ps))} |")
    else:
        A("_None detected from path references (run audit_3_path_refs.py first)._")
    A("")

    A("## Recommended next pass (NOT performed in this audit)\n")
    A("1. Confirm & remove redundant copies from `duplicates_exact.csv` after "
      "choosing a canonical home (e.g. the known stray "
      "`Open Finance/BCB/Egan_et_al_2025_Rep/processed`).")
    A("2. Resolve `collisions_name.csv` version drift (keep the newest "
      "authoritative copy; delete/rename stale ones).")
    A("3. Promote the shared datasets above into `Open-Finance/shared/<DOMAIN>/`.")
    A("4. Give each project a `paths` module mirroring "
      "`Code/Egan_et_al_2025_Rep/utils/paths.py`; route every literal through it.")

    out_md = args.out / "SUMMARY.md"
    out_md.write_text("\n".join(L), encoding="utf-8")
    LOG.info("Wrote %s", out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""audit/audit_2_duplicates.py — Phase 2: redundant copies & name collisions.

Reads ``inventory.csv`` (from audit_1_inventory.py) and emits:

  duplicates_exact.csv  — identical content (same sha256) in >=2 paths.  These
                          are truly redundant; safe-delete candidates once a
                          canonical home is chosen.
  collisions_name.csv   — same filename in >=2 directories but with differing
                          (or unknown) content: version drift / stale copies.
                          These need a human decision, never auto-delete.

Read-only: this script only reads the CSV; it never touches the data files.
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

LOG = logging.getLogger("audit.duplicates")


def load(inv_csv: Path) -> list[dict]:
    with open(inv_csv, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def span(projects: set[str]) -> str:
    return "cross-project" if len(projects) > 1 else "within-project"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Detect duplicate/colliding data files from inventory.csv.")
    ap.add_argument("--out", type=Path,
                    default=paths.OPEN_FINANCE.parent / "_audit_output")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    inv = args.out / "inventory.csv"
    rows = load(inv)
    LOG.info("Loaded %d rows from %s", len(rows), inv)

    # --- exact duplicates: group by non-empty sha256 -----------------------
    by_hash: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["sha256"]:
            by_hash[r["sha256"]].append(r)
    exact = {h: rs for h, rs in by_hash.items() if len(rs) > 1}

    exact_csv = args.out / "duplicates_exact.csv"
    reclaim = 0
    ordered = sorted(
        exact.items(),
        key=lambda kv: -int(kv[1][0]["size_bytes"]) * (len(kv[1]) - 1),
    )
    with open(exact_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["sha256", "n_copies", "size_bytes", "reclaimable_bytes",
                    "span", "projects", "paths"])
        for h, rs in ordered:
            size = int(rs[0]["size_bytes"])
            rbytes = size * (len(rs) - 1)
            reclaim += rbytes
            projects = {r["project"] for r in rs}
            w.writerow([h, len(rs), size, rbytes, span(projects),
                        "|".join(sorted(projects)),
                        " || ".join(r["path"] for r in rs)])
    LOG.info("Wrote %s (%d hash groups, %.1f MB reclaimable).",
             exact_csv, len(exact), reclaim / 1e6)

    # --- name collisions: same filename, differing/unknown content ---------
    by_name: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_name[os.path.basename(r["rel_path"])].append(r)

    coll_csv = args.out / "collisions_name.csv"
    n_coll = 0
    with open(coll_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "n_locations", "distinct_hashes", "span",
                    "projects", "sizes", "paths"])
        for name, rs in sorted(by_name.items()):
            if len(rs) < 2:
                continue
            hashes = {r["sha256"] for r in rs if r["sha256"]}
            # Skip groups proven byte-identical (covered by duplicates_exact).
            if len(hashes) == 1 and all(r["sha256"] for r in rs):
                continue
            n_coll += 1
            projects = {r["project"] for r in rs}
            w.writerow([name, len(rs), len(hashes), span(projects),
                        "|".join(sorted(projects)),
                        "|".join(r["size_bytes"] for r in rs),
                        " || ".join(r["path"] for r in rs)])
    LOG.info("Wrote %s (%d name-collision groups).", coll_csv, n_coll)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

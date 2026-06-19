"""audit/audit_1_inventory.py — Phase 1 of the Open Finance folder audit.

Walks the whole grandparent "Open Finance" tree and records every *data* file
as metadata only.  OneDrive cloud-only placeholders are detected and NOT
downloaded (we never open them).  Emits ``inventory.csv`` into the audit output
directory.

This script is strictly READ-ONLY with respect to the scanned tree: it never
creates, moves, or deletes anything outside ``--out``.

Anchor reuse: the scan root defaults to ``paths.OPEN_FINANCE.parent`` — i.e. the
grandparent ``Open Finance`` (with a space), so the known stray copy at
``Open Finance/BCB/Egan_et_al_2025_Rep/processed`` is in scope.

    python audit/audit_1_inventory.py                 # full tree
    python audit/audit_1_inventory.py --root "<dir>"   # one subtree (dry run)
    python audit/audit_1_inventory.py --no-hash        # metadata only, no hashing
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Make ``from utils import paths`` work no matter where this is run from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import paths  # noqa: E402

LOG = logging.getLogger("audit.inventory")

# --- config -----------------------------------------------------------------
DATA_EXTS = {
    "csv", "parquet", "dta", "rds", "rdata", "feather", "xlsx", "xls",
    "json", "zip", "pkl", "npy", "npz", "h5", "hdf5", "sas7bdat", "sav",
}
SKIP_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    ".ipynb_checkpoints", ".Rproj.user", "Bibliography", "_audit_output",
}

# Windows cloud / offline attribute bits (OneDrive Files On-Demand).
FILE_ATTRIBUTE_OFFLINE               = 0x00001000
FILE_ATTRIBUTE_RECALL_ON_OPEN        = 0x00040000
FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS = 0x00400000
_CLOUD_BITS = (
    FILE_ATTRIBUTE_OFFLINE
    | FILE_ATTRIBUTE_RECALL_ON_OPEN
    | FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS
)
_INVALID = 0xFFFFFFFF


def is_cloud_only(path: Path) -> bool:
    """True if *path* is a OneDrive placeholder not materialized locally.

    Uses ``GetFileAttributesW`` (metadata only) — never opens the file, so it
    cannot trigger a download.  Always False on non-Windows.
    """
    if os.name != "nt":
        return False
    attrs = ctypes.windll.kernel32.GetFileAttributesW(str(path))
    if attrs == _INVALID:
        return False
    return bool(attrs & _CLOUD_BITS)


def project_of(rel_parts: tuple[str, ...]) -> str:
    """Project/domain label for a file path relative to the scan root.

    Uses up to TWO leading directory levels so e.g. the Egan repo's 127 GB data
    tree (``BCB/Egan_et_al_2025_Rep``) is not lumped with the rest of ``BCB``.
    Grandparent-level trees (anything NOT under ``Open-Finance``) are prefixed
    with ``~`` so the stray ``Open Finance/BCB`` never collides with the real
    ``Open-Finance/BCB``.  The trailing filename is dropped before labelling.
    """
    dirs = rel_parts[:-1]                     # drop the filename itself
    if not dirs:
        return "(root-file)"                  # data file sitting at the scan root
    if dirs[0] != "Open-Finance":
        return "~" + "/".join(dirs[:2])       # grandparent-level (stray) tree
    sub = dirs[1:]
    return "/".join(sub[:2]) if sub else "Open-Finance/(root)"


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def walk(scan_root: Path):
    """Yield (full_path, rel_parts, stat_result, cloud_only) per data file."""
    for dirpath, dirnames, filenames in os.walk(scan_root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
            if ext not in DATA_EXTS:
                continue
            full = Path(dirpath) / name
            try:
                st = full.stat()
            except OSError as e:
                LOG.warning("stat failed: %s (%s)", full, e)
                continue
            rel = full.relative_to(scan_root)
            yield full, rel.parts, st, is_cloud_only(full)


FIELDS = ["path", "rel_path", "project", "ext", "size_bytes", "mtime",
          "materialized", "sha256"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Inventory data files under the Open Finance tree (read-only).")
    ap.add_argument("--root", type=Path, default=paths.OPEN_FINANCE.parent,
                    help="Scan root (default: grandparent 'Open Finance').")
    ap.add_argument("--out", type=Path,
                    default=paths.OPEN_FINANCE.parent / "_audit_output",
                    help="Output directory for inventory.csv.")
    ap.add_argument("--no-hash", action="store_true",
                    help="Skip hashing entirely (metadata-only run).")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    scan_root = args.root.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    LOG.info("Scanning %s", scan_root)

    rows: list[dict] = []
    by_size: dict[int, list[int]] = defaultdict(list)  # size -> materialized row idxs
    for full, rel_parts, st, cloud in walk(scan_root):
        idx = len(rows)
        rows.append({
            "path": str(full),
            "rel_path": str(Path(*rel_parts)) if rel_parts else full.name,
            "project": project_of(rel_parts),
            "ext": full.suffix.lower().lstrip("."),
            "size_bytes": st.st_size,
            "mtime": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
            "materialized": (not cloud),
            "sha256": "",
        })
        if not cloud:
            by_size[st.st_size].append(idx)

    LOG.info("Found %d data files (%d materialized).",
             len(rows), sum(1 for r in rows if r["materialized"]))

    # Two-pass hashing: only hash files whose size collides with another.
    if not args.no_hash:
        groups = [ids for ids in by_size.values() if len(ids) > 1]
        to_hash = [i for ids in groups for i in ids]
        LOG.info("Hashing %d files across %d size-collision groups.",
                 len(to_hash), len(groups))
        for n, i in enumerate(to_hash, 1):
            try:
                rows[i]["sha256"] = sha256_of(Path(rows[i]["path"]))
            except OSError as e:
                LOG.warning("hash failed: %s (%s)", rows[i]["path"], e)
            if n % 200 == 0:
                LOG.info("  hashed %d/%d", n, len(to_hash))

    out_csv = args.out / "inventory.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    LOG.info("Wrote %s (%d rows).", out_csv, len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

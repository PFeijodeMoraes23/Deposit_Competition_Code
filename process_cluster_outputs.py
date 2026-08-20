#!/usr/bin/env python3
"""
process_cluster_outputs.py -- unpack the BBL and CF cluster archives into the local tree.

The BLP return trip is automated by process_blp_outputs.py (zip -> cluster_raw/ +
cluster_processed/ + INDEX.json + SUMMARY.md). BBL and CF had no counterpart:
make_bbl_cost_tables.py and export_cf1_franchise.py both assume somebody already unzipped
the download by hand into the right place. This is that counterpart, and it deliberately
mirrors process_blp_outputs.py -- same cluster_raw/cluster_processed split, same
newest-zip-wins discovery, same INDEX.json beside the processed artifacts -- rather than
inventing a second set of conventions for the same job.

    --kind bbl    bbl_outputs_<jobid>.zip / psi_cost.zip / cf2_outputs.zip
                  -> BBL_OUTPUT/cluster_raw/       the archive itself (copied)
                  -> BBL_OUTPUT/cluster_processed/ cost_params_E{k}_spec_{s}_{stage}.json
                  The psi_eq / psi_dev shards stay INSIDE the archive on purpose: a full
                  psi run is ~400 shard parquets and make_bbl_cost_tables.py reads them
                  from the zip in place ("so 400 shard parquets are not duplicated onto
                  disk"). The --psi-zip path to hand it is printed at the end.

    --kind cf     foundation/cf1/cf3/cf4/cf5/cf6_outputs[_tag].zip
                  -> CF_OUTPUT/cluster_raw/  the archive itself (copied)
                  -> CF_FOUNDATION/          everything under the archive's output/cf/,
                                             keeping any sub-directory (the sharded
                                             cf3_jacobi_E*/cf5_E*/cf6_E* folders) so
                                             export_cf1_franchise.py and friends find the
                                             flat top-level parquets exactly where they
                                             look for them.

A source zip is only ever READ or COPIED, never moved out of a download folder and never
deleted -- these archives are the only copy of some cluster output (the psi shards have no
second copy anywhere), and this script is re-runnable precisely because it leaves them be.

Usage:
    python process_cluster_outputs.py --kind bbl
    python process_cluster_outputs.py --kind cf
    python process_cluster_outputs.py --kind cf --cf cf1 --zip D:/downloads/cf1_outputs.zip
    python process_cluster_outputs.py --kind bbl --dry-run
"""
from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils import paths                                   # noqa: E402
import process_blp_outputs as blp                         # noqa: E402  (move_into reused)

EST_OUT = paths.estimation_output()

# Archive families, newest of each is processed unless --zip/--cf pins one. The names are
# fixed by the producers: submit_bbl_all.sh writes bbl_outputs_<jobid>.zip, and
# zip_cf_outputs.sh writes <cf>_outputs[_<tag>].zip with member paths relative to data/.
BBL_FAMILIES = ("bbl_outputs", "psi_cost", "cf2_outputs")
CF_FAMILIES = ("foundation_outputs", "cf1_outputs", "cf3_outputs",
               "cf4_outputs", "cf5_outputs", "cf6_outputs")

# Member prefixes inside the archives, as written by zip_cf_outputs.sh (paths relative to
# data/). bbl_outputs_*.zip is flat (zip -j), so a bare basename counts as output/cost.
CF_PREFIX = "output/cf/"
COST_PREFIX = "output/cost/"


def tree_for(kind):
    """-> (raw_dir, processed_dir, index_path). processed_dir is where the artifacts the
    table generators read end up; INDEX.json sits beside them, as in process_blp_outputs."""
    if kind == "bbl":
        root = paths.bbl_output_dir()
        proc = root / "cluster_processed"
        return root / "cluster_raw", proc, proc / "INDEX.json"
    root = paths.cf_foundation_dir()
    return EST_OUT / "CF_OUTPUT" / "cluster_raw", root, root / "INDEX.json"


def search_dirs(kind, extra):
    raw, _, _ = tree_for(kind)
    cands = [Path(d) for d in extra]
    cands += [raw, EST_OUT, paths.bbl_output_dir(), paths.cf_foundation_dir(),
              EST_OUT / "CF_OUTPUT" / "cluster_raw",
              Path.home() / "Downloads", Path.home() / "Desktop"]
    seen, out = set(), []
    for d in cands:
        if d.is_dir() and str(d).lower() not in seen:
            seen.add(str(d).lower())
            out.append(d)
    return out


def _job_id(name):
    """Trailing _<digits> before .zip, as submit_bbl_all.sh stamps it. -1 when absent, so a
    family whose archives carry no job id falls back to mtime ordering."""
    stem = name[:-4] if name.lower().endswith(".zip") else name
    tail = stem.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else -1


def newest(paths_list):
    """Highest job id wins; among archives with no job id, newest mtime wins. Mirrors
    pick_zip() in submit_cf_all.sh / submit_bbl_all.sh."""
    if not paths_list:
        return None
    return max(paths_list, key=lambda p: (_job_id(p.name), p.stat().st_mtime))


def discover(kind, families, dirs):
    """-> {family: Path}. One archive per family: the newest found anywhere we looked."""
    found = {}
    for fam in families:
        hits = []
        for d in dirs:
            hits += [Path(p) for p in glob.glob(str(d / f"{fam}*.zip"))]
        pick = newest(hits)
        if pick is not None:
            found[fam] = pick
    return found


def fmt_size(n):
    for unit, div in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if n >= div:
            return f"{n / div:.1f} {unit}"
    return f"{n} B"


def sha256_file(path: Path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def _safe_rel(member: str):
    """Reject absolute members and any '..' traversal before writing anything to disk."""
    rel = member.replace("\\", "/").lstrip("/")
    parts = [p for p in rel.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts) or (len(member) > 1 and member[1] == ":"):
        return None
    return "/".join(parts)


def classify(kind, member):
    """-> (dest_rel | None, note). dest_rel is relative to the kind's processed dir."""
    rel = _safe_rel(member)
    if rel is None or rel.endswith("/"):
        return None, "unsafe member name" if rel is None else None
    base = os.path.basename(rel)
    if kind == "bbl":
        # cost_params are the artifact make_bbl_cost_tables.py reads out of
        # cluster_processed; psi_eq / psi_dev shards stay archived.
        if base.startswith("cost_params_") and base.endswith(".json"):
            return base, None
        if base.startswith("psi_eq_") or base.startswith("psi_dev_"):
            return None, "psi (left in the archive; pass it with --psi-zip)"
        return None, "not a BBL artifact"
    if rel.startswith(CF_PREFIX):
        return rel[len(CF_PREFIX):], None
    if rel.startswith(COST_PREFIX) or base.startswith("cost_params_"):
        return None, "cost artifact -- ingest this archive with --kind bbl instead"
    return None, "not a CF artifact"


def load_index(path: Path):
    if path.is_file():
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            print(f"  [index] {path.name} unreadable -- starting a fresh one")
    return {}


def ingest(kind, family, zip_path: Path, raw_dir: Path, proc_dir: Path, index, dry):
    """Copy the archive into cluster_raw/, extract its artifacts into proc_dir.
    -> (record, [(dest_rel, size)])."""
    zmtime = zip_path.stat().st_mtime
    zstamp = datetime.datetime.fromtimestamp(zmtime).isoformat(timespec="seconds")
    print(f"\n-- {family}: {zip_path.name} "
          f"({fmt_size(zip_path.stat().st_size)}, "
          f"{datetime.datetime.fromtimestamp(zmtime):%Y-%m-%d %H:%M})")
    print(f"   source: {zip_path.parent}")

    landed, skipped = [], {}
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.namelist() if not m.endswith("/")]
        for m in members:
            dest_rel, note = classify(kind, m)
            if dest_rel is None:
                skipped[note or "unknown"] = skipped.get(note or "unknown", 0) + 1
                continue
            # Several archives carry the same cost_params_E{k}_spec_{s}_{stage}.json -- the
            # solve writes them into data/output/cost, and both bbl_outputs_<jobid>.zip and
            # the psi/cf2 archive pick them up. The copies are NOT interchangeable: the ones
            # inside an older psi archive predate the c_bar/ridge fields the solve now
            # persists, and make_bbl_cost_tables.py reads those fields. So a file already
            # attributed to a NEWER archive is never overwritten by an older one.
            prev = index.get("files", {}).get(dest_rel)
            if prev and prev.get("zip_mtime", "") > zstamp:
                skipped[f"kept the copy from {prev['zip']} (newer archive)"] = \
                    skipped.get(f"kept the copy from {prev['zip']} (newer archive)", 0) + 1
                continue
            info = z.getinfo(m)
            if not dry:
                out = proc_dir / dest_rel
                out.parent.mkdir(parents=True, exist_ok=True)
                with z.open(m) as src, open(out, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            landed.append((dest_rel, info.file_size))

    # The archive goes to cluster_raw/ so the tree is self-contained. A zip found loose in
    # the destination tree is relocated (blp.move_into, the same helper process_blp_outputs
    # uses for that case); one that came from anywhere else is COPIED and the original left
    # exactly where it was.
    raw_copy = raw_dir / zip_path.name
    if not dry:
        raw_dir.mkdir(parents=True, exist_ok=True)
        if zip_path.resolve() != raw_copy.resolve():
            if zip_path.parent.resolve() in (proc_dir.resolve(), raw_dir.parent.resolve()):
                blp.move_into(str(zip_path), str(raw_dir))
                print(f"   archive: moved into {raw_dir.name}/ (it was loose in the tree)")
            else:
                shutil.copy2(zip_path, raw_copy)
                print(f"   archive: copied into {raw_dir.name}/ (original left in place)")
        else:
            print(f"   archive: already in {raw_dir.name}/")

    for note, n in sorted(skipped.items()):
        print(f"   {n:>4} member(s) skipped -- {note}")
    print(f"   {len(landed)} file(s) -> {proc_dir}")
    for dest_rel, size in sorted(landed)[:40]:
        print(f"        {fmt_size(size):>9}  {dest_rel}")
    if len(landed) > 40:
        print(f"        ... and {len(landed) - 40} more")

    record = {
        "family": family,
        "zip": zip_path.name,
        "zip_source": str(zip_path.parent),
        "zip_size": zip_path.stat().st_size,
        "zip_mtime": zstamp,
        "zip_sha256": None if dry else sha256_file(raw_copy if raw_copy.is_file() else zip_path),
        "ingested": datetime.datetime.now().isoformat(timespec="seconds"),
        "n_files": len(landed),
        "files": sorted(d for d, _ in landed),
    }
    return record, landed


def main():
    ap = argparse.ArgumentParser(
        description="Unpack the BBL / CF cluster archives into the local tree.")
    ap.add_argument("--kind", required=True, choices=("bbl", "cf"))
    ap.add_argument("--zip", default=None, help="pin one archive instead of discovering it")
    ap.add_argument("--cf", default=None,
                    help="restrict to one CF family: foundation, cf1, cf3, cf4, cf5, cf6")
    ap.add_argument("--search-dir", action="append", default=[],
                    help="extra directory to look for archives in (repeatable)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would land where; write nothing")
    args = ap.parse_args()

    raw_dir, proc_dir, index_path = tree_for(args.kind)
    print("=" * 88)
    print(f"process_cluster_outputs -- kind={args.kind}"
          f"{'  (DRY RUN, nothing written)' if args.dry_run else ''}")
    print("=" * 88)
    print(f"  archives -> {raw_dir}")
    print(f"  artifacts-> {proc_dir}")
    print(f"  index    -> {index_path}")

    if args.zip:
        zp = Path(args.zip).expanduser()
        if not zp.is_file():
            sys.exit(f"ERROR: --zip not found: {zp}")
        fam = next((f for f in (BBL_FAMILIES + CF_FAMILIES) if zp.name.startswith(f)),
                   zp.stem)
        picks = {fam: zp}
    else:
        families = BBL_FAMILIES if args.kind == "bbl" else CF_FAMILIES
        if args.cf:
            want = args.cf if args.cf.endswith("_outputs") else f"{args.cf}_outputs"
            families = tuple(f for f in families if f == want)
            if not families:
                sys.exit(f"ERROR: --cf {args.cf} is not one of "
                         f"{', '.join(f[:-8] for f in CF_FAMILIES)}")
        dirs = search_dirs(args.kind, args.search_dir)
        print(f"  searched : {len(dirs)} directories")
        picks = discover(args.kind, families, dirs)
        if not picks:
            print()
            print(f"No {args.kind.upper()} archive found. Looked for "
                  f"{', '.join(f + '*.zip' for f in families)} in:")
            for d in dirs:
                print(f"    {d}")
            print("Download the archive from the cluster (it is written to "
                  "data/output/ there), or pass --zip.")
            return 1

    if not args.dry_run:
        raw_dir.mkdir(parents=True, exist_ok=True)
        proc_dir.mkdir(parents=True, exist_ok=True)

    index = load_index(index_path)
    index.setdefault("kind", args.kind)
    index.setdefault("ingests", [])
    index.setdefault("files", {})

    # Oldest archive first, so that where two archives carry the same artifact the newest
    # one is written last and wins (the per-file guard in ingest() enforces the same rule
    # across separate runs, using the provenance recorded in INDEX.json).
    total = 0
    for fam in sorted(picks, key=lambda f: picks[f].stat().st_mtime):
        record, landed = ingest(args.kind, fam, picks[fam], raw_dir, proc_dir,
                                index, args.dry_run)
        total += len(landed)
        index["ingests"] = [r for r in index["ingests"]
                            if not (r.get("family") == fam and r.get("zip") == record["zip"])]
        index["ingests"].append(record)
        for dest_rel, size in landed:
            index["files"][dest_rel] = {"from_zip": record["zip"], "family": fam,
                                        "size": size, "ingested": record["ingested"],
                                        "zip_mtime": record["zip_mtime"]}
    index["updated"] = datetime.datetime.now().isoformat(timespec="seconds")

    if not args.dry_run:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)

    print()
    print("=" * 88)
    verb = "would land" if args.dry_run else "landed"
    print(f"{total} file(s) {verb} from {len(picks)} archive(s); "
          f"{len(index['files'])} file(s) tracked in {index_path.name}")
    if args.kind == "bbl":
        psi = index["ingests"]
        psi = [r for r in psi if r["family"] in ("psi_cost", "cf2_outputs")]
        if psi:
            print()
            print("Next:  python make_bbl_cost_tables.py"
                  "                       (cost tables, from cost_params)")
            print(f"       python make_bbl_cost_tables.py --from-psi --psi-zip "
                  f"{raw_dir / psi[-1]['zip']}")
            print("                                                        (ridge diagnostic)")
        else:
            print()
            print("Next:  python make_bbl_cost_tables.py")
            print("       The ridge diagnostic also needs the psi archive (psi_cost.zip or "
                  "cf2_outputs.zip);")
            print("       it is produced on the cluster by "
                  "`KEEP=1 CFS=\"cf2\" bash zip_all_cf.sh` -- download it and re-run "
                  "with --kind bbl.")
    else:
        print()
        print("Next:  python export_cf1_franchise.py --estim <k> --spec 12 --stage extended")
    print("Source archives were left where they were found; nothing was deleted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

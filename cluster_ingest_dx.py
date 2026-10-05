"""
cluster_ingest_dx.py
====================
Local ingest of the `dx` demand variant (the D-type dummy in the linear part; blp_dx.jl,
dx_suite_20261001.sh): takes the variant's members out of the downloaded cluster archives and
lands them in the variant's own folder, ESTIMATION_OUTPUT/DX_VARIANT.

    <dx>/blp/   blp_results_E{k}_spec_12_{stage}_dx.{jls,json}   the RC ladder
                blp_summary_E{k}_{stage}_gpu_ift_dx.json
                blp_meta_E{k}_spec_12_{stage}_dx.json            SE guard verdict per stage
                blp_linear_E{k}_spec_12_dx.json                  the theta2 = 0 nesting check
                blp_compare_E{k}_spec_12_dx.json                 main vs dx (alpha, theta2, delta)
                blp_bblfit_E{k}_spec_12_extended_dx_ms<S>.json   sha256 of the fit a BBL run was
                                                                 simulated on (make_dx_tables.py
                                                                 checks the fit on disk against it)
    <dx>/bbl/   cost_params_E{k}_spec_12_extended_dx_ms<S>.json
                psi_eq_/psi_starts_E{k}_spec_12_extended_dx_ms<S>.*   (psi_dev shards: --with-psi-dev)

A member is taken only when its NAME is a variant name (the patterns below); every other member of
the archive is left alone and counted. Nothing is written outside <dx>: the main ingests
(cluster_ingest_blp.py, cluster_ingest_bbl_cf.py) own BLP_RESULTS, BBL_OUTPUT and COST_FWD, and
this script refuses a destination inside any of them. An existing file with other content is kept
unless --force is given.

WARNING: THE VARIANT'S ARCHIVES CARRY THE MAIN FAMILY'S NAMES.
cluster_archive.sh names every archive <set>_outputs_<tag>.zip and takes no other prefix, so the
variant's archives are bbl_outputs_<jobid>_dx.zip and blp_outputs_<jobid>_dx.zip (the suite tags
them <jobid>_dx). The main ingests find archives by that prefix:
  * cluster_ingest_bbl_cf.py --kind bbl, and cluster_ingest.py, prefer the archive with the highest
    NUMERIC job id; the tag <jobid>_dx is not a number, so they pick a variant archive only when no
    main archive with a job id is in the folders they scan. If one did, it would land the
    variant's cost_params_*_dx_ms982.json next to the main ones in BBL_OUTPUT/cluster_processed
    (other names, nothing overwritten, but the variant's files do not belong there).
  * cluster_ingest_blp.py takes the NEWEST blp_outputs*.zip by modification time in BLP_RESULTS or
    cluster_raw. A variant BLP archive holds the whole blp step folder, main results included; fed
    to that ingest it refreshes the main results from the same cluster folder and files the `_dx`
    members under their own names in cluster_raw/, using none of them.
The safe routine: keep the variant's archives in a folder of their own (not BLP_RESULTS, not the
folder the main ingests scan) and always name the archive explicitly.

  the variant:   python cluster_ingest_dx.py --zip <dir>/blp_outputs_<jobid>_dx.zip --zip <dir>/bbl_outputs_<jobid>_dx.zip
  the main BBL:  python cluster_ingest_bbl_cf.py --kind bbl --zip <dir>/bbl_outputs_<main jobid>.zip
  the main BLP:  python cluster_ingest_blp.py      (with only the main blp_outputs_<jobid>.zip in BLP_RESULTS)

Usage
-----
  python cluster_ingest_dx.py --zip blp_outputs_123_dx.zip --zip bbl_outputs_456_dx.zip
  python cluster_ingest_dx.py --zip <archive> --dry-run
  python cluster_ingest_dx.py --zip <archive> --dx-dir <folder>          # a sandbox
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import argparse
import hashlib
import os
import pathlib
import re
import sys
import zipfile

from utils import paths as _paths

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DX_SUFFIX = "_dx"
DX_DIR_DEFAULT = _paths.estimation_output() / "DX_VARIANT"

# Variant member names -> subfolder. Anchored on the whole basename.
BLP_RE = re.compile(
    r"^blp_(?:results|meta)_E\d+_spec_\d+_[A-Za-z0-9]+_dx\.(?:jls|json)$"
    r"|^blp_summary_E\d+_[A-Za-z0-9]+_gpu_ift_dx\.json$"
    r"|^blp_(?:linear|compare)_E\d+_spec_\d+_dx\.json$"
    r"|^blp_bblfit_E\d+_spec_\d+_[A-Za-z0-9]+_dx_[A-Za-z0-9]+\.json$")
BBL_RE = re.compile(
    r"^cost_params_E\d+_spec_\d+_[A-Za-z0-9]+_dx_[A-Za-z0-9]+\.json$"
    r"|^psi_(?:eq|starts)_E\d+_spec_\d+_[A-Za-z0-9]+_dx_[A-Za-z0-9]+\.(?:parquet|json)$")
PSI_DEV_RE = re.compile(
    r"^psi_dev_E\d+_spec_\d+_[A-Za-z0-9]+_dx_[A-Za-z0-9]+(?:_shard\d+of\d+)?\.parquet$")


def classify(name: str, with_psi_dev: bool = False):
    """'blp' | 'bbl' | None for an archive member (by basename)."""
    b = name.replace("\\", "/").rsplit("/", 1)[-1]
    if BLP_RE.match(b):
        return "blp"
    if BBL_RE.match(b) or (with_psi_dev and PSI_DEV_RE.match(b)):
        return "bbl"
    return None


def protected_dirs():
    """Folders the main ingests and generators own; the variant never lands inside one."""
    return [_paths.blp_results_dir(), _paths.bbl_output_dir(), _paths.cost_fwd_dir(),
            _paths.estimation_output() / "Rout", _paths.drafts_dir(),
            _paths.estimation_output() / "COST_POLFUNC", _paths.demand_parquet_dir()]


def check_dx_dir(dx_dir: pathlib.Path):
    r = dx_dir.resolve()
    for p in protected_dirs():
        pr = pathlib.Path(p).resolve()
        if r == pr or pr in r.parents:
            raise SystemExit(f"[ingest-dx] REFUSING: the destination {r} is inside {pr}, a folder of the "
                             "main specification. The variant has its own folder.")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def ingest_zip(zpath: pathlib.Path, dx_dir: pathlib.Path, *, with_psi_dev=False, force=False, dry=False):
    counts = {"landed": 0, "same": 0, "kept": 0, "updated": 0, "other": 0}
    with zipfile.ZipFile(zpath) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            kind = classify(info.filename, with_psi_dev)
            if kind is None:
                counts["other"] += 1
                continue
            base = info.filename.replace("\\", "/").rsplit("/", 1)[-1]
            dest = dx_dir / kind / base
            data = zf.read(info)
            if dest.exists():
                if _sha(dest.read_bytes()) == _sha(data):
                    counts["same"] += 1
                    continue
                if not force:
                    counts["kept"] += 1
                    print(f"  KEPT (differs from the archive; --force replaces): {dest.name}")
                    continue
                tag = "updated"
            else:
                tag = "landed"
            counts[tag] += 1
            print(f"  {'[dry-run] would have ' if dry else ''}{tag}: {kind}/{base}")
            if dry:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, dest)
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser(description="Ingest the dx demand variant's cluster outputs.")
    ap.add_argument("--zip", action="append", default=[], required=True,
                    help="a downloaded archive (blp_outputs_*.zip, bbl_outputs_*.zip); repeatable")
    ap.add_argument("--dx-dir", default=str(DX_DIR_DEFAULT))
    ap.add_argument("--with-psi-dev", action="store_true", help="also land the psi_dev shards")
    ap.add_argument("--force", action="store_true", help="replace an existing file that differs")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    dx_dir = pathlib.Path(a.dx_dir)
    check_dx_dir(dx_dir)
    total = {}
    for z in a.zip:
        zp = pathlib.Path(z)
        if not zp.is_file():
            raise SystemExit(f"[ingest-dx] no such archive: {zp}")
        print(f"[ingest-dx] {zp.name} -> {dx_dir}")
        c = ingest_zip(zp, dx_dir, with_psi_dev=a.with_psi_dev, force=a.force, dry=a.dry_run)
        print(f"  {c['landed']} landed, {c['updated']} updated, {c['same']} already identical, "
              f"{c['kept']} kept (differ), {c['other']} member(s) of other specifications left alone")
        for k, v in c.items():
            total[k] = total.get(k, 0) + v
    if total.get("landed", 0) + total.get("updated", 0) + total.get("same", 0) + total.get("kept", 0) == 0:
        print("[ingest-dx] no variant member in the archive(s).")
        return 1
    print("[ingest-dx] next:  python make_dx_tables.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

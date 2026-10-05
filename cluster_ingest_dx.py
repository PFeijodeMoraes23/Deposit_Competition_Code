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
                                                                 simulated on, and the job ids of
                                                                 its BBL archives
    <dx>/bbl/   cost_params_E{k}_spec_12_extended_dx_ms<S>.json
                cost_params_E{k}_spec_12_extended_dx_ms<S>.ingest.json   written HERE, not on the
                                                                 cluster: the archive each
                                                                 cost_params file came from
                psi_eq_/psi_starts_E{k}_spec_12_extended_dx_ms<S>.*   (psi_dev shards: --with-psi-dev)

A member is taken only when its NAME is a variant name (the patterns below); every other member of
the archive is left alone and counted. Nothing is written outside <dx>: the main ingests
(cluster_ingest_blp.py, cluster_ingest_bbl_cf.py) own BLP_RESULTS, BBL_OUTPUT and COST_FWD, and
this script refuses a destination inside any of them. An existing file with other content is kept
unless --force is given.

THE DEMAND FIT, ITS RECORD AND THE COST PARAMS MOVE TOGETHER OR NOT AT ALL.
The cost params of a BBL run mean something only beside the demand fit they were simulated on. The
suite writes, at the BBL launch, blp_bblfit_<key>.json: the sha256 of that fit, and the job id of
every archive job it sequenced for the run, which is the number in the BBL archive's name
(bbl_outputs_<jobid>_dx.zip). Nothing inside a BBL archive says which launch it comes from, so
that number is the tie. The whole call is planned before a file is written, and it is REFUSED,
with nothing landed, when it would
  * land or update cost params while the archives' fit record or extended fit of that run differs
    from the file on disk and would be kept (no --force);
  * leave on disk a fit that is not the one its record names;
  * leave on disk cost params from a BBL archive whose job id the record does not list.
So give the BLP and the BBL archive of one run in the same call, under the names they were
downloaded with. Where the tie cannot be told (no record, a record without archive jobs, a renamed
archive) the ingest goes on and says so; make_dx_tables.py then reports the cost params as not
verifiable instead of bound. Beside each cost_params file it lands, the ingest writes
cost_params_<key>.ingest.json (archive name, its job id, sha256): the tables read it.

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
import datetime
import hashlib
import json
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


def _sha_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _json(b):
    try:
        d = json.loads(b)
        return d if isinstance(d, dict) else None
    except (TypeError, ValueError):
        return None


# ── the pairing: demand fit, fit record and cost params of one BBL run ───────────────────────
# <key> = E{k}_spec_{s}_{stage}_dx<tag>, as in cost_params_<key>.json and blp_bblfit_<key>.json.
COST_RE = re.compile(r"^cost_params_(E\d+_spec_\d+_[A-Za-z0-9]+_dx_[A-Za-z0-9]+)\.json$")
RECORD_RE = re.compile(r"^blp_bblfit_(E\d+_spec_\d+_[A-Za-z0-9]+_dx_[A-Za-z0-9]+)\.json$")
KEY_RE = re.compile(r"^(E\d+_spec_\d+_[A-Za-z0-9]+)_dx_[A-Za-z0-9]+$")
ARCHIVE_JOB_RE = re.compile(r"^bbl_outputs_(\d+)_dx\.zip$")


def record_name(key: str) -> str:
    return f"blp_bblfit_{key}.json"


def fit_names(key: str):
    """The demand fit of a BBL key: the `.jls` the forward simulation reads and the engine's `.json`."""
    base = KEY_RE.match(key).group(1)
    return f"blp_results_{base}{DX_SUFFIX}.jls", f"blp_results_{base}{DX_SUFFIX}.json"


def source_name(key: str) -> str:
    """The provenance file this ingest writes beside a cost_params file it landed."""
    return f"cost_params_{key}.ingest.json"


def archive_job(zip_name: str):
    """The job id in a BBL archive's name, bbl_outputs_<jobid>_dx.zip, or None."""
    m = ARCHIVE_JOB_RE.match(pathlib.Path(zip_name).name)
    return m.group(1) if m else None


def plan_ingest(zips, dx_dir: pathlib.Path, *, with_psi_dev=False, force=False):
    """What ingesting `zips` in order would do, without writing: a list of entries
    {zip, member, kind, name, dest, sha, state, data?} with state landed | updated | same | kept,
    and per archive the count of members left alone. `data` is kept for the fit records only."""
    entries, others, after = [], {}, {}
    for zp in zips:
        others[zp] = 0
        with zipfile.ZipFile(zp) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                kind = classify(info.filename, with_psi_dev)
                if kind is None:
                    others[zp] += 1
                    continue
                name = info.filename.replace("\\", "/").rsplit("/", 1)[-1]
                dest = dx_dir / kind / name
                data = zf.read(info)
                sha = _sha(data)
                have = after.get(dest, _sha_file(dest) if dest.exists() else None)
                if have is None:
                    state = "landed"
                elif have == sha:
                    state = "same"
                else:
                    state = "updated" if force else "kept"
                if state in ("landed", "updated"):
                    after[dest] = sha
                entries.append({"zip": zp, "member": info.filename, "kind": kind, "name": name, "dest": dest,
                                "sha": sha, "state": state, "data": data if RECORD_RE.match(name) else None})
    return entries, others


def check_pairing(entries, dx_dir: pathlib.Path):
    """The rule of the ingest: the demand fit, the fit record and the cost params of a BBL run move
    together or not at all. -> (refusals, notes), both lists of sentences; any refusal stops the
    ingest before a file is written.

    For every BBL key of which this call changes something (lands or updates its cost params, its
    record, or its extended fit):
      1. cost params land or are updated while the archives' record or fit of that key differs
         from the one on disk and is kept (no --force): refused;
      2. the fit that would be on disk afterwards is not the one its record names: refused;
      3. the cost params that would be on disk afterwards come from a BBL archive whose job id the
         record does not list: refused.
    Where 2 or 3 cannot be told (no record, a record without archive jobs, an archive whose name
    carries no job id) the ingest goes on and says the pairing is not verifiable."""
    by_name = {}
    for e in entries:
        by_name.setdefault(e["name"], []).append(e)

    def last(name, states):
        hits = [e for e in by_name.get(name, []) if e["state"] in states]
        return hits[-1] if hits else None

    keys = set()
    for e in entries:
        if e["state"] not in ("landed", "updated"):
            continue
        for rx in (COST_RE, RECORD_RE):
            m = rx.match(e["name"])
            if m:
                keys.add(m.group(1))
    for e in entries:      # a changed extended fit touches every key built on it that is on disk or arriving
        if e["state"] in ("landed", "updated") and e["kind"] == "blp":
            cands = {m.group(1) for n in by_name for m in [COST_RE.match(n)] if m}
            cands |= {m.group(1) for p in (dx_dir / "bbl").glob("cost_params_*.json") for m in [COST_RE.match(p.name)] if m}
            keys |= {k for k in cands if e["name"] in fit_names(k)}
    refusals, notes = [], []
    for key in sorted(keys):
        cost_n, rec_n, (fit_n, fitj_n) = f"cost_params_{key}.json", record_name(key), fit_names(key)
        cost_new = last(cost_n, ("landed", "updated"))
        kept = [n for n in (rec_n, fit_n, fitj_n) if last(n, ("kept",))]
        if cost_new and kept:
            refusals.append(
                f"{key}: its cost params would be {cost_new['state']} from {cost_new['zip'].name}, while "
                f"{', '.join(kept)} in the archives differ(s) from the file on disk and would be kept. The demand "
                f"fit, its record and the cost params move together: pass --force to replace all of them, or "
                f"ingest into an empty folder")
            continue
        # the state after this call
        rec_e = last(rec_n, ("landed", "updated", "same"))
        rec_p = dx_dir / "blp" / rec_n
        rec = _json(rec_e["data"]) if rec_e else (_json(rec_p.read_bytes()) if rec_p.is_file() else None)
        fit_e = last(fit_n, ("landed", "updated", "same"))
        fit_p = dx_dir / "blp" / fit_n
        fit_sha = fit_e["sha"] if fit_e else (_sha_file(fit_p) if fit_p.is_file() else None)
        cost_e = last(cost_n, ("landed", "updated", "same"))
        cost_p = dx_dir / "bbl" / cost_n
        if cost_e:
            job, where = archive_job(cost_e["zip"].name), cost_e["zip"].name
        elif cost_p.is_file():
            src = _json((dx_dir / "bbl" / source_name(key)).read_bytes()) if (dx_dir / "bbl" / source_name(key)).is_file() else None
            ok_src = bool(src) and src.get("sha256") == _sha_file(cost_p)
            job, where = (src.get("archive_job"), src.get("archive")) if ok_src else (None, "the file on disk")
        else:
            continue        # no cost params of this key on disk after the call: nothing is paired yet
        if rec and rec.get("fit_sha256") and fit_sha and rec["fit_sha256"] != fit_sha:
            refusals.append(
                f"{key}: after this ingest {fit_n} would have sha256 {fit_sha[:12]}..., but {rec_n} says its BBL was "
                f"simulated on {str(rec['fit_sha256'])[:12]}...: the fit and the record belong to different runs. "
                f"Ingest the BLP archive that goes with this record (with --force), or first remove {cost_n} and "
                f"{rec_n} of the earlier run from {dx_dir}")
            continue
        jobs = str((rec or {}).get("bbl_archive_jobs") or "").split()
        if rec and jobs and job and str(job) not in jobs:
            refusals.append(
                f"{key}: the cost params ({where}) come from the BBL archive of job {job}, but {rec_n} lists the "
                f"archive job(s) {', '.join(jobs)} for the fit it records: another launch. Ingest the BLP and the "
                f"BBL archive of the same run in one call (with --force), or first remove {cost_n} and "
                f"{source_name(key)} from {dx_dir / 'bbl'}")
            continue
        if not rec:
            notes.append(f"{key}: no fit record {rec_n}: the pairing of the cost params with the demand fit is not verifiable")
        elif not jobs:
            notes.append(f"{key}: {rec_n} lists no BBL archive job: the cost params are not verifiable against the record")
        elif not job:
            notes.append(f"{key}: {where} carries no job id (expected bbl_outputs_<jobid>_dx.zip): the cost params are "
                         f"not verifiable against the record")
    return refusals, notes


def write_sources(entries, dry=False):
    """Beside every cost_params file this call landed, updated or found identical: where it came
    from (archive name, the job id in it, sha256). make_dx_tables.py reads it to tie the cost
    params to the fit record. Not rewritten when it already says the same."""
    for e in entries:
        m = COST_RE.match(e["name"])
        if not m or e["state"] == "kept":
            continue
        p = e["dest"].with_name(source_name(m.group(1)))
        old = _json(p.read_bytes()) if p.is_file() else None
        new = {"cost_params": e["name"], "sha256": e["sha"], "archive": e["zip"].name,
               "archive_job": archive_job(e["zip"].name)}
        if old and all(old.get(k) == v for k, v in new.items()):
            continue
        if dry:
            continue
        new["ingested"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(new, indent=1), encoding="utf-8")
        os.replace(tmp, p)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Ingest the dx demand variant's cluster outputs.")
    ap.add_argument("--zip", action="append", default=[], required=True,
                    help="a downloaded archive (blp_outputs_*.zip, bbl_outputs_*.zip); repeatable. Give the "
                         "BLP and the BBL archive of one run in the same call")
    ap.add_argument("--dx-dir", default=str(DX_DIR_DEFAULT))
    ap.add_argument("--with-psi-dev", action="store_true", help="also land the psi_dev shards")
    ap.add_argument("--force", action="store_true", help="replace an existing file that differs")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    dx_dir = pathlib.Path(a.dx_dir)
    check_dx_dir(dx_dir)
    zips = [pathlib.Path(z) for z in a.zip]
    for zp in zips:
        if not zp.is_file():
            raise SystemExit(f"[ingest-dx] no such archive: {zp}")
    entries, others = plan_ingest(zips, dx_dir, with_psi_dev=a.with_psi_dev, force=a.force)
    refusals, notes = check_pairing(entries, dx_dir)
    if refusals:
        raise SystemExit("[ingest-dx] REFUSING, nothing was landed:\n  " + "\n  ".join(refusals))
    for zp in zips:
        mine = [e for e in entries if e["zip"] == zp]
        print(f"[ingest-dx] {zp.name} -> {dx_dir}")
        with zipfile.ZipFile(zp) as zf:
            for e in mine:
                if e["state"] == "kept":
                    print(f"  KEPT (differs from the archive; --force replaces): {e['name']}")
                if e["state"] not in ("landed", "updated"):
                    continue
                print(f"  {'[dry-run] would have ' if a.dry_run else ''}{e['state']}: {e['kind']}/{e['name']}")
                if a.dry_run:
                    continue
                e["dest"].parent.mkdir(parents=True, exist_ok=True)
                tmp = e["dest"].with_name(e["dest"].name + ".tmp")
                tmp.write_bytes(zf.read(e["member"]))
                os.replace(tmp, e["dest"])
        c = {s: sum(1 for e in mine if e["state"] == s) for s in ("landed", "updated", "same", "kept")}
        print(f"  {c['landed']} landed, {c['updated']} updated, {c['same']} already identical, "
              f"{c['kept']} kept (differ), {others[zp]} member(s) of other specifications left alone")
    write_sources(entries, dry=a.dry_run)
    for n in notes:
        print(f"  [pairing] {n}")
    if not entries:
        print("[ingest-dx] no variant member in the archive(s).")
        return 1
    print("[ingest-dx] next:  python make_dx_tables.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
bbl_shards.py -- the ONE implementation of "which psi_dev shards does this BBL run have?"
====================================================================================

bbl_fwd_sim.jl writes one `psi_dev_<key>_shard{i}of{N}.parquet` per shard, where
key = E{k}_spec_{s}_{stage}{suffix}{psi_tag}, and shard 0 also writes `psi_eq_<key>.parquet`
(and, under --multi-start, `psi_starts_<key>.json`). Two consumers need to know which of the N
indices are really there:

  bbl_solve.py        refuses to solve on a partial set (it imports `shard_coverage` as its
                      `_shard_coverage`, so the solve and the sweep can never disagree about
                      what "complete" means);
  bbl_job.sh sweep    runs `python bbl_shards.py coverage ...` after every fwd_sim array and
                      resubmits EXACTLY the indices it reports missing.

"Present" is stricter for the sweep than for the solve's name check. With --validate a file only
counts when it is a readable parquet with the expected columns: two jobs writing the same shard,
or a job killed mid-write, used to be able to leave a truncated file under the final name, and a
file-exists check passes it. (bbl_fwd_sim.jl now writes through a temporary name and renames, so
a truncated FINAL file should no longer be possible -- the validation is the second lock.) With
--newer-than a file older than the run is treated as stale: a fresh run that reuses a tag must
never count a shard left over from an earlier design as its own.

STDLIB ONLY at import time -- no numpy, no pandas -- so it runs in the sweep's small job and in
the local tests alike. pyarrow is used by --validate when it is importable; without it the check
falls back to the parquet magic bytes and footer length, which is still enough to catch a
truncated file.

It also carries the Python reader of bbl_discount.env (`read_bbl_discount`), the registry of the
BBL discount factor and forward-simulation horizon, for the Python consumers (bbl_solve.py's
provenance note, diag_cf1_franchise_dataonly.py).

CLI
  python bbl_shards.py coverage --dir DIR --key KEY --n-shards N [--validate] [--newer-than EPOCH]
                                [--expect-starts | --expect-sidecar] [--expect-beta B] [--expect-horizon T]
                                [--expect-phi-path P] [--expect-z-path Z] [--expect-rdep-timing R]
                                [--expect-sim-version V] [--policy-csv CSV]
      exit 0 complete, 1 gaps (MISSING_* lines name them), 2 a problem no re-run can fix
  python bbl_shards.py compress 3 50 51 52          -> 3,50-52
  python bbl_shards.py expand 3,50-52               -> 3 50 51 52
  python bbl_shards.py pack --k 4 --spec 3,50-60    -> one line per task
  python bbl_shards.py discount                      -> BBL_BETA=... / BBL_HORIZON=...
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

SHARD_RE = re.compile(r"_shard(\d+)of(\d+)$")
PARQUET_MAGIC = b"PAR1"
# The psi block layout bbl_fwd_sim.jl writes (psi3_gamma_* vary with the cost shifters, so they
# are not required by name). start_q is added under --multi-start.
DEV_COLS = ("shock", "firm", "is_B", "psi1", "psi2_omega", "psi4_zeta")
EQ_COLS = ("firm", "is_B", "psi1", "psi2_omega", "psi4_zeta")


# ==========================================================================
# The discount registry: bbl_discount.env
# ==========================================================================
def read_bbl_discount(path=None):
    """{'BBL_BETA': float, 'BBL_HORIZON': int} from bbl_discount.env.

    The file sits beside this module (the scripts folder on the cluster); BBL_DISCOUNT_ENV names
    another. Plain KEY=VALUE lines with '#' comments -- the same file bash reads with
    cl_bbl_discount_get and Julia with bbl_discount(). A missing file or key raises: a fallback
    literal is how a discount factor nobody chose reaches a run.
    """
    p = Path(path or os.environ.get("BBL_DISCOUNT_ENV")
             or Path(__file__).resolve().parent / "bbl_discount.env")
    if not p.is_file():
        raise FileNotFoundError(f"BBL discount registry not found: {p} (it sets BBL_BETA and "
                                f"BBL_HORIZON; pass --beta/--horizon explicitly or restore it)")
    vals = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        v = v.split("#", 1)[0].strip()
        if v:
            vals[k.strip()] = v
    missing = [k for k in ("BBL_BETA", "BBL_HORIZON") if k not in vals]
    if missing:
        raise KeyError(f"{', '.join(missing)} not set in {p}")
    return {"BBL_BETA": float(vals["BBL_BETA"]), "BBL_HORIZON": int(vals["BBL_HORIZON"])}


def psi_discount(bbl_dir, key):
    """(beta, T, source): what the psi of run `key` were SIMULATED under, from that run's records.

    First psi_starts_<key>.json, which shard 0 of the forward sim writes under --multi-start (the
    sim's own record of the values it used); else the launch record bbl_run.sh writes for every
    run, .dispatch/<key>/context.env (CL_BBL_DISPATCH_ROOT relocates it). (None, None, why) when
    neither exists -- a run launched by hand with neither record.
    """
    d = Path(bbl_dir)
    sj = d / f"psi_starts_{key}.json"
    if sj.is_file():
        try:
            meta = json.loads(sj.read_text(encoding="utf-8"))
            if "beta" in meta and "T" in meta:
                return float(meta["beta"]), int(meta["T"]), sj.name
        except (OSError, ValueError):
            pass
    root = Path(os.environ.get("CL_BBL_DISPATCH_ROOT") or d / ".dispatch")
    ctx = root / key / "context.env"
    if ctx.is_file():
        vals = {}
        for line in ctx.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^(BETA|HORIZON)=(.*)$", line.strip())
            if m:
                vals[m.group(1)] = m.group(2).strip().strip("'\"")
        try:
            return (float(vals["BETA"]), int(vals["HORIZON"]),
                    f".dispatch/{key}/context.env (the bbl_run.sh launch record)")
        except (KeyError, ValueError):
            pass
    return None, None, "not recorded (no psi_starts sidecar and no launch context)"


# The model switches of bbl_fwd_sim.jl and the files behind them, as the psi record them. A psi
# written before the switches existed records none, and was simulated with these values.
SIM_SWITCH_DEFAULTS = {"phi_path": "frozen", "z_path": "frozen", "rdep_timing": "contemporaneous"}
SIM_PROV_KEYS = ("phi_path", "z_path", "rdep_timing", "sleep_link", "sleep_link_sha256",
                 "state_transitions", "state_transitions_sha256", "phi_t0_max_abs_diff",
                 "phi_d_rows", "sim_version", "spread_units", "policy_csv", "policy_csv_sha256")


def file_sha256(path, chunk=1 << 22):
    """sha256 of a file, hex (the value bbl_fwd_sim.jl records for its inputs)."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def parquet_kv_metadata(path):
    """The file-level key-value metadata of a parquet file (bbl_fwd_sim.jl writes the 'bbl.<field>'
    provenance there) as str -> str; {} when pyarrow is unavailable or the footer is unreadable."""
    try:
        import pyarrow.parquet as pq
        kv = pq.ParquetFile(path).metadata.metadata or {}
        return {k.decode(): v.decode() for k, v in kv.items()}
    except Exception:   # noqa: BLE001 -- a shard whose footer is unreadable already fails validation
        return {}


def psi_sim_paths(bbl_dir, key):
    """dict of SIM_PROV_KEYS plus 'source': how the psi of run `key` were simulated.

    First psi_starts_<key>.json (shard 0, --multi-start), else the file-level key-value metadata
    ('bbl.<field>') of psi_eq_<key>.parquet, which every run writes. A switch neither records is
    reported at its SIM_SWITCH_DEFAULTS value, with 'source' saying so.
    """
    d = Path(bbl_dir)
    out, src = {}, None
    sj = d / f"psi_starts_{key}.json"
    if sj.is_file():
        try:
            meta = json.loads(sj.read_text(encoding="utf-8"))
            out = {k: meta[k] for k in SIM_PROV_KEYS if k in meta}
            src = sj.name
        except (OSError, ValueError):
            pass
    if not out:
        eq = d / f"psi_eq_{key}.parquet"
        if eq.is_file():
            try:
                import pyarrow.parquet as pq
                kv = pq.ParquetFile(eq).metadata.metadata or {}
                kv = {k.decode(): v.decode() for k, v in kv.items()}
                out = {k: kv[f"bbl.{k}"] for k in SIM_PROV_KEYS if f"bbl.{k}" in kv}
                src = f"{eq.name} (parquet metadata)" if out else None
            except Exception:   # noqa: BLE001 -- provenance only; the solve must not die on it
                pass
    missing = [k for k in SIM_SWITCH_DEFAULTS if k not in out]
    for k in missing:
        out[k] = SIM_SWITCH_DEFAULTS[k]
    out["source"] = (src or "not recorded") + (
        f"; {', '.join(missing)} not recorded -> the pre-switch value" if missing else "")
    return out


# ==========================================================================
# Coverage from names alone (the solve's gate)
# ==========================================================================
def shard_coverage(dev_files):
    """(N, present, missing) for a list of psi_dev paths, read from their names alone.

    N is the shard count every file agrees on -- 0 for an unsharded run, which has no index to
    cover and returns ([], []). The forward sim writes `psi_dev_<tag>_shard{i}of{N}.parquet`
    once the whole shard is simulated, so a task that died mid-shard leaves NO file: an index
    absent here is a slice of the (firm x shock) grid that was never computed. The deviations
    are split round-robin over that grid (`(i-1) % nsh == sid` in bbl_fwd_sim.jl), so each
    missing shard removes roughly one shock from nearly every firm.

    Mixed shard counts return (sorted counts, present, []) -- the solve refuses them first.
    """
    idx, n_of = set(), set()
    for p in dev_files:
        m = SHARD_RE.search(Path(p).stem)
        if m:
            idx.add(int(m.group(1)))
            n_of.add(int(m.group(2)))
        else:
            n_of.add(0)
    if len(n_of) != 1:
        return (sorted(n_of), sorted(idx), [])
    N = n_of.pop()
    if N <= 1:
        return (N, sorted(idx), [])
    return (N, sorted(idx), sorted(set(range(N)) - idx))


# ==========================================================================
# Index specs: the form sbatch --array takes and the form a human can read
# ==========================================================================
def compress_ranges(idx):
    """[3, 50, 51, ..., 299] -> "3,50-299". Sorted, de-duplicated; "" for an empty list."""
    xs = sorted({int(i) for i in idx})
    out, i = [], 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[j] + 1:
            j += 1
        out.append(str(xs[i]) if i == j else f"{xs[i]}-{xs[j]}")
        i = j + 1
    return ",".join(out)


def expand_spec(spec):
    """"3,50-52" -> [3, 50, 51, 52]. Accepts commas or blanks; a %throttle suffix is ignored."""
    spec = str(spec).split("%", 1)[0]
    out = set()
    for part in re.split(r"[,\s]+", spec.strip()):
        if not part:
            continue
        m = re.fullmatch(r"(\d+)-(\d+)", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if b < a:
                raise ValueError(f"descending range '{part}'")
            out.update(range(a, b + 1))
        elif part.isdigit():
            out.add(int(part))
        else:
            raise ValueError(f"not an index or a range: '{part}'")
    return sorted(out)


def pack(indices, k):
    """Chunk the sorted, de-duplicated index LIST into tasks of k (the last may be shorter).

    The twin of cluster_lib.sh's cl_bbl_pack_tasks, which is what actually builds the map files
    at submit time; the test suite runs both on the same inputs.
    """
    k = int(k)
    if k < 1:
        raise ValueError("k must be >= 1")
    xs = sorted({int(i) for i in indices})
    return [xs[i:i + k] for i in range(0, len(xs), k)]


# ==========================================================================
# File validation
# ==========================================================================
def _magic_ok(path):
    """Header and footer magic plus a sane footer length -- what a truncated write breaks."""
    size = path.stat().st_size
    if size < 12:
        return False, f"{size} bytes, smaller than an empty parquet"
    with open(path, "rb") as f:
        head = f.read(4)
        f.seek(size - 8)
        tail = f.read(8)
    if head != PARQUET_MAGIC:
        return False, "no PAR1 header"
    if tail[4:] != PARQUET_MAGIC:
        return False, "no PAR1 footer (truncated write)"
    flen = int.from_bytes(tail[:4], "little")
    if flen <= 0 or flen > size - 12:
        return False, f"footer length {flen} does not fit a {size}-byte file"
    return True, ""


def validate_parquet(path, required=()):
    """-> (ok, reason, nrows). nrows is None when pyarrow is unavailable."""
    path = Path(path)
    try:
        ok, why = _magic_ok(path)
    except OSError as e:
        return False, f"unreadable ({e.__class__.__name__}: {e})", None
    if not ok:
        return False, why, None
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return True, "", None
    try:
        tbl = pq.read_table(path)
    except Exception as e:  # noqa: BLE001 -- any read failure makes the shard unusable
        return False, f"pyarrow cannot read it ({e.__class__.__name__}: {str(e)[:120]})", None
    miss = [c for c in required if c not in tbl.column_names]
    if miss:
        return False, f"missing column(s) {','.join(miss)}", tbl.num_rows
    return True, "", tbl.num_rows


# ==========================================================================
# The sweep's report
# ==========================================================================
def coverage_report(bbl_dir, key, n_shards, validate=False, newer_than=None,
                    expect_starts=False, expect_beta=None, expect_horizon=None,
                    expect_switches=None, expect_sim_version=None, policy_csv=None,
                    expect_sidecar=False):
    """Everything the sweep decides from. `missing` already includes bad and stale indices.

    `expect_starts` (a multi-start run) requires the start_q column and psi_starts_<key>.json;
    `expect_sidecar` (a single-curve run) requires the sidecar alone. Either way the sidecar's
    provenance is checked against the other expectations.

    `policy_csv` (with `validate`) is also checked shard by shard: every psi file records the
    sha256 of the policy CSV it was simulated around (parquet metadata bbl.policy_csv_sha256),
    and a shard that records another one is re-run (`policy_stale`, counted in `missing`), so the
    set the solve pools holds ONE policy even when a partial launch put a second one under the
    tag. A shard that records none is left to the sidecar and version checks."""
    d = Path(bbl_dir)
    n_shards = int(n_shards)
    rep = dict(key=key, n=n_shards, present=[], missing=[], bad=[], stale=[], empty=[],
               policy_stale=[], problems=[], eq="ok", starts="n/a", tmp_debris=0)
    policy_sha = None
    if validate and policy_csv is not None and Path(policy_csv).is_file():
        policy_sha = file_sha256(policy_csv)
    if not d.is_dir():
        rep["problems"].append(f"no such directory: {d}")
        return rep
    pre = f"psi_dev_{key}"
    names = sorted(p.name for p in d.iterdir() if p.is_file())
    rep["tmp_debris"] = sum(1 for n in names if n.startswith(".tmp_") and key in n)

    # The solve's own filter: only the writer's two spellings after the key ("" and
    # _shard{i}of{N}), so a different psi tag sharing the prefix is never counted here.
    by_idx, other_n, unsharded = {}, set(), False
    for n in names:
        if not (n.startswith(pre) and n.endswith(".parquet")):
            continue
        rest = n[len(pre):-len(".parquet")]
        if rest == "":
            unsharded = True
            continue
        m = re.fullmatch(r"_shard(\d+)of(\d+)", rest)
        if not m:
            continue
        i, of = int(m.group(1)), int(m.group(2))
        if of != n_shards:
            other_n.add(of)
            continue
        by_idx[i] = d / n
    if other_n:
        rep["problems"].append(
            f"psi_dev_{key}_shard*of{{{','.join(str(x) for x in sorted(other_n))}}} exists beside "
            f"the of{n_shards} family -- the solve refuses mixed shard counts. N_SHARDS must never "
            f"change between a run and its re-runs; move the other family aside.")
    if unsharded:
        rep["problems"].append(
            f"an unsharded psi_dev_{key}.parquet exists beside the sharded family -- the solve "
            f"refuses the mix; move it aside.")

    dev_cols = DEV_COLS + (("start_q",) if expect_starts else ())
    for i in range(n_shards):
        p = by_idx.get(i)
        if p is None:
            rep["missing"].append(i)
            continue
        if newer_than is not None and p.stat().st_mtime < newer_than:
            rep["stale"].append(i)
            rep["missing"].append(i)
            continue
        if validate:
            ok, why, nrows = validate_parquet(p, dev_cols)
            if not ok:
                rep["bad"].append((p.name, why))
                rep["missing"].append(i)
                continue
            if policy_sha is not None:
                rec = parquet_kv_metadata(p).get("bbl.policy_csv_sha256", "")
                if rec and rec != policy_sha:
                    rep["policy_stale"].append(i)
                    rep["missing"].append(i)
                    continue
            if nrows == 0:
                rep["empty"].append(i)
        rep["present"].append(i)

    # Shard 0 is the only writer of psi_eq (and psi_starts): a shard-0 psi_dev without a usable
    # psi_eq is a run the solve cannot open, and re-running shard 0 is what rewrites it.
    eq = d / f"psi_eq_{key}.parquet"
    if not eq.is_file():
        rep["eq"] = "missing"
    elif newer_than is not None and eq.stat().st_mtime < newer_than:
        rep["eq"] = "stale"
    elif validate:
        ok, why, _ = validate_parquet(eq, EQ_COLS + (("start_q",) if expect_starts else ()))
        if not ok:
            rep["eq"] = f"bad ({why})"
    if expect_starts or expect_sidecar:
        sj = d / f"psi_starts_{key}.json"
        if not sj.is_file():
            rep["starts"] = "missing"
        elif newer_than is not None and sj.stat().st_mtime < newer_than:
            rep["starts"] = "stale"
        else:
            try:
                meta = json.loads(sj.read_text(encoding="utf-8"))
                rep["starts"] = "ok"
                # Provenance: the one place a shard set records what it was simulated under.
                # A mismatch is not a gap -- re-running shard 0 would give a set whose other N-1
                # shards still carry the old design -- so it is a problem, not a retry.
                if expect_beta is not None and "beta" in meta \
                        and abs(float(meta["beta"]) - float(expect_beta)) > 1e-12:
                    rep["problems"].append(
                        f"psi_starts_{key}.json records beta={meta['beta']} but this run uses "
                        f"beta={expect_beta}: the shards on disk were simulated under another "
                        f"discount factor. Use a new --psi-tag or move the old files aside.")
                if expect_horizon is not None and "T" in meta \
                        and int(meta["T"]) != int(expect_horizon):
                    rep["problems"].append(
                        f"psi_starts_{key}.json records T={meta['T']} but this run uses "
                        f"T={expect_horizon}: the shards on disk were simulated under another "
                        f"horizon. Use a new --psi-tag or move the old files aside.")
                # The model switches. A sidecar that records none (a run launched before they
                # existed: _ms1, _ms979) is refused whatever this run expects -- its psi were
                # simulated frozen / frozen / contemporaneous under ANOTHER design version, so no
                # shard this code simulates can complete it.
                for sw, want in (expect_switches or {}).items():
                    if want is None:
                        continue
                    if sw not in meta:
                        rep["problems"].append(
                            f"psi_starts_{key}.json records no {sw}: the run was launched before "
                            f"the model switches existed, under another design. Launch it again "
                            f"under a new --psi-tag.")
                        continue
                    got = meta[sw]
                    if got != want:
                        rep["problems"].append(
                            f"psi_starts_{key}.json records {sw}={got} but this run uses "
                            f"{sw}={want}: the shards on disk were simulated under another model. "
                            f"Use a new --psi-tag or move the old files aside.")
                # The design version: a sidecar of another version, or of none (every run
                # launched before 2026-09-29), belongs to another design.
                if expect_sim_version is not None and meta.get("sim_version") != expect_sim_version:
                    rep["problems"].append(
                        f"psi_starts_{key}.json records sim_version={meta.get('sim_version', 'none')} "
                        f"but this code is {expect_sim_version}: the shards on disk were simulated "
                        f"under another design. Launch it again under a new --psi-tag.")
                # The policy the re-run shards would read must be the one these psi were
                # simulated around: a polfunc re-fit since then changes sigma-hat under them.
                if policy_csv is not None:
                    pcsv = Path(policy_csv)
                    if not pcsv.is_file():
                        rep["problems"].append(f"policy CSV {policy_csv} not found: no shard can be re-run.")
                    else:
                        now = policy_sha or file_sha256(pcsv)
                        then = meta.get("policy_csv_sha256", "")
                        if now != then:
                            rep["problems"].append(
                                f"{pcsv.name} (sha256 {now[:12]}) is not the policy CSV the psi on disk "
                                f"were simulated around (psi_starts_{key}.json records "
                                f"{then[:12] or 'none'}): re-running shards now would pool two "
                                f"policies. Restore that CSV, or launch again under a new --psi-tag.")
            except (OSError, ValueError) as e:
                rep["starts"] = f"bad ({e.__class__.__name__})"
    if rep["eq"] != "ok" or rep["starts"] not in ("ok", "n/a"):
        if 0 not in rep["missing"]:
            rep["missing"].append(0)
            rep["missing"].sort()
            if 0 in rep["present"]:
                rep["present"].remove(0)
    return rep


def _print_report(rep):
    miss = rep["missing"]
    print(f"COVERAGE key={rep['key']} n={rep['n']} present={len(rep['present'])} "
          f"missing={len(miss)} bad={len(rep['bad'])} stale={len(rep['stale'])} "
          f"empty={len(rep['empty'])} eq={rep['eq'].split(' ')[0]} "
          f"starts={rep['starts'].split(' ')[0]} tmp_debris={rep['tmp_debris']}")
    print(f"MISSING_SPEC={compress_ranges(miss)}")
    print(f"MISSING_LIST={' '.join(str(i) for i in miss)}")
    for name, why in rep["bad"]:
        print(f"BAD_FILE={name}|{why}")
    if rep.get("policy_stale"):
        print(f"POLICY_STALE_SPEC={compress_ranges(rep['policy_stale'])}")
    if rep["stale"]:
        print(f"STALE_SPEC={compress_ranges(rep['stale'])}")
    if rep["eq"] != "ok":
        print(f"EQ_STATUS={rep['eq']}")
    if rep["starts"] not in ("ok", "n/a"):
        print(f"STARTS_STATUS={rep['starts']}")
    for p in rep["problems"]:
        print(f"PROBLEM={p}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="BBL psi_dev shard coverage and index specs.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("coverage", help="which shards of one run are present and usable")
    c.add_argument("--dir", required=True, help="the bbl step folder (CF_COST_FWD)")
    c.add_argument("--key", required=True, help="E{k}_spec_{s}_{stage}{suffix}{psi_tag}")
    c.add_argument("--n-shards", type=int, required=True)
    c.add_argument("--validate", action="store_true", help="open every parquet, not just stat it")
    c.add_argument("--newer-than", type=float, default=None,
                   help="epoch seconds: an older file is stale and counts as missing")
    c.add_argument("--expect-starts", action="store_true",
                   help="multi-start run: require start_q and psi_starts_<key>.json")
    c.add_argument("--expect-sidecar", action="store_true",
                   help="single-curve run: require psi_starts_<key>.json (no start_q)")
    c.add_argument("--expect-beta", type=float, default=None)
    c.add_argument("--expect-horizon", type=int, default=None)
    c.add_argument("--expect-phi-path", default=None, choices=("evolving", "frozen"))
    c.add_argument("--expect-z-path", default=None, choices=("mean_reverting", "frozen"))
    c.add_argument("--expect-rdep-timing", default=None, choices=("lagged", "contemporaneous"))
    c.add_argument("--expect-sim-version", default=None,
                   help="the design version the sidecar must record (cluster_lib.sh BBL_SIM_VERSION)")
    c.add_argument("--policy-csv", default=None,
                   help="the policy CSV re-run shards would read; its sha256 must be the sidecar's")
    p = sub.add_parser("compress", help="indices -> 3,50-299")
    p.add_argument("idx", nargs="*", type=int)
    e = sub.add_parser("expand", help="3,50-299 -> indices")
    e.add_argument("spec")
    k = sub.add_parser("pack", help="index list -> one task per line, k shards each")
    k.add_argument("--k", type=int, required=True)
    k.add_argument("--spec", required=True)
    g = sub.add_parser("discount", help="print BBL_BETA / BBL_HORIZON from bbl_discount.env")
    g.add_argument("--path", default=None)
    a = ap.parse_args(argv)

    if a.cmd == "discount":
        for key, val in read_bbl_discount(a.path).items():
            print(f"{key}={val}")
        return 0
    if a.cmd == "compress":
        print(compress_ranges(a.idx))
        return 0
    if a.cmd == "expand":
        print(" ".join(str(i) for i in expand_spec(a.spec)))
        return 0
    if a.cmd == "pack":
        for t in pack(expand_spec(a.spec), a.k):
            print(" ".join(str(i) for i in t))
        return 0
    rep = coverage_report(a.dir, a.key, a.n_shards, validate=a.validate,
                          newer_than=a.newer_than, expect_starts=a.expect_starts,
                          expect_beta=a.expect_beta, expect_horizon=a.expect_horizon,
                          expect_switches={"phi_path": a.expect_phi_path,
                                           "z_path": a.expect_z_path,
                                           "rdep_timing": a.expect_rdep_timing},
                          expect_sim_version=a.expect_sim_version, policy_csv=a.policy_csv,
                          expect_sidecar=a.expect_sidecar)
    _print_report(rep)
    if rep["problems"]:
        return 2
    return 1 if rep["missing"] else 0


if __name__ == "__main__":
    sys.exit(main())

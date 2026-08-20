#!/usr/bin/env python3
"""
stage_cluster_upload.py -- pre-flight the Bouchet inputs and bundle them into two zips.

The upload itself is MANUAL (Yale's web interface), so this script optimises for a human
doing it by hand: it checks every input declared in cluster/upload_manifest.txt, refuses to
bundle a missing or stale one, and then produces exactly TWO files to upload:

    <out>/<YYYYMMDD>/bouchet_code_<YYYYMMDD>.zip   all *.jl / *.sh / Project+Manifest.toml
    <out>/<YYYYMMDD>/bouchet_data_<YYYYMMDD>.zip   parquets / .bin / .csv / .json inputs

Two bundles rather than one because they change at different rates: the code changes almost
every run and is small, the data changes only when the estimation vintage changes and is
large. A code-only fix therefore costs a small upload, not a repeat of the big one.

Each zip mirrors the cluster tree (`data/input/...`, `scripts/...`), so ONE unzip at the
project root puts every file where it belongs -- nothing to move by hand. Each zip also
carries, at its root, a plain-text UPLOAD_README.txt with the numbered steps and a
sha256SUMS.txt covering its own members.

WHY THE STALENESS CHECK IS THE POINT. Every input here derives from another artefact: a
demand parquet from its routine's estimation_results.pkl, a logit delta from its demand
parquet, polfunc_fitted.csv from market_panel.csv. An input OLDER than its parent looks
perfectly valid on the cluster -- it is a real file of the right name -- and produces a
mixed-vintage run whose results are wrong in a way no cluster-side preflight can see. So an
entry whose mtime predates its parent's is refused here, before it is ever uploaded.

Usage:
    python stage_cluster_upload.py                       # pre-flight only (default)
    python stage_cluster_upload.py --stage               # pre-flight, then build both zips
    python stage_cluster_upload.py --vintage-root C:\\egan_relineup_20260819
    python stage_cluster_upload.py --stages blp,bbl,cf,cf4
    python stage_cluster_upload.py --stage --allow-stale # explicit override; says so loudly
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import os
import stat
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

MANIFEST = REPO / "cluster" / "upload_manifest.txt"

# Project root on Bouchet. Taken from the #SBATCH --output lines in submit_blp_1_draws.sh /
# setup_julia_env.sh, which are the only absolute cluster paths committed to this repo; the
# submit scripts themselves resolve data as `$(pwd)/../data` from scripts/.
CLUSTER_ROOT = "/nfs/roberts/project/pi_mf2263/pf382/dep_comp"

DEFAULT_OUT_ROOT = Path(r"C:\egan_cluster_stage")
DEFAULT_STAGES = ("blp", "bbl", "cf")

# Members stored uncompressed: parquet and the raw delta binaries are already compressed or
# incompressible, and deflating 250 MB of them costs minutes for ~1% of size.
STORE_SUFFIXES = {".parquet", ".bin", ".jls", ".so", ".zip"}
# Rewritten to LF on the way in: bash reads a CRLF shebang as a syntax error on line 1, and
# Julia scripts are read by the same shell wrappers.
LF_SUFFIXES = {".sh", ".jl"}
EXEC_SUFFIXES = {".sh"}


# ---------------------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------------------
class Row:
    """One raw manifest line, before {k}/{prefix} expansion."""

    __slots__ = ("name", "bundle", "stages", "root", "pattern", "dest",
                 "producer", "parent_root", "parent_pattern")

    def __init__(self, fields):
        (self.name, self.bundle, stages, self.root, self.pattern, self.dest,
         self.producer, self.parent_root, self.parent_pattern) = fields
        self.stages = tuple(s for s in stages.split(",") if s)


class Entry:
    """One resolved file (or, for code globs, one group of files) to upload."""

    def __init__(self, name, bundle, stages, paths, dest, producer, parent, is_glob):
        self.name = name
        self.bundle = bundle
        self.stages = stages
        self.paths = paths            # list[Path]; may be empty when nothing matched
        self.dest = dest              # cluster-relative destination directory
        self.producer = producer
        self.parent = parent          # Path | None
        self.is_glob = is_glob
        self.status = "?"
        self.detail = ""

    @property
    def display(self):
        if self.is_glob:
            return f"{len(self.paths)} file(s)"
        return self.paths[0].name if self.paths else "(missing)"

    @property
    def producer_short(self):
        if self.producer == "-":
            return "(in git)"
        for tok in self.producer.split():
            if tok.endswith(".py") or tok.endswith(".jl"):
                return tok
        return self.producer.split()[0]


def parse_manifest(path: Path):
    """-> (directives, rows). Directives are the '#!' lines; everything else starting with
    '#', and every blank line, is a comment."""
    directives = {"SPEC": "12", "ROUTINES": [], "PREFIX": {}}
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#!"):
            parts = line[2:].split()
            if not parts:
                continue
            key, args = parts[0], parts[1:]
            if key == "SPEC" and args:
                directives["SPEC"] = args[0]
            elif key == "ROUTINES":
                directives["ROUTINES"] = [int(a) for a in args]
            elif key == "PREFIX" and len(args) >= 2:
                directives["PREFIX"][int(args[0])] = args[1]
            continue
        if line.startswith("#"):
            continue
        fields = [f.strip() for f in line.split("|")]
        if len(fields) != 9:
            raise SystemExit(f"ERROR: manifest line has {len(fields)} fields, expected 9:\n  {line}")
        rows.append(Row(fields))
    if not directives["ROUTINES"]:
        raise SystemExit("ERROR: manifest declares no '#!ROUTINES'.")
    missing_pfx = [k for k in directives["ROUTINES"] if k not in directives["PREFIX"]]
    if missing_pfx:
        raise SystemExit(f"ERROR: manifest has no '#!PREFIX' for routine(s) {missing_pfx}.")
    return directives, rows


def make_root_resolver(paths_mod):
    """Root token -> Path, entirely through utils/paths.py accessors."""
    est_out = paths_mod.estimation_output()

    def resolve(token: str, k=None) -> Path:
        if token.startswith("EST") and token[3:].isdigit():
            return paths_mod.est_dir(int(token[3:]))
        if token == "EST{k}":
            return paths_mod.est_dir(k)
        table = {
            "DEMAND_PREP":   paths_mod.demand_prep_root(),
            "SIGMA_DIR":     paths_mod.PROCESSED / "ESTIMATION_OUTPUT" / "DEMAND_PREP",
            "BLP_LOGIT":     est_out / "BLP_RESULTS" / "logit",
            "COST_FWD":      est_out / "COST_FWD",
            "COST_POLFUNC":  est_out / "COST_POLFUNC",
            "CF_FOUNDATION": est_out / "CF_FOUNDATION",
            "PROCESSED":     paths_mod.PROCESSED,
            "REPO":          REPO,
        }
        if token not in table:
            raise SystemExit(f"ERROR: manifest uses unknown root token '{token}'.")
        return table[token]

    return resolve


def expand(rows, directives, resolve):
    """Expand {k}/{prefix}/{spec} over the declared routines -> list[Entry]."""
    spec = directives["SPEC"]
    entries = []
    for row in rows:
        ks = directives["ROUTINES"] if "{k}" in (row.name + row.pattern) else [None]
        for k in ks:
            def sub(s, k=k):
                if k is None:
                    return s.replace("{spec}", spec)
                return (s.replace("{k}", str(k))
                         .replace("{prefix}", directives["PREFIX"][k])
                         .replace("{spec}", spec))

            root = resolve(row.root, k)
            pattern = sub(row.pattern)
            is_glob = any(c in pattern for c in "*?[")
            if is_glob:
                hits = sorted(p for p in root.glob(pattern) if p.is_file())
            else:
                p = root / pattern
                hits = [p] if p.is_file() else []
            parent = None
            if row.parent_root != "-" and row.parent_pattern != "-":
                parent = resolve(row.parent_root, k) / sub(row.parent_pattern)
            entries.append(Entry(sub(row.name), row.bundle, row.stages, hits,
                                 row.dest, sub(row.producer), parent, is_glob))
    return entries


# ---------------------------------------------------------------------------------------
# pre-flight
# ---------------------------------------------------------------------------------------
def evaluate(entry: Entry):
    """OK / MISSING / STALE / UNKNOWN. STALE means older than the artefact it derives from,
    which is the failure this whole script exists to prevent."""
    if not entry.paths:
        entry.status = "MISSING"
        entry.detail = "not on disk"
        return
    if entry.parent is None:
        entry.status = "OK"
        return
    if not entry.parent.is_file():
        entry.status = "UNKNOWN"
        entry.detail = f"cannot verify freshness: parent {entry.parent.name} is not on disk"
        return
    p_mtime = entry.parent.stat().st_mtime
    oldest = min(p.stat().st_mtime for p in entry.paths)
    if oldest < p_mtime:
        lag = (p_mtime - oldest) / 3600.0
        entry.status = "STALE"
        entry.detail = (f"{lag:.1f} h older than {entry.parent.name} "
                        f"({_fmt_time(p_mtime)})")
        return
    entry.status = "OK"


def _fmt_time(ts):
    return _dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def _fmt_size(n):
    for unit, div in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if n >= div:
            return f"{n / div:.1f} {unit}"
    return f"{n} B"


def entry_size(entry):
    return sum(p.stat().st_size for p in entry.paths)


def required(entry, selected):
    return "all" in entry.stages or any(s in selected for s in entry.stages)


def print_checklist(entries, selected, directives, vintage_root, verbose):
    print("=" * 100)
    print("PRE-FLIGHT CHECKLIST")
    print("=" * 100)
    print(f"  spec            : {directives['SPEC']}")
    print(f"  routines        : {' '.join(str(k) for k in directives['ROUTINES'])}")
    print(f"  vintage root    : {vintage_root}")
    print(f"  stages required : {', '.join(selected)}")
    print(f"  manifest        : {MANIFEST}")
    print()
    hdr = f"  {'STATUS':<8}{'NAME':<20}{'REQ':<5}{'SIZE':>9}  {'MODIFIED':<17}{'PRODUCER':<30}FILE"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    fixes = []
    for e in entries:
        req = "yes" if required(e, selected) else "no"
        if e.paths:
            size = _fmt_size(entry_size(e))
            mtime = _fmt_time(min(p.stat().st_mtime for p in e.paths))
        else:
            size, mtime = "-", "-"
        print(f"  {e.status:<8}{e.name:<20}{req:<5}{size:>9}  {mtime:<17}"
              f"{e.producer_short:<30}{e.display}")
        if verbose:
            print(f"           dest {e.dest}/  |  builds with: {e.producer}")
            if e.parent is not None:
                print(f"           derives from: {e.parent}")
        if e.status != "OK":
            if e.detail:
                print(f"           {e.detail}")
            if e.producer != "-":
                tag = "" if req == "yes" else "   (optional for the selected stages)"
                print(f"           -> run: {e.producer}{tag}")
                if req == "yes" and e.producer not in fixes:
                    fixes.append(e.producer)
    print()
    return fixes


# Producers that resolve DEMAND_PREP themselves instead of going through utils/paths, so
# SLEEP_OUT_ROOT does not move them. Reading the demand parquets from the production tree
# while the vintage sits in a sandbox is how they find nothing and exit reporting success.
HARDCODED_PRODUCERS = (
    ("blp_1_logit.jl",         "get_paths(): input_dir = processed/ESTIMATION_OUTPUT/DEMAND_PREP"),
    ("cf_forward_rf.py",       "globs DEMAND_PREP/demand_3_index*spec_12.parquet"),
    ("cf_4_upsilon_export.py", "globs DEMAND_PREP/demand_{k}_* and reads DEMAND_PREP/est{k}"),
)


def warn_vintage_split(paths_mod, directives):
    """The TO FIX commands above only work if the producer can SEE the active vintage.
    Three of them resolve DEMAND_PREP for themselves, so when the vintage is redirected
    they read the production tree and quietly find nothing."""
    active = paths_mod.demand_prep_root().resolve()
    production = (paths_mod.PROCESSED / "ESTIMATION_OUTPUT" / "DEMAND_PREP").resolve()
    if active == production:
        return
    have = sorted(production.glob("demand_*_spec_12.parquet")) if production.is_dir() else []
    print("BEFORE YOU RUN THOSE -- the active vintage is NOT the production tree:")
    print(f"    active     {active}")
    print(f"    production {production}   "
          f"({len(have)} demand_*_spec_12.parquet)")
    print("  These producers resolve DEMAND_PREP themselves and do NOT follow SLEEP_OUT_ROOT,")
    print("  so from the production tree they see no demand parquets, write nothing, and can")
    print("  still exit 0:")
    for script, how in HARDCODED_PRODUCERS:
        print(f"    {script:<24} {how}")
    print("  Either copy this vintage's demand_*_spec_12.parquet into the production tree")
    print("  before running them, or run them against a tree where those files exist.")
    print()


# ---------------------------------------------------------------------------------------
# bundling
# ---------------------------------------------------------------------------------------
def _member_bytes(path: Path):
    data = path.read_bytes()
    if path.suffix.lower() in LF_SUFFIXES:
        data = data.replace(b"\r\n", b"\n")
    return data


def _zip_add(zf, arcname, data, suffix):
    info = zipfile.ZipInfo(arcname, date_time=_dt.datetime.now().timetuple()[:6])
    mode = 0o755 if suffix in EXEC_SUFFIXES else 0o644
    info.external_attr = (stat.S_IFREG | mode) << 16
    info.compress_type = (zipfile.ZIP_STORED if suffix in STORE_SUFFIXES
                          else zipfile.ZIP_DEFLATED)
    zf.writestr(info, data)


def build_bundle(kind, entries, out_dir: Path, datestamp: str, next_steps, omitted):
    """Write one zip; -> (path, [(arcname, size, sha256, producer)])."""
    zip_path = out_dir / f"bouchet_{kind}_{datestamp}.zip"
    members = []
    with zipfile.ZipFile(zip_path, "w", allowZip64=True) as zf:
        for e in entries:
            for p in e.paths:
                arc = f"{e.dest}/{p.name}"
                data = _member_bytes(p)
                _zip_add(zf, arc, data, p.suffix.lower())
                members.append((arc, len(data), hashlib.sha256(data).hexdigest(),
                                e.producer_short))
        sums = "".join(f"{h}  {a}\n" for a, _, h, _ in sorted(members))
        _zip_add(zf, "sha256SUMS.txt", sums.encode("ascii"), ".txt")
        readme = render_readme(kind, datestamp, members, next_steps, omitted)
        _zip_add(zf, "UPLOAD_README.txt", readme.encode("ascii"), ".txt")
    verify_bundle(zip_path, members)
    return zip_path, members


def verify_bundle(zip_path: Path, members):
    """Re-open the finished zip and confirm what we claim about it: every member present,
    every hash matching, and not one CR left in a .sh or .jl."""
    want = {a: h for a, _, h, _ in members}
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        missing = sorted(set(want) - names)
        if missing:
            raise SystemExit(f"ERROR: {zip_path.name} is missing {len(missing)} member(s): "
                             f"{missing[:3]}")
        crlf = []
        for arc, h in want.items():
            data = zf.read(arc)
            if hashlib.sha256(data).hexdigest() != h:
                raise SystemExit(f"ERROR: {zip_path.name}:{arc} hash mismatch after write.")
            if arc.rsplit(".", 1)[-1].lower() in ("sh", "jl") and b"\r\n" in data:
                crlf.append(arc)
        if crlf:
            raise SystemExit(f"ERROR: CRLF survived in {len(crlf)} member(s): {crlf[:3]}")


def sha256_file(path: Path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def _wrap(text, width=76, indent=""):
    out, line = [], indent
    for word in text.split():
        if line.strip() and len(line) + 1 + len(word) > width:
            out.append(line)
            line = indent + word
        else:
            line = (line + " " + word) if line.strip() else line + word
    if line.strip():
        out.append(line)
    return out


def render_readme(kind, datestamp, members, next_steps, omitted):
    payload = [m for m in members]
    total = sum(sz for _, sz, _, _ in payload)
    what = ("the CODE for the run: every Julia and shell script in the repository, plus "
            "Project.toml and Manifest.toml so `julia --project=.` can resolve."
            if kind == "code" else
            "the DATA INPUTS for the run. Every file listed at the bottom of this README "
            "is in the bundle; nothing else is.")
    dest = "scripts/" if kind == "code" else "data/input/"
    lines = [
        "=" * 78,
        f"  bouchet_{kind}_{datestamp}.zip",
        "=" * 78,
        "",
        "WHAT THIS IS:",
    ]
    lines += _wrap(what, indent="  ")
    lines += [
        "",
        f"BUILT ON    : {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')} (local machine)",
        f"CONTAINS    : {len(payload)} files, {_fmt_size(total)} uncompressed,",
        f"              all under {dest}",
    ]
    gone = [(n, p) for n, p, st, _ in omitted if st == "MISSING"]
    stale = [(n, p, d) for n, p, st, d in omitted if st == "STALE"]
    if gone:
        lines += ["", "NOT IN THIS BUNDLE -- the manifest declares these, but they were not on",
                  "the local disk when it was built. If a cluster stage needs one, its own",
                  "preflight will say so; build it locally and upload a second bundle:"]
        for name, producer in gone:
            lines.append(f"    {name:<20} build with: {producer}")
    if stale:
        lines += ["", "WARNING -- these ARE in the bundle but are OLDER than the artefact they",
                  "derive from, so this upload mixes estimation vintages. Rebuild and re-upload",
                  "them before trusting any result that depends on them:"]
        for name, producer, detail in stale:
            lines.append(f"    {name:<20} {detail}")
            lines.append(f"    {'':<20} rebuild with: {producer}")
    if kind == "data":
        lines += [
            "",
            "Everything inside is an UPLOADED INPUT. On this cluster data/input holds only",
            "files that came from the local machine and data/output holds only files the",
            "cluster made. Never let an archiver script run in move mode over data/input:",
            "zip_cf_outputs.sh deletes what it archives unless KEEP=1, and it once emptied",
            "a directory that way.",
        ]
    else:
        lines += [
            "",
            "Every .sh here is stored with Unix line endings. That is not cosmetic: bash",
            "reads a Windows line ending on the first line as part of the shebang and every",
            "submit script fails at line 1. If you ever copy one of these across by hand",
            "instead of unzipping, check it with `file submit_cf_all.sh` first.",
        ]
    lines += [
        "",
        "-" * 78,
        "STEPS",
        "-" * 78,
        "",
        f"1. Put this zip in your home directory on Bouchet:   ~/bouchet_{kind}_{datestamp}.zip",
        "",
        "2. Open a terminal on the login node and go to the PROJECT ROOT.",
        "   This is the directory you unzip FROM. It is the parent of both scripts/ and data/:",
        "",
        f"      cd {CLUSTER_ROOT}",
        "",
        "3. Unzip. The paths inside the zip already match the tree, so this one command puts",
        "   every file in its final place - nothing to move by hand:",
        "",
        f"      unzip -o ~/bouchet_{kind}_{datestamp}.zip",
        "",
        "4. Check the file count. It should print exactly this many files:",
        "",
        f"      unzip -l ~/bouchet_{kind}_{datestamp}.zip | tail -1",
        f"          expected: {len(payload) + 2} files "
        f"({len(payload)} + this README + sha256SUMS.txt)",
        "",
        "5. Check the contents are intact. From the same directory:",
        "",
        "      sha256sum -c sha256SUMS.txt",
        "",
        f"   Every line must end in OK ({len(payload)} lines). If any line says FAILED the",
        "   upload was corrupted - upload the zip again and repeat from step 3.",
        "",
        "6. Tidy up the two bookkeeping files, which land in the project root:",
        "",
        "      rm -f UPLOAD_README.txt sha256SUMS.txt",
        "",
        "   Do this only AFTER step 5 passes. If you are uploading both the code and the data",
        "   bundle, finish steps 3-5 for one bundle before unzipping the other: they each",
        "   carry their own sha256SUMS.txt and the second overwrites the first.",
        "",
        "-" * 78,
        "THEN RUN",
        "-" * 78,
        "",
    ]
    lines += next_steps
    lines += [
        "",
        "-" * 78,
        "FILES IN THIS BUNDLE",
        "-" * 78,
        "",
    ]
    for arc, sz, _, producer in sorted(payload):
        lines.append(f"  {_fmt_size(sz):>9}  {arc:<58}  built by {producer}")
    lines.append("")
    return "\n".join(lines) + "\n"


def next_steps_for(kind):
    if kind == "code":
        return [
            f"  cd {CLUSTER_ROOT}/scripts",
            "  chmod +x *.sh",
            "",
            "  Only if Project.toml or Manifest.toml changed in this bundle:",
            "      sbatch setup_julia_env.sh          # 5-15 min, wait for it to finish",
            "",
            "  Then, in this order (each waits on the one before it):",
            "      sbatch submit_blp_1_draws.sh       # only if data/output/BLP_DRAWS is empty",
            "      bash   submit_blp_2_rc_default.sh  # the RC-BLP chains",
            "      bash   submit_bbl_all.sh           # BBL costs (needs the RC results)",
            "      bash   submit_cf_all.sh            # counterfactuals (needs the BBL costs)",
            "",
            "  Each of those prints its own preflight and stops with a list of anything it",
            "  cannot find, so a missing piece is reported before any job is submitted.",
        ]
    return [
        "  Nothing. This bundle is only input data - no job reads it until a submit script",
        "  runs. Upload the code bundle too, then follow ITS instructions.",
        "",
        "  To confirm the files arrived where the submit scripts look for them:",
        f"      ls -la {CLUSTER_ROOT}/data/input",
    ]


# ---------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Pre-flight the Bouchet inputs and bundle them into two upload zips.")
    ap.add_argument("--stage", action="store_true",
                    help="build the zips (default is pre-flight only)")
    ap.add_argument("--dry-run", action="store_true",
                    help="pre-flight only; the default, accepted for explicitness")
    ap.add_argument("--stages", default=",".join(DEFAULT_STAGES),
                    help="cluster stages this run needs, comma separated: blp,bbl,cf,cf4. "
                         "An input is REQUIRED when one of these consumes it.")
    ap.add_argument("--routines", default=None,
                    help="override the manifest's routine lineup, e.g. '3,4'")
    ap.add_argument("--vintage-root", default=None,
                    help="the estimation vintage to stage from; sets SLEEP_OUT_ROOT so every "
                         "utils/paths accessor follows it")
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT),
                    help=f"where the dated staging folder is written (default {DEFAULT_OUT_ROOT})")
    ap.add_argument("--date", default=None, help="datestamp for the zips (default: today)")
    ap.add_argument("--allow-stale", action="store_true",
                    help="bundle even when a required input is missing or stale")
    ap.add_argument("--verbose", action="store_true", help="print the full producer command")
    args = ap.parse_args()

    if args.vintage_root:
        os.environ["SLEEP_OUT_ROOT"] = args.vintage_root
    from utils import paths  # imported AFTER SLEEP_OUT_ROOT so the override is honoured

    if not MANIFEST.is_file():
        raise SystemExit(f"ERROR: manifest not found: {MANIFEST}")
    directives, rows = parse_manifest(MANIFEST)
    if args.routines:
        directives["ROUTINES"] = [int(x) for x in args.routines.replace(",", " ").split()]
    resolve = make_root_resolver(paths)
    entries = expand(rows, directives, resolve)
    for e in entries:
        evaluate(e)

    selected = [s.strip() for s in args.stages.split(",") if s.strip()]
    fixes = print_checklist(entries, selected, directives,
                            paths.demand_prep_root(), args.verbose)

    blockers = [e for e in entries if required(e, selected) and e.status in ("MISSING", "STALE")]
    unknown = [e for e in entries if required(e, selected) and e.status == "UNKNOWN"]
    ok = [e for e in entries if e.paths and e.status != "MISSING"]

    if fixes:
        print("TO FIX -- run these locally, then re-run this script:")
        for i, cmd in enumerate(fixes, 1):
            print(f"  {i}. {cmd}")
        print()
        warn_vintage_split(paths, directives)

    data_entries = [e for e in ok if e.bundle == "data"]
    code_entries = [e for e in ok if e.bundle == "code"]
    data_bytes = sum(entry_size(e) for e in data_entries)
    code_bytes = sum(entry_size(e) for e in code_entries)
    print(f"BUNDLE SIZES (uncompressed, of what is currently stageable)")
    print(f"  code : {sum(len(e.paths) for e in code_entries):>4} files  {_fmt_size(code_bytes):>9}")
    print(f"  data : {sum(len(e.paths) for e in data_entries):>4} files  {_fmt_size(data_bytes):>9}")
    if data_bytes >= 4 << 30:
        print()
        print("  NOTE: the data bundle is over 4 GB. A single manual upload through a browser")
        print("  is likely to time out at this size. The per-file sizes are in the table above")
        print("  so you can decide how to split it -- this script deliberately does not split")
        print("  it for you, because a partial bundle that looks complete is worse than a big")
        print("  one. Consider uploading only the entries whose stage you are actually running")
        print("  (--stages), or transferring the largest files separately.")
    elif data_bytes >= 2 << 30:
        print("  NOTE: the data bundle is over 2 GB -- expect a slow upload, but it should go")
        print("  through in one piece.")
    print()

    if not args.stage:
        print("Pre-flight only (no files written). Re-run with --stage to build the zips.")
        if blockers:
            print(f"As it stands --stage would REFUSE: {len(blockers)} required input(s) "
                  f"missing or stale.")
        return 0 if not blockers else 1

    if blockers and not args.allow_stale:
        print("REFUSING TO STAGE.")
        for e in blockers:
            print(f"  {e.status:<8}{e.name:<20}{e.detail}")
        print()
        print("Uploading a stale input is how a mixed-vintage run happens: the file is real,")
        print("the cluster preflight passes, and the results are wrong. Build the artefacts")
        print("listed under TO FIX above, then re-run. --allow-stale overrides this.")
        return 1
    if blockers and args.allow_stale:
        print(f"WARNING: --allow-stale -- bundling with {len(blockers)} missing/stale "
              f"required input(s). This run will be mixed-vintage.")
        print()
    if unknown:
        print(f"NOTE: {len(unknown)} input(s) could not be freshness-checked "
              f"(their parent artefact is not on disk).")
        print()

    datestamp = args.date or _dt.datetime.now().strftime("%Y%m%d")
    out_dir = Path(args.out_root) / datestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print(f"STAGING -> {out_dir}")
    print("=" * 100)
    results = []
    for kind, ents in (("code", code_entries), ("data", data_entries)):
        if not ents:
            print(f"  {kind}: nothing stageable -- skipped")
            continue
        omitted = [(e.name, e.producer, e.status, e.detail) for e in entries
                   if e.bundle == kind and e.status in ("MISSING", "STALE")]
        zp, members = build_bundle(kind, ents, out_dir, datestamp,
                                   next_steps_for(kind), omitted)
        results.append((kind, zp, members))
        print(f"  {kind}: {len(members)} files -> {zp.name} "
              f"({_fmt_size(zp.stat().st_size)} on disk)")
        print(f"         LF check passed for every .sh/.jl member; "
              f"sha256SUMS.txt + UPLOAD_README.txt written at the zip root")
    print()

    print("=" * 100)
    print("WHAT YOU DO NEXT")
    print("=" * 100)
    print(f"  1. Open  {out_dir}")
    for i, (kind, zp, members) in enumerate(results, start=2):
        print(f"  {i}. Upload  {zp.name}  ({_fmt_size(zp.stat().st_size)}, "
              f"{len(members)} files) to your Bouchet home directory.")
    n = len(results) + 2
    print(f"  {n}. On Bouchet:  cd {CLUSTER_ROOT}")
    print(f"  {n + 1}. For EACH zip, in turn: unzip it, check the count, run "
          f"`sha256sum -c sha256SUMS.txt`.")
    print(f"  {n + 2}. Then follow the THEN RUN section of the code bundle's UPLOAD_README.txt.")
    print()
    print("  The full numbered instructions are inside each zip as UPLOAD_README.txt, so you")
    print("  do not need this terminal open while you upload.")
    print()
    print("SHA256 OF EACH ZIP (compare after transfer with `sha256sum <file>` on Bouchet):")
    for kind, zp, _ in results:
        print(f"  {sha256_file(zp)}  {zp.name}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

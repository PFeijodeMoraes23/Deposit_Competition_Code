"""build_stagec_bundle.py -- the stage C upload for a FRESH Bouchet tree, computed at build time.

    python -I build_stagec_bundle.py                    # REPORT only
    python -I build_stagec_bundle.py --stage OUTDIR     # + the upload

The cluster is empty, so everything the chain executes and everything it reads ships from here.

  CODE   the closure of the chain, found by a crawl from every entry point (the shell drivers and
         job scripts the chain submits, the wrapper's modes, env_job.sh's warm-up steps): Python
         imports, Julia include()s, and every quoted or bare file name a script mentions that
         exists in the repo (subprocess lists, `run_py x.py`, "${CL_ROOT}/x.sh"). Plus Project.toml,
         Manifest.toml (the local 1.12.6 set) and Manifest-v1.11.toml: the set resolved locally
         under Julia 1.11.4 (jl111/resolve_111.jl) and validated there (jl111/validate_111.log:
         check c identical), which the cluster's Julia 1.11 reads first and the wrapper's env_julia
         instantiates without ever resolving; its sha256 is pinned below and in the pin file. Also
         the config and the upload manifest the preflight reads. Text files are shipped with LF
         endings. Superseded files are refused: utils/demand_repair.py, panel_demand_repair*.py,
         blp_dx*.jl, dx_*.sh.
  DATA   every input the chain reads, listed explicitly below (never globbed), sha256 recorded:
         the 12 data rows of cluster/upload_manifest.txt (cluster_preflight.sh blocks on any one
         missing), the demand-characteristics file and the policy (sha256 must be fd208b5f...).
         No sleepiness estimate ships: the chain estimates E1-E4 on the cluster (Pedro, 10-15).
         Never shipped: demand_repair*, blp_dx*, dx_*, and the weak-IV battery (EXCLUDE: not in
         the chain, old instrument names). No local reference values ship either: the cluster's
         gates are automatic sanity checks.
  PYTHON the third-party modules the executed Python imports; the wrapper probes them, and its
         refusal prints a conda create line pinned to this machine's versions.

Upload (--stage OUTDIR):
  stagec_code_<stamp>.zip   scripts/... (with scripts/stagec_arrival_<stamp>.sha256 and the pin file)
  stagec_data_<stamp>.zip   data/input/... (every input; ~0.15 GB)
  stagec_upload_<stamp>.md5 md5 of both zips, for `md5sum -c` in the project root
Exit 0 = ready; 1 = refused (reasons printed).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata as md
import json
import re
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(r"C:\Users\pedro\OneDrive\Documentos\Yale\Year 3 (2024 - 2025)\Open Finance\Open-Finance\Code\Egan_et_al_2025_Rep")
OF = REPO.parents[1]
PROCESSED = OF / "BCB" / "Egan_et_al_2025_Rep" / "processed"
EO = PROCESSED / "ESTIMATION_OUTPUT"
WRAPPER = HERE / "stagec_suite_20261012.sh"
PINS_NAME = "stagec_pins_20261012.env"
STAMP = dt.date.today().strftime("%Y%m%d")
CHARS_NAME = f"demand_chars_{STAMP}.parquet"
POLICY_UPLOAD = "polfunc_fitted_liverows_20260930.csv"
POLICY_SHA256 = "fd208b5fc57b9697eefc010ed4a1076af776d9510b89f3bbc1edca009dc29728"
PART_BYTES = 1 << 30
CONDA_ENV = "dep_comp_blp"          # cluster_lib.sh's CONDA_ENV default
# The Julia 1.11 package set (D7, user 10-13): resolved under Julia 1.11.4 from the 1.12.6 manifest
# (172 of 173 registered packages at the same version; PrecompileTools 1.3.3 -> 1.2.1) for THIS
# Project.toml, validated locally (load test + stage C check c identical to the 1.12.6 run).
# In the repo since 10-14 (D2): the 1.11 manifest sits next to Project.toml, and the Project.toml it
# was resolved for is pinned by sha256 (the resolve and validate scripts and their logs stay in
# checkpoints_20260928/stageC_launch_20261012/jl111).
JL111 = HERE
PROJECT_SHA256 = "5833c9080e892416a5421aaa9d70e394dca74a6282b440d34f82fc7087057ec0"
JL_MANIFEST = "Manifest-v1.11.toml"
JL_MANIFEST_SHA256 = "781049148b0e82809180f072799e94e07220080e9b6cf2483f071158fb50a4e9"

# ── DATA: destination name in data/input -> local source. Explicit; never a glob. ──────────
DATA = {
    "market_panel.parquet":            PROCESSED / "market_panel.parquet",
    "digital_banks_diagnostic.csv":    PROCESSED / "PANEL_INTERMED" / "digital_banks_diagnostic.csv",
    "estban_own_branches.csv":         PROCESSED / "PANEL_INTERMED" / "estban_own_branches.csv",
    "bcb_banked_mca_panel.csv":        OF / "BCB" / "Inclusion" / "bcb_banked_mca_panel.csv",
    "state_centering_means.json":      EO / "DEMAND_PREP" / "state_centering_means.json",
    "state_centering_tier0.csv":       EO / "DEMAND_PREP" / "state_centering_tier0.csv",
    "demographics_sigma.parquet":      EO / "DEMAND_PREP" / "demographics_sigma.parquet",
    "forward_rf_qoq.csv":              EO / "COST_FWD" / "forward_rf_qoq.csv",
    "forward_rf_vintages.csv":         EO / "COST_FWD" / "forward_rf_vintages.csv",
    "bbl_transitions.json":            EO / "COST_FWD" / "bbl_transitions.json",
    "sleep_link_E3_spec_12.json":      EO / "COST_FWD" / "sleep_link_E3_spec_12.json",
    "sleep_link_E4_spec_12.json":      EO / "COST_FWD" / "sleep_link_E4_spec_12.json",
    CHARS_NAME:                        PROCESSED / "demand_chars.parquet",
    POLICY_UPLOAD:                     EO / "COST_POLFUNC" / "polfunc_fitted.csv",
}
CHARS_REPORT = PROCESSED / "demand_chars_report.json"


# Hard pins (coordinator, 10-13 15:04). utils/demand_chars.apply refuses any market panel but this
# one, and the side file was rebuilt with the panel md5 in its schema metadata. Never overridable.
PANEL_MD5 = "6613228b2fe6b0ba95d0f16d1d5a743c"
CHARS_SHA256 = "55fa8698f76341d3d7a8937af2b9d5cfaa23bed022e5ec485be46df75d6c9c72"
# The demographic sigma file the draws read (blp_draws.jl; nothing else in the bundle reads it or
# derives from it). panel_demographics_sigma.py rebuilt it on 2026-10-14 20:11 (the two ANATEL
# sigma columns, connections_per100_sigma and frac_4g5g_sigma) and again on 2026-10-15 16:09,
# which dropped the 8 demographic mean columns the draws never read (they take the means from
# the demand panels). The 10-14 file (sha256 9e0cfc96...) has the same keys and bit-identical
# sigma columns, so it gives the same draws; the pin is the 10-15 file. The file the 10-13
# bundle shipped (08-26) is refused by name.
SIGMA_NAME = "demographics_sigma.parquet"
SIGMA_SHA256 = "d96c647298a94c8b3428302132524558c28c1a35577f746d0833fb49f7cd21e6"
SIGMA_STALE_SHA256 = {"6506ecc7012041b95957d4e89733b6147f393e9da9150c61b48e8b1d225343d8": "the 08-26 file the 10-13 bundle shipped"}
# md5 each BBL input had in the _ms982 run (bbl_relaunch / dx pins, verified on the cluster). A
# local file that differs changes the BBL for a reason other than stage C: refused unless
# --accept-input-change NAME.
MS982_MD5 = {
    "forward_rf_qoq.csv": "e544d34079c60c8169f0995faa171873",
    "forward_rf_vintages.csv": "13f321168f7bbd6e29f76b2b3f6e3b37",
    "bbl_transitions.json": "621c621f40f9f615fa0864259bc4f60d",
    "sleep_link_E3_spec_12.json": "14b73a495175b282226a23e14713649b",
    "sleep_link_E4_spec_12.json": "ef594035f1d0b21a04e878415265be3b",
    "market_panel.parquet": "6613228b2fe6b0ba95d0f16d1d5a743c",
}

# ── CODE: the entry points of the chain ────────────────────────────────────────────────────
ENTRIES = [
    "cluster_lib.sh", "env_job.sh", "cluster_preflight.sh", "sleep_run.sh", "sleep_job.sh",
    "logit_job.sh", "blp_draws_job.sh", "blp_run.sh", "blp_stage_job.sh", "bbl_run.sh",
    "bbl_job.sh", "bbl_status.sh", "bbl_cancel.sh", "bbl_sizing.sh", "cluster_archive.sh",
    "bbl_discount.env", "Project.toml", "Manifest.toml", "config/routines.toml",
    "config/table_notes.toml", "cluster/upload_manifest.txt", "cluster/RUNBOOK.md",
    "cluster/LAUNCH_SEQUENCE.txt", "BBL_RUNBOOK.md",
    # Python and Julia the job scripts call by name (also found by the literal scan; listed so a
    # change in a job script's spelling cannot drop them silently)
    "sleep_pipeline.py", "sleep_demand_prep.py", "sleep_demand_prep_link.py", "sleep_demand_prep_e1.py",
    "sleep_demand_prep_e2.py", "utils/demand_chars.py", "blp_logit.jl", "blp_rc.jl", "blp_draws.jl",
    "blp_build_sysimage.jl", "bbl_fwd_sim.jl", "bbl_solve.py", "bbl_shards.py", "make_bbl_cost_tables.py",
    "cf_demand_eval.jl", "cf_deposit_sim.jl",
    # 10-14 consolidation: the one-process smoke driver and the request file the wrapper sources
    "blp_smoke.jl", "stagec_requests.env",
]
EXECUTED_PY = ["sleep_pipeline.py", "sleep_demand_prep.py", "sleep_demand_prep_link.py",
               "sleep_demand_prep_e1.py", "sleep_demand_prep_e2.py", "bbl_solve.py", "bbl_shards.py",
               "make_bbl_cost_tables.py"]
FORBIDDEN = re.compile(r"(^|/)(demand_repair\.py|panel_demand_repair[^/]*\.py|blp_dx[^/]*|dx_[^/]*\.sh)$")
# Not in the chain and not shipped: the demand weak-IV battery (old instrument names) and its job,
# and the sleepiness battery that imports it. Never followed by the crawl; reported if reached.
EXCLUDE = {"blp_weakiv_job.sh", "blp_weak_iv.py", "blp_cue_linear.py", "blp_moment_reduction.py",
           "blp_diag_degeneracy.py", "sleep_weak_iv.py"}
NEVER_FOLLOW = re.compile(r"^(diagnostics/|scratch/|tests?/)")   # mentioned in comments, never run
TEXT_SUFFIXES = {".sh", ".jl", ".py", ".env", ".toml", ".txt", ".md"}
PIP_TO_CONDA = {"sklearn": "scikit-learn", "yaml": "pyyaml", "bs4": "beautifulsoup4"}
DIST_OF = {"sklearn": "scikit-learn", "yaml": "PyYAML", "bs4": "beautifulsoup4"}

PY_IMPORT = re.compile(r"^\s*(?:from\s+([A-Za-z_][\w.]*)\s+import\s+([\w, ()*]+)|import\s+([A-Za-z_][\w., ]*))", re.M)
JL_INCLUDE = re.compile(r'include\(\s*(?:joinpath\(\s*@__DIR__\s*,\s*)?"([^"]+\.jl)"')
LITERAL = re.compile(r"(?<![\w.-])((?:[\w-]+/)?[\w.-]+\.(?:py|jl|sh|toml|env))(?![\w])")
# In Python and Julia only QUOTED names count (code that builds a command line or a path);
# a bare name in a comment or a docstring is not a call.
QUOTED = re.compile(r"[\"']((?:[\w-]+/)?[\w.-]+\.(?:py|jl|sh|toml|env))[\"']")
OPTIONAL_MODULES = {"of_paths", "bs4"}   # imported inside try/except (utils/paths.py, utils/toon_parser.py); optional


def lf(p: Path) -> bytes:
    b = p.read_bytes()
    return b.replace(b"\r\n", b"\n") if p.suffix in TEXT_SUFFIXES else b


def h(b: bytes, algo="sha256") -> str:
    return hashlib.new(algo, b).hexdigest()


def hfile(p: Path, algo="sha256") -> str:
    x = hashlib.new(algo)
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            x.update(chunk)
    return x.hexdigest()


def repo_file(name: str):
    """A name mentioned in a script -> the repo file it denotes, if any."""
    for cand in (name, name.lstrip("./")):
        p = REPO / cand
        if p.is_file() and not NEVER_FOLLOW.match(cand):
            return cand.replace("\\", "/")
    return None


def crawl(entries, literals=True):
    seen, todo, why = set(), list(entries), {}
    for e in entries:
        why[e] = "entry"
    while todo:
        rel = todo.pop()
        if rel in seen or not (REPO / rel).is_file():
            continue
        seen.add(rel)
        p = REPO / rel
        if p.suffix not in (".py", ".jl", ".sh", ".env"):
            continue
        txt = p.read_text(encoding="utf-8", errors="replace")
        found = []
        if p.suffix == ".py":
            for frm, names, imp in PY_IMPORT.findall(txt):
                mods = [frm] if frm else [m.strip().split(" ")[0] for m in imp.split(",")]
                for mod in mods:
                    found.append(mod.replace(".", "/") + ".py")
                    found.append(mod.replace(".", "/") + "/__init__.py")
                    if mod == "utils" and names:
                        found += [f"utils/{n.strip().split(' ')[0]}.py" for n in names.replace("(", "").replace(")", "").split(",")]
        if p.suffix == ".jl":
            found += JL_INCLUDE.findall(txt)
        if literals:
            if p.suffix in (".sh", ".env"):   # shell: every name on a line that is not a comment
                code_lines = chr(10).join(l for l in txt.splitlines() if not l.lstrip().startswith("#"))
                found += LITERAL.findall(code_lines)
            else:
                found += QUOTED.findall(txt)
        # a module inside a package needs the package's __init__.py
        for f in list(found):
            parent = Path(f).parent
            if str(parent) not in ("", ".") and (REPO / parent / "__init__.py").is_file():
                found.append(f"{parent.as_posix()}/__init__.py")
        for f in found:
            r = repo_file(f)
            if r in EXCLUDE:
                EXCLUDED_REACHED.setdefault(r, rel)
                continue
            if r and r not in seen:
                why.setdefault(r, rel)
                todo.append(r)
    return seen, why


EXCLUDED_REACHED: dict = {}


def third_party(py_files):
    std = set(sys.stdlib_module_names)
    mods = set()
    for rel in py_files:
        txt = (REPO / rel).read_text(encoding="utf-8", errors="replace")
        for frm, _names, imp in PY_IMPORT.findall(txt):
            for mod in ([frm] if frm else [m.strip().split(" ")[0] for m in imp.split(",")]):
                top = mod.split(".")[0]
                if not top or top in std or top == "__future__" or top in OPTIONAL_MODULES:
                    continue
                if (REPO / f"{top}.py").is_file() or (REPO / top).is_dir():
                    continue
                mods.add(top)
    return sorted(mods)


def main() -> int:
    global STAMP, CHARS_NAME
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default=None)
    ap.add_argument("--accept-input-change", nargs="*", default=[])
    ap.add_argument("--stamp", default=None, help="the bundle stamp (default: today). A rebuild of a bundle whose data "
                    "zip is already uploaded keeps that bundle's stamp (the chars file name carries it)")
    ap.add_argument("--reuse-data-zip", default=None, help="a data zip already built (and maybe uploaded): every member "
                    "is checked equal to its current local source, then the zip is COPIED byte for byte (a fresh "
                    "zip would differ: zip headers carry the write time)")
    a = ap.parse_args()
    refuse, notes = [], []
    if a.stamp and a.stamp != STAMP:
        old = CHARS_NAME
        STAMP, CHARS_NAME = a.stamp, f"demand_chars_{a.stamp}.parquet"
        DATA[CHARS_NAME] = DATA.pop(old)

    # ── code ──
    closure, why = crawl(ENTRIES)
    for f in sorted(closure):
        if FORBIDDEN.search(f):
            refuse.append(f"{f}: superseded, must not ship (reached from {why.get(f)})")
    for f in ENTRIES:
        if not (REPO / f).is_file():
            refuse.append(f"entry {f} does not exist in the repo")
    code = {f"scripts/{f}": lf(REPO / f) for f in sorted(closure) if not FORBIDDEN.search(f)}
    code[f"scripts/{WRAPPER.name}"] = lf(WRAPPER)
    # the 1.11 manifest: shipped byte for byte as validated, for the Project.toml it was resolved for
    jl_m = (JL111 / JL_MANIFEST).read_bytes() if (JL111 / JL_MANIFEST).is_file() else b""
    if h(jl_m) != JL_MANIFEST_SHA256:
        refuse.append(f"{JL111 / JL_MANIFEST}: sha256 {h(jl_m)[:12]}..., not the validated {JL_MANIFEST_SHA256[:12]}...")
    if bytes([13]) in jl_m:
        refuse.append(f"{JL_MANIFEST}: carries CR bytes")
    if h(lf(REPO / "Project.toml")) != PROJECT_SHA256:
        refuse.append(f"the repo Project.toml (sha256 {h(lf(REPO / 'Project.toml'))[:12]}...) is not the one {JL_MANIFEST} was resolved for ({PROJECT_SHA256[:12]}...): re-resolve under Julia 1.11 and re-validate")
    code[f"scripts/{JL_MANIFEST}"] = jl_m
    why[JL_MANIFEST] = "jl111/ (resolved and validated under Julia 1.11.4)"
    for m, b in code.items():
        if m.endswith((".sh", ".jl", ".py", ".env")) and b"\r" in b:
            refuse.append(f"{m}: a CR survives LF normalisation")

    # ── python stack ──
    py_closure = [f for f in closure if f.endswith(".py")]
    exec_py, _ = crawl(EXECUTED_PY, literals=False)   # what the executed Python IMPORTS
    tp = third_party([f for f in exec_py if f.endswith(".py")])
    req = sorted(set(tp) | {"numpy", "pandas", "scipy", "pyarrow", "statsmodels", "matplotlib"})
    pins = []
    for m in req:
        try:
            pins.append(f"{PIP_TO_CONDA.get(m, m)}={md.version(DIST_OF.get(m, m))}")
        except md.PackageNotFoundError:
            notes.append(f"python module {m}: no local distribution found; left unpinned")
            pins.append(PIP_TO_CONDA.get(m, m))
    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    # --override-channels --strict-channel-priority (m12/D14): conda-forge only, so a non-interactive
    # create never stops on the default channels' terms-of-service prompt.
    conda_create = (f"module load miniconda && conda create -y -n {CONDA_ENV} --override-channels --strict-channel-priority "
                    f"-c conda-forge python={py} " + " ".join(pins))

    # ── data ──
    data_members, data_sha = {}, {}
    for name, src in DATA.items():
        if not src.is_file():
            refuse.append(f"data/input/{name}: local source missing ({src})"); continue
        b = src.read_bytes(); data_members[f"data/input/{name}"] = b; data_sha[name] = h(b)
        if name in MS982_MD5 and h(b, "md5") != MS982_MD5[name]:
            msg = f"data/input/{name}: local md5 {h(b, 'md5')} differs from the _ms982 run's {MS982_MD5[name]}"
            (notes if name in a.accept_input_change else refuse).append(msg + (" (accepted)" if name in a.accept_input_change else " (--accept-input-change to ship it)"))
    if h(data_members.get("data/input/market_panel.parquet", b""), "md5") != PANEL_MD5:
        refuse.append(f"market_panel.parquet: local md5 {h(data_members.get('data/input/market_panel.parquet', b''), 'md5')}, "
                      f"not the pinned {PANEL_MD5} that utils/demand_chars.apply requires (was the panel rebuilt?)")
    if data_sha.get(CHARS_NAME) != CHARS_SHA256:
        refuse.append(f"demand_chars.parquet: sha256 {data_sha.get(CHARS_NAME)}, not the pinned {CHARS_SHA256}")
    if data_sha.get(SIGMA_NAME) in SIGMA_STALE_SHA256:
        refuse.append(f"{SIGMA_NAME}: STALE ({SIGMA_STALE_SHA256[data_sha[SIGMA_NAME]]}); rebuild it with panel_demographics_sigma.py")
    elif data_sha.get(SIGMA_NAME) != SIGMA_SHA256:
        refuse.append(f"{SIGMA_NAME}: sha256 {data_sha.get(SIGMA_NAME)}, not the pinned {SIGMA_SHA256} (the 10-15 rebuild); "
                      f"a newer rebuild needs SIGMA_SHA256 updated here after the user confirms it")
    if data_sha.get(POLICY_UPLOAD) and data_sha[POLICY_UPLOAD] != POLICY_SHA256:
        refuse.append(f"the policy has sha256 {data_sha[POLICY_UPLOAD]}, not the pinned {POLICY_SHA256}")
    # The prep's own check (utils/demand_chars.check_panel_md5): the side file's schema metadata
    # must record the md5 of the panel the prep reads -- on the cluster, the shipped panel.
    import pyarrow.parquet as _pq
    _meta = _pq.ParquetFile(str(DATA[CHARS_NAME])).schema_arrow.metadata or {}
    _rec = _meta.get(b"demand_chars.market_panel_md5", b"").decode()
    _pan = h(data_members.get("data/input/market_panel.parquet", b""), "md5")
    if _rec != _pan:
        refuse.append(f"demand_chars.parquet records market_panel md5 '{_rec or '(none)'}' in its schema metadata, the shipped panel is {_pan}: "
                      f"the prep refuses it (check_panel_md5). Rebuild it with the current panel_demand_chars.py")
    else:
        notes.append(f"demand_chars.parquet schema metadata: market_panel md5 {_rec} = the shipped panel (check_panel_md5 will pass)")
    if CHARS_REPORT.is_file():
        rep = json.loads(CHARS_REPORT.read_text(encoding="utf-8"))
        pm = rep.get("inputs", {}).get("market_panel.parquet")
        if pm != h(data_members.get("data/input/market_panel.parquet", b""), "md5"):
            refuse.append(f"demand_chars_report.json records market_panel.parquet md5 {pm}, not the shipped panel's")
        if rep.get("output_md5") != h(data_members.get(f"data/input/{CHARS_NAME}", b""), "md5"):
            refuse.append("demand_chars_report.json records another output_md5 than the shipped demand_chars.parquet")
        notes.append(f"demand_chars.parquet: built from market_panel.parquet md5 {pm} (the shipped panel), output md5 {rep.get('output_md5')}")
    else:
        refuse.append(f"no {CHARS_REPORT}: the demand-characteristics file cannot be tied to the panel")

    # ── report ──
    csize = sum(len(b) for b in code.values()); dsize = sum(len(b) for b in data_members.values())
    print(f"CODE: {len(code)} files, {csize/1e6:.1f} MB (closure of {len(ENTRIES)} entries)")
    for m in sorted(code):
        print(f"  {m}   <- {why.get(m[len('scripts/'):], 'wrapper')}")
    print(f"DATA: {len(data_members)} files, {dsize/1e6:.1f} MB (no sleepiness estimate: made on the cluster)")
    for n in sorted(data_members):
        print(f"  {n}  {len(data_members[n]):>12,}  sha256 {data_sha[n[len('data/input/'):]][:12]}")
    for r, by in sorted(EXCLUDED_REACHED.items()):
        print(f"EXCLUDED (not in the chain, not shipped): {r}  (named by {by})")
    print(f"PYTHON ({CONDA_ENV}): imports {', '.join(req)}")
    print(f"  create line: {conda_create}")
    if a.reuse_data_zip:
        rz = Path(a.reuse_data_zip)
        want = {m: h(b) for m, b in data_members.items()}
        if rz.name != f"stagec_data_{STAMP}.zip":
            refuse.append(f"--reuse-data-zip {rz.name}: not stagec_data_{STAMP}.zip (pass --stamp)")
        with zipfile.ZipFile(rz) as zf:
            names = set(zf.namelist())
            if names != set(want):
                refuse.append(f"--reuse-data-zip: members differ (missing {sorted(set(want) - names)}, extra {sorted(names - set(want))})")
            for m in sorted(names & set(want)):
                x = hashlib.sha256()
                with zf.open(m) as fh:
                    for chunk in iter(lambda: fh.read(1 << 24), b""):
                        x.update(chunk)
                if x.hexdigest() != want[m]:
                    refuse.append(f"--reuse-data-zip: {m} differs from its current local source (the data zip cannot stay as uploaded)")
        notes.append(f"--reuse-data-zip {rz}: {len(names)} members, each equal to its current local source; copied byte for byte")
    for n in notes:
        print(f"NOTE: {n}")
    if refuse:
        print("REFUSED:"); [print(f"  {r}") for r in refuse]
        return 1
    if not a.stage:
        print("READY (report only; --stage OUTDIR writes the upload)")
        return 0

    out = Path(a.stage); out.mkdir(parents=True, exist_ok=True)
    targets = [out / f"stagec_code_{STAMP}.zip", out / f"stagec_data_{STAMP}.zip"]
    if any(t.exists() for t in targets):
        print(f"REFUSED: {out} already holds a bundle of {STAMP}"); return 1
    bad_members = [m for m in code if FORBIDDEN.search(m) or Path(m).name in EXCLUDE]
    if bad_members:
        print(f"REFUSED: forbidden or excluded files among the members: {bad_members}"); return 1
    arrival = {m: h(b) for m, b in code.items()}
    arrival.update({m: h(b) for m, b in data_members.items()})
    pins_txt = "\n".join([
        "# stagec_pins_20261012.env -- written by build_stagec_bundle.py; sourced by stagec_suite_20261012.sh",
        f"ARRIVAL_LIST=stagec_arrival_{STAMP}.sha256",
        f"CHARS_NAME={CHARS_NAME}", f"CHARS_SHA256={data_sha[CHARS_NAME]}",
        f"SIGMA_SHA256={data_sha[SIGMA_NAME]}",
        f"POLICY_UPLOAD={POLICY_UPLOAD}",
        f"JL_MANIFEST_SHA256={JL_MANIFEST_SHA256}",
        f'PY_REQ="{", ".join(req)}"',
        f'CONDA_CREATE="{conda_create}"', ""]).encode()
    code[f"scripts/{PINS_NAME}"] = pins_txt
    arrival[f"scripts/{PINS_NAME}"] = h(pins_txt)
    arr_txt = "".join(f"{v}  {m}\n" for m, v in sorted(arrival.items())).encode()
    code[f"scripts/stagec_arrival_{STAMP}.sha256"] = arr_txt
    with zipfile.ZipFile(targets[0], "w", zipfile.ZIP_DEFLATED) as zf:
        for m, b in sorted(code.items()):
            zi = zipfile.ZipInfo(m, date_time=(2026, 10, 13, 0, 0, 0))
            zi.external_attr = (0o755 if m.endswith(".sh") else 0o644) << 16
            zf.writestr(zi, b, compress_type=zipfile.ZIP_DEFLATED)
    if a.reuse_data_zip:
        import shutil
        shutil.copyfile(a.reuse_data_zip, targets[1])
    else:
        with zipfile.ZipFile(targets[1], "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
            for m, b in sorted(data_members.items()):
                zf.writestr(m, b)
    up = [*targets]
    md5_txt = "".join(f"{hfile(p, 'md5')}  {p.name}\n" for p in up)
    (out / f"stagec_upload_{STAMP}.md5").write_text(md5_txt, encoding="utf-8", newline="\n")
    print(f"-> {out}: {len(up)} files to upload, {sum(p.stat().st_size for p in up)/1e9:.2f} GB; md5 list stagec_upload_{STAMP}.md5")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
BLP pipeline runner.

Local / Cluster split
---------------------
  LOCAL  (run on your PC):
    1. python run_blp_pipeline.py --logit
         Runs blp_1_logit.jl: non-RC logit for every routine auto-discovered from the
         demand-prep parquets (spec 12). Writes logit_summary_spec_12.json, the
         logit_delta_E*.{jls,bin} warm-start checkpoints, and the LaTeX tables. Use
         --est N for a single routine.
    2. python run_blp_pipeline.py --draws --R 50  (optional validation)
         R<=500 draws finish in a few minutes locally and let you test the
         dry-run timing before submitting to the cluster.
    3. rsync processed/ESTIMATION_OUTPUT/BLP_RESULTS/logit_delta_E*.bin
            <cluster>:../data/output/
         Transfer delta checkpoints so the cluster sigma stage warm-starts
         from logit delta* instead of log-shares (saves ~50-200 iters/call).

  CLUSTER (submit SLURM jobs):
    4. sbatch submit_blp_1_draws.sh         # R=2000 scrambled Halton draws
    5. bash  submit_blp_2_rc_default.sh     # E5-E8 RC-BLP dependency chains

Modes
-----
  --logit          Run blp_1_logit.jl (non-RC, routines auto-discovered from the
                   demand-prep parquets, spec 12) and build the logit tables. Writes
                   logit_summary_spec_12.json + logit_delta_E{k}_spec_12.{jls,bin}. Use
                   --est N to run one routine (e.g. --est 7). Local.
  --draws          Run blp_1_draws.jl (local R<=500 fine; R=2000 use cluster).
  --estimate       Run blp_1_estimation.jl with pre-computed draws.
  --all            Run logit -> draws -> estimate in sequence (full local test).
  --rc             Run the RC-BLP estimation via blp_2_rc.jl (routines auto-discovered).
                   GPU only — on the cluster pass --hpc; locally use --dry-run on a GPU
                   box. DEFAULT runs E5-E8 (single-index links + +Time; spec 12, extended
                   RC sequence); pick a subset with --est (digits, e.g. --est 5678), or
                   --est all for every routine found. Each E{k} warm-starts from
                   logit_delta_E{k}_spec_12.bin.

Options (used with --draws and --estimate)
------------------------------------------
  --est 12345      Estimation strategies: "1","12","12345","all" (default: "12")
  --spec 12        Spec IDs (default: "12")
  --stage          logit | sigma | full | extended | sequence (default: sequence)
  --R              Simulation draws (default: 2000; use 50 for local testing)
  --seed           RNG seed (default: 42)
  --workers        Julia threads (default: auto)
  --dry-run        Dry-run: time 10 inner iterations and exit
  --local-dir      Override 'processed' data directory
  --tol-inner      Inner contraction tolerance (default: 1e-12)
  --max-inner      Inner contraction max iterations (default: 5000)
  --tol-outer      Outer GMM tolerance (default: 1e-6)
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import subprocess
import pathlib
import sys
import shutil
import argparse
import concurrent.futures
import os



ROOT = pathlib.Path(__file__).resolve().parent
PYTHON_EXE = sys.executable
LOGIT_SCRIPT       = ROOT / "blp_1_logit.jl"
DRAWS_SCRIPT       = ROOT / "blp_1_draws.jl"
ESTIM_SCRIPT       = ROOT / "blp_1_estimation.jl"
RC_SCRIPT          = ROOT / "blp_2_rc.jl"


# ── Julia mode ───────────────────────────────────────────────────────────────

def _find_julia() -> str:
    jl = shutil.which("julia")
    if jl is None:
        print("ERROR: 'julia' not found on PATH.")
        sys.exit(1)
    return jl


JULIA_PACKAGES = [
    "Parquet2", "DataFrames", "Optim", "QuasiMonteCarlo",
    "Distributions", "JSON3", "ArgParse", "SparseArrays",
]


def _ensure_julia_packages(jl_exe: str):
    manifest = ROOT / "Manifest.toml"
    if manifest.exists():
        return
    print("[Julia] No Manifest.toml found — installing packages...")
    pkg_list = '["' + '", "'.join(JULIA_PACKAGES) + '"]'
    script = f'using Pkg; Pkg.activate(raw"{ROOT}"); Pkg.add({pkg_list}); Pkg.instantiate()'
    try:
        subprocess.run([jl_exe, f"--project={ROOT}", "-e", script], check=True)
        print("[Julia] Package installation complete.")
    except subprocess.CalledProcessError as e:
        print(f"[Julia] Package install FAILED (exit {e.returncode}).")
        sys.exit(1)


# ── Mode: logit (non-RC, auto-discovered routines) ────────────────────────────

def run_logit(args):
    """Run the non-RC logit (routines auto-discovered, spec 12).

    Reads the self-contained demand_*_spec_12.parquet files, writes
    logit_summary_spec_12.json + per-routine JLS + delta checkpoints, then builds
    est{id}_spec12_logit.tex tables. blp_1_logit.jl auto-discovers the routine list from
    the parquets, so E7/E8/... are included with no code change.

    By default runs ALL discovered routines in one Julia process (avoids repeated JIT
    warmup). Use --est N to run a single routine, e.g. --est 7 (forwarded to the
    orchestrator's own --est selector).
    """
    jl_exe = _find_julia()
    _ensure_julia_packages(jl_exe)

    threads = f"--threads={args.workers}" if args.workers else "--threads=auto"
    env = os.environ.copy()
    env["PYTHON"] = PYTHON_EXE
    env["JULIA_PYTHONCALL_EXE"] = PYTHON_EXE

    if not LOGIT_SCRIPT.exists():
        print(f"ERROR: {LOGIT_SCRIPT} not found."); sys.exit(1)

    # Single-routine selection: a bare single integer (e.g. "7"). The default "12" and
    # other multi-digit/sentinel values mean "all discovered routines".
    single = None
    if args.est not in ("12", "123", "123456", "12345678", "all"):
        s = args.est.strip()
        if s.isdigit() and len(s) == 1:
            single = int(s)

    cmd = [jl_exe, f"--project={ROOT}", threads, str(LOGIT_SCRIPT)]
    if single is not None:
        cmd += ["--est", str(single)]
        print(f"\n=== Logit (E{single}) | CMD: {' '.join(cmd)} ===")
    else:
        print(f"\n=== Logit (all discovered routines) | CMD: {' '.join(cmd)} ===")
    # blp_1_logit.jl writes the est{id}_spec12_logit.tex tables itself (the LaTeX
    # generator is in-file), so no separate table step.
    subprocess.run(cmd, env=env, check=True)


# ── Mode: draws ───────────────────────────────────────────────────────────────

def run_draws(args):
    jl_exe = _find_julia()
    _ensure_julia_packages(jl_exe)
    if not DRAWS_SCRIPT.exists():
        print(f"ERROR: {DRAWS_SCRIPT} not found."); sys.exit(1)
    cmd = [jl_exe, f"--project={ROOT}", "--threads=auto",
           str(DRAWS_SCRIPT),
           "--R",    str(args.R),
           "--seed", str(args.seed),
           "--spec", args.spec,
           "--estim", "1",  # use E1 as reference panel for key extraction
          ]
    if args.local_dir:
        cmd += ["--local-dir", args.local_dir]
    print(f"\n=== Draws | CMD: {' '.join(cmd)} ===")
    env = os.environ.copy()
    env["PYTHON"] = PYTHON_EXE
    env["JULIA_PYTHONCALL_EXE"] = PYTHON_EXE
    subprocess.run(cmd, env=env, check=True)


# ── Mode: estimate ────────────────────────────────────────────────────────────

def run_estimation(est_id: int, jl_exe: str, args):
    if not ESTIM_SCRIPT.exists():
        print(f"ERROR: {ESTIM_SCRIPT} not found."); return (est_id, False, "missing")
    cmd = [jl_exe, f"--project={ROOT}",
           f"--threads={args.workers}" if args.workers else "--threads=auto",
           str(ESTIM_SCRIPT),
           "--estim",    str(est_id),
           "--spec",     args.spec,
           "--stage",    args.stage,
           "--R",        str(args.R),
           "--seed",     str(args.seed),
           "--tol-inner",str(args.tol_inner),
           "--max-inner",str(args.max_inner),
           "--tol-outer",str(args.tol_outer),
          ]
    if args.dry_run:
        cmd.append("--dry-run")
    if args.local_dir:
        cmd += ["--local-dir", args.local_dir]
    print(f"\n=== Estimation {est_id} | CMD: {' '.join(cmd)} ===")
    env = os.environ.copy()
    env["PYTHON"] = PYTHON_EXE
    env["JULIA_PYTHONCALL_EXE"] = PYTHON_EXE
    try:
        subprocess.run(cmd, env=env, check=True, text=True)
        return (est_id, True, "")
    except subprocess.CalledProcessError as e:
        return (est_id, False, f"Exit {e.returncode}")


def run_estimate_pipeline(args):
    jl_exe = _find_julia()
    _ensure_julia_packages(jl_exe)
    est_str = args.est
    est_ids = list(range(1, 6)) if est_str == "all" else [int(c) for c in est_str if c.isdigit()]
    print(f"=== BLP Estimation | E{est_ids} | stage={args.stage} | R={args.R} ===")
    for eid in est_ids:
        eid_result, success, msg = run_estimation(eid, jl_exe, args)
        if success:
            print(f"[+] E{eid_result} DONE.")
        else:
            print(f"[!] E{eid_result} FAILED: {msg}"); sys.exit(1)
    print("\n=== Estimation Pipeline Complete ===")


# ── Mode: rc (GPU RC-BLP, default E5-E8; routines auto-discovered) ────────────────

def run_rc(args):
    """Run the RC-BLP estimation via blp_2_rc.jl.

    Routines are auto-discovered on the cluster from the demand-prep parquets; E{k}
    reads its parquet (e.g. demand_3_logistic, demand_6_index, demand_7_sijoint) and
    warm-starts from BLP_RESULTS/logit_delta_E{k}_spec_12.bin. Requires a CUDA GPU
    (cluster gpu_h200); locally use --dry-run on a GPU box.

    The DEFAULT run is routines E5-E8 (the single-index links and their +Time variants),
    spec 12, with the full extended random-coefficient sequence. Pick a different subset
    with --est (digits, e.g. "5678"); --est all runs every routine found. Set
    BLP_ENGINE=numerical to use the finite-difference engine instead of IFT.
    """
    jl_exe = _find_julia()
    _ensure_julia_packages(jl_exe)
    if not RC_SCRIPT.exists():
        print(f"ERROR: {RC_SCRIPT} not found."); sys.exit(1)
    # Default (the global --est default "12", or explicit "5678") → E5-E8 (the single-
    # index links + their +Time variants). "all" runs every discovered routine (delegated
    # to --all-routines); otherwise parse the requested digits.
    if args.est in ("12", "5678"):
        est_ids = [5, 6, 7, 8]
    elif args.est == "all":
        est_ids = None  # → --all-routines (driver discovers every parquet on disk)
    else:
        est_ids = [int(c) for c in args.est if c.isdigit()]
        if not est_ids:
            print("ERROR: --rc routines must be digits (use --est, e.g. 5678)."); sys.exit(1)
    # est_ids None → run every discovered routine in one process via --all-routines;
    # otherwise one invocation per requested routine.
    invocations = ([("all", ["--all-routines"])] if est_ids is None
                   else [(eid, ["--estim", str(eid)]) for eid in est_ids])
    label = "all (discovered)" if est_ids is None else f"E{est_ids}"
    print(f"=== RC-BLP | {label} | spec=12 | stage={args.stage} | R={args.R} ===")
    for tag, sel in invocations:
        cmd = [jl_exe, f"--project={ROOT}",
               f"--threads={args.workers}" if args.workers else "--threads=auto",
               str(RC_SCRIPT), *sel,
               "--stage",     args.stage,
               "--R",         str(args.R),
               "--seed",      str(args.seed),
               "--tol-inner", str(args.tol_inner),
               "--max-inner", str(args.max_inner),
               "--tol-outer", str(args.tol_outer),
              ]
        if args.hpc:
            cmd.append("--hpc")
        if args.dry_run:
            cmd.append("--dry-run")
        if args.local_dir:
            cmd += ["--local-dir", args.local_dir]
        print(f"\n=== RC-BLP {tag} | CMD: {' '.join(cmd)} ===")
        env = os.environ.copy()
        env["PYTHON"] = PYTHON_EXE
        env["JULIA_PYTHONCALL_EXE"] = PYTHON_EXE
        try:
            subprocess.run(cmd, env=env, check=True)
            print(f"[+] RC-BLP {tag} DONE.")
        except subprocess.CalledProcessError as e:
            print(f"[!] RC-BLP {tag} FAILED: exit {e.returncode}"); sys.exit(1)
    print("\n=== RC-BLP Estimation Complete ===")


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BLP pipeline")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--logit",    action="store_true", default=False,
                      help="Run blp_1_logit.jl (non-RC, routines auto-discovered, "
                           "spec 12) and generate logit tables. Use --est N for one routine.")
    mode.add_argument("--draws",    action="store_true", default=False,
                      help="Run blp_1_draws.jl to pre-compute simulation draws")
    mode.add_argument("--estimate", action="store_true", default=False,
                      help="Run blp_1_estimation.jl")
    mode.add_argument("--all",      action="store_true", default=False,
                      help="Run logit -> draws -> estimate in sequence")
    mode.add_argument("--rc", action="store_true", default=False,
                      help="Run the RC-BLP estimation via blp_2_rc.jl (GPU; routines "
                           "auto-discovered). Default runs E5-E8 (spec 12); use "
                           "--est 5678 for a subset, --est all for every routine.")

    g = p.add_argument_group("Estimation options")
    g.add_argument("--hpc",        action="store_true", default=False,
                   help="Pass --hpc to the engine (cluster data paths; used with --rc)")
    g.add_argument("--est",        type=str,   default="12")
    g.add_argument("--spec",       type=str,   default="12")
    g.add_argument("--stage",      type=str,   default="sequence",
                   choices=["logit", "sigma", "rc2", "rc3", "rc4",
                            "full", "ext1", "ext2", "extended", "sequence"])
    g.add_argument("--R",          type=int,   default=2000)
    g.add_argument("--seed",       type=int,   default=42)
    g.add_argument("--workers",    type=int,   default=None)
    g.add_argument("--dry-run",    action="store_true", dest="dry_run")
    g.add_argument("--local-dir",  type=str,   default=None, dest="local_dir")
    g.add_argument("--tol-inner",  type=float, default=1e-12, dest="tol_inner")
    g.add_argument("--max-inner",  type=int,   default=5000,  dest="max_inner")
    g.add_argument("--tol-outer",  type=float, default=1e-6,  dest="tol_outer")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.all:
        run_logit(args)
        run_draws(args)
        run_estimate_pipeline(args)
    elif args.logit:
        run_logit(args)
    elif args.draws:
        run_draws(args)
    elif args.estimate:
        run_estimate_pipeline(args)
    elif args.rc:
        run_rc(args)


if __name__ == "__main__":
    main()

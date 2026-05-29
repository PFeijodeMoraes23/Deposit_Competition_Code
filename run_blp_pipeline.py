"""
BLP pipeline runner.

Local / Cluster split
---------------------
  LOCAL  (run on your PC):
    1. python run_blp_pipeline.py --logit
         Runs blp_logit_local.jl: non-RC baseline + saves logit_delta_E*.jls
         checkpoints to BLP_RESULTS/ for delta warm-start on the cluster.
    2. python run_blp_pipeline.py --draws --R 50  (optional validation)
         R<=500 draws finish in a few minutes locally and let you test the
         dry-run timing before submitting to the cluster.
    3. rsync processed/ESTIMATION_OUTPUT/BLP_RESULTS/logit_delta_E*.jls
            <cluster>:../data/output/
         Transfer delta checkpoints so the cluster sigma stage warm-starts
         from logit delta* instead of log-shares (saves ~50-200 iters/call).

  CLUSTER (submit SLURM jobs):
    4. sbatch submit_blp_draws.sh          # R=2000 scrambled Halton draws
    5. bash  submit_blp_chain.sh 1 500     # E1 preliminary (sigma/full/extended)
       bash  submit_blp_chain.sh 1 2000    # E1 production
       (repeat for E2-E5 in parallel)

Modes
-----
  --latex          Build LaTeX table fragments (default).
  --logit          Run blp_logit_local.jl (non-RC, all strategies, ~1 min).
                   Also saves logit_delta_E*.jls delta checkpoints.
  --draws          Run blp_draws.jl (local R<=500 fine; R=2000 use cluster).
  --estimate       Run blp_estimation.jl with pre-computed draws.
  --all            Run logit -> draws -> estimate in sequence (full local test).

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
LOGIT_TABLE_SCRIPT = ROOT / "make_blp_logit_table.py"
LOGIT_SCRIPT       = ROOT / "blp_logit_local.jl"
DRAWS_SCRIPT       = ROOT / "blp_draws.jl"
ESTIM_SCRIPT       = ROOT / "blp_estimation.jl"


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


# ── Mode: logit ───────────────────────────────────────────────────────────────

def run_logit(args):
    jl_exe = _find_julia()
    _ensure_julia_packages(jl_exe)
    if not LOGIT_SCRIPT.exists():
        print(f"ERROR: {LOGIT_SCRIPT} not found."); sys.exit(1)
    cmd = [jl_exe, f"--project={ROOT}",
           f"--threads={args.workers}" if args.workers else "--threads=auto",
           str(LOGIT_SCRIPT)]
    print(f"\n=== Logit (non-RC) | CMD: {' '.join(cmd)} ===")
    env = os.environ.copy()
    env["PYTHON"] = PYTHON_EXE
    env["JULIA_PYTHONCALL_EXE"] = PYTHON_EXE
    subprocess.run(cmd, env=env, check=True)

    # Generate logit table automatically after estimation
    print(f"\n=== Generating Logit Table ===")
    if not LOGIT_TABLE_SCRIPT.exists():
        print(f"WARNING: {LOGIT_TABLE_SCRIPT} not found. Skipping table generation.")
        return
    try:
        subprocess.run([PYTHON_EXE, str(LOGIT_TABLE_SCRIPT)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"WARNING: Table generation failed (exit {e.returncode})")


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


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BLP pipeline")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--logit",    action="store_true", default=False,
                      help="Run blp_logit_local.jl (non-RC, local) and generate logit tables")
    mode.add_argument("--draws",    action="store_true", default=False,
                      help="Run blp_draws.jl to pre-compute simulation draws")
    mode.add_argument("--estimate", action="store_true", default=False,
                      help="Run blp_estimation.jl")
    mode.add_argument("--all",      action="store_true", default=False,
                      help="Run logit -> draws -> estimate in sequence")

    g = p.add_argument_group("Estimation options")
    g.add_argument("--est",        type=str,   default="12")
    g.add_argument("--spec",       type=str,   default="12")
    g.add_argument("--stage",      type=str,   default="sequence",
                   choices=["logit", "sigma", "full", "extended", "sequence"])
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


if __name__ == "__main__":
    main()


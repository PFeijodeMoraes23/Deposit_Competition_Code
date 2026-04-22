"""
BLP pipeline runner.

Modes
-----
  --latex          Build LaTeX tables (default).
  --logit          Run blp_logit_local.jl locally (non-RC, all strategies).
  --draws          Run blp_draws.jl to pre-compute simulation draws.
  --estimate       Run blp_estimation.jl (load draws + full BLP GMM).
  --all            Run logit -> draws -> estimate in sequence.

Options (used with --draws and --estimate)
------------------------------------------
  --est 12345      Estimation strategies: "1","12","12345","all" (default: "12")
  --spec 12        Spec IDs (default: "12")
  --stage          logit | sigma | full | extended | sequence (default: sequence)
  --R              Simulation draws (default: 2000)
  --seed           RNG seed (default: 42)
  --workers        Julia threads (default: auto)
  --dry-run        Dry-run mode
  --local-dir      Override 'processed' data directory
  --tol-inner      Inner contraction tolerance (default: 1e-12)
  --max-inner      Inner contraction max iterations (default: 5000)
  --tol-outer      Outer GMM tolerance (default: 1e-6)

Recommended usage
-----------------
  # Step 1 — logit sanity check (local, fast):
  python run_blp_pipeline.py --logit

  # Step 2 — pre-compute draws (local R=50 or cluster R=2000):
  python run_blp_pipeline.py --draws --R 50 --seed 42

  # Step 3 — estimate (dry-run):
  python run_blp_pipeline.py --estimate --est 1 --spec 12 --stage sigma --R 50 --dry-run

  # Full sequence:
  python run_blp_pipeline.py --all --R 2000 --workers 4
"""

import subprocess
import pathlib
import sys
import shutil
import argparse
import concurrent.futures



ROOT = pathlib.Path(__file__).resolve().parent
PYTHON_EXE = sys.executable
LATEX_SCRIPT   = ROOT / "make_blp_latex_tables.py"
LOGIT_SCRIPT   = ROOT / "blp_logit_local.jl"
DRAWS_SCRIPT   = ROOT / "blp_draws.jl"
ESTIM_SCRIPT   = ROOT / "blp_estimation.jl"

# ── LaTeX mode ───────────────────────────────────────────────────────────────
def run_latex_for_est(est_id: int):
    print(f"Launching LaTeX builder for Estimation {est_id}...")
    cmd = [PYTHON_EXE, str(LATEX_SCRIPT), "--est", str(est_id)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return (est_id, True, res.stdout)
    except subprocess.CalledProcessError as e:
        return (est_id, False, f"Error:\n{e.stderr}\n{e.stdout}")

def run_latex_pipeline():
    routines = [1, 2, 3, 4, 5]
    if not LATEX_SCRIPT.exists():
        print(f"Cannot find script at: {LATEX_SCRIPT}")
        sys.exit(1)

    print(f"=== Beginning Parallel BLP LaTeX Pipeline for Routines {routines} ===")
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(routines)) as executor:
        futures = {executor.submit(run_latex_for_est, r): r for r in routines}
        for future in concurrent.futures.as_completed(futures):
            est_id, success, output = future.result()
            print(f"\n--- Output from Estimation {est_id} ---")
            print(output.strip())
            if not success:
                print(f"[!] Estimation {est_id} LaTeX generation FAILED.")
            else:
                print(f"[+] Estimation {est_id} LaTeX generation SUCCESSFUL.")
    print("\n=== BLP LaTeX Pipeline Complete ===")


# ── Python mode ──────────────────────────────────────────────────────────────

def run_python_estimation(est_id: int, args: argparse.Namespace):
    script = PYTHON_SCRIPTS[est_id]
    if not script.exists():
        print(f"ERROR: Python script not found: {script}")
        return (est_id, False, "Script missing")

    cmd = [
        PYTHON_EXE,
        str(script),
        "--spec",      args.spec,
        "--stage",     args.stage,
        "--R",         str(args.R),
        "--seed",      str(args.seed),
        "--tol-inner", str(args.tol_inner),
        "--max-inner", str(args.max_inner),
        "--tol-outer", str(args.tol_outer),
    ]
    if args.workers:
        cmd += ["--workers", str(args.workers)]
    if args.chunk_size:
        cmd += ["--chunk-size", str(args.chunk_size)]
    if args.dry_run:
        cmd.append("--dry-run")
    if args.local_dir:
        cmd += ["--local-dir", args.local_dir]
    if est_id == 5 and args.alt:
        cmd += ["--alt", args.alt]

    print(f"\n=== Estimation {est_id} | Running Python ===")
    print(f"  CMD: {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, text=True, check=True)
        return (est_id, True, "")
    except subprocess.CalledProcessError as e:
        return (est_id, False, f"Exit code {e.returncode}")


def run_python_pipeline(args: argparse.Namespace):
    est_str = args.est
    if est_str == "all":
        est_ids = sorted(PYTHON_SCRIPTS.keys())
    else:
        est_ids = [int(c) for c in est_str if c.isdigit()]
    est_ids = [e for e in est_ids if e in PYTHON_SCRIPTS]

    if not est_ids:
        print("ERROR: No valid estimation IDs. Available: 1-5.")
        sys.exit(1)

    print(f"=== Python Local BLP Pipeline | Estimations {est_ids} ===")
    for eid in est_ids:
        eid_result, success, msg = run_python_estimation(eid, args)
        if success:
            print(f"[+] Estimation {eid_result} DONE.")
        else:
            print(f"[!] Estimation {eid_result} FAILED: {msg}")
            sys.exit(1)
    print("\n=== Python Pipeline Complete ===")


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
    subprocess.run(cmd, check=True)


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
    subprocess.run(cmd, check=True)


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
    try:
        subprocess.run(cmd, check=True, text=True)
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
    mode.add_argument("--latex",    action="store_true", default=False)
    mode.add_argument("--logit",    action="store_true", default=False,
                      help="Run blp_logit_local.jl (non-RC, local)")
    mode.add_argument("--draws",    action="store_true", default=False,
                      help="Run blp_draws.jl to pre-compute simulation draws")
    mode.add_argument("--estimate", action="store_true", default=False,
                      help="Run blp_estimation.jl")
    mode.add_argument("--all",      action="store_true", default=False,
                      help="Run logit -> draws -> estimate in sequence")
    mode.add_argument("--julia-then-latex", action="store_true", default=False,
                      dest="julia_then_latex",
                      help="Run estimate then build LaTeX tables")
    mode.add_argument("--logit-then-latex", action="store_true", default=False,
                      dest="logit_then_latex",
                      help="Run blp_logit_local.jl then build LaTeX tables from available results")

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
    elif args.julia_then_latex:
        run_estimate_pipeline(args)
        run_latex_pipeline()
    elif args.logit_then_latex:
        run_logit(args)
        run_latex_pipeline()
    else:
        run_latex_pipeline()


if __name__ == "__main__":
    main()


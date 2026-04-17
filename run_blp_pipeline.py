"""
BLP pipeline runner.

Modes
-----
  --latex          Build LaTeX tables for all 5 estimation routines (default).
  --julia          Run Julia estimation scripts locally (all 5 routines).

Julia-specific flags (used with --julia)
----------------------------------------
  --est 12345      Estimation routines to run, e.g. "1", "12", "12345", "all" (default: "12")
  --spec 12        Spec IDs forwarded to each Julia script (default: "12")
  --stage          logit | sigma | full | extended | sequence (default: sequence)
  --R              Number of simulation draws (default: 100)
  --seed           RNG seed (default: 42)
  --workers        Julia threads (default: system nthreads)
  --chunk-size     Chunk allocation size; 0 = unlimited (default: 0)
  --dry-run        Dry-run mode (no optimisation, just load + sanity check)
  --local-dir      Override 'processed' data directory
  --tol-inner      Inner contraction tolerance (default: 1e-12)
  --max-inner      Inner contraction max iterations (default: 2000)
  --tol-outer      Outer GMM tolerance (default: 1e-6)
  --alt            Estimation 5 only: alt1 | alt2 | alt2linear | alt2logistic | all (default: alt2)

Recommended usage (16 GB RAM local machine)
--------------------------------------------
  # Full run — all 5 routines, R=2000, chunked to fit in memory:
  python run_blp_pipeline.py --julia --est 12345 --spec 12 --R 2000 --chunk-size 200 --workers 4 --stage sequence

  # Dry-run sanity check — single routine, single spec:
  python run_blp_pipeline.py --julia --est 1 --spec 1 --R 2000 --chunk-size 200 --workers 4 --dry-run
"""
import subprocess
import pathlib
import sys
import shutil
import argparse
import concurrent.futures

ROOT = pathlib.Path(__file__).resolve().parent
PYTHON_EXE = sys.executable
LATEX_SCRIPT = ROOT / "make_blp_latex_tables.py"

JULIA_SCRIPTS = {
    1: ROOT / "estimation_1_demand_2_loop_ju.jl",
    2: ROOT / "estimation_2_demand_2_loop_ju.jl",
    3: ROOT / "estimation_3_demand_2_loop_ju.jl",
    4: ROOT / "estimation_4_demand_2_loop_ju.jl",
    5: ROOT / "estimation_5_demand_2_loop_ju.jl",
}

PYTHON_SCRIPTS = {
    1: ROOT / "estimation_1_demand_2_loop.py",
    2: ROOT / "estimation_2_demand_2_loop.py",
    3: ROOT / "estimation_3_demand_2_loop.py",
    4: ROOT / "estimation_4_demand_2_loop.py",
    5: ROOT / "estimation_5_demand_2_loop.py",
}

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
    "Distributions", "JSON3", "ArgParse",
]


def _ensure_julia_packages(jl_exe: str):
    """Install required Julia packages into the repo-local project env if not already present."""
    manifest = ROOT / "Manifest.toml"
    if manifest.exists():
        return  # already instantiated
    print("[Julia] No Manifest.toml found — installing packages into project env...")
    pkg_list = '["' + '", "'.join(JULIA_PACKAGES) + '"]'
    script = f'using Pkg; Pkg.activate(raw"{ROOT}"); Pkg.add({pkg_list}); Pkg.instantiate()'
    try:
        subprocess.run(
            [jl_exe, f"--project={ROOT}", "-e", script],
            check=True,
        )
        print("[Julia] Package installation complete.")
    except subprocess.CalledProcessError as e:
        print(f"[Julia] Package installation FAILED (exit {e.returncode}). "
              "Run manually: julia --project=<repo_root> -e 'using Pkg; Pkg.instantiate()'")
        sys.exit(1)

def run_julia_estimation(est_id: int, jl_exe: str, args: argparse.Namespace):
    script = JULIA_SCRIPTS[est_id]
    if not script.exists():
        print(f"ERROR: Julia script not found: {script}")
        return (est_id, False, "Script missing")

    cmd = [
        jl_exe,
        f"--project={ROOT}",
        f"--threads={args.workers}" if args.workers else "--threads=auto",
        str(script),
        "--spec",      args.spec,
        "--stage",     args.stage,
        "--R",         str(args.R),
        "--seed",      str(args.seed),
        "--tol-inner", str(args.tol_inner),
        "--max-inner", str(args.max_inner),
        "--tol-outer", str(args.tol_outer),
    ]
    if args.chunk_size:
        cmd += ["--chunk-size", str(args.chunk_size)]
    if args.dry_run:
        cmd.append("--dry-run")
    if args.local_dir:
        cmd += ["--local-dir", args.local_dir]
    if est_id == 5 and args.alt:
        cmd += ["--alt", args.alt]

    print(f"\n=== Estimation {est_id} | Running Julia ===")
    print(f"  CMD: {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, text=True, check=True)
        return (est_id, True, "")
    except subprocess.CalledProcessError as e:
        return (est_id, False, f"Exit code {e.returncode}")

def run_julia_pipeline(args: argparse.Namespace):
    jl_exe = _find_julia()

    # Determine which estimation routines to run
    est_str = args.est
    if est_str == "all":
        est_ids = sorted(JULIA_SCRIPTS.keys())
    else:
        est_ids = [int(c) for c in est_str if c.isdigit()]
    est_ids = [e for e in est_ids if e in JULIA_SCRIPTS]

    if not est_ids:
        print("ERROR: No valid estimation IDs. Available: 1-5.")
        sys.exit(1)

    _ensure_julia_packages(jl_exe)

    print(f"=== Julia Local BLP Pipeline | Estimations {est_ids} ===")
    for eid in est_ids:
        eid_result, success, msg = run_julia_estimation(eid, jl_exe, args)
        if success:
            print(f"[+] Estimation {eid_result} DONE.")
        else:
            print(f"[!] Estimation {eid_result} FAILED: {msg}")
            sys.exit(1)
    print("\n=== Julia Pipeline Complete ===")


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BLP pipeline: LaTeX tables or local Julia estimation")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--latex", action="store_true", default=False,
                      help="Build LaTeX result tables (default if no mode given)")
    mode.add_argument("--julia", action="store_true", default=False,
                      help="Run Julia BLP estimation locally")
    mode.add_argument("--python", action="store_true", default=False,
                      help="Run Python BLP estimation locally")
    mode.add_argument("--julia-then-latex", action="store_true", default=False,
                      dest="julia_then_latex",
                      help="Run Julia estimation then build LaTeX tables")

    # Julia-specific args (ignored in --latex mode)
    jg = p.add_argument_group("Julia estimation options")
    jg.add_argument("--est",        type=str,   default="12",    help='Estimation routines: "1","12","12345","all" (default: 12)')
    jg.add_argument("--spec",       type=str,   default="12",    help='Spec IDs forwarded to Julia: "1","12","all" (default: 12)')
    jg.add_argument("--stage",      type=str,   default="sequence",
                    choices=["logit", "sigma", "full", "extended", "sequence"])
    jg.add_argument("--R",          type=int,   default=100)
    jg.add_argument("--seed",       type=int,   default=42)
    jg.add_argument("--workers",    type=int,   default=None,    help="Julia threads (default: auto)")
    jg.add_argument("--chunk-size", type=int,   default=0,       dest="chunk_size")
    jg.add_argument("--dry-run",    action="store_true",         dest="dry_run")
    jg.add_argument("--local-dir",  type=str,   default=None,    dest="local_dir",
                    help='Override path to "processed" directory')
    jg.add_argument("--tol-inner",  type=float, default=1e-9,    dest="tol_inner")
    jg.add_argument("--max-inner",  type=int,   default=5000,    dest="max_inner")
    jg.add_argument("--tol-outer",  type=float, default=1e-6,    dest="tol_outer")
    jg.add_argument("--alt",        type=str,   default="alt2",
                    help='Est 5 only: alt1|alt2|alt2linear|alt2logistic|all (default: alt2)')
    return p

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.julia_then_latex:
        run_julia_pipeline(args)
        run_latex_pipeline()
    elif args.julia:
        run_julia_pipeline(args)
    elif args.python:
        run_python_pipeline(args)
    else:
        # Default: LaTeX mode (whether --latex given or not)
        run_latex_pipeline()

if __name__ == "__main__":
    main()

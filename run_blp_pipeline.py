"""
BLP pipeline runner.

Local / Cluster split
---------------------
  LOCAL  (run on your PC):
    1. python run_blp_pipeline.py --logit
         Runs blp_1_logit_local.jl: non-RC baseline + saves logit_delta_E*.jls
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
  --logit          Run blp_1_logit_local.jl (non-RC, legacy 5 strategies, ~1 min).
                   Also saves logit_delta_E*.jls delta checkpoints.
  --logit-coherence  Run the post-coherence-fix non-RC logit (3 routines E1/E2/E3,
                   spec 12) via blp_1_logit_coherence.jl, then build *_coherence
                   tables. Reads demand_{1,2,3[_logistic]}_spec_12.parquet; writes
                   logit_summary_spec_12_coherence.json + logit_delta_E{k}_spec_12_
                   coherence.{jls,bin}. Use --est {1,2,3} to run one routine. Local.
  --draws          Run blp_1_draws.jl (local R<=500 fine; R=2000 use cluster).
  --estimate       Run blp_1_estimation.jl with pre-computed draws.
  --all            Run logit -> draws -> estimate in sequence (full local test).
  --coherence      Run the post-fix RC-BLP for the 3 coherence routines (E1/E2/E3,
                   spec 12) via blp_coherence_orchestrator.jl. GPU only — on the
                   cluster pass --hpc; locally use --dry-run on a GPU box. Pick
                   routines with --est (e.g. --est 13); default runs all three.
                   Each E{k} warm-starts from logit_delta_E{k}_spec_12.bin.

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
LOGIT_SCRIPT       = ROOT / "blp_1_logit_local.jl"
DRAWS_SCRIPT       = ROOT / "blp_1_draws.jl"
ESTIM_SCRIPT       = ROOT / "blp_1_estimation.jl"
COHERENCE_SCRIPT   = ROOT / "blp_2_rc_coherence.jl"

# Coherence (post-coherence-fix) LOGIT: non-RC 3-routine orchestrator + per-routine
# entrypoints. Distinct from COHERENCE_SCRIPT above (that is the GPU RC-BLP estimation).
LOGIT_COH_ORCH     = ROOT / "blp_1_logit_coherence.jl"
LOGIT_COH_BY_EST   = {
    1: ROOT / "blp_1_logit_e1_coherence.jl",
    2: ROOT / "blp_1_logit_e2_coherence.jl",
    3: ROOT / "blp_1_logit_e3_coherence.jl",
}


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


# ── Mode: logit-coherence (non-RC, post-coherence-fix, 3 routines) ─────────────

def run_logit_coherence(args):
    """Run the post-coherence-fix non-RC logit (3 routines E1/E2/E3, spec 12).

    Reads the new self-contained demand_{1,2,3[_logistic]}_spec_12.parquet files,
    writes logit_summary_spec_12_coherence.json + per-routine JLS + coherence delta
    checkpoints, then builds est{id}_spec12_logit_coherence.tex tables.

    By default runs all three routines via the orchestrator (single Julia process —
    avoids 3x JIT warmup). Use --est with a single digit in {1,2,3} to run just one
    routine via its thin entrypoint.
    """
    jl_exe = _find_julia()
    _ensure_julia_packages(jl_exe)

    threads = f"--threads={args.workers}" if args.workers else "--threads=auto"
    env = os.environ.copy()
    env["PYTHON"] = PYTHON_EXE
    env["JULIA_PYTHONCALL_EXE"] = PYTHON_EXE

    # Decide single-routine vs orchestrator. --est default is "12"; treat that and
    # "123"/"all" as "all three via orchestrator".
    single = None
    if args.est not in ("12", "123", "all"):
        digits = [int(c) for c in args.est if c in "123"]
        if len(digits) == 1:
            single = digits[0]
        elif not digits:
            print("ERROR: --logit-coherence --est must be among {1,2,3}."); sys.exit(1)

    if single is not None:
        script = LOGIT_COH_BY_EST[single]
        if not script.exists():
            print(f"ERROR: {script} not found."); sys.exit(1)
        cmd = [jl_exe, f"--project={ROOT}", threads, str(script)]
        print(f"\n=== Logit Coherence (E{single}) | CMD: {' '.join(cmd)} ===")
    else:
        if not LOGIT_COH_ORCH.exists():
            print(f"ERROR: {LOGIT_COH_ORCH} not found."); sys.exit(1)
        cmd = [jl_exe, f"--project={ROOT}", threads, str(LOGIT_COH_ORCH)]
        print(f"\n=== Logit Coherence (E1/E2/E3) | CMD: {' '.join(cmd)} ===")
    subprocess.run(cmd, env=env, check=True)

    # Generate coherence logit tables automatically after estimation.
    print(f"\n=== Generating Logit Coherence Tables ===")
    if not LOGIT_TABLE_SCRIPT.exists():
        print(f"WARNING: {LOGIT_TABLE_SCRIPT} not found. Skipping table generation.")
        return
    table_cmd = [PYTHON_EXE, str(LOGIT_TABLE_SCRIPT), "--coherence"]
    if single is not None:
        table_cmd += ["--est", str(single)]
    try:
        subprocess.run(table_cmd, check=True)
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


# ── Mode: coherence (GPU RC-BLP, 3 routines) ──────────────────────────────────

def run_coherence(args):
    """Run the post-fix 'coherence' RC-BLP estimation via blp_coherence_orchestrator.jl.

    Routines E1/E2/E3 (spec 12 only): E{k} reads demand_{k}_final_spec_12.parquet and
    warm-starts from BLP_RESULTS/logit_delta_E{k}_spec_12.bin. Requires a CUDA GPU
    (cluster gpu_h200); locally use --dry-run on a GPU box. Pick routines with --est
    (digits among 1,2,3; default runs all three). Set BLP_COHERENCE_ENGINE=numerical
    to use the finite-difference engine instead of IFT.
    """
    jl_exe = _find_julia()
    _ensure_julia_packages(jl_exe)
    if not COHERENCE_SCRIPT.exists():
        print(f"ERROR: {COHERENCE_SCRIPT} not found."); sys.exit(1)
    # Default to all three coherence routines; --est selects a subset.
    if args.est in ("12", "123", "all"):
        est_ids = [1, 2, 3]
    else:
        est_ids = [int(c) for c in args.est if c in "123"]
    if not est_ids:
        print("ERROR: --coherence routines must be among {1,2,3} (use --est)."); sys.exit(1)
    print(f"=== Coherence RC-BLP | E{est_ids} | spec=12 | stage={args.stage} | R={args.R} ===")
    for eid in est_ids:
        cmd = [jl_exe, f"--project={ROOT}",
               f"--threads={args.workers}" if args.workers else "--threads=auto",
               str(COHERENCE_SCRIPT),
               "--estim",     str(eid),
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
        print(f"\n=== Coherence E{eid} | CMD: {' '.join(cmd)} ===")
        env = os.environ.copy()
        env["PYTHON"] = PYTHON_EXE
        env["JULIA_PYTHONCALL_EXE"] = PYTHON_EXE
        try:
            subprocess.run(cmd, env=env, check=True)
            print(f"[+] Coherence E{eid} DONE.")
        except subprocess.CalledProcessError as e:
            print(f"[!] Coherence E{eid} FAILED: exit {e.returncode}"); sys.exit(1)
    print("\n=== Coherence Estimation Complete ===")


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BLP pipeline")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--logit",    action="store_true", default=False,
                      help="Run blp_1_logit_local.jl (non-RC, local, legacy 5 routines) and generate logit tables")
    mode.add_argument("--logit-coherence", action="store_true", default=False, dest="logit_coherence",
                      help="Run the post-coherence-fix non-RC logit (3 routines E1/E2/E3, spec 12) "
                           "and generate *_coherence logit tables. Use --est {1,2,3} for one routine.")
    mode.add_argument("--draws",    action="store_true", default=False,
                      help="Run blp_1_draws.jl to pre-compute simulation draws")
    mode.add_argument("--estimate", action="store_true", default=False,
                      help="Run blp_1_estimation.jl")
    mode.add_argument("--all",      action="store_true", default=False,
                      help="Run logit -> draws -> estimate in sequence")
    mode.add_argument("--coherence", action="store_true", default=False,
                      help="Run the post-fix RC-BLP for the 3 coherence routines "
                           "(E1/E2/E3, spec 12) via blp_coherence_orchestrator.jl (GPU)")

    g = p.add_argument_group("Estimation options")
    g.add_argument("--hpc",        action="store_true", default=False,
                   help="Pass --hpc to the engine (cluster data paths; used with --coherence)")
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
    elif args.logit_coherence:
        run_logit_coherence(args)
    elif args.draws:
        run_draws(args)
    elif args.estimate:
        run_estimate_pipeline(args)
    elif args.coherence:
        run_coherence(args)


if __name__ == "__main__":
    main()


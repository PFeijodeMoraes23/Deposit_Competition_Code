"""
run_data_pipeline.py
====================
Master runner for the full Brazilian Open Finance data pipeline.

Executes every download / processing / panel-building script in the correct
dependency order.  Each step is a separate Python script run as a subprocess
so side-effects (imports, large DataFrames) never bleed between steps.

Pipeline stages
---------------
  Stage 0 - Raw data downloads
    0a. if_data_scrape_1.py          ESTBAN monthly files + IF Data via Olinda API

  Stage 1 - IBGE demographics
    1a. ibge_demographics_panel.py   Municipal population, GDP, age structure -> MCA panel

  Stage 2 - Market-characteristic panels  (independent; can be run in any order)
    2a. scrape_pix_panel.py          Process existing BCB PIX municipality files -> MCA panel
    2b. scrape_anatel.py             Download ANATEL mobile connections -> MCA panel
    2d. scrape_bcb_inclusion.py      BCB banking access-points (branches, correspondents) -> MCA panel
    2e. scrape_cadunico.py           CadUnico low-income families -> MCA panel

  Stage 3 - Deposit panel
    3a. deposits_panel_build.py      ESTBAN + IF Data -> conglomerate x municipality x quarter

  Stage 4 - Master analysis panel
    4a. build_market_panel.py        All MCA panels + deposit panel -> single merged dataset

Usage
-----
  python run_data_pipeline.py                   # run all stages
  python run_data_pipeline.py --from 2          # restart from stage 2 onward
  python run_data_pipeline.py --only 3          # run only stage 3
  python run_data_pipeline.py --skip 2b,2c      # skip specific sub-steps

Notes
-----
  * If any step exits with a non-zero return code the pipeline halts and prints
    the failed step.  Fix the issue and rerun with --from <stage> to resume.
  * Stage 2 sub-steps (2a-2e) are independent and could in principle be
    parallelized; they are serialized here for simplicity and to avoid hitting
    rate limits on BCB / ANATEL / SAGI APIs simultaneously.
  * Stage 4 requires ALL panels to exist.  If some stage-2 downloads failed
    (data source unavailable), build_market_panel.py handles missing files
    gracefully (those columns will be NaN).
"""

import argparse
import concurrent.futures
import os
os.environ["PYTHONWARNINGS"] = "ignore"
import subprocess
import sys
import threading
import time
from pathlib import Path
try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None

if ensure_project_venv is not None:
    ensure_project_venv(__file__)



try:
    from utils.toon_parser import dump_context_json, get_script_config, load_default_toon_context
except Exception:
    dump_context_json = None
    get_script_config = None
    load_default_toon_context = None

# Force the console encoding to match the terminal (CP1252 / Latin-1 on Windows)
# so that printed characters are never garbled.  All print strings use plain ASCII.
# line_buffering=True ensures output is flushed after every newline even when
# stdout/stderr are connected to a pipe (e.g. Tee-Object), so crash messages are
# never lost due to a full-buffer that was never flushed.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="latin-1", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="latin-1", errors="replace", line_buffering=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON     = sys.executable


def _load_toon_runtime_context() -> dict:
    """Load optional TOON runtime context from local Gemini exports."""
    if load_default_toon_context is None:
        return {}

    try:
        return load_default_toon_context(Path(SCRIPT_DIR) / "utils", quiet=True)
    except Exception:
        return {}

# -- Pipeline definition --------------------------------------------------------
# Each entry: (stage_int, step_id_str, script_filename, description)
STEPS = [
    # Stage 0 -- raw downloads
    (0, "0a", "if_data_scrape_1.py",
     "ESTBAN monthly files + IF Data (Olinda API)"),

    # Stage 1 -- IBGE demographics
    (1, "1a", "ibge_demographics_panel.py",
     "IBGE population, GDP, age structure -> MCA demographics panel"),

    # Stage 2 -- market characteristic panels
    (2, "2a", "scrape_pix_panel.py",
     "Process BCB PIX municipality files -> MCA PIX adoption panel"),
    (2, "2b", "scrape_anatel.py",
     "Download ANATEL mobile connections -> MCA connectivity panel"),
    (2, "2d", "scrape_bcb_inclusion.py",
     "BCB banking access-points (branches + correspondents) -> MCA inclusion panel"),
    (2, "2e", "scrape_cadunico.py",
     "CadUnico low-income families -> MCA poverty panel"),
    (2, "2f", "tarifas_scrape_1.py",
     "BCB bank fee schedules (PF + PJ) -> tarifas conglomerate panel + fee summary"),
    (3, "3a", "deposits_panel_build.py",
     "ESTBAN + IF Data -> conglomerate x municipality x quarter deposit panel"),
    (3, "3b", "append_rates.py",
     "Compute and append deposit rates/spreads (COSIF + SGS) to deposit panel"),
    (3, "3c", "bank_chars_panel_build.py",
     "IF Data -> conglomerate x quarter bank size and solvency characteristics panel"),

    # Stage 4 -- master analysis panel
    (4, "4a", "build_market_panel.py",
     "Merge all MCA panels + deposit panel -> master analysis dataset"),
]


# -- Runner helpers -------------------------------------------------------------

# Registry of currently-running subprocesses; populated by run_step so that
# run_wave can kill sibling processes the moment any one step fails.
_running_procs: dict[str, subprocess.Popen] = {}
_running_procs_lock = threading.Lock()


def _kill_running_procs(exclude_sid: str | None = None) -> None:
    """Kill all registered pipeline subprocesses except `exclude_sid`."""
    with _running_procs_lock:
        for sid, proc in list(_running_procs.items()):
            if sid == exclude_sid:
                continue
            try:
                proc.kill()
            except Exception:
                pass

_W = 72   # display width

def _banner(text: str, char: str = "-") -> None:
    print(char * _W)
    print(f"  {text}")
    print(char * _W)


_print_lock = threading.Lock()  # serialize console output across parallel workers


def run_step(step_id: str, script: str, description: str) -> float:
    """
    Run one pipeline script as a subprocess, capturing its output so that
    parallel runs don't interleave on the console.  Raises RuntimeError on
    failure (safe to use inside ThreadPoolExecutor worker threads).
    """
    path = os.path.join(SCRIPT_DIR, script)
    if not os.path.exists(path):
        with _print_lock:
            print(f"  [SKIP] {step_id} -- script not found: {script}", flush=True)
        return 0.0

    with _print_lock:
        print(f"  > [{step_id}] starting: {description}", flush=True)

    t0   = time.perf_counter()
    env = os.environ.copy()
    toon_ctx_path = os.environ.get("TOON_CONTEXT_PATH", "").strip()
    if toon_ctx_path:
        env["TOON_CONTEXT_PATH"] = toon_ctx_path

    proc = subprocess.Popen(
        [PYTHON, path], cwd=SCRIPT_DIR,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=env,
    )
    with _running_procs_lock:
        _running_procs[step_id] = proc
    try:
        raw_out, raw_err = proc.communicate()
    finally:
        with _running_procs_lock:
            _running_procs.pop(step_id, None)
    elapsed     = time.perf_counter() - t0
    stdout_text = raw_out.decode("utf-8", errors="replace")
    stderr_text = raw_err.decode("utf-8", errors="replace")
    returncode  = proc.returncode

    # Print buffered output atomically so concurrent steps don't interleave
    with _print_lock:
        print(f"\n{'-' * _W}")
        print(f"  [{step_id}]  {description}  ({elapsed:.1f}s)")
        print(f"{'-' * _W}")
        if stdout_text:
            print(stdout_text, end="")
        if stderr_text:
            print(stderr_text, end="", file=sys.stderr)
        if returncode != 0:
            print(f"\n  {'!' * (_W - 2)}")
            print(f"  FAILED  [{step_id}] {script}  (exit code {returncode})")
            print(f"  {'!' * (_W - 2)}")
        print(flush=True)

    if returncode != 0:
        raise RuntimeError(
            f"[{step_id}] {script} failed (exit code {returncode})."
        )

    return elapsed


# -- Execution waves -----------------------------------------------------------
# Each inner list runs concurrently; waves are sequential.
# Dependency rationale:
#   Wave 1 -- 0a alone: downloads ESTBAN + IF Data raw files from BCB Olinda.
#             1a is NOT here because it takes ~700s and would block Wave 2 from
#             starting until IBGE finishes. 1a has no dependency on 0a.
#   Wave 2 -- After 0a completes: all characteristic panels run in parallel.
#             1a=IBGE SIDRA (long, ~700s but independent of 0a),
#             2a=PIX (local files), 2b=ANATEL, 2d=BCB Olinda inclusion,
#             2e=SAGI CadUnico, 2f=BCB tarifas, 3a=deposits (reads 0a output).
#             All are mutually independent -> run in parallel.
#   Wave 3 -- 3b (deposit rates) runs after 3a constructs the deposit panel.
#   Wave 4 -- 4a (master merge) needs everything above -> serial.
WAVES: list[list[str]] = [
    ["0a"],                                     # Wave 1: ESTBAN + IF Data raw download
    ["1a", "2a", "2b", "2d", "2e", "2f", "3a"], # Wave 2: all characteristic panels + deposits
    ["3b", "3c"],                               # Wave 3: deposit rates/spreads + bank chars (parallel)
    ["4a"],                                     # Wave 4: master merge
]

def run_wave(
    step_ids:   list[str],
    steps_dict: dict[str, tuple[str, str]],
    skip_set:   set[str],
) -> dict[str, float]:
    """
    Run a group of pipeline steps concurrently using a thread pool.
    Steps in `skip_set` are silently omitted.  Returns {label: elapsed_s}.
    If any step fails the error is collected and sys.exit(1) is called after
    all futures complete (so remaining steps still finish printing).
    """
    active = [
        (sid, steps_dict[sid][0], steps_dict[sid][1])
        for sid in step_ids
        if sid in steps_dict and sid not in skip_set
    ]
    if not active:
        return {}

    timings: dict[str, float] = {}

    if len(active) == 1:
        sid, script, desc = active[0]
        with _print_lock:
            print(f"\n  Wave (serial): [{sid}]", flush=True)
        try:
            timings[f"[{sid}] {script}"] = run_step(sid, script, desc)
        except RuntimeError as exc:
            with _print_lock:
                print(f"\n  *** {exc} ***", file=sys.stderr, flush=True)
            sys.exit(1)
        return timings

    with _print_lock:
        ids_str = ", ".join(sid for sid, *_ in active)
        print(f"\n  Wave (parallel x{len(active)}): {ids_str}", flush=True)

    errors: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(active)) as pool:
        future_map = {
            pool.submit(run_step, sid, script, desc): (sid, script)
            for sid, script, desc in active
        }
        for future in concurrent.futures.as_completed(future_map):
            sid, script_ = future_map[future]
            try:
                timings[f"[{sid}] {script_}"] = future.result()
            except RuntimeError as exc:
                errors.append(str(exc))
                # Kill all still-running sibling subprocesses immediately
                _kill_running_procs()
                # Cancel futures that haven't started yet
                for f in future_map:
                    f.cancel()

    if errors:
        with _print_lock:
            for err in errors:
                print(f"\n  *** {err} ***", file=sys.stderr, flush=True)
        sys.exit(1)

    return timings


# -- Argument parsing -----------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the full Brazilian Open Finance data pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--from", dest="from_stage", type=int, default=0, metavar="N",
        help="Start from this stage number (0-4).  Skips all earlier stages.",
    )
    p.add_argument(
        "--only", dest="only_stage", type=int, default=None, metavar="N",
        help="Run only this stage number (0-4).  All others are skipped.",
    )
    p.add_argument(
        "--skip", dest="skip_steps", type=str, default="", metavar="IDs",
        help="Comma-separated list of step IDs to skip (e.g. '2b,2c').",
    )
    p.add_argument(
        "--list", action="store_true",
        help="Print the pipeline steps and exit.",
    )
    return p.parse_args()


# -- Main -----------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    toon_ctx = _load_toon_runtime_context()
    if toon_ctx and get_script_config is not None:
        cfg = get_script_config(toon_ctx, "run_data_pipeline")
        note = cfg.get("note") if isinstance(cfg, dict) else None
        if isinstance(note, str) and note.strip():
            _banner(f"TOON note: {note.strip()}", "=")

    if toon_ctx and dump_context_json is not None:
        toon_ctx_file = Path(SCRIPT_DIR) / "utils" / "toon_context.json"
        try:
            dump_context_json(toon_ctx, toon_ctx_file)
            os.environ["TOON_CONTEXT_PATH"] = str(toon_ctx_file)
            print(f"[TOON] Runtime context exported to: {toon_ctx_file}")
        except Exception as exc:
            print(f"[TOON] Warning: failed to export runtime context ({exc}).")

    # Build step lookup: {step_id: (script, description)}
    steps_dict: dict[str, tuple[str, str]] = {
        step_id: (script, desc) for _, step_id, script, desc in STEPS
    }

    if args.list:
        _banner("Pipeline steps", "=")
        for stage, step_id, script, desc in STEPS:
            print(f"  [{step_id}]  Stage {stage}  {script:<40}  {desc}")
        print("=" * _W)
        print("\nExecution waves (steps within each wave run in parallel):")
        for i, wave in enumerate(WAVES, 1):
            print(f"  Wave {i}: {wave}")
        return

    skip_set = {s.strip() for s in args.skip_steps.split(",") if s.strip()}

    # Determine which stage numbers are active
    stage_of   = {step_id: stage for stage, step_id, *_ in STEPS}
    all_stages = sorted({stage for stage, *_ in STEPS})
    if args.only_stage is not None:
        active_stages: set[int] = {args.only_stage}
    else:
        active_stages = {s for s in all_stages if s >= args.from_stage}

    # Filter WAVES to only include steps whose stage is active
    filtered_waves: list[list[str]] = []
    for wave in WAVES:
        filtered = [sid for sid in wave if stage_of.get(sid, -1) in active_stages]
        if filtered:
            filtered_waves.append(filtered)

    t_start = time.perf_counter()
    _banner(
        f"Brazilian Open Finance -- Full Data Pipeline  "
        f"({time.strftime('%Y-%m-%d %H:%M:%S')})", "="
    )

    if args.only_stage is not None:
        print(f"  Running ONLY stage {args.only_stage}")
    elif args.from_stage > 0:
        print(f"  Starting from stage {args.from_stage}")
    if skip_set:
        print(f"  Skipping steps: {', '.join(sorted(skip_set))}")

    all_timings: dict[str, float] = {}
    for wave_steps in filtered_waves:
        wave_timings = run_wave(wave_steps, steps_dict, skip_set)
        all_timings.update(wave_timings)

    total = time.perf_counter() - t_start
    print(f"\n{'=' * _W}")
    print("  Summary")
    print(f"{'-' * _W}")
    for label, t in all_timings.items():
        print(f"  {label:<52} {t:>7.1f}s")
    print(f"{'-' * _W}")
    print(f"  Total :  {total:.1f}s  ({total / 60:.1f} min)")
    print(f"  Finished {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * _W)


if __name__ == "__main__":
    main()

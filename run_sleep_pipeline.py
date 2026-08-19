"""
run_sleep_pipeline.py — Sleepiness Estimation Pipeline Orchestrator

CLI Options:
------------
usage: run_sleep_pipeline.py [-h] [--only-spec-12] [--skip-sleep] [--sleep-only]
                             [--skip-steps STEP [STEP ...]]

Run the full Sleepiness Estimation Pipeline (3 estimators → exports → demand prep).

options:
  -h, --help              show this help message and exit
  --only-spec-12          Only run specification 12 for demand prep (not all specs)
  --skip-sleep            Skip the estimation steps (1–3), run only exports & demand prep (4–6)
  --sleep-only            Run only the estimation steps (1–3), skip exports & demand prep
  --skip-steps STEP ...   Skip specific step IDs (1–6)

All steps run sequentially. Steps 1–4 are sleep estimators; each spawns its own
ProcessPoolExecutor internally. Running them concurrently exhausted Windows non-paged
pool via simultaneous IPC pipe traffic for 400K-row DataFrames (WinError 1450).

  1. estimation_1_sleep.py              (E1: Local B-type)
  2. estimation_2_sleep.py              (E2: Pooled B+D Linear)
  3. estimation_sleep_common.py --est 3 (E3: Pooled Single-Index)
  4. estimation_sleep_common.py --est 4 (E4: Pooled Single-Index + Time)
  5. export_results.py                  (Export 1st/2nd Stage Summaries, Est 1-4)
  6. estimation_demand_1_prep.py        (Universal Demand Prep Orchestrator, Est 1-4)
  7. export_analyze_spec12.py           (Analyze Specification 12 Results)
  8. desc_3.py                          (Cluster-imbalance / deposit-concentration table)
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)
import argparse
import subprocess
import sys
import time
from pathlib import Path

# Line-buffer our OWN output (mirrors run_data_pipeline.py).  Python block-buffers stdout when
# it is a file/pipe rather than a tty, so without this the [STARTING]/Finished banners sit in
# the buffer for hours and the log looks stalled even though the pipeline is healthy.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

_EMAIL_WARNED_MISSING_PWD = False

def send_notification_email(finished_step, elapsed_seconds, next_step):
    global _EMAIL_WARNED_MISSING_PWD
    import os
    import smtplib
    from email.message import EmailMessage

    # To use this without prompts, you must set an App Password in your environment variables.
    # For example: 
    # $env:SYS_EMAIL_USER="your-email@gmail.com"
    # $env:SYS_EMAIL_PWD="your-16-digit-app-password"
    
    sender = os.environ.get("SYS_EMAIL_USER", "pedro.feijo25@gmail.com")
    pwd = os.environ.get("SYS_EMAIL_PWD")
    recipient = "pedro.feijodemoraes@yale.edu"

    if not pwd:
        if not _EMAIL_WARNED_MISSING_PWD:
            print("  -> Email notifications disabled (set SYS_EMAIL_PWD to enable).")
            _EMAIL_WARNED_MISSING_PWD = True
        return

    try:
        msg = EmailMessage()
        mins, secs = divmod(int(elapsed_seconds), 60)
        
        msg['Subject'] = f"[Sleep Pipeline] Finished: {finished_step}"
        msg['From'] = sender
        msg['To'] = recipient

        body = f"The sleep pipeline has successfully completed {finished_step}.\n"
        body += f"Duration: {mins} minutes and {secs} seconds.\n\n"
        
        if next_step:
            body += f"Next step starting now: {next_step}\n"
        else:
            body += "This was the last step. The pipeline is fully complete!\n"
            
        msg.set_content(body)

        # Assuming Gmail for defaults, but this applies to Yale/Office365 with correct SMTP
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender, pwd)
            server.send_message(msg)
            
        print("  -> Notification email sent via SMTP!")
    except Exception as e:
        print(f"  -> Could not send SMTP notification: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Run the full Sleepiness Estimation Pipeline.",
        epilog="""
All steps run sequentially. Steps 1–4 are sleep estimators; each spawns its own
ProcessPoolExecutor internally. Running them concurrently exhausted Windows non-paged
pool via simultaneous IPC pipe traffic for 400K-row DataFrames (WinError 1450).

  1. estimation_1_sleep.py              (E1: Local B-type)
  2. estimation_2_sleep.py              (E2: Pooled B+D Linear)
  3. estimation_sleep_common.py --est 3 (E3: Pooled Single-Index)
  4. estimation_sleep_common.py --est 4 (E4: Pooled Single-Index + Time)
  5. export_results.py                  (Export 1st/2nd Stage Summaries, Est 1-4)
  6. estimation_demand_1_prep.py        (Universal Demand Prep Orchestrator, Est 1-4)
  7. export_analyze_spec12.py           (Analyze Specification 12 Results)
  8. desc_3.py                          (Cluster-imbalance / deposit-concentration table)
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--only-spec-12",
        action="store_true",
        help=("Only run specification 12 for E2-E4 + demand prep instead of all specifications. "
              "NOTE: E1 (estimation_1_sleep.py) has no spec selector -- it always runs the full "
              "12-spec grid -- so this flag does not reduce E1's runtime.")
    )

    parser.add_argument(
        "--skip-sleep",
        action="store_true",
        help="Skip executing sleepiness estimators 1-4 and only run exports and demand prep."
    )

    parser.add_argument(
        "--sleep-only",
        action="store_true",
        help="Only execute the sleepiness estimation steps, skipping exports and demand prep."
    )
    parser.add_argument(
        "--skip-steps",
        nargs="+",
        type=int,
        default=[],
        help="Skip executing specific steps (1-8). E.g., --skip-steps 7 to skip export_analyze."
    )
    args = parser.parse_args()

    spec_arg = "12" if args.only_spec_12 else "all"
    spec12_arg = ["--spec12"] if args.only_spec_12 else []

    if getattr(args, 'skip_sleep', False) and getattr(args, 'sleep_only', False):
        print("[ERROR] Cannot use both --skip-sleep and --sleep-only simultaneously.")
        sys.exit(1)

    scripts_to_run = []

    if not getattr(args, 'skip_sleep', False):
        scripts_to_run.extend([
            {"id": 1, "file": "estimation_1_sleep.py", "desc": "E1: Local B-type Estimation"},
            {"id": 2, "file": "estimation_2_sleep.py", "args": spec12_arg, "desc": "E2: Pooled B+D Linear"},
            # E3/E4 are config-driven via the shared dispatcher estimation_sleep_common.py --est N.
            {"id": 3, "file": "estimation_sleep_common.py", "args": ["--est", "3"] + spec12_arg, "desc": "E3: Pooled Single-Index"},
            {"id": 4, "file": "estimation_sleep_common.py", "args": ["--est", "4"] + spec12_arg, "desc": "E4: Pooled Single-Index + Time block"},
        ])

    if not getattr(args, 'sleep_only', False):
        scripts_to_run.extend([
            {"id": 5, "file": "export_results.py", "args": ["--estimation", "all"], "desc": "Export 1st/2nd Stage Summaries (Est 1-4)"},
            {"id": 6, "file": "estimation_demand_1_prep.py", "args": ["--estimation", "all", "--spec", spec_arg], "desc": "Universal Demand Prep Orchestrator & Panel Serialization (Est 1-4)"},
            {"id": 7, "file": "export_analyze_spec12.py", "args": [], "desc": "Analyze Specification 12 Results"},
            # desc_3 reads the E3 second-stage sample (est3/market_panel_phis.csv), so it
            # runs after estimation; it is the canonical cluster-imbalance / deposit-
            # concentration exhibit that justifies the wild cluster bootstrap.
            {"id": 8, "file": "desc_3.py", "args": [], "desc": "Cluster-imbalance & deposit-concentration table (WCB justification)"},
        ])

    import os

    # Filter skipped steps
    scripts_to_run = [s for s in scripts_to_run if s.get('id') not in args.skip_steps]

    cwd = Path(__file__).resolve().parent
    start_time_all = time.time()

    sleep_scripts = []
    heavy_scripts = []
    post_scripts = []

    _sleep_files = {'estimation_sleep_common.py'}  # config-driven E3/E4 dispatcher
    for s in scripts_to_run:
        if (s['file'].startswith('estimation_') and s['file'].endswith('_sleep.py')) or s['file'] in _sleep_files:
            sleep_scripts.append(s)
        else:
            post_scripts.append(s)

    def run_script(step, total_count, n_parallel_slots=1):
        script = step['file']
        desc = step['desc']
        args = step.get('args', [])
        
        print(f"\n[STARTING] {script}: {desc}", flush=True)
        # -u: run the child UNBUFFERED.  Without it Python block-buffers the child's stdout
        # whenever this pipeline is redirected to a file (the normal way it is run), so a
        # healthy multi-hour estimator emits NOTHING to the log for over an
        # hour and looks dead.  That cost a killed-and-restarted run on 2026-07-23; the
        # process was fine, only its output was invisible.  Never remove this.
        cmd = [sys.executable, "-u", script] + args
        start_time_script = time.time()

        # Limit numpy/scipy core thrashing so multiple heavy processes don't freeze the OS
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"   # belt-and-braces alongside -u
        for v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
            env[v] = "2" # Keep heavily numeric compute per process down to ~2 threads
        # Force a headless matplotlib backend: the project .venv lives inside OneDrive,
        # whose on-demand sync can dehydrate matplotlib's Tk window-icon PNG, crashing
        # the default interactive backend mid-batch (FileNotFoundError on matplotlib.png).
        env["MPLBACKEND"] = "Agg"
        # Windows consoles default to cp1252, so ANY non-ASCII character in a child's
        # print (phi, arrows, Upsilon) raises UnicodeEncodeError. It usually fires on a
        # STATUS line after the real work is done, so the step exits non-zero and reports
        # failure for a computation that actually succeeded -- three such false failures on
        # 2026-07-29 (cf_forward_rf, cf_1_franchise_dataonly). One env var here covers every
        # child the pipeline launches, which is why this is preferable to editing each script.
        env["PYTHONIOENCODING"] = "utf-8"
        # Tell each child script how many peer scripts share the CPU pool so it
        # can scale down its own ProcessPoolExecutor / Parallel n_jobs accordingly.
        env["SLEEP_PIPELINE_NSLOTS"] = str(max(1, n_parallel_slots))
            
        result = subprocess.run(cmd, cwd=cwd, env=env)
        
        elapsed_script = time.time() - start_time_script
        print(f"\n-> Finished {script} in {elapsed_script:.2f} seconds.")
        
        if result.returncode != 0:
            print(f"\n[ERROR] Script '{script}' failed with exit code: {result.returncode}")
            raise RuntimeError(f"'{script}' exited with code {result.returncode}")
            
        send_notification_email(script, elapsed_script, "Next in queue")
        return script, elapsed_script

    print("\n=====================================================================")
    try:
        # Run sleep estimators sequentially.
        # Running them in parallel caused WinError 1450 (Windows non-paged pool exhausted)
        # because each script spawns its own ProcessPoolExecutor and sends large DataFrames
        # (400K+ rows) via IPC pipes. Sequential execution lets each script use all CPUs
        # for its inner ProcessPool without competing for kernel IPC resources.
        if sleep_scripts:
            print("====== Running Sleep Estimations Sequentially ======")
            for s in sleep_scripts:
                run_script(s, len(sleep_scripts), n_parallel_slots=1)

        # Run memory-heavy scripts sequentially after the parallel batch finishes
        if heavy_scripts:
            print("\n====== Running Heavy Estimations Sequentially ======")
            for step in heavy_scripts:
                run_script(step, len(heavy_scripts))

        # Run post-processors sequentially because they aggregate the results from the estimations
        if post_scripts:
            print("\n====== Running Post-Processing Sequentially ======")
            for idx, step in enumerate(post_scripts, 1):
                run_script(step, len(post_scripts))

    except RuntimeError as err:
        print("\n=====================================================================")
        print(f" [FAILED] Sleepiness pipeline stopped: {err}")
        print("=====================================================================")
        sys.exit(1)

    elapsed_all = time.time() - start_time_all
    print("\n=====================================================================")
    print(f" SLEEPINESS PIPELINE COMPLETED SUCCESSFULLY IN {elapsed_all:.2f} SECONDS")
    print("=====================================================================")
if __name__ == '__main__':
    main()








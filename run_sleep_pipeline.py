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

All steps run sequentially. Steps 1–3 are sleep estimators; each spawns its own
ProcessPoolExecutor internally. Running them concurrently exhausted Windows non-paged
pool via simultaneous IPC pipe traffic for 400K-row DataFrames (WinError 1450).

  1. estimation_1_sleep.py              (Local B-type Estimation)
  2. estimation_2_sleep.py              (Pooled B+D Linear)
  3. estimation_3_sleep.py              (Pooled B+D Logistic, AME)
  4. estimation_4_sleep.py              (Pooled B+D Constrained Linear, uniform)
  5. estimation_5_sleep.py              (Pooled B+D Probit, AME)
  6. estimation_6_sleep.py              (Pooled B+D Single-Index, nonparametric)
  7. export_results.py                  (Export 1st/2nd Stage Summaries, Est 1-6)
  8. estimation_demand_1_prep.py        (Universal Demand Prep Orchestrator, Est 1-6)
  9. export_analyze_spec12.py           (Analyze Specification 12 Results)
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)
import argparse
import subprocess
import sys
import time
from pathlib import Path

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
All steps run sequentially. Steps 1–3 are sleep estimators; each spawns its own
ProcessPoolExecutor internally. Running them concurrently exhausted Windows non-paged
pool via simultaneous IPC pipe traffic for 400K-row DataFrames (WinError 1450).

  1. estimation_1_sleep.py              (Local B-type Estimation)
  2. estimation_2_sleep.py              (Pooled B+D Linear)
  3. estimation_3_sleep.py              (Pooled B+D Logistic, AME)
  4. estimation_4_sleep.py              (Pooled B+D Constrained Linear, uniform)
  5. estimation_5_sleep.py              (Pooled B+D Probit, AME)
  6. estimation_6_sleep.py              (Pooled B+D Single-Index, nonparametric)
  7. export_results.py                  (Export 1st/2nd Stage Summaries, Est 1-6)
  8. estimation_demand_1_prep.py        (Universal Demand Prep Orchestrator, Est 1-6)
  9. export_analyze_spec12.py           (Analyze Specification 12 Results)
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--only-spec-12",
        action="store_true",
        help="Only run specification 12 for the demand prep scripts instead of all specifications."
    )

    parser.add_argument(
        "--skip-sleep",
        action="store_true",
        help="Skip executing sleepiness estimators 1-6 and only run exports and demand prep."
    )

    parser.add_argument(
        "--sleep-only",
        action="store_true",
        help="Only execute the 6 sleepiness estimation steps, skipping exports and demand prep."
    )
    parser.add_argument(
        "--skip-steps",
        nargs="+",
        type=int,
        default=[],
        help="Skip executing specific steps (1-9). E.g., --skip-steps 9 to skip export_analyze."
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
            {"id": 1, "file": "estimation_1_sleep.py", "desc": "Local B-type Estimation"},
            {"id": 2, "file": "estimation_2_sleep.py", "args": spec12_arg, "desc": "Pooled B+D Linear"},
            {"id": 3, "file": "estimation_3_sleep.py", "args": spec12_arg, "desc": "Pooled B+D Logistic (AME)"},
            {"id": 4, "file": "estimation_4_sleep.py", "args": spec12_arg, "desc": "Pooled B+D Constrained Linear (uniform)"},
            {"id": 5, "file": "estimation_5_sleep.py", "args": spec12_arg, "desc": "Pooled B+D Probit (AME)"},
            {"id": 6, "file": "estimation_6_sleep.py", "args": spec12_arg, "desc": "Pooled B+D Single-Index (nonparametric)"},
        ])

    if not getattr(args, 'sleep_only', False):
        scripts_to_run.extend([
            {"id": 7, "file": "export_results.py", "args": ["--estimation", "all"], "desc": "Export 1st/2nd Stage Summaries"},
            {"id": 8, "file": "estimation_demand_1_prep.py", "args": ["--estimation", "all", "--spec", spec_arg], "desc": "Universal Demand Prep Orchestrator & Panel Serialization"},
            {"id": 9, "file": "export_analyze_spec12.py", "args": [], "desc": "Analyze Specification 12 Results"},
        ])

    import os

    # Filter skipped steps
    scripts_to_run = [s for s in scripts_to_run if s.get('id') not in args.skip_steps]

    cwd = Path(__file__).resolve().parent
    start_time_all = time.time()

    sleep_scripts = []
    heavy_scripts = []
    post_scripts = []

    for s in scripts_to_run:
        if s['file'].startswith('estimation_') and s['file'].endswith('_sleep.py'):
            sleep_scripts.append(s)
        else:
            post_scripts.append(s)

    def run_script(step, total_count, n_parallel_slots=1):
        script = step['file']
        desc = step['desc']
        args = step.get('args', [])
        
        print(f"\n[STARTING] {script}: {desc}")
        cmd = [sys.executable, script] + args
        start_time_script = time.time()
        
        # Limit numpy/scipy core thrashing so multiple heavy processes don't freeze the OS
        env = os.environ.copy()
        for v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
            env[v] = "2" # Keep heavily numeric compute per process down to ~2 threads
        # Force a headless matplotlib backend: the project .venv lives inside OneDrive,
        # whose on-demand sync can dehydrate matplotlib's Tk window-icon PNG, crashing
        # the default interactive backend mid-batch (FileNotFoundError on matplotlib.png).
        env["MPLBACKEND"] = "Agg"
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








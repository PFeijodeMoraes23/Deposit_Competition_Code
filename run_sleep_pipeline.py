"""
run_sleep_pipeline.py

CLI Options:
------------
usage: run_sleep_pipeline.py [-h] [--only-spec-12]

Run the full Sleepiness Estimation Pipeline.

options:
  -h, --help      show this help message and exit
  --only-spec-12  Only run specification 12 for the demand prep scripts
                  instead of all specifications.

This script sequentially runs the following steps:
  1. estimation_1_sleep.py         (Local Estimation of Sleepness)
  2. estimation_2_sleep.py         (Robustness bounds for B-firms)
  3. estimation_3_sleep.py         (Robustness bounds for pooled B and D firms)
  4. estimation_4_sleep.py         (NLLS logistic structural estimation)
  5. estimation_5_sleep.py         (Robustness bounds with cooperative/state controls)
  6. export_results.py             (Export 1st/2nd Stage Summaries across all estimators)
  7. estimation_demand_1_prep.py   (Universal Demand Prep Orchestrator & Panel Serialization)
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

def send_notification_email(finished_step, elapsed_seconds, next_step):
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
        print("  -> Skipped email notification: 'SYS_EMAIL_PWD' environment variable is not set.")
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
This script sequentially runs the following steps:
  1. estimation_1_sleep.py         (Local Estimation of Sleepness)
  2. estimation_2_sleep.py         (Robustness bounds for B-firms)
  3. estimation_3_sleep.py         (Robustness bounds for pooled B and D firms)
  4. estimation_4_sleep.py         (NLLS logistic structural estimation)
  5. estimation_5_sleep.py         (Robustness bounds with cooperative/state controls)
  6. export_results.py             (Export 1st/2nd Stage Summaries across all estimators)
  7. estimation_demand_1_prep.py   (Universal Demand Prep Orchestrator & Panel Serialization)
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
        help="Only execute the first 6 sleepiness estimation steps and plot scripts, skipping exports and demand prep."
    )
    parser.add_argument(
        "--skip-steps",
        nargs="+",
        type=int,
        default=[],
        help="Skip executing specific steps (1-7). E.g., --skip-steps 1 2"
    )
    args = parser.parse_args()


    

    spec_arg = "12" if args.only_spec_12 else "all"
    spec12_arg = ["--spec12"] if args.only_spec_12 else []

    if getattr(args, 'skip_sleep', False) and getattr(args, 'sleep_only', False):
        print("[ERROR] Cannot use both --skip-sleep and --sleep-only simultaneously.")
        sys.exit(1)

    # Define the scripts and their arguments precisely as requested
    scripts_to_run = []
    
    if not getattr(args, 'skip_sleep', False):
        scripts_to_run.extend([
            {"id": 1, "file": "estimation_1_sleep.py", "desc": "Local Estimation of Sleepness"},
            {"id": 2, "file": "estimation_2_sleep.py", "args": spec12_arg, "desc": "Robustness bounds for B-firms"},
            {"id": 3, "file": "estimation_3_sleep.py", "args": spec12_arg, "desc": "Robustness bounds for pooled B and D firms"},
            {"id": 4, "file": "estimation_4_sleep.py", "args": spec12_arg, "desc": "NLLS logistic structural estimation"},
            {"id": 5, "file": "estimation_5_sleep.py", "args": ["--model-type", "both", "--alt", "2"] + (["--spec12-only"] if args.only_spec_12 else []), "desc": "Robustness bounds with cooperative/state controls"},
        ])

    if not getattr(args, 'sleep_only', False):
        scripts_to_run.extend([
            {"id": 6, "file": "export_results.py", "args": ["--estimation", "all"], "desc": "Export 1st/2nd Stage Summaries across all estimators"},
            {"id": 7, "file": "estimation_demand_1_prep.py", "args": ["--estimation", "all", "--spec", spec_arg], "desc": "Universal Demand Prep Orchestrator & Panel Serialization"},
            {"id": 8, "file": "export_analyze_spec12.py", "args": ["--skip-est2"], "desc": "Analyze Specification 12 Results"}
        ])

    import concurrent.futures
    import os

    # Filter skipped steps
    scripts_to_run = [s for s in scripts_to_run if s.get('id') not in args.skip_steps]

    cwd = Path(__file__).resolve().parent
    start_time_all = time.time()

    sleep_scripts = []
    post_scripts = []
    
    # Split into sleep estimators and post-processors
    for s in scripts_to_run:
        if s['file'].startswith('estimation_') and s['file'].endswith('_sleep.py'):
            sleep_scripts.append(s)
        else:
            post_scripts.append(s)

    def run_script(step, total_count):
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
            
        result = subprocess.run(cmd, cwd=cwd, env=env)
        
        elapsed_script = time.time() - start_time_script
        print(f"\n-> Finished {script} in {elapsed_script:.2f} seconds.")
        
        if result.returncode != 0:
            print(f"\n[ERROR] Script '{script}' failed with exit code: {result.returncode}")
            raise RuntimeError(f"'{script}' exited with code {result.returncode}")
            
        send_notification_email(script, elapsed_script, "Next in queue")
        return script, elapsed_script

    print("\n=====================================================================")
    # Run sleep estimators in parallel (max 3 at a time to save RAM footprint on a 32GB machine)
    if sleep_scripts:
        print("====== Running Sleep Estimations in Parallel (max 3 concurrent) ======")
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(run_script, s, len(sleep_scripts)) for s in sleep_scripts]
            for future in concurrent.futures.as_completed(futures):
                future.result() # Will raise if sys.exit was called

    # Run post-processors sequentially because they aggregate the results from the estimations
    if post_scripts:
        print("\n====== Running Post-Processing Sequentially ======")
        for idx, step in enumerate(post_scripts, 1):
            run_script(step, len(post_scripts))

    elapsed_all = time.time() - start_time_all
    print("\n=====================================================================")
    print(f" SLEEPINESS PIPELINE COMPLETED SUCCESSFULLY IN {elapsed_all:.2f} SECONDS")
    print("=====================================================================")
if __name__ == '__main__':
    main()








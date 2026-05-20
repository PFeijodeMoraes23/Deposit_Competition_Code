"""
run_d_rate_scrape_pipeline.py
==============================
Master orchestrator for the digital-bank advertised-rate scraping chain.

Pipeline (strictly sequential — each stage depends on the previous):

  Stage 1 - d_rate_scrape_1_targets.py
            Build the target URL list for each digital-bank conglomerate.

  Stage 2 - d_rate_scrape_2_cdx.py
            Query the Wayback Machine CDX index for archived snapshots of
            those URLs. Writes scraper_urls.json.

  Stage 3 - d_rate_scrape_3_fetch.py
            Download archived HTML/PDF snapshots into archive_html/ and
            archive_pdfs/. RESUMABLE: skips any URL whose target file is
            already on disk. Slowest step (Wayback Machine rate limits).

  Stage 4 - d_rate_scrape_4_parse.py
            Extract CDI/SELIC rate mentions from cached HTML/PDFs into
            extracted_historical_rates.csv. CPU-bound, ~17 min.

  Stage 5 - d_rate_scrape_5_format.py
            Aggregate raw mentions to a quarter x conglomerate panel.
            Produces advertised_rates_quarterly{,_wide}.csv and a dropped
            audit CSV.

  Stage 6 - d_rate_scrape_6_diagnose.py
            Compare scraped rates against the panel_3_rates.py output;
            decide which CDI-fallback cells (deposit types 4 & 5 only) to
            override. Produces recommended_merge.csv.

Usage
-----
  python run_d_rate_scrape_pipeline.py                    # all stages
  python run_d_rate_scrape_pipeline.py --from 4           # start from stage 4
  python run_d_rate_scrape_pipeline.py --only 6           # run only stage 6
  python run_d_rate_scrape_pipeline.py --skip 1,2         # skip listed stages
  python run_d_rate_scrape_pipeline.py --list             # print steps & exit
  python run_d_rate_scrape_pipeline.py --dry-run-fetch    # pass --dry-run to stage 3
  python run_d_rate_scrape_pipeline.py --fetch-workers 2  # raise fetch concurrency
  python run_d_rate_scrape_pipeline.py --parse-workers 8 --parse-pages 5

If any stage exits non-zero, the pipeline halts. Fix the issue and resume
with `--from <stage>`.  Stage 3 is idempotent: a partial fetch can be
resumed by simply re-running, no extra flag needed.
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Match the parent pipeline's terminal encoding behavior for consistency on
# Windows consoles.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="latin-1", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="latin-1", errors="replace", line_buffering=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON     = sys.executable

# Each entry: (stage_int, script_filename, description)
STEPS = [
    (1, "d_rate_scrape_1_targets.py",
        "Build target URL list for digital-bank conglomerates"),
    (2, "d_rate_scrape_2_cdx.py",
        "Wayback CDX -> scraper_urls.json"),
    (3, "d_rate_scrape_3_fetch.py",
        "Fetch archive snapshots (RESUMABLE — skips cached files)"),
    (4, "d_rate_scrape_4_parse.py",
        "Extract CDI/SELIC mentions -> extracted_historical_rates.csv"),
    (5, "d_rate_scrape_5_format.py",
        "Aggregate -> advertised_rates_quarterly{,_wide}.csv"),
    (6, "d_rate_scrape_6_diagnose.py",
        "Compare to panel_3 fallbacks -> recommended_merge.csv"),
]

_W = 72

def _banner(text, char="="):
    print(char * _W)
    print(f"  {text}")
    print(char * _W, flush=True)


def _resolve_args_for_stage(stage, cli):
    """Translate top-level CLI flags into per-stage subprocess args."""
    args = []
    if stage == 1:
        if cli.target_top_n is not None:
            args += ["--top-n", str(cli.target_top_n)]
        if cli.target_all:
            args.append("--all")
        if cli.target_only_native_k5:
            args.append("--only-native-k5")
        if cli.target_no_aggregators:
            args.append("--no-aggregators")
    elif stage == 3:
        if cli.dry_run_fetch:
            args.append("--dry-run")
        if cli.fetch_workers is not None:
            args += ["--workers", str(cli.fetch_workers)]
        if cli.fetch_sleep_min is not None:
            args += ["--sleep-min", str(cli.fetch_sleep_min)]
        if cli.fetch_sleep_max is not None:
            args += ["--sleep-max", str(cli.fetch_sleep_max)]
        if cli.fetch_include_images:
            args.append("--include-images")
        if cli.fetch_no_url_filter:
            args.append("--no-url-filter")
        if cli.fetch_skip_url_patterns is not None:
            args += ["--skip-url-patterns", cli.fetch_skip_url_patterns]
        if cli.fetch_mime_priority is not None:
            args += ["--mime-priority", cli.fetch_mime_priority]
    elif stage == 4:
        if cli.parse_workers is not None:
            args += ["--workers", str(cli.parse_workers)]
        if cli.parse_pages is not None:
            args += ["--pages", str(cli.parse_pages)]
        if cli.parse_ocr_images:
            args.append("--ocr-images")
    elif stage == 5:
        if cli.manual_overrides is not None:
            args += ["--manual-overrides", cli.manual_overrides]
    return args


def run_step(stage, script, description, extra_args):
    path = os.path.join(SCRIPT_DIR, script)
    if not os.path.exists(path):
        print(f"  [SKIP] stage {stage}: script not found ({script})", flush=True)
        return 0.0
    _banner(f"Stage {stage}: {script}")
    print(f"  {description}")
    if extra_args:
        print(f"  args: {' '.join(extra_args)}")
    print(flush=True)
    t0 = time.perf_counter()
    rc = subprocess.call([PYTHON, path] + extra_args, cwd=SCRIPT_DIR)
    elapsed = time.perf_counter() - t0
    if rc != 0:
        print(flush=True)
        _banner(f"FAILED stage {stage} ({script}) — exit code {rc}", char="!")
        sys.exit(rc)
    print(f"\n  [OK] stage {stage} done in {elapsed:,.1f}s", flush=True)
    return elapsed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="start", type=int, default=None,
                    help="Start from this stage (1-6).")
    ap.add_argument("--only", type=int, default=None,
                    help="Run only this stage.")
    ap.add_argument("--skip", default="",
                    help="Comma-separated stages to skip, e.g. '1,2'.")
    ap.add_argument("--list", action="store_true",
                    help="Print the pipeline steps and exit.")

    # Stage-3 (fetch) passthrough
    ap.add_argument("--dry-run-fetch", action="store_true",
                    help="Stage 3: report cached/pending counts and exit without fetching.")
    ap.add_argument("--fetch-workers", type=int, default=None,
                    help="Stage 3: concurrent workers (default 1).")
    ap.add_argument("--fetch-sleep-min", type=float, default=None,
                    help="Stage 3: min seconds of jitter between requests.")
    ap.add_argument("--fetch-sleep-max", type=float, default=None,
                    help="Stage 3: max seconds of jitter between requests.")
    ap.add_argument("--fetch-include-images", action="store_true",
                    help="Stage 3: also fetch png/jpg/webp/gif snapshots for OCR.")
    ap.add_argument("--fetch-no-url-filter", action="store_true",
                    help="Stage 3: disable URL-pattern denylist (fetch everything).")
    ap.add_argument("--fetch-skip-url-patterns", default=None,
                    help="Stage 3: comma-separated URL substrings to skip.")
    ap.add_argument("--fetch-mime-priority", choices=("html", "pdf", "none"),
                    default="html",
                    help="Stage 3: queue ordering. Default 'html' drains HTML "
                         "before PDFs so rate mentions appear faster.")

    # Stage-1 (targets) passthrough
    ap.add_argument("--target-top-n", type=int, default=None,
                    help="Stage 1: number of largest digital-bank conglomerates (default 40).")
    ap.add_argument("--target-all", action="store_true",
                    help="Stage 1: include every D-type firm as a target (~386).")
    ap.add_argument("--target-only-native-k5", action="store_true",
                    help="Stage 1: restrict to firms with native IP rate data.")
    ap.add_argument("--target-no-aggregators", action="store_true",
                    help="Stage 1: omit supplementary aggregator-site targets.")

    # Stage-4 (parse) passthrough
    ap.add_argument("--parse-workers", type=int, default=None,
                    help="Stage 4: ProcessPool workers (default CPU-1).")
    ap.add_argument("--parse-pages", type=int, default=None,
                    help="Stage 4: max pages parsed per PDF.")
    ap.add_argument("--parse-ocr-images", action="store_true",
                    help="Stage 4: also OCR images in archive_images/ (needs pytesseract).")

    # Stage-5 (format) passthrough
    ap.add_argument("--manual-overrides", default=None,
                    help="Stage 5: path to manual-seed CSV. Default uses "
                         "IP_SCRAPE/manual_advertised_rates.csv if present.")
    args = ap.parse_args()

    if args.list:
        for stage, script, desc in STEPS:
            print(f"  [{stage}] {script:<35s} {desc}")
        return

    skip_stages = {int(s) for s in args.skip.split(",") if s.strip()}

    selected = []
    for stage, script, desc in STEPS:
        if args.only is not None and stage != args.only:
            continue
        if args.start is not None and stage < args.start:
            continue
        if stage in skip_stages:
            continue
        selected.append((stage, script, desc))

    if not selected:
        print("No stages selected.")
        return

    _banner(f"d_rate scrape pipeline — {len(selected)} stage(s) to run")
    for stage, script, desc in selected:
        print(f"   [{stage}] {script}")
    print(flush=True)

    t_total = time.perf_counter()
    for stage, script, desc in selected:
        extra = _resolve_args_for_stage(stage, args)
        run_step(stage, script, desc, extra)
    _banner(f"All stages complete in {time.perf_counter() - t_total:,.1f}s")


if __name__ == "__main__":
    main()

"""
run_openfinance_monthly.py
==========================
One-shot driver for the Open Finance Brasil fee collection + verification, meant to
be run on a WEEKLY SWEEP around each month's update window.

Pipeline (in order)
-------------------
  1. scrape_19_openfinance_fees   — pull the open-data fee APIs (all families by
                                     default), append a data_coleta-tagged snapshot.
  2. diag_openfinance_freshness   — infer each bank's update-landing business-day
                                     from the accumulated snapshots (the payload has
                                     no reference-date field, so freshness is measured).
  3. panel_11_openfinance_fees    — roll the fees up to the prudential conglomerate
                                     and build the cross-bank comparable measure.

Why weekly, and why not "the 15th"
----------------------------------
OFB values publish on the 10th BUSINESS day (BCB IN 32/2020), which lands on ~calendar
day 12-16 (e.g. it IS the 15th in May and January 2026 — see utils/br_calendar.py). So a
day-15 pull is borderline. A weekly sweep (≈ days 8, 15, 22) both (a) catches the update
as soon as it lands and (b) feeds the freshness verifier the multiple in-month snapshots
it needs to measure the true per-bank lag. Once the lag is known, you can drop to a
single monthly pull on a safe buffer day (~18-20).

Scheduling (Windows Task Scheduler — local is Windows)
------------------------------------------------------
Run WEEKLY (fires ~4x/month; the append+dedup design makes re-runs idempotent per day):

  schtasks /Create /TN "OFB_fees_weekly" /SC WEEKLY /D SUN /ST 06:00 ^
    /TR "\"C:\\...\\Egan_et_al_2025_Rep\\.venv\\Scripts\\python.exe\" ^
         \"C:\\...\\Egan_et_al_2025_Rep\\run_openfinance_monthly.py\""

(Adjust the absolute paths. Do NOT auto-create the task; create it deliberately.)

Cloud alternative (more durable archive): a GitHub Actions workflow on a monthly/weekly
cron (UTC — day-18 06:00 America/Sao_Paulo ≈ 09:00 UTC) that runs this script and commits
the appended panel + raw JSONL.

Usage
-----
  python run_openfinance_monthly.py
  python run_openfinance_monthly.py --families accounts,creditcards,loans
  python run_openfinance_monthly.py --skip-scrape        # re-run diag + rollup only
  python run_openfinance_monthly.py --no-cache
"""

from __future__ import annotations

import argparse
import logging
import traceback

from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_of_monthly")


def _step(label: str, fn) -> bool:
    log.info("──────── %s ────────", label)
    try:
        fn()
        return True
    except Exception:
        log.error("STEP FAILED: %s\n%s", label, traceback.format_exc())
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description="OFB monthly fee collection + verification")
    ap.add_argument("--families", default="all",
                    help="Families for the scrape (default: all). See scrape_19 --families.")
    ap.add_argument("--customer-type", default="PF", choices=["PF", "PJ"],
                    help="Person type for the conglomerate rollup (default PF).")
    ap.add_argument("--skip-scrape", action="store_true",
                    help="Skip the network scrape; only re-run diagnostics + rollup.")
    ap.add_argument("--no-cache", action="store_true",
                    help="Bypass the scraper's disk cache (force fresh fetch).")
    args = ap.parse_args()

    ok = True
    if not args.skip_scrape:
        import scrape_19_openfinance_fees as s19
        ok &= _step(
            f"1/3 scrape_19 (families={args.families})",
            lambda: s19.main(test_n=None, use_cache=not args.no_cache, families=args.families),
        )
    else:
        log.info("Skipping scrape (--skip-scrape).")

    import diag_openfinance_freshness as diag
    ok &= _step("2/3 freshness verifier", diag.main)

    import panel_11_openfinance_fees as p11
    ok &= _step("3/3 conglomerate rollup", lambda: p11.main(customer_type=args.customer_type))

    log.info("=== %s ===", "ALL STEPS OK" if ok else "COMPLETED WITH FAILURES (see log)")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

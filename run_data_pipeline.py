"""
run_data_pipeline.py
====================
Master runner for the full Brazilian Open Finance data pipeline.

Context on Digital Bank Classifications (D-Firms):
--------------------------------------------------
Banks originally classified as wholesale ("Tesouraria e Negócios") with zero 
branch presence often morph into digital retail contenders by adding a checking 
account platform. For instance, BTG Pactual's retail deposits hovered between 
R$ 100M - 350M (2014-2019) from corporate management, but exploded to R$ 1.07B 
in 2020 Q1 and R$ 4.49B in 2024 Q3 due to their digital app.

Similar trajectories occurred for:
1. Banco Inter (C0080996): ~R$ 300M (2018) -> R$ 4.86B (2024)
2. Grupo Bonsucesso / BS2 (C0080422): ~R$ 31M -> R$ 677M
3. Brasil Plural / Genial (C0080941): ~R$ 24M -> R$ 771M
4. Sofisa (C0080271): ~R$ 119M -> R$ 575M (Sofisa Direto)
5. Rendimento (C0080659): ~R$ 178M -> R$ 641M

These institutions are manually whitelisted as True Digital Retail Banks 
to ensure they are correctly assigned to the Tier-2 (Digital) segment, 
overriding their legacy wholesale regulatory classifications.

Executes every download / processing / panel-building script in the correct
dependency order.  Each step is a separate Python script run as a subprocess
so side-effects (imports, large DataFrames) never bleed between steps.

Pipeline stages
---------------
  Stage 0 - Raw data downloads
    0a. scrape_1_bcb_estban_if_data.py          ESTBAN monthly files + IF Data via Olinda API
    0b. scrape_2_estban_concat.py              Concatenate raw ESTBAN monthlies -> processed ESTBAN.csv
                                                  (Python port of the deprecated ESTBAN_Process_1.R; auto-extends as new months arrive)
    0c. scrape_3_ifdata_aggregate.py           Aggregate per-period IF_DATA_Values_* -> Aggregated Data type/report CSVs
                                                  (Python port of the deprecated if_data_process_1.py; consumed by panel_1 & panel_4)

  Stage 1 - IBGE demographics
    1a. scrape_4_ibge_demographics.py           Municipal population, GDP, age structure -> MCA demographics panel

  Stage 2 - Market characteristic panels
    2a. scrape_5_pix_panel.py                   Process BCB PIX municipality files -> MCA PIX adoption panel
    2b. scrape_6_anatel.py                      Download ANATEL mobile connections -> MCA connectivity panel
    2d. scrape_7_bcb_inclusion.py               BCB banking access-points (branches + correspondents) -> MCA inclusion panel
    2d2. scrape_8_bcb_banked.py                  BCB ESTBAN deposit balances (Dec snapshot) + WB Findex -> MCA banked-fraction proxy panel
    2e. scrape_9_cadunico.py                     CadUnico low-income families -> MCA poverty panel
    2f. scrape_10_fees.py                        BCB bank fee schedules (PF + PJ) -> tarifas conglomerate panel + fee summary

  Stage 2c - COSIF download + processing (feeds deposit-rate panels)
    2k. scrape_21_cosif_download.py             COSIF ZIP downloader for missing months (and future updates) -> shared/COSIF
    2l. cosif_process_1_extract.py              Extract COSIF -> custos_implicitos_<TAXONOMY>.csv (canonical, consumed by panel_3) + per-type foundation
    2m. cosif_process_2_calibrate.py            Per-bank + prudential-segment-shrunk corrected k=4 (CDB) rate -> cosif_cdb_rate_corrected.csv

  Stage 3 - Deposit panel, rates, characteristics & digital flags
    3a. panel_1_deposits.py                 ESTBAN + IF Data -> conglomerate x municipality x quarter deposit panel
    3b. panel_2_rates_ip.py         Extract IP explicit deposit rates from raw COSIF (parallel to deposits)
    3c. panel_3_master_panel_build.py            Compute and append deposit rates/spreads (COSIF + SGS; corrected k=4 CDB rate) to deposit panel
    3d. panel_4_bank_chars.py               IF Data -> conglomerate x quarter bank size and solvency characteristics panel
    3e. panel_5_flag_digital.py                   Analyze raw ESTBAN to identify purely digital banks -> PANEL_INTERMED

  Stage 4 - Master analysis panel   [ORDER IS LOAD-BEARING - see the STEPS comment]
    4a. panel_6_market.py                   Merge all MCA panels + deposit panel -> master analysis dataset
    4b. panel_7_instruments.py       Compute LOO instruments and FGC dummy -> OVERWRITES market_panel.csv
    4c. panel_8_demographics_sigma.py        Within-MCA demographic sigma for BLP parametric draws
    4d. panel_9_cosif_fees.py --patch-market  Fee columns -> market_panel_with_fees.csv (MUST follow 4b).
                                              OPTIONAL since 2026-08-04: the estimators read
                                              market_panel.csv unless USE_FEE_PANEL=1.

  Stage 5 - Descriptive statistics
    5a. desc_1.py                        Generate unweighted overview descriptive tables
    5b. desc_1.py --weight-col pop_total Generate market-weighted descriptive tables
    5c. desc_2.py                        Generate unweighted compressed/thematic tables
    5d. desc_2.py --weight-col pop_total Generate market-weighted compressed/thematic tables
    (desc_3.py and export_analyze_spec12.py are POST-ESTIMATION — they read the sleep's
     market_panel_phis.csv — and live in run_sleep_pipeline.py steps 11-12, not here.)

  Stage 6 - Firm-disclosure & count data (extensive-vs-intensive margin)
    6a. scrape_11_edgar_disclosures.py        SEC EDGAR firm customers + deposits
    6b. scrape_12_parent_disclosures.py      Mercado Pago MAU (MELI 8-K); C6/PicPay template
    6c. scrape_13_incumbent_clients.py       CVM-IPE earnings PDFs -> client counts (Tier-2/4 cascade)
    6d. scrape_14_bcb_accounts.py            BCB account-count report archival + template
    6e. scrape_15_worldbank_findex.py        Findex national demographic ownership
    6f. scrape_16_fgc_statistics.py          FGC bracket template + report archival
    6g. scrape_17_cvm_disclosures.py         CVM deposits in BRL (incl. Banco do Brasil)
    6h.  analysis_1_disclosure_join.py       Join disclosures -> account-vs-volume table

  NOTE: the non-pipeline scrapers scrape_18_cosif_service_fees, scrape_19_openfinance_fees,
  scrape_20_bcb_tariff_vigencia, scrape_22_pix_participants, scrape_23_pix_roster,
  scrape_24_cvm_fee_income, scrape_inss and the fee-merge panel_9_cosif_fees are run
  outside this master runner.

Usage
-----
  python run_data_pipeline.py                   # run all stages
  python run_data_pipeline.py --from 2          # restart from stage 2 onward
  python run_data_pipeline.py --only 3          # run only stage 3
  python run_data_pipeline.py --skip 2b,2e      # skip specific sub-steps

Notes
-----
  * If any step exits with a non-zero return code the pipeline halts and prints
    the failed step.  Fix the issue and rerun with --from <stage> to resume.
  * Stage 2 sub-steps (2a-2e) are independent and could in principle be
    parallelized; they are serialized here for simplicity and to avoid hitting
    rate limits on BCB / ANATEL / SAGI APIs simultaneously.
  * Stage 4 requires ALL panels to exist.  If some stage-2 downloads failed
    (data source unavailable), panel_6_market.py handles missing files
    gracefully (those columns will be NaN).

CLI Options:
------------
usage: run_data_pipeline.py [-h] [--from N] [--only N] [--skip IDs] [--list]

Run the full Brazilian Open Finance data pipeline.

options:
  -h, --help  show this help message and exit
  --from N    Start from this stage number (0-6). Skips all earlier stages.
  --only N    Run only this stage number (0-6). All others are skipped.
  --skip IDs  Comma-separated list of step IDs to skip (e.g. '9,12').
  --list      Print the pipeline steps and exit.

Scripts called by the data pipeline (contiguous step IDs in execution order):
-----------------------------------------------------------------------------
[1] scrape_1_bcb_estban_if_data.py
[2] scrape_2_estban_concat.py
[3] scrape_3_ifdata_aggregate.py
[4] scrape_4_ibge_demographics.py
[5] scrape_5_pix_panel.py
[6] scrape_6_anatel.py
[7] scrape_7_bcb_inclusion.py
[8] scrape_8_bcb_banked.py
[9] scrape_9_cadunico.py
[10] scrape_10_fees.py
[11] scrape_21_cosif_download.py
[12] cosif_process_1_extract.py
[13] cosif_process_2_calibrate.py
[14] panel_1_deposits.py
[15] panel_2_rates_ip.py
[16] panel_3_master_panel_build.py
[17] panel_4_bank_chars.py
[18] panel_5_flag_digital.py
[19] panel_6_market.py
[20] panel_7_instruments.py
[21] panel_8_demographics_sigma.py
[22] desc_1.py
[23] desc_1.py --weight-col pop_total
[24] desc_2.py
[25] desc_2.py --weight-col pop_total
[26] scrape_11_edgar_disclosures.py
[27] scrape_12_parent_disclosures.py
[28] scrape_13_incumbent_clients.py
[29] scrape_14_bcb_accounts.py
[30] scrape_15_worldbank_findex.py
[31] scrape_16_fgc_statistics.py
[32] scrape_17_cvm_disclosures.py
[33] analysis_1_disclosure_join.py
"""

import argparse
import concurrent.futures
import os
os.environ["PYTHONWARNINGS"] = "ignore"
import subprocess
import sys
import threading
import time
import contextlib
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
# Step IDs are contiguous 1..N in execution order.
STEPS = [
    # Stage 0 -- raw downloads
    (0, "1", "scrape_1_bcb_estban_if_data.py",
     "ESTBAN monthly files + IF Data (Olinda API)"),
    (0, "2", "scrape_2_estban_concat.py",
     "Concatenate raw ESTBAN monthlies -> processed ESTBAN.csv (Python port of deprecated R)"),
    (0, "3", "scrape_3_ifdata_aggregate.py",
     "Aggregate per-period IF_DATA_Values -> Aggregated Data reports (Python port of deprecated if_data_process_1.py)"),

    # Stage 1 -- IBGE demographics
    (1, "4", "scrape_4_ibge_demographics.py",
     "IBGE population, GDP, age structure -> MCA demographics panel"),

    # Stage 2 -- market characteristic panels
    (2, "5", "scrape_5_pix_panel.py",
     "Process BCB PIX municipality files -> MCA PIX adoption panel"),
    (2, "6", "scrape_6_anatel.py",
     "Download ANATEL mobile connections -> MCA connectivity panel"),
    (2, "7", "scrape_7_bcb_inclusion.py",
     "BCB banking access-points (branches + correspondents) -> MCA inclusion panel"),
    (2, "8", "scrape_8_bcb_banked.py",
     "BCB ESTBAN deposit balances (Dec snapshot) + WB Findex -> MCA banked-fraction proxy panel"),
    (2, "9", "scrape_9_cadunico.py",
     "CadUnico low-income families -> MCA poverty panel"),
    (2, "10", "scrape_10_fees.py",
     "BCB bank fee schedules (PF + PJ) -> tarifas conglomerate panel + fee summary"),

    # Stage 2c -- COSIF download + processing (feeds panel_2/panel_3 deposit rates)
    (2, "11", "scrape_21_cosif_download.py",
     "Download missing monthly COSIF ZIPs (BANCOS) into shared/COSIF"),
    (2, "12", "cosif_process_1_extract.py",
     "Extract COSIF -> custos_implicitos_<TAXONOMY>.csv (canonical) + per-type foundation"),
    (2, "13", "cosif_process_2_calibrate.py",
     "Per-bank + segment-shrunk corrected k=4 (CDB) rate -> cosif_cdb_rate_corrected.csv"),

    # Stage 3 -- deposit panel, rates, characteristics & digital flags
    (3, "14", "panel_1_deposits.py",
     "ESTBAN + IF Data -> conglomerate x municipality x quarter deposit panel"),
    (3, "15", "panel_2_rates_ip.py",
     "Extract IP explicit deposit rates from raw COSIF (parallel to deposits)"),
    (3, "16", "panel_3_master_panel_build.py",
     "Compute and append deposit rates/spreads (COSIF + SGS; corrected k=4 CDB rate) to deposit panel"),
    (3, "17", "panel_4_bank_chars.py",
     "IF Data -> conglomerate x quarter bank size and solvency characteristics panel"),
    (3, "18", "panel_5_flag_digital.py",
     "Analyze raw ESTBAN to identify purely digital banks -> PANEL_INTERMED"),

    # Stage 4 -- master analysis panel
    #
    # ORDER IS LOAD-BEARING, and two steps were missing here until 2026-07-14:
    #   * panel_7 OVERWRITES market_panel.csv with the LOO instruments (IV_BLP_LOO) + the FGC dummy.
    #     Rebuilding panel_6 without then re-running panel_7 leaves market_panel with NO
    #     loo_log_assets / mean_loo_log_assets / loo_equity_ratio / loo_basileia /
    #     leave_one_out_mean_spread -- and blp_1_estimation's build_regressor_matrices SILENTLY drops
    #     any instrument not present, so the BLP ends up identified off the cost shifters alone.
    #   * panel_9 builds market_panel_with_fees.csv. It must still run LAST in this stage, after
    #     panel_7's overwrite, so the fee panel is not built from a half-finished base.
    #     RESOLVED 2026-08-04 — the shadowing hazard this warning described actually fired.
    #     Estimation scripts USED to prefer with_fees whenever it existed, so a stale copy
    #     silently shadowed a freshly rebuilt market_panel. Run-order discipline was the only
    #     guard and it failed: panel_7/10/12 overwrote market_panel.csv on 08-03, panel_9 was
    #     not re-run, and every downstream estimate was built from the 07-28 fee panel. The
    #     preference is now gone (utils/paths.market_panel_csv reads market_panel.csv; opt in
    #     with USE_FEE_PANEL=1), so a stale fee panel can no longer shadow anything.
    # See counterfactuals_plan.md §0B.
    (4, "19", "panel_6_market.py",
     "Merge all MCA panels + deposit panel -> master analysis dataset"),
    (4, "20", "panel_7_instruments.py",
     "Compute LOO instruments and FGC dummy -> overwrites market_panel.csv"),
    (4, "21", "panel_8_demographics_sigma.py",
     "Within-MCA demographic σ for BLP parametric draws -> demographics_sigma.parquet"),
    # panel_10 (ESTBAN rival-branch IV) and panel_12 (local CDB spread) were manual-only
    # until 2026-07-23. A full panel_6 rebuild wipes market_panel, so estban_rival_branches_lag
    # and the local spread must be re-applied here, AFTER panel_7's overwrite and BEFORE panel_9.
    (4, "36", "panel_10_estban_instruments.py",
     "ESTBAN rival-branch instrument -> market_panel.csv (estban_rival_branches_lag; MUST follow panel_7)"),
    (4, "37", "panel_12_local_cdb_spread.py",
     "Member-CNPJ LOCAL type-4 (CDB) spread patch, toggle LOCAL_CDB_SPREAD (default ON) -> "
     "market_panel.csv (MUST follow panel_10, precede panel_9)"),
    (4, "22", "panel_9_cosif_fees.py --patch-market",
     "COSIF/Tarifas fee columns -> market_panel_with_fees.csv (MUST follow panel_7/panel_10/panel_12). "
     "Optional for estimation: the estimators read market_panel.csv unless USE_FEE_PANEL=1"),

    # Stage 5 -- descriptive statistics
    (5, "23", "desc_1.py",
     "Generate unweighted overview descriptive tables"),
    (5, "24", "desc_1.py --weight-col pop_total",
     "Generate market-weighted descriptive tables"),
    (5, "25", "desc_2.py",
     "Generate unweighted compressed/thematic descriptive tables"),
    (5, "26", "desc_2.py --weight-col pop_total",
     "Generate market-weighted compressed/thematic descriptive tables"),
    # NB: desc_3.py and export_analyze_spec12.py are NOT here. They read market_panel_phis.csv, which
    # is a SLEEP output — so they are POST-ESTIMATION steps and live in run_sleep_pipeline.py (steps
    # 11-12), which runs after the estimators. Putting them in this data pipeline would silently read
    # whatever the last sleep happened to leave on disk. desc_1/desc_2 belong here: they read
    # market_panel.csv, which this pipeline builds.

    # Stage 6 -- firm-disclosure & count data (extensive-vs-intensive margin diagnostic)
    # External-API scrapers, serialized (one per wave) to respect SEC/CVM/WB rate
    # limits, mirroring the Stage-2 convention. The join depends on BOTH the firm
    # disclosures and the Stage-4 market_panel.csv, so it runs last.
    (6, "28", "scrape_11_edgar_disclosures.py",
     "SEC EDGAR: firm customers (BR/consolidated) + deposits (XBRL)"),
    (6, "29", "scrape_12_parent_disclosures.py",
     "MELI 8-K: Mercado Pago fintech MAU; C6/PicPay manual template"),
    (6, "30", "scrape_13_incumbent_clients.py",
     "CVM-IPE earnings PDFs -> client counts (Pan/BMG/Banrisul/BB); vision fallback"),
    (6, "31", "scrape_14_bcb_accounts.py",
     "BCB RCF/REB report archival + account-count manual template"),
    (6, "32", "scrape_15_worldbank_findex.py",
     "World Bank Findex: national demographic account ownership"),
    (6, "33", "scrape_16_fgc_statistics.py",
     "FGC: bracket manual template + report archival"),
    (6, "34", "scrape_17_cvm_disclosures.py",
     "CVM DFP/ITR: deposits in BRL (incl. Banco do Brasil)"),
    (6, "35", "analysis_1_disclosure_join.py",
     "Join disclosures -> conglomerate x quarter account-vs-volume table"),
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
            with contextlib.suppress(Exception):
                proc.kill()

_W = 72   # display width

def _banner(text: str, char: str = "-") -> None:
    print(char * _W)
    print(f"  {text}")
    print(char * _W)


_print_lock = threading.Lock()  # serialize console output across parallel workers


import shlex

def run_step(step_id: str, script: str, description: str) -> float:
    """
    Run one pipeline script as a subprocess, capturing its output so that
    parallel runs don't interleave on the console.  Raises RuntimeError on
    failure (safe to use inside ThreadPoolExecutor worker threads).
    """
    script_args = shlex.split(script)
    script_file = script_args[0]
    path = os.path.join(SCRIPT_DIR, script_file)
    if not os.path.exists(path):
        with _print_lock:
            print(f"  [SKIP] {step_id} -- script not found: {script_file}", flush=True)
        return 0.0

    with _print_lock:
        print(f"  > [{step_id}] starting: {description}", flush=True)

    t0   = time.perf_counter()
    env = os.environ.copy()
    # Windows consoles default to cp1252, so any non-ASCII character in a child's print
    # (phi, arrows, Upsilon) raises UnicodeEncodeError -- typically on a STATUS line AFTER
    # the work is done, so the step reports failure for a computation that succeeded. One
    # env var here covers every child this pipeline launches. See run_sleep_pipeline.py.
    env["PYTHONIOENCODING"] = "utf-8"
    if toon_ctx_path := os.environ.get("TOON_CONTEXT_PATH", "").strip():
        env["TOON_CONTEXT_PATH"] = toon_ctx_path

    proc = subprocess.Popen(
        [PYTHON, path] + script_args[1:], cwd=SCRIPT_DIR,
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
#   Wave 2 -- After 0a completes: all characteristic panels run in stages.
#             1a=IBGE SIDRA (long, ~700s but independent of 0a),
#             2a=PIX (local files), 2b=ANATEL, 2d=BCB Olinda inclusion,
#             2e=SAGI CadUnico, 2f=BCB tarifas, 3a=deposits (reads 0a output),
#             and 3b=IP rates (reads raw COSIF).
#             These are mutually independent -> run in stages.
#   Wave 3 -- 3c (deposit rates) runs after 3a constructs the deposit panel and 3b extracts IP rates.
#   Wave 4 -- 4a (master merge) needs everything above -> serial.
WAVES: list[list[str]] = [
    ["1"],                                     # Wave 1: ESTBAN + IF Data raw download
    ["2"],                                    # Wave 1b: concat raw ESTBAN monthlies -> ESTBAN.csv
    ["3"],                                    # Wave 1c: aggregate per-period IF Data -> Aggregated Data reports
    ["4"],                                     # Wave 2a: IBGE
    ["5"],                                     # Wave 2b: Stage 2 scrapers CANNOT be parallelized
    ["6"],                                     # Wave 2c: ANATEL
    ["7"],                                     # Wave 2d: BCB inclusion
    ["8"],                                    # Wave 2d2: BCB ESTBAN banked-fraction proxy
    ["9"],                                     # Wave 2e: CadUnico
    ["10"],                                     # Wave 2f: fees
    # Stage 2c: COSIF download -> extract -> calibrate (strict serial chain;
    # must finish before panel_2/panel_3 consume the COSIF processed output).
    ["11"],                                    # Wave 2g: COSIF download
    ["12"],                                    # Wave 2h: COSIF extract (custos_implicitos + foundation)
    ["13"],                                    # Wave 2i: COSIF calibrate (corrected k=4 CDB rate)
    ["14", "15"],                               # Wave 3: deposits + IP rates
    ["16", "17", "18"],                         # Wave 4: deposit rates/spreads + bank chars + digital flags (parallel)
    ["19"],                                    # Wave 5: master merge (panel_6)
    ["20", "21"],                             # Wave 6: LOO instruments (panel_7, OVERWRITES market_panel)
                                              #         + demographics sigma (panel_8, independent)
    ["36"],                                   # Wave 6b: ESTBAN rival-branch IV (panel_10) — after panel_7 overwrite
    ["37"],                                   # Wave 6c: LOCAL CDB spread patch (panel_12) — after panel_10, before panel_9
    ["22"],                                   # Wave 7: panel_9 fee patch — MUST follow panel_7/panel_10/panel_12
    ["23", "24", "25", "26"],                 # Wave 8: descriptives (desc_1/desc_2 x {unweighted, weighted})
                                              #         desc_3 + export_analyze_spec12 are post-estimation
                                              #         -> run_sleep_pipeline.py steps 11-12
    # Stage 6: firm-disclosure scrapers, one per wave (serial) to respect external
    # rate limits (EDGAR/parent both hit SEC), then the join once all are present.
    ["28"], ["29"], ["30"], ["31"], ["32"], ["33"], ["34"],  # Wave 9-15: disclosure scrapers
    ["35"],                                                          # Wave 16: disclosure join
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
        # Cap workers to 6 to safely parallelize heavily across 32GB RAM machines, keeping 4-6GB headroom for TeXStudio
        workers = min(len(active), 6)
        print(f"\n  Wave (concurrency: {workers}): {ids_str}", flush=True)

    errors: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
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
        epilog="""
Scripts called by the data pipeline (contiguous step IDs in execution order):
-----------------------------------------------------------------------------
[1] scrape_1_bcb_estban_if_data.py
[2] scrape_2_estban_concat.py
[3] scrape_3_ifdata_aggregate.py
[4] scrape_4_ibge_demographics.py
[5] scrape_5_pix_panel.py
[6] scrape_6_anatel.py
[7] scrape_7_bcb_inclusion.py
[8] scrape_8_bcb_banked.py
[9] scrape_9_cadunico.py
[10] scrape_10_fees.py
[11] scrape_21_cosif_download.py
[12] cosif_process_1_extract.py
[13] cosif_process_2_calibrate.py
[14] panel_1_deposits.py
[15] panel_2_rates_ip.py
[16] panel_3_master_panel_build.py
[17] panel_4_bank_chars.py
[18] panel_5_flag_digital.py
[19] panel_6_market.py
[20] panel_7_instruments.py
[21] panel_8_demographics_sigma.py
[22] desc_1.py / [24] desc_2.py (each x2: unweighted + --weight-col pop_total)
[26] scrape_11_edgar_disclosures.py ... [32] scrape_17_cvm_disclosures.py
[33] analysis_1_disclosure_join.py
"""
    )
    p.add_argument(
        "--from", dest="from_stage", type=int, default=0, metavar="N",
        help="Start from this stage number (0-6).  Skips all earlier stages.",
    )
    p.add_argument(
        "--only", dest="only_stage", type=int, default=None, metavar="N",
        help="Run only this stage number (0-6).  All others are skipped.",
    )
    p.add_argument(
        "--to", "--max-stage", dest="max_stage", type=int, default=None, metavar="N",
        help="Run only up to and including this stage number (e.g. '--to 5' skips Stage 6).",
    )
    p.add_argument(
        "--skip", dest="skip_steps", type=str, default="", metavar="IDs",
        help="Comma-separated list of step IDs to skip (e.g. '9,12').",
    )
    p.add_argument(
        "--list", action="store_true",
        help="Print the pipeline steps and exit.",
    )
    return p.parse_args()


def send_notification_email(finished_step: str, elapsed_seconds: float, next_step: str | None) -> None:
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
        with _print_lock:
            print("  -> Skipped email notification: 'SYS_EMAIL_PWD' environment variable is not set.", flush=True)
        return

    try:
        msg = EmailMessage()
        mins, secs = divmod(int(elapsed_seconds), 60)
        
        msg['Subject'] = f"[Data Pipeline] Finished: {finished_step}"
        msg['From'] = sender
        msg['To'] = recipient

        body = f"The data pipeline has successfully completed {finished_step}.\n"
        body += f"Duration: {mins} minutes and {secs} seconds.\n\n"
        
        if next_step:
            body += f"Next step starting now: {next_step}\n"
        else:
            body += "This was the last step. The pipeline is fully complete!\n"
            
        msg.set_content(body)

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender, pwd)
            server.send_message(msg)
            
        with _print_lock:
            print("  -> Notification email sent via SMTP!", flush=True)
    except Exception as e:
        with _print_lock:
            print(f"  -> Could not send SMTP notification: {e}", flush=True)

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
        active_stages = {
            s for s in all_stages
            if s >= args.from_stage and (args.max_stage is None or s <= args.max_stage)
        }

    # Filter WAVES to only include steps whose stage is active
    filtered_waves: list[list[str]] = []
    for wave in WAVES:
        if filtered := [
            sid for sid in wave if stage_of.get(sid, -1) in active_stages
        ]:
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
    for i, wave_steps in enumerate(filtered_waves):
        wave_start = time.perf_counter()
        wave_timings = run_wave(wave_steps, steps_dict, skip_set)
        wave_elapsed = time.perf_counter() - wave_start
        all_timings |= wave_timings

        # Only notify if we actually ran something in this wave
        if wave_timings:
            finished_str = ", ".join(steps_dict[sid][0] for sid in wave_steps if sid not in skip_set)
            
            # Find next step
            next_str = None
            if i + 1 < len(filtered_waves):
                next_wave = filtered_waves[i + 1]
                next_str = ", ".join(steps_dict[sid][0] for sid in next_wave if sid not in skip_set)
                if not next_str:
                    next_str = None
            
            send_notification_email(finished_str, wave_elapsed, next_str)

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

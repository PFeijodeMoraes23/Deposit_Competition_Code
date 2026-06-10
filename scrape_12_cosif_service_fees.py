"""
scrape_12_cosif_service_fees.py
================================
Extract banking service-fee revenue and deposit volumes from COSIF BANCOS ZIPs,
building a panel that improves on Nakane et al. (2006) along three dimensions:

  1. Time: monthly, 2013-present (not just Dec 2002 / Dec 2003)
  2. Institutions: all banks including digital (Nubank, Inter, C6, …)
  3. Deposit-type disaggregation: separate denominators for demand, savings,
     and time deposits → three distinct fee/deposit ratios

Key COSIF accounts extracted
-----------------------------
Pre-2023 format (old account plan):
  71700009  Rendas De Prestacao De Servicos  (service-fee revenue, aggregate)

Post-2023 format (new account plan — richer sub-accounts):
  7170000005  Receita de Prestação de Serviços  (aggregate)
  7170100008  Receita de Tarifas - PN e MEI     (retail / PF tariff revenue)
  7170200001  Receita de Tarifas - PJ           (corporate / PJ tariff revenue)
  All captured by prefix "717"

Deposit accounts (unchanged across formats):
  41100000  Depositos A Vista               (demand deposits)
  41200003  Depositos De Poupanca           (savings deposits)
  41500002  Depositos A Prazo               (time deposits)

DOCUMENTO codes
---------------
  4010  individual institution (always present; used for institution panel)
  4020  prudential conglomerate (only in pre-2023 files; absent in newer data)

File naming conventions
-----------------------
  Pre-2023:  {YYYYMM}BANCOS.ZIP       (inner CSV has 3 header rows)
  2023+:     {YYYYMM}BANCOS.csv.zip   (same structure, different extension)

Output files  (BCB/Tarifas/processed/)
---------------------------------------
  cosif_service_fees_institution.parquet   institution × month panel
  cosif_service_fees_conglomerate.parquet  conglomerate × month panel
  cosif_service_fees_institution.csv       same, CSV copy
  cosif_service_fees_conglomerate.csv      same, CSV copy

Design
------
  - ProcessPoolExecutor: one worker per ZIP file (I/O-bound, ~119 files)
  - Polars for final aggregation and I/O (fast parquet write)
  - Pandas inside workers (pickling-friendly; already project standard)
  - Falls back gracefully if Polars not installed (pure pandas output)

Usage
-----
  python scrape_12_cosif_service_fees.py
  python scrape_12_cosif_service_fees.py --workers 4   # override CPU count
  python scrape_12_cosif_service_fees.py --test        # first 5 ZIPs only
"""

from __future__ import annotations

import argparse
import logging
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
from utils import paths
_REPO = Path(__file__).resolve().parents[2]
COSIF_RAW = paths.COSIF_RAW
OUT_DIR   = paths.TARIFAS_PROC

# ---------------------------------------------------------------------------
# Account filters
# ---------------------------------------------------------------------------
SVC_PREFIX     = "717"       # All service-fee revenue accounts (both old and new plan)
# New plan sub-accounts (2023+):
#   7170000005 = aggregate, 7170100xxx = PF/retail, 7170200xxx = PJ/corporate
SVC_PF_PREFIX  = "71701"     # Retail (Pessoa Física + MEI) tariff revenue
SVC_PJ_PREFIX  = "71702"     # Corporate (Pessoa Jurídica) tariff revenue

DEP_DEMAND     = "41100"     # Depositos A Vista
DEP_SAVINGS    = "41200"     # Depositos De Poupanca
DEP_TIME       = "41500"     # Depositos A Prazo
DEP_ALL_PREFIX = "410"       # Total deposits header (41000007)

KEEP_PREFIXES = (SVC_PREFIX, DEP_DEMAND, DEP_SAVINGS, DEP_TIME, DEP_ALL_PREFIX)

INST_DOC  = "4010"           # institution-level (always present)
CONG_DOC  = "4020"           # conglomerate-level (only pre-2023 files)


# ---------------------------------------------------------------------------
# Worker function  (must be module-level for pickling on Windows)
# ---------------------------------------------------------------------------

def _process_zip(zip_path: str) -> list[dict]:
    """
    Read one COSIF file (ZIP, csv.zip, or plain CSV), return relevant rows.
    Runs in a subprocess — must not use any unpicklable globals.
    """
    import pandas as pd
    import zipfile
    from io import BytesIO
    from pathlib import Path as _Path

    path = zip_path
    p = _Path(path)

    # Load bytes: either from inside a ZIP or directly from a plain CSV
    try:
        if p.suffix.lower() in (".zip",):
            with zipfile.ZipFile(path) as zf:
                raw_bytes = zf.open(zf.namelist()[0]).read()
            src = BytesIO(raw_bytes)
        else:
            # Plain .csv — read directly
            src = path
    except Exception as exc:
        return [{"_error": str(exc), "_file": path}]

    try:
        df = pd.read_csv(src, sep=";", encoding="latin1", dtype=str, skiprows=3)
    except Exception:
        try:
            df = pd.read_csv(src, sep=";", encoding="latin1", dtype=str)
        except Exception as exc:
            return [{"_error": str(exc), "_file": path}]

    df.columns = [c.replace("#", "").strip() for c in df.columns]

    required = {"DATA_BASE", "DOCUMENTO", "CNPJ", "NOME_INSTITUICAO",
                "COD_CONGL", "NOME_CONGL", "CONTA", "NOME_CONTA", "SALDO"}
    if not required.issubset(df.columns):
        return []

    # Filter to relevant documents and accounts
    doc_mask  = df["DOCUMENTO"].isin([INST_DOC, CONG_DOC])
    acct_mask = df["CONTA"].str.startswith(KEEP_PREFIXES, na=False)
    sub = df[doc_mask & acct_mask].copy()

    if sub.empty:
        return []

    # Parse balance (Brazilian decimal: "1.234.567,89" -> 1234567.89)
    sub["saldo_num"] = (
        sub["SALDO"]
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    sub["saldo_num"] = pd.to_numeric(sub["saldo_num"], errors="coerce")

    # Normalise CNPJ
    sub["cnpj"] = sub["CNPJ"].str.strip().str.zfill(8)
    sub["data_base"] = pd.to_numeric(sub["DATA_BASE"], errors="coerce").astype("Int64")

    keep_cols = ["data_base", "DOCUMENTO", "cnpj", "NOME_INSTITUICAO",
                 "COD_CONGL", "NOME_CONGL", "CONTA", "NOME_CONTA", "saldo_num"]
    sub = sub[[c for c in keep_cols if c in sub.columns]]
    sub = sub.rename(columns={
        "DOCUMENTO": "documento",
        "NOME_INSTITUICAO": "nome_instituicao",
        "COD_CONGL": "cod_congl",
        "NOME_CONGL": "nome_congl",
        "CONTA": "conta",
        "NOME_CONTA": "nome_conta",
    })

    return sub.to_dict("records")


# ---------------------------------------------------------------------------
# Build ratio panel from long records
# ---------------------------------------------------------------------------

def _build_panel(records: list[dict], doc_code: str) -> pd.DataFrame:
    """
    Pivot long records into a wide panel with fee/deposit ratios.

    doc_code: INST_DOC or CONG_DOC
    """
    df = pd.DataFrame(records)
    df = df[df["documento"] == doc_code].copy()

    if df.empty:
        return pd.DataFrame()

    id_col   = "cnpj"     if doc_code == INST_DOC else "cod_congl"
    name_col = "nome_instituicao" if doc_code == INST_DOC else "nome_congl"

    # Fill NaN strings so groupby doesn't silently drop rows
    for col in (id_col, name_col, "conta"):
        if col in df.columns:
            df[col] = df[col].fillna("")

    # Aggregate (sum) by entity × month × account
    grp = (
        df.groupby(["data_base", id_col, name_col, "conta"], as_index=False)
        ["saldo_num"].sum()
    )

    # Map account to category
    # Pre-2023: 71700009 (single aggregate)
    # Post-2023: 7170000005 (total), 7170100xxx (PF/retail), 7170200xxx (PJ/corp)
    def _cat(conta: str) -> str:
        if conta.startswith(SVC_PF_PREFIX):  # 71701... → retail tariffs
            return "svc_revenue_pf"
        if conta.startswith(SVC_PJ_PREFIX):  # 71702... → corporate tariffs
            return "svc_revenue_pj"
        if conta.startswith(SVC_PREFIX):     # 717... → aggregate (catches old + new total)
            return "svc_revenue"
        if conta.startswith(DEP_DEMAND):
            return "dep_demand"
        if conta.startswith(DEP_SAVINGS):
            return "dep_savings"
        if conta.startswith(DEP_TIME):
            return "dep_time"
        return "dep_total"

    grp["category"] = grp["conta"].map(_cat)

    # Pivot: one row per entity × month
    wide = grp.pivot_table(
        index=["data_base", id_col, name_col],
        columns="category",
        values="saldo_num",
        aggfunc="sum",
    ).reset_index()
    wide.columns.name = None

    # Ensure all expected columns exist
    for col in ("svc_revenue", "svc_revenue_pf", "svc_revenue_pj",
                "dep_demand", "dep_savings", "dep_time", "dep_total"):
        if col not in wide.columns:
            wide[col] = float("nan")

    # For 2023+ files that report PF+PJ sub-accounts but not the aggregate,
    # reconstruct the aggregate as PF + PJ when total is missing.
    has_total = wide["svc_revenue"].notna() & (wide["svc_revenue"] > 0)
    pf_pj_sum = wide[["svc_revenue_pf","svc_revenue_pj"]].sum(axis=1, min_count=1)
    wide["svc_revenue"] = wide["svc_revenue"].where(has_total, pf_pj_sum)

    # Compute fee ratios (Nakane-style: revenue / deposit volume)
    # We divide by 6 as Nakane did (semi-annual flow / 6 = monthly equivalent).
    # COSIF saldos here are CUMULATIVE 6-month income statement flows (Jan-Jun or Jul-Dec).
    svc_monthly = wide["svc_revenue"] / 6
    wide["fee_ratio_demand"]  = svc_monthly / wide["dep_demand"]
    wide["fee_ratio_savings"] = svc_monthly / wide["dep_savings"]
    wide["fee_ratio_time"]    = svc_monthly / wide["dep_time"]

    # For the aggregate ratio: traditional banks use sum of (demand+savings+time);
    # payment institutions (Nubank, PagSeguro, Stone …) have those as NaN and
    # instead report all balances in dep_total (account 41000007 = Outros Depositos
    # for IPs, or the header total for banks). Fall back to dep_total when the
    # sub-account sum is zero or missing.
    dep_sub_sum = wide[["dep_demand", "dep_savings", "dep_time"]].sum(axis=1, min_count=1)
    dep_sub_sum = dep_sub_sum.where(dep_sub_sum > 0, other=wide["dep_total"])
    wide["fee_ratio_all"] = svc_monthly / dep_sub_sum

    # Universal ratio using the header total (works for all institution types)
    wide["fee_ratio_total_deposits"] = svc_monthly / wide["dep_total"]

    # PF / PJ breakdowns (2023+ only; NaN for older data)
    wide["fee_ratio_pf"] = (wide["svc_revenue_pf"] / 6) / dep_sub_sum
    wide["fee_ratio_pj"] = (wide["svc_revenue_pj"] / 6) / dep_sub_sum

    return wide


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(workers: int | None = None, test: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Include all three COSIF entity types and both filename conventions:
    #   Pre-2023:  {YYYYMM}TYPE.ZIP
    #   2023+:     {YYYYMM}TYPE.csv.zip  (same inner structure, different extension)
    def _gather(pattern_upper: str) -> list[Path]:
        return sorted(set(
            list(COSIF_RAW.glob(f"*{pattern_upper}.ZIP")) +
            list(COSIF_RAW.glob(f"*{pattern_upper}.zip")) +
            list(COSIF_RAW.glob(f"*{pattern_upper}.csv.zip")) +
            list(COSIF_RAW.glob(f"*{pattern_upper}.csv"))   # plain CSV (unzipped)
        ))

    bancos       = _gather("BANCOS")
    sociedades   = _gather("SOCIEDADES")
    cooperativas = _gather("COOPERATIVAS")
    zip_files    = bancos + sociedades + cooperativas

    if not zip_files:
        log.error("No COSIF ZIPs found under %s", COSIF_RAW)
        return

    if test:
        zip_files = zip_files[:5]
        log.info("TEST MODE: processing %d ZIPs", len(zip_files))
    else:
        log.info("Found %d ZIPs (%d BANCOS + %d SOCIEDADES + %d COOPERATIVAS | %s → %s)",
                 len(zip_files), len(bancos), len(sociedades), len(cooperativas),
                 zip_files[0].name, zip_files[-1].name)

    n_workers = workers or min(8, (os.cpu_count() or 4))
    log.info("Using %d workers", n_workers)

    all_records: list[dict] = []
    errors = 0

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_process_zip, str(z)): z for z in zip_files}
        done = 0
        for fut in as_completed(futures):
            done += 1
            zname = futures[fut].name
            if done % 20 == 0 or done == len(zip_files):
                log.info("  [%d/%d] last: %s", done, len(zip_files), zname)
            try:
                rows = fut.result()
                error_rows = [r for r in rows if "_error" in r]
                if error_rows:
                    log.warning("  Error in %s: %s", zname, error_rows[0]["_error"])
                    errors += 1
                else:
                    all_records.extend(rows)
            except Exception as exc:
                log.error("  Unhandled exception for %s: %s", zname, exc)
                errors += 1

    log.info("Collected %d rows from %d ZIPs (%d errors)",
             len(all_records), len(zip_files), errors)

    if not all_records:
        log.error("No records collected — aborting.")
        return

    # Build institution panel (4010)
    log.info("Building institution panel …")
    inst_panel = _build_panel(all_records, INST_DOC)

    # Build conglomerate panel (4020)
    log.info("Building conglomerate panel …")
    cong_panel = _build_panel(all_records, CONG_DOC)

    # Save with Polars (fast parquet) or pandas fallback
    try:
        import polars as pl
        _use_polars = True
    except ImportError:
        _use_polars = False
        log.warning("polars not installed — saving as CSV only (run: pip install polars)")

    for panel, tag in [(inst_panel, "institution"), (cong_panel, "conglomerate")]:
        if panel.empty:
            log.warning("Panel %s is empty — skipping.", tag)
            continue

        csv_path = OUT_DIR / f"cosif_service_fees_{tag}.csv"
        panel.to_csv(csv_path, index=False)
        log.info("%s panel: %d rows × %d cols → %s",
                 tag, len(panel), len(panel.columns), csv_path.name)

        if _use_polars:
            pq_path = OUT_DIR / f"cosif_service_fees_{tag}.parquet"
            pl.from_pandas(panel).write_parquet(str(pq_path), compression="zstd")
            log.info("  parquet → %s", pq_path.name)

    log.info("=== Done ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COSIF service-fee panel builder")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: min(8, cpu_count))")
    parser.add_argument("--test", action="store_true",
                        help="Process only the first 5 ZIPs")
    args = parser.parse_args()
    main(workers=args.workers, test=args.test)

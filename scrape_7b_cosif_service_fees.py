"""
scrape_7b_cosif_service_fees.py
================================
Extract banking service-fee revenue and deposit volumes from COSIF BANCOS ZIPs,
building a panel that improves on Nakane et al. (2006) along three dimensions:

  1. Time: monthly, 2013-present (not just Dec 2002 / Dec 2003)
  2. Institutions: all banks including digital (Nubank, Inter, C6, …)
  3. Deposit-type disaggregation: separate denominators for demand, savings,
     and time deposits → three distinct fee/deposit ratios

Key COSIF accounts extracted
-----------------------------
  71700009  Rendas De Prestacao De Servicos  (service-fee revenue)

  41100000  Depositos A Vista               (demand deposits)
  41200003  Depositos De Poupanca           (savings deposits)
  41500002  Depositos A Prazo               (time deposits)

DOCUMENTO codes
---------------
  4010  individual institution (used for the institution-level panel)
  4020  prudential conglomerate (used for the conglomerate-level panel)

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
  python scrape_7b_cosif_service_fees.py
  python scrape_7b_cosif_service_fees.py --workers 4   # override CPU count
  python scrape_7b_cosif_service_fees.py --test        # first 5 ZIPs only
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
_REPO = Path(__file__).resolve().parents[2]
COSIF_RAW = _REPO / "BCB" / "Egan_et_al_2025_Rep" / "raw" / "COSIF_RAW"
OUT_DIR   = _REPO / "BCB" / "Tarifas" / "processed"

# ---------------------------------------------------------------------------
# Account filters
# ---------------------------------------------------------------------------
SVC_PREFIX  = "717"          # Rendas de Prestacao de Servicos
DEP_DEMAND  = "41100"        # Depositos A Vista
DEP_SAVINGS = "41200"        # Depositos De Poupanca
DEP_TIME    = "41500"        # Depositos A Prazo
DEP_ALL_PREFIX = "410"       # Total deposits header (41000007)

KEEP_PREFIXES = (SVC_PREFIX, DEP_DEMAND, DEP_SAVINGS, DEP_TIME, DEP_ALL_PREFIX)

INST_DOC  = "4010"           # institution-level balancete
CONG_DOC  = "4020"           # conglomerate-level balancete


# ---------------------------------------------------------------------------
# Worker function  (must be module-level for pickling on Windows)
# ---------------------------------------------------------------------------

def _process_zip(zip_path: str) -> list[dict]:
    """
    Read one COSIF BANCOS ZIP, return relevant rows as a list of dicts.
    Runs in a subprocess — must not use any unpicklable globals.
    """
    import pandas as pd
    import zipfile
    from io import BytesIO

    path = zip_path  # str
    try:
        with zipfile.ZipFile(path) as zf:
            inner = zf.namelist()[0]
            raw_bytes = zf.open(inner).read()
    except Exception as exc:
        return [{"_error": str(exc), "_file": path}]

    try:
        df = pd.read_csv(
            BytesIO(raw_bytes),
            sep=";",
            encoding="latin1",
            dtype=str,
            skiprows=3,
        )
    except Exception:
        # Some older ZIPs have no header rows
        try:
            df = pd.read_csv(BytesIO(raw_bytes), sep=";", encoding="latin1", dtype=str)
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
    def _cat(conta: str) -> str:
        if conta.startswith(SVC_PREFIX):
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
    for col in ("svc_revenue", "dep_demand", "dep_savings", "dep_time", "dep_total"):
        if col not in wide.columns:
            wide[col] = float("nan")

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

    return wide


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(workers: int | None = None, test: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Include all three COSIF entity types:
    #   BANCOS     – commercial and universal banks
    #   SOCIEDADES – payment institutions (Nubank, PagSeguro, Stone, PicPay …)
    #   COOPERATIVAS – credit unions (relevant local competitors in many municipalities)
    bancos       = sorted(COSIF_RAW.glob("*BANCOS.ZIP"))
    sociedades   = sorted(COSIF_RAW.glob("*SOCIEDADES.ZIP"))
    cooperativas = sorted(COSIF_RAW.glob("*COOPERATIVAS.ZIP"))
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

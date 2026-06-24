## scrape_22_pix_participants.py
# Author: Pedro Feijó de Moraes
#
# Last edited: 2026-06-09
#
# Purpose: Build an INSTITUTION-LEVEL Pix entry timeline — i.e. when each
#          participant (ISPB) first appears in the Pix environment. This is the
#          "key-based" leg of a cross-validated timeline; the "roster-based" leg
#          (participation type: direct/indirect, mandatory/voluntary) is built by
#          scrape_23_pix_roster.py and the two are reconciled there.
#
#          Distinct from scrape_5_pix_panel.py, which uses the geographic
#          TransacoesPixPorMunicipio endpoint (MCA × quarter adoption). This
#          script uses the ChavesPix endpoint (monthly stock of registered Pix
#          keys by participant), which nothing else in the pipeline touches.
#
#   Source (BCB Open Data / Olinda OData, verified live 2026-06-09):
#     https://olinda.bcb.gov.br/olinda/servico/Pix_DadosAbertos/versao/v1/odata/
#         ChavesPix(Data=@Data)?@Data='YYYY-MM-DD'
#     Schema (one row per month-end × ISPB × NaturezaUsuario × TipoChave):
#       Data            : month-end date (last business day), e.g. 2020-11-30
#       ISPB            : 8-digit participant identifier (join key to IF.data etc.)
#       Nome            : reduced institution name
#       NaturezaUsuario : PF | PJ
#       TipoChave       : CPF | CNPJ | Celular | e-mail | EVP (aleatoria)
#       qtdChaves       : count of registered keys
#     Coverage confirmed back to 2020-10-31 (pre-launch registration).
#
#   Method:
#     entry = the FIRST month in which an ISPB appears holding any Pix keys.
#     This is an observable proxy for "operational in Pix". It will (slightly)
#     lag pure adhesion/authorisation, and indirect participants that issue no
#     keys of their own may be under-captured — which is exactly why we
#     cross-validate against the participant roster in scrape_7.
#
#   Outputs (BCB/PIX/):
#     1. pix_participant_keys_panel.csv
#          AnoMes, Data, ISPB, Nome, qtd_chaves, qtd_pf, qtd_pj
#          (one row per month × ISPB; key counts summed across TipoChave)
#     2. pix_participant_entry.csv
#          ISPB, Nome, entry_anomes, entry_date, keys_at_entry,
#          n_months_observed, last_anomes, keys_last
#          (one row per ISPB; the institutional entry timeline)
#
# Notes:
#   • The ChavesPix function-import requires a Data parameter but returns the
#     full panel regardless; real filtering is done with $filter on Data. We
#     pull one month at a time (range filter month-start..month-end) and cache
#     each month to BCB/PIX/ChavesPix_YYYYMM.csv so re-runs are incremental.
#   • Olinda CSV here is plain comma-delimited UTF-8 with integer counts — no
#     Brazilian decimal-comma cleaning needed (unlike the municipality files).
###─────────────────────────────────────────────────────────────────────────────

import os
import ssl
import time
import glob
import logging
import calendar
import urllib.request

try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None

if ensure_project_venv is not None:
    ensure_project_venv(__file__)

import pandas as pd

try:
    from utils.toon_runtime import resolve_script_paths
except Exception:
    resolve_script_paths = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

## ─────────────────────────────────────────────────────────────────────────────
## 1) PATHS & CONSTANTS
## ─────────────────────────────────────────────────────────────────────────────
from utils import paths
from utils import refresh
BASE       = str(paths.OPEN_FINANCE)
PIX_DIR    = str(paths.PIX_DIR)   # processed pix_participant_* outputs
PIX_RAW    = str(paths.PIX_RAW)   # raw ChavesPix_* inputs
PANEL_CSV  = os.path.join(PIX_DIR, "pix_participant_keys_panel.csv")
ENTRY_CSV  = os.path.join(PIX_DIR, "pix_participant_entry.csv")

if resolve_script_paths is not None:
    _paths = resolve_script_paths(
        "scrape_22_pix_participants",
        {"pix_dir": PIX_DIR, "panel_csv": PANEL_CSV, "entry_csv": ENTRY_CSV},
        script_dir=os.path.dirname(os.path.abspath(__file__)),
    )
    PIX_DIR   = _paths["pix_dir"]
    PANEL_CSV = _paths["panel_csv"]
    ENTRY_CSV = _paths["entry_csv"]

OLINDA_BASE = ("https://olinda.bcb.gov.br/olinda/servico/Pix_DadosAbertos/"
               "versao/v1/odata/ChavesPix(Data=@Data)")

# ChavesPix coverage begins at the pre-launch registration snapshot.
SERIES_START = pd.Period("2020-10", "M")


## ─────────────────────────────────────────────────────────────────────────────
## 2) DOWNLOAD MONTHLY ChavesPix SNAPSHOTS (incremental, idempotent)
## ─────────────────────────────────────────────────────────────────────────────

def _month_url(period: pd.Period) -> str:
    """Build the Olinda CSV URL for a single month's key-stock snapshot."""
    y, m   = period.year, period.month
    last   = calendar.monthrange(y, m)[1]
    start  = f"{y:04d}-{m:02d}-01"
    end    = f"{y:04d}-{m:02d}-{last:02d}"
    # @Data is a required-but-ignored placeholder; $filter does the real work.
    return (
        f"{OLINDA_BASE}?@Data=%27{end}%27"
        f"&%24filter=Data%20ge%20{start}%20and%20Data%20le%20{end}"
        f"&%24format=text/csv&%24top=1000000"
    )


def download_missing_months():
    """Fetch any month from 2020-10 to the current month not yet cached."""
    os.makedirs(PIX_RAW, exist_ok=True)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    today_p = pd.Period(pd.Timestamp("today"), "M")
    for p in pd.period_range(SERIES_START, today_p):
        yyyymm   = p.strftime("%Y%m")
        filepath = os.path.join(PIX_RAW, f"ChavesPix_{yyyymm}.csv")
        # Skip only if cached AND outside the rolling refresh window (BCB revises/
        # backfills recent months, so the trailing year is always re-fetched).
        if os.path.exists(filepath) and not refresh.is_recent_month(p.year, p.month):
            continue

        url = _month_url(p)
        logging.info(f"Downloading ChavesPix for {yyyymm} …")
        for attempt in range(4):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
                    content = resp.read().decode("utf-8", errors="replace")
                # A valid response has a header line plus at least one data row.
                if len(content.splitlines()) < 2:
                    logging.info(f"  No ChavesPix data for {yyyymm} yet — stopping.")
                    return
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)
                logging.info(f"  Saved {yyyymm} ({len(content.splitlines())-1:,} rows)")
                break
            except Exception as e:
                if attempt == 3:
                    logging.error(f"  Failed {yyyymm} after 4 attempts: {e}")
                else:
                    time.sleep(2)


## ─────────────────────────────────────────────────────────────────────────────
## 3) LOAD + AGGREGATE TO MONTH × ISPB
## ─────────────────────────────────────────────────────────────────────────────

def load_all_months() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(PIX_RAW, "ChavesPix_*.csv")))
    if not files:
        raise FileNotFoundError(f"No ChavesPix_*.csv files in {PIX_RAW}. Run the download first.")

    frames = []
    for fp in files:
        try:
            df = pd.read_csv(fp, dtype={"ISPB": str}, encoding="utf-8")
            if {"Data", "ISPB", "qtdChaves"}.issubset(df.columns):
                frames.append(df)
            else:
                logging.warning(f"Unexpected columns in {fp}: {list(df.columns)}")
        except Exception as e:
            logging.warning(f"Could not read {fp}: {e}")

    raw = pd.concat(frames, ignore_index=True)
    raw["ISPB"]      = raw["ISPB"].str.zfill(8)
    raw["qtdChaves"] = pd.to_numeric(raw["qtdChaves"], errors="coerce").fillna(0)
    raw["Data"]      = pd.to_datetime(raw["Data"], errors="coerce")
    raw = raw.dropna(subset=["Data", "ISPB"])
    raw["AnoMes"]    = raw["Data"].dt.year * 100 + raw["Data"].dt.month
    logging.info(f"Loaded {len(raw):,} key rows across {raw['AnoMes'].nunique()} months, "
                 f"{raw['ISPB'].nunique()} ISPBs.")
    return raw


def build_month_ispb_panel(raw: pd.DataFrame) -> pd.DataFrame:
    """Sum keys across TipoChave, split PF/PJ, one row per month × ISPB."""
    raw = raw.copy()
    raw["qtd_pf"] = raw["qtdChaves"].where(raw["NaturezaUsuario"].eq("PF"), 0)
    raw["qtd_pj"] = raw["qtdChaves"].where(raw["NaturezaUsuario"].eq("PJ"), 0)

    panel = (
        raw.groupby(["AnoMes", "Data", "ISPB"], as_index=False)
           .agg(qtd_chaves=("qtdChaves", "sum"),
                qtd_pf=("qtd_pf", "sum"),
                qtd_pj=("qtd_pj", "sum"))
    )
    # Attach the most recent name seen for each ISPB (names drift over time).
    latest_name = (
        raw.sort_values("Data")
           .groupby("ISPB")["Nome"].last()
           .rename("Nome")
    )
    panel = panel.merge(latest_name, on="ISPB", how="left")
    panel = panel[["AnoMes", "Data", "ISPB", "Nome", "qtd_chaves", "qtd_pf", "qtd_pj"]]
    return panel.sort_values(["ISPB", "AnoMes"]).reset_index(drop=True)


## ─────────────────────────────────────────────────────────────────────────────
## 4) DERIVE INSTITUTIONAL ENTRY TIMELINE
## ─────────────────────────────────────────────────────────────────────────────

def build_entry_table(panel: pd.DataFrame) -> pd.DataFrame:
    """entry = first month an ISPB holds any keys."""
    active = panel[panel["qtd_chaves"] > 0].copy()

    first = (active.sort_values("AnoMes")
                   .groupby("ISPB")
                   .first()
                   .rename(columns={"AnoMes": "entry_anomes",
                                    "Data": "entry_date",
                                    "qtd_chaves": "keys_at_entry"}))
    last = (active.sort_values("AnoMes")
                  .groupby("ISPB")
                  .last()
                  .rename(columns={"AnoMes": "last_anomes",
                                   "qtd_chaves": "keys_last"}))
    n_months = active.groupby("ISPB")["AnoMes"].nunique().rename("n_months_observed")

    entry = (first[["Nome", "entry_anomes", "entry_date", "keys_at_entry"]]
             .join(n_months)
             .join(last[["last_anomes", "keys_last"]])
             .reset_index())
    entry["entry_date"] = entry["entry_date"].dt.date
    return entry.sort_values(["entry_anomes", "ISPB"]).reset_index(drop=True)


## ─────────────────────────────────────────────────────────────────────────────
## 5) MAIN
## ─────────────────────────────────────────────────────────────────────────────

def main():
    download_missing_months()

    raw   = load_all_months()
    panel = build_month_ispb_panel(raw)
    entry = build_entry_table(panel)

    panel.to_csv(PANEL_CSV, index=False)
    entry.to_csv(ENTRY_CSV, index=False)

    logging.info(f"Saved month × ISPB panel  -> {PANEL_CSV}  ({len(panel):,} rows)")
    logging.info(f"Saved participant entries -> {ENTRY_CSV}  ({len(entry):,} ISPBs)")

    by_year = (entry.assign(year=entry["entry_anomes"] // 100)
                    .groupby("year").size())
    print("\nPix participant entry timeline (institutions first observed holding keys)")
    print(f"  Total participants:     {len(entry):,}")
    print(f"  Entry-month range:      {entry['entry_anomes'].min()} – {entry['entry_anomes'].max()}")
    print("  New participants / year:")
    for y, n in by_year.items():
        print(f"     {y}: {n:>4}")
    print("\n  Earliest 8 entrants:")
    print(entry.head(8)[["ISPB", "Nome", "entry_anomes", "keys_at_entry"]].to_string(index=False))


if __name__ == "__main__":
    main()

## scr_panel_build.py
# Author: Pedro Feijo de Moraes
# Last edited: 2026-06-01
#
# Objective: Build a segment-level credit structure panel from the BCB's
#            public SCR aggregate data (Document 3040 / "planilha" files)
#            for use as instruments in the Egan et al. (2025) deposit demand
#            model.
#
# Data context:
#   The public SCR planilha files (BCB/SCR/SCRData/planilhaYYYY/) aggregate
#   credit registry data by:
#       tcb × sr × uf × cliente × ocupacao × cnae × porte × modalidade ×
#       origem × indexador
#   There are NO individual institution (CNPJ) identifiers.  The finest
#   institution proxy available is the prudential segment (`sr` = S1–S5),
#   which maps directly to the `segment` column in egan_panel_institution.csv.
#
# Instrument rationale:
#   Segment-level credit structure metrics are valid BLP cost instruments
#   for deposit rates because:
#     1. They are determined by the aggregate lending behaviour of all banks
#        in a prudential segment, not by any individual bank's deposit pricing.
#        → Exclusion restriction holds (exogenous to individual demand shocks).
#     2. Banks with heavier long-term or fixed-rate loan books face larger
#        asset-liability duration gaps → higher pressure to raise long-term
#        deposits → shifts deposit rate supply curve.
#
#   The maturity bucket dimension is NOT captured by the IF Data accounts
#   already in egan_panel_build_2.py (which only gives total credit / NPL),
#   so these instruments add genuinely new variation.
#
# Output variables (per prudential segment × AnoMes):
#   scr_long_term_share    – (a_vencer_de_1801_ate_5400 + a_vencer_acima_5400)
#                            / carteira_ativa  [share credit > 5 years maturity]
#   scr_medium_term_share  – (a_vencer_de_361_ate_1080 + a_vencer_de_1081_ate_1800)
#                            / carteira_ativa  [share credit 1–5 years maturity]
#   scr_short_term_share   – (a_vencer_ate_90 + a_vencer_de_91_ate_360)
#                            / carteira_ativa  [share credit < 1 year maturity]
#   scr_npl_rate           – vencido_acima_de_15_dias / carteira_ativa
#   scr_fixed_rate_share   – carteira_ativa(Prefixado) / carteira_total
#   scr_problematic_share  – ativo_problematico / carteira_ativa
#
#   All variables are then lagged 1 quarter when merged downstream to
#   maintain the causal timing convention used throughout the panel.
#
# Join key:
#   The output contains columns (segment, AnoMes), which join to the
#   institution panel on the same keys.  Institutions inherit the aggregate
#   credit structure profile of their prudential segment.
#
# Input:  BCB/SCR/SCRData/planilhaYYYY/planilha_YYYYMM.csv
# Output: BCB/Egan_et_al_2025_Rep/processed/PANEL_INTERMED/
#             scr_segment_panel.csv
#             scr_segment_panel.parquet
#
# Usage: python scr_panel_build.py
# ──────────────────────────────────────────────────────────────────────────────

import logging
import os
import glob
import re
from pathlib import Path

try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None

if ensure_project_venv is not None:
    ensure_project_venv(__file__)

import numpy as np
import pandas as pd
from logging.handlers import RotatingFileHandler

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
PARENT_DIR  = SCRIPT_DIR.parents[1]
BCB_PATH    = PARENT_DIR / "BCB"
SCR_DATA    = BCB_PATH / "SCR" / "SCRData"
PANEL_DIR   = BCB_PATH / "Egan_et_al_2025_Rep" / "processed" / "PANEL_INTERMED"

OUT_CSV     = PANEL_DIR / "scr_segment_panel.csv"
OUT_PARQUET = PANEL_DIR / "scr_segment_panel.parquet"

PANEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────
_log_file = SCRIPT_DIR / "scr_panel_build.log"
_handler  = RotatingFileHandler(
    str(_log_file), maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
_console  = logging.StreamHandler()
_console.setLevel(logging.INFO)
logging.basicConfig(
    handlers=[_handler, _console],
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)

# ── Constants ──────────────────────────────────────────────────────────────

# Maturity bucket columns in the raw planilha files
MATURITY_COLS = [
    "a_vencer_ate_90_dias",            # 0–90 days
    "a_vencer_de_91_ate_360_dias",     # 91–360 days
    "a_vencer_de_361_ate_1080_dias",   # 361–1080 days (~1–3 years)
    "a_vencer_de_1081_ate_1800_dias",  # 1081–1800 days (~3–5 years)
    "a_vencer_de_1801_ate_5400_dias",  # 1801–5400 days (~5–15 years)
    "a_vencer_acima_de_5400_dias",     # > 5400 days (>15 years)
]

VALUE_COLS = MATURITY_COLS + [
    "vencido_acima_de_15_dias",    # overdue > 15 days (NPL proxy)
    "carteira_ativa",              # total active portfolio
    "ativo_problematico",          # problematic assets
]

# Recognised prudential segments
VALID_SEGMENTS = {"S1", "S2", "S3", "S4", "S5"}

# Fixed-rate indexer label (varies slightly across files; use prefix match)
PREFIXADO_LABEL = "prefixado"


# ── Helper functions ────────────────────────────────────────────────────────

def _find_planilha_files() -> list[Path]:
    """
    Locate all planilha_YYYYMM.csv files under SCR_DATA.
    They are organised in year-subdirectories: planilha2012/, planilha2013/, ...
    """
    pattern = str(SCR_DATA / "planilha*" / "planilha_??????.csv")
    files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No planilha CSV files found under: {SCR_DATA}\n"
            "Expected path pattern: SCRData/planilhaYYYY/planilha_YYYYMM.csv"
        )
    logging.info(f"Found {len(files):,} planilha files "
                 f"({Path(files[0]).name} … {Path(files[-1]).name})")
    return [Path(f) for f in files]


def _yyyymm_from_path(p: Path) -> int:
    """Extract YYYYMM integer from a filename like planilha_202403.csv."""
    m = re.search(r"planilha_(\d{6})\.csv", p.name, re.IGNORECASE)
    if not m:
        raise ValueError(f"Could not extract YYYYMM from filename: {p.name}")
    return int(m.group(1))


def _yyyymm_to_anomes(yyyymm: int) -> int:
    """Convert YYYYMM int (e.g. 202403) to the IF-Data AnoMes convention.
    IF Data uses YYYYMM directly (e.g. 202403 = March 2024 quarter end)."""
    return yyyymm


def _quarter_end_months() -> set[int]:
    """
    Quarter-end months: 3 (March), 6 (June), 9 (September), 12 (December).
    The SCR planilha is published monthly; we keep only quarter-end snapshots
    to match the quarterly IF Data panel.
    """
    return {3, 6, 9, 12}


def _parse_br_float(series: pd.Series) -> pd.Series:
    """
    Convert Brazilian-formatted numbers (comma decimal, period thousand-sep)
    to float.  Handles empty strings and '-' gracefully.
    """
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(r"^\-$", "0", regex=True)   # lone dash → 0
        .str.replace(r"^\s*$", "0", regex=True)  # blank → 0
        .str.replace(r"\.", "", regex=True)       # remove thousand separator
        .str.replace(",", ".", regex=False)       # decimal separator
    )
    return pd.to_numeric(cleaned, errors="coerce")


def load_planilha_file(path: Path, yyyymm: int) -> pd.DataFrame | None:
    """
    Load a single planilha CSV and return a summary row per
    (sr, indexador_is_fixed) with summed value columns.

    Uses pandas' decimal=',' to handle Brazilian number format efficiently,
    loading only the columns required for aggregation.

    Returns None if the file is empty or has no valid rows.
    """
    # Determine which columns to load (avoids reading unused text columns)
    usecols_candidates = ["sr", "indexador"] + VALUE_COLS

    try:
        df = pd.read_csv(
            str(path),
            sep=";",
            encoding="latin-1",
            decimal=",",
            thousands=".",
            low_memory=False,
            # Read all columns; we'll filter below after fixing the BOM name
        )
    except Exception as e:
        logging.warning(f"  Could not read {path.name}: {e}")
        return None

    # Fix BOM in first column name
    df.columns = [
        c.replace("\ufeff", "").replace("\xef\xbb\xbf", "").replace("ï»¿", "").strip()
        for c in df.columns
    ]

    # Keep only the columns we need (sr, indexador, value cols)
    present = set(df.columns)
    keep = [c for c in usecols_candidates if c in present]
    if "sr" not in keep:
        logging.warning(f"  No 'sr' column in {path.name} — skip.")
        return None
    df = df[keep].copy()

    # Filter to known prudential segments
    df = df[df["sr"].isin(VALID_SEGMENTS)]
    if df.empty:
        return None

    # Ensure value columns are numeric (they should be after decimal=',' parsing)
    missing_cols = [c for c in VALUE_COLS if c not in df.columns]
    if missing_cols:
        logging.warning(f"  {path.name}: missing value columns {missing_cols}")
        for c in missing_cols:
            df[c] = np.nan
    for c in VALUE_COLS:
        if c in df.columns and df[c].dtype == object:
            # Fallback: manual string conversion
            df[c] = (
                df[c].astype(str).str.strip()
                .str.replace(r"\.", "", regex=True)
                .str.replace(",", ".", regex=False)
            )
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Fixed-rate flag (case-insensitive prefix match on indexador)
    if "indexador" in df.columns:
        df = df.copy()   # avoid SettingWithCopyWarning
        df["_is_fixed"] = (
            df["indexador"].fillna("").str.lower().str.startswith(PREFIXADO_LABEL)
        ).astype(int)
    else:
        df = df.copy()
        df["_is_fixed"] = 0

    # Aggregate to (sr, _is_fixed) — collapse all other dimensions
    val_cols_present = [c for c in VALUE_COLS if c in df.columns]
    grouped = (
        df.groupby(["sr", "_is_fixed"], observed=True)[val_cols_present]
        .sum(numeric_only=True)
        .reset_index()
    )
    # Add any missing value columns as NaN
    for c in VALUE_COLS:
        if c not in grouped.columns:
            grouped[c] = np.nan
    grouped["AnoMes"] = yyyymm
    return grouped


def build_segment_panel() -> pd.DataFrame:
    """
    Load all quarter-end planilha files, aggregate to (sr × AnoMes), and
    compute credit-structure ratios.
    """
    qend_months = _quarter_end_months()
    files       = _find_planilha_files()

    chunks: list[pd.DataFrame] = []
    skipped = 0
    for path in files:
        yyyymm = _yyyymm_from_path(path)
        month  = yyyymm % 100
        if month not in qend_months:
            continue   # skip non-quarter-end months
        df_chunk = load_planilha_file(path, yyyymm)
        if df_chunk is not None and not df_chunk.empty:
            chunks.append(df_chunk)
        else:
            skipped += 1

    if not chunks:
        raise RuntimeError("No valid planilha data loaded — check SCR data path.")

    logging.info(
        f"Loaded {len(chunks)} quarter-end files | {skipped} files skipped."
    )

    df = pd.concat(chunks, ignore_index=True)

    # Sum further to (sr × AnoMes) by collapsing the _is_fixed dimension
    # so we can compute fixed-rate share from the _is_fixed=1 sub-total
    # ── Total per (sr × AnoMes)
    total = (
        df.groupby(["sr", "AnoMes"], observed=True)[VALUE_COLS]
        .sum()
        .reset_index()
    )
    # ── Fixed-rate sub-total per (sr × AnoMes)
    fixed = (
        df[df["_is_fixed"] == 1]
        .groupby(["sr", "AnoMes"], observed=True)[["carteira_ativa"]]
        .sum()
        .rename(columns={"carteira_ativa": "_fixed_ativa"})
        .reset_index()
    )
    seg = total.merge(fixed, on=["sr", "AnoMes"], how="left")
    seg["_fixed_ativa"] = seg["_fixed_ativa"].fillna(0.0)

    # ── Compute ratios ──────────────────────────────────────────────────────
    ca = seg["carteira_ativa"].replace(0, np.nan)  # avoid divide-by-zero

    # Maturity shares
    seg["scr_short_term_share"]  = (
        (seg["a_vencer_ate_90_dias"] + seg["a_vencer_de_91_ate_360_dias"]) / ca
    )
    seg["scr_medium_term_share"] = (
        (seg["a_vencer_de_361_ate_1080_dias"] + seg["a_vencer_de_1081_ate_1800_dias"]) / ca
    )
    seg["scr_long_term_share"]   = (
        (seg["a_vencer_de_1801_ate_5400_dias"] + seg["a_vencer_acima_de_5400_dias"]) / ca
    )

    # NPL rate and problematic share
    seg["scr_npl_rate"]          = seg["vencido_acima_de_15_dias"] / ca
    seg["scr_problematic_share"] = seg["ativo_problematico"] / ca

    # Fixed-rate credit share
    seg["scr_fixed_rate_share"]  = seg["_fixed_ativa"] / ca

    # Rename sr → segment to match institution panel
    seg = seg.rename(columns={"sr": "segment"})

    keep_cols = [
        "segment", "AnoMes",
        "scr_short_term_share",
        "scr_medium_term_share",
        "scr_long_term_share",
        "scr_npl_rate",
        "scr_fixed_rate_share",
        "scr_problematic_share",
    ]
    seg = seg[keep_cols].sort_values(["segment", "AnoMes"]).reset_index(drop=True)

    # Sanity-check: ratios should be in [0, 1]
    ratio_cols = [c for c in keep_cols if c not in ("segment", "AnoMes")]
    for c in ratio_cols:
        n_out = ((seg[c] < -0.001) | (seg[c] > 1.001)).sum()
        if n_out > 0:
            logging.warning(
                f"  {c}: {n_out} values outside [0,1] — setting to NaN."
            )
            seg.loc[(seg[c] < -0.001) | (seg[c] > 1.001), c] = np.nan

    return seg


def add_lagged_columns(seg: pd.DataFrame) -> pd.DataFrame:
    """
    Add one-quarter-lagged versions of all ratio columns.
    Lag is computed within each prudential segment group.

    The institution panel uses lagged instruments throughout (e.g.
    indice_basileia_lag) — we follow the same convention here.
    """
    ratio_cols = [c for c in seg.columns if c not in ("segment", "AnoMes")]
    seg = seg.sort_values(["segment", "AnoMes"]).copy()
    for c in ratio_cols:
        seg[f"{c}_lag"] = seg.groupby("segment", observed=True)[c].shift(1)
    return seg


def print_summary(seg: pd.DataFrame) -> None:
    logging.info("=" * 60)
    logging.info("SCR segment panel summary:")
    logging.info(f"  Rows: {len(seg):,} | Segments: {seg['segment'].nunique()} | "
                 f"AnoMes: {seg['AnoMes'].min()}–{seg['AnoMes'].max()}")
    ratio_cols = [c for c in seg.columns if "scr_" in c]
    for c in ratio_cols:
        n_ok  = seg[c].notna().sum()
        mu    = seg[c].mean()
        std   = seg[c].std()
        logging.info(f"  {c:<35s}: mean={mu:.4f}  std={std:.4f}  n={n_ok:,}")
    logging.info("=" * 60)


def save_output(seg: pd.DataFrame) -> None:
    seg.to_csv(str(OUT_CSV), index=False)
    logging.info(f"Saved CSV     -> {OUT_CSV}")
    try:
        seg.to_parquet(str(OUT_PARQUET), index=False, compression="snappy")
        logging.info(f"Saved parquet -> {OUT_PARQUET}")
    except Exception as e:
        logging.warning(f"Could not save parquet: {e}")


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    logging.info("=" * 60)
    logging.info("scr_panel_build.py  START")
    logging.info("=" * 60)

    seg      = build_segment_panel()
    seg      = add_lagged_columns(seg)
    print_summary(seg)
    save_output(seg)

    logging.info("scr_panel_build.py  DONE")


if __name__ == "__main__":
    main()

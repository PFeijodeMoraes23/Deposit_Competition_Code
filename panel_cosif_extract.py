## panel_cosif_extract.py
# Author: Pedro Feijo de Moraes
# Objective: CANONICAL COSIF extractor for the deposit-rate pipeline.  Reads the
#            shared monthly COSIF *BANCOS* files and writes the processed
#            custos_implicitos_*.csv that panel_deposit_rates.py consumes via
#            load_cosif_processed(), PLUS the per-type FOUNDATION table that
#            panel_cosif_calibrate.py consumes to build the corrected k=4 rate.
#
# TWO OUTPUTS (single read pass):
#
#  (1) CANONICAL monthly per-taxonomy files  custos_implicitos_<TAXONOMY>.csv
#      The contract panel_deposit_rates.load_cosif_processed reads:
#        * one file per COSIF TAXONOMIA (filename suffix -> cosif_taxonomy),
#        * monthly DATA_BASE,
#        * Estoque_Total / Estoque_Prepago: total-deposit / prepaid STOCK.  These
#          intentionally retain the Jun/Dec 4010+4016 double-count and the
#          group-8 SEMESTER-CUMULATIVE Despesa_Captacao_Marginal, BECAUSE
#          panel_deposit_rates._correct_cosif_time_aggregation expects raw
#          values and applies its own halving + disaccumulation.
#        * Estoque_Total_Lag, Custo_Efetivo_Blended, Custo_Efetivo_Prepago.
#      Plus per-type monthly columns (stk_time/stk_repos/stk_bills/stk_savings/
#      stk_interb from DOC 4010 only, and the 2025+ exp_* expense leaves).
#      load_cosif_processed ignores them; panel_cosif_calibrate reads them.
#
#  (2) FOUNDATION table  custos_implicitos_v2_foundation.csv
#      CNPJ x quarter, DOC 4010 only, per-type stocks + disaccumulated expense +
#      2025+ leaves, already collapsed to quarterly, with `src` (BANCOS or SOCIEDADES).
#      It also holds the credit-finance companies (SCFI) of the SOCIEDADES files, the
#      deposit-taking entity of conglomerates with no bank.  Input of the calibration.
#
# WHY the per-type detail: the blended LUMPED expense (8.1.1, code 81100008) / TOTAL
# deposits gives a CDB rate far outside the plausible 90-115%-of-CDI band.
# panel_cosif_calibrate uses the per-type detail emitted here to build a
# per-bank, segment-shrunk corrected CDB rate.
#
# Raw input : shared/COSIF/*BANCOS* and *SOCIEDADES* (SCFI rows) monthly (201301-202603).
## ---------------------------------------------------------------------------

import os
import re
import sys
import zipfile
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from utils import paths  # noqa: E402

COSIF_RAW_DIR = str(paths.COSIF_RAW)                       # shared/COSIF
PROCESSED_PATH = str(paths.PROCESSED)
COSIF_PROCESSED_PATH = os.path.join(PROCESSED_PATH, "COSIF_PROCESSED")
os.makedirs(COSIF_PROCESSED_PATH, exist_ok=True)

OUT_FOUNDATION = os.path.join(
    COSIF_PROCESSED_PATH, "custos_implicitos_v2_foundation.csv"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("panel_cosif_extract")

# ---------------------------------------------------------------------------
# COSIF account codes (raw CONTA has NO dots/dashes: 8-digit pre-2025,
# 10-digit from the 2025 recode).  Confirmed against the raw data.
# ---------------------------------------------------------------------------
# Pre-2025 (8-digit) STOCKS (parent rows):
PRE = {
    "dep_total":   "41000007",  # DEPOSITOS (total)
    "dep_demand":  "41100000",  # Depositos A Vista
    "dep_savings": "41200003",  # Depositos De Poupanca
    "dep_interb":  "41300006",  # Depositos Interfinanceiros
    "dep_time":    "41500002",  # Depositos A Prazo (CDB)
    "repos":       "42000006",  # Obrigacoes por Operacoes Compromissadas
    "bills":       "43000005",  # Aceites/Letras/Debentures e similares
    "prepaid":     "41930",     # (prefix) Conta Pag. Pre-Paga -- absent pre-2025
}
# Pre-2025 EXPENSE (cumulative within semester; LUMPED = deposits+repos+bills):
PRE_EXP_LUMP = "81100008"       # (-) Despesas de Captacao

# 2025+ (10-digit) STOCKS (parent rows):
NEW = {
    "dep_total":   "4100000009",
    "dep_demand":  "4110000006",
    "dep_savings": "4120000003",
    "dep_interb":  "4130000000",
    "dep_time":    "4150000004",  # Depositos a Prazo (parent total > 4151 child)
    "repos":       "4200000002",
    "bills":       "4300000005",  # Outros Instrumentos de Divida (group 4.3 parent)
    "prepaid":     "4193000009",
}
# 2025+ EXPENSE leaves:
NEW_EXP = {
    "lump":     "8110000002",  # parent = old lump
    "savings":  "8111000001",
    "interb":   "8112000000",
    "cdb":      "8113000009",  # (-) Despesas de Depositos a Prazo  == EXACT CDB cost
    "repos":    "8115000007",
}
NEW_EXP_BILLS = ["8116500001", "8117500000", "8117700006", "8118200000"]  # LCA/LCI/LIG/LF
NEW_EXP_PREPAID = "8119800007"

# All exact codes we keep (parent rows only; do NOT sum children).
KEEP_CODES = set(PRE.values()) | {PRE_EXP_LUMP} \
    | set(NEW.values()) | set(NEW_EXP.values()) \
    | set(NEW_EXP_BILLS) | {NEW_EXP_PREPAID}
# 'prepaid' pre-2025 is a prefix not an exact code; handle separately.
KEEP_CODES.discard("41930")

# Map raw CONTA -> a canonical short column name in the output.
CODE2COL = {
    # stocks (canonical, era-merged)
    PRE["dep_total"]: "stk_total",  NEW["dep_total"]: "stk_total",
    PRE["dep_demand"]: "stk_demand", NEW["dep_demand"]: "stk_demand",
    PRE["dep_savings"]: "stk_savings", NEW["dep_savings"]: "stk_savings",
    PRE["dep_interb"]: "stk_interb", NEW["dep_interb"]: "stk_interb",
    PRE["dep_time"]: "stk_time",    NEW["dep_time"]: "stk_time",
    PRE["repos"]: "stk_repos",      NEW["repos"]: "stk_repos",
    PRE["bills"]: "stk_bills",      NEW["bills"]: "stk_bills",
    NEW["prepaid"]: "stk_prepaid",
    # expense
    PRE_EXP_LUMP: "exp_lump",       NEW_EXP["lump"]: "exp_lump",
    NEW_EXP["savings"]: "exp_savings",
    NEW_EXP["interb"]: "exp_interb",
    NEW_EXP["cdb"]: "exp_cdb",
    NEW_EXP["repos"]: "exp_repos",
    NEW_EXP_PREPAID: "exp_prepaid",
}
for _c in NEW_EXP_BILLS:
    CODE2COL[_c] = "exp_bills"  # several leaves sum into one column

EXPENSE_COLS = [
    "exp_lump", "exp_savings", "exp_interb", "exp_cdb",
    "exp_repos", "exp_bills", "exp_prepaid",
]
STOCK_COLS = [
    "stk_total", "stk_demand", "stk_savings", "stk_interb",
    "stk_time", "stk_repos", "stk_bills", "stk_prepaid",
]


# ---------------------------------------------------------------------------
# Raw file reading
# ---------------------------------------------------------------------------
# Filename: YYYYMM<KIND>.{ZIP|zip|csv.zip|CSV|csv}, KIND = BANCOS or SOCIEDADES.  Skip the " (1)"
# duplicates.  From the SOCIEDADES files only the credit-finance companies (SCFI) are kept: they are
# the deposit-taking entity of conglomerates with no bank (Nu Financeira, Mercado Credito, Neon,
# Will, Facta, ...), whose time deposits the market panel carries from IF.data.
SOURCES = ("BANCOS", "SOCIEDADES")
SCFI_TAXONOMY = "FINANCIAMENTO E INVESTIMENTO"     # "SOC. DE CREDITO, FINANCIAMENTO E INVESTIMENTO"


def discover_files(raw_dir, kind="BANCOS"):
    """Return {YYYYMM: filepath} of the `kind` files, preferring one canonical file per month.
    Preference order: .csv.zip / .zip  >  .CSV / .csv (plain).  Skips the
    duplicate ' (1)' copies entirely."""
    file_re = re.compile(rf"^(\d{{6}}){kind}\.(zip|csv\.zip|csv|ZIP|CSV)$", re.IGNORECASE)
    by_month = {}
    for fname in os.listdir(raw_dir):
        m = file_re.match(fname)
        if not m:
            continue
        ym = m.group(1)
        ext = m.group(2).lower()
        rank = 0 if ext in ("zip", "csv.zip") else 1   # prefer zipped
        prev = by_month.get(ym)
        if prev is None or rank < prev[0]:
            by_month[ym] = (rank, os.path.join(raw_dir, fname))
    return {ym: v[1] for ym, v in sorted(by_month.items())}


def _read_raw(filepath):
    if filepath.upper().endswith(".ZIP"):
        with zipfile.ZipFile(filepath) as zf:
            csvs = [n for n in zf.namelist() if n.upper().endswith(".CSV")]
            if not csvs:
                return pd.DataFrame()
            with zf.open(csvs[0]) as f:
                df = pd.read_csv(f, sep=";", encoding="latin-1", skiprows=3,
                                 dtype=str, on_bad_lines="skip")
    else:
        df = pd.read_csv(filepath, sep=";", encoding="latin-1", skiprows=3,
                         dtype=str, on_bad_lines="skip")
    return df


def read_month(args):
    """Read ONE month of one source (BANCOS, or the SCFI rows of SOCIEDADES), keep only parent
    rows for the codes we need, parse SALDO (reais).  Returns a tidy long frame:
        [CNPJ, TAXONOMIA, SRC, DATA_BASE, DOC, col, SALDO]
    DOC distinguishes 4010 (balancete; the only series with group-8 expenses and
    the clean per-type detail) from 4016 (semester balanco; duplicates group-4
    stocks at Jun/Dec).  Both are kept here so the CANONICAL Estoque_Total can
    retain the 4010+4016 Jun/Dec double-count that panel_deposit_rates halves, while
    the per-type detail columns are taken from 4010 only downstream."""
    ym, filepath, kind = args
    try:
        df = _read_raw(filepath)
        if df.empty:
            return pd.DataFrame()
        df.columns = [c.upper().lstrip("#").strip() for c in df.columns]
        need = {"DOCUMENTO", "CNPJ", "CONTA", "SALDO"}
        if not need.issubset(df.columns):
            log.warning("%s %s missing columns %s", ym, kind, need - set(df.columns))
            return pd.DataFrame()
        if kind == "SOCIEDADES":
            if "TAXONOMIA" not in df.columns:
                log.warning("%s SOCIEDADES has no TAXONOMIA column: SCFI rows cannot be told apart", ym)
                return pd.DataFrame()
            df = df[df["TAXONOMIA"].astype(str).str.upper().str.contains(SCFI_TAXONOMY, regex=False)]

        df["DOC"] = df["DOCUMENTO"].astype(str).str.strip()
        df = df[df["DOC"].isin(["4010", "4016"])]
        if df.empty:
            return pd.DataFrame()

        df["CONTA"] = df["CONTA"].astype(str).str.strip()

        # Keep prepaid pre-2025 by prefix; everything else by exact code.
        is_prepaid_pre = df["CONTA"].str.startswith(PRE["prepaid"])
        mask = df["CONTA"].isin(KEEP_CODES) | is_prepaid_pre
        df = df[mask].copy()
        if df.empty:
            return pd.DataFrame()

        # SALDO in reais, Brazilian format ('1.234.567,89').
        df["SALDO"] = pd.to_numeric(
            df["SALDO"].astype(str)
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False),
            errors="coerce",
        )
        df["col"] = df["CONTA"].map(CODE2COL)
        df.loc[is_prepaid_pre, "col"] = "stk_prepaid"
        df = df.dropna(subset=["col"])

        df["DATA_BASE"] = pd.to_datetime(ym, format="%Y%m")
        df["CNPJ"] = df["CNPJ"].astype(str).str.zfill(8)
        if "TAXONOMIA" in df.columns:
            df["TAXONOMIA"] = df["TAXONOMIA"].astype(str)
        else:
            df["TAXONOMIA"] = kind
        df["SRC"] = kind

        return df[["CNPJ", "TAXONOMIA", "SRC", "DATA_BASE", "DOC", "col", "SALDO"]]
    except Exception as e:   # noqa: BLE001
        log.error("read %s %s failed: %s", ym, kind, e)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Disaccumulation (the rule panel_deposit_rates._correct_cosif_time_aggregation applies)
# ---------------------------------------------------------------------------
def disaccumulate_within_semester(df, value_col, id_cols, date_col="DATA_BASE"):
    """COSIF group-8 result accounts accumulate within the SEMESTER (reset Jan &
    Jul).  Difference within each (id, year, semester) run to recover the true
    monthly flow; the first month of a run keeps its level.  Differencing across
    an internal gap still yields the correct cumulative increment."""
    df = df.sort_values(id_cols + [date_col]).copy()
    yr = df[date_col].dt.year
    sem = (df[date_col].dt.month > 6).astype(int)
    grp = id_cols + ["_yr", "_sem"]
    df["_yr"] = yr
    df["_sem"] = sem
    df[value_col] = (
        df.groupby(grp)[value_col]
        .transform(lambda s: s.diff().where(s.shift(1).notna(), s))
    )
    df.drop(columns=["_yr", "_sem"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# Read all months once into a single long frame.
# ---------------------------------------------------------------------------
def read_all():
    tasks = []
    for kind in SOURCES:
        files = discover_files(COSIF_RAW_DIR, kind)
        if not files:
            log.error("No %s files under %s", kind, COSIF_RAW_DIR)
            sys.exit(1)
        log.info("Reading %d %s months (%s..%s)", len(files), kind, min(files), max(files))
        tasks += [(ym, fp, kind) for ym, fp in files.items()]
    frames = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(read_month, t): t for t in tasks}
        done = 0
        for fut in as_completed(futs):
            done += 1
            d = fut.result()
            if d is not None and not d.empty:
                frames.append(d)
            if done % 30 == 0:
                log.info("  ... %d/%d months read", done, len(tasks))
    if not frames:
        log.error("All months empty.")
        sys.exit(1)
    long = pd.concat(frames, ignore_index=True)
    del frames
    log.info("Raw long rows: %d", len(long))
    return long


# ---------------------------------------------------------------------------
# (1) CANONICAL monthly per-taxonomy custos_implicitos_<TAXONOMY>.csv
# ---------------------------------------------------------------------------
def build_canonical_taxonomy_files(long):
    """Emit the custos_implicitos_<TAXONOMY>.csv files that
    panel_deposit_rates.load_cosif_processed() consumes.

    The canonical columns are raw, so panel_deposit_rates'
    _correct_cosif_time_aggregation (Jun/Dec halving + within-semester
    disaccumulation) applies correctly:
      * Estoque_Total / Estoque_Prepago : total / prepaid STOCK summed over
        DOC 4010+4016 (retains the Jun/Dec double-count),
      * Despesa_Captacao_Marginal / Desp_Prepago_Marginal : group-8 expense from
        DOC 4010 only, left SEMESTER-CUMULATIVE (panel_deposit_rates disaccumulates it).
    Plus per-type detail columns (DOC 4010 only) for panel_cosif_calibrate.
    """
    # ---- canonical totals: 4010+4016 for stocks, 4010-only for expenses ----
    stock_cols_canon = {"stk_total": "Estoque_Total", "stk_prepaid": "Estoque_Prepago"}
    exp_cols_canon = {"exp_lump": "Despesa_Captacao_Marginal",
                      "exp_prepaid": "Desp_Prepago_Marginal"}

    # Stocks (Estoque_Total / Estoque_Prepago): both documents.
    stk = (long[long["col"].isin(stock_cols_canon)]
           .groupby(["TAXONOMIA", "CNPJ", "DATA_BASE", "col"])["SALDO"]
           .sum().unstack("col").reset_index().rename(columns=stock_cols_canon))

    # Expenses (semester-cumulative): DOC 4010 only.
    exp = (long[(long["DOC"] == "4010") & long["col"].isin(exp_cols_canon)]
           .groupby(["TAXONOMIA", "CNPJ", "DATA_BASE", "col"])["SALDO"]
           .sum().unstack("col").reset_index().rename(columns=exp_cols_canon))

    # EXTRA per-type detail (DOC 4010 only): all stk_*/exp_* except the two
    # canonical stocks (already covered) -- kept raw-monthly for panel_cosif_calibrate.
    detail_cols = [c for c in (STOCK_COLS + EXPENSE_COLS)
                   if c not in ("stk_total", "stk_prepaid")]
    det = (long[(long["DOC"] == "4010") & long["col"].isin(detail_cols)]
           .groupby(["TAXONOMIA", "CNPJ", "DATA_BASE", "col"])["SALDO"]
           .sum().unstack("col").reset_index())

    canon = stk.merge(exp, on=["TAXONOMIA", "CNPJ", "DATA_BASE"], how="outer")
    canon = canon.merge(det, on=["TAXONOMIA", "CNPJ", "DATA_BASE"], how="outer")
    for c in ["Estoque_Total", "Estoque_Prepago",
              "Despesa_Captacao_Marginal", "Desp_Prepago_Marginal"]:
        if c not in canon.columns:
            canon[c] = np.nan
    canon = canon.sort_values(["TAXONOMIA", "CNPJ", "DATA_BASE"])

    # Lag + blended/prepaid effective cost (legacy convenience columns, monthly %).
    canon["Estoque_Total_Lag"] = canon.groupby(["TAXONOMIA", "CNPJ"])["Estoque_Total"].shift(1)
    canon["Estoque_Prepago_Lag"] = canon.groupby(["TAXONOMIA", "CNPJ"])["Estoque_Prepago"].shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        canon["Custo_Efetivo_Blended"] = (
            canon["Despesa_Captacao_Marginal"].abs() / canon["Estoque_Total_Lag"] * 100)
        canon["Custo_Efetivo_Prepago"] = (
            canon["Desp_Prepago_Marginal"].abs() / canon["Estoque_Prepago_Lag"] * 100)
    for c in ["Custo_Efetivo_Blended", "Custo_Efetivo_Prepago"]:
        canon.loc[~np.isfinite(canon[c]), c] = np.nan

    # One file per taxonomy (filename suffix -> cosif_taxonomy on read).
    n_files = 0
    for taxo, g in canon.groupby("TAXONOMIA"):
        safe = re.sub(r'[\\/*?:"<>|]', "", str(taxo)).replace(" ", "_")
        out = os.path.join(COSIF_PROCESSED_PATH, f"custos_implicitos_{safe}.csv")
        g.drop(columns=["TAXONOMIA"]).to_csv(out, index=False)
        n_files += 1
    log.info("Wrote %d canonical custos_implicitos_<TAXONOMY>.csv files", n_files)


# ---------------------------------------------------------------------------
# (2) CNPJ x quarter FOUNDATION table (DOC 4010 only, disaccumulated)
# ---------------------------------------------------------------------------
def build_foundation(long):
    sub = long[long["DOC"] == "4010"]
    # The source is a property of the CNPJ-month: an institution can file as an SCFI for years and
    # as a bank afterwards.  In a month where it appears in both files only its BANCOS rows count
    # (summing the two would count it twice).
    in_bancos = (sub["SRC"] == "BANCOS").groupby([sub["CNPJ"], sub["DATA_BASE"]]).transform("any")
    sub = sub[(sub["SRC"] == "BANCOS") | ~in_bancos]
    src_month = sub.groupby(["CNPJ", "DATA_BASE"])["SRC"].first().reset_index()
    # Pivot CNPJ x month x col.  Sum handles the multi-leaf 'exp_bills' column.
    wide = (
        sub.groupby(["CNPJ", "DATA_BASE", "col"])["SALDO"]
        .sum().unstack("col").reset_index()
    )
    for c in STOCK_COLS + EXPENSE_COLS:
        if c not in wide.columns:
            wide[c] = np.nan

    # --- Disaccumulate every expense flow within (CNPJ, semester) ---
    for c in EXPENSE_COLS:
        if wide[c].notna().any():
            wide = disaccumulate_within_semester(wide, c, ["CNPJ"])

    # --- Collapse monthly -> quarterly ---
    wide["year"] = wide["DATA_BASE"].dt.year
    wide["month"] = wide["DATA_BASE"].dt.month
    wide["q_month"] = ((wide["month"] - 1) // 3 + 1) * 3
    wide["AnoMes"] = wide["year"] * 100 + wide["q_month"]

    # Stocks: quarter-end (months 3,6,9,12).
    stocks_q = (
        wide[wide["month"].isin([3, 6, 9, 12])]
        [["CNPJ", "AnoMes"] + STOCK_COLS].copy()
    )
    # Expense: sum the (disaccumulated) monthly flows in the quarter.
    exp_q = (
        wide.groupby(["CNPJ", "AnoMes"])[EXPENSE_COLS]
        .sum(min_count=1).reset_index()
    )
    found = stocks_q.merge(exp_q, on=["CNPJ", "AnoMes"], how="outer")
    found = found.sort_values(["CNPJ", "AnoMes"]).reset_index(drop=True)
    # src of a CNPJ-quarter: BANCOS if any of its months is read from the BANCOS files, else
    # SOCIEDADES (the SCFI rows).  panel_cosif_calibrate uses SCFI rows only for a
    # conglomerate-quarter without a BANCOS member.
    src_month["AnoMes"] = (src_month["DATA_BASE"].dt.year * 100
                           + ((src_month["DATA_BASE"].dt.month - 1) // 3 + 1) * 3)
    src_q = (src_month.assign(_bank=src_month["SRC"] == "BANCOS")
             .groupby(["CNPJ", "AnoMes"])["_bank"].any().reset_index())
    found = found.merge(src_q, on=["CNPJ", "AnoMes"], how="left")
    found["src"] = np.where(found["_bank"].fillna(False).astype(bool), "BANCOS", "SOCIEDADES")
    found = found.drop(columns=["_bank"])

    # Lagged stocks (previous quarter) per CNPJ for rate denominators.
    for c in STOCK_COLS:
        found[c + "_lag"] = found.groupby("CNPJ")[c].shift(1)

    found.to_csv(OUT_FOUNDATION, index=False)
    log.info("Wrote foundation: %s  (%d rows, %d CNPJs, %d quarters)",
             OUT_FOUNDATION, len(found), found["CNPJ"].nunique(),
             found["AnoMes"].nunique())
    return found


def main():
    long = read_all()
    # the per-taxonomy files panel_deposit_rates reads stay on the BANCOS files
    build_canonical_taxonomy_files(long[long["SRC"] == "BANCOS"])
    build_foundation(long)


if __name__ == "__main__":
    main()

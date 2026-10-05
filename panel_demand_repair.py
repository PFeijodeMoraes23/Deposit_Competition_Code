"""
panel_demand_repair.py
======================
Writes the demand-side repair file, `processed/demand_repair.parquet`, which
`utils/demand_repair.py` applies inside the three demand preps when `DEMAND_REPAIR=1`.

It WRITES NO EXISTING FILE. market_panel.csv / .parquet, bank_chars_panel.csv and
deposits_panel.csv are read and left as they are, so the sleepiness estimators and the policy
function, which read the panel directly, are unaffected by this step.

What the file repairs (each value carries a flag; see FLAGS below)
-----------------------------------------------------------------
1. The lags of 2016Q1.
   a. Balances of 2013-2015. panel_deposits.load_estban_raw_pre2016 lists the raw monthly
      ESTBAN files with `glob("*.CSV") + glob("*.csv")`; on a case-insensitive file system each
      file is listed twice and the build sums both copies. The factor is MEASURED here,
      municipality by municipality, against the raw files (it must be exactly 2 or exactly 1 in
      every municipality of a quarter), and the stored balances are divided by it.
   b. The 2015Q4 balance of the national firms that only IF.data reports (their conglomerate
      has no value in the prudential report before 2016-03): the 2016Q1 balance of the panel
      times the one-quarter change of the member institutions (individual-institution report).
   c. The lagged characteristics of 2016Q1. The prudential report of 2015 holds credit
      co-operatives only, so no bank has a 2015Q4 row. They are built as
          X_prud(2015Q4) = X_prud(2016Q1) * X_members(2015Q4) / X_members(2016Q1)
      (a difference in logs for assets): the level from the prudential basis, the one-quarter
      change from the member institutions, members fixed at the List of 2016-03. Flow ratios use
      quarterly flows (the statement is cumulative within the semester: Q4 = December minus
      September). The Basel index exists on no basis for 2015Q4: the firm's own 2016Q1 value.
   d. The rival-branch count of 2015Q4 (estban_rival_branches_lag of 2016Q1) from the raw
      2015-12 ESTBAN file, with panel_estban_instrument.py's own mapping.
2. The other firm-quarters with no lagged characteristics (and those whose lagged total assets
   are a stored zero, i.e. a blank filing): the TRUE lag when the firm filed at t-1 and only its
   row at t is absent; else the nearest filing within MAX_DISTANCE quarters (a tie goes to the
   earlier quarter); else none, and the prep drops the rows.
3. Prudential segments: a firm-quarter with no recorded segment takes the firm's first recorded
   segment (quarters before it) or the last recorded one before the quarter (holes after it).
   A firm that was never classified stays S1 and is flagged.
4. The ratio of borrowings and onlendings (IF.data Passivo, line 78295) to total assets, built
   at one date and lagged like every other ratio, and its leave-one-out pair, under the names
   borrow_assets_lag, loo_borrow_assets, mean_loo_borrow_assets.

Leave-one-out pairs are rebuilt with panel_loo_instruments.compute_loo_instruments' rule
unchanged (a rival with a missing value counts as 0 in the sum and stays in the count), once
after repair 1 (`s1__*`) and once after repairs 1 and 2 (`s2__*`). A market-quarter in which
no firm was repaired keeps its stored values.

FLAGS (on every row of 2016-2024)
---------------------------------
chars_src           0 as stored; 1 true lag recovered by calendar; 2 the 2015Q4 values of 1c;
                    3 nearest filing; 9 none within MAX_DISTANCE (rows dropped by the prep)
chars_dist          quarters between the source filing and t-1 (0 unless chars_src == 3)
basel_imputed       1 where the Basel index (t-1) is the firm's own value at t (chars_src == 2)
segment_src         0 recorded; 1 the firm's first recorded segment; 2 carried from the firm's
                    neighbouring quarter; 3 never classified, left at S1
lag_from_2015q4     1 on the rows of 2016Q1
lag_dep_src         2016Q1 rows: 1 lagged balance from a corrected ESTBAN row, 2 from member
                    institutions (1b), 0 none
estban_lag_src      1 where estban_rival_branches_lag comes from the raw 2015-12 file
borrow_changed      1 where the re-timed ratio differs from a finite stored credit_assets_lag
borrow_den_guarded  1 where the stored ratio divided by a zero lagged total

Reads
-----
processed/market_panel.parquet, processed/bank_chars_panel.csv, processed/deposits_panel.csv,
processed/COSIF_PROCESSED/cosif_cdb_rate_corrected.csv,
IF_DATA/Individual Institutions/IF_DATA_Values_{2015_9, 2015_12, 2016_3, 2016_6}.csv,
IF_DATA/List/IF_DATA_List_2016_3.csv (+ every List file, through the CNPJ -> conglomerate map of
panel_cosif_calibrate), ESTBAN/Relatório por município/{2013..2015 quarter-ends, 201603,
201606}_ESTBAN.CSV, IBGE/muni_mca_regions_2010_2024_panel.csv.

Writes
------
processed/demand_repair.parquet          the repair file
processed/demand_repair_report.json      counts and the results of the identity checks

Usage:  python panel_demand_repair.py
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys

try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None
if ensure_project_venv is not None:
    ensure_project_venv(__file__)

import numpy as np
import pandas as pd

from utils import paths
from utils import demand_repair as dr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("panel_demand_repair")

C = "CodConglomeradoPrudencial"
KEYS = dr.KEYS
PANEL_PARQUET = paths.PROCESSED / "market_panel.parquet"
BANK_CHARS_CSV = paths.PROCESSED / "bank_chars_panel.csv"
DEPOSITS_CSV = paths.PROCESSED / "deposits_panel.csv"
CORRECTED_RATE_CSV = paths.PROCESSED / "COSIF_PROCESSED" / "cosif_cdb_rate_corrected.csv"
MCA_XWALK = paths.IBGE_DIR / "muni_mca_regions_2010_2024_panel.csv"
OUT_PARQUET = paths.PROCESSED / "demand_repair.parquet"
OUT_REPORT = paths.PROCESSED / "demand_repair_report.json"

MAX_DISTANCE = 4            # quarters, nearest-filing rule
RATE_A4_MAX_MULT = 3.0      # panel_deposit_rates.py's plausibility band for the type-4 rate
MIN_YEAR, MAX_YEAR = 2016, 2024
WINSORIZED = ["personnel_cost_ratio_lag", "admin_cost_ratio_lag", "tax_cost_ratio_lag",
              "indice_basileia_lag"]          # clipped in the panel (panel_loo_instruments.py)
BASEL_BOUNDS = (-1.0, 2.0)                    # utils/winsorize.VALIDITY_BOUNDS

# IF.data lines (reports 1-4), as panel_bank_chars.py and panel_deposits.py read them.
LINES = {78182: "total_assets", 78186: "equity", 78192: "npl_provision",
         78295: "emprestimos_repasses", 78218: "personnel_expenses", 78219: "admin_expenses",
         78220: "tax_expenses", 78282: "dep_a1", 78283: "dep_a2", 78286: "dep_a4"}
FLOWS = ["personnel_expenses", "admin_expenses", "tax_expenses"]
# characteristic -> (column of bank_chars_panel.csv at the source filing, panel column at t)
LEVEL_OF = {"log_total_assets_lag": "log_total_assets", "equity_ratio_lag": "equity_ratio",
            "personnel_cost_ratio_lag": "personnel_cost_ratio", "admin_cost_ratio_lag": "admin_cost_ratio",
            "tax_cost_ratio_lag": "tax_cost_ratio", "indice_basileia_lag": "indice_basileia",
            "npl_provision_ratio_lag": "npl_provision_ratio", "total_assets_lag": "total_assets"}
LOO_CHARS = {"log_assets": "log_total_assets_lag", "equity_ratio": "equity_ratio_lag",
             "basileia": "indice_basileia_lag", "credit_assets": "credit_assets_lag",
             "npl_provision": "npl_provision_ratio_lag"}


def tq(year, quarter):
    return np.asarray(year) * 4 + np.asarray(quarter)


def md5(path, chunk=1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# ----------------------------------------------------------------------------- inputs
def load_panel() -> pd.DataFrame:
    cols = KEYS + ["is_B", "Source", "dep_a1", "dep_a2", "dep_a4", "risk_free_qoq", "rate_a4",
                   "spread_a1", "spread_a2", "spread_ann_a1", "spread_ann_a2",
                   "segment", "seg_S2", "seg_S3", "seg_S4", "seg_S5", "total_assets",
                   "emprestimos_repasses", "credit_assets_lag", "n_rivals",
                   "estban_rival_branches_lag"] + list(LEVEL_OF) + dr.LOO_COLS
    p = pd.read_parquet(PANEL_PARQUET, columns=list(dict.fromkeys(cols)))
    p[C] = p[C].astype(str)
    p["mca_code"] = p["mca_code"].astype(str)
    if p.duplicated(KEYS).any():
        raise ValueError("market panel is not unique on its key")
    return p


def load_bank_chars() -> pd.DataFrame:
    use = [C, "year", "quarter", "segment", "emprestimos_repasses"] + sorted(set(LEVEL_OF.values()))
    b = pd.read_csv(BANK_CHARS_CSV, usecols=use, low_memory=False)
    b[C] = b[C].astype(str)
    b["tq"] = tq(b["year"], b["quarter"])
    b = b.sort_values([C, "tq"]).reset_index(drop=True)
    pos = b["total_assets"] > 0
    b["log_total_assets"] = np.where(pos, np.log(b["total_assets"].where(pos)), np.nan)
    # borrowings and onlendings over total assets of the SAME date; missing if the total is not positive
    b["borrow_assets"] = b["emprestimos_repasses"] / b["total_assets"].where(pos)
    return b


def read_member_values(year: int, month: int) -> pd.DataFrame:
    """Individual-institution report (type 3) of one date: one row per institution and line."""
    fp = paths.IF_DATA_INDIVIDUAL / f"IF_DATA_Values_{year}_{month}.csv"
    d = pd.read_csv(fp, sep=",", encoding="latin1", dtype=str,
                    usecols=["CodInst", "NumeroRelatorio", "Conta", "Saldo"])
    d["Conta"] = pd.to_numeric(d["Conta"], errors="coerce")
    d = d[d["Conta"].isin(LINES) & d["NumeroRelatorio"].isin(["1", "2", "3", "4"])].copy()
    d["val"] = pd.to_numeric(d["Saldo"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    d["CodInst"] = d["CodInst"].astype(str).str.strip()
    d["line"] = d["Conta"].map(LINES)
    # total assets is carried by the Resumo and by the Ativo: one copy per institution and line
    return d.sort_values("NumeroRelatorio").drop_duplicates(["CodInst", "Conta"], keep="first")


def member_map() -> pd.Series:
    """Institution (CNPJ) -> prudential conglomerate, from the List of 2016-03; an institution with
    no conglomerate stands for itself."""
    L = pd.read_csv(paths.IF_DATA_LIST / "IF_DATA_List_2016_3.csv", dtype=str, encoding="latin1")
    L.columns = [c.strip() for c in L.columns]
    for c in ("CodInst", "CodConglomeradoPrudencial"):
        L[c] = L[c].astype(str).str.strip().replace({"null": np.nan, "nan": np.nan, "": np.nan})
    inst = L[~L["CodInst"].str.startswith("C")].drop_duplicates("CodInst")
    prud = inst["CodConglomeradoPrudencial"].where(inst["CodConglomeradoPrudencial"].notna(), inst["CodInst"])
    return pd.Series(prud.to_numpy(), index=inst["CodInst"].to_numpy())


def member_sums(year: int, month: int, memb: pd.Series) -> pd.DataFrame:
    """Lines summed over the member institutions of each prudential conglomerate."""
    d = read_member_values(year, month)
    d["prud"] = d["CodInst"].map(memb)
    d = d.dropna(subset=["prud"])
    w = d.pivot_table(index="prud", columns="line", values="val",
                      aggfunc=lambda s: s.sum(min_count=1))
    for c in LINES.values():
        if c not in w.columns:
            w[c] = np.nan
    w["n_members"] = d.groupby("prud")["CodInst"].nunique()
    return w


def quarter_flows(cum_first: pd.DataFrame, cum_second: pd.DataFrame) -> pd.DataFrame:
    """Second quarter of a semester: reported (semester-cumulative) minus the first quarter's."""
    out = cum_second.copy()
    for c in FLOWS:
        out[c] = cum_second[c] - cum_first[c].reindex(cum_second.index)
    return out


def member_chars(w: pd.DataFrame) -> pd.DataFrame:
    """The characteristics on the member-sum basis (flows in `w` must be single-quarter)."""
    a = w["total_assets"].where(w["total_assets"] > 0)
    o = pd.DataFrame(index=w.index)
    o["log_total_assets"] = np.log(a)
    o["equity_ratio"] = w["equity"] / a
    o["personnel_cost_ratio"] = w["personnel_expenses"].abs() / a
    o["admin_cost_ratio"] = w["admin_expenses"].abs() / a
    o["tax_cost_ratio"] = w["tax_expenses"].abs() / a
    o["npl_provision_ratio"] = w["npl_provision"].abs() / a
    o["borrow_assets"] = w["emprestimos_repasses"] / a
    return o


SPLICED = ["log_total_assets", "equity_ratio", "personnel_cost_ratio", "admin_cost_ratio",
           "tax_cost_ratio", "npl_provision_ratio", "borrow_assets"]


def splice_back(own_T: pd.DataFrame, mem_prev: pd.DataFrame, mem_T: pd.DataFrame):
    """One quarter backwards: the level from the prudential basis at T, the change from the members.

    own_T     prudential characteristics at T (index = conglomerate)
    mem_prev  member-sum characteristics at T-1;  mem_T  at T
    Returns (values at T-1, how each value was formed: 1 splice, 2 member value where the prudential
    ratio is zero, 3 the prudential value of T where the members give no ratio). A firm with no
    member assets at either date gets nothing."""
    idx = own_T.index
    out = pd.DataFrame(index=idx, columns=SPLICED, dtype=float)
    how = pd.DataFrame(0, index=idx, columns=SPLICED, dtype="int8")
    ok_firm = mem_prev["log_total_assets"].reindex(idx).notna() & mem_T["log_total_assets"].reindex(idx).notna() \
        & own_T["log_total_assets"].notna()
    for v in SPLICED:
        own, ip, it = own_T[v], mem_prev[v].reindex(idx), mem_T[v].reindex(idx)
        if v == "log_total_assets":
            val = own + (ip - it)
            h = pd.Series(1, index=idx)
        else:
            ratio_ok = ip.notna() & it.notna() & (it.abs() > 0)
            val = (own * (ip / it.where(ratio_ok))).where(ratio_ok)
            h = pd.Series(np.where(ratio_ok, 1, 0), index=idx)
            use_member = val.isna() & (own == 0) & ip.notna()
            val = val.where(~use_member, ip); h = h.where(~use_member, 2)
            carry = val.isna() & own.notna()
            val = val.where(~carry, own); h = h.where(~carry, 3)
        out[v] = val.where(ok_firm)
        how[v] = h.where(ok_firm, 0).astype("int8")
    return out, how


# ----------------------------------------------------------------------------- 1a balances
def read_raw_estban(yyyymm: str, cols: dict) -> pd.DataFrame:
    """One raw monthly ESTBAN file with the read options of panel_deposits.process_raw_estban_csv.
    `cols` maps an output name to a substring of the raw header."""
    fp = paths.ESTBAN_RAW_MUN / f"{yyyymm}_ESTBAN.CSV"
    d = pd.read_csv(fp, sep=";", decimal=",", encoding="latin1", skiprows=2, low_memory=False,
                    dtype={"CNPJ": str})
    d.columns = [c.strip().upper().replace(" ", "_") for c in d.columns]
    out = pd.DataFrame({"CNPJ": d["CNPJ"].astype(str).str.strip().str.zfill(8),
                        "CODMUN_IBGE": pd.to_numeric(d["CODMUN_IBGE"], errors="coerce")})
    for name, sub in cols.items():
        hit = [c for c in d.columns if sub in c]
        if not hit:
            raise KeyError(f"{fp.name}: no column containing {sub!r}")
        out[name] = pd.to_numeric(d[hit[0]], errors="coerce")
    out = out[out["CODMUN_IBGE"].notna() & (out["CODMUN_IBGE"] != 0)].copy()
    out["CODMUN_IBGE"] = out["CODMUN_IBGE"].astype(int)
    return out


DEP_SUBSTR = {"dep_a1": "VERBETE_401_", "dep_a2": "VERBETE_420_", "dep_a4": "VERBETE_432_"}


def pre2016_scale(report: dict) -> dict:
    """Stored balance over raw balance for each quarter-end of 2013-2015, measured municipality by
    municipality on deposits_panel.csv (Source == ESTBAN). Every municipality of a quarter must give
    the same integer (2 where the raw file was read twice, 1 where it was read once)."""
    dp = pd.read_csv(DEPOSITS_CSV, low_memory=False,
                     usecols=["CODMUN_IBGE", "Year", "Quarter", "Source"] + dr.DEP_COLS)
    dp = dp[(dp["Source"] == "ESTBAN") & (dp["Year"] <= 2016)]
    scale, detail = {}, {}
    quarters = [(y, q) for y in (2013, 2014, 2015) for q in (1, 2, 3, 4)] + [(2016, 1)]
    for (y, q) in quarters:
        raw = read_raw_estban(f"{y}{3 * q:02d}", DEP_SUBSTR).groupby("CODMUN_IBGE")[dr.DEP_COLS].sum()
        pan = dp[(dp["Year"] == y) & (dp["Quarter"] == q)].groupby("CODMUN_IBGE")[dr.DEP_COLS].sum()
        j = raw.join(pan, lsuffix="_raw", rsuffix="_pan", how="outer")
        ratios = []
        for c in dr.DEP_COLS:
            ok = j[f"{c}_raw"] > 0
            ratios.append((j.loc[ok, f"{c}_pan"] / j.loc[ok, f"{c}_raw"]).to_numpy())
        r = np.concatenate(ratios)
        f = float(np.round(np.nanmedian(r)))
        worst = float(np.nanmax(np.abs(r - f)))
        if f not in (1.0, 2.0) or not np.isfinite(worst) or worst > 1e-9:
            raise AssertionError(f"balances of {y}Q{q}: stored/raw is not one integer in every municipality "
                                 f"(median {np.nanmedian(r):.6f}, worst deviation {worst:.3e})")
        scale[(y, q)] = f
        detail[f"{y}Q{q}"] = {"factor": f, "municipalities": int(len(j)), "cells": int(len(r)),
                              "worst_deviation": worst}
    if scale.pop((2016, 1)) != 1.0:
        raise AssertionError("balances of 2016Q1 are not on the scale of the raw file")
    report["pre2016_balance_scale"] = detail
    return scale


def check_panel_against_deposits(panel: pd.DataFrame, report: dict) -> None:
    """The market panel's 2015Q4 balances are the deposits file's, summed to the market: so the
    factor measured on the deposits file is the panel's."""
    dp = pd.read_csv(DEPOSITS_CSV, low_memory=False,
                     usecols=[C, "CODMUN_IBGE", "Year", "Quarter", "Source"] + dr.DEP_COLS)
    dp = dp[(dp["Year"] == 2015) & (dp["Quarter"] == 4) & (dp["Source"] == "ESTBAN")].copy()
    dp[C] = dp[C].astype(str)
    xw = pd.read_csv(MCA_XWALK, usecols=["municipality_code", "mca_code", "year"],
                     dtype={"municipality_code": int, "mca_code": str, "year": int})
    xw = xw[xw["year"] == 2015]
    dp = dp.merge(xw[["municipality_code", "mca_code"]], left_on="CODMUN_IBGE",
                  right_on="municipality_code", how="left")
    # panel_market.attach_mca_code keeps a municipality the crosswalk does not hold as its own market
    unmapped = dp["mca_code"].isna()
    dp.loc[unmapped, "mca_code"] = "UNKNOWN_" + dp.loc[unmapped, "CODMUN_IBGE"].astype(int).astype(str)
    p = panel[(panel["year"] == 2015) & (panel["quarter"] == 4)]
    loc = p[p["is_B"] == 1].set_index([C, "mca_code"])[dr.DEP_COLS]
    agg = dp.groupby([C, "mca_code"])[dr.DEP_COLS].sum(min_count=1)
    j = loc.join(agg, rsuffix="_dep", how="left")
    same = np.ones(len(j), bool)
    for c in dr.DEP_COLS:
        a, b = j[c].to_numpy(float), j[c + "_dep"].to_numpy(float)
        same &= np.isclose(a, b, rtol=1e-9, atol=1e-6, equal_nan=True)
    nat = p[p["is_B"] == 0].set_index(C)[dr.DEP_COLS]
    aggn = dp.groupby(C)[dr.DEP_COLS].sum(min_count=1).reindex(nat.index)
    same_n = np.ones(len(nat), bool)
    for c in dr.DEP_COLS:
        same_n &= np.isclose(nat[c].to_numpy(float), aggn[c].to_numpy(float), rtol=1e-9, atol=1e-6, equal_nan=True)
    report["panel_equals_deposits_file_2015q4"] = {
        "local_rows": int(len(j)), "local_rows_equal": int(same.sum()),
        "national_rows": int(len(nat)), "national_rows_equal": int(same_n.sum())}
    share = (same.sum() + same_n.sum()) / max(1, len(j) + len(nat))
    if share < 0.999:
        raise AssertionError(f"only {share:.4f} of the panel's 2015Q4 rows equal the deposits file summed "
                             "to the market: the factor measured on that file cannot be applied to the panel")


# ----------------------------------------------------------------------------- 1c, 2 characteristics
def stored_cutoffs(panel: pd.DataFrame) -> dict:
    """The clip bounds the panel build applied to each winsorized column, by firm type: the smallest
    and the largest stored value (a clipped column sits exactly on its bounds)."""
    out = {}
    for c in WINSORIZED:
        for t in (0, 1):
            v = panel.loc[panel["is_B"] == t, c].dropna()
            out[(c, t)] = (float(v.min()), float(v.max()))
    return out


def clip_like_panel(values: pd.DataFrame, is_b: pd.Series, cuts: dict) -> pd.DataFrame:
    values = values.copy()
    if "indice_basileia_lag" in values:
        values["indice_basileia_lag"] = values["indice_basileia_lag"].clip(*BASEL_BOUNDS)
    for c in WINSORIZED:
        for t in (0, 1):
            m = (is_b == t).to_numpy()
            lo, hi = cuts[(c, t)]
            values.loc[m, c] = values.loc[m, c].clip(lo, hi)
    return values


def build_firm_quarters(panel: pd.DataFrame, bc: pd.DataFrame, memb: pd.Series, report: dict):
    """One row per firm-quarter of 2016-2024: the repaired lagged characteristics, the segment, the
    re-timed borrowings ratio, and their flags."""
    win = panel[panel["year"].between(MIN_YEAR, MAX_YEAR)]
    stored_cols = list(LEVEL_OF) + ["segment"] + dr.SEG_COLS + ["total_assets", "emprestimos_repasses",
                                                              "credit_assets_lag", "is_B"]
    fq = win.drop_duplicates([C, "year", "quarter"])[[C, "year", "quarter"] + stored_cols].reset_index(drop=True)
    fq["tq"] = tq(fq["year"], fq["quarter"])
    is_q1 = (fq["year"] == 2016) & (fq["quarter"] == 1)

    # --- 1c: 2015Q4 by the member splice -------------------------------------------------------
    m15_9, m15_12, m16_3 = member_sums(2015, 9, memb), member_sums(2015, 12, memb), member_sums(2016, 3, memb)
    mem_prev = member_chars(quarter_flows(m15_9, m15_12))       # 2015Q4 flows = December - September
    mem_T = member_chars(m16_3)                                 # 2016Q1 flows as reported in March
    own = bc[(bc["year"] == 2016) & (bc["quarter"] == 1)].set_index(C)
    own_T = own[SPLICED]
    spl, how = splice_back(own_T, mem_prev, mem_T)
    spl_ok = spl["log_total_assets"].notna()
    report["splice_2015q4"] = {
        "firms_with_a_2016q1_filing": int(len(own_T)), "firms_spliced": int(spl_ok.sum()),
        "values_by_rule": {v: {str(k): int(n) for k, n in how.loc[spl_ok, v].value_counts().sort_index().items()}
                           for v in SPLICED}}

    # --- which firm-quarters need a value ------------------------------------------------------
    need = fq["log_total_assets_lag"].isna() | (fq["total_assets_lag"] == 0) \
        | (is_q1 & fq[C].str.startswith("C"))
    fq["stored_zero_assets"] = (fq["total_assets_lag"] == 0).astype("int8")
    fq["chars_src"] = 0
    fq["chars_dist"] = 0
    fq["src_tq"] = -1
    rep = pd.DataFrame(np.nan, index=fq.index, columns=list(LEVEL_OF) + ["borrow_assets_lag"])

    # filings that can serve as a source: positive total assets, 2016 on (the prudential report of
    # 2015 holds co-operatives only; a 2015 row under a bank's code is not that bank)
    src_rows = bc[(bc["total_assets"] > 0) & (bc["year"] >= 2016)]
    by_firm = {c: g for c, g in src_rows.groupby(C)}
    level_cols = sorted(set(LEVEL_OF.values()))

    q1_spliced = is_q1 & need & fq[C].isin(spl.index[spl_ok])
    codes = fq.loc[q1_spliced, C]
    fq.loc[q1_spliced, "chars_src"] = 2
    fq.loc[q1_spliced, "src_tq"] = 2015 * 4 + 4
    for col, lvl in LEVEL_OF.items():
        if lvl in SPLICED:
            rep.loc[q1_spliced, col] = spl[lvl].reindex(codes).to_numpy()
    rep.loc[q1_spliced, "total_assets_lag"] = np.exp(spl["log_total_assets"].reindex(codes).to_numpy())
    rep.loc[q1_spliced, "indice_basileia_lag"] = own["indice_basileia"].reindex(codes).to_numpy()
    rep.loc[q1_spliced, "borrow_assets_lag"] = spl["borrow_assets"].reindex(codes).to_numpy()
    fq["basel_imputed"] = q1_spliced.astype("int8")

    # --- 2: true lag by calendar, else nearest filing ------------------------------------------
    todo = fq.index[need & ~q1_spliced]
    for i in todo:
        code, t = fq.at[i, C], int(fq.at[i, "tq"])
        g = by_firm.get(code)
        if g is None:
            fq.at[i, "chars_src"] = dr.CHARS_DROPPED
            continue
        d = (g["tq"] - (t - 1)).abs()
        dmin = int(d.min())
        if dmin > MAX_DISTANCE:
            fq.at[i, "chars_src"] = dr.CHARS_DROPPED
            continue
        row = g.loc[d[d == dmin].index].sort_values("tq").iloc[0]      # a tie goes to the earlier quarter
        fq.at[i, "chars_src"] = 1 if dmin == 0 else 3
        fq.at[i, "chars_dist"] = dmin
        fq.at[i, "src_tq"] = int(row["tq"])
        for col, lvl in LEVEL_OF.items():
            rep.at[i, col] = row[lvl]
        rep.at[i, "borrow_assets_lag"] = row["borrow_assets"]

    # --- the panel's own bounds on the winsorized columns ---------------------------------------
    cuts = stored_cutoffs(panel)
    before = rep[WINSORIZED].copy()
    rep[list(LEVEL_OF)] = clip_like_panel(rep[list(LEVEL_OF)], fq["is_B"], cuts)
    report["clipped_to_panel_bounds"] = {c: int(((rep[c] != before[c]) & before[c].notna()).sum()) for c in WINSORIZED}
    report["panel_bounds"] = {f"{c}|is_B={t}": list(v) for (c, t), v in cuts.items()}

    # --- 4: the borrowings ratio at one date, lagged --------------------------------------------
    # firm-quarters whose characteristics are the stored ones take the ratio of the same filing the
    # stored lags come from: the firm's previous row in bank_chars_panel.csv
    prev = bc[[C, "tq", "borrow_assets"]].copy()
    prev["borrow_prev_row"] = prev.groupby(C)["borrow_assets"].shift(1)
    stored_src = fq["chars_src"] == 0
    look = fq.loc[stored_src, [C, "tq"]].merge(prev[[C, "tq", "borrow_prev_row"]], on=[C, "tq"], how="left")
    rep.loc[stored_src, "borrow_assets_lag"] = look["borrow_prev_row"].to_numpy()
    old = fq["credit_assets_lag"].to_numpy(float)
    new = rep["borrow_assets_lag"].to_numpy(float)
    fin_old = np.isfinite(old)
    changed = fin_old & ~(np.isfinite(new) & (np.abs(new - old) <= 1e-12))
    fq["borrow_changed"] = changed.astype("int8")
    fq["borrow_den_guarded"] = (fq["total_assets_lag"] == 0).astype("int8")
    dif = (new - old)[fin_old & np.isfinite(new)]
    report["borrow_ratio"] = {
        "firm_quarters": int(len(fq)), "stored_finite": int(fin_old.sum()),
        "stored_infinite": int(np.isinf(old).sum()), "stored_missing": int(np.isnan(old).sum()),
        "new_missing": int(np.isnan(new).sum()), "new_max": float(np.nanmax(new)),
        "stored_finite_and_changed": int(changed.sum()),
        "stored_finite_new_missing": int((fin_old & ~np.isfinite(new)).sum()),
        "abs_change_quantiles": {q: float(np.quantile(np.abs(dif), float(q))) for q in ("0.5", "0.9", "0.99", "1.0")},
        "abs_change_above_0.01": int((np.abs(dif) > 0.01).sum()), "abs_change_above_0.10": int((np.abs(dif) > 0.10).sum())}

    # --- 3: segments ---------------------------------------------------------------------------
    seg = bc.dropna(subset=["segment"])[[C, "tq", "segment"]].sort_values([C, "tq"])
    first = seg.drop_duplicates(C, keep="first").set_index(C)
    seg_by_firm = {c: (g["tq"].to_numpy(), g["segment"].to_numpy()) for c, g in seg.groupby(C)}
    fq["segment_src"] = 0
    fq["r__segment"] = fq["segment"]
    for i in fq.index[fq["segment"].isna()]:
        code, t = fq.at[i, C], int(fq.at[i, "tq"])
        if code not in seg_by_firm:
            fq.at[i, "r__segment"], fq.at[i, "segment_src"] = "S1", 3
            continue
        tqs, segs = seg_by_firm[code]
        if t < tqs[0]:
            fq.at[i, "r__segment"], fq.at[i, "segment_src"] = segs[0], 1
        else:
            k = int(np.searchsorted(tqs, t, side="right")) - 1          # last recorded quarter <= t
            fq.at[i, "r__segment"], fq.at[i, "segment_src"] = segs[max(k, 0)], 2
    for s in ("S2", "S3", "S4", "S5"):
        dummy = (fq["r__segment"] == s).astype(float)
        fq[f"r__seg_{s}"] = np.where(fq["segment_src"] == 0, fq[f"seg_{s}"], dummy)
    report["segments"] = {"firm_quarters_by_src": {str(k): int(v) for k, v in fq["segment_src"].value_counts().sort_index().items()},
                          "firms_never_classified": int(fq.loc[fq["segment_src"] == 3, C].nunique()),
                          "firms_with_a_recorded_segment": int(len(first))}
    for col in rep.columns:
        fq["r__" + col if col != "borrow_assets_lag" else col] = rep[col]
    report["chars"] = {"firm_quarters": int(len(fq)),
                       "by_src": {str(k): int(v) for k, v in fq["chars_src"].value_counts().sort_index().items()},
                       "stored_zero_assets": int(fq["stored_zero_assets"].sum()),
                       "nearest_by_distance": {str(k): int(v) for k, v in fq.loc[fq["chars_src"] == 3, "chars_dist"].value_counts().sort_index().items()}}
    return fq, (spl, how, own_T, mem_prev, mem_T)


def backtest_splice(bc: pd.DataFrame, memb: pd.Series) -> pd.DataFrame:
    """The splice where the truth is known: the prudential values of 2016Q1 predicted from 2016Q2
    (same function, one quarter backwards), against the 2016Q1 filing and against the value of 2016Q2
    carried back unchanged."""
    m3, m6 = member_sums(2016, 3, memb), member_sums(2016, 6, memb)
    mem_prev, mem_T = member_chars(m3), member_chars(quarter_flows(m3, m6))
    own_T = bc[(bc["year"] == 2016) & (bc["quarter"] == 2)].set_index(C)[SPLICED]
    truth = bc[(bc["year"] == 2016) & (bc["quarter"] == 1)].set_index(C)[SPLICED]
    pred, _ = splice_back(own_T, mem_prev, mem_T)
    idx = pred.index[pred["log_total_assets"].notna()].intersection(truth.index[truth["log_total_assets"].notna()])
    rows = []
    for v in SPLICED:
        e_s = (pred.loc[idx, v] - truth.loc[idx, v]).abs().dropna()
        e_c = (own_T.loc[idx, v] - truth.loc[idx, v]).abs().dropna()
        rows.append({"variable": v, "n": int(len(e_s)), "splice_median_abs_err": float(e_s.median()),
                     "splice_p90_abs_err": float(e_s.quantile(0.9)), "splice_max_abs_err": float(e_s.max()),
                     "carry_median_abs_err": float(e_c.median()), "carry_p90_abs_err": float(e_c.quantile(0.9))})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 1d rival branches
def rival_branches(yyyymm: str) -> pd.DataFrame:
    """Rival branch count per (conglomerate, market) at one month, as panel_estban_instrument.py
    builds it from ESTBAN.csv: branches summed to the conglomerate (time-varying List crosswalk)
    and the market (latest crosswalk row of the municipality), rivals = market total minus own."""
    import panel_cosif_calibrate as cc
    est = read_raw_estban(yyyymm, {"AGEN_PROCESSADAS": "AGEN_PROCESSADAS"})
    est["AGEN_PROCESSADAS"] = est["AGEN_PROCESSADAS"].fillna(0.0)
    est["AnoMes"] = int(yyyymm)
    uniq = est[["CNPJ", "AnoMes"]].drop_duplicates()
    cmap = cc.map_foundation_to_congl(uniq, cc.build_cnpj_to_congl())[["CNPJ", "AnoMes", C]]
    est = est.merge(cmap, on=["CNPJ", "AnoMes"], how="inner")
    xw = pd.read_csv(MCA_XWALK, usecols=["municipality_code", "year", "mca_code"])
    xw = xw.sort_values("year").drop_duplicates("municipality_code", keep="last")
    xw["mca_code"] = xw["mca_code"].astype(str)
    est = est.merge(xw[["municipality_code", "mca_code"]], left_on="CODMUN_IBGE",
                    right_on="municipality_code", how="inner")
    own = est.groupby([C, "mca_code"], as_index=False)["AGEN_PROCESSADAS"].sum().rename(
        columns={"AGEN_PROCESSADAS": "own_br"})
    own["rival_br"] = (own.groupby("mca_code")["own_br"].transform("sum") - own["own_br"]).clip(lower=0.0)
    own[C] = own[C].astype(str)
    return own


# ----------------------------------------------------------------------------- leave-one-out pairs
def loo_pairs(frame: pd.DataFrame, chars: dict) -> pd.DataFrame:
    """panel_loo_instruments.compute_loo_instruments on `frame` (unique on its key): for each
    characteristic the sum over the OTHER firms of the market-quarter, a missing value counted as 0,
    and that sum over the number of rivals (0 where there is none)."""
    grp = [frame["mca_code"], frame["year"], frame["quarter"]]
    n_riv = (frame.groupby(["mca_code", "year", "quarter"])[C].transform("size") - 1).clip(lower=0)
    out = pd.DataFrame(index=frame.index)
    for name, col in chars.items():
        v0 = frame[col].fillna(0)
        loo = v0.groupby(grp).transform("sum") - v0
        out[f"loo_{name}"] = loo
        out[f"mean_loo_{name}"] = np.where(n_riv > 0, loo / n_riv, 0)
    return out


def same_values(a, b, tol=1e-9) -> np.ndarray:
    a, b = np.asarray(a, float), np.asarray(b, float)
    both_nan = np.isnan(a) & np.isnan(b)
    both_inf = np.isinf(a) & np.isinf(b) & (np.sign(a) == np.sign(b))
    with np.errstate(invalid="ignore"):
        close = np.abs(a - b) <= tol * np.maximum(1.0, np.abs(b))
    return both_nan | both_inf | (np.isfinite(a) & np.isfinite(b) & close)


# ----------------------------------------------------------------------------- main
def main() -> None:
    report = {"inputs": {p.name: md5(p) for p in (PANEL_PARQUET, BANK_CHARS_CSV, DEPOSITS_CSV)},
              "max_distance_quarters": MAX_DISTANCE}
    panel = load_panel()
    bc = load_bank_chars()
    memb = member_map()
    log.info(f"panel rows {len(panel):,}; bank-characteristics firm-quarters {len(bc):,}; "
             f"member institutions mapped {len(memb):,}")

    # ---- 1a: balances of 2013-2015
    scale = pre2016_scale(report)
    check_panel_against_deposits(panel, report)
    pre = panel[panel["year"] < 2016].copy()
    if not (pre["Source"] == "ESTBAN").all():
        raise AssertionError("a pre-2016 panel row does not come from ESTBAN: its balance has no measured factor")
    f = np.array([scale[(int(y), int(q))] for y, q in zip(pre["year"], pre["quarter"])])
    pre_out = pre[KEYS].copy()
    for c in dr.DEP_COLS:
        pre_out["r__" + c] = pre[c].to_numpy() / f
    pre_out["row_kind"] = "pre2016"
    report["pre2016_rows"] = int(len(pre_out))

    # ---- firm-quarter repairs (1c, 2, 3, 4)
    fq, (spl, how, own_T, mem_prev, mem_T) = build_firm_quarters(panel, bc, memb, report)
    bt = backtest_splice(bc, memb)
    report["splice_backtest_2016q2_to_2016q1"] = bt.to_dict("records")

    win = panel[panel["year"].between(MIN_YEAR, MAX_YEAR)].copy().reset_index(drop=True)
    fcols = ["chars_src", "chars_dist", "basel_imputed", "segment_src", "borrow_changed", "borrow_den_guarded",
             "r__segment", "borrow_assets_lag"] + ["r__" + c for c in LEVEL_OF] + ["r__" + c for c in dr.SEG_COLS]
    win = win.merge(fq[[C, "year", "quarter"] + fcols], on=[C, "year", "quarter"], how="left", validate="many_to_one")
    is_q1 = ((win["year"] == 2016) & (win["quarter"] == 1)).to_numpy()
    win["lag_from_2015q4"] = is_q1.astype("int8")

    # ---- 1d: rival branches of 2015Q4, checked on 2016Q1 -> 2016Q2 against the stored column
    rb15, rb16 = rival_branches("201512"), rival_branches("201603")
    chk = win[(win["year"] == 2016) & (win["quarter"] == 2)][[C, "mca_code", "estban_rival_branches_lag"]].merge(
        rb16[[C, "mca_code", "rival_br"]], on=[C, "mca_code"], how="left")
    mine = np.log1p(chk["rival_br"].fillna(0.0))
    eq = np.isclose(mine, chk["estban_rival_branches_lag"], rtol=0, atol=1e-12)
    report["estban_replication_2016q2"] = {"rows": int(len(chk)), "equal": int(eq.sum()),
                                           "rows_with_a_2016q1_pair": int(chk["rival_br"].notna().sum())}
    q1keys = win.loc[is_q1, [C, "mca_code"]].merge(rb15[[C, "mca_code", "rival_br"]], on=[C, "mca_code"], how="left")
    win["estban_lag_src"] = 0
    win["r__estban_rival_branches_lag"] = np.nan
    found = q1keys["rival_br"].notna().to_numpy()
    idx_q1 = np.flatnonzero(is_q1)
    win.loc[idx_q1[found], "estban_lag_src"] = 1
    win.loc[idx_q1[found], "r__estban_rival_branches_lag"] = np.log1p(q1keys.loc[found, "rival_br"].to_numpy())
    report["estban_2016q1"] = {"rows": int(is_q1.sum()), "rows_with_a_2015q4_pair": int(found.sum()),
                               "of_which_positive": int((q1keys.loc[found, "rival_br"] > 0).sum())}

    # ---- leave-one-out pairs: stored characteristics reproduce the stored columns
    base = loo_pairs(win, LOO_CHARS)
    rep_loo = {c: int((~same_values(base[c], win[c])).sum()) for c in dr.LOO_COLS}
    report["loo_replication_rows_differing"] = rep_loo
    if any(rep_loo.values()):
        raise AssertionError(f"the leave-one-out rule here does not reproduce the stored columns: {rep_loo}")

    def with_chars(src_set):
        fr = win[KEYS].copy()
        m = win["chars_src"].isin(src_set).to_numpy()
        for name, col in LOO_CHARS.items():
            if col == "credit_assets_lag":
                tal = np.where(m, win["r__total_assets_lag"], win["total_assets_lag"])
                with np.errstate(divide="ignore", invalid="ignore"):
                    fr[col] = np.where(m, win["emprestimos_repasses"].to_numpy(float) / tal, win[col].to_numpy(float))
            else:
                fr[col] = np.where(m, win["r__" + col], win[col])
        return fr, m

    for tag, src_set in (("s1", {2}), ("s2", {1, 2, 3})):
        fr, m = with_chars(src_set)
        new = loo_pairs(fr, LOO_CHARS)
        touched = pd.Series(m).groupby([win["mca_code"], win["year"], win["quarter"]]).transform("any").to_numpy()
        for c in dr.LOO_COLS:
            win[f"{tag}__{c}"] = np.where(touched, new[c], win[c])
        report[f"loo_{tag}"] = {"rows_in_a_touched_market_quarter": int(touched.sum()),
                                "rows_changed": {c: int((~same_values(win[f"{tag}__{c}"], win[c])).sum()) for c in dr.LOO_COLS},
                                "infinite_left": {c: int(np.isinf(win[f"{tag}__{c}"].to_numpy(float)).sum()) for c in dr.LOO_COLS}}
    bor = loo_pairs(win[KEYS + ["borrow_assets_lag"]], {"borrow_assets": "borrow_assets_lag"})
    win["loo_borrow_assets"], win["mean_loo_borrow_assets"] = bor["loo_borrow_assets"], bor["mean_loo_borrow_assets"]
    st_fin = np.isfinite(win["loo_credit_assets"].to_numpy(float))
    d = (win["loo_borrow_assets"] - win["loo_credit_assets"]).to_numpy(float)[st_fin]
    report["loo_borrow"] = {"rows": int(len(win)), "stored_finite": int(st_fin.sum()),
                            "stored_infinite_or_missing": int((~st_fin).sum()),
                            "stored_finite_and_changed": int((np.abs(d) > 1e-9).sum()),
                            "abs_change_quantiles": {q: float(np.quantile(np.abs(d), float(q))) for q in ("0.5", "0.9", "0.99", "1.0")},
                            "infinite_or_missing_now": int((~np.isfinite(win["loo_borrow_assets"].to_numpy(float))).sum())}

    # ---- 1b: 2015Q4 balances of the national firms that only IF.data reports
    m15, m16 = member_sums(2015, 12, memb), member_sums(2016, 3, memb)
    q4 = panel[(panel["year"] == 2015) & (panel["quarter"] == 4)]
    q1 = panel[(panel["year"] == 2016) & (panel["quarter"] == 1)]
    have_q4 = set(zip(q4[C], q4["mca_code"]))
    cand = q1[(q1["is_B"] == 0) & (q1["mca_code"] == "NATIONAL")].copy()
    cand = cand[[k not in have_q4 for k in zip(cand[C], cand["mca_code"])]]
    rf = q4["risk_free_qoq"].dropna().unique()
    s1, s2 = q4["spread_a1"].dropna().unique(), q4["spread_a2"].dropna().unique()
    a1, a2 = q4["spread_ann_a1"].dropna().unique(), q4["spread_ann_a2"].dropna().unique()
    for name, arr in (("risk_free_qoq", rf), ("spread_a1", s1), ("spread_a2", s2), ("spread_ann_a1", a1), ("spread_ann_a2", a2)):
        if len(arr) == 0 or float(arr.max() - arr.min()) > 1e-12:
            raise AssertionError(f"{name} is not one national value in 2015Q4")
    rf = float(rf[0])
    corr = pd.read_csv(CORRECTED_RATE_CSV, dtype={C: str})
    corr = corr[corr["AnoMes"] == 201512].set_index(C)["rate_a4_corrected_qoq"]
    added = []
    for r in cand.itertuples(index=False):
        code = getattr(r, C)
        vals, any_val = {}, False
        for c in dr.DEP_COLS:
            i15 = m15[c].get(code, np.nan) if code in m15.index else np.nan
            i16 = m16[c].get(code, np.nan) if code in m16.index else np.nan
            own_bal = getattr(r, c)
            if np.isfinite(i15) and np.isfinite(i16) and i16 > 0 and np.isfinite(own_bal):
                vals["r__" + c] = own_bal * i15 / i16
                any_val = True
            else:
                vals["r__" + c] = np.nan
        if not any_val:
            continue
        rate, rate_src = corr.get(code, np.nan), 1
        if not np.isfinite(rate) or rate > RATE_A4_MAX_MULT * rf:
            rate, rate_src = r.rate_a4, 2                      # the firm's own type-4 rate of 2016Q1
        if not np.isfinite(rate):
            rate, rate_src = rf, 3
        vals.update({C: code, "mca_code": r.mca_code, "year": 2015, "quarter": 4, "row_kind": "added_2015q4",
                     "r__risk_free_qoq": rf, "r__spread_a1": float(s1[0]), "r__spread_a2": float(s2[0]),
                     "r__spread_a4": rf - rate, "r__spread_ann_a1": float(a1[0]), "r__spread_ann_a2": float(a2[0]),
                     "r__spread_ann_a4": (1 + rf) ** 4 - (1 + rate) ** 4, "lag_rate_src": rate_src})
        added.append(vals)
    add_out = pd.DataFrame(added)
    report["national_2015q4_balances"] = {
        "national_firms_in_2016q1": int(((q1["is_B"] == 0) & (q1["mca_code"] == "NATIONAL")).sum()),
        "without_a_2015q4_row": int(len(cand)), "rows_added": int(len(add_out)),
        "with_dep_a1": int(add_out["r__dep_a1"].gt(0).sum()) if len(add_out) else 0,
        "with_dep_a2": int(add_out["r__dep_a2"].gt(0).sum()) if len(add_out) else 0,
        "with_dep_a4": int(add_out["r__dep_a4"].gt(0).sum()) if len(add_out) else 0,
        "type4_rate_source": {str(k): int(v) for k, v in add_out["lag_rate_src"].value_counts().sort_index().items()} if len(add_out) else {}}
    win["lag_dep_src"] = 0
    k1 = list(zip(win.loc[is_q1, C], win.loc[is_q1, "mca_code"]))
    added_keys = set(zip(add_out[C], add_out["mca_code"])) if len(add_out) else set()
    win.loc[idx_q1, "lag_dep_src"] = [1 if k in have_q4 else (2 if k in added_keys else 0) for k in k1]

    # ---- assemble and write
    keep = KEYS + dr.FLAG_COLS + ["r__" + c for c in dr.CHAR_COLS] + ["r__segment"] + ["r__" + c for c in dr.SEG_COLS] \
        + ["r__estban_rival_branches_lag"] + [f"{t}__{c}" for t in ("s1", "s2") for c in dr.LOO_COLS] + dr.BORROW_COLS
    win_out = win[keep].copy()
    win_out["row_kind"] = "window"
    out = pd.concat([win_out, pre_out, add_out], ignore_index=True, sort=False)
    for c in dr.FLAG_COLS + ["lag_rate_src"]:
        out[c] = out[c].astype("float32")                          # missing outside the rows a flag describes
    if out.duplicated(KEYS).any():
        raise AssertionError("the repair file is not unique on its key")
    out.to_parquet(OUT_PARQUET, index=False)
    report["rows"] = {"window": int(len(win_out)), "pre2016": int(len(pre_out)), "added_2015q4": int(len(add_out))}
    report["output_md5"] = md5(OUT_PARQUET)
    with open(OUT_REPORT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    log.info(f"wrote {OUT_PARQUET} ({len(out):,} rows) and {OUT_REPORT.name}")
    print(json.dumps({k: report[k] for k in ("rows", "chars", "segments", "estban_2016q1", "national_2015q4_balances")}, indent=2))


if __name__ == "__main__":
    sys.exit(main())

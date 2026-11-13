"""
diag_lai025440_rates.py
=======================
The panel's time-deposit (type-4) rate against the BCB's Document 5005 issuance rates, by prudential segment
and quarter, 2017Q3-2023Q4: the first check of the panel's type-4 rate that does not use COSIF
(claude/MAIN_IMPR_CHANNEL.md, log 11-13 11:02 MAIN, Task 2). Read-only; it builds no paper keys.

Inputs (read only)
  DATA_RAW/BCB/registry_025440/raw/Captações resgates taxas segmento.xlsx (raw_paths.BCB_REGISTRY_025440): the
      BCB's answer to LAI 18810.025440/2026-48, sheet Dados, one row per business day and segment S1-S5:
      CAP_<k> the day's issues (R$) and TX_<k> their issuance-weighted mean issuance rate, in percent A YEAR
      (sheet Notas, note 3), for k = PRE (fixed), POS (TR/TBF), CDI, OUTRAS (other floating). TX_<k> = 0
      exactly where CAP_<k> = 0. TX_CDI is a rate, not a percentage of the CDI, so it is not converted.
  processed/market_panel.parquet (pyarrow column subset): CodConglomeradoPrudencial, mca_code, year, quarter,
      is_B, dep_a4, rate_a4_national (the conglomerate's type-4 rate, a fraction PER QUARTER, the same in all
      its markets; panel_deposit_rates.py), rate_a4 (the local version, panel_local_cdb_spread.py),
      rate_a4_carried, rate_a4_src, risk_free_qoq, risk_free_ann, spread_ann_a4_national
  Rout/lambda_registry/conglomerate_segment_quarter.csv (make_lambda_registry.py): the segment map of registry
      025399, panel conglomerate x quarter -> S1-S5 from the IF.data register (Sr)
  processed/PANEL_INTERMED/quarterly_macro_rates.csv (panel_deposit_rates.py): cdi_qoq and selic_qoq, percent
      per quarter, compounded from the daily series

Definitions (segment s, quarter t)
  BCB      TX_k(s,t) = sum_d CAP_k(s,d) TX_k(s,d) / sum_d CAP_k(s,d) over the quarter's business days d,
           percent a year
  panel    r_q(s,t) = sum_j D_jt r_jt / sum_j D_jt over the conglomerates j of s in t with D_jt > 0 and r_jt
           observed; r_jt = rate_a4_national, D_jt = dep_a4 summed over j's markets; annualized by
           compounding, 100 ((1 + r_q)^4 - 1), percent a year
  CDI      100 ((1 + cdi_qoq / 100)^4 - 1), percent a year; every rate is also given as a ratio to it
  gap      panel minus BCB, percentage points a year
  S1-S5    a pooled row: the same means over every segment
Variants (robustness): panel 'measured' (rate_a4_carried = 0 only); panel 'local' (rate_a4 by market, weighted
  by the market's dep_a4); BCB 'daily' (each day's rate as a daily rate (1 + TX)^(1/252) - 1, issuance-weighted,
  then compounded over 252 days); 'premium-free' quarters only (the old-rule savings premium, savings_old_qoq -
  savings_rate_qoq of the macro file, is zero: 2022Q1-2023Q4 in the window), because a type-4 rate that nets
  savings out at the new-rule yield carries part of that premium where it is positive
Checks (the script stops if one fails): the workbook's md5 and header; no repeated (DATA, SEG); TX_k = 0 exactly
  where CAP_k = 0; rate_a4_national one value per conglomerate-quarter; spread_ann_a4_national = risk_free_ann -
  ((1 + rate_a4_national)^4 - 1), so the rate is per quarter; the panel's risk_free_qoq equals the macro file's
  selic_qoq / 100; every panel conglomerate-quarter of 2016-2024 is in the segment map with the same dep_a4.

Outputs (paths.rout_dir()/diag_lai025440/, LF)
  bcb_quarterly.csv          segment x quarter: issues and redemptions (R$ bn), issue days and TX by indexation,
                             the CDI and PRE rates in the daily-rate variant
  panel_quarterly.csv        segment x quarter: dep_a4 (R$ bn), coverage, the carried and D shares, the rate in
                             the three panel variants, the dep_a4 shares by rate_a4_src
  panel_conglomerate_quarterly.csv   conglomerate x quarter 2017Q3-2023Q4 (S1-S5): segment, dep_a4 and its share of
                             the segment, rate_a4_national, annualized and over the CDI, carried flag, rate_a4_src
  compare.csv                segment x quarter 2017Q3-2023Q4: CDI, Selic, panel and BCB rates, gaps and ratios
  gap_path_by_year.csv       segment x year: mean rates, gaps and ratios to the CDI and the Selic
  summary_by_segment.csv     segment: level gap, correlations over time (levels, ratios to the CDI, changes),
                             the variants, the premium-free quarters
  ordering_by_quarter.csv    quarter: the segments' ranks on each side, Spearman, concordant pairs
  summary.json               inputs with md5s, checks, the tables above, the cross-correlations by lag
  MD5SUMS.txt                LF md5 of each output

Usage:  python diagnostics/diag_lai025440_rates.py [--any-panel]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import hashlib
import itertools
import json
import time
from datetime import datetime

import numpy as np
import openpyxl
import pandas as pd
import pyarrow.parquet as pq

from utils import paths, raw_paths

XLSX = raw_paths.BCB_REGISTRY_025440 / "raw" / "Captações resgates taxas segmento.xlsx"
XLSX_MD5 = "ecb89db5ec9e340a29bf5e2442ddb79b"
PANEL = paths.PROCESSED / "market_panel.parquet"
PANEL_EXPECT = "fa38794d"                 # the panel make_lambda_registry.py built the segment map on
SEGMAP_DIR = paths.rout_dir() / "lambda_registry"
SEGMAP = SEGMAP_DIR / "conglomerate_segment_quarter.csv"
MACRO = paths.PROCESSED / "PANEL_INTERMED" / "quarterly_macro_rates.csv"
OUT = paths.rout_dir() / "diag_lai025440"

HEADER = ("DATA", "SEG", "CAP_PRE", "CAP_POS", "CAP_CDI", "CAP_OUTRAS", "RESG_PRE", "RESG_POS", "RESG_CDI",
          "RESG_OUTRAS", "TX_PRE", "TX_POS", "TX_CDI", "TX_OUTRAS")
KINDS = ("PRE", "POS", "CDI", "OUTRAS")
SEGS = ("S1", "S2", "S3", "S4", "S5")
POOLED = "S1-S5"
Q0, Q1 = (2017, 3), (2023, 4)             # the comparison window
YEARS_MAP = (2016, 2024)                  # the segment map's years
BIZ_DAYS = 252
PANEL_COLS = ["CodConglomeradoPrudencial", "NomeInstituicao", "mca_code", "year", "quarter", "is_B", "dep_a4",
              "rate_a4", "rate_a4_national", "rate_a4_carried", "rate_a4_src", "risk_free_qoq", "risk_free_ann",
              "spread_ann_a4_national"]
SRC_KINDS = ("statement", "benchmarked", "imputed", "leaf", "carried")
LAGS = range(0, 5)
T0 = time.time()


def stamp(msg):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:4.0f}s] {msg}", flush=True)


def md5_file(p, lf=False):
    b = Path(p).read_bytes()
    return hashlib.md5(b.replace(b"\r", b"") if lf else b).hexdigest()


def qi(y, q):
    return np.asarray(y, dtype=int) * 4 + np.asarray(q, dtype=int) - 1


def qlabel(i):
    return f"{int(i) // 4}Q{int(i) % 4 + 1}"


def ann(r_q):
    """A rate per quarter (fraction) to percent a year, compounded."""
    return 100.0 * ((1.0 + np.asarray(r_q, dtype=float)) ** 4 - 1.0)


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~np.isnan(a) & ~np.isnan(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


# ------------------------------------------------------------------------------------------- BCB side
def read_bcb():
    got = md5_file(XLSX)
    if got != XLSX_MD5:
        raise SystemExit(f"{XLSX.name}: md5 {got}, expected {XLSX_MD5}")
    wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
    rows = list(wb["Dados"].iter_rows(values_only=True))
    notas = [r[0] for r in wb["Notas"].iter_rows(values_only=True) if r and r[0] is not None]
    if tuple(rows[0]) != HEADER:
        raise SystemExit(f"{XLSX.name}: header {rows[0]}, expected {HEADER}")
    filled = [i for i, r in enumerate(rows) if any(v is not None for v in r)]
    if filled != list(range(len(filled))):
        raise SystemExit(f"{XLSX.name}: a blank row inside the data")
    d = pd.DataFrame(rows[1:len(filled)], columns=list(HEADER))
    d["DATA"] = pd.to_datetime(d["DATA"])
    num = list(HEADER[2:])
    d[num] = d[num].astype(float)
    if (~d["SEG"].isin(SEGS)).any() or d[num].isna().any().any() or (d[num] < 0).any().any():
        raise SystemExit(f"{XLSX.name}: unexpected segment, a missing value or a negative value")
    if d.duplicated(["DATA", "SEG"]).any():
        raise SystemExit(f"{XLSX.name}: repeated (DATA, SEG)")
    for k in KINDS:
        if ((d[f"CAP_{k}"] == 0) != (d[f"TX_{k}"] == 0)).any():
            raise SystemExit(f"{XLSX.name}: TX_{k} = 0 does not coincide with CAP_{k} = 0")
    d["year"], d["quarter"] = d["DATA"].dt.year, d["DATA"].dt.quarter
    info = dict(path=str(XLSX), md5=got, rows=int(len(d)), blank_rows_after=int(len(rows) - len(filled)),
                first=str(d["DATA"].min().date()), last=str(d["DATA"].max().date()),
                days_by_segment={s: int((d["SEG"] == s).sum()) for s in SEGS},
                first_day_by_segment={s: str(d.loc[d["SEG"] == s, "DATA"].min().date()) for s in SEGS},
                notas=notas)
    return d, info


def bcb_quarterly(d: pd.DataFrame) -> pd.DataFrame:
    out = []
    for seg, x in [(s, d[d["SEG"] == s]) for s in SEGS] + [(POOLED, d)]:
        for (y, q), g in x.groupby(["year", "quarter"]):
            row = dict(segment=seg, year=int(y), quarter=int(q), qlabel=qlabel(qi(y, q)),
                       days=int(g["DATA"].nunique()))
            for k in KINDS:
                w, tx = g[f"CAP_{k}"].to_numpy(), g[f"TX_{k}"].to_numpy()
                tot = w.sum()
                row[f"cap_{k.lower()}_bn"] = tot / 1e9
                row[f"resg_{k.lower()}_bn"] = g[f"RESG_{k}"].sum() / 1e9
                row[f"issue_days_{k.lower()}"] = int(g.loc[g[f"CAP_{k}"] > 0, "DATA"].nunique())
                row[f"tx_{k.lower()}"] = float((w * tx).sum() / tot) if tot > 0 else np.nan
                if k in ("CDI", "PRE"):
                    daily = (1.0 + tx / 100.0) ** (1.0 / BIZ_DAYS) - 1.0
                    row[f"tx_{k.lower()}_daily"] = (100.0 * ((1.0 + (w * daily).sum() / tot) ** BIZ_DAYS - 1.0)
                                                    if tot > 0 else np.nan)
            out.append(row)
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------- panel side
def read_macro():
    m = pd.read_csv(MACRO)
    m["year"], m["quarter"] = m["AnoMes"] // 100, m["AnoMes"] % 100 // 3
    m["cdi_ann"] = 100.0 * ((1.0 + m["cdi_qoq"] / 100.0) ** 4 - 1.0)
    m["selic_ann"] = 100.0 * ((1.0 + m["selic_qoq"] / 100.0) ** 4 - 1.0)
    # the old-rule savings premium (deposits up to 2012-05-03 keep 0.5 % a month + TR): where it is positive,
    # a type-4 rate that nets savings out at the new-rule yield carries part of it (utils/macro_rates.py)
    m["savings_premium_qoq"] = m["savings_old_qoq"] - m["savings_rate_qoq"]
    m["premium_free"] = (m["savings_premium_qoq"].abs() < 1e-9).astype(int)
    return m[["year", "quarter", "cdi_qoq", "selic_qoq", "cdi_ann", "selic_ann", "savings_premium_qoq",
              "premium_free"]]


def panel_cq(any_panel: bool, macro: pd.DataFrame):
    pmd5 = md5_file(PANEL)
    if not pmd5.startswith(PANEL_EXPECT) and not any_panel:
        raise SystemExit(f"{PANEL.name}: md5 {pmd5[:8]}, expected {PANEL_EXPECT} (--any-panel to override)")
    t = pq.read_table(PANEL, columns=PANEL_COLS).to_pandas()
    t = t[(t["year"] >= YEARS_MAP[0]) & (t["year"] <= YEARS_MAP[1])].copy()
    checks = {}

    # units: rate_a4_national is per quarter, annualized by compounding in the pipeline
    ok = t[["rate_a4_national", "risk_free_ann", "spread_ann_a4_national"]].notna().all(axis=1)
    err = (t.loc[ok, "risk_free_ann"] - ((1.0 + t.loc[ok, "rate_a4_national"]) ** 4 - 1.0)
           - t.loc[ok, "spread_ann_a4_national"]).abs().max()
    err_rf = (t["risk_free_ann"] - ((1.0 + t["risk_free_qoq"]) ** 4 - 1.0)).abs().max()
    if not (err < 1e-9 and err_rf < 1e-12):
        raise SystemExit(f"units: spread_ann_a4_national off by {err:.3g}, risk_free_ann by {err_rf:.3g}")
    checks["units_rate_a4_national_per_quarter_max_abs_err"] = float(err)
    checks["units_rows_checked"] = int(ok.sum())

    t["code"] = t["CodConglomeradoPrudencial"].astype(str)
    g = t.groupby(["code", "year", "quarter"])
    nun = g[["rate_a4_national", "rate_a4_carried", "rate_a4_src", "is_B", "risk_free_qoq"]].nunique(dropna=False)
    if (nun > 1).any().any():
        raise SystemExit(f"a conglomerate-quarter has two values of {list(nun.columns[(nun > 1).any()])}")
    t["w_loc"] = t["dep_a4"].where(t["rate_a4"].notna() & (t["dep_a4"] > 0), 0.0).fillna(0.0)
    t["wr_loc"] = t["w_loc"] * t["rate_a4"].fillna(0.0)
    cq = g.agg(name=("NomeInstituicao", "first"), dep_a4=("dep_a4", "sum"), r_nat=("rate_a4_national", "first"),
               carried=("rate_a4_carried", "first"), src=("rate_a4_src", "first"), is_B=("is_B", "max"),
               rf_q=("risk_free_qoq", "first"), w_loc=("w_loc", "sum"), wr_loc=("wr_loc", "sum"),
               rows=("mca_code", "size")).reset_index()
    cq["r_loc"] = np.where(cq["w_loc"] > 0, cq["wr_loc"] / cq["w_loc"].where(cq["w_loc"] > 0), np.nan)

    # the macro file is the vintage the panel was built with
    chk = cq[["year", "quarter", "rf_q"]].merge(macro, on=["year", "quarter"], how="left")
    d_rf = (chk["rf_q"] * 100.0 - chk["selic_qoq"]).abs()
    if d_rf.isna().any() or not d_rf.max() < 1e-10:
        raise SystemExit(f"{MACRO.name}: selic_qoq differs from the panel's risk_free_qoq (max {d_rf.max():.3g}, "
                         f"{int(d_rf.isna().sum())} conglomerate-quarters unmatched)")
    checks["macro_selic_vs_panel_max_abs_diff_pct"] = float(d_rf.max())

    # the segment map of registry 025399
    man = pd.read_csv(SEGMAP_DIR / "md5_manifest.csv").set_index("file")["md5"]
    smd5 = md5_file(SEGMAP)
    if man.get(SEGMAP.name) != smd5:
        raise SystemExit(f"{SEGMAP.name}: md5 {smd5} differs from its md5_manifest.csv")
    smap = pd.read_csv(SEGMAP, dtype={"code": str, "seg": str, "seg_src": str})
    m = cq.merge(smap[["code", "year", "quarter", "seg", "seg_src", "dep_a4"]].rename(columns={"dep_a4": "map_dep_a4"}),
                 on=["code", "year", "quarter"], how="outer", indicator=True)
    if (m["_merge"] != "both").any():
        raise SystemExit(f"segment map and panel differ: {m['_merge'].value_counts().to_dict()}")
    dd = (m["dep_a4"] - m["map_dep_a4"]).abs() / m["map_dep_a4"].abs().clip(lower=1.0)
    if not dd.max() < 1e-9:
        raise SystemExit(f"segment map dep_a4 differs from the panel's (max rel {dd.max():.3g})")
    checks["segmap_rows"] = int(len(m))
    checks["segmap_dep_a4_max_rel_diff"] = float(dd.max())
    m = m.drop(columns=["_merge", "map_dep_a4"])
    m["qi"] = qi(m["year"], m["quarter"])
    info = dict(panel=dict(path=str(PANEL), md5=pmd5, rows_2016_2024=int(len(t)),
                           conglomerate_quarters=int(len(cq))),
                segment_map=dict(path=str(SEGMAP), md5=smd5, generator="make_lambda_registry.py"),
                macro=dict(path=str(MACRO), md5=md5_file(MACRO)))
    return m, info, checks


def wavg(r, w):
    r, w = np.asarray(r, float), np.asarray(w, float)
    ok = ~np.isnan(r) & (w > 0)
    return (float(np.sum(r[ok] * w[ok]) / np.sum(w[ok])) if ok.any() else np.nan), float(np.sum(w[ok]))


def panel_quarterly(m: pd.DataFrame) -> pd.DataFrame:
    out = []
    groups = [(s, m[m["seg"] == s]) for s in SEGS] + [(POOLED, m[m["seg"].isin(SEGS)])]
    for seg, x in groups:
        for (y, q), g in x.groupby(["year", "quarter"]):
            pos = g[g["dep_a4"] > 0]
            dep = pos["dep_a4"].sum()
            r_all, w_all = wavg(pos["r_nat"], pos["dep_a4"])
            meas = pos[pos["carried"] == 0]
            r_meas, w_meas = wavg(meas["r_nat"], meas["dep_a4"])
            w_loc = pos["w_loc"].sum()
            r_loc = float(pos["wr_loc"].sum() / w_loc) if w_loc > 0 else np.nan
            rated = pos[pos["r_nat"].notna()]
            row = dict(segment=seg, year=int(y), quarter=int(q), qlabel=qlabel(qi(y, q)),
                       n_conglomerates=int(len(pos)), n_rated=int(len(rated)), dep_a4_bn=dep / 1e9,
                       share_dep_rated=w_all / dep if dep > 0 else np.nan,
                       share_rated_carried=(rated.loc[rated["carried"] == 1, "dep_a4"].sum() / w_all
                                            if w_all > 0 else np.nan),
                       share_dep_D=pos.loc[pos["is_B"] == 0, "dep_a4"].sum() / dep if dep > 0 else np.nan,
                       r_q=r_all, panel_ann=float(ann(r_all)),
                       panel_ann_measured=float(ann(r_meas)), share_dep_measured=w_meas / dep if dep > 0 else np.nan,
                       panel_ann_local=float(ann(r_loc)))
            for sk in SRC_KINDS:
                row[f"share_rated_src_{sk}"] = (rated.loc[rated["src"] == sk, "dep_a4"].sum() / w_all
                                                if w_all > 0 else np.nan)
            out.append(row)
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------- comparison
def compare(bq: pd.DataFrame, pqt: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    c = pqt.merge(bq, on=["segment", "year", "quarter", "qlabel"], how="inner").merge(macro, on=["year", "quarter"])
    c["qi"] = qi(c["year"], c["quarter"])
    c = c[(c["qi"] >= qi(*Q0)) & (c["qi"] <= qi(*Q1))].copy()
    for k in ("cdi", "pre"):
        c[f"gap_{k}_pp"] = c["panel_ann"] - c[f"tx_{k}"]
        c[f"bcb_{k}_ratio_cdi"] = c[f"tx_{k}"] / c["cdi_ann"]
    c["panel_ratio_cdi"] = c["panel_ann"] / c["cdi_ann"]
    c["panel_measured_ratio_cdi"] = c["panel_ann_measured"] / c["cdi_ann"]
    c["panel_local_ratio_cdi"] = c["panel_ann_local"] / c["cdi_ann"]
    c["bcb_cdi_ratio_selic"] = c["tx_cdi"] / c["selic_ann"]
    c["panel_ratio_selic"] = c["panel_ann"] / c["selic_ann"]
    cols = ["segment", "year", "quarter", "qlabel", "cdi_ann", "selic_ann", "panel_ann", "tx_cdi", "tx_pre",
            "gap_cdi_pp", "gap_pre_pp", "panel_ratio_cdi", "bcb_cdi_ratio_cdi", "bcb_pre_ratio_cdi",
            "panel_ratio_selic", "bcb_cdi_ratio_selic", "panel_ann_measured", "panel_measured_ratio_cdi",
            "panel_ann_local", "panel_local_ratio_cdi", "tx_cdi_daily", "tx_pre_daily", "cap_cdi_bn", "cap_pre_bn",
            "issue_days_cdi", "issue_days_pre", "days", "dep_a4_bn", "n_conglomerates", "n_rated", "share_dep_rated",
            "share_rated_carried", "share_dep_D", "savings_premium_qoq", "premium_free", "qi"]
    return c[cols].sort_values(["segment", "qi"]).reset_index(drop=True)


def conglomerate_table(m: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """The panel side by conglomerate, in the window: what each segment mean is made of."""
    x = m[m["seg"].isin(SEGS) & (m["qi"] >= qi(*Q0)) & (m["qi"] <= qi(*Q1))].merge(
        macro[["year", "quarter", "cdi_ann"]], on=["year", "quarter"])
    tot = x[x["dep_a4"] > 0].groupby(["seg", "qi"])["dep_a4"].sum().rename("seg_dep")
    x = x.join(tot, on=["seg", "qi"])
    x["share_of_segment_dep_a4"] = np.where(x["dep_a4"] > 0, x["dep_a4"] / x["seg_dep"], 0.0)
    x["panel_ann"] = ann(x["r_nat"])
    x["panel_ratio_cdi"] = x["panel_ann"] / x["cdi_ann"]
    x["qlabel"] = [qlabel(i) for i in x["qi"]]
    x["dep_a4_bn"] = x["dep_a4"] / 1e9
    cols = ["seg", "qlabel", "year", "quarter", "code", "name", "is_B", "seg_src", "dep_a4_bn",
            "share_of_segment_dep_a4", "r_nat", "panel_ann", "cdi_ann", "panel_ratio_cdi", "carried", "src"]
    return (x[cols].rename(columns={"seg": "segment", "r_nat": "rate_a4_national_qoq", "carried": "rate_a4_carried",
                                    "src": "rate_a4_src"})
            .sort_values(["segment", "qlabel", "share_of_segment_dep_a4"], ascending=[True, True, False])
            .reset_index(drop=True))


def summary_by_segment(c: pd.DataFrame):
    rows, lags = [], {}
    window = range(int(qi(*Q0)), int(qi(*Q1)) + 1)
    for seg, g in c.groupby("segment", sort=False):
        # consecutive quarters (a quarter missing on either side stays as a gap), so changes and lags line up
        g = g.set_index("qi").reindex(window)
        g["qlabel"] = [qlabel(i) for i in g.index]
        both = g.dropna(subset=["panel_ann", "tx_cdi"])
        pf = g[g["premium_free"] == 1]
        imax, imin = both["gap_cdi_pp"].idxmax(), both["gap_cdi_pp"].idxmin()
        row = dict(segment=seg, quarters=int(len(both)), first=both["qlabel"].iloc[0], last=both["qlabel"].iloc[-1],
                   mean_cdi=both["cdi_ann"].mean(), mean_panel=both["panel_ann"].mean(),
                   mean_bcb_cdi=both["tx_cdi"].mean(), mean_bcb_pre=g["tx_pre"].mean(),
                   mean_gap_cdi_pp=both["gap_cdi_pp"].mean(), sd_gap_cdi_pp=both["gap_cdi_pp"].std(),
                   min_gap_cdi_pp=both.loc[imin, "gap_cdi_pp"], min_gap_q=both.loc[imin, "qlabel"],
                   max_gap_cdi_pp=both.loc[imax, "gap_cdi_pp"], max_gap_q=both.loc[imax, "qlabel"],
                   mean_gap_pre_pp=g["gap_pre_pp"].mean(),
                   mean_panel_ratio_cdi=both["panel_ratio_cdi"].mean(),
                   mean_bcb_cdi_ratio_cdi=both["bcb_cdi_ratio_cdi"].mean(),
                   mean_bcb_pre_ratio_cdi=g["bcb_pre_ratio_cdi"].mean(),
                   corr_level_cdi=corr(g["panel_ann"], g["tx_cdi"]),
                   corr_level_pre=corr(g["panel_ann"], g["tx_pre"]),
                   corr_ratio_cdi=corr(g["panel_ratio_cdi"], g["bcb_cdi_ratio_cdi"]),
                   corr_ratio_pre=corr(g["panel_ratio_cdi"], g["bcb_pre_ratio_cdi"]),
                   corr_dlevel_cdi=corr(g["panel_ann"].diff(), g["tx_cdi"].diff()),
                   corr_dratio_cdi=corr(g["panel_ratio_cdi"].diff(), g["bcb_cdi_ratio_cdi"].diff()),
                   corr_ratio_cdi_measured=corr(g["panel_measured_ratio_cdi"], g["bcb_cdi_ratio_cdi"]),
                   corr_ratio_cdi_local=corr(g["panel_local_ratio_cdi"], g["bcb_cdi_ratio_cdi"]),
                   corr_level_cdi_bcb_daily=corr(g["panel_ann"], g["tx_cdi_daily"]),
                   quarters_premium_free=int(pf["panel_ann"].notna().sum()),
                   corr_ratio_cdi_premium_free=corr(pf["panel_ratio_cdi"], pf["bcb_cdi_ratio_cdi"]),
                   corr_level_cdi_premium_free=corr(pf["panel_ann"], pf["tx_cdi"]),
                   mean_gap_cdi_pp_premium_free=pf["gap_cdi_pp"].mean(),
                   mean_gap_cdi_pp_premium=g.loc[g["premium_free"] == 0, "gap_cdi_pp"].mean(),
                   mean_gap_cdi_pp_measured=(g["panel_ann_measured"] - g["tx_cdi"]).mean(),
                   mean_gap_cdi_pp_local=(g["panel_ann_local"] - g["tx_cdi"]).mean(),
                   max_abs_local_minus_national_pp=(g["panel_ann_local"] - g["panel_ann"]).abs().max(),
                   max_abs_bcb_cdi_daily_minus_mean_pp=(g["tx_cdi_daily"] - g["tx_cdi"]).abs().max(),
                   max_abs_bcb_pre_daily_minus_mean_pp=(g["tx_pre_daily"] - g["tx_pre"]).abs().max(),
                   mean_share_dep_rated=g["share_dep_rated"].mean(),
                   mean_share_rated_carried=g["share_rated_carried"].mean(),
                   mean_share_dep_D=g["share_dep_D"].mean(), mean_cap_cdi_bn=g["cap_cdi_bn"].mean(),
                   mean_cap_pre_bn=g["cap_pre_bn"].mean(), mean_dep_a4_bn=g["dep_a4_bn"].mean())
        rows.append(row)
        # the stock reprices with a lag: correlation of the panel's ratio in t with the BCB's in t - L
        pr, br = g["panel_ratio_cdi"].to_numpy(), g["bcb_cdi_ratio_cdi"].to_numpy()
        lags[seg] = {str(L): corr(pr[L:], br[:len(br) - L] if L else br) for L in LAGS}
    return pd.DataFrame(rows), lags


def gap_path(c: pd.DataFrame) -> pd.DataFrame:
    agg = dict(quarters=("qi", "size"), cdi_ann=("cdi_ann", "mean"), selic_ann=("selic_ann", "mean"),
               panel_ann=("panel_ann", "mean"), tx_cdi=("tx_cdi", "mean"), tx_pre=("tx_pre", "mean"),
               gap_cdi_pp=("gap_cdi_pp", "mean"), gap_pre_pp=("gap_pre_pp", "mean"),
               panel_ratio_cdi=("panel_ratio_cdi", "mean"), bcb_cdi_ratio_cdi=("bcb_cdi_ratio_cdi", "mean"),
               bcb_pre_ratio_cdi=("bcb_pre_ratio_cdi", "mean"), panel_ratio_selic=("panel_ratio_selic", "mean"),
               bcb_cdi_ratio_selic=("bcb_cdi_ratio_selic", "mean"), share_rated_carried=("share_rated_carried", "mean"))
    return c.groupby(["segment", "year"], sort=False).agg(**agg).reset_index()


def ordering(c: pd.DataFrame):
    rows = []
    for i, g in c[c["segment"].isin(SEGS)].groupby("qi"):
        g = g.set_index("segment")
        row = dict(qlabel=qlabel(i))
        for name, segs in (("S1-S4", SEGS[:4]), ("S1-S5", SEGS)):
            h = g.reindex(list(segs)).dropna(subset=["panel_ann", "tx_cdi"])
            for k in ("cdi", "pre"):
                hh = h.dropna(subset=[f"tx_{k}"])
                rp, rb = hh["panel_ann"].rank(), hh[f"tx_{k}"].rank()
                pairs = list(itertools.combinations(hh.index, 2))
                conc = [np.sign(hh.at[a, "panel_ann"] - hh.at[b, "panel_ann"])
                        == np.sign(hh.at[a, f"tx_{k}"] - hh.at[b, f"tx_{k}"]) for a, b in pairs]
                row[f"{name}_{k}_n"] = int(len(hh))
                row[f"{name}_{k}_spearman"] = corr(rp, rb)
                row[f"{name}_{k}_same_order"] = int(len(hh) > 1 and (rp == rb).all())
                row[f"{name}_{k}_concordant_pairs"] = float(np.mean(conc)) if conc else np.nan
            if name == "S1-S4":
                row["panel_order_low_to_high"] = " < ".join(h["panel_ann"].sort_values().index)
                row["bcb_cdi_order_low_to_high"] = " < ".join(h["tx_cdi"].sort_values().index)
                row["bcb_pre_order_low_to_high"] = " < ".join(h["tx_pre"].dropna().sort_values().index)
        rows.append(row)
    o = pd.DataFrame(rows)
    means = c[c["segment"].isin(SEGS)].groupby("segment")[["panel_ann", "tx_cdi", "tx_pre", "panel_ratio_cdi",
                                                            "bcb_cdi_ratio_cdi"]].mean()
    s14 = means.loc[list(SEGS[:4])]
    tot = dict(quarters=int(len(o)),
               S1_S4_cdi_same_order_quarters=int(o["S1-S4_cdi_same_order"].sum()),
               S1_S4_cdi_mean_spearman=float(o["S1-S4_cdi_spearman"].mean()),
               S1_S4_cdi_concordant_pairs=float(o["S1-S4_cdi_concordant_pairs"].mean()),
               S1_S4_pre_same_order_quarters=int(o["S1-S4_pre_same_order"].sum()),
               S1_S4_pre_mean_spearman=float(o["S1-S4_pre_spearman"].mean()),
               S1_S4_pre_concordant_pairs=float(o["S1-S4_pre_concordant_pairs"].mean()),
               S1_S5_cdi_mean_spearman=float(o["S1-S5_cdi_spearman"].mean()),
               S1_S5_cdi_concordant_pairs=float(o["S1-S5_cdi_concordant_pairs"].mean()),
               window_means_order_panel=" < ".join(s14["panel_ann"].sort_values().index),
               window_means_order_bcb_cdi=" < ".join(s14["tx_cdi"].sort_values().index),
               window_means_order_bcb_pre=" < ".join(s14["tx_pre"].sort_values().index),
               window_means_spearman_cdi=corr(s14["panel_ann"].rank(), s14["tx_cdi"].rank()),
               window_means_spearman_pre=corr(s14["panel_ann"].rank(), s14["tx_pre"].rank()),
               most_common_panel_order=o["panel_order_low_to_high"].value_counts().head(3).to_dict(),
               most_common_bcb_cdi_order=o["bcb_cdi_order_low_to_high"].value_counts().head(3).to_dict())
    # each pair of segments: in how many quarters the two sides order it the same way
    for k in ("cdi", "pre"):
        pc = {}
        for a, b in itertools.combinations(SEGS, 2):
            x = c[c["segment"] == a].set_index("qi")
            y = c[c["segment"] == b].set_index("qi")
            j = x.index.intersection(y.index)
            sp = np.sign(x.loc[j, "panel_ann"] - y.loc[j, "panel_ann"])
            sb = np.sign(x.loc[j, f"tx_{k}"] - y.loc[j, f"tx_{k}"])
            ok = sp.notna() & sb.notna()
            pc[f"{a}-{b}"] = dict(quarters=int(ok.sum()), same_order=int((sp[ok] == sb[ok]).sum()),
                                  panel_first_lower=int((sp[ok] < 0).sum()), bcb_first_lower=int((sb[ok] < 0).sum()))
        tot[f"pair_concordance_{k}"] = pc
    return o, tot, means


# ------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--any-panel", action="store_true",
                    help=f"run on a panel other than {PANEL_EXPECT} (the outputs record its md5)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    written = []

    def write_csv(df, name):
        p = OUT / name
        df.to_csv(p, index=False, encoding="utf-8", lineterminator="\n", float_format="%.10g")
        written.append(p)

    d, bcb_info = read_bcb()
    stamp(f"workbook: {bcb_info['rows']} rows, {bcb_info['first']}..{bcb_info['last']}")
    bq = bcb_quarterly(d)
    macro = read_macro()
    m, pinfo, checks = panel_cq(a.any_panel, macro)
    stamp(f"panel {pinfo['panel']['md5'][:8]}: {pinfo['panel']['conglomerate_quarters']} conglomerate-quarters; "
          f"segment map {pinfo['segment_map']['md5'][:8]}")
    pqt = panel_quarterly(m)
    c = compare(bq, pqt, macro)
    summ, lags = summary_by_segment(c)
    path = gap_path(c)
    o, otot, means = ordering(c)

    write_csv(bq, "bcb_quarterly.csv")
    write_csv(pqt, "panel_quarterly.csv")
    write_csv(conglomerate_table(m, macro), "panel_conglomerate_quarterly.csv")
    write_csv(c.drop(columns=["qi"]), "compare.csv")
    write_csv(path, "gap_path_by_year.csv")
    write_csv(summ, "summary_by_segment.csv")
    write_csv(o, "ordering_by_quarter.csv")

    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (float, np.floating)):
            return float(x) if np.isfinite(x) else None
        return x
    res = dict(generated=datetime.now().isoformat(timespec="seconds"), generator="diagnostics/diag_lai025440_rates.py",
               window=f"{qlabel(qi(*Q0))}-{qlabel(qi(*Q1))}",
               units=dict(bcb="TX_<k>: percent a year as delivered (workbook note 3); quarterly = issuance-weighted "
                              "mean over the quarter's business days; not converted",
                          panel="rate_a4_national: fraction per quarter; segment mean weighted by dep_a4 over "
                                "conglomerates, then 100((1+r)^4-1)",
                          cdi="100((1+cdi_qoq/100)^4-1) from PANEL_INTERMED/quarterly_macro_rates.csv"),
               inputs=dict(workbook=bcb_info, **pinfo), checks=checks,
               summary_by_segment=summ.to_dict("records"), ratio_cross_corr_by_lag=lags,
               ordering=otot, window_means=means.reset_index().to_dict("records"),
               gap_path_by_year=path.to_dict("records"))
    p = OUT / "summary.json"
    p.write_text(json.dumps(clean(res), indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    written.append(p)
    with open(OUT / "MD5SUMS.txt", "w", encoding="utf-8", newline="\n") as fh:
        for q in written:
            fh.write(f"{md5_file(q, lf=True)}  {q.name}\n")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    show = ["segment", "quarters", "mean_cdi", "mean_panel", "mean_bcb_cdi", "mean_bcb_pre", "mean_gap_cdi_pp",
            "sd_gap_cdi_pp", "mean_panel_ratio_cdi", "mean_bcb_cdi_ratio_cdi", "corr_level_cdi", "corr_ratio_cdi",
            "corr_dratio_cdi", "corr_ratio_cdi_measured", "corr_ratio_cdi_premium_free", "mean_gap_cdi_pp_premium_free",
            "corr_level_pre", "corr_ratio_pre"]
    print("\nBy segment, 2017Q3-2023Q4 (percent a year; ratios to the quarter's CDI):")
    print(summ[show].round(3).to_string(index=False))
    print("\nGap path, panel minus BCB CDI-indexed issuance rate (pp a year), by year:")
    print(path.pivot(index="year", columns="segment", values="gap_cdi_pp").round(2).to_string())
    print("\nRatio to the CDI by year: panel / BCB CDI-indexed:")
    pv = path.assign(x=lambda z: z["panel_ratio_cdi"].round(2).astype(str) + " / "
                     + z["bcb_cdi_ratio_cdi"].round(2).astype(str))
    print(pv.pivot(index="year", columns="segment", values="x").to_string())
    print("\nOrdering:", json.dumps(clean({k: v for k, v in otot.items() if not k.startswith("pair_")}),
                                    indent=1, ensure_ascii=False))
    for k in ("cdi", "pre"):
        print(f"Pairs, panel vs BCB {k.upper()}-indexed (quarters ordered alike / quarters):",
              "; ".join(f"{p} {v['same_order']}/{v['quarters']}" for p, v in otot[f"pair_concordance_{k}"].items()))
    print("\nRatio cross-correlation, panel t vs BCB t-L:", json.dumps(clean(lags)))
    stamp(f"wrote {len(written)} files and MD5SUMS.txt to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
diag_new_money.py
=================
Step 1 of the market size built from measured new money (MAIN_IMPR_CHANNEL.md, A1.6, A1.13, A3; Pedro's go
of 11-12). Read-only: no estimation, nothing in production. The D-market fix is part of the construction: every
national aggregate runs over the B markets and the D institutions once (the D rows' NATIONAL pseudo-market never
enters a sum over markets), and the national sleepy share phi_t is the population-weighted mean of phi_mt over
the B markets.

Symbols (the paper's): P the awake pool, C carried wealth, C^0 its part outside deposits, N new money.

Cells. A cell c is conglomerate x market x deposit type k in {1, 2, 4, 5}; a B cell's market is its MCA, a D
cell's is the NATIONAL row. Quarters 2016Q1-2024Q4 (2015Q4 for the lags).
  D_c       booked balance, market_panel.parquet dep_a{k} (dep_a4 net of judicial deposits, as production;
            dep_a4_incl_judicial for the judicial sensitivity). Missing = 0, so entries and exits are cells.
  C_c       R_c D_c,t-1 with R = gross_return_lag of (11-B): the frame's own value where the cell is in the
            demand frame, else the prep's formula 1 + risk_free_qoq_{t-1} - spread_a{k}_{t-1} from the panel,
            else the type-quarter deposit-weighted R of the frame.
  phi_c     B: phi_mt of the market-quarter (frame; one value per market-quarter), filled from the same market
            within four quarters where the market-quarter has no stored row, else phi_t; D: phi_t over the B
            markets. Uncapped in (11-B); the caps 0.99 and 0.999 enter only the pools.
  Dep_Act   max{0, D - phi C}                       (11-B)
  A         max{(1 - phi) C, C - D}                 money that did not stay (Pedro, 20:32)
  identity  Dep_Act = (D - C) + A, so on deposits alone sum Dep_Act = N + sum A           (gate G1)
National (quarter t):
  W_t = inside + outside stocks; C_t = sum_c C_c + sum_k C^0_k; N_t = W_t - C_t
  P^B_t = sum_c A_c + sum_k max{(1 - phi_t) C^0_k, C^0_k - D^0_k} + N_t      the main pool (Pedro, 20:32)
  P^A_t = (1 - phi_t) C_t + N_t                                             the 20:10 pool, for comparison
  outside share = 1 - sum Dep_Act / P (truncated Dep_Act of (11-B); and untruncated, sum (D - phi C))
  realized awake share = (P^B - N) / C against 1 - phi_t and the carry-weighted sum (1 - phi) C / sum C
Universes:
  dep      the panel's four deposit types alone (all holders): P^B = sum Dep_Act exactly (gate G1)
  disk     dep + what is on disk outside deposits: IF.Data letras (LCI 78289, LCA 78290, LF 78291, other
           78293; liabilities to all holders, carried at the CDI) and the Tesouro Direto stock (B3 RI,
           persons, carried at the Selic)
  agg      the BCB aggregates (16 SGS series, DATA_RAW/BCB/monetary_aggregates_sgs) once downloaded: the panel
           cells rescaled to the aggregates' stocks by type and quarter (f_k,t = aggregate / panel; type 5 has no
           aggregate and keeps f = 1), outside = currency (zero return), letras financeiras / de credito / other
           (CDI), money-market fund quotas (CDI; sensitivity CDI - 1 % a year), repos (Selic), federal
           securities (Selic; N_t also without them). Skipped, with the columns left empty, while the files are
           absent.
Local pools (B markets): P_mt = sum_{B cells of m} A_c + a_mt X_t, X_t = P_t - sum_{B cells} A_c (the D cells'
  awake money, the outside part and N_t), a_mt one of three allocation rules: L1 population x access, L2 the
  lagged local booked B stock, L3 their mean. With and without the symmetric netting of booking moves inside a
  bank (a bank's net losses in some markets matched by its net gains in others, types netted within a
  bank-market): the matched amount leaves both the losing market's awake money and the gaining market's
  active balances.

Steps: --step national,local (default both). Gates: G1 (identity), G2 (the rebuilt cells against the stored
frame's own columns), G3 (phi_t with the NATIONAL row reproduces the stored phi_t), W (W4.1's production pool
numbers reproduce from the K2 context, through diag_market_size_alt.W4).

Outputs: paths.rout_dir()/diag_new_money/ (CSV, JSON, MD5SUMS.txt with LF md5s).
Usage: python diagnostics/diag_new_money.py [--routine 3] [--step national,local] [--agg-dir <dir>]
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from utils import paths, raw_paths
from utils import macro_rates

T0 = time.time()
BANK = "CodConglomeradoPrudencial"
TYPES = (1, 2, 4, 5)
PANEL = paths.PROCESSED / "market_panel.parquet"
FRAME_DIR = Path(r"C:\Users\pedro\AppData\Local\Temp\msize_1112\base\demand_prep")   # production-equivalent copy
FRAMES = {3: "demand_3_index_spec_12.parquet", 4: "demand_4_index_time_spec_12.parquet"}
SPEC_COL = "phi_t_IV_RivalSpread_x_Tech"            # spec 12 in est{k}/national_phi_t.csv
AGG_DIR = getattr(raw_paths, "BCB_MONETARY_AGGREGATES_SGS", raw_paths.DATA_RAW / "BCB" / "monetary_aggregates_sgs")
B3_RI = raw_paths.DATA_RAW / "B3" / "ri_banco_de_dados" / "BD_Operational Figures_dl20261027.xlsx"
FGC = paths.RAW / "FGC_CENSO" / "censo_produto.parquet"
OUT = paths.rout_dir() / "diag_new_money"
SCRATCH = Path(r"C:\Users\pedro\AppData\Local\Temp\claude\e04f72edbe14fbd1\21a30b32-2059-49bd-aaac-15d47097282e\scratchpad")
CAPS = {"none": 1.0, "0.99": 0.99, "0.999": 0.999}
Q0, Q1 = 2016 * 4, 2024 * 4 + 3
IFD_ACC = {"78289": "LCI", "78290": "LCA", "78291": "LF", "78293": "other_bills"}
AGG = {   # code: (name, role, return)
    27789: ("currency", "outside", "zero"), 27790: ("demand", "inside1", None), 27791: ("M1", "check", None),
    1835: ("savings", "inside2", None), 27805: ("time", "inside4", None), 27806: ("LF", "outside", "cdi"),
    27807: ("LC", "outside", "cdi"), 27808: ("other_priv", "outside", "cdi"), 27809: ("priv_total", "check", None),
    27810: ("M2", "check", None), 27811: ("fund_quotas", "outside", "cdi"), 27812: ("repo_priv", "outside", "selic"),
    1839: ("repo_fed", "outside", "selic"), 27813: ("M3", "check", None), 27814: ("federal", "outside", "selic"),
    27815: ("M4", "total", None)}
QS = (0.10, 0.25, 0.50, 0.75, 0.90)
K5_FIRST_FLOW_Q = 2020 * 4 + 2      # 2020Q3: the first quarter with a lagged type-5 balance


def stamp(msg):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:5.0f}s] {msg}", flush=True)


def md5_file(p, lf=False):
    b = Path(p).read_bytes()
    return hashlib.md5(b.replace(b"\r", b"") if lf else b).hexdigest()


def qlabel(q):
    q = np.asarray(q, int)
    return pd.Series(q // 4).astype(str).str.cat(pd.Series(q % 4 + 1).astype(str), sep="Q").to_numpy()


def wquant(v, w, q):
    v, w = np.asarray(v, float), np.asarray(w, float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[ok], w[ok]
    if not len(v):
        return np.nan
    o = np.argsort(v)
    cw = np.cumsum(w[o]) / w.sum()
    return float(v[o][min(np.searchsorted(cw, q), len(v) - 1)])


# ------------------------------------------------------------------------------------------ inputs
def read_frame(k):
    f = FRAME_DIR / FRAMES[k]
    cols = [BANK, "mca_code", "deposit_type", "time_id", "is_B", "deposit_balance", "lagged_deposits",
            "gross_return_lag", "phi_mt", "phi_t", "Dep_Act", "pop_total"]
    d = pq.read_table(f, columns=cols).to_pandas()
    d[BANK] = d[BANK].astype(str)
    d["mca_code"] = d["mca_code"].astype(str)
    d["is_B"] = d["is_B"].astype(bool)
    d["k"] = d["deposit_type"].astype(int)
    d["q"] = d["time_id"].str[:4].astype(int) * 4 + d["time_id"].str[-1].astype(int) - 1
    meta = pq.read_schema(f).metadata
    return d, {"frame": str(f), "frame_md5": md5_file(f),
               "frame_panel_md5": meta[b"demand_chars.market_panel_md5"].decode()}


def read_cells():
    """Panel cells (B and D), quarters 2015Q4-2024Q4, lags by calendar quarter (outer, so entries and exits)."""
    cols = [BANK, "mca_code", "is_B", "year", "quarter", "dep_a1", "dep_a2", "dep_a4", "dep_a5",
            "dep_a4_incl_judicial", "spread_a1", "spread_a2", "spread_a4", "spread_a5", "risk_free_qoq",
            "pop_total", "gdp_per_capita"]
    p = pq.read_table(PANEL, columns=cols).to_pandas()
    p = p.drop_duplicates(subset=[BANK, "mca_code", "year", "quarter"])
    p = p[p["year"].between(2015, 2024)].copy()
    p[BANK] = p[BANK].astype(str)
    p["mca_code"] = p["mca_code"].astype(str)
    p["is_B"] = pd.to_numeric(p["is_B"], errors="coerce") == 1
    p["q"] = p["year"].astype(int) * 4 + p["quarter"].astype(int) - 1
    rf = p.groupby("q")["risk_free_qoq"].median()           # one national value per quarter
    mk = (p[p["is_B"]].groupby(["mca_code", "q"]).agg(pop=("pop_total", "first"), gdppc=("gdp_per_capita", "first"))
          .reset_index())
    parts = []
    for k in TYPES:
        x = p[[BANK, "mca_code", "is_B", "q", f"dep_a{k}", f"spread_a{k}"]].rename(
            columns={f"dep_a{k}": "D", f"spread_a{k}": "spread"})
        x["Dj"] = p["dep_a4_incl_judicial"].where(p["dep_a4_incl_judicial"].notna(), p["dep_a4"]) if k == 4 else x["D"]
        x["k"] = k
        parts.append(x)
    long = pd.concat(parts, ignore_index=True)
    long[["D", "Dj"]] = long[["D", "Dj"]].fillna(0.0)
    key = [BANK, "mca_code", "k"]
    lag = long[key + ["q", "is_B", "D", "Dj", "spread"]].assign(q=long["q"] + 1).rename(
        columns={"D": "D_lag", "Dj": "Dj_lag", "spread": "spread_lag", "is_B": "is_B_lag"})
    c = long.merge(lag, on=key + ["q"], how="outer")
    c["is_B"] = c["is_B"].fillna(c["is_B_lag"]).astype(bool)
    c = c.drop(columns=["is_B_lag"])
    c[["D", "Dj", "D_lag", "Dj_lag"]] = c[["D", "Dj", "D_lag", "Dj_lag"]].fillna(0.0)
    c = c[c["q"].between(Q0, Q1)].copy()
    c = c[(c["D"] > 0) | (c["D_lag"] > 0)].copy()
    c["rf_lag"] = (c["q"] - 1).map(rf)
    c["R_rebuilt"] = 1.0 + c["rf_lag"] - c["spread_lag"]
    return c.reset_index(drop=True), mk, {"panel_md5": md5_file(PANEL), "panel_rows_2015_2024": int(len(p))}


def national_phi(fr, k):
    """phi_t over the B markets (the fix), the stored column, the version with the NATIONAL row (the bug, gate
    G3), and est{k}/national_phi_t.csv."""
    b = fr[fr["is_B"]].groupby(["mca_code", "q"]).agg(phi=("phi_mt", "first"), pop=("pop_total", "first")).reset_index()
    b = b[b["pop"].notna()]
    num = (b["phi"] * b["pop"]).groupby(b["q"]).sum()
    den = b["pop"].groupby(b["q"]).sum()
    phiB = (num / den).rename("phi_t_B")
    dd = fr[~fr["is_B"]].groupby("q").agg(phi=("phi_mt", "first"), pop=("pop_total", "first"))
    phi_bug = ((num + dd["phi"] * dd["pop"]) / (den + dd["pop"])).rename("phi_t_with_national_row")
    stored = fr.groupby("q")["phi_t"].first().rename("phi_t_stored")
    est = pd.read_csv(paths.est_dir(k) / "national_phi_t.csv")
    est["q"] = est["year_quarter"].str[:4].astype(int) * 4 + est["year_quarter"].str[-1].astype(int) - 1
    estc = est.set_index("q")[SPEC_COL].rename("phi_t_est_csv")
    out = pd.concat([phiB, phi_bug, stored, estc], axis=1).loc[Q0:Q1]
    return out, {"est_csv": str(paths.est_dir(k) / "national_phi_t.csv"), "est_col": SPEC_COL,
                 "G3_max_abs_bug_minus_stored": float((out["phi_t_with_national_row"] - out["phi_t_stored"]).abs().max())}


def build_cells(k, fr, cells, phiN):
    c = cells.copy()
    f = fr[[BANK, "mca_code", "k", "q", "deposit_balance", "lagged_deposits", "gross_return_lag", "Dep_Act",
            "phi_t"]].rename(columns={"Dep_Act": "Dep_Act_frame", "phi_t": "phi_t_frame"})
    c = c.merge(f, on=[BANK, "mca_code", "k", "q"], how="left")
    c["in_frame"] = c["Dep_Act_frame"].notna()
    # R: the frame's own value, else the prep's formula, else the type-quarter deposit-weighted R of the frame
    Rtq = (fr.assign(w=fr["gross_return_lag"] * fr["lagged_deposits"]).groupby(["k", "q"])["w"].sum()
           / fr.groupby(["k", "q"])["lagged_deposits"].sum()).rename("R_tq").reset_index()
    c = c.merge(Rtq, on=["k", "q"], how="left")
    c["R_src"] = np.where(c["in_frame"], "frame", np.where(np.isfinite(c["R_rebuilt"]), "rebuilt", "type_quarter"))
    c["R"] = np.where(c["in_frame"], c["gross_return_lag"], np.where(np.isfinite(c["R_rebuilt"]), c["R_rebuilt"], c["R_tq"]))
    # phi: B from the market-quarter; D from phi_t over the B markets
    mq = fr[fr["is_B"]].groupby(["mca_code", "q"])["phi_mt"].first().rename("phi_mq").reset_index()
    c = c.merge(mq, on=["mca_code", "q"], how="left")
    c["phi_src"] = np.where(c["phi_mq"].notna(), "frame", "")
    miss = c["is_B"] & c["phi_mq"].isna()
    if miss.any():
        mm = c.loc[miss, ["mca_code", "q"]].drop_duplicates()
        cand = mm.merge(mq.rename(columns={"q": "q2"}), on="mca_code", how="inner")
        cand = cand[(cand["q2"] - cand["q"]).abs() <= 4]
        cand["dist"] = (cand["q2"] - cand["q"]).abs() * 2 + (cand["q2"] > cand["q"])
        near = cand.sort_values("dist").groupby(["mca_code", "q"]).head(1)[["mca_code", "q", "phi_mq"]]
        c = c.merge(near.rename(columns={"phi_mq": "phi_near"}), on=["mca_code", "q"], how="left")
        fill = miss & c["phi_near"].notna()
        c.loc[fill, "phi_mq"] = c.loc[fill, "phi_near"]
        c.loc[fill, "phi_src"] = "near"
        c = c.drop(columns=["phi_near"])
    c["phi_t_B"] = c["q"].map(phiN["phi_t_B"])
    c["phi_t_stored"] = c["q"].map(phiN["phi_t_stored"])
    left = c["is_B"] & c["phi_mq"].isna()
    c.loc[left, "phi_mq"] = c.loc[left, "phi_t_B"]
    c.loc[left, "phi_src"] = "phi_t"
    c["phi"] = np.where(c["is_B"], c["phi_mq"], c["phi_t_B"])
    c.loc[~c["is_B"], "phi_src"] = "phi_t_B"
    c["phi_prod"] = np.where(c["is_B"], c["phi_mq"], c["phi_t_stored"])     # production's (11-B), D with the bug
    c["C"] = c["R"] * c["D_lag"]
    c["Cj"] = c["R"] * c["Dj_lag"]
    c["Dep_Act"] = np.maximum(0.0, c["D"] - c["phi"] * c["C"])
    c["Dep_Act_prod"] = np.maximum(0.0, c["D"] - c["phi_prod"] * c["C"])
    c["X"] = c["D"] - c["C"]
    for lab, cap in CAPS.items():
        ph = np.minimum(c["phi"], cap)
        c[f"A_{lab}"] = np.maximum((1 - ph) * c["C"], c["C"] - c["D"])
    return c


def gates_cells(c):
    f = c[c["in_frame"]]
    g = {}
    rel = lambda a, b: float(np.nanmax(np.abs(a - b) / np.maximum(np.abs(b), 1.0))) if len(a) else np.nan
    g["cells"] = int(len(c))
    g["cells_in_frame"] = int(len(f))
    g["D_vs_frame_max_rel"] = rel(f["D"].to_numpy(), f["deposit_balance"].to_numpy())
    for lab, sub in (("all", f), ("2016Q1", f[f["q"] == Q0]), ("after_2016Q1", f[f["q"] > Q0])):
        g[f"Dlag_vs_frame_max_rel_{lab}"] = rel(sub["D_lag"].to_numpy(), sub["lagged_deposits"].to_numpy())
        g[f"Dlag_vs_frame_n_rel_gt_1e-6_{lab}"] = int((np.abs(sub["D_lag"] - sub["lagged_deposits"])
                                                      / np.maximum(sub["lagged_deposits"].abs(), 1.0) > 1e-6).sum())
    ok = np.isfinite(f["R_rebuilt"])
    g["R_rebuilt_vs_frame_max_abs"] = float(np.abs(f.loc[ok, "R_rebuilt"] - f.loc[ok, "gross_return_lag"]).max())
    g["R_rebuilt_vs_frame_share_within_1e-9"] = float((np.abs(f.loc[ok, "R_rebuilt"] - f.loc[ok, "gross_return_lag"]) < 1e-9).mean())
    # (11-B) with the frame's own lag and R and production's phi: must equal the stored Dep_Act
    da = np.maximum(0.0, f["deposit_balance"] - f["phi_prod"] * f["gross_return_lag"] * f["lagged_deposits"])
    g["DepAct_frame_inputs_vs_stored_max_rel"] = rel(da.to_numpy(), f["Dep_Act_frame"].to_numpy())
    g["DepAct_mine_prod_phi_vs_stored_max_rel_after_2016Q1"] = rel(
        f.loc[f["q"] > Q0, "Dep_Act_prod"].to_numpy(), f.loc[f["q"] > Q0, "Dep_Act_frame"].to_numpy())
    pos_out = (c["Dep_Act_prod"] > 1e-6) & ~c["in_frame"]
    g["positive_DepAct_not_in_frame"] = {
        "cells": int(pos_out.sum()), "entries_D_lag_0": int((pos_out & (c["D_lag"] <= 0)).sum()),
        "DepAct_share_of_all_cells_DepAct": float(c.loc[pos_out, "Dep_Act_prod"].sum() / c["Dep_Act_prod"].sum()),
        "DepAct_share_entries": float(c.loc[pos_out & (c["D_lag"] <= 0), "Dep_Act_prod"].sum() / c["Dep_Act_prod"].sum())}
    g["R_src_carry_share"] = {s: float(c.loc[c["R_src"] == s, "C"].sum() / c["C"].sum()) for s in ("frame", "rebuilt", "type_quarter")}
    g["phi_src_carry_share_B"] = {s: float(c.loc[c["is_B"] & (c["phi_src"] == s), "C"].sum() / c.loc[c["is_B"], "C"].sum())
                                  for s in ("frame", "near", "phi_t")}
    return g


# --------------------------------------------------------------------------------- outside stocks
IFD_R3 = {"78282": "ifd_a1", "78283": "ifd_a2", "78286": "ifd_a4", "110560": "ifd_a5", "78285": "ifd_a6",
          "78289": "LCI", "78290": "LCA", "78291": "LF", "78293": "other_bills", "78294": "bills_total",
          "78288": "repos_liab"}


def ifdata_by_institution():
    """IF.Data type 1 (prudential conglomerates and independent institutions), per institution and quarter:
    the report 3 stocks of IFD_R3 and report 4's funding expense 78209, with the register's prudential
    segment (List, column Sr). The files are read as latin-1 (a few bytes are undefined in cp1252; only
    ASCII columns are used)."""
    rows = []
    for y in range(2015, 2025):
        for m in (3, 6, 9, 12):
            if (y, m) < (2015, 12):
                continue
            f = paths.IF_DATA_PRUDENTIAL / f"IF_DATA_Values_{y}_{m}.csv"
            d = pd.read_csv(f, encoding="latin-1", dtype=str, usecols=["CodInst", "NumeroRelatorio", "Conta", "Saldo"])
            d = d[((d["NumeroRelatorio"] == "3") & d["Conta"].isin(IFD_R3)) |
                  ((d["NumeroRelatorio"] == "4") & (d["Conta"] == "78209"))].copy()
            d["v"] = pd.to_numeric(d["Saldo"].str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
                                   errors="coerce")
            d["q"] = y * 4 + m // 3 - 1
            L = pd.read_csv(paths.IF_DATA_LIST / f"IF_DATA_List_{y}_{m}.csv", dtype=str, usecols=["CodInst", "Sr", "NomeInstituicao", "Tcb"])
            d = d.merge(L, on="CodInst", how="left")
            rows.append(d[["q", "CodInst", "NomeInstituicao", "Sr", "Tcb", "Conta", "v"]])
    return pd.concat(rows, ignore_index=True)


def ifdata_national(inst):
    """National sums by quarter; 78209 is cumulative within the semester in the files, so it is
    de-cumulated per institution (Q2 - Q1 and Q4 - Q3)."""
    r3 = inst[inst["Conta"] != "78209"]
    t = r3.pivot_table(index="q", columns="Conta", values="v", aggfunc="sum").rename(columns=IFD_R3)
    e = inst[inst["Conta"] == "78209"].groupby(["q", "CodInst"])["v"].sum()
    flow = {}
    for q in sorted(e.index.get_level_values(0).unique()):
        cur = e.loc[q]
        if q % 2 == 1 and (q - 1) in e.index.get_level_values(0):
            prev = e.loc[q - 1].reindex(cur.index).fillna(0.0)
            flow[q] = float((cur - prev).sum())
        else:
            flow[q] = float(cur.sum())
    t["ifd_funding_expense"] = -pd.Series(flow)
    return t


def cosif_expense():
    f = paths.PROCESSED / "COSIF_PROCESSED" / "custos_implicitos_v2_foundation.csv"
    d = pd.read_csv(f, usecols=["CNPJ", "AnoMes", "exp_lump"])
    s = -d.groupby("AnoMes")["exp_lump"].sum()
    s.index = (s.index // 100) * 4 + (s.index % 100) // 3 - 1
    return s.rename("cosif_exp_lump"), str(f)


def tesouro_direto():
    import openpyxl
    wb = openpyxl.load_workbook(B3_RI, read_only=True, data_only=True)
    ws = wb[wb.sheetnames[3]]
    rows = list(ws.iter_rows(min_row=40, max_row=42, values_only=True))
    dates, stock = rows[0][1:], rows[2][1:]
    if not str(rows[2][0]).startswith("Estoque"):
        raise SystemExit(f"B3 RI: row 42 is {rows[2][0]!r}, not the Tesouro Direto stock")
    s = pd.Series({pd.Timestamp(d): float(v) * 1e9 for d, v in zip(dates, stock) if d is not None and v is not None})
    s = s[s.index.month.isin([3, 6, 9, 12])]
    s.index = s.index.year * 4 + s.index.month // 3 - 1
    return s.rename("TD")


def read_aggregates(agg_dir):
    files = {c: Path(agg_dir) / f"sgs_{c}_2015-2024.json" for c in AGG}
    if not all(f.exists() for f in files.values()):
        return None, {"agg_dir": str(agg_dir), "present": int(sum(f.exists() for f in files.values()))}
    out = {}
    for c, f in files.items():
        j = json.loads(f.read_text(encoding="utf-8"))
        s = pd.Series({pd.to_datetime(r["data"], dayfirst=True): float(r["valor"]) * 1e3 for r in j})
        s = s[s.index.month.isin([3, 6, 9, 12])]
        s.index = s.index.year * 4 + s.index.month // 3 - 1
        out[AGG[c][0]] = s
    return pd.DataFrame(out), {"agg_dir": str(agg_dir), "present": 16, "md5": {c: md5_file(f) for c, f in files.items()}}


def fgc_persons_share():
    t = pq.read_table(FGC).to_pandas()
    t = t[(t["cod_competencia"] % 100 % 3 == 0) & (t["cod_competencia"] <= 202412)]
    grp = {1: "demand", 2: "savings", 3: "time", 4: "time", 7: "LC", 8: "LC"}
    t["g"] = t["num_produto"].map(grp)
    t = t[t["g"].notna()]
    s = t.groupby(["cod_competencia", "g", "num_titularidade"])["val_deposito"].sum().unstack("num_titularidade").fillna(0)
    sh = (s[1] / s.sum(axis=1)).unstack("g")
    sh.index = (sh.index // 100) * 4 + (sh.index % 100) // 3 - 1
    return sh.add_prefix("fgc_persons_share_")


# --------------------------------------------------------------------------------------- universes
def factors(c, target, q_idx):
    """f_k,t = target stock / panel stock (all holders), for the types in `target` ({k: series by q}); the other
    types keep f = 1. The panel stock at t-1 is the cells' summed lagged balance at t (entries and exits in)."""
    rows = []
    for kk, A in target.items():
        sub = c[c["k"] == kk]
        S = sub.groupby("q")["D"].sum().reindex(q_idx)
        S_prev = sub.groupby("q")["D_lag"].sum().reindex(q_idx)
        rows.append(pd.DataFrame({"q": q_idx, "k": kk, "f": (A.reindex(q_idx) / S).to_numpy(),
                                  "f_lag": (A.reindex(q_idx - 1).to_numpy() / S_prev.to_numpy())}))
    return pd.concat(rows, ignore_index=True)


def cell_view(c, fac=None):
    """The cells of a universe: the panel's (fac None) or with their STOCKS rescaled by type and quarter (D by f_t,
    the lagged balance by f_t-1) and (11-B) recomputed on them: f_t D_t - phi R f_t-1 D_t-1. Also kept, for the
    comparison only: the active balance scaled after the fact, f_t max{0, D_t - phi R D_t-1}."""
    v = c[[BANK, "mca_code", "q", "k", "is_B", "phi", "R", "D", "D_lag"]].copy()
    v["f"] = 1.0
    v["DA_fscaled"] = np.maximum(0.0, v["D"] - v["phi"] * v["R"] * v["D_lag"])
    if fac is not None:
        v = v.merge(fac[["q", "k", "f", "f_lag"]], on=["q", "k"], how="left", suffixes=("_x", ""))
        v = v.drop(columns=["f_x"])
        v[["f", "f_lag"]] = v[["f", "f_lag"]].fillna(1.0)
        v["DA_fscaled"] = v["f"] * v["DA_fscaled"]
        v["D"] = v["f"] * v["D"]
        v["D_lag"] = v["f_lag"] * v["D_lag"]
    v["C"] = v["R"] * v["D_lag"]
    v["X"] = v["D"] - v["C"]
    v["Dep_Act"] = np.maximum(0.0, v["D"] - v["phi"] * v["C"])
    v["Dep_Act_untrunc"] = v["D"] - v["phi"] * v["C"]
    for lab, cap in CAPS.items():
        v[f"A_{lab}"] = np.maximum((1 - np.minimum(v["phi"], cap)) * v["C"], v["C"] - v["D"])
    return v.reset_index(drop=True)


def outside_pools(comps, q_idx):
    O = pd.DataFrame(index=q_idx)
    for n, (s, R) in comps.items():
        s = s.reindex(range(Q0 - 1, Q1 + 1))
        O[f"Out_{n}"] = s.reindex(q_idx).to_numpy()
        O[f"C0_{n}"] = R.reindex(q_idx).to_numpy() * s.shift(1).reindex(q_idx).to_numpy()
    return O


def national_universe(v, comps, phi_t_B, q_idx):
    g = v.groupby("q")
    R = pd.DataFrame(index=q_idx)
    R["S_in"] = g["D"].sum().reindex(q_idx)
    R["C_in"] = g["C"].sum().reindex(q_idx)
    R["N_in"] = R["S_in"] - R["C_in"]
    O = outside_pools(comps, q_idx)
    names = list(comps)
    R["Out"] = O[[f"Out_{n}" for n in names]].sum(axis=1, min_count=len(names)) if names else 0.0
    R["C0"] = O[[f"C0_{n}" for n in names]].sum(axis=1, min_count=len(names)) if names else 0.0
    R["N_out"] = R["Out"] - R["C0"]
    R["W"] = R["S_in"] + R["Out"]
    R["C"] = R["C_in"] + R["C0"]
    R["N"] = R["W"] - R["C"]
    R["N_over_W"] = R["N"] / R["W"]
    R["N_over_inside"] = R["N"] / R["S_in"]
    R["Out_share_of_W"] = R["Out"] / R["W"]
    R["DepAct_trunc"] = g["Dep_Act"].sum().reindex(q_idx)
    R["DepAct_untrunc"] = g["Dep_Act_untrunc"].sum().reindex(q_idx)
    R["DepAct_fscaled_after"] = g["DA_fscaled"].sum().reindex(q_idx)
    R["DepAct_B"] = v[v["is_B"]].groupby("q")["Dep_Act"].sum().reindex(q_idx)
    R["DepAct_D"] = v[~v["is_B"]].groupby("q")["Dep_Act"].sum().reindex(q_idx)
    ph0 = phi_t_B.reindex(q_idx).to_numpy()
    for lab, cap in CAPS.items():
        ph = np.minimum(ph0, cap)
        outA = np.zeros(len(q_idx))
        for n in names:
            C0n, D0n = O[f"C0_{n}"].to_numpy(), O[f"Out_{n}"].to_numpy()
            outA = outA + np.maximum((1 - ph) * C0n, C0n - D0n)
        R[f"A_in_{lab}"] = g[f"A_{lab}"].sum().reindex(q_idx)
        R[f"A_B_{lab}"] = v[v["is_B"]].groupby("q")[f"A_{lab}"].sum().reindex(q_idx)
        R[f"A_out_{lab}"] = outA
        R[f"PB_{lab}"] = R[f"A_in_{lab}"] + outA + R["N"]
        R[f"PA_{lab}"] = (1 - ph) * R["C"] + R["N"]
        R[f"outside_share_PB_{lab}"] = 1 - R["DepAct_trunc"] / R[f"PB_{lab}"]
        R[f"outside_share_PA_trunc_{lab}"] = 1 - R["DepAct_trunc"] / R[f"PA_{lab}"]
        R[f"outside_share_PA_untrunc_{lab}"] = 1 - R["DepAct_untrunc"] / R[f"PA_{lab}"]
        R[f"realized_awake_share_{lab}"] = (R[f"PB_{lab}"] - R["N"]) / R["C"]
        R[f"flag_PA_below_active_{lab}"] = R[f"PA_{lab}"] < R["DepAct_trunc"]
        R[f"flag_PB_below_active_{lab}"] = R[f"PB_{lab}"] < R["DepAct_trunc"] * (1 - 1e-9)
    R["phi_t_B"] = ph0
    R["one_minus_phi_t"] = 1 - ph0
    R["carry_weighted_one_minus_phi_in"] = (((1 - v["phi"]) * v["C"]).groupby(v["q"]).sum().reindex(q_idx) / R["C_in"])
    R["flag_N_negative"] = R["N"] < 0
    R["gate_G1_rel_gap"] = (R["N_in"] + R["A_in_none"] - R["DepAct_trunc"]) / R["DepAct_trunc"]
    return R.join(O)


# ------------------------------------------------------------------------------------------- local
def netting(v):
    """Symmetric netting of booking moves inside a bank (B cells): per bank and quarter, the net losses of some
    of its bank-markets (deposit types netted inside a bank-market) matched by the net gains of others, B_j =
    min(gains, losses). The matched amount leaves the losing cells' awake money (pro rata to their losses) and the
    gaining cells' active balances (pro rata to their gains). Returns per-cell gain and loss taken, and B by q."""
    b = v["is_B"]
    jm = v[b].groupby([BANK, "mca_code", "q"])["X"].sum().rename("Xjm").reset_index()
    jm["G"] = jm["Xjm"].clip(lower=0)
    jm["L"] = (-jm["Xjm"]).clip(lower=0)
    jt = jm.groupby([BANK, "q"])[["G", "L"]].sum().rename(columns={"G": "Gj", "L": "Lj"}).reset_index()
    jt["Bj"] = np.minimum(jt["Gj"], jt["Lj"])
    jm = jm.merge(jt, on=[BANK, "q"], how="left")
    jm["g_jm"] = np.where(jm["Gj"] > 0, jm["Bj"] * jm["G"] / jm["Gj"].where(jm["Gj"] > 0, 1.0), 0.0)
    jm["l_jm"] = np.where(jm["Lj"] > 0, jm["Bj"] * jm["L"] / jm["Lj"].where(jm["Lj"] > 0, 1.0), 0.0)
    cc = v[[BANK, "mca_code", "q", "is_B", "X"]].merge(jm[[BANK, "mca_code", "q", "g_jm", "l_jm"]],
                                                       on=[BANK, "mca_code", "q"], how="left")
    gain, loss = cc["X"].clip(lower=0), (-cc["X"]).clip(lower=0)
    key = [cc[BANK], cc["mca_code"], cc["q"]]
    sg = gain.groupby(key).transform("sum")
    sl = loss.groupby(key).transform("sum")
    g_c = np.where(cc["is_B"] & (sg > 0), cc["g_jm"].fillna(0) * gain / sg.where(sg > 0, 1.0), 0.0)
    l_c = np.where(cc["is_B"] & (sl > 0), cc["l_jm"].fillna(0) * loss / sl.where(sl > 0, 1.0), 0.0)
    return g_c, l_c, jt.groupby("q")["Bj"].sum()


def local_universe(v, R, acc, uname, k, w4=True):
    rows, wrows = [], []
    b = v["is_B"].to_numpy()
    q_idx = R.index
    for netted in (False, True):
        if netted:
            g_c, l_c, Bq = netting(v)
        else:
            g_c, l_c, Bq = np.zeros(len(v)), np.zeros(len(v)), pd.Series(0.0, index=q_idx)
        DA = np.maximum(0.0, v["Dep_Act"].to_numpy() - g_c)
        lagpos = (v["D_lag"].to_numpy() > 0) & b
        tr_cells = float(((DA <= 1e-6) & lagpos).sum() / lagpos.sum())
        tr_stock = float(v["D_lag"].to_numpy()[(DA <= 1e-6) & lagpos].sum() / v["D_lag"].to_numpy()[lagpos].sum())
        for lab in CAPS:
            A = np.maximum(0.0, v[f"A_{lab}"].to_numpy() - l_c)
            cb = pd.DataFrame({"mca_code": v["mca_code"].to_numpy()[b], "q": v["q"].to_numpy()[b],
                               "bank": v[BANK].to_numpy()[b], "DA": DA[b], "A": A[b], "C": v["C"].to_numpy()[b],
                               "D_lag": v["D_lag"].to_numpy()[b]})
            m = cb.groupby(["mca_code", "q"]).agg(ACT=("DA", "sum"), AWK=("A", "sum"), CAR=("C", "sum"),
                                                  S_lag=("D_lag", "sum")).reset_index()
            m = m.merge(acc[["mca_code", "q", "L1w"]], on=["mca_code", "q"], how="left")
            m["L1w"] = m["L1w"].fillna(0.0)
            Xt = (R[f"PB_{lab}"] - R[f"A_B_{lab}"])                  # unchanged by the netting
            Dawk = (R[f"A_in_{lab}"] - R[f"A_B_{lab}"])
            m["X"] = m["q"].map(Xt)
            m["Dawk"] = m["q"].map(Dawk)
            w1 = m["L1w"] / m.groupby("q")["L1w"].transform("sum")
            w2 = m["S_lag"] / m.groupby("q")["S_lag"].transform("sum")
            for rule, a in (("L1_pop_access", w1), ("L2_lagged_stock", w2), ("L3_mix", 0.5 * (w1 + w2))):
                P = m["AWK"] + a * m["X"]
                valid = P > 0
                osh = 1 - m["ACT"] / P
                row = {"routine": f"E{k}", "universe": uname, "cap": lab, "netting": "symmetric" if netted else "none",
                       "rule": rule, "market_quarters": int(len(m)), "P_le_0": int((~valid).sum()),
                       "share_ACT_gt_0.80P": float(((m["ACT"] > 0.80 * P) | ~valid).mean()),
                       "share_ACT_gt_0.95P": float(((m["ACT"] > 0.95 * P) | ~valid).mean()),
                       "share_ACT_gt_P": float(((m["ACT"] > P) | ~valid).mean()),
                       "stockw_share_ACT_gt_0.80P": float(m.loc[(m["ACT"] > 0.80 * P) | ~valid, "S_lag"].sum() / m["S_lag"].sum()),
                       "stockw_share_ACT_gt_0.95P": float(m.loc[(m["ACT"] > 0.95 * P) | ~valid, "S_lag"].sum() / m["S_lag"].sum()),
                       "trunc_share_cells_B": tr_cells, "trunc_share_lagstock_B": tr_stock,
                       "netted_share_of_B_carry": float(Bq.sum() / v.loc[v["is_B"], "C"].sum())}
                for qq in QS:
                    row[f"outside_share_p{int(qq * 100)}"] = float(np.quantile(osh[valid], qq)) if valid.any() else np.nan
                    row[f"outside_share_stockw_p{int(qq * 100)}"] = wquant(osh[valid], m.loc[valid, "S_lag"], qq)
                rows.append(row)
                if w4 and lab == "none":
                    jm = cb.groupby(["bank", "mca_code", "q"]).agg(DA=("DA", "sum"), A=("A", "sum"), C=("C", "sum")).reset_index()
                    jm = jm[jm["DA"] > 1e-6]
                    mt = m[["mca_code", "q", "AWK", "X", "Dawk"]].assign(a=a.to_numpy(), P=P.to_numpy())
                    jm = jm.merge(mt, on=["mca_code", "q"], how="left")
                    jm = jm[jm["P"] > 0]
                    own = jm["A"] / jm["P"]
                    riv = (jm["AWK"] - jm["A"] + jm["a"] * jm["Dawk"]) / jm["P"]
                    new = jm["a"] * (jm["X"] - jm["Dawk"]) / jm["P"]
                    w = jm["C"].to_numpy()
                    wr = {"routine": f"E{k}", "universe": uname, "netting": "symmetric" if netted else "none",
                          "rule": rule, "dest_units": int(len(jm))}
                    for nm, val in (("own", own), ("rivals", riv), ("new", new)):
                        val = val.to_numpy()
                        wr[f"{nm}_p10"] = float(np.quantile(val, 0.10))
                        wr[f"{nm}_p50"] = float(np.median(val))
                        wr[f"{nm}_p90"] = float(np.quantile(val, 0.90))
                        wr[f"{nm}_carryw_p50"] = wquant(val, w, 0.5)
                        wr[f"{nm}_carryw_mean"] = float(np.average(val, weights=np.maximum(w, 1e-12)))
                    wrows.append(wr)
    return rows, wrows


def gate_w4():
    """W4.1's production numbers from the K2 context, through diag_market_size_alt.W4 (its gate W)."""
    import diag_market_size_alt as dma
    w = dma.W4(dma.CTX_DEFAULT)
    ref = json.loads(dma.W41_JSON.read_text(encoding="utf-8"))
    m = w.metrics(w.M_hat)
    mine = {"b_new_dest_p50": m["b_new_dest"]["p50"], "b_new_dest_p10": m["b_new_dest"]["p10"],
            "b_new_dest_stock_w_p50": m["b_new_dest_own_stock_weighted"]["p50"],
            "new_share_W_p50": m["W_composition_availability_0p5"]["new"]["p50"]}
    theirs = {"b_new_dest_p50": ref["pools"]["base_cap099"]["b_new"]["unweighted"]["p50"],
              "b_new_dest_p10": ref["pools"]["base_cap099"]["b_new"]["unweighted"]["p10"],
              "b_new_dest_stock_w_p50": ref["pools"]["base_cap099"]["b_new"]["own_stock_weighted"]["p50"],
              "new_share_W_p50": ref["shifters"]["base_cap099"]["cost_only"]["composition_new"]["p50"]}
    d = max(abs(mine[x] - theirs[x]) for x in mine)
    return {"mine": mine, "w4_1": theirs, "max_abs_diff": d, "pass": bool(d < 1e-12), "ctx": str(dma.CTX_DEFAULT),
            "W_composition_availability_0p5_production": m["W_composition_availability_0p5"]}


# -------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routine", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--step", default="national,local")
    ap.add_argument("--agg-dir", default=str(AGG_DIR))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--no-w4-gate", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    steps = set(args.step.split(","))
    q_idx = pd.Index(np.arange(Q0, Q1 + 1), name="q")
    stamp(f"reading {PANEL.name}")
    cells, mk, prec = read_cells()
    # type 5 is reported from 2020Q2 (IF.Data 110560): no flow is formed before 2020Q3, as in the prep
    cells = cells[~((cells["k"] == 5) & (cells["q"] <= K5_FIRST_FLOW_Q - 1))].reset_index(drop=True)
    stamp(f"cells {len(cells):,}; panel md5 {prec['panel_md5'][:8]}")
    macro = macro_rates.make_quarterly_macro(macro_rates.load_macro_series())
    macro["q"] = (macro["AnoMes"] // 100) * 4 + (macro["AnoMes"] % 100) // 3 - 1
    rate = macro.set_index("q")
    ifd_inst = ifdata_by_institution()
    ifd = ifdata_national(ifd_inst)
    cexp, cexp_file = cosif_expense()
    td = tesouro_direto()
    agg, agg_info = read_aggregates(args.agg_dir)
    if agg is None:
        raise SystemExit(f"the 16 aggregates are not all in {args.agg_dir}: {agg_info}")
    fgc = fgc_persons_share()
    stamp("inputs read")
    cdiR = pd.Series(1 + rate["cdi_qoq"], index=rate.index).reindex(range(Q0 - 1, Q1 + 1))
    selR = pd.Series(1 + rate["selic_qoq"], index=rate.index).reindex(range(Q0 - 1, Q1 + 1))
    zero = pd.Series(1.0, index=range(Q0 - 1, Q1 + 1))
    ret = {"zero": zero, "cdi": cdiR, "selic": selR}
    out_comps = {AGG[cc][0]: (agg[AGG[cc][0]], ret[AGG[cc][2]]) for cc in AGG if AGG[cc][1] == "outside"}
    m4_check = (agg["currency"] + agg["demand"] + agg["savings"] + agg["time"] + agg["LF"] + agg["LC"]
                + agg["other_priv"] + agg["fund_quotas"] + agg["repo_priv"] + agg["repo_fed"] + agg["federal"])
    summary = {"inputs": {**prec, "aggregates": agg_info, "ifdata_dir": str(paths.IF_DATA_PRUDENTIAL),
                          "b3_ri": str(B3_RI), "fgc": str(FGC), "cosif_foundation": cexp_file,
                          "prep_lf_md5": md5_file(Path(__file__).resolve().parents[1] / "sleep_demand_prep_link.py", lf=True),
                          "driver_lf_md5": md5_file(Path(__file__), lf=True),
                          "M4_identity_max_rel_gap": float(((m4_check - agg["M4"]) / agg["M4"]).abs().max())},
               "routines": {}}
    if "local" in steps and not args.no_w4_gate:
        summary["gate_W"] = gate_w4()
        stamp(f"gate W: pass {summary['gate_W']['pass']} (max diff {summary['gate_W']['max_abs_diff']:.2e})")
    acc = mk.copy()
    acc["year"] = acc["q"] // 4
    acc["acc"] = (acc["gdppc"] / acc.groupby("year")["gdppc"].transform("median")).clip(0.35, 2.5).fillna(1.0)
    acc["L1w"] = acc["pop"].fillna(0.0) * acc["acc"]
    written = []
    for k in args.routine:
        fr, finfo = read_frame(k)
        if finfo["frame_panel_md5"] != prec["panel_md5"]:
            raise SystemExit(f"E{k}: the frame's panel {finfo['frame_panel_md5'][:8]} is not {prec['panel_md5'][:8]}")
        phiN, pinfo = national_phi(fr, k)
        c = build_cells(k, fr, cells, phiN)
        g = gates_cells(c)
        stamp(f"E{k}: {len(c):,} cells; G2 {g['DepAct_frame_inputs_vs_stored_max_rel']:.1e} / R {g['R_rebuilt_vs_frame_max_abs']:.1e}")
        # ---- the national inputs: phi_t, stocks by type and B / D, IF.Data, the aggregates, the reconciliations
        T = pd.DataFrame(index=q_idx)
        T["time_id"] = qlabel(q_idx)
        T = T.join(phiN)
        for bb, nmB in ((True, "B"), (False, "D")):
            for kk in TYPES:
                T[f"S_{nmB}_k{kk}"] = c[(c["is_B"] == bb) & (c["k"] == kk)].groupby("q")["D"].sum()
        T["S_panel_k4_incl_judicial"] = c[c["k"] == 4].groupby("q")["Dj"].sum()
        T["judicial_in_panel"] = T["S_panel_k4_incl_judicial"] - c[c["k"] == 4].groupby("q")["D"].sum()
        T = T.join(ifd)
        T = T.join(agg.add_prefix("agg_"))
        for kk, col in ((1, "ifd_a1"), (2, "ifd_a2"), (4, "ifd_a4"), (5, "ifd_a5")):
            T[f"recon1_panel_over_ifdata_k{kk}"] = (T[f"S_B_k{kk}"] + T[f"S_D_k{kk}"]) / T[col]
        T["recon1_panel_incl_judicial_over_ifdata_k4"] = T["S_panel_k4_incl_judicial"] / T["ifd_a4"]
        acc_k = (c["R"] - 1) * c["D_lag"]
        for kk in TYPES:
            sel = c["k"] == kk
            T[f"accrual_k{kk}"] = acc_k[sel].groupby(c.loc[sel, "q"]).sum()
            T[f"accrual_rate_k{kk}"] = T[f"accrual_k{kk}"] / c[sel].groupby("q")["D_lag"].sum()
        T["accrual_all"] = acc_k.groupby(c["q"]).sum()
        T["cosif_exp_lump"] = cexp
        for col in ("cdi_qoq", "selic_qoq", "savings_rate_qoq", "meta_selic"):
            T[col] = rate[col].reindex(q_idx)
        T["d_meta_selic_pp"] = rate["meta_selic"].diff().reindex(q_idx)
        T["flag_selic_move_ge_1pp"] = T["d_meta_selic_pp"].abs() >= 1.0
        T["nonDeposit_expense_proxy"] = T["repos_liab"].shift(1) * T["selic_qoq"] + T["bills_total"].shift(1) * T["cdi_qoq"]
        T["recon2_accrual_over_ifd_expense"] = T["accrual_all"] / T["ifd_funding_expense"]
        T["recon2_accrual_over_ifd_expense_net_proxy"] = T["accrual_all"] / (T["ifd_funding_expense"] - T["nonDeposit_expense_proxy"])
        T["TD_stock"] = td.reindex(q_idx)
        T = T.join(fgc)
        # ---- the factors: split (i) inside = the aggregates' deposit components; split (ii) inside = IF.Data
        S5 = c[c["k"] == 5].groupby("q")["D"].sum()
        tgt_i = {1: agg["demand"], 2: agg["savings"], 4: agg["time"]}
        ifd_q = ifd.reindex(range(Q0 - 1, Q1 + 1))
        jud = (c[c["k"] == 4].groupby("q")["Dj"].sum() - c[c["k"] == 4].groupby("q")["D"].sum())
        jud_prev = (c[c["k"] == 4].groupby("q")["Dj_lag"].sum() - c[c["k"] == 4].groupby("q")["D_lag"].sum())
        jud_full = pd.concat([pd.Series({Q0 - 1: float(jud_prev.loc[Q0])}), jud]).sort_index()
        tgt_ii = {1: ifd_q["ifd_a1"], 2: ifd_q["ifd_a2"], 4: ifd_q["ifd_a4"] - jud_full.reindex(ifd_q.index), 5: ifd_q["ifd_a5"]}
        fac_i, fac_ii = factors(c, tgt_i, q_idx), factors(c, tgt_ii, q_idx)
        for kk in (1, 2, 4):
            T[f"factor_i_k{kk}"] = fac_i[fac_i["k"] == kk].set_index("q")["f"]
        for kk in (1, 2, 4, 5):
            T[f"factor_ii_k{kk}"] = fac_ii[fac_ii["k"] == kk].set_index("q")["f"]
        # institutions in IF.Data type 1 but not in the panel (credit unions first), by register type Tcb
        codes = set(c[BANK].astype(str))
        nip = ifd_inst[~ifd_inst["CodInst"].isin(codes) & ifd_inst["Conta"].isin(["78282", "78283", "78286", "110560"])].copy()
        nip["typ"] = nip["Conta"].map({"78282": "k1", "78283": "k2", "78286": "k4", "110560": "k5"})
        nip["grp"] = nip["Tcb"].where(nip["Tcb"].isin(["B3S", "B3C", "B1", "B2"]), "other")
        npt = nip.pivot_table(index="q", columns=["grp", "typ"], values="v", aggfunc="sum").fillna(0.0)
        npt.columns = [f"notpanel_{a}_{b}" for a, b in npt.columns]
        npt = npt.reindex(range(Q0 - 1, Q1 + 1)).fillna(0.0)
        for col in npt.columns:
            T[col] = npt[col].reindex(q_idx)
        cu = {kk: npt.get(f"notpanel_B3S_k{kk}", pd.Series(0.0, index=npt.index)) for kk in (1, 2, 4)}
        allnp = {kk: sum(npt[cn] for cn in npt.columns if cn.endswith(f"_k{kk}")) for kk in (1, 2, 4)}
        tgt_cu = {1: agg["demand"] - cu[1].reindex(agg.index).fillna(0.0), 2: agg["savings"] - cu[2].reindex(agg.index).fillna(0.0),
                  4: agg["time"] - cu[4].reindex(agg.index).fillna(0.0)}
        tgt_np = {1: agg["demand"] - allnp[1].reindex(agg.index).fillna(0.0), 2: agg["savings"] - allnp[2].reindex(agg.index).fillna(0.0),
                  4: agg["time"] - allnp[4].reindex(agg.index).fillna(0.0)}
        fac_cu, fac_np = factors(c, tgt_cu, q_idx), factors(c, tgt_np, q_idx)
        for kk in (1, 2, 4):
            T[f"factor_i_cmp_noB3S_k{kk}"] = fac_cu[fac_cu["k"] == kk].set_index("q")["f"]
            T[f"factor_i_cmp_noNotPanel_k{kk}"] = fac_np[fac_np["k"] == kk].set_index("q")["f"]
        T["M4"] = agg["M4"].reindex(q_idx)
        T["W_M4_plus_prepaid"] = T["M4"] + S5.reindex(q_idx).fillna(0.0)
        T.to_csv(out / f"national_inputs_E{k}.csv", index_label="q")
        written.append(out / f"national_inputs_E{k}.csv")
        # ---- the universes
        v_dep = cell_view(c)
        v_i = cell_view(c, fac_i)
        v_ii = cell_view(c, fac_ii)
        res = {}
        res["dep"] = (v_dep, national_universe(v_dep, {}, phiN["phi_t_B"], q_idx))
        R_i = national_universe(v_i, out_comps, phiN["phi_t_B"], q_idx)
        res["split_i_agg"] = (v_i, R_i)
        res["split_i_agg_no_federal"] = (v_i, national_universe(v_i, {n: x for n, x in out_comps.items() if n != "federal"},
                                                                phiN["phi_t_B"], q_idx))
        res["split_i_agg_funds_cdi_minus_1pct"] = (v_i, national_universe(
            v_i, {**out_comps, "fund_quotas": (agg["fund_quotas"], cdiR * 0.99 ** 0.25)}, phiN["phi_t_B"], q_idx))
        # split (i) with the single credit unions in the outside good (they are not goods of the demand model):
        # the panel's stocks scaled to aggregate - B3S, and the B3S stocks as outside components (demand at zero,
        # savings at the savings rate, time at the CDI)
        savR = pd.Series(1 + rate["savings_rate_qoq"], index=rate.index).reindex(range(Q0 - 1, Q1 + 1))
        v_cu = cell_view(c, fac_cu)
        cu_comps = {**out_comps, "cu_demand": (cu[1], zero), "cu_savings": (cu[2], savR), "cu_time": (cu[4], cdiR)}
        res["split_i_agg_cu_outside"] = (v_cu, national_universe(v_cu, cu_comps, phiN["phi_t_B"], q_idx))
        # split (ii): the same wealth (M4 + prepaid) and the same N_t; the outside is the rest, one block, with the
        # carry that keeps C_t equal to split (i)'s
        W_full = (agg["M4"] + S5.reindex(agg.index).fillna(0.0)).reindex(range(Q0 - 1, Q1 + 1))
        S_in_ii = v_ii.groupby("q")["D"].sum()
        S_in_ii_prev = v_ii.groupby("q")["D_lag"].sum()
        out_ii = W_full - pd.concat([pd.Series({Q0 - 1: float(S_in_ii_prev.loc[Q0])}), S_in_ii]).sort_index().reindex(W_full.index)
        C0_ii = (R_i["C"] - v_ii.groupby("q")["C"].sum().reindex(q_idx))
        R_eff = pd.Series(C0_ii.to_numpy() / out_ii.shift(1).reindex(q_idx).to_numpy(), index=q_idx).reindex(range(Q0 - 1, Q1 + 1))
        R_ii = national_universe(v_ii, {"rest_of_M4": (out_ii, R_eff)}, phiN["phi_t_B"], q_idx)
        res["split_ii_ifdata"] = (v_ii, R_ii)
        for uname, (v, Rn) in res.items():
            Rn.insert(0, "time_id", qlabel(q_idx))
            Rn.to_csv(out / f"national_{uname}_E{k}.csv", index_label="q")
            written.append(out / f"national_{uname}_E{k}.csv")
        Rd = res["dep"][1]
        g["G1_PB_minus_DepAct_dep_max_abs_rel"] = float(((Rd["PB_none"] - Rd["DepAct_trunc"]) / Rd["DepAct_trunc"]).abs().max())
        g["N_identical_split_i_ii_max_abs_rel"] = float(((R_ii["N"] - R_i["N"]) / R_i["N"].abs()).abs().max())
        g["W_split_i_over_M4_plus_prepaid_max_abs_rel"] = float(((R_i["W"] - T["W_M4_plus_prepaid"]) / T["W_M4_plus_prepaid"]).abs().max())
        stamp(f"E{k}: gate G1 {g['G1_PB_minus_DepAct_dep_max_abs_rel']:.2e}; N(ii)=N(i) {g['N_identical_split_i_ii_max_abs_rel']:.2e}; "
              f"W(i)=M4+S5 {g['W_split_i_over_M4_plus_prepaid_max_abs_rel']:.2e}")
        # ---- the realized awake share by deposit type and year (panel cells, phi uncapped)
        c["year"] = c["q"] // 4
        aw = []
        for scope, sub in (("B+D", c), ("B", c[c["is_B"]]), ("D", c[~c["is_B"]])):
            for by in (["k", "year"], ["k"], ["year"], []):
                it = sub.groupby(by) if by else [((), sub)]
                for key, s_ in it:
                    key = key if isinstance(key, tuple) else (key,)
                    r = dict(zip(by, key))
                    aw.append({"routine": f"E{k}", "scope": scope, "k": r.get("k", "all"), "year": r.get("year", "all"),
                               "realized_awake_share": float(s_["A_none"].sum() / s_["C"].sum()),
                               "carry_weighted_one_minus_phi": float(((1 - s_["phi"]) * s_["C"]).sum() / s_["C"].sum()),
                               "realized_awake_share_cap_0.99": float(s_["A_0.99"].sum() / s_["C"].sum()),
                               "realized_awake_share_cap_0.999": float(s_["A_0.999"].sum() / s_["C"].sum()),
                               "carry_bn": float(s_["C"].sum() / 1e9)})
        pd.DataFrame(aw).to_csv(out / f"realized_awake_E{k}.csv", index=False)
        written.append(out / f"realized_awake_E{k}.csv")
        summary["routines"][f"E{k}"] = {"frame": finfo, "phi": pinfo, "gates": g}
        if "local" in steps:
            L, W = [], []
            for uname in ("dep", "split_i_agg", "split_i_agg_cu_outside", "split_ii_ifdata"):
                v, Rn = res[uname]
                rr, ww = local_universe(v, Rn, acc, uname, k)
                L += rr
                W += ww
                stamp(f"E{k}: local {uname} done")
            pd.DataFrame(L).to_csv(out / f"local_pools_E{k}.csv", index=False)
            pd.DataFrame(W).to_csv(out / f"local_w4_weights_E{k}.csv", index=False)
            written += [out / f"local_pools_E{k}.csv", out / f"local_w4_weights_E{k}.csv"]
        (out / "diag_new_money_summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    summary["seconds"] = round(time.time() - T0, 1)
    try:
        import psutil
        summary["peak_working_set_GB"] = round(psutil.Process().memory_info().peak_wset / 2 ** 30, 2)
    except Exception:
        pass
    (out / "diag_new_money_summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    written.append(out / "diag_new_money_summary.json")
    with open(out / "MD5SUMS.txt", "w", encoding="utf-8", newline="\n") as fh:
        for pth in written:
            fh.write(f"{md5_file(pth, lf=True)}  {pth.name}\n")
    stamp(f"done; {len(written)} files in {out}")


if __name__ == "__main__":
    main()

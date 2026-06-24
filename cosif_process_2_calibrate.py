## cosif_process_2_calibrate.py
# Author: Pedro Feijo de Moraes
# Objective: Build the CORRECTED k=4 (CDB / time-deposit) implicit rate per
#            (CodConglomeradoPrudencial x quarter), fixing the artifact in the
#            old blended/residual approach.
#
# Method (per-bank effective rates + precision-weighted shrinkage to segment):
#   1. From each conglomerate's 2025+ 10-digit COSIF leaves, estimate effective
#      repo / bills rates (as x CDI) and a per-bank de-bias factor k_i:
#        k_i = (true CDB rate from leaf 8113000009) / (strip-implied CDB rate),
#      where the strip uses the SAME residual formula applied pre-2025.
#   2. Pool to a PRUDENTIAL SEGMENT (S1-S5) profile (deposit-weighted over banks
#      in the segment with 2025 data); shrink each bank toward its segment with
#      w_i = n_i / (n_i + N0), N0 = 2 (n_i = # of the bank's valid 2025 CDB qtrs).
#      Banks with no 2025 data -> full segment profile; segment also missing ->
#      global fallback.
#   3. Apply:
#        pre-2025: rate_a4 = k_shrunk * strip_cdb_rate(repo_shrunk, bills_shrunk)
#        2025+   : rate_a4 = |8113000009| / lag(time-deposit stock)   (direct leaf)
#
# Inputs  : custos_implicitos_v2_foundation.csv      (cosif_process_1_extract)
#           PANEL_INTERMED/quarterly_macro_rates.csv  (macro; written by panel_3)
#           market_panel.csv  (segment per conglomerate; read-only)  -- with a
#           bank_chars_panel.csv fallback for the segment if market_panel absent.
#           IF-Data List  (CNPJ<->conglomerate merge_asof crosswalk)
# Output  : COSIF_PROCESSED/cosif_cdb_rate_corrected.csv
#             columns: CodConglomeradoPrudencial, AnoMes, rate_a4_corrected_qoq,
#                      rate_a4_corrected_ann, tier
#
# NOTE: macro *_qoq in quarterly_macro_rates.csv are stored in PERCENT; this
# module converts them to fractions internally.  rate_a4_corrected_qoq is a
# fraction (e.g. 0.025 = 2.5% per quarter), matching panel_3's internal qoq scale
# BEFORE its final *100 conversion.
## ---------------------------------------------------------------------------

import os
import sys
import glob
import logging

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from utils import paths  # noqa: E402

PROCESSED = str(paths.PROCESSED)
COSIF_PROC = os.path.join(PROCESSED, "COSIF_PROCESSED")
PANEL_INTERMED = os.path.join(PROCESSED, "PANEL_INTERMED")
LIST_DIR = str(paths.IF_DATA_LIST)

FOUNDATION = os.path.join(COSIF_PROC, "custos_implicitos_v2_foundation.csv")
MACRO = os.path.join(PANEL_INTERMED, "quarterly_macro_rates.csv")
MARKET_PANEL = os.path.join(PROCESSED, "market_panel.csv")
BANK_CHARS = os.path.join(PROCESSED, "bank_chars_panel.csv")
OUT_CORRECTED = os.path.join(COSIF_PROC, "cosif_cdb_rate_corrected.csv")

N0 = 2                  # precision-shrinkage prior count
RATE_Q_CAP = 0.5        # economically implausible quarterly-rate guard

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("cosif_calibrate")

STOCK_COLS = ["stk_total", "stk_demand", "stk_savings", "stk_interb",
              "stk_time", "stk_repos", "stk_bills", "stk_prepaid"]
EXPENSE_COLS = ["exp_lump", "exp_savings", "exp_interb", "exp_cdb",
                "exp_repos", "exp_bills", "exp_prepaid"]


def annualize(q):
    return (1.0 + q) ** 4 - 1.0


def wmean(s, w):
    s = np.asarray(s, dtype=float)
    w = np.asarray(w, dtype=float)
    m = np.isfinite(s) & np.isfinite(w) & (w > 0)
    return float(np.average(s[m], weights=w[m])) if m.any() else np.nan


def load_macro_fraction():
    m = pd.read_csv(MACRO)
    for c in ["cdi_qoq", "selic_qoq", "savings_rate_qoq"]:
        if c in m.columns:
            m[c] = m[c] / 100.0
    return m


def build_cnpj_to_congl():
    """CNPJ(=CodInst for Td=='I') -> CodConglomeradoPrudencial, time-varying."""
    files = sorted(glob.glob(os.path.join(LIST_DIR, "IF_DATA_List_*.csv")))
    if not files:
        raise FileNotFoundError(f"No IF-Data List files in {LIST_DIR}")
    frames = []
    for fp in files:
        try:
            df = pd.read_csv(fp, dtype=str, encoding="latin-1")
        except Exception:
            df = pd.read_csv(fp, dtype=str, encoding="utf-8")
        df.columns = [c.strip() for c in df.columns]
        mask = (df["Td"] == "I") & (df["Situacao"] == "A")
        sub = df.loc[mask, ["CodInst", "Data", "CodConglomeradoPrudencial"]].copy()
        standalone = (sub["CodConglomeradoPrudencial"].isna()
                      | (sub["CodConglomeradoPrudencial"] == "null"))
        sub.loc[standalone, "CodConglomeradoPrudencial"] = sub.loc[standalone, "CodInst"]
        sub.rename(columns={"CodInst": "CNPJ", "Data": "AnoMes_list"}, inplace=True)
        frames.append(sub)
    m = pd.concat(frames, ignore_index=True)
    m["AnoMes_list"] = pd.to_numeric(m["AnoMes_list"], errors="coerce").astype("Int64")
    m["CNPJ"] = m["CNPJ"].str.strip().str.zfill(8)
    return m.drop_duplicates(subset=["CNPJ", "AnoMes_list"])


def map_foundation_to_congl(found, df_map):
    """As-of merge: each (CNPJ, quarter) -> nearest List snapshot (backward,
    then earliest-forward for pre-2016 quarters)."""
    f = found.sort_values(["AnoMes", "CNPJ"]).copy()
    dm = df_map.dropna(subset=["AnoMes_list"]).copy()
    dm["AnoMes_list"] = dm["AnoMes_list"].astype(int)
    dm = dm.sort_values(["AnoMes_list", "CNPJ"])
    back = pd.merge_asof(f, dm, left_on="AnoMes", right_on="AnoMes_list",
                         by="CNPJ", direction="backward")
    fwd = pd.merge_asof(f, dm, left_on="AnoMes", right_on="AnoMes_list",
                        by="CNPJ", direction="forward")
    merged = back.copy()
    fill = merged["CodConglomeradoPrudencial"].isna()
    merged.loc[fill, "CodConglomeradoPrudencial"] = fwd.loc[fill, "CodConglomeradoPrudencial"]
    return merged.dropna(subset=["CodConglomeradoPrudencial"])


def load_segment_map():
    """conglomerate -> prudential segment (S1-S5), latest known.  Prefer
    market_panel.csv; fall back to bank_chars_panel.csv; else empty."""
    for path, code_col, ym_cols in [
        (MARKET_PANEL, "CodConglomeradoPrudencial", ("year", "quarter")),
        (BANK_CHARS, "CodConglPrud", ("AnoMes",)),
    ]:
        if not os.path.exists(path):
            continue
        try:
            cols = pd.read_csv(path, nrows=0).columns
            if "segment" not in cols or code_col not in cols:
                continue
            use = [code_col, "segment"] + [c for c in ym_cols if c in cols]
            df = pd.read_csv(path, usecols=use, dtype={code_col: str})
            df = df.dropna(subset=["segment"])
            if len(ym_cols) == 2 and all(c in df.columns for c in ym_cols):
                df = df.sort_values(list(ym_cols))
            elif ym_cols[0] in df.columns:
                df = df.sort_values(ym_cols[0])
            seg = (df.drop_duplicates(code_col, keep="last")
                   [[code_col, "segment"]]
                   .rename(columns={code_col: "CodConglomeradoPrudencial"}))
            log.info("Segment map: %d conglomerates from %s",
                     len(seg), os.path.basename(path))
            return seg
        except Exception as e:   # noqa: BLE001
            log.warning("Could not read segment from %s: %s", path, e)
    log.warning("No segment source found -- all banks use the global profile.")
    return pd.DataFrame(columns=["CodConglomeradoPrudencial", "segment"])


def strip_cdb_expense(d, repo_ratio, bills_ratio):
    """|lump| - repo - bills - savings - interbank.  Same formula used pre-2025
    and to derive k_i.  repo_ratio/bills_ratio scalar or per-row Series."""
    cdi = d["cdi_qoq"]
    deposit_exp = (
        d["exp_lump"].abs()
        - d["stk_repos_lag"].fillna(0) * repo_ratio * cdi
        - d["stk_bills_lag"].fillna(0) * bills_ratio * cdi
    )
    cdb_exp = (
        deposit_exp
        - d["savings_rate_qoq"].fillna(0) * d["stk_savings_lag"].fillna(0)
        - cdi.fillna(0) * d["stk_interb_lag"].fillna(0)
    )
    return cdb_exp.clip(lower=0)


def build_congl_panel(found, macro):
    df_map = build_cnpj_to_congl()
    merged = map_foundation_to_congl(found, df_map)
    agg_cols = [c for c in STOCK_COLS + EXPENSE_COLS if c in merged.columns]
    congl = (merged.groupby(["CodConglomeradoPrudencial", "AnoMes"])[agg_cols]
             .sum(min_count=1).reset_index())
    congl["year"] = congl["AnoMes"] // 100
    congl = congl.sort_values(["CodConglomeradoPrudencial", "AnoMes"])
    for c in STOCK_COLS:
        if c in congl.columns:
            congl[c + "_lag"] = congl.groupby("CodConglomeradoPrudencial")[c].shift(1)
    return congl.merge(macro[["AnoMes", "cdi_qoq", "savings_rate_qoq"]],
                       on="AnoMes", how="left")


def estimate_perbank(post):
    """Per-conglomerate 2025+ repo/bills ratios, true & strip CDB rates, k_i, n_i."""
    def ratio_rows(d, exp_col, stk_lag_col):
        x = d.dropna(subset=[exp_col, stk_lag_col, "cdi_qoq"]).copy()
        x = x[(x[stk_lag_col] > 0) & (x["cdi_qoq"] > 0)]
        x["r"] = (x[exp_col].abs() / x[stk_lag_col]) / x["cdi_qoq"]
        return x[(x["r"] >= 0) & (x["r"] < 5)]

    rows = []
    for code, g in post.groupby("CodConglomeradoPrudencial"):
        seg = g["segment"].dropna().iloc[0] if g["segment"].notna().any() else None
        rp = ratio_rows(g, "exp_repos", "stk_repos_lag")
        repo_i = wmean(rp["r"], rp["stk_repos_lag"]) if len(rp) else np.nan
        bl = ratio_rows(g, "exp_bills", "stk_bills_lag")
        bills_i = wmean(bl["r"], bl["stk_bills_lag"]) if len(bl) else np.nan
        cdbq = g.dropna(subset=["exp_cdb", "stk_time_lag"]).copy()
        cdbq = cdbq[cdbq["stk_time_lag"] > 0]
        n_i = len(cdbq)
        if n_i:
            cdbq["true_x"] = (cdbq["exp_cdb"].abs() / cdbq["stk_time_lag"]) / cdbq["cdi_qoq"]
            true_x = wmean(cdbq["true_x"], cdbq["stk_time_lag"])
        else:
            true_x = np.nan
        if n_i and pd.notna(repo_i) and pd.notna(bills_i):
            se = strip_cdb_expense(cdbq, repo_i, bills_i)
            cdbq["strip_x"] = (se / cdbq["stk_time_lag"]) / cdbq["cdi_qoq"]
            strip_x = wmean(cdbq["strip_x"], cdbq["stk_time_lag"])
        else:
            strip_x = np.nan
        k_i = (true_x / strip_x) if (pd.notna(strip_x) and strip_x > 0) else np.nan
        rows.append({"CodConglomeradoPrudencial": code, "segment": seg,
                     "repo_ratio_i": repo_i, "bills_ratio_i": bills_i,
                     "k_i": k_i, "n_i": n_i,
                     "dep_w": cdbq["stk_time_lag"].sum() if n_i else 0.0})
    return pd.DataFrame(rows)


def main():
    if not os.path.exists(FOUNDATION):
        log.error("Foundation not found: %s -- run cosif_process_1_extract first.",
                  FOUNDATION)
        sys.exit(1)

    macro = load_macro_fraction()
    found = pd.read_csv(FOUNDATION, dtype={"CNPJ": str})
    found["CNPJ"] = found["CNPJ"].str.zfill(8)

    congl = build_congl_panel(found, macro)
    seg_map = load_segment_map()
    congl = congl.merge(seg_map, on="CodConglomeradoPrudencial", how="left")

    post = congl[congl["year"] >= 2025].copy()
    pb = estimate_perbank(post)

    # ---- segment profiles (deposit-weighted over banks with 2025 data) ----
    def seg_profile(g):
        gg = g[g["n_i"] > 0]
        return pd.Series({
            "repo_ratio_seg": wmean(gg["repo_ratio_i"], gg["dep_w"]),
            "bills_ratio_seg": wmean(gg["bills_ratio_i"], gg["dep_w"]),
            "k_seg": wmean(gg["k_i"], gg["dep_w"]),
        })
    seg_prof = (pb.dropna(subset=["segment"]).groupby("segment")
                .apply(seg_profile, include_groups=False))
    gall = pb[pb["n_i"] > 0]
    glob = {"repo": wmean(gall["repo_ratio_i"], gall["dep_w"]),
            "bills": wmean(gall["bills_ratio_i"], gall["dep_w"]),
            "k": wmean(gall["k_i"], gall["dep_w"])}
    log.info("Segment profiles:\n%s", seg_prof.to_string())
    log.info("Global fallback: repo=%.3f bills=%.3f k=%.3f",
             glob["repo"], glob["bills"], glob["k"])

    def seg_val(seg, col, fallback):
        if seg in seg_prof.index and pd.notna(seg_prof.loc[seg, col]):
            return seg_prof.loc[seg, col]
        return fallback

    # ---- shrinkage (w_i = n_i/(n_i+N0)); per-component fallback to segment ----
    pb["repo_seg"] = pb["segment"].apply(lambda s: seg_val(s, "repo_ratio_seg", glob["repo"]))
    pb["bills_seg"] = pb["segment"].apply(lambda s: seg_val(s, "bills_ratio_seg", glob["bills"]))
    pb["k_segt"] = pb["segment"].apply(lambda s: seg_val(s, "k_seg", glob["k"]))
    pb["w_i"] = pb["n_i"] / (pb["n_i"] + N0)
    for own, segc, out in [("repo_ratio_i", "repo_seg", "repo_shrunk"),
                           ("bills_ratio_i", "bills_seg", "bills_shrunk"),
                           ("k_i", "k_segt", "k_shrunk")]:
        w = pb["w_i"].where(pb[own].notna(), 0.0)
        pb[out] = w * pb[own].fillna(0.0) + (1 - w) * pb[segc]

    # ---- assign params to EVERY conglomerate (incl. those absent in 2025) ----
    all_codes = congl[["CodConglomeradoPrudencial"]].drop_duplicates().merge(
        seg_map, on="CodConglomeradoPrudencial", how="left")
    params = all_codes.merge(
        pb[["CodConglomeradoPrudencial", "repo_shrunk", "bills_shrunk",
            "k_shrunk", "n_i"]], on="CodConglomeradoPrudencial", how="left")
    params["n_i"] = params["n_i"].fillna(0)
    params["repo_shrunk"] = params.apply(
        lambda r: r["repo_shrunk"] if pd.notna(r["repo_shrunk"])
        else seg_val(r["segment"], "repo_ratio_seg", glob["repo"]), axis=1)
    params["bills_shrunk"] = params.apply(
        lambda r: r["bills_shrunk"] if pd.notna(r["bills_shrunk"])
        else seg_val(r["segment"], "bills_ratio_seg", glob["bills"]), axis=1)
    params["k_shrunk"] = params.apply(
        lambda r: r["k_shrunk"] if pd.notna(r["k_shrunk"])
        else seg_val(r["segment"], "k_seg", glob["k"]), axis=1)

    def tier(r):
        if r["n_i"] >= 2:
            return "perbank"
        if r["n_i"] >= 1:
            return "perbank_weak"
        s = r["segment"]
        if s in seg_prof.index and pd.notna(seg_prof.loc[s, "k_seg"]):
            return "segment"
        return "global"
    params["tier"] = params.apply(tier, axis=1)

    congl = congl.merge(
        params[["CodConglomeradoPrudencial", "repo_shrunk", "bills_shrunk",
                "k_shrunk", "tier"]], on="CodConglomeradoPrudencial", how="left")

    # ---- apply: pre-2025 strip*k ; 2025+ direct leaf ----
    is_post = congl["year"] >= 2025
    strip_exp = strip_cdb_expense(congl, congl["repo_shrunk"], congl["bills_shrunk"])
    rate_pre = (strip_exp / congl["stk_time_lag"]).clip(lower=0) * congl["k_shrunk"]
    rate_post = (congl["exp_cdb"].abs() / congl["stk_time_lag"]).clip(lower=0)
    congl["rate_a4_corrected_qoq"] = np.where(is_post, rate_post, rate_pre)
    bad = ~np.isfinite(congl["rate_a4_corrected_qoq"])
    congl.loc[bad, "rate_a4_corrected_qoq"] = np.nan
    congl.loc[congl["rate_a4_corrected_qoq"] > RATE_Q_CAP, "rate_a4_corrected_qoq"] = np.nan
    congl["rate_a4_corrected_ann"] = annualize(congl["rate_a4_corrected_qoq"])

    out = congl[["CodConglomeradoPrudencial", "AnoMes",
                 "rate_a4_corrected_qoq", "rate_a4_corrected_ann", "tier"]].copy()
    out = out.dropna(subset=["rate_a4_corrected_qoq"])
    out.to_csv(OUT_CORRECTED, index=False)
    log.info("Wrote corrected CDB rate: %s  (%d rows, %d conglomerates, %d quarters)",
             OUT_CORRECTED, len(out), out["CodConglomeradoPrudencial"].nunique(),
             out["AnoMes"].nunique())


if __name__ == "__main__":
    main()

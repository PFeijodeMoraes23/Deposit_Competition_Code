## diag_cosif_k4_validate_perbank.py
# Author: Pedro Feijo de Moraes (k=4 fix, v2 = per-bank + segment shrinkage)
# Objective: Replace the v1 UNIFORM repo/bills rates and UNIFORM strip de-bias
#            factor with PER-BANK effective rates and a per-bank de-bias factor
#            k_i, each precision-weighted-SHRUNK toward the bank's PRUDENTIAL
#            SEGMENT (S1-S5) group profile.  Re-validate vs OLD (market_panel)
#            and vs v1-uniform.
#
# "Bank" here = prudential conglomerate (CodConglomeradoPrudencial): the strip
# is applied to conglomerate-aggregated COSIF stocks/expenses (foundation summed
# over member CNPJs), and OLD rate_a4 is conglomerate-level, so per-bank
# estimates are estimated at the conglomerate level too -- consistent units.
#
# Reuses, unchanged:
#   * custos_implicitos_v2_foundation.csv  (from panel_cosif_extract.py)
#   * the CNPJ<->conglomerate merge_asof mapping  (imported from v1 module)
#   * the macro fraction loader and the 2025+ leaf column scheme
#
# GUARDRAILS: NEW files only; does NOT modify panel_deposit_rates.py / panel_market.py,
# does NOT overwrite canonical custos_implicitos_*/market_panel.csv, does NOT
# rebuild market_panel or demand parquets.  PAUSE after validating.

import os
import sys
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from utils import paths
# Reuse the v1 building blocks verbatim.
from diag_cosif_k4_validate import (
    annualize, load_macro_fraction, build_cnpj_to_congl,
    map_foundation_to_congl, STOCK_COLS, EXPENSE_COLS, FOUNDATION,
)

PROCESSED = str(paths.PROCESSED)
COSIF_PROC = os.path.join(PROCESSED, "COSIF_PROCESSED")
CF_FOUNDATION = os.path.join(COSIF_PROC, "CF_FOUNDATION")
os.makedirs(CF_FOUNDATION, exist_ok=True)

MARKET_PANEL = os.path.join(PROCESSED, "market_panel.csv")
MACRO = os.path.join(PROCESSED, "PANEL_INTERMED", "quarterly_macro_rates.csv")

FIG_OUT = os.path.join(CF_FOUNDATION, "cdb_rate_fix_v2_perbank_top10.png")
TABLE_OUT = os.path.join(CF_FOUNDATION, "cdb_rate_fix_v2_perbank_top10.csv")
NEWRATE_OUT = os.path.join(CF_FOUNDATION, "cosif_rate_a4_v2_perbank_by_congl_quarter.csv")
PROFILE_OUT = os.path.join(CF_FOUNDATION, "segment_profiles_v2.csv")
PERBANK_OUT = os.path.join(CF_FOUNDATION, "perbank_estimates_v2.csv")

N0_DEFAULT = 2          # shrinkage prior count
N0_SENSITIVITY = [1, 2, 4]
RATE_Q_CAP = 0.5        # economically implausible quarterly rate guard


def wmean(s, w):
    m = pd.Series(s).notna() & pd.Series(w).notna() & (pd.Series(w) > 0)
    if not m.any():
        return np.nan
    return float(np.average(np.asarray(s)[m.values], weights=np.asarray(w)[m.values]))


# ---------------------------------------------------------------------------
# Build conglomerate x quarter panel (reuse v1 mapping + aggregation).
# ---------------------------------------------------------------------------
def build_congl_panel(macro):
    found = pd.read_csv(FOUNDATION, dtype={"CNPJ": str})
    found["CNPJ"] = found["CNPJ"].str.zfill(8)
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
    congl = congl.merge(macro[["AnoMes", "cdi_qoq", "savings_rate_qoq"]],
                        on="AnoMes", how="left")
    return congl


# ---------------------------------------------------------------------------
# Strip-implied CDB expense (the SAME formula used pre-2025 and for k_i).
# ---------------------------------------------------------------------------
def strip_cdb_expense(d, repo_ratio, bills_ratio):
    """|lump| - repo - bills - savings - interbank, per row.  repo_ratio /
    bills_ratio may be scalars or per-row Series aligned to d."""
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


def main():
    macro = load_macro_fraction()
    congl = build_congl_panel(macro)

    # ===================================================================
    # Segment map: conglomerate -> prudential segment (latest known).
    # ===================================================================
    mp_all = pd.read_csv(
        MARKET_PANEL,
        usecols=["CodConglomeradoPrudencial", "year", "quarter", "segment",
                 "dep_a4", "rate_a4", "risk_free_ann", "NomeInstituicao"],
        dtype={"CodConglomeradoPrudencial": str})
    seg_map = (mp_all.dropna(subset=["segment"])
               .sort_values(["year", "quarter"])
               .drop_duplicates("CodConglomeradoPrudencial", keep="last")
               [["CodConglomeradoPrudencial", "segment"]])
    congl = congl.merge(seg_map, on="CodConglomeradoPrudencial", how="left")

    post = congl[congl["year"] >= 2025].copy()

    # ===================================================================
    # 1) PER-BANK (conglomerate) estimates from 2025+ leaves.
    # ===================================================================
    def ratio_rows(d, exp_col, stk_lag_col):
        x = d.dropna(subset=[exp_col, stk_lag_col, "cdi_qoq"]).copy()
        x = x[(x[stk_lag_col] > 0) & (x["cdi_qoq"] > 0)]
        x["r"] = (x[exp_col].abs() / x[stk_lag_col]) / x["cdi_qoq"]
        x = x[(x["r"] >= 0) & (x["r"] < 5)]   # ratio-to-CDI sanity
        return x

    rows = []
    for code, g in post.groupby("CodConglomeradoPrudencial"):
        seg = g["segment"].dropna().iloc[0] if g["segment"].notna().any() else None

        rp = ratio_rows(g, "exp_repos", "stk_repos_lag")
        repo_i = wmean(rp["r"], rp["stk_repos_lag"]) if len(rp) else np.nan

        bl = ratio_rows(g, "exp_bills", "stk_bills_lag")
        bills_i = wmean(bl["r"], bl["stk_bills_lag"]) if len(bl) else np.nan

        # precision n_i: quarters with valid 8113 leaf AND non-trivial time stock
        cdbq = g.dropna(subset=["exp_cdb", "stk_time_lag"]).copy()
        cdbq = cdbq[cdbq["stk_time_lag"] > 0]
        n_i = len(cdbq)

        # true CDB-rate-as-x-CDI (deposit-weighted across the bank's 2025 qtrs)
        if n_i:
            cdbq["true_x"] = (cdbq["exp_cdb"].abs() / cdbq["stk_time_lag"]) / cdbq["cdi_qoq"]
            true_x_i = wmean(cdbq["true_x"], cdbq["stk_time_lag"])
        else:
            true_x_i = np.nan

        # strip-implied CDB rate (x CDI) using the bank's OWN stocks + own ratios.
        # Use the bank's own repo/bills ratio if available, else NaN -> k_i NaN.
        if n_i and pd.notna(repo_i) and pd.notna(bills_i):
            d = cdbq.copy()
            strip_exp = strip_cdb_expense(d, repo_i, bills_i)
            d["strip_x"] = (strip_exp / d["stk_time_lag"]) / d["cdi_qoq"]
            strip_x_i = wmean(d["strip_x"], d["stk_time_lag"])
        else:
            strip_x_i = np.nan

        k_i = (true_x_i / strip_x_i) if (pd.notna(strip_x_i) and strip_x_i > 0) else np.nan

        rows.append({
            "CodConglomeradoPrudencial": code, "segment": seg,
            "repo_ratio_i": repo_i, "bills_ratio_i": bills_i,
            "true_x_i": true_x_i, "strip_x_i": strip_x_i, "k_i": k_i,
            "n_i": n_i,
            "dep_w": cdbq["stk_time_lag"].sum() if n_i else 0.0,
        })
    pb = pd.DataFrame(rows)

    # ===================================================================
    # 2) SEGMENT PROFILES: deposit-weighted across banks with 2025 data.
    # ===================================================================
    def seg_profile(g):
        gg = g[g["n_i"] > 0]
        return pd.Series({
            "repo_ratio_seg": wmean(gg["repo_ratio_i"], gg["dep_w"]),
            "bills_ratio_seg": wmean(gg["bills_ratio_i"], gg["dep_w"]),
            "k_seg": wmean(gg["k_i"], gg["dep_w"]),
            "n_banks": (gg["k_i"].notna()).sum(),
            "k_median": gg["k_i"].median(),
            "k_q25": gg["k_i"].quantile(0.25),
            "k_q75": gg["k_i"].quantile(0.75),
        })
    seg_prof = pb.dropna(subset=["segment"]).groupby("segment").apply(
        seg_profile, include_groups=False).reset_index()

    # Global fallback profile (deposit-weighted over all 2025 banks).
    gall = pb[pb["n_i"] > 0]
    global_prof = {
        "repo_ratio": wmean(gall["repo_ratio_i"], gall["dep_w"]),
        "bills_ratio": wmean(gall["bills_ratio_i"], gall["dep_w"]),
        "k": wmean(gall["k_i"], gall["dep_w"]),
    }
    seg_prof.to_csv(PROFILE_OUT, index=False)

    print("=" * 78)
    print("SEGMENT PROFILES (deposit-weighted over banks with 2025 data)")
    print("=" * 78)
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(seg_prof.to_string(index=False))
    print(f"\nGLOBAL fallback profile: repo={global_prof['repo_ratio']:.3f}xCDI  "
          f"bills={global_prof['bills_ratio']:.3f}xCDI  k={global_prof['k']:.3f}")

    print("\n" + "-" * 78)
    print("DISTRIBUTION of k_i by segment (count / median / IQR)")
    print("-" * 78)
    print(seg_prof[["segment", "n_banks", "k_median", "k_q25", "k_q75",
                    "k_seg"]].to_string(index=False))

    # ===================================================================
    # 3) SHRINKAGE: per-bank shrunk toward own segment (fallback -> global).
    # ===================================================================
    seg_lookup = seg_prof.set_index("segment")

    def seg_val(seg, col, glob):
        if seg in seg_lookup.index and pd.notna(seg_lookup.loc[seg, col]):
            return seg_lookup.loc[seg, col]
        return glob

    def compute_shrunk(pb_in, n0):
        out = pb_in.copy()
        # segment targets (with global fallback)
        out["repo_seg"] = out["segment"].apply(
            lambda s: seg_val(s, "repo_ratio_seg", global_prof["repo_ratio"]))
        out["bills_seg"] = out["segment"].apply(
            lambda s: seg_val(s, "bills_ratio_seg", global_prof["bills_ratio"]))
        out["k_segt"] = out["segment"].apply(
            lambda s: seg_val(s, "k_seg", global_prof["k"]))
        out["w_i"] = out["n_i"] / (out["n_i"] + n0)
        # banks with no own estimate -> rely fully on segment (w forced 0)
        for own, seg, col in [("repo_ratio_i", "repo_seg", "repo_shrunk"),
                              ("bills_ratio_i", "bills_seg", "bills_shrunk"),
                              ("k_i", "k_segt", "k_shrunk")]:
            w = out["w_i"].where(out[own].notna(), 0.0)
            out[col] = w * out[own].fillna(0.0) + (1 - w) * out[seg]
        return out

    pb_shrunk = compute_shrunk(pb, N0_DEFAULT)
    pb_shrunk.to_csv(PERBANK_OUT, index=False)
    print(f"\nWrote per-bank estimates: {PERBANK_OUT}")

    # ===================================================================
    # 4) APPLY to the full conglomerate panel.
    #    Banks with NO 2025 row at all (absent from pb) -> full segment / global.
    # ===================================================================
    all_codes = congl[["CodConglomeradoPrudencial"]].drop_duplicates()
    all_codes = all_codes.merge(seg_map, on="CodConglomeradoPrudencial", how="left")
    params = all_codes.merge(
        pb_shrunk[["CodConglomeradoPrudencial", "repo_shrunk", "bills_shrunk",
                   "k_shrunk", "n_i"]],
        on="CodConglomeradoPrudencial", how="left")
    # banks absent in 2025 (no pb row): n_i=0, params NaN -> fill from segment.
    miss = params["k_shrunk"].isna()
    params.loc[miss, "n_i"] = 0
    params["repo_shrunk"] = params.apply(
        lambda r: r["repo_shrunk"] if pd.notna(r["repo_shrunk"])
        else seg_val(r["segment"], "repo_ratio_seg", global_prof["repo_ratio"]),
        axis=1)
    params["bills_shrunk"] = params.apply(
        lambda r: r["bills_shrunk"] if pd.notna(r["bills_shrunk"])
        else seg_val(r["segment"], "bills_ratio_seg", global_prof["bills_ratio"]),
        axis=1)
    params["k_shrunk"] = params.apply(
        lambda r: r["k_shrunk"] if pd.notna(r["k_shrunk"])
        else seg_val(r["segment"], "k_seg", global_prof["k"]),
        axis=1)
    # tier label for coverage
    def tier(r):
        if r["n_i"] >= 2:
            return "perbank"        # per-bank dominated (w_i>=0.5)
        if r["n_i"] >= 1:
            return "perbank_weak"   # has some 2025 data but shrinkage-dominated
        # no 2025 data: did its segment have a profile?
        s = r["segment"]
        if s in seg_lookup.index and pd.notna(seg_lookup.loc[s, "k_seg"]):
            return "segment"
        return "global"
    params["tier"] = params.apply(tier, axis=1)

    congl = congl.merge(
        params[["CodConglomeradoPrudencial", "repo_shrunk", "bills_shrunk",
                "k_shrunk", "tier"]],
        on="CodConglomeradoPrudencial", how="left")

    cdi = congl["cdi_qoq"]
    is_post = congl["year"] >= 2025
    strip_exp_pre = strip_cdb_expense(congl, congl["repo_shrunk"], congl["bills_shrunk"])
    rate_pre = (strip_exp_pre / congl["stk_time_lag"]).clip(lower=0) * congl["k_shrunk"]
    rate_post = (congl["exp_cdb"].abs() / congl["stk_time_lag"]).clip(lower=0)
    congl["rate_a4_new_qoq"] = np.where(is_post, rate_post, rate_pre)
    congl.loc[~np.isfinite(congl["rate_a4_new_qoq"]), "rate_a4_new_qoq"] = np.nan
    congl.loc[congl["rate_a4_new_qoq"] > RATE_Q_CAP, "rate_a4_new_qoq"] = np.nan
    congl["rate_a4_new_ann"] = annualize(congl["rate_a4_new_qoq"])
    congl.to_csv(NEWRATE_OUT, index=False)
    print(f"Wrote per-congl v2 rate: {NEWRATE_OUT}")

    # ===================================================================
    # 5) COVERAGE: share of PRE-2025 total CDB deposit stock by tier.
    # ===================================================================
    pre = congl[congl["year"] < 2025].copy()
    cov = (pre.groupby("tier")["stk_time"].sum())
    cov_share = (cov / cov.sum()).reindex(
        ["perbank", "perbank_weak", "segment", "global"]).fillna(0.0)
    print("\n" + "=" * 78)
    print("COVERAGE: share of pre-2025 total CDB deposit STOCK by tier")
    print("=" * 78)
    for t, s in cov_share.items():
        n_banks = params[params["tier"] == t]["CodConglomeradoPrudencial"].nunique()
        print(f"  {t:14s}: {s*100:6.2f}%   ({n_banks} conglomerates)")
    print(f"  (a) per-bank-dominated (perbank+weak): "
          f"{(cov_share['perbank']+cov_share['perbank_weak'])*100:.2f}%")
    print(f"  (b) segment fallback                 : {cov_share['segment']*100:.2f}%")
    print(f"  (c) global fallback / pre-2025-exit  : {cov_share['global']*100:.2f}%")

    # ===================================================================
    # 6) Load v1 (uniform) per-congl rate for the 3-way comparison.
    # ===================================================================
    v1_path = os.path.join(CF_FOUNDATION, "cosif_rate_a4_v2_by_congl_quarter.csv")
    v1 = pd.read_csv(v1_path, dtype={"CodConglomeradoPrudencial": str})
    v1 = v1[["CodConglomeradoPrudencial", "AnoMes", "rate_a4_new_ann"]].rename(
        columns={"rate_a4_new_ann": "v1_rate_ann"})

    # ===================================================================
    # 7) Top-10 + before/after table & figure.
    # ===================================================================
    mp = mp_all.copy()
    mp["AnoMes"] = mp["year"] * 100 + mp["quarter"] * 3
    top = (mp.groupby("CodConglomeradoPrudencial")
           .agg(dep=("dep_a4", "sum"), name=("NomeInstituicao", "first"))
           .reset_index().sort_values("dep", ascending=False).head(10))
    top_codes = top["CodConglomeradoPrudencial"].tolist()

    mp_top = mp[mp["CodConglomeradoPrudencial"].isin(top_codes)].copy()
    old = (mp_top.groupby(["CodConglomeradoPrudencial", "AnoMes"])
           .apply(lambda g: pd.Series({
               "old_rate_ann": wmean(g["rate_a4"], g["dep_a4"].clip(lower=0) + 1e-9),
               "rf_ann": wmean(g["risk_free_ann"], g["dep_a4"].clip(lower=0) + 1e-9),
               "dep_a4": g["dep_a4"].sum(),
           }), include_groups=False).reset_index())

    new = congl[congl["CodConglomeradoPrudencial"].isin(top_codes)][
        ["CodConglomeradoPrudencial", "AnoMes", "rate_a4_new_ann"]].rename(
        columns={"rate_a4_new_ann": "v2_rate_ann"})

    macro2 = macro.copy()
    macro2["cdi_ann"] = annualize(macro2["cdi_qoq"])

    comp = (old.merge(new, on=["CodConglomeradoPrudencial", "AnoMes"], how="left")
            .merge(v1, on=["CodConglomeradoPrudencial", "AnoMes"], how="left"))
    comp["year"] = comp["AnoMes"] // 100
    comp = comp.merge(macro2[["AnoMes", "cdi_ann"]], on="AnoMes", how="left")
    comp = comp[(comp["year"] >= 2013) & (comp["year"] <= 2025)]

    annual = (comp.groupby("year").apply(lambda g: pd.Series({
        "old_rate_a4": wmean(g["old_rate_ann"], g["dep_a4"]),
        "v1_rate_a4": wmean(g["v1_rate_ann"], g["dep_a4"]),
        "v2_rate_a4": wmean(g["v2_rate_ann"], g["dep_a4"]),
        "cdi": g["cdi_ann"].mean(),
        "rf": wmean(g["rf_ann"], g["dep_a4"]),
    }), include_groups=False).reset_index())
    annual["band_lo"] = 0.90 * annual["cdi"]
    annual["band_hi"] = 1.15 * annual["cdi"]
    annual["old_x"] = annual["old_rate_a4"] / annual["cdi"]
    annual["v1_x"] = annual["v1_rate_a4"] / annual["cdi"]
    annual["v2_x"] = annual["v2_rate_a4"] / annual["cdi"]
    annual["v2_in_band"] = ((annual["v2_rate_a4"] >= annual["band_lo"]) &
                            (annual["v2_rate_a4"] <= annual["band_hi"]))
    annual.to_csv(TABLE_OUT, index=False)

    print("\n" + "=" * 78)
    print("ANNUAL DEPOSIT-WEIGHTED CDB RATE (x CDI): OLD vs v1-UNIFORM vs v2-PERBANK")
    print("=" * 78)
    print(annual[["year", "old_x", "v1_x", "v2_x", "cdi", "v2_in_band"]]
          .to_string(index=False))
    print(f"\nWrote table: {TABLE_OUT}")

    # Success metrics over 2013-2024 (exclude 2025 which is direct-leaf).
    def near(col, lo=0.85, hi=1.20):
        w = annual[(annual["year"] >= 2013) & (annual["year"] <= 2024)]
        return ((w[col] >= lo) & (w[col] <= hi)).sum(), len(w)
    v1n, tot = near("v1_x")
    v2n, _ = near("v2_x")
    oldn, _ = near("old_x")
    v2b = annual[(annual["year"] >= 2013) & (annual["year"] <= 2024)]["v2_in_band"].sum()
    print(f"\nNEAR band [0.85,1.20]xCDI over 2013-2024:  "
          f"OLD {oldn}/{tot}  |  v1 {v1n}/{tot}  |  v2 {v2n}/{tot}")
    print(f"v2 STRICTLY in 90-115% band over 2013-2024: {v2b}/{tot}")

    # ===================================================================
    # 8) n0 sensitivity (report value-weighted near-band count).
    # ===================================================================
    print("\n" + "-" * 78)
    print("n0 SENSITIVITY (2013-2024 top-10 value-weighted)")
    print("-" * 78)
    for n0 in N0_SENSITIVITY:
        pbs = compute_shrunk(pb, n0)
        prm = all_codes.merge(
            pbs[["CodConglomeradoPrudencial", "repo_shrunk", "bills_shrunk",
                 "k_shrunk"]], on="CodConglomeradoPrudencial", how="left")
        prm["repo_shrunk"] = prm.apply(
            lambda r: r["repo_shrunk"] if pd.notna(r["repo_shrunk"])
            else seg_val(r["segment"], "repo_ratio_seg", global_prof["repo_ratio"]), axis=1)
        prm["bills_shrunk"] = prm.apply(
            lambda r: r["bills_shrunk"] if pd.notna(r["bills_shrunk"])
            else seg_val(r["segment"], "bills_ratio_seg", global_prof["bills_ratio"]), axis=1)
        prm["k_shrunk"] = prm.apply(
            lambda r: r["k_shrunk"] if pd.notna(r["k_shrunk"])
            else seg_val(r["segment"], "k_seg", global_prof["k"]), axis=1)
        c2 = congl.drop(columns=["repo_shrunk", "bills_shrunk", "k_shrunk"]).merge(
            prm[["CodConglomeradoPrudencial", "repo_shrunk", "bills_shrunk", "k_shrunk"]],
            on="CodConglomeradoPrudencial", how="left")
        se = strip_cdb_expense(c2, c2["repo_shrunk"], c2["bills_shrunk"])
        rp = (se / c2["stk_time_lag"]).clip(lower=0) * c2["k_shrunk"]
        rpost = (c2["exp_cdb"].abs() / c2["stk_time_lag"]).clip(lower=0)
        c2["rq"] = np.where(c2["year"] >= 2025, rpost, rp)
        c2.loc[~np.isfinite(c2["rq"]), "rq"] = np.nan
        c2.loc[c2["rq"] > RATE_Q_CAP, "rq"] = np.nan
        c2["rann"] = annualize(c2["rq"])
        nn = c2[c2["CodConglomeradoPrudencial"].isin(top_codes)][
            ["CodConglomeradoPrudencial", "AnoMes", "rann"]]
        cc = (old.merge(nn, on=["CodConglomeradoPrudencial", "AnoMes"], how="left"))
        cc["year"] = cc["AnoMes"] // 100
        cc = cc.merge(macro2[["AnoMes", "cdi_ann"]], on="AnoMes", how="left")
        cc = cc[(cc["year"] >= 2013) & (cc["year"] <= 2024)]
        ann = (cc.groupby("year").apply(lambda g: pd.Series({
            "r": wmean(g["rann"], g["dep_a4"]), "cdi": g["cdi_ann"].mean()}),
            include_groups=False).reset_index())
        ann["x"] = ann["r"] / ann["cdi"]
        nb = ((ann["x"] >= 0.85) & (ann["x"] <= 1.20)).sum()
        strict = ((ann["r"] >= 0.90 * ann["cdi"]) & (ann["r"] <= 1.15 * ann["cdi"])).sum()
        print(f"  n0={n0}: near-band {nb}/{len(ann)}  strict-band {strict}/{len(ann)}  "
              f"mean x-CDI={ann['x'].mean():.3f}")

    # ===================================================================
    # 9) Figure.
    # ===================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    yrs = annual["year"]
    ax.fill_between(yrs, 100 * annual["band_lo"], 100 * annual["band_hi"],
                    color="green", alpha=0.12, label="Plausible band (90-115% CDI)")
    ax.plot(yrs, 100 * annual["cdi"], "k--", lw=1.5, label="CDI (annual)")
    ax.plot(yrs, 100 * annual["old_rate_a4"], "o-", color="firebrick", lw=1.8,
            alpha=0.8, label="OLD rate_a4 (market_panel)")
    ax.plot(yrs, 100 * annual["v1_rate_a4"], "^--", color="darkorange", lw=1.6,
            alpha=0.8, label="v1 UNIFORM")
    ax.plot(yrs, 100 * annual["v2_rate_a4"], "s-", color="navy", lw=2.2,
            label="v2 PER-BANK + segment shrinkage")
    ax.set_xlabel("Year")
    ax.set_ylabel("Annualized rate (%)")
    ax.set_title("Top-10 banks: CDB (k=4) implicit rate -- OLD vs v1 vs v2\n"
                 "(deposit-weighted; per-bank repo/bills/k shrunk to segment, n0=2)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_OUT, dpi=130)
    print(f"\nWrote figure: {FIG_OUT}")


if __name__ == "__main__":
    main()

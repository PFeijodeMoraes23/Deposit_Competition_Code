## diag_cosif_k4_validate.py
# Author: Pedro Feijo de Moraes (validation, paired with panel_cosif_extract.py)
# Objective: Calibrate repo/bills effective rates, compute the CORRECTED k=4
#            (CDB) implicit rate from the v2 foundation table, and produce a
#            top-10 before/after validation (table + ONE figure).
#
# Conglomerate mapping: built from the IF-Data List exactly as panel_deposit_rates.py
# does -- the List CodInst for individual institutions (Td=='I') IS the 8-digit
# CNPJ root that COSIF uses, so we aggregate ALL member CNPJs up to each
# CodConglomeradoPrudencial (the conglomerate's deposit-taking subsidiary, e.g.
# Itau Unibanco S.A. 60701190, is NOT the holding leader CNPJ 60872504).
#
# GUARDRAILS: reads canonical inputs read-only; writes NEW files only under
# COSIF_PROCESSED/CF_FOUNDATION/.  Does NOT rebuild market_panel or demand parquets.

import os
import sys
import glob
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from utils import paths

PROCESSED = str(paths.PROCESSED)
COSIF_PROC = os.path.join(PROCESSED, "COSIF_PROCESSED")
PANEL_INTERMED = os.path.join(PROCESSED, "PANEL_INTERMED")
CF_FOUNDATION = os.path.join(COSIF_PROC, "CF_FOUNDATION")
LIST_DIR = str(paths.IF_DATA_LIST)
os.makedirs(CF_FOUNDATION, exist_ok=True)

FOUNDATION = os.path.join(COSIF_PROC, "custos_implicitos_v2_foundation.csv")
MARKET_PANEL = os.path.join(PROCESSED, "market_panel.csv")
MACRO = os.path.join(PANEL_INTERMED, "quarterly_macro_rates.csv")

FIG_OUT = os.path.join(CF_FOUNDATION, "cdb_rate_fix_before_after_top10.png")
TABLE_OUT = os.path.join(CF_FOUNDATION, "cdb_rate_fix_before_after_top10.csv")
NEWRATE_OUT = os.path.join(CF_FOUNDATION, "cosif_rate_a4_v2_by_congl_quarter.csv")

# Repo/bills rates are calibrated below from 2025+ leaves.  Savings/interbank
# strip uses the macro savings rate and CDI respectively.

STOCK_COLS = ["stk_total", "stk_demand", "stk_savings", "stk_interb",
              "stk_time", "stk_repos", "stk_bills", "stk_prepaid"]
EXPENSE_COLS = ["exp_lump", "exp_savings", "exp_interb", "exp_cdb",
                "exp_repos", "exp_bills", "exp_prepaid"]


def annualize(q):
    return (1.0 + q) ** 4 - 1.0


def load_macro_fraction():
    """quarterly_macro_rates.csv stores *_qoq in PERCENT; return fractions."""
    m = pd.read_csv(MACRO)
    for c in ["cdi_qoq", "selic_qoq", "savings_rate_qoq"]:
        if c in m.columns:
            m[c] = m[c] / 100.0
    return m


def build_cnpj_to_congl():
    """CNPJ(=CodInst for Td=='I') -> CodConglomeradoPrudencial, time-varying.
    Returns df_map[CNPJ, AnoMes_list, CodConglomeradoPrudencial]."""
    files = sorted(glob.glob(os.path.join(LIST_DIR, "IF_DATA_List_*.csv")))
    frames = []
    for fp in files:
        try:
            df = pd.read_csv(fp, dtype=str, encoding="latin-1")
        except Exception:
            df = pd.read_csv(fp, dtype=str, encoding="utf-8")
        df.columns = [c.strip() for c in df.columns]
        mask = (df["Td"] == "I") & (df["Situacao"] == "A")
        sub = df.loc[mask, ["CodInst", "Data",
                            "CodConglomeradoPrudencial"]].copy()
        standalone = (sub["CodConglomeradoPrudencial"].isna()
                      | (sub["CodConglomeradoPrudencial"] == "null"))
        sub.loc[standalone, "CodConglomeradoPrudencial"] = sub.loc[standalone, "CodInst"]
        sub.rename(columns={"CodInst": "CNPJ", "Data": "AnoMes_list"}, inplace=True)
        frames.append(sub)
    m = pd.concat(frames, ignore_index=True)
    m["AnoMes_list"] = pd.to_numeric(m["AnoMes_list"], errors="coerce").astype("Int64")
    m["CNPJ"] = m["CNPJ"].str.strip().str.zfill(8)
    m = m.drop_duplicates(subset=["CNPJ", "AnoMes_list"])
    return m


def map_foundation_to_congl(found, df_map):
    """Map foundation (CNPJ x quarter) to conglomerate via the List.

    The List snapshots begin 2016, while COSIF runs from 2013, and a CNPJ's
    conglomerate is near-static.  So we use an as-of merge: each (CNPJ, quarter)
    is matched to the NEAREST List snapshot for that CNPJ (backward first, then
    the earliest available snapshot for pre-2016 quarters).  This keeps the
    big banks' 2013-2015 rows instead of dropping them (an inner contemporaneous
    merge would lose every pre-2016 quarter)."""
    # merge_asof needs both frames globally sorted by the 'on' key.
    f = found.sort_values(["AnoMes", "CNPJ"]).copy()
    dm = df_map.dropna(subset=["AnoMes_list"]).copy()
    dm["AnoMes_list"] = dm["AnoMes_list"].astype(int)
    dm = dm.sort_values(["AnoMes_list", "CNPJ"])

    # backward as-of: snapshot at or before the quarter
    back = pd.merge_asof(
        f, dm, left_on="AnoMes", right_on="AnoMes_list", by="CNPJ",
        direction="backward",
    )
    # forward fallback (earliest snapshot) for quarters before first snapshot
    fwd = pd.merge_asof(
        f, dm, left_on="AnoMes", right_on="AnoMes_list", by="CNPJ",
        direction="forward",
    )
    merged = back.copy()
    fill = merged["CodConglomeradoPrudencial"].isna()
    merged.loc[fill, "CodConglomeradoPrudencial"] = \
        fwd.loc[fill, "CodConglomeradoPrudencial"]
    merged = merged.dropna(subset=["CodConglomeradoPrudencial"])
    return merged


def main():
    macro = load_macro_fraction()

    found = pd.read_csv(FOUNDATION, dtype={"CNPJ": str})
    found["CNPJ"] = found["CNPJ"].str.zfill(8)
    found["year"] = found["AnoMes"] // 100

    # ===================================================================
    # 1) CALIBRATE repo & bills effective rates from clean 2025+ leaves.
    # ===================================================================
    fc = found.merge(macro[["AnoMes", "cdi_qoq"]], on="AnoMes", how="left")
    post = fc[fc["year"] >= 2025].copy()

    def calib(exp_col, stk_lag_col):
        d = post.dropna(subset=[exp_col, stk_lag_col, "cdi_qoq"]).copy()
        d = d[(d[stk_lag_col] > 0) & (d["cdi_qoq"] > 0)]
        d["rate"] = d[exp_col].abs() / d[stk_lag_col]
        d = d[(d["rate"] >= 0) & (d["rate"] < 0.5)]
        w = d[stk_lag_col]
        agg_rate = np.average(d["rate"], weights=w)
        agg_cdi = np.average(d["cdi_qoq"], weights=w)
        return agg_rate, agg_cdi, agg_rate / agg_cdi, len(d)

    repo_rate, repo_cdi, repo_ratio, n_repo = calib("exp_repos", "stk_repos_lag")
    bills_rate, bills_cdi, bills_ratio, n_bills = calib("exp_bills", "stk_bills_lag")
    cdb_rate, cdb_cdi, cdb_ratio, n_cdb = calib("exp_cdb", "stk_time_lag")

    print("=" * 72)
    print("CALIBRATED EFFECTIVE RATES (2025+ leaves, deposit-weighted, qoq)")
    print(f"  repo  : rate_q={repo_rate:.5f}  cdi_q={repo_cdi:.5f}  "
          f"ratio_to_CDI={repo_ratio:.3f}  (n={n_repo})")
    print(f"  bills : rate_q={bills_rate:.5f}  cdi_q={bills_cdi:.5f}  "
          f"ratio_to_CDI={bills_ratio:.3f}  (n={n_bills})")
    print(f"  CDB   : rate_q={cdb_rate:.5f}  cdi_q={cdb_cdi:.5f}  "
          f"ratio_to_CDI={cdb_ratio:.3f}  (n={n_cdb})   [self-check]")
    print("=" * 72)

    # ===================================================================
    # 2) Aggregate foundation to CONGLOMERATE x quarter, then compute rate.
    # ===================================================================
    df_map = build_cnpj_to_congl()
    merged = map_foundation_to_congl(found, df_map)

    # Sum stocks + (disaccumulated, quarterly) expenses across member CNPJs.
    agg_cols = [c for c in STOCK_COLS + EXPENSE_COLS if c in merged.columns]
    congl = (merged.groupby(["CodConglomeradoPrudencial", "AnoMes"])[agg_cols]
             .sum(min_count=1).reset_index())
    congl["year"] = congl["AnoMes"] // 100
    congl = congl.sort_values(["CodConglomeradoPrudencial", "AnoMes"])

    # Lagged stocks at the CONGLOMERATE level (so the rate denominator and the
    # repo/bills strip use conglomerate-aggregated lagged stocks).
    for c in STOCK_COLS:
        if c in congl.columns:
            congl[c + "_lag"] = congl.groupby("CodConglomeradoPrudencial")[c].shift(1)

    congl = congl.merge(macro[["AnoMes", "cdi_qoq", "savings_rate_qoq"]],
                        on="AnoMes", how="left")
    cdi = congl["cdi_qoq"]
    repo_q = repo_ratio * cdi
    bills_q = bills_ratio * cdi
    is_post = congl["year"] >= 2025

    # ---- pre-2025: strip repo+bills+savings+interbank out of the lump ----
    deposit_expense = (
        congl["exp_lump"].abs()
        - congl["stk_repos_lag"].fillna(0) * repo_q
        - congl["stk_bills_lag"].fillna(0) * bills_q
    )
    cdb_expense_pre = (
        deposit_expense
        - congl["savings_rate_qoq"].fillna(0) * congl["stk_savings_lag"].fillna(0)
        - cdi.fillna(0) * congl["stk_interb_lag"].fillna(0)
    )
    rate_pre = (cdb_expense_pre / congl["stk_time_lag"]).clip(lower=0)
    # ---- 2025+: exact CDB leaf / lagged time stock ----
    rate_post = (congl["exp_cdb"].abs() / congl["stk_time_lag"]).clip(lower=0)

    # ---- STRIP DE-BIASING -----------------------------------------------
    # The pre-2025 residual strip systematically OVERstates the true CDB cost
    # because the lumped expense 81100008 also funds non-CDB liabilities that
    # the strip does not net out (demand has small cost, plus special/judicial/
    # other deposits 4140/4155/4160/4190).  On the 2025+ overlap we observe
    # BOTH the strip (from the lump) and the true CDB leaf 8113000009, so we
    # calibrate a single deposit-weighted multiplicative correction
    #   k = sum(true CDB expense) / sum(strip CDB expense)
    # over the 2025+ big banks, then apply k to the pre-2025 strip.  This
    # anchors the residual to the only ground truth available.
    overlap = congl[is_post].copy()
    ov_strip_exp = (
        overlap["exp_lump"].abs()
        - overlap["stk_repos_lag"].fillna(0) * repo_ratio * overlap["cdi_qoq"]
        - overlap["stk_bills_lag"].fillna(0) * bills_ratio * overlap["cdi_qoq"]
        - overlap["savings_rate_qoq"].fillna(0) * overlap["stk_savings_lag"].fillna(0)
        - overlap["cdi_qoq"].fillna(0) * overlap["stk_interb_lag"].fillna(0)
    ).clip(lower=0)
    ov_true_exp = overlap["exp_cdb"].abs()
    ov_ok = ov_strip_exp.notna() & ov_true_exp.notna() & (ov_strip_exp > 0) \
        & overlap["stk_time_lag"].fillna(0).gt(0)
    # deposit-weighted (by lagged CDB stock) aggregate correction factor
    w = overlap.loc[ov_ok, "stk_time_lag"]
    strip_bias = float(np.average(
        (ov_strip_exp[ov_ok] / ov_true_exp[ov_ok]), weights=w))
    k_correction = 1.0 / strip_bias
    print(f"\nSTRIP DE-BIAS (2025+ overlap, deposit-weighted): "
          f"strip/leaf={strip_bias:.3f}  ->  k_correction={k_correction:.3f} "
          f"(n={int(ov_ok.sum())})")

    rate_pre = (rate_pre * k_correction).clip(lower=0)

    congl["rate_a4_new_qoq"] = np.where(is_post, rate_post, rate_pre)
    congl.loc[~np.isfinite(congl["rate_a4_new_qoq"]), "rate_a4_new_qoq"] = np.nan
    congl.loc[congl["rate_a4_new_qoq"] > 0.5, "rate_a4_new_qoq"] = np.nan
    congl["rate_a4_new_ann"] = annualize(congl["rate_a4_new_qoq"])

    congl.to_csv(NEWRATE_OUT, index=False)
    print(f"Wrote per-congl new rate: {NEWRATE_OUT}")

    # ===================================================================
    # 3) Top-10 conglomerates by total CDB deposits (market_panel dep_a4),
    #    and OLD congl-level rate_a4 (annual, deposit-weighted within congl).
    # ===================================================================
    mp_cols = ["CodConglomeradoPrudencial", "NomeInstituicao", "year",
               "quarter", "dep_a4", "rate_a4", "risk_free_ann"]
    mp = pd.read_csv(MARKET_PANEL, usecols=mp_cols,
                     dtype={"CodConglomeradoPrudencial": str})
    mp["AnoMes"] = mp["year"] * 100 + mp["quarter"] * 3

    top = (mp.groupby("CodConglomeradoPrudencial")
           .agg(dep=("dep_a4", "sum"), name=("NomeInstituicao", "first"))
           .reset_index().sort_values("dep", ascending=False).head(10))
    top_codes = top["CodConglomeradoPrudencial"].tolist()
    print("\nTOP-10 conglomerates by total CDB deposits:")
    print(top[["CodConglomeradoPrudencial", "name", "dep"]].to_string(index=False))

    mp_top = mp[mp["CodConglomeradoPrudencial"].isin(top_codes)].copy()
    old = (mp_top.groupby(["CodConglomeradoPrudencial", "AnoMes"])
           .apply(lambda g: pd.Series({
               "old_rate_ann": np.average(g["rate_a4"],
                                          weights=g["dep_a4"].clip(lower=0) + 1e-9),
               "rf_ann": np.average(g["risk_free_ann"],
                                    weights=g["dep_a4"].clip(lower=0) + 1e-9),
               "dep_a4": g["dep_a4"].sum(),
           }), include_groups=False).reset_index())

    new = congl[congl["CodConglomeradoPrudencial"].isin(top_codes)][
        ["CodConglomeradoPrudencial", "AnoMes", "rate_a4_new_ann"]].copy()

    macro2 = macro.copy()
    macro2["cdi_ann"] = annualize(macro2["cdi_qoq"])

    comp = old.merge(new, on=["CodConglomeradoPrudencial", "AnoMes"], how="left")
    comp["year"] = comp["AnoMes"] // 100
    comp = comp.merge(macro2[["AnoMes", "cdi_ann"]], on="AnoMes", how="left")
    comp = comp[(comp["year"] >= 2013) & (comp["year"] <= 2025)]

    def wavg(s, w):
        mm = s.notna() & w.notna() & (w > 0)
        return np.average(s[mm], weights=w[mm]) if mm.any() else np.nan

    annual = (comp.groupby("year")
              .apply(lambda g: pd.Series({
                  "old_rate_a4": wavg(g["old_rate_ann"], g["dep_a4"]),
                  "new_rate_a4": wavg(g["rate_a4_new_ann"], g["dep_a4"]),
                  "cdi": g["cdi_ann"].mean(),
                  "rf": wavg(g["rf_ann"], g["dep_a4"]),
                  "n_new": g["rate_a4_new_ann"].notna().sum(),
              }), include_groups=False).reset_index())
    annual["band_lo"] = 0.90 * annual["cdi"]
    annual["band_hi"] = 1.15 * annual["cdi"]
    annual["old_pct_cdi"] = annual["old_rate_a4"] / annual["cdi"]
    annual["new_pct_cdi"] = annual["new_rate_a4"] / annual["cdi"]
    annual["new_in_band"] = ((annual["new_rate_a4"] >= annual["band_lo"]) &
                             (annual["new_rate_a4"] <= annual["band_hi"]))

    annual.to_csv(TABLE_OUT, index=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("\n" + "=" * 72)
    print("ANNUAL DEPOSIT-WEIGHTED CDB RATE: OLD vs NEW vs CDI (top-10)")
    print("=" * 72)
    print(annual[["year", "old_rate_a4", "new_rate_a4", "cdi", "band_lo",
                  "band_hi", "old_pct_cdi", "new_pct_cdi", "new_in_band"]]
          .to_string(index=False))
    print(f"\nWrote table: {TABLE_OUT}")

    win = annual[(annual["year"] >= 2015) & (annual["year"] <= 2021)]
    near = ((win["new_pct_cdi"] >= 0.85) & (win["new_pct_cdi"] <= 1.20))
    in_band = win["new_in_band"]
    print(f"\nSUCCESS CHECK 2015-2021:")
    print(f"  NEW strictly in 90-115% band : {in_band.sum()}/{len(win)} years")
    print(f"  NEW near band [0.85,1.20]xCDI: {near.sum()}/{len(win)} years")
    old_near = ((win["old_pct_cdi"] >= 0.85) & (win["old_pct_cdi"] <= 1.20))
    print(f"  (OLD near band, for contrast): {old_near.sum()}/{len(win)} years")

    # ===================================================================
    # 4) Figure.
    # ===================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    yrs = annual["year"]
    ax.fill_between(yrs, 100 * annual["band_lo"], 100 * annual["band_hi"],
                    color="green", alpha=0.12,
                    label="Plausible band (90-115% of CDI)")
    ax.plot(yrs, 100 * annual["cdi"], "k--", lw=1.5, label="CDI (annual)")
    ax.plot(yrs, 100 * annual["old_rate_a4"], "o-", color="firebrick",
            lw=2, label="OLD rate_a4 (market_panel)")
    ax.plot(yrs, 100 * annual["new_rate_a4"], "s-", color="navy",
            lw=2, label="NEW rate_a4 (corrected)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Annualized rate (%)")
    ax.set_title("Top-10 banks: CDB (k=4) implicit rate -- before vs after fix\n"
                 f"(deposit-weighted; calibrated repo={repo_ratio:.2f}xCDI, "
                 f"bills={bills_ratio:.2f}xCDI)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_OUT, dpi=130)
    print(f"\nWrote figure: {FIG_OUT}")


if __name__ == "__main__":
    main()

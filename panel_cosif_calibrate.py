## panel_cosif_calibrate.py
# Author: Pedro Feijo de Moraes
# Objective: Build the CORRECTED k=4 (CDB / time-deposit) implicit rate per
#            (CodConglomeradoPrudencial x quarter), fixing the artifact in the
#            old blended/residual approach.
#
# Method (per-bank effective rates + precision-weighted shrinkage to segment):
#   1. From each conglomerate's 2025+ 10-digit COSIF leaves, estimate effective
#      repo / bills rates (as x CDI), pool them to a PRUDENTIAL SEGMENT (S1-S5)
#      profile (deposit-weighted over banks in the segment with 2025 data) and
#      shrink each bank toward it with w_i = n_i / (n_i + N0), N0 = 2, where n_i is
#      the number of the bank's 2025+ quarters with a CDB leaf, a lagged time stock
#      and a CDI.  A bank with no own ratio takes the segment's.  Only banks with both
#      own ratios get an own k_i (step 2).
#   2. The per-bank de-bias factor
#        k_i = (true CDB rate from leaf 8113000009) / (strip-implied CDB rate),
#      with the strip built from the SHRUNK ratios of step 1: the same residual
#      formula, with the same ratios, that the pre-2025 rate is built from.  k_i is
#      pooled and shrunk like the ratios.  Banks with no 2025 data -> the segment
#      profile; segment also missing -> the global one.
#   3. Apply:
#        pre-2025: rate_a4 = k_shrunk * strip_cdb_rate(repo_shrunk, bills_shrunk)
#        2025+   : rate_a4 = |8113000009| / lag(time-deposit stock)   (direct leaf)
#
# Inputs  : custos_implicitos_v2_foundation.csv      (panel_cosif_extract.py, step 12; BANCOS
#             members, and the SCFI members used only where a conglomerate has no bank)
#           SGS_RAW/macro_series_full.csv  (quarterly macro rates computed here through
#             utils.macro_rates, as fractions; step 16 writes the same rates in percent)
#           IF-Data List  (CNPJ<->conglomerate merge_asof crosswalk, utils.conglomerates,
#             keyed as deposits_panel.csv files each CNPJ; and the segment Sr per conglomerate)
# Output  : COSIF_PROCESSED/cosif_cdb_rate_corrected.csv
#             columns: CodConglomeradoPrudencial, AnoMes, rate_a4_corrected_qoq,
#                      rate_a4_corrected_ann, tier
#
# NOTE: rate_a4_corrected_qoq is a fraction (e.g. 0.025 = 2.5% per quarter), the scale
# panel_deposit_rates.py uses internally.
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
from utils.conglomerates import (build_cnpj_to_congl, map_foundation_to_congl,  # noqa: E402
                                 canonicalize_codes, deposit_panel_codes)
from utils.macro_rates import load_macro_series, make_quarterly_macro  # noqa: E402

PROCESSED = str(paths.PROCESSED)
COSIF_PROC = os.path.join(PROCESSED, "COSIF_PROCESSED")

FOUNDATION = os.path.join(COSIF_PROC, "custos_implicitos_v2_foundation.csv")
IF_LIST_DIR = str(paths.IF_DATA_LIST)
SEGMENTS = ["S1", "S2", "S3", "S4", "S5"]
OUT_CORRECTED = os.path.join(COSIF_PROC, "cosif_cdb_rate_corrected.csv")

N0 = 2                  # precision-shrinkage prior count
RATE_Q_CAP = 0.5        # economically implausible quarterly-rate guard

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("panel_cosif_calibrate")

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
    """Quarterly macro rates as fractions, computed from the staged SGS series. Step 16 writes
    the same rates, in percent, to PANEL_INTERMED/quarterly_macro_rates.csv; computing them
    here keeps this step from reading the output of a later step."""
    return make_quarterly_macro(load_macro_series())


def load_segment_map():
    """conglomerate -> prudential segment (S1-S5), from the IF-Data List files: for each
    canonical code, the modal Sr over the rows of its latest snapshot that carry a valid Sr
    (ties -> the lower segment number).  The List files are a raw input, so this step reads
    no later pipeline output.  On the 2026-10-15 panel this agrees with the market panel's
    latest segment for 385 of the 386 conglomerates both carry.  Raises if no List file has
    an Sr column: every bank would otherwise fall back to the global profile."""
    files = sorted(glob.glob(os.path.join(IF_LIST_DIR, "IF_DATA_List_*.csv")))
    frames = []
    for f in files:
        if os.path.getsize(f) <= 500:          # headers-only stub
            continue
        if "Sr" not in pd.read_csv(f, nrows=0, encoding="latin1").columns:
            continue
        frames.append(pd.read_csv(f, encoding="latin1", dtype=str,
                                  usecols=["Data", "CodConglomeradoPrudencial", "Sr"]))
    if not frames:
        raise RuntimeError(
            f"No IF-Data List file in {IF_LIST_DIR} carries an Sr column; every bank "
            "would fall back to the global profile.")
    lf = pd.concat(frames, ignore_index=True).dropna(subset=["CodConglomeradoPrudencial"])
    lf["Sr"] = lf["Sr"].astype(str).str.strip().str.upper()
    lf = lf[lf["Sr"].isin(SEGMENTS)].copy()
    lf["Data"] = pd.to_numeric(lf["Data"], errors="coerce")
    lf["CodConglomeradoPrudencial"] = canonicalize_codes(lf["CodConglomeradoPrudencial"])
    tail = lf[lf["Data"] == lf.groupby("CodConglomeradoPrudencial")["Data"].transform("max")]
    seg = (tail.groupby(["CodConglomeradoPrudencial", "Sr"]).size().rename("n").reset_index()
               .sort_values(["CodConglomeradoPrudencial", "n", "Sr"], ascending=[True, False, True])
               .drop_duplicates("CodConglomeradoPrudencial")
               .rename(columns={"Sr": "segment"})[["CodConglomeradoPrudencial", "segment"]])
    log.info("Segment map: %d conglomerates from %d IF-Data List files", len(seg), len(frames))
    return seg


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
    # key each CNPJ by the code deposits_panel.csv files its deposits under
    df_map["CodConglomeradoPrudencial"] = deposit_panel_codes(df_map["CNPJ"], df_map["CodConglomeradoPrudencial"])
    merged = map_foundation_to_congl(found, df_map)
    # A conglomerate-quarter is measured on its BANCOS members; its credit-finance companies
    # (src SOCIEDADES) count only when it has no BANCOS member that quarter, i.e. for the
    # conglomerates whose deposit-taking entity is an SCFI (Nubank, Mercado Pago, Neon, ...).
    merged["scfi"] = False
    if "src" in merged.columns:
        cq = ["CodConglomeradoPrudencial", "AnoMes"]
        has_bank = (merged["src"] == "BANCOS").groupby([merged[c] for c in cq]).transform("any")
        merged["scfi"] = ~has_bank
        n_scfi = int(((merged["src"] != "BANCOS") & ~has_bank).sum())
        merged = merged[(merged["src"] == "BANCOS") | ~has_bank]
        log.info("SCFI rows used for conglomerate-quarters without a BANCOS member: %d", n_scfi)
    agg_cols = [c for c in STOCK_COLS + EXPENSE_COLS if c in merged.columns]
    # scfi: the conglomerate-quarter is measured on credit-finance companies, not banks
    congl = (merged.groupby(["CodConglomeradoPrudencial", "AnoMes"])
             .agg(**{c: (c, lambda s: s.sum(min_count=1)) for c in agg_cols}, scfi=("scfi", "any"))
             .reset_index())
    congl["year"] = congl["AnoMes"] // 100
    congl = congl.sort_values(["CodConglomeradoPrudencial", "AnoMes"])
    for c in STOCK_COLS:
        if c in congl.columns:
            congl[c + "_lag"] = congl.groupby("CodConglomeradoPrudencial")[c].shift(1)
    return congl.merge(macro[["AnoMes", "cdi_qoq", "savings_rate_qoq"]],
                       on="AnoMes", how="left")


def cdb_quarters(g):
    """The 2025+ quarters that measure the bank's CDB cost: a leaf expense, a positive lagged
    time-deposit stock and a CDI (the macro file can end before the COSIF data)."""
    q = g.dropna(subset=["exp_cdb", "stk_time_lag", "cdi_qoq"])
    return q[(q["stk_time_lag"] > 0) & (q["cdi_qoq"] > 0)].copy()


def estimate_perbank(post):
    """Per-conglomerate 2025+ own repo/bills ratios (as x CDI), n_i (the bank's CDB quarters,
    cdb_quarters) and the weight dep_w.  k_i comes later, from strip_k, once the ratios it is
    applied with are known."""
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
        cdbq = cdb_quarters(g)
        n_i = len(cdbq)
        rows.append({"CodConglomeradoPrudencial": code, "segment": seg,
                     "repo_ratio_i": repo_i, "bills_ratio_i": bills_i, "n_i": n_i,
                     "dep_w": cdbq["stk_time_lag"].sum() if n_i else 0.0,
                     # an SCFI conglomerate is measured on SCFIs in every 2025+ quarter; a bank group
                     # whose bank member stops reporting late in the sample stays a bank
                     "scfi": bool(g["scfi"].all()) if "scfi" in g else False})
    return pd.DataFrame(rows)


def strip_k(post, ratios):
    """k_i = true CDB rate (leaf) / strip-implied CDB rate, both x CDI and time-stock-weighted
    over cdb_quarters.  The strip uses the repo/bills ratios the pre-2025 application uses
    (`ratios`: CodConglomeradoPrudencial, repo_shrunk, bills_shrunk, repo_ratio_i,
    bills_ratio_i), so k corrects the formula as it is applied.  Only a bank with its own repo
    AND its own bills ratio gets an own k; the others take the segment's.  Extending own k to
    banks that hold no repos or bills was measured (2026-10-16) and rejected: their strips are
    small next to the CDB leaf, and k ran from 0.19 to 13.3."""
    r = ratios.set_index("CodConglomeradoPrudencial")
    k = {}
    for code, g in post.groupby("CodConglomeradoPrudencial"):
        cdbq = cdb_quarters(g)
        if code not in r.index or not len(cdbq):
            continue
        if pd.isna(r.at[code, "repo_ratio_i"]) or pd.isna(r.at[code, "bills_ratio_i"]):
            continue
        w = cdbq["stk_time_lag"]
        true_x = wmean((cdbq["exp_cdb"].abs() / w) / cdbq["cdi_qoq"], w)
        se = strip_cdb_expense(cdbq, r.at[code, "repo_shrunk"], r.at[code, "bills_shrunk"])
        strip_x = wmean((se / w) / cdbq["cdi_qoq"], w)
        k[code] = true_x / strip_x if (np.isfinite(strip_x) and strip_x > 0) else np.nan
    return pd.Series(k, name="k_i", dtype=float)


def build_corrected():
    """The corrected type-4 rate (the frame main() writes) and the per-bank calibration table."""
    macro = load_macro_fraction()
    found = pd.read_csv(FOUNDATION, dtype={"CNPJ": str})
    found["CNPJ"] = found["CNPJ"].str.zfill(8)

    congl = build_congl_panel(found, macro)
    seg_map = load_segment_map()
    congl = congl.merge(seg_map, on="CodConglomeradoPrudencial", how="left")

    post = congl[congl["year"] >= 2025].copy()
    pb = estimate_perbank(post)
    has_q = pb["n_i"] > 0

    # ---- segment profiles (deposit-weighted over BANK conglomerates with 2025 data; the
    #      SCFI-measured ones have a different funding structure and do not enter them) ----
    bank = has_q & ~pb["scfi"]

    def profile(col):
        prof = {s: wmean(g.loc[bank.loc[g.index], col], g.loc[bank.loc[g.index], "dep_w"])
                for s, g in pb.dropna(subset=["segment"]).groupby("segment")}
        return prof, wmean(pb.loc[bank, col], pb.loc[bank, "dep_w"])

    def seg_val(prof, seg, fallback):
        v = prof.get(seg, np.nan)
        return v if pd.notna(v) else fallback

    # ---- shrinkage (w_i = n_i/(n_i+N0)); per-component fallback to segment ----
    pb["w_i"] = pb["n_i"] / (pb["n_i"] + N0)

    def shrink(own, prof, fb):
        segv = pb["segment"].apply(lambda s: seg_val(prof, s, fb))
        w = pb["w_i"].where(pb[own].notna(), 0.0)
        return w * pb[own].fillna(0.0) + (1 - w) * segv

    repo_prof, repo_fb = profile("repo_ratio_i")
    bills_prof, bills_fb = profile("bills_ratio_i")
    pb["repo_shrunk"] = shrink("repo_ratio_i", repo_prof, repo_fb)
    pb["bills_shrunk"] = shrink("bills_ratio_i", bills_prof, bills_fb)
    # k is measured with the ratios it is applied with (strip_k), then shrunk the same way
    pb["k_i"] = pb["CodConglomeradoPrudencial"].map(
        strip_k(post, pb[["CodConglomeradoPrudencial", "repo_shrunk", "bills_shrunk",
                          "repo_ratio_i", "bills_ratio_i"]]))
    k_prof, k_fb = profile("k_i")
    pb["k_shrunk"] = shrink("k_i", k_prof, k_fb)
    seg_prof = pd.DataFrame({"repo_ratio_seg": repo_prof, "bills_ratio_seg": bills_prof,
                             "k_seg": k_prof})
    glob_fb = {"repo": repo_fb, "bills": bills_fb, "k": k_fb}
    log.info("Segment profiles:\n%s", seg_prof.to_string())
    log.info("Global fallback: repo=%.3f bills=%.3f k=%.3f",
             glob_fb["repo"], glob_fb["bills"], glob_fb["k"])
    log.info("Own k_i for %d of %d conglomerates with 2025 CDB quarters",
             int(pb.loc[has_q, "k_i"].notna().sum()), int(has_q.sum()))

    def seg_val_col(seg, col, fallback):
        return seg_val(seg_prof[col].to_dict(), seg, fallback)

    # ---- assign params to EVERY conglomerate (incl. those absent in 2025) ----
    all_codes = congl[["CodConglomeradoPrudencial"]].drop_duplicates().merge(
        seg_map, on="CodConglomeradoPrudencial", how="left")
    params = all_codes.merge(
        pb[["CodConglomeradoPrudencial", "repo_shrunk", "bills_shrunk",
            "k_shrunk", "n_i"]], on="CodConglomeradoPrudencial", how="left")
    params["n_i"] = params["n_i"].fillna(0)
    params["repo_shrunk"] = params.apply(
        lambda r: r["repo_shrunk"] if pd.notna(r["repo_shrunk"])
        else seg_val_col(r["segment"], "repo_ratio_seg", glob_fb["repo"]), axis=1)
    params["bills_shrunk"] = params.apply(
        lambda r: r["bills_shrunk"] if pd.notna(r["bills_shrunk"])
        else seg_val_col(r["segment"], "bills_ratio_seg", glob_fb["bills"]), axis=1)
    params["k_shrunk"] = params.apply(
        lambda r: r["k_shrunk"] if pd.notna(r["k_shrunk"])
        else seg_val_col(r["segment"], "k_seg", glob_fb["k"]), axis=1)

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
    return out, pb


def main():
    if not os.path.exists(FOUNDATION):
        log.error("Foundation not found: %s -- run panel_cosif_extract.py first.",
                  FOUNDATION)
        sys.exit(1)
    out, _ = build_corrected()
    out.to_csv(OUT_CORRECTED, index=False)
    log.info("Wrote corrected CDB rate: %s  (%d rows, %d conglomerates, %d quarters)",
             OUT_CORRECTED, len(out), out["CodConglomeradoPrudencial"].nunique(),
             out["AnoMes"].nunique())


if __name__ == "__main__":
    main()

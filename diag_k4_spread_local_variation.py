"""
diag_k4_spread_local_variation.py
=================================
READ-ONLY diagnostic.  Does NOT modify any panel.

QUESTION.  For deposit type k=4 (time deposits / CDB) held by B-type
(municipal-reporting) institutions, is the SPREAD built from national data only,
even though ESTBAN carries municipal (local) time-deposit data?

WHAT THIS SCRIPT ESTABLISHES.
  Stage A.  Variance decomposition of the wide market panel: for each of
            spread_a4, rate_a4, dep_a4, and the type-4 conditional market share,
            what share of the variance is BETWEEN (conglomerate x quarter) vs
            WITHIN (across the MCAs a conglomerate serves)?  The claim to test:
            the PRICE (spread_a4/rate_a4) has ZERO within variation (pure
            national broadcast), while the QUANTITY (dep_a4, share) — sourced
            from ESTBAN V432 per municipality — has large within variation.

  Stage B.  Geography of the type-4 book for B-type conglomerates: how many
            MCAs a conglomerate serves, and what share of the national CDB book
            sits in multi-MCA conglomerates (i.e. is "exposed" to the missing
            local price variation).

  Stage C.  CEILING on genuine local rate variation obtainable from ESTBAN.
            ESTBAN has municipal BALANCES (V432) but NO municipal interest
            expense, so a true municipal CDB rate is not directly measurable.
            The only genuine-data route is member-CNPJ compositional variation:
            a prudential conglomerate can contain several member CNPJs, each with
            its own national CDB rate, and ESTBAN says which CNPJ holds the CDB
            book in which municipality.  This stage builds the counterfactual
            municipal rate
               r(congl, muni, t) = Σ_{cnpj∈congl} w_V432(cnpj,muni,t) · r_cnpj,t
            and measures the within-(congl x quarter) dispersion it would create,
            i.e. how much local price variation is even available.  Per-CNPJ
            CDB rate r_cnpj,t uses the real 2025+ COSIF leaf exp_cdb/stk_time_lag.
            A period-agnostic structural count (how many conglomerates are
            multi-CNPJ in the CDB book at all) is also reported.

OUTPUTS (under processed/ESTIMATION_OUTPUT/DIAG_K4_SPREAD/):
  report_k4_spread_local.txt           full text report
  tableA_variance_decomp.csv
  tableB_geography.csv
  tableC_member_cnpj_ceiling.csv
  fig_k4_local_variation.png

Usage:  python diag_k4_spread_local_variation.py
"""
from __future__ import annotations

try:
    from utils.venv_guard import ensure_project_venv
    ensure_project_venv(__file__)
except Exception:
    pass

import os
import sys
os.environ.setdefault("MPLBACKEND", "Agg")  # headless / OneDrive-safe
try:                                          # Windows console defaults to cp1252
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import paths
import panel_cosif_calibrate as cc

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("diag_k4")

PANEL_CSV = paths.PROCESSED / "market_panel.csv"
FOUNDATION = paths.PROCESSED / "COSIF_PROCESSED" / "custos_implicitos_v2_foundation.csv"
ESTBAN_CSV = paths.ESTBAN_CSV
OUT_DIR = paths.PROCESSED / "ESTIMATION_OUTPUT" / "DIAG_K4_SPREAD"

RATE_Q_CAP = 0.5          # economically-implausible quarterly rate guard (matches cosif_process_2)
_report_lines: list[str] = []


def say(msg: str = "") -> None:
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"))
    _report_lines.append(msg)


def between_within(df: pd.DataFrame, value: str, group_cols: list[str]) -> dict:
    """Decompose Var(value) into between-group and within-group components using
    only groups with >=2 non-null rows (so 'within' is defined).  Returns the
    within-group share of total variance and how many multi-row groups carry ANY
    within-group dispersion."""
    d = df[group_cols + [value]].dropna(subset=[value]).copy()
    g = d.groupby(group_cols)[value]
    n = g.transform("size")
    multi = d[n > 1].copy()
    if multi.empty:
        return dict(n_groups_multi=0, within_share=np.nan, groups_with_variation_pct=np.nan,
                    mean_within_std=np.nan, total_std=np.nan)
    gm = multi.groupby(group_cols)[value]
    grp_mean = gm.transform("mean")
    within_dev = multi[value] - grp_mean
    total_var = np.nanvar(multi[value].to_numpy())
    within_var = np.nanvar(within_dev.to_numpy())
    # per-group dispersion summary
    per = gm.agg(["size", "nunique", "std"])
    per = per[per["size"] > 1]
    return dict(
        n_groups_multi=int(len(per)),
        within_share=float(within_var / total_var) if total_var > 0 else np.nan,
        groups_with_variation_pct=float(100 * (per["nunique"] > 1).mean()),
        mean_within_std=float(per["std"].mean()),
        total_std=float(np.sqrt(total_var)),
    )


# ----------------------------------------------------------------------------
# Stage A — variance decomposition (national broadcast test)
# ----------------------------------------------------------------------------
def stage_a(panel: pd.DataFrame) -> pd.DataFrame:
    say("\n" + "=" * 78)
    say("STAGE A — Is the type-4 spread national-only? (variance decomposition)")
    say("=" * 78)
    say("Sample: B-type rows only (mca_code != 'NATIONAL'); group = conglomerate x year x quarter;")
    say("'within' = variation across the MCAs a conglomerate serves in that quarter.\n")

    b = panel[panel["mca_code"].astype(str).str.upper() != "NATIONAL"].copy()
    say(f"B-type rows: {len(b):,}  |  conglomerates: {b['CodConglomeradoPrudencial'].nunique():,}  "
        f"|  MCAs: {b['mca_code'].nunique():,}")

    # type-4 conditional market share (quantity signal that DOES vary locally)
    b["mkt_dep_a4"] = b.groupby(["mca_code", "year", "quarter"])["dep_a4"].transform("sum")
    b["share_a4"] = b["dep_a4"] / b["mkt_dep_a4"].replace(0.0, np.nan)
    b["log_dep_a4"] = np.log1p(b["dep_a4"].clip(lower=0))

    grp = ["CodConglomeradoPrudencial", "year", "quarter"]
    rows = []
    for var, kind in [("spread_a4", "PRICE"), ("rate_a4", "PRICE"),
                      ("spread_ann_a4", "PRICE"),
                      ("dep_a4", "QUANTITY"), ("log_dep_a4", "QUANTITY"),
                      ("share_a4", "QUANTITY")]:
        if var not in b.columns:
            continue
        r = between_within(b, var, grp)
        r.update(variable=var, kind=kind)
        rows.append(r)
    ta = pd.DataFrame(rows)[["variable", "kind", "n_groups_multi", "within_share",
                             "groups_with_variation_pct", "mean_within_std", "total_std"]]
    say("")
    say(f"{'variable':16s} {'kind':9s} {'multiMCA_grps':>13s} {'within_var%':>11s} "
        f"{'grps_w_var%':>11s} {'mean_within_sd':>14s}")
    for _, r in ta.iterrows():
        say(f"{r['variable']:16s} {r['kind']:9s} {r['n_groups_multi']:13d} "
            f"{100*r['within_share']:11.4f} {r['groups_with_variation_pct']:11.2f} "
            f"{r['mean_within_std']:14.6g}")
    say("")
    say("READING: within_var% ≈ 0 and grps_w_var% ≈ 0 ⇒ the variable is a pure national")
    say("broadcast (identical across every MCA of a conglomerate-quarter).  A large value ⇒")
    say("the variable carries genuine local (municipal/MCA) variation.")
    return ta


# ----------------------------------------------------------------------------
# Stage B — geography of the B-type type-4 book
# ----------------------------------------------------------------------------
def stage_b(panel: pd.DataFrame) -> pd.DataFrame:
    say("\n" + "=" * 78)
    say("STAGE B — Geography of the B-type CDB (type-4) book")
    say("=" * 78)
    b = panel[panel["mca_code"].astype(str).str.upper() != "NATIONAL"].copy()
    b = b[b["dep_a4"].fillna(0) > 0]

    n_mca = b.groupby(["CodConglomeradoPrudencial", "year", "quarter"])["mca_code"].nunique()
    say("\nMCAs served per (conglomerate x quarter), type-4 book > 0:")
    for q in [0.5, 0.75, 0.9, 0.99, 1.0]:
        say(f"  p{int(q*100):02d} = {n_mca.quantile(q):.0f}")
    say(f"  mean = {n_mca.mean():.1f}   share of congl-quarters in >1 MCA = "
        f"{100*(n_mca > 1).mean():.1f}%")

    # share of the national CDB book that sits in multi-MCA conglomerates
    congl_q = b.groupby(["CodConglomeradoPrudencial", "year", "quarter"]).agg(
        dep=("dep_a4", "sum"), n_mca=("mca_code", "nunique")).reset_index()
    tot = congl_q["dep"].sum()
    multi_share = 100 * congl_q.loc[congl_q["n_mca"] > 1, "dep"].sum() / max(tot, 1e-9)
    say(f"\nShare of total B-type type-4 deposits held by multi-MCA conglomerates: {multi_share:.1f}%")
    say("(this is the fraction of the CDB book 'exposed' to the missing local price variation)")

    rows = []
    for q in [0.5, 0.75, 0.9, 0.99, 1.0]:
        rows.append(dict(stat=f"p{int(q*100):02d}_mcas_per_congl_qtr", value=float(n_mca.quantile(q))))
    rows.append(dict(stat="mean_mcas_per_congl_qtr", value=float(n_mca.mean())))
    rows.append(dict(stat="pct_congl_qtrs_multi_mca", value=float(100*(n_mca > 1).mean())))
    rows.append(dict(stat="pct_type4_deposits_in_multi_mca_congl", value=float(multi_share)))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Stage C — ceiling on genuine local rate variation (member-CNPJ route)
# ----------------------------------------------------------------------------
def _per_cnpj_rates(found: pd.DataFrame) -> pd.DataFrame:
    """Per-CNPJ quarterly funding rates from the foundation table.  Expenses are
    stored NEGATIVE (despesas) so we take abs(); flows are already disaccumulated.
      r_cdb  = |exp_cdb| / stk_time_lag   -- the exact CDB rate, but the direct
               COSIF leaf 8113 only exists from the 2025 recode (2025+).
      r_lump = |exp_lump| / stk_total_lag -- total funding cost, available all
               years (2013+); a PROXY for whether member CNPJs differ in price.
    Both guarded to (0, RATE_Q_CAP]."""
    f = found.copy()
    for c in ["exp_cdb", "exp_lump", "stk_time", "stk_time_lag", "stk_total_lag"]:
        f[c] = pd.to_numeric(f[c], errors="coerce")

    def _rate(num_abs, den):
        r = num_abs.abs() / den.replace(0.0, np.nan)
        r = r.where(np.isfinite(r))
        return r.where((r > 0) & (r <= RATE_Q_CAP))

    f["r_cdb"] = _rate(f["exp_cdb"], f["stk_time_lag"])
    f["r_lump"] = _rate(f["exp_lump"], f["stk_total_lag"])
    # preferred per-CNPJ CDB rate: real leaf where present, else lump proxy
    f["r_cnpj"] = f["r_cdb"].fillna(f["r_lump"])
    return f


def _weighted_muni_dispersion(fc: pd.DataFrame, rate_col: str, label: str,
                              years: list[int] | None = None) -> pd.DataFrame:
    """Build the ESTBAN V432-weighted counterfactual municipal rate and return
    per-(congl x quarter) within-dispersion across municipalities."""
    fr = fc.dropna(subset=[rate_col]).copy()
    ryears = sorted({int(a) // 100 for a in fr["AnoMes"].tolist()})
    if years is not None:
        ryears = [y for y in ryears if y in years]
    if not ryears:
        say(f"    [{label}] no rated years; skip.")
        return pd.DataFrame()
    est = pd.read_csv(ESTBAN_CSV, usecols=["CNPJ", "CODMUN_IBGE", "YEAR", "MONTH", "V432"],
                      dtype={"CNPJ": str}, low_memory=False, encoding="latin-1")
    est = est[est["MONTH"].isin([3, 6, 9, 12]) & est["YEAR"].isin(ryears)].copy()
    est["CNPJ"] = est["CNPJ"].str.strip().str.zfill(8)
    est["V432"] = pd.to_numeric(est["V432"], errors="coerce").fillna(0.0).clip(lower=0)
    est["AnoMes"] = (est["YEAR"].astype(int) * 100 + est["MONTH"].astype(int)).astype(int)
    est = est.merge(fr[["CNPJ", "AnoMes", "CodConglomeradoPrudencial", rate_col]],
                    on=["CNPJ", "AnoMes"], how="inner")
    est = est[(est["V432"] > 0) & est[rate_col].notna()]
    if est.empty:
        say(f"    [{label}] no (CNPJ x muni x qtr) ESTBAN V432 rows with a rated CNPJ; skip.")
        return pd.DataFrame()
    key = ["CodConglomeradoPrudencial", "CODMUN_IBGE", "AnoMes"]
    est["wr"] = est["V432"] * est[rate_col]
    muni = est.groupby(key).agg(wr=("wr", "sum"), w=("V432", "sum")).reset_index()
    muni["r_muni"] = muni["wr"] / muni["w"]
    disp = muni.groupby(["CodConglomeradoPrudencial", "AnoMes"]).agg(
        n_muni=("CODMUN_IBGE", "nunique"), r_sd=("r_muni", "std"),
        r_min=("r_muni", "min"), r_max=("r_muni", "max")).reset_index()
    disp = disp[disp["n_muni"] > 1]
    say(f"    [{label}] years={ryears}; congl-quarters with >1 municipality & rated book: {len(disp):,}")
    if len(disp):
        say(f"    [{label}] within-congl-qtr SD of counterfactual municipal rate (bps/qtr):")
        for q in [0.5, 0.9, 0.99, 1.0]:
            say(f"       p{int(q*100):02d} = {1e4*disp['r_sd'].quantile(q):.2f}")
        say(f"       mean = {1e4*disp['r_sd'].mean():.2f}   |  "
            f"congl-qtrs with local SD > 5 bps/qtr: {100*(disp['r_sd'] > 5e-4).mean():.1f}%")
    disp["measure"] = label
    return disp


def stage_c(panel: pd.DataFrame) -> pd.DataFrame:
    say("\n" + "=" * 78)
    say("STAGE C — Ceiling on genuine local rate variation from ESTBAN (member-CNPJ route)")
    say("=" * 78)
    say("ESTBAN has municipal BALANCES (V432) but NO municipal interest expense, so a true")
    say("municipal CDB rate is not directly measurable.  The only genuine-data route to local")
    say("price variation is member-CNPJ composition: r(congl,muni,t) = Σ w_V432(cnpj,muni,t)·r_cnpj,t.\n")

    found = pd.read_csv(FOUNDATION, dtype={"CNPJ": str})
    found["CNPJ"] = found["CNPJ"].str.strip().str.zfill(8)
    found["AnoMes"] = pd.to_numeric(found["AnoMes"], errors="coerce").astype("Int64")
    found = _per_cnpj_rates(found)

    cmap = cc.build_cnpj_to_congl()
    fc = cc.map_foundation_to_congl(
        found[["CNPJ", "AnoMes", "stk_time", "r_cdb", "r_lump", "r_cnpj"]]
        .dropna(subset=["AnoMes"]).assign(AnoMes=lambda d: d["AnoMes"].astype(int)), cmap)

    # --- C1. Structural multi-CNPJ counts + cross-CNPJ rate dispersion ---
    fc["has_book"] = pd.to_numeric(fc["stk_time"], errors="coerce").fillna(0) > 0
    bk = fc[fc["has_book"]]
    per_cq = bk.groupby(["CodConglomeradoPrudencial", "AnoMes"]).agg(
        n_cnpj=("CNPJ", "nunique"),
        n_cnpj_rated=("r_cnpj", lambda s: s.notna().sum()),
        rate_sd=("r_cnpj", "std")).reset_index()
    say("C1. Structural member-CNPJ presence in the time-deposit book (all quarters, stk_time>0):")
    say(f"    conglomerate-quarters with a CDB book              : {len(per_cq):,}")
    say(f"    ... with >=2 member CNPJs holding a CDB book        : {100*(per_cq['n_cnpj'] > 1).mean():.1f}%")
    say(f"    ... with >=2 member CNPJs that are rate-identified   : {100*(per_cq['n_cnpj_rated'] > 1).mean():.1f}%")
    rated_multi = per_cq[per_cq["n_cnpj_rated"] > 1]
    if len(rated_multi):
        say(f"    cross-CNPJ SD of r_cnpj within congl-qtr (bps/qtr), multi-CNPJ-rated only "
            f"(n={len(rated_multi):,}):")
        for q in [0.5, 0.9, 0.99, 1.0]:
            say(f"       p{int(q*100):02d} = {1e4*rated_multi['rate_sd'].quantile(q):.1f}")
        say(f"       mean = {1e4*rated_multi['rate_sd'].mean():.1f}")
    say("    (UPPER BOUND on achievable local rate variation if member footprints were disjoint)\n")

    # --- C2. Realistic ESTBAN V432-weighted municipal-rate dispersion ---
    say("C2. Realistic municipal-rate dispersion (ESTBAN V432-weighted member-CNPJ rates):")
    d_cdb = _weighted_muni_dispersion(fc, "r_cdb", "real-CDB-leaf (2025+)")
    d_lump = _weighted_muni_dispersion(fc, "r_lump", "lump-proxy (all years)")
    say("")
    out = pd.concat([d for d in [d_cdb, d_lump] if len(d)], ignore_index=True) \
        if (len(d_cdb) or len(d_lump)) else per_cq
    return out


# ----------------------------------------------------------------------------
def make_figure(panel: pd.DataFrame, tableC: pd.DataFrame) -> None:
    b = panel[panel["mca_code"].astype(str).str.upper() != "NATIONAL"].copy()
    grp = ["CodConglomeradoPrudencial", "year", "quarter"]
    per = b.groupby(grp).agg(
        n_mca=("mca_code", "nunique"),
        sd_spread=("spread_a4", "std"),
        sd_logdep=("dep_a4", lambda s: np.log1p(s.clip(lower=0)).std())).reset_index()
    per = per[per["n_mca"] > 1]

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax[0].hist(per["sd_spread"].fillna(0), bins=60, color="#c0392b", alpha=.85,
               label="within-SD of spread_a4 (PRICE)")
    ax[0].hist(per["sd_logdep"].fillna(0), bins=60, color="#2980b9", alpha=.55,
               label="within-SD of log dep_a4 (QUANTITY)")
    ax[0].set(title="Within-(congl×qtr) dispersion across MCAs\n(multi-MCA B-type conglomerates)",
              xlabel="within-group standard deviation", ylabel="conglomerate-quarters")
    ax[0].legend(fontsize=8)
    ax[0].text(.5, .55, "PRICE is a spike at 0\n(national broadcast)", transform=ax[0].transAxes,
               fontsize=9, color="#c0392b", ha="center")

    cols = getattr(tableC, "columns", [])
    if tableC is not None and "r_sd" in cols:
        sub = tableC
        if "measure" in cols and (tableC["measure"] == "lump-proxy (all years)").any():
            sub = tableC[tableC["measure"] == "lump-proxy (all years)"]
        vals = 1e4 * sub["r_sd"].dropna()
        ax[1].hist(vals.clip(upper=vals.quantile(0.99)), bins=40, color="#27ae60", alpha=.85)
        ax[1].set(title="Ceiling: within-congl-qtr SD of counterfactual\nmunicipal rate "
                        "(ESTBAN V432-weighted member CNPJs)",
                  xlabel="within-congl SD of municipal rate (bps/qtr)", ylabel="conglomerate-quarters")
        ax[1].axvline(vals.median(), color="k", ls="--", lw=.8)
        ax[1].text(.97, .9, f"median={vals.median():.1f} bps\nmean={vals.mean():.1f} bps",
                   transform=ax[1].transAxes, ha="right", fontsize=8)
    else:
        ax[1].axis("off")
        ax[1].text(.5, .5, "Stage C2 not computable\n(no ESTBAN×rated-CNPJ overlap);\nsee C1 structural ceiling",
                   ha="center", va="center", fontsize=10)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "fig_k4_local_variation.png"
    fig.tight_layout(); fig.savefig(out, dpi=130)
    say(f"\nWrote figure -> {out}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    say("K=4 (CDB / time-deposit) SPREAD — LOCAL vs NATIONAL DIAGNOSTIC")
    say(f"panel : {PANEL_CSV}")
    usecols = ["CodConglomeradoPrudencial", "CNPJ_Lider", "CODMUN_IBGE", "mca_code",
               "year", "quarter", "dep_a4", "spread_a4", "rate_a4", "spread_ann_a4"]
    have = pd.read_csv(PANEL_CSV, nrows=0).columns
    panel = pd.read_csv(PANEL_CSV, usecols=[c for c in usecols if c in have],
                        dtype={"mca_code": str}, low_memory=False)
    for c in ["dep_a4", "spread_a4", "rate_a4", "spread_ann_a4"]:
        if c in panel.columns:
            panel[c] = pd.to_numeric(panel[c], errors="coerce")

    tableA = stage_a(panel)
    tableB = stage_b(panel)
    tableC = stage_c(panel)
    make_figure(panel, tableC)

    tableA.to_csv(OUT_DIR / "tableA_variance_decomp.csv", index=False)
    tableB.to_csv(OUT_DIR / "tableB_geography.csv", index=False)
    if tableC is not None:
        tableC.to_csv(OUT_DIR / "tableC_member_cnpj_ceiling.csv", index=False)
    (OUT_DIR / "report_k4_spread_local.txt").write_text("\n".join(_report_lines), encoding="utf-8")
    say(f"\nWrote report + tables -> {OUT_DIR}")


if __name__ == "__main__":
    main()

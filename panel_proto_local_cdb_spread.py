"""
panel_proto_local_cdb_spread.py
==================================
SEPARATE-PANEL PROTOTYPE (Option 1).  Does NOT touch market_panel.csv or any
production demand parquet.  Builds a parallel type-4 demand panel in which the
CDB spread carries GENUINE local (MCA) variation, then runs the real weak-IV /
demand battery on it to see how the data behaves.

The local variation is REAL, not injected: for each conglomerate we take its
member CNPJs' own COSIF CDB rates (r_cnpj = |exp_cdb|/stk_time_lag, real 2025+
leaf; |exp_lump|/stk_total_lag proxy pre-2025) and weight them by each CNPJ's
ESTBAN V432 (time-deposit) balance in each municipality.  The only thing added to
the panel's calibrated national rate is the WITHIN-conglomerate deviation

    dev_qoq(congl, mca, t) = r_mca(congl, mca, t) − r_congl(congl, t)

so the national level/identification is untouched; we add exactly the local
price variation Option 1 would provide and nothing else.

Outputs (processed/ESTIMATION_OUTPUT/DIAG_K4_SPREAD/):
  proto1_local_spread_panel.parquet   type-4 rows + spread_ann_national/_local
  proto1_sensitivity.csv              α̂, SE, first-stage F, partial R² tables
  report_proto1.txt

Usage:  python panel_proto_local_cdb_spread.py [--routine 4]
"""
from __future__ import annotations
import os, sys, glob, argparse
os.environ.setdefault("MPLBACKEND", "Agg")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import numpy as np
import pandas as pd

from utils import paths
import panel_cosif_calibrate as cc
import blp_weak_iv as wia   # reuse the validated battery (import has no side effects)

FOUNDATION = paths.PROCESSED / "COSIF_PROCESSED" / "custos_implicitos_v2_foundation.csv"
ESTBAN_CSV = paths.ESTBAN_CSV
MARKET_CSV = paths.PROCESSED / "market_panel.csv"
MCA_XWALK = paths.IBGE_DIR / "muni_mca_regions_2010_2024_panel.csv"
DEMAND_PREP = paths.PROCESSED / "ESTIMATION_OUTPUT" / "DEMAND_PREP"
OUT_DIR = paths.PROCESSED / "ESTIMATION_OUTPUT" / "DIAG_K4_SPREAD"
RATE_Q_CAP = 0.5
_lines: list[str] = []


def say(m: str = ""):
    try:
        print(m)
    except UnicodeEncodeError:
        print(m.encode("ascii", "replace").decode())
    _lines.append(m)


# ── member-CNPJ V432-weighted local CDB rate → within-conglomerate deviation ──
def build_local_deviation() -> pd.DataFrame:
    """Return (CodConglomeradoPrudencial, mca_code, year, quarter) with
    dev_qoq = r_mca − r_congl (qoq fraction) and n_cnpj (member CNPJs in the mca)."""
    f = pd.read_csv(FOUNDATION, dtype={"CNPJ": str})
    f["CNPJ"] = f["CNPJ"].str.strip().str.zfill(8)
    f["AnoMes"] = pd.to_numeric(f["AnoMes"], errors="coerce")
    for c in ["exp_cdb", "exp_lump", "stk_time_lag", "stk_total_lag"]:
        f[c] = pd.to_numeric(f[c], errors="coerce")

    def _rate(num, den):
        r = num.abs() / den.replace(0.0, np.nan)
        return r.where(np.isfinite(r) & (r > 0) & (r <= RATE_Q_CAP))

    f["r_cnpj"] = _rate(f["exp_cdb"], f["stk_time_lag"]).fillna(_rate(f["exp_lump"], f["stk_total_lag"]))
    f = f.dropna(subset=["AnoMes", "r_cnpj"]).assign(AnoMes=lambda d: d["AnoMes"].astype(int))

    cmap = cc.build_cnpj_to_congl()
    fc = cc.map_foundation_to_congl(f[["CNPJ", "AnoMes", "r_cnpj"]], cmap)[
        ["CNPJ", "AnoMes", "CodConglomeradoPrudencial", "r_cnpj"]]

    say(f"member-CNPJ rates: {len(fc):,} CNPJ-quarters, "
        f"{fc['CodConglomeradoPrudencial'].nunique():,} conglomerates, "
        f"years {fc['AnoMes'].min()//100}-{fc['AnoMes'].max()//100}")

    est = pd.read_csv(ESTBAN_CSV, usecols=["CNPJ", "CODMUN_IBGE", "YEAR", "MONTH", "V432"],
                      dtype={"CNPJ": str}, low_memory=False, encoding="latin-1")
    est = est[est["MONTH"].isin([3, 6, 9, 12])].copy()
    est["CNPJ"] = est["CNPJ"].str.strip().str.zfill(8)
    est["V432"] = pd.to_numeric(est["V432"], errors="coerce").fillna(0.0).clip(lower=0)
    est["CODMUN_IBGE"] = pd.to_numeric(est["CODMUN_IBGE"], errors="coerce")
    est = est.dropna(subset=["CODMUN_IBGE"])
    est["CODMUN_IBGE"] = est["CODMUN_IBGE"].astype(int)
    est["AnoMes"] = (est["YEAR"].astype(int) * 100 + est["MONTH"].astype(int)).astype(int)
    est["year"] = est["YEAR"].astype(int)
    est["quarter"] = (est["MONTH"] // 3).astype(int)

    xw = pd.read_csv(MCA_XWALK, usecols=["municipality_code", "year", "mca_code"])
    xw = xw.sort_values("year").drop_duplicates("municipality_code", keep="last")
    xw["mca_code"] = xw["mca_code"].astype(str)
    est = est.merge(xw[["municipality_code", "mca_code"]],
                    left_on="CODMUN_IBGE", right_on="municipality_code", how="inner")

    est = est.merge(fc, on=["CNPJ", "AnoMes"], how="inner")
    est = est[est["V432"] > 0]

    est["wr"] = est["V432"] * est["r_cnpj"]
    C, M = "CodConglomeradoPrudencial", "mca_code"
    mca = est.groupby([C, M, "year", "quarter"], as_index=False).agg(
        wr=("wr", "sum"), w=("V432", "sum"), n_cnpj=("CNPJ", "nunique"))
    mca["r_mca"] = mca["wr"] / mca["w"]
    congl = est.groupby([C, "year", "quarter"], as_index=False).agg(wr=("wr", "sum"), w=("V432", "sum"))
    congl["r_congl"] = congl["wr"] / congl["w"]
    out = mca.merge(congl[[C, "year", "quarter", "r_congl"]], on=[C, "year", "quarter"], how="left")
    out["dev_qoq"] = out["r_mca"] - out["r_congl"]
    out[C] = out[C].astype(str)
    out["mca_code"] = out["mca_code"].astype(str)
    say(f"local deviation table: {len(out):,} (congl x mca x quarter) cells; "
        f"non-zero |dev|>1bp: {100*(out['dev_qoq'].abs() > 1e-4).mean():.1f}%")
    return out[[C, "mca_code", "year", "quarter", "dev_qoq", "n_cnpj"]]


# ── build the separate type-4 panel with national vs local spread ────────────
def build_panel(routine: int) -> pd.DataFrame:
    fs = sorted(glob.glob(str(DEMAND_PREP / f"demand_{routine}_*spec_12.parquet")), key=os.path.getmtime)
    if not fs:
        raise FileNotFoundError(f"no demand parquet for routine {routine}")
    need = list(dict.fromkeys(
        ["spread_ann", "share_D", "share_B_cond", "is_B", "deposit_type", "mca_code",
         "CodConglomeradoPrudencial", "year", "quarter"] + wia.X_COLS + wia.IV_COLS))
    df = pd.read_parquet(fs[-1], columns=None)
    df = df[df["deposit_type"].astype(int) == 4].copy()
    df["CodConglomeradoPrudencial"] = df["CodConglomeradoPrudencial"].astype(str)
    df["mca_code"] = df["mca_code"].astype(str)
    say(f"type-4 rows from {os.path.basename(fs[-1])}: {len(df):,}")

    # national base rate (qoq) from market_panel to annualise the deviation consistently
    mp = pd.read_csv(MARKET_CSV, usecols=["CodConglomeradoPrudencial", "mca_code", "year", "quarter", "rate_a4"],
                     dtype={"mca_code": str}, low_memory=False)
    mp["CodConglomeradoPrudencial"] = mp["CodConglomeradoPrudencial"].astype(str)
    df = df.merge(mp, on=["CodConglomeradoPrudencial", "mca_code", "year", "quarter"], how="left")

    dev = build_local_deviation()
    df = df.merge(dev, on=["CodConglomeradoPrudencial", "mca_code", "year", "quarter"], how="left")
    df["dev_qoq"] = df["dev_qoq"].fillna(0.0)
    df["n_cnpj"] = df["n_cnpj"].fillna(1)

    # national spread (annual fraction) exactly as weak_iv builds it
    df["spread_nat"] = df["spread_ann"].fillna(0.0) / 100.0
    # local = national − (annualised local rate deviation); base rate = panel rate_a4 (qoq)
    r0 = pd.to_numeric(df["rate_a4"], errors="coerce").fillna(0.0)
    dev_ann = (1.0 + r0 + df["dev_qoq"]) ** 4 - (1.0 + r0) ** 4
    df["spread_loc"] = df["spread_nat"] - dev_ann
    df["dev_ann_bps"] = 1e4 * dev_ann
    return df


def _matrices(df: pd.DataFrame):
    is_B = df["is_B"].fillna(False).astype(bool).to_numpy()
    sD = df["share_D"].fillna(0.0).to_numpy(float)
    sB = df["share_B_cond"].fillna(0.0).to_numpy(float)
    delta = np.where(is_B, np.log(np.clip(sB, 1e-15, None)), np.log(np.clip(sD, 1e-15, None)))
    X = np.column_stack([np.ones(len(df))] +
                        [df[c].fillna(0.0).to_numpy(float) if c in df else np.zeros(len(df)) for c in wia.X_COLS])
    iv_avail = [c for c in wia.IV_COLS if c in df.columns]
    Z = np.column_stack([np.nan_to_num(df[c].to_numpy(float), posinf=0.0, neginf=0.0) for c in iv_avail])
    clus = df["CodConglomeradoPrudencial"].astype(str).to_numpy()
    return delta, X, Z, clus, iv_avail


def evaluate(df: pd.DataFrame, label: str, rows: list):
    delta, X, Z, clus, iv_avail = _matrices(df)
    grp = df.groupby(["CodConglomeradoPrudencial", "year", "quarter"])
    within_sd_nat = grp["spread_nat"].transform("std").fillna(0.0)
    within_sd_loc = grp["spread_loc"].transform("std").fillna(0.0)
    for tag, spread in [("national", df["spread_nat"].to_numpy(float)),
                        ("local", df["spread_loc"].to_numpy(float))]:
        ok = np.isfinite(delta) & np.isfinite(spread) & np.all(np.isfinite(X), 1) & np.all(np.isfinite(Z), 1)
        d, s, Xo, Zo, cl = delta[ok], spread[ok], X[ok], Z[ok], clus[ok]
        zmask = Zo.std(0) > 1e-10
        Zo = Zo[:, zmask]
        # drop zero-variance controls (e.g. fgc_covered≡1 for type-4) — collinear with const
        Xo = Xo[:, np.r_[True, Xo[:, 1:].std(0) > 1e-10]] if Xo.shape[1] > 1 else Xo
        uniq, codes = np.unique(cl, return_inverse=True); G = len(uniq)
        fsF = wia._fs_F(s, Xo, Zo, codes, G)
        rec = {"n_obs": int(ok.sum()), "n_clusters": G}
        rec = wia._add_linearmodels(rec, d, s, Xo, Zo, cl)
        rows.append(dict(sample=label, spread=tag, n_obs=rec["n_obs"], G=G,
                         alpha=rec.get("alpha_2sls"), se=rec.get("alpha_se"),
                         eff_F=fsF["eff_F"], kp_F=fsF["kp_F"], partial_R2=fsF["partial_R2"],
                         within_sd_spread=float((within_sd_loc if tag == "local" else within_sd_nat)[ok].mean())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routine", type=int, default=4)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    say("=" * 78)
    say("PROTO 1 — separate type-4 panel with member-CNPJ LOCAL CDB spread (REAL data)")
    say("=" * 78)
    df = build_panel(args.routine)

    tail = df[df["dev_ann_bps"].abs() > 0.5]
    say(f"\nrows with a non-trivial local deviation (|dev|>0.5 bps/yr): {len(tail):,} "
        f"({100*len(tail)/len(df):.2f}% of type-4 rows)")
    say(f"deviation size (bps/yr) among those: mean|dev|={tail['dev_ann_bps'].abs().mean():.1f}, "
        f"p99={tail['dev_ann_bps'].abs().quantile(.99):.1f}, max={tail['dev_ann_bps'].abs().max():.1f}")

    rows: list = []
    say("\nRunning weak-IV / 2SLS battery: national spread vs local spread ...")
    evaluate(df, "all type-4", rows)
    if len(tail) > 200:
        # conglomerates that ever get a deviation → the 'exposed' subsample
        exposed_congl = set(tail["CodConglomeradoPrudencial"])
        sub = df[df["CodConglomeradoPrudencial"].isin(exposed_congl)]
        evaluate(sub, "multi-CNPJ congls", rows)

    res = pd.DataFrame(rows)
    say("")
    say(f"{'sample':20s} {'spread':9s} {'N':>8s} {'G':>4s} {'alpha':>9s} {'se':>8s} "
        f"{'eff_F':>8s} {'KP_F':>8s} {'partR2':>8s} {'within_sd':>10s}")
    for _, r in res.iterrows():
        def fx(x, d=4): return f"{x:.{d}f}" if isinstance(x, (int, float)) and x is not None and np.isfinite(x) else "NA"
        say(f"{r['sample']:20s} {r['spread']:9s} {r['n_obs']:8,d} {r['G']:4d} "
            f"{fx(r['alpha']):>9s} {fx(r['se']):>8s} {fx(r['eff_F'],2):>8s} {fx(r['kp_F'],2):>8s} "
            f"{fx(r['partial_R2']):>8s} {fx(r['within_sd_spread']):>10s}")

    # headline deltas
    say("\nHEADLINE (Δ from swapping national → local spread):")
    for samp in res["sample"].unique():
        n = res[(res["sample"] == samp) & (res["spread"] == "national")].iloc[0]
        l = res[(res["sample"] == samp) & (res["spread"] == "local")].iloc[0]
        da = (l["alpha"] - n["alpha"]) if (n["alpha"] is not None and l["alpha"] is not None) else np.nan
        rel = 100 * da / abs(n["alpha"]) if (n["alpha"] not in (None, 0) and np.isfinite(da)) else np.nan
        say(f"  [{samp}] Δalpha = {da:+.5f} ({rel:+.2f}% of national alpha);  "
            f"within-congl spread SD: {n['within_sd_spread']:.4f} → {l['within_sd_spread']:.4f} (annual frac)")

    df_keep = df[["CodConglomeradoPrudencial", "mca_code", "year", "quarter", "deposit_type",
                  "share_B_cond", "is_B", "spread_ann", "spread_nat", "spread_loc",
                  "dev_ann_bps", "n_cnpj"]]
    df_keep.to_parquet(OUT_DIR / "proto1_local_spread_panel.parquet", index=False)
    res.to_csv(OUT_DIR / "proto1_sensitivity.csv", index=False)
    (OUT_DIR / "report_proto1.txt").write_text("\n".join(_lines), encoding="utf-8")
    say(f"\nWrote separate panel + sensitivity -> {OUT_DIR}")


if __name__ == "__main__":
    main()

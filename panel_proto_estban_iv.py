"""
panel_proto_estban_iv.py
===================================
SEPARATE-PANEL PROTOTYPE (Option 3).  Does NOT touch market_panel.csv.  Builds
candidate ESTBAN-derived instruments for the NATIONAL type-4 deposit spread and
measures their first-stage strength with the validated weak-IV battery, so we can
see whether local ESTBAN structure helps identify the (national) markdown better
than the existing instrument set.

IDENTIFICATION GRAIN (why congl x quarter).  The endogenous spread is constant
across the MCAs of a conglomerate-quarter (proved: 0% within-variance).  So an
instrument must vary at (conglomerate x quarter) to have matching price variation;
MCA-level detail is noise to the first stage.  Every candidate here is therefore
built per (congl, mca, quarter) from ESTBAN and then collapsed to (congl, quarter)
as an OWN-V432-weighted mean across the conglomerate's MCAs, lagged one quarter and
z-standardised — matching the national spread's grain.

Candidates (all leave-one-out / rival-based, BLP differentiation-IV logic):
  iv_rival_v432        log1p rival time-deposit balances in the MCA (balance analog
                       of the adopted estban_rival_branches_lag)
  iv_rival_v432_hhi    HHI of RIVAL time-deposit shares in the MCA (competitor concentration)
  iv_rival_branch_hhi  HHI of RIVAL branch shares in the MCA
  iv_own_geo_hhi       HHI of OWN V432 across the conglomerate's MCAs (geographic concentration)
  iv_own_branch_dens   own branches per unit own V432 (scale/cost shifter)

Outputs (processed/ESTIMATION_OUTPUT/DIAG_K4_SPREAD/):
  proto3_local_instruments.parquet    (congl x quarter) instrument values
  proto3_first_stage.csv              first-stage strength table
  report_proto3.txt

Usage:  python panel_proto_estban_iv.py [--routine 4]
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
import blp_weak_iv as wia

ESTBAN_CSV = paths.ESTBAN_CSV
MCA_XWALK = paths.IBGE_DIR / "muni_mca_regions_2010_2024_panel.csv"
DEMAND_PREP = paths.PROCESSED / "ESTIMATION_OUTPUT" / "DEMAND_PREP"
OUT_DIR = paths.PROCESSED / "ESTIMATION_OUTPUT" / "DIAG_K4_SPREAD"
KEYS = ["CodConglomeradoPrudencial", "year", "quarter"]
NEW_IVS = ["iv_rival_v432", "iv_rival_v432_hhi", "iv_rival_branch_hhi",
           "iv_own_geo_hhi", "iv_own_branch_dens"]
_lines: list[str] = []


def say(m: str = ""):
    try:
        print(m)
    except UnicodeEncodeError:
        print(m.encode("ascii", "replace").decode())
    _lines.append(m)


def _hhi_rivals(values: np.ndarray) -> np.ndarray:
    """For each element i, HHI of the shares of the OTHER elements (leave-one-out)."""
    tot = values.sum()
    out = np.zeros_like(values, dtype=float)
    for i in range(len(values)):
        rest = tot - values[i]
        if rest > 0:
            sh = values[np.arange(len(values)) != i] / rest
            out[i] = float((sh ** 2).sum())
    return out


def build_instruments() -> pd.DataFrame:
    say("Reading ESTBAN (CNPJ x muni x quarter: branches + V432)")
    est = pd.read_csv(ESTBAN_CSV, usecols=["CNPJ", "CODMUN_IBGE", "YEAR", "MONTH",
                                           "AGEN_PROCESSADAS", "V432"],
                      dtype={"CNPJ": str}, low_memory=False, encoding="latin-1")
    est = est[est["MONTH"].isin([3, 6, 9, 12])].copy()
    est["CNPJ"] = est["CNPJ"].str.strip().str.zfill(8)
    est["CODMUN_IBGE"] = pd.to_numeric(est["CODMUN_IBGE"], errors="coerce")
    est = est.dropna(subset=["CODMUN_IBGE"])
    est["CODMUN_IBGE"] = est["CODMUN_IBGE"].astype(int)
    est["br"] = pd.to_numeric(est["AGEN_PROCESSADAS"], errors="coerce").fillna(0.0).clip(lower=0)
    est["v432"] = pd.to_numeric(est["V432"], errors="coerce").fillna(0.0).clip(lower=0)
    est["AnoMes"] = (est["YEAR"].astype(int) * 100 + est["MONTH"].astype(int)).astype("int64")
    est["year"] = est["YEAR"].astype(int)
    est["quarter"] = (est["MONTH"] // 3).astype(int)

    uniq = est[["CNPJ", "AnoMes"]].drop_duplicates()
    cmap = cc.map_foundation_to_congl(uniq, cc.build_cnpj_to_congl())[
        ["CNPJ", "AnoMes", "CodConglomeradoPrudencial"]]
    est = est.merge(cmap, on=["CNPJ", "AnoMes"], how="inner")

    xw = pd.read_csv(MCA_XWALK, usecols=["municipality_code", "year", "mca_code"])
    xw = xw.sort_values("year").drop_duplicates("municipality_code", keep="last")
    xw["mca_code"] = xw["mca_code"].astype(str)
    est = est.merge(xw[["municipality_code", "mca_code"]],
                    left_on="CODMUN_IBGE", right_on="municipality_code", how="inner")

    C, M = "CodConglomeradoPrudencial", "mca_code"
    own = est.groupby([C, M, "year", "quarter"], as_index=False).agg(
        own_br=("br", "sum"), own_v432=("v432", "sum"))

    say("Building (congl x mca x quarter) rival aggregates + HHIs")
    frames = []
    for (yr, q), g in own.groupby(["year", "quarter"], sort=False):
        g = g.reset_index(drop=True)
        for mcode, gm in g.groupby(M, sort=False):
            v = gm["own_v432"].to_numpy(float)
            b = gm["own_br"].to_numpy(float)
            tv, tb = v.sum(), b.sum()
            rec = gm[[C, M, "year", "quarter", "own_br", "own_v432"]].copy()
            rec["rival_v432"] = tv - v
            rec["rival_v432_hhi"] = _hhi_rivals(v)
            rec["rival_branch_hhi"] = _hhi_rivals(b)
            frames.append(rec)
    mca = pd.concat(frames, ignore_index=True)

    # collapse to (congl, quarter): own-V432-weighted mean across the conglomerate's MCAs
    def wavg(df, col, wcol="own_v432"):
        w = df[wcol].to_numpy(float)
        x = df[col].to_numpy(float)
        sw = w.sum()
        return float((x * w).sum() / sw) if sw > 0 else float(np.nan)

    rows = []
    for (c, yr, q), g in mca.groupby([C, "year", "quarter"], sort=False):
        v = g["own_v432"].to_numpy(float)
        vg = v / v.sum() if v.sum() > 0 else v
        rows.append({
            C: c, "year": yr, "quarter": q,
            "iv_rival_v432": np.log1p(wavg(g, "rival_v432")),
            "iv_rival_v432_hhi": wavg(g, "rival_v432_hhi"),
            "iv_rival_branch_hhi": wavg(g, "rival_branch_hhi"),
            "iv_own_geo_hhi": float((vg ** 2).sum()),                      # own geographic concentration
            "iv_own_branch_dens": float(g["own_br"].sum() / (g["own_v432"].sum() + 1.0)),
        })
    cq = pd.DataFrame(rows)
    cq[C] = cq[C].astype(str)

    # lag one quarter within conglomerate, z-standardise
    cq = cq.sort_values([C, "year", "quarter"])
    for col in NEW_IVS:
        lag = cq.groupby(C)[col].shift(1)
        mu, sd = lag.mean(), lag.std(ddof=0)
        cq[col] = ((lag - mu) / (sd if sd and sd > 0 else 1.0)).fillna(0.0)
    say(f"Built {len(NEW_IVS)} congl-quarter instruments: {len(cq):,} rows, "
        f"years {int(cq['year'].min())}-{int(cq['year'].max())}")
    return cq[KEYS + NEW_IVS]


def fs_eval(df: pd.DataFrame, iv_names: list, label: str) -> dict:
    """First-stage strength of instrument set `iv_names` on the NATIONAL type-4 spread."""
    spread = df["spread_ann"].fillna(0.0).to_numpy(float) / 100.0
    is_B = df["is_B"].fillna(False).astype(bool).to_numpy()
    sD = df["share_D"].fillna(0.0).to_numpy(float)
    sB = df["share_B_cond"].fillna(0.0).to_numpy(float)
    delta = np.where(is_B, np.log(np.clip(sB, 1e-15, None)), np.log(np.clip(sD, 1e-15, None)))
    X = np.column_stack([np.ones(len(df))] +
                        [df[c].fillna(0.0).to_numpy(float) if c in df else np.zeros(len(df)) for c in wia.X_COLS])
    avail = [c for c in iv_names if c in df.columns]
    Z = np.column_stack([np.nan_to_num(df[c].to_numpy(float), posinf=0.0, neginf=0.0) for c in avail])
    clus = df["CodConglomeradoPrudencial"].astype(str).to_numpy()
    ok = np.isfinite(delta) & np.isfinite(spread) & np.all(np.isfinite(X), 1) & np.all(np.isfinite(Z), 1)
    d, s, Xo, Zo, cl = delta[ok], spread[ok], X[ok], Z[ok], clus[ok]
    Zo = Zo[:, Zo.std(0) > 1e-10]
    Xo = Xo[:, np.r_[True, Xo[:, 1:].std(0) > 1e-10]] if Xo.shape[1] > 1 else Xo
    if Zo.shape[1] == 0:
        return dict(set=label, n_iv=0)
    uniq, codes = np.unique(cl, return_inverse=True); G = len(uniq)
    fsF = wia._fs_F(s, Xo, Zo, codes, G)
    a, se = wia._iv2sls_alpha(d, s, Xo, Zo, codes, G)
    return dict(set=label, n_obs=int(ok.sum()), G=G, n_iv=int(Zo.shape[1]),
                eff_F=fsF["eff_F"], kp_F=fsF["kp_F"], partial_R2=fsF["partial_R2"],
                k_eff=fsF["k_eff"], alpha=a, se=se)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routine", type=int, default=4)
    ap.add_argument("--reuse", action="store_true", help="reuse cached instrument parquet")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    say("=" * 78)
    say("PROTO 3 — separate ESTBAN local-instrument battery for the NATIONAL type-4 spread")
    say("=" * 78)

    cache = OUT_DIR / "proto3_local_instruments.parquet"
    if cache.exists() and args.reuse:
        say(f"Reusing cached instruments {cache.name}")
        inst = pd.read_parquet(cache)
        inst["CodConglomeradoPrudencial"] = inst["CodConglomeradoPrudencial"].astype(str)
    else:
        inst = build_instruments()
        inst.to_parquet(cache, index=False)

    fs = sorted(glob.glob(str(DEMAND_PREP / f"demand_{args.routine}_*spec_12.parquet")), key=os.path.getmtime)
    df = pd.read_parquet(fs[-1])
    df = df[df["deposit_type"].astype(int) == 4].copy()
    df["CodConglomeradoPrudencial"] = df["CodConglomeradoPrudencial"].astype(str)
    df = df.merge(inst, on=KEYS, how="left")
    for c in NEW_IVS:
        df[c] = df[c].fillna(0.0)
    say(f"type-4 rows: {len(df):,}; merged {len(NEW_IVS)} local instruments\n")

    existing = [c for c in wia.IV_COLS if c in df.columns]
    pars = [c for c in wia.PARSIMONIOUS_IV if c in df.columns]
    tests = [
        ("baseline (existing IV set)", existing),
        ("existing estban_rival_branches ALONE", ["estban_rival_branches_lag"]),
        ("baseline + local ESTBAN (5)", existing + NEW_IVS),
        ("baseline + rival_branch_hhi (best new)", existing + ["iv_rival_branch_hhi"]),
        ("parsimonious loo (5)", pars),
        ("parsimonious loo + local (10)", pars + NEW_IVS),
        ("parsimonious loo + rival_branch_hhi", pars + ["iv_rival_branch_hhi"]),
        ("local ESTBAN ONLY (5)", NEW_IVS),
    ] + [(f"  alone: {iv}", [iv]) for iv in NEW_IVS]

    recs = [fs_eval(df, ivs, name) for name, ivs in tests]
    res = pd.DataFrame(recs)

    say(f"{'instrument set':40s} {'K':>3s} {'eff_F':>8s} {'KP_F':>9s} {'partR2':>8s} "
        f"{'alpha':>9s} {'se':>8s}")
    for _, r in res.iterrows():
        def fx(x, d=4): return f"{x:.{d}f}" if isinstance(x, (int, float)) and x is not None and np.isfinite(x) else "NA"
        say(f"{r['set']:40s} {int(r.get('n_iv') or 0):3d} {fx(r.get('eff_F'),2):>8s} "
            f"{fx(r.get('kp_F'),2):>9s} {fx(r.get('partial_R2')):>8s} "
            f"{fx(r.get('alpha')):>9s} {fx(r.get('se')):>8s}")
    say("")
    say("READING: eff_F is the cluster-robust Montiel-Olea–Pflueger effective F (weak if < ~10-23).")
    say("Compare 'baseline' vs 'baseline + local ESTBAN' (does F/partial-R² rise?) and each local")
    say("instrument ALONE vs the existing branch instrument alone.")

    res.to_csv(OUT_DIR / "proto3_first_stage.csv", index=False)
    (OUT_DIR / "report_proto3.txt").write_text("\n".join(_lines), encoding="utf-8")
    say(f"\nWrote instruments + first-stage table -> {OUT_DIR}")


if __name__ == "__main__":
    main()

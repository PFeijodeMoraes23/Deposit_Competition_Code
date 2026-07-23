"""
panel_10b_estban_rival_chars.py
===============================
Branch-overlap-weighted rival CHARACTERISTIC instruments, at the
CONGLOMERATE x QUARTER grain.

WHY THIS GRAIN (read before "improving" it).  The endogenous deposit spread is
NATIONAL: it is built at CodConglomeradoPrudencial x deposit_type x AnoMes and is
*exactly constant across the MCAs* of a conglomerate-quarter (verified:
std(spread across MCA within congl x type x quarter) = 0 in 100% of multi-MCA
cells; panel_6_market.py:300-302 documents it). So an MCA-level instrument has no
matching price variation to identify off of. The only variation the spread has is
between conglomerates and over time — so an instrument must live at (conglomerate
x quarter) to help. panel_10's ESTBAN branch COUNT already helps only via its
conglomerate-quarter aggregate; this file generalises that from a count to the
branch-overlap-weighted average of RIVAL characteristics (the Berry-Levinsohn-Pakes
differentiation-IV idea, but weighted by geographic/branch overlap rather than
equal cell membership, which is what may make it less collinear than the plain
mean_loo block).

CONSTRUCTION.  For conglomerate c, quarter t, and each characteristic X:
    rival_X_{c,t} = Σ_{c'≠c} w_{c,c',t} · X_{c',t} / Σ_{c'≠c} w_{c,c',t}
    w_{c,c',t}    = Σ_mca  br_{c,mca,t} · br_{c',mca,t}    (branch-overlap product)
i.e. rivals that share more branches with c across space get more weight. Branches
br come from ESTBAN AGEN_PROCESSADAS summed to (conglomerate, mca, quarter);
characteristics X come from market_panel (conglomerate-level, lagged). Each column
is lagged one quarter and cross-sectionally standardised.

The 5 characteristics mirror the LOO set: log_total_assets, equity_ratio,
indice_basileia, credit_assets, npl_provision. Output columns:
    estban_rival_{log_assets,equity_ratio,basileia,credit_assets,npl_provision}_wtd_lag

PIPELINE PLACEMENT.  Run AFTER panel_7 (LOO) and panel_10 (branch count), BEFORE
panel_9 --patch-market. Merges 5 columns onto market_panel keyed on
(CodConglomeradoPrudencial, year, quarter) — broadcast across the conglomerate's
MCAs, matching the national spread grain. Idempotent.

Usage:  python panel_10b_estban_rival_chars.py           # write into market_panel.csv
        (or import build_estban_rival_chars() for a diagnostic preview)
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd

from utils import paths
import cosif_process_2_calibrate as cc

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

PANEL_CSV = paths.PROCESSED / "market_panel.csv"
ESTBAN_CSV = paths.ESTBAN_CSV
MCA_XWALK = paths.IBGE_DIR / "muni_mca_regions_2010_2024_panel.csv"

# (output_suffix, market_panel source column) — mirrors panel_7's LOO characteristics.
CHARS = [
    ("log_assets",     "log_total_assets_lag"),
    ("equity_ratio",   "equity_ratio_lag"),
    ("basileia",       "indice_basileia_lag"),
    ("credit_assets",  "credit_assets_lag"),
    ("npl_provision",  "npl_provision_ratio_lag"),
]
OUT_COLS = [f"estban_rival_{name}_wtd_lag" for name, _ in CHARS]
KEYS = ["CodConglomeradoPrudencial", "year", "quarter"]


def _own_branches() -> pd.DataFrame:
    """(conglomerate, mca_code, year, quarter) -> own_br  (ESTBAN branches). Mirrors panel_10."""
    est = pd.read_csv(ESTBAN_CSV, usecols=["CNPJ", "CODMUN_IBGE", "YEAR", "MONTH", "AGEN_PROCESSADAS"],
                      dtype={"CNPJ": str}, low_memory=False, encoding="latin-1")
    est = est[est["MONTH"].isin([3, 6, 9, 12])].copy()
    est["CNPJ"] = est["CNPJ"].str.strip().str.zfill(8)
    est["CODMUN_IBGE"] = pd.to_numeric(est["CODMUN_IBGE"], errors="coerce")
    est = est.dropna(subset=["CODMUN_IBGE"])
    est["CODMUN_IBGE"] = est["CODMUN_IBGE"].astype(int)
    est["AGEN_PROCESSADAS"] = pd.to_numeric(est["AGEN_PROCESSADAS"], errors="coerce").fillna(0.0)
    est["AnoMes"] = (est["YEAR"].astype(int) * 100 + est["MONTH"].astype(int)).astype("int64")
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

    own = est.groupby(["CodConglomeradoPrudencial", "mca_code", "YEAR", "quarter"],
                      as_index=False)["AGEN_PROCESSADAS"].sum().rename(
        columns={"AGEN_PROCESSADAS": "own_br", "YEAR": "year"})
    own["CodConglomeradoPrudencial"] = own["CodConglomeradoPrudencial"].astype(str)
    return own


def _congl_chars() -> pd.DataFrame:
    """(conglomerate, year, quarter) -> the 5 lagged characteristics (conglomerate-level)."""
    src = [c for _, c in CHARS]
    mp = pd.read_csv(PANEL_CSV, low_memory=False,
                     usecols=["CodConglomeradoPrudencial", "year", "quarter"] + src)
    mp["CodConglomeradoPrudencial"] = mp["CodConglomeradoPrudencial"].astype(str)
    # characteristics are constant within (congl, year, quarter); take the first non-null.
    ch = mp.groupby(["CodConglomeradoPrudencial", "year", "quarter"], as_index=False)[src].first()
    return ch


def build_estban_rival_chars() -> pd.DataFrame:
    """Return (CodConglomeradoPrudencial, year, quarter) + the 5 estban_rival_*_wtd_lag columns."""
    logging.info("Building branch-overlap-weighted rival characteristics (congl x quarter)")
    own = _own_branches()
    ch = _congl_chars()
    src = [c for _, c in CHARS]

    frames = []
    for (yr, q), g in own.groupby(["year", "quarter"]):
        # branch matrix B: rows = conglomerates present this quarter, cols = MCAs.
        B = g.pivot_table(index="CodConglomeradoPrudencial", columns="mca_code",
                          values="own_br", aggfunc="sum", fill_value=0.0)
        congls = B.index.to_numpy()
        Bm = B.to_numpy(dtype=float)
        W = Bm @ Bm.T                                  # (nc x nc) branch-overlap product
        np.fill_diagonal(W, 0.0)                        # leave-one-out own

        # align characteristics to this quarter's conglomerate order
        chq = ch[(ch["year"] == yr) & (ch["quarter"] == q)].set_index("CodConglomeradoPrudencial")
        chq = chq.reindex(congls)
        out = {"CodConglomeradoPrudencial": congls, "year": yr, "quarter": q}
        for name, col in CHARS:
            x = chq[col].to_numpy(dtype=float)         # may contain NaN
            valid = np.isfinite(x)
            if valid.sum() == 0:
                out[f"estban_rival_{name}_wtd"] = np.full(len(congls), np.nan)
                continue
            Wv = W[:, valid]
            num = Wv @ x[valid]
            den = Wv.sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[f"estban_rival_{name}_wtd"] = np.where(den > 0, num / den, np.nan)
        frames.append(pd.DataFrame(out))

    wide = pd.concat(frames, ignore_index=True)

    # lag one quarter within conglomerate, then z-standardise each column cross-sectionally.
    wide = wide.sort_values(["CodConglomeradoPrudencial", "year", "quarter"])
    for name, _ in CHARS:
        raw = f"estban_rival_{name}_wtd"
        lagged = wide.groupby("CodConglomeradoPrudencial")[raw].shift(1)
        mu, sd = lagged.mean(), lagged.std(ddof=0)
        wide[f"estban_rival_{name}_wtd_lag"] = (lagged - mu) / (sd if sd and sd > 0 else 1.0)
    out = wide[KEYS + OUT_COLS].copy()
    for c in OUT_COLS:
        out[c] = out[c].fillna(0.0)                    # no-rival / pre-sample -> 0 (standardised mean)
    logging.info(f"Built {len(OUT_COLS)} rival-char instruments: {len(out):,} congl-quarters, "
                 f"years {int(out['year'].min())}-{int(out['year'].max())}")
    return out


def main() -> None:
    if not PANEL_CSV.exists():
        raise FileNotFoundError(f"market_panel.csv not found at {PANEL_CSV} — run panel_7 first.")
    inst = build_estban_rival_chars()

    panel = pd.read_csv(PANEL_CSV, low_memory=False)
    panel["CodConglomeradoPrudencial"] = panel["CodConglomeradoPrudencial"].astype(str)
    inst["CodConglomeradoPrudencial"] = inst["CodConglomeradoPrudencial"].astype(str)
    drop = [c for c in OUT_COLS if c in panel.columns]
    if drop:
        panel = panel.drop(columns=drop)
    merged = panel.merge(inst, on=KEYS, how="left")
    for c in OUT_COLS:
        merged[c] = merged[c].fillna(0.0)
    merged.to_csv(PANEL_CSV, index=False)
    logging.info(f"Wrote {PANEL_CSV} with {OUT_COLS}")


if __name__ == "__main__":
    main()

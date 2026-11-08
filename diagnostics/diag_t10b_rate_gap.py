"""
diag_t10b_rate_gap.py
=====================
T10b (claude/MAIN_REPLY_2026-11-05a.md, Addendum 13:00): the B versus D deposit-rate gap with
inference, and the digital issuers' posted rates against the panel's rates, on two panels: the
production market_panel.parquet and a read-only copy of an earlier one. Reads only; writes three
files to Rout.

1. Table 2 reproduced. make_desc_compressed_tables.build_table4 (the D firms' pre- versus
   post-Pix comparison that writes Compressed_DFirm_PrePost_Pix_*.csv and the
   desc.t2.spread_ann_a4.* keys) run on each panel through its own prepare_panel; the
   spread_ann_a4 row (D firm-quarters with time deposits, post minus pre, CRV1 by conglomerate,
   p from t(G*)), plus the wild cluster bootstrap of the same regression.
2. The gap. One row per conglomerate-quarter (2016-2024; type 5 from its first reported
   balance), where the conglomerate holds the type and the panel has its rate: the type's rate as
   a percent of the CDI (both compounded to annual rates) and the spread over the Selic in pp, on
   a D indicator with quarter effects; unweighted and weighted by the type's stock. Pooled, and by
   year (D x year indicators, every year's gap; joint tests that all are zero and that all are
   equal). Standard errors CRV1 by conglomerate; p values and 95 percent intervals from the
   restricted wild cluster bootstrap (Webb weights, B draws, null imposed; intervals by test
   inversion); G and the effective number of clusters G* = G / (1 + cv^2) of the cluster sizes,
   the formula Table 2 uses.
3. Posted against panel (descriptive, no inference). Type 5: the balance rules of
   config/type5_balance_rates.csv and the 100-percent balance rules that file leaves to the rate
   chain (posted_rates.csv rows below). Type 4: the digital issuers' own CDB/RDB rules and
   disclosed stock averages (posted_rates.csv rows below). Issuers are mapped to conglomerates by
   the Pix ISPB map (DATA_RAW/BCB/pix_participants/compiled/ispb_conglomerate_map.csv, by name);
   PicPay is C0088022 in every quarter in the panel (utils/firm_remaps.py), so the map's
   time-invariant code applies as it is. Beside them, seven B banks' type-4 rates as a percent of
   the CDI by year, over statement-sourced quarters (rate_a4_src = statement) and over all.

Outputs (Rout)
  t10b_rate_gap.csv          every estimate: panel, block, type, measure, weight, year, coef,
                             se_crv1, p_wcb, ci_lo, ci_hi, stat, G, G_star, N, n_B, n_D, mean_B,
                             mean_D
  t10b_posted_vs_panel.csv   issuer x quarter: posted rule against the panel's type-4 and type-5
                             rates on both panels, and the B banks' rates by year
  t10b_rate_gap_note.md      the tables, with the inputs' md5s

Usage:  python -I diagnostics/diag_t10b_rate_gap.py [--draws 999] [--seed 20261105]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: utils/
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp

import make_desc_compressed_tables as mdt
from utils import paths
from utils.window import MIN_YEAR, MAX_YEAR

PANELS = {
    "fa38794d": paths.PROCESSED / "market_panel.parquet",
    "0b9385b5": Path(r"C:\Users\pedro\AppData\Local\Temp\claude\e04f72edbe14fbd1"
                     r"\21a30b32-2059-49bd-aaac-15d47097282e\scratchpad\rebuild_1104"
                     r"\panel_0b9385b5\market_panel.parquet"),
}
COLS = ["CodConglomeradoPrudencial", "NomeInstituicao", "CODMUN_IBGE", "is_B", "year", "quarter",
        "dep_a1", "dep_a2", "dep_a4", "dep_a5", "total_assets", "equity_ratio", "spread_a4",
        "spread_a5", "risk_free_qoq", "rate_a4_national", "rate_a5", "rate_a5_posted",
        "rate_a4_src", "rate_a4_fallback"]
MACRO = paths.PROCESSED / "PANEL_INTERMED" / "quarterly_macro_rates.csv"
T5_RULES = Path(__file__).resolve().parents[1] / "config" / "type5_balance_rates.csv"
OUT = paths.rout_dir()
WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])

# Conglomerates of the issuers, from the ISPB map (nome -> conglomerate).
CONGL = {"Nubank": "C0084693", "Mercado Pago": "C0084820", "PagBank": "C0084813",
         "PicPay": "C0088022", "Neon": "C0085702", "C6": "C0084844", "iti (Itau)": "C0080099",
         "Bradesco (bitz, Next)": "C0080075"}
# Balance rules (type 5) at 100 percent of the CDI or more that config/type5_balance_rates.csv
# leaves to the rate chain: (posted_rates.csv row, issuer, lo, hi, start, end, note).
T5_EXTRA = [
    (3, "Nubank", 100, 100, "2019-05-30", "2022-07-24", "100% of CDI on all balance"),
    (4, "Nubank", 100, 100, "2022-07-25", "2024-12-31", "100% of CDI from day 30, retroactive"),
    (11, "Mercado Pago", 100, 100, "2020-09-23", "2024-03-04", "Rende 100% do CDI"),
    (12, "Mercado Pago", 100, 105, "2024-04-05", "2024-12-31", "ate 105% do CDI (conditions not saved)"),
    (15, "PagBank", 100, 100, "2019-10-07", "2020-06-06", "109-110% of poupanca = 100% of CDI, 30 days"),
    (17, "PagBank", 5, 100, "2020-09-06", "2021-03-02", "100% up to R$1,000 held 30 days, 5% above"),
    (19, "PagBank", 100, 100, "2021-06-05", "2024-09-02", "100% of CDI, no cap, 30 days"),
    (56, "iti (Itau)", 100, 100, "2021-11-05", "2023-07-24", "rende 100% do CDI"),
    (142, "iti (Itau)", 0, 100, "2023-10-05", "2024-12-31", "main balance 0, Metas 100% (split unknown)"),
    (89, "Bradesco (bitz, Next)", 100, 100, "2021-05-13", "2022-12-10", "bitz: above R$10, 100% of CDI"),
]
# Own CDB/RDB rules and disclosed averages (type 4): (row, issuer, lo, hi, start, end, kind, note).
T4_RULES = [
    (59, "Nubank", 100, 100, "2019-06-01", "2024-12-31", "posted", "RDB daily liquidity 100% of CDI"),
    (66, "Nubank", 102, 126, "2020-12-31", "2020-12-31", "range", "scheduled RDB range 3-36 months"),
    (66, "Nubank", 102, 126, "2021-09-30", "2021-12-31", "range", "scheduled RDB range 3-36 months"),
    (67, "Nubank", 107, 107, "2021-12-31", "2021-12-31", "stock average", "scheduled RDB weighted average"),
    (67, "Nubank", 104, 104, "2022-12-31", "2022-12-31", "stock average", "scheduled RDB weighted average"),
    (68, "Nubank", 105, 106, "2023-12-31", "2023-12-31", "stock average", "scheduled RDB weighted average (FY2024 / FY2023 20-F)"),
    (69, "Nubank", 105, 105, "2024-12-31", "2024-12-31", "stock average", "scheduled RDB weighted average"),
    (20, "PagBank", np.nan, 170, "2020-09-06", "2020-09-06", "ceiling", "CDBs 'ate 170% do CDI'"),
    (21, "PagBank", 164, 164, "2020-12-31", "2020-12-31", "stock average", "CDB stock average (20-F)"),
    (23, "PagBank", 163, 163, "2021-12-31", "2021-12-31", "stock average", "CDB stock average (20-F)"),
    (25, "PagBank", 117, 117, "2022-12-31", "2022-12-31", "stock average", "CDB stock average (20-F)"),
    (27, "PagBank", 110, 110, "2023-12-31", "2023-12-31", "stock average", "CDB stock average (20-F)"),
    (41, "Neon", 80, 80, "2016-12-04", "2018-02-05", "posted", "CDB (Objetivos) 80% of CDI"),
    (42, "Neon", 100, 100, "2018-06-24", "2018-06-24", "posted", "CDB 100% of CDI"),
    (43, "Neon", 95, 101, "2022-05-16", "2022-05-17", "posted", "progressive CDB 95 -> 101"),
    (44, "Neon", 100, 104, "2023-03-09", "2023-03-26", "posted", "progressive CDB 100 -> 104"),
    (45, "Neon", 105, 113, "2023-11-26", "2023-12-07", "posted", "progressive CDB 105 -> 113"),
    (46, "Neon", 100, 113, "2024-06-14", "2024-06-14", "posted", "ladder 100-113 by holding time"),
    (49, "C6", 100, 125, "2020-06-26", "2020-06-26", "range", "CDB 100 (daily) to 125 (60 months)"),
    (50, "C6", 104, 104, "2024-02-29", "2024-02-29", "posted", "daily-liquidity CDB (blog)"),
    (50, "C6", 102.5, 103.5, "2024-11-09", "2024-11-09", "posted", "daily-liquidity CDB (blog)"),
    (73, "Bradesco (bitz, Next)", np.nan, 103, "2023-01-10", "2023-05-30", "ceiling", "Next CDB Fidelidade 'ate 103%'"),
    (74, "Bradesco (bitz, Next)", np.nan, 102, "2023-09-21", "2024-01-06", "ceiling", "Next CDB Fidelidade 'ate 102%'"),
    (75, "Bradesco (bitz, Next)", np.nan, 101, "2024-06-23", "2024-11-06", "ceiling", "Next CDB Fidelidade 'ate 101%'"),
]
# B banks whose type-4 rate the addendum lists, by name in the panel.
B_BANKS = {"BB": r"^BCO DO BRASIL|^BANCO DO BRASIL", "Caixa": r"CAIXA ECONOMICA",
           "Bradesco": r"^BCO BRADESCO|^BANCO BRADESCO", "Santander": r"SANTANDER",
           "BRB": r"^BRB|BCO DE BRASILIA", "Banese": r"BANESE|BCO DO EST\. DE SE|SERGIPE",
           "Banestes": r"BANESTES"}


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------- inference
class WCB:
    """y on X with fixed effects `fe` (absorbed), optional weights, CRV1 by cluster; restricted
    wild cluster bootstrap (Webb) for one coefficient (with test-inversion intervals) and for
    joint linear restrictions R beta = 0."""

    def __init__(self, y, X, fe, cl, w=None, draws=999, seed=0):
        y = np.asarray(y, float)
        X = np.asarray(X, float).reshape(len(y), -1)
        n, K = X.shape
        self.s = np.sqrt(np.asarray(w, float)) if w is not None else np.ones(n)
        fe_u, fe_inv = np.unique(np.asarray(fe), return_inverse=True)
        cl_u, self.cl_inv = np.unique(np.asarray(cl).astype(str), return_inverse=True)
        Q, self.G = len(fe_u), len(cl_u)
        self.F = sp.csr_matrix((self.s, (fe_inv, np.arange(n))), shape=(Q, n))
        self.fe_inv, self.sw = fe_inv, np.bincount(fe_inv, weights=self.s ** 2)
        self.C = sp.csr_matrix((np.ones(n), (self.cl_inv, np.arange(n))), shape=(self.G, n))
        self.yt = self.mf(self.s * y)
        self.Xt = self.mf(self.s[:, None] * X)
        self.A = np.linalg.inv(self.Xt.T @ self.Xt)
        self.beta = self.A @ (self.Xt.T @ self.yt)
        self.e = self.yt - self.Xt @ self.beta
        self.c = self.G / (self.G - 1) * (n - 1) / (n - K - Q)
        S = self.C @ (self.Xt * self.e[:, None])
        self.V = self.c * self.A @ (S.T @ S) @ self.A
        self.v = WEBB[np.random.default_rng(seed).integers(0, 6, size=(self.G, draws))]
        sizes = np.bincount(self.cl_inv)
        self.G_star = float(self.G / (1.0 + (sizes.std() / sizes.mean()) ** 2))
        self.n, self.K = n, K

    def mf(self, Z):
        """Annihilate the fixed effects in the sqrt(w)-scaled space."""
        Z = np.asarray(Z, float)
        two = Z.ndim == 2
        Z2 = Z if two else Z[:, None]
        m = (self.F @ Z2) / self.sw[:, None]
        out = Z2 - self.s[:, None] * m[self.fe_inv]
        return out if two else out[:, 0]

    def _boot(self, ur):
        Z = ur[:, None] * self.v[self.cl_inv]
        dev = self.A @ (self.Xt.T @ Z)                      # beta_b - beta_restricted
        E = self.mf(Z) - self.Xt @ dev
        S = np.stack([np.asarray(self.C @ (self.Xt[:, [k]] * E)) for k in range(self.K)], axis=1)
        return dev, S                                       # K x B, G x K x B

    def coef(self, k, level=0.95):
        """(beta_k, CRV1 se, WCB p for beta_k = 0, interval by inverting the WCB test)."""
        b, se = float(self.beta[k]), float(np.sqrt(self.V[k, k]))
        ak = self.A[:, k]
        h = self.Xt @ ak / self.A[k, k]
        da, Sa = self._boot(self.e + h * b)                 # restricted residual at r: (e + h b) - r h
        db, Sb = self._boot(-h)
        n0, n1 = da[k], db[k]
        qa, qb = np.einsum("gkb,k->gb", Sa, ak), np.einsum("gkb,k->gb", Sb, ak)
        Q00, Q01, Q11 = (qa * qa).sum(0), (qa * qb).sum(0), (qb * qb).sum(0)

        def pval(r):
            t2 = (b - r) ** 2 / se ** 2
            tb2 = (n0 + r * n1) ** 2 / (self.c * (Q00 + 2 * r * Q01 + r * r * Q11))
            return float(np.mean(tb2 >= t2))

        alpha = 1.0 - level

        def edge(sign):
            step = max(se, 1e-6)
            inner, outer = b, b + sign * step
            for _ in range(60):
                if pval(outer) < alpha:
                    break
                inner, outer = outer, outer + sign * step
                step *= 2.0
            else:
                return np.nan
            for _ in range(60):
                mid = 0.5 * (inner + outer)
                if pval(mid) < alpha:
                    outer = mid
                else:
                    inner = mid
            return 0.5 * (inner + outer)
        return b, se, pval(0.0), edge(-1.0), edge(1.0)

    def joint(self, R):
        """(Wald statistic, WCB p) for R beta = 0."""
        R = np.atleast_2d(np.asarray(R, float))
        RAR = R @ self.A @ R.T
        beta_r = self.beta - self.A @ R.T @ np.linalg.solve(RAR, R @ self.beta)
        Rb = R @ self.beta
        W = float(Rb @ np.linalg.solve(R @ self.V @ R.T, Rb))
        dev, S = self._boot(self.yt - self.Xt @ beta_r)
        RA = R @ self.A
        T = np.einsum("jk,gkb->gjb", RA, S)
        RVR = self.c * np.einsum("gjb,gib->bji", T, T)
        Rd = R @ dev
        Wb = np.einsum("jb,bji,ib->b", Rd, np.linalg.inv(RVR), Rd)
        return W, float(np.mean(Wb >= W))


# ------------------------------------------------------------------------------- data
def load(tag: str, path: Path):
    got = md5(path)
    if not got.startswith(tag):
        raise SystemExit(f"{path} has md5 {got}, not {tag}")
    df = pq.read_table(path, columns=COLS).to_pandas()
    return mdt.prepare_panel(df), got


def cdi_table() -> pd.DataFrame:
    m = pd.read_csv(MACRO)
    m["year"], m["quarter"] = m["AnoMes"] // 100, (m["AnoMes"] % 100) // 3
    return m[["year", "quarter", "cdi_qoq"]]


def congl_quarters(df: pd.DataFrame) -> pd.DataFrame:
    """One row per conglomerate-quarter with the type-4 and type-5 rates as a percent of the CDI
    and the spreads over the Selic in pp, all compounded to annual rates."""
    k = ["CodConglomeradoPrudencial", "year", "quarter"]
    g = df.groupby(k, as_index=False).agg(
        name=("NomeInstituicao", "first"), bank_type=("bank_type", "first"),
        dep_a4=("dep_a4", "sum"), dep_a5=("dep_a5", "sum"), r4=("rate_a4_national", "first"),
        r5=("rate_a5", "first"), r5_n=("rate_a5", "nunique"), src=("rate_a4_src", "first"),
        fb=("rate_a4_fallback", "first"), posted5=("rate_a5_posted", "first"),
        rf=("risk_free_qoq", "first"))
    g = g.merge(cdi_table(), on=["year", "quarter"], how="left")
    cdi = g["cdi_qoq"] / 100.0
    for t in ("4", "5"):
        r = g[f"r{t}"]
        g[f"pcdi{t}"] = 100.0 * ((1 + r) ** 4 - 1) / ((1 + cdi) ** 4 - 1)
        g[f"sp{t}"] = 100.0 * ((1 + g["rf"]) ** 4 - (1 + r) ** 4)
    g["t"] = g["year"] * 4 + g["quarter"] - 1
    g["D"] = (g["bank_type"] == "D").astype(float)
    return g


# ------------------------------------------------------------------------------- estimates
def table2(tag, df, draws, seed):
    res, meta = mdt.build_table4(df)
    r = res.set_index("var").loc["spread_ann_a4"]
    d = df[df["bank_type"] == "D"]
    y = mdt.annualized_spread_pp(d["spread_a4"], d["risk_free_qoq"]).where(d["dep_a4"] > 0)
    ok = y.notna()
    f = WCB(y[ok], d.loc[ok, "post"], np.zeros(int(ok.sum())), d.loc[ok, "CodConglomeradoPrudencial"],
            draws=draws, seed=seed)
    b, se, p, lo, hi = f.coef(0)
    return dict(panel=tag, block="table2", type="4", measure="spread_pp", weight="none", year="all",
                coef=b, se_crv1=se, p_wcb=p, ci_lo=lo, ci_hi=hi, stat=np.nan, G=f.G, G_star=f.G_star,
                N=f.n, n_B=0, n_D=f.n, mean_B=r["pre_mean"], mean_D=r["post_mean"],
                t2_diff=r["diff"], t2_se=r["se"], t2_p=r["p"], t2_G=r["G"], t2_G_star=r["G_star"],
                t2_n_pre=r["n_pre"], t2_n_post=r["n_post"])


def gaps(tag, g, draws, seed):
    rows = []
    for typ in ("4", "5"):
        s0 = g[(g[f"dep_a{typ}"] > 0) & g[f"r{typ}"].notna()].copy()
        for measure, col in (("pct_cdi", f"pcdi{typ}"), ("spread_pp", f"sp{typ}")):
            s = s0[np.isfinite(s0[col])]
            for weight in ("none", "stock"):
                w = s[f"dep_a{typ}"].to_numpy() if weight == "stock" else None
                ww = w if w is not None else np.ones(len(s))
                base = dict(panel=tag, type=typ, measure=measure, weight=weight)

                def means(m):
                    x, z = s.loc[m, col].to_numpy(), ww[m.to_numpy()]
                    return float(np.average(x, weights=z)) if len(x) else np.nan
                f = WCB(s[col], s["D"], s["t"], s["CodConglomeradoPrudencial"], w=w, draws=draws, seed=seed)
                b, se, p, lo, hi = f.coef(0)
                isD = s["D"] == 1
                rows.append(dict(**base, block="pooled", year="all", coef=b, se_crv1=se, p_wcb=p,
                                 ci_lo=lo, ci_hi=hi, stat=np.nan, G=f.G, G_star=f.G_star, N=f.n,
                                 n_B=int((~isD).sum()), n_D=int(isD.sum()), mean_B=means(~isD),
                                 mean_D=means(isD)))
                yrs = [y for y in sorted(s["year"].unique())
                       if ((s["year"] == y) & isD).any() and ((s["year"] == y) & ~isD).any()]
                sy = s[s["year"].isin(yrs)]
                wy = sy[f"dep_a{typ}"].to_numpy() if weight == "stock" else None
                X = np.column_stack([(sy["D"] * (sy["year"] == y)).to_numpy() for y in yrs])
                fy = WCB(sy[col], X, sy["t"], sy["CodConglomeradoPrudencial"], w=wy, draws=draws, seed=seed)
                isDy = sy["D"] == 1
                wwy = wy if wy is not None else np.ones(len(sy))
                for j, y in enumerate(yrs):
                    b, se, p, lo, hi = fy.coef(j)
                    my = (sy["year"] == y).to_numpy()
                    mB, mD = my & ~isDy.to_numpy(), my & isDy.to_numpy()
                    rows.append(dict(**base, block="by_year", year=str(y), coef=b, se_crv1=se, p_wcb=p,
                                     ci_lo=lo, ci_hi=hi, stat=np.nan, G=fy.G, G_star=fy.G_star,
                                     N=int(my.sum()), n_B=int(mB.sum()), n_D=int(mD.sum()),
                                     mean_B=float(np.average(sy[col].to_numpy()[mB], weights=wwy[mB])),
                                     mean_D=float(np.average(sy[col].to_numpy()[mD], weights=wwy[mD]))))
                K = len(yrs)
                for block, R in (("joint_zero", np.eye(K)),
                                 ("joint_equal", np.diff(np.eye(K), axis=0) if K > 1 else None)):
                    if R is None:
                        continue
                    W, p = fy.joint(R)
                    rows.append(dict(**base, block=block, year=f"{yrs[0]}-{yrs[-1]}", coef=np.nan,
                                     se_crv1=np.nan, p_wcb=p, ci_lo=np.nan, ci_hi=np.nan, stat=W,
                                     G=fy.G, G_star=fy.G_star, N=fy.n, n_B=int((~isDy).sum()),
                                     n_D=int(isDy.sum()), mean_B=np.nan, mean_D=np.nan))
    return rows


def quarters(start, end):
    a, b = pd.Timestamp(start), pd.Timestamp(end)
    out, q = [], pd.Period(a, "Q")
    while q <= pd.Period(b, "Q"):
        out.append((q.year, q.quarter))
        q += 1
    return out


def posted_vs_panel(cq: dict) -> pd.DataFrame:
    rules = []
    t5 = pd.read_csv(T5_RULES, comment="#")
    for r in t5.itertuples():
        rules.append(dict(source="config/type5_balance_rates.csv", row=np.nan, issuer=r.issuer,
                          conglomerate=r.conglomerate, type="5", kind=f"panel rule {r.decision}",
                          posted_lo=r.pct_cdi, posted_hi=r.pct_cdi, start=r.start, end=r.end,
                          note=str(r.evidence)[:120]))
    for row, iss, lo, hi, a, b, note in T5_EXTRA:
        rules.append(dict(source="posted_rates.csv", row=row, issuer=iss, conglomerate=CONGL[iss],
                          type="5", kind="posted (chain rate in the panel)", posted_lo=lo,
                          posted_hi=hi, start=a, end=b, note=note))
    for row, iss, lo, hi, a, b, kind, note in T4_RULES:
        rules.append(dict(source="posted_rates.csv", row=row, issuer=iss, conglomerate=CONGL[iss],
                          type="4", kind=kind, posted_lo=lo, posted_hi=hi, start=a, end=b, note=note))
    out = []
    for r in rules:
        for y, q in quarters(r["start"], r["end"]):
            if not MIN_YEAR <= y <= MAX_YEAR:
                continue
            rec = dict(r, year=y, quarter=q)
            for tag, g in cq.items():
                m = g[(g["CodConglomeradoPrudencial"] == r["conglomerate"]) & (g["year"] == y)
                      & (g["quarter"] == q)]
                x = m.iloc[0] if len(m) else None
                t = r["type"]
                rec[f"panel_pcdi{t}_{tag}"] = float(x[f"pcdi{t}"]) if x is not None else np.nan
                rec[f"dep_a{t}_{tag}"] = float(x[f"dep_a{t}"]) if x is not None else np.nan
                rec[f"src_{tag}"] = (x["src"] if t == "4" else x["posted5"]) if x is not None else np.nan
            out.append(rec)
    return pd.DataFrame(out)


def b_bank_rates(cq: dict) -> pd.DataFrame:
    rows = []
    for tag, g in cq.items():
        b = g[(g["bank_type"] == "B") & (g["dep_a4"] > 0) & g["r4"].notna()]
        for bank, pat in B_BANKS.items():
            codes = b.loc[b["name"].astype(str).str.upper().str.contains(pat, regex=True), "CodConglomeradoPrudencial"].unique()
            if len(codes) != 1:
                rows.append(dict(panel=tag, bank=bank, year="all", note=f"{len(codes)} conglomerates match {pat!r}"))
                continue
            x = b[b["CodConglomeradoPrudencial"] == codes[0]]
            for y, xy in x.groupby("year"):
                st = xy[xy["src"] == "statement"]
                rows.append(dict(panel=tag, bank=bank, conglomerate=codes[0], year=int(y),
                                 quarters=len(xy), statement_quarters=len(st),
                                 pcdi4_statement=float(np.average(st["pcdi4"], weights=st["dep_a4"])) if len(st) else np.nan,
                                 pcdi4_all=float(np.average(xy["pcdi4"], weights=xy["dep_a4"])),
                                 src_mix=";".join(f"{k}:{v}" for k, v in xy["src"].fillna("none").value_counts().items())))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------- note
def f3(x):
    """Three decimals with thousands separators; '' for a missing value; never -0.000."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    s = f"{x:,.3f}"
    return s[1:] if s.startswith("-") and float(s.replace(",", "")) == 0.0 else s


def note(est, pv, bb, md5s, draws, seed, checks):
    L = [f"# T10b: the B versus D rate gap, and posted rates against the panel",
         "",
         f"Written {datetime.now():%Y-%m-%d %H:%M} by `diagnostics/diag_t10b_rate_gap.py` (read only). "
         f"Panels: " + "; ".join(f"`{k}` = `{v}`" for k, v in md5s.items()) + ". Wild cluster bootstrap: "
         f"restricted (null imposed), Webb weights, {draws:,} draws, seed {seed}; 95 percent intervals by "
         "inverting the bootstrap test; standard errors CRV1 by conglomerate; G* = G / (1 + cv^2) of the "
         "cluster sizes (Table 2's formula). Rates are compounded to annual rates before the ratio to the "
         "CDI or the difference from the Selic.",
         "",
         "## 1. Table 2 reproduced",
         "",
         "Table 2 (`Compressed_DFirm_PrePost_Pix`, `make_desc_compressed_tables.build_table4`) is the D "
         "firms' own type-4 spread after Pix minus before Pix (D firm-quarters with time deposits). It is "
         "not a B versus D difference; section 2 estimates that gap.",
         "",
         "| panel | pre | post | diff | se (CRV1) | p, t(G*) | G | G* | n pre | n post | WCB p | WCB 95% interval | Rout CSV diff |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in est[est["block"] == "table2"].itertuples():
        L.append(f"| `{r.panel}` | {f3(r.mean_B)} | {f3(r.mean_D)} | {f3(r.t2_diff)} | {f3(r.t2_se)} | {f3(r.t2_p)} | "
                 f"{int(r.t2_G)} | {f3(r.t2_G_star)} | {int(r.t2_n_pre):,} | {int(r.t2_n_post):,} | {f3(r.p_wcb)} | "
                 f"[{f3(r.ci_lo)}, {f3(r.ci_hi)}] | {checks.get(r.panel, '')} |")
    L += ["", "## 2. The B versus D gap: rate (or spread) on a D indicator with quarter effects", "",
          "One row per conglomerate-quarter holding the type with a panel rate. pct_cdi: the rate as a "
          "percent of the CDI; spread_pp: Selic minus the rate, pp a year. coef = D minus B.", "",
          "| panel | type | measure | weight | N (B / D) | mean B | mean D | D - B | se | WCB p | 95% interval | G | G* |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in est[est["block"] == "pooled"].itertuples():
        L.append(f"| `{r.panel}` | {r.type} | {r.measure} | {r.weight} | {r.n_B:,} / {r.n_D:,} | {f3(r.mean_B)} | "
                 f"{f3(r.mean_D)} | {f3(r.coef)} | {f3(r.se_crv1)} | {f3(r.p_wcb)} | [{f3(r.ci_lo)}, {f3(r.ci_hi)}] | "
                 f"{r.G} | {f3(r.G_star)} |")
    L += ["", "### By year (D x year indicators; percent of the CDI)", "",
          "| panel | type | weight | year | n B / D | D - B | se | WCB p | 95% interval |", "|---|---|---|---|---|---|---|---|---|"]
    for r in est[(est["block"] == "by_year") & (est["measure"] == "pct_cdi")].itertuples():
        L.append(f"| `{r.panel}` | {r.type} | {r.weight} | {r.year} | {r.n_B:,} / {r.n_D:,} | {f3(r.coef)} | "
                 f"{f3(r.se_crv1)} | {f3(r.p_wcb)} | [{f3(r.ci_lo)}, {f3(r.ci_hi)}] |")
    L += ["", "### Joint tests of the yearly gaps (WCB, Wald)", "",
          "| panel | type | measure | weight | test | years | Wald | WCB p |", "|---|---|---|---|---|---|---|---|"]
    for r in est[est["block"].str.startswith("joint")].itertuples():
        L.append(f"| `{r.panel}` | {r.type} | {r.measure} | {r.weight} | {r.block} | {r.year} | {f3(r.stat)} | {f3(r.p_wcb)} |")
    L += ["", "## 3. Posted rates against the panel (descriptive; no inference)", "",
          "Issuers mapped to conglomerates by the Pix ISPB map (by name): " + ", ".join(f"{k} {v}" for k, v in CONGL.items())
          + ". Neon's CDBs are issued by Banco Neon; the map gives Neon Pagamentos IP (C0085702). "
          "Type 5: the panel takes the rules of config/type5_balance_rates.csv as its rate (rate_a5_posted = 1); "
          "the 100-percent balance rules are not in that file and the panel keeps the chain rate there. "
          "Type 4: the panel's rate is the conglomerate's national type-4 rate with its rate_a4_src. "
          "Median over the quarters a rule covers; full detail in `t10b_posted_vs_panel.csv`.", "",
          "| issuer | type | kind | posted (% of CDI) | rows | quarters | panel median, fa38794d | panel median, 0b9385b5 | panel source, fa38794d |",
          "|---|---|---|---|---|---|---|---|---|"]
    for (iss, typ, kind, lo, hi, row, a, b), x in pv.groupby(["issuer", "type", "kind", "posted_lo", "posted_hi", "row", "start", "end"], dropna=False, sort=False):
        post = f"{lo:g}" if lo == hi else (f"up to {hi:g}" if pd.isna(lo) else f"{lo:g} to {hi:g}")
        src = ";".join(f"{k}:{v}" for k, v in x["src_fa38794d"].fillna("none").astype(str).value_counts().items())
        L.append(f"| {iss} | {typ} | {kind} | {post} | {'' if pd.isna(row) else int(row)} | {a} to {b} ({len(x)}) | "
                 f"{f3(x['panel_pcdi' + typ + '_fa38794d'].median())} | {f3(x['panel_pcdi' + typ + '_0b9385b5'].median())} | {src} |")
    L += ["", "### Seven B banks: type-4 rate as a percent of the CDI by year (stock-weighted over quarters)", "",
          "| panel | bank | year | quarters (statement) | statement-sourced | all quarters | sources |", "|---|---|---|---|---|---|---|"]
    for r in bb.itertuples():
        if r.year == "all":
            L.append(f"| `{r.panel}` | {r.bank} | | | | | {r.note} |")
            continue
        L.append(f"| `{r.panel}` | {r.bank} | {r.year} | {r.quarters} ({r.statement_quarters}) | {f3(r.pcdi4_statement)} | "
                 f"{f3(r.pcdi4_all)} | {r.src_mix} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=999)
    ap.add_argument("--seed", type=int, default=20261105)
    args = ap.parse_args()
    est, cq, md5s, checks = [], {}, {}, {}
    for tag, path in PANELS.items():
        print(f"{datetime.now():%H:%M:%S} {tag}: reading {path}")
        df, md5s[tag] = load(tag, path)
        est.append(table2(tag, df, args.draws, args.seed))
        cq[tag] = congl_quarters(df)
        print(f"  rate_a5 values per conglomerate-quarter, max {int(cq[tag]['r5_n'].max())}")
        est += gaps(tag, cq[tag], args.draws, args.seed)
        print(f"{datetime.now():%H:%M:%S} {tag}: estimates done")
    t2csv = OUT / "Compressed_DFirm_PrePost_Pix_unweighted.csv"
    if t2csv.is_file():
        r = pd.read_csv(t2csv).set_index("var").loc["spread_ann_a4"]
        checks["fa38794d"] = f"{r['diff']:.3f} (se {r['se']:.3f}, p {r['p']:.3f})"
    est = pd.DataFrame(est)
    pv = posted_vs_panel(cq)
    bb = b_bank_rates(cq)
    est.to_csv(OUT / "t10b_rate_gap.csv", index=False)
    pd.concat([pv.assign(table="posted_vs_panel"), bb.assign(table="b_banks_by_year")]).to_csv(
        OUT / "t10b_posted_vs_panel.csv", index=False)
    (OUT / "t10b_rate_gap_note.md").write_text(note(est, pv, bb, md5s, args.draws, args.seed, checks),
                                               encoding="utf-8", newline="\n")
    print(f"wrote {OUT / 't10b_rate_gap.csv'}, t10b_posted_vs_panel.csv, t10b_rate_gap_note.md")


if __name__ == "__main__":
    main()

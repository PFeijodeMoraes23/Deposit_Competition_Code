"""
diag_estban_expense_rate.py
===========================
READ-ONLY diagnostic.  Does NOT modify any panel and writes no file.

QUESTION.  ESTBAN carries, next to the municipal balances, the municipal RESULT
accounts (V711 revenues, V712 expenses).  Can V712 price time deposits by market,
which the conglomerate-level type-4 rate cannot do?

WHAT THIS SCRIPT ESTABLISHES.
  Stage A.  Validation by institution.  Municipality x quarter panel, 2016Q1-2024Q4,
            municipality and quarter fixed effects, SE clustered by municipality:

               expense / liabilities = sum_k c_k * share_k
                                       + thS * (savings rate * savings share)
                                       + thT * (Selic * time-deposit share) + FE

            The savings rate is administered and identical everywhere, so thS = 1 if
            the institution books deposit interest where the deposit sits.  thT is
            then the time-deposit rate as a fraction of Selic.

  Stage B.  Local variation, for the institutions that pass Stage A: thT by group of
            municipalities (macro-region; number of institutions in the municipality),
            with a Wald test of equality across groups.

ACCOUNT MECHANICS.  V712 is stored NEGATIVE and accumulates within the semester,
restarting in January and July (June and December are not zeroed).  The quarterly
flow is the balance in Q1 and Q3 and the increase over the previous quarter in Q2
and Q4.  It bundles ALL expenses, so V712 / V432 is not a rate: only the regression
coefficient is.  One coefficient across all of an institution's municipalities is a
national number; variation across markets exists only across the groups of Stage B.

Usage:  python diagnostics/diag_estban_expense_rate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: utils/ and pipeline modules
try:
    from utils.venv_guard import ensure_project_venv
    ensure_project_venv(__file__)
except Exception:
    pass

try:                                          # Windows console defaults to cp1252
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
from scipy import stats

from utils import paths

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)

MIN_MUNIS = 150          # median municipalities per quarter for an institution to enter Stage A
MIN_OBS = 3000

USE = ["UF", "CNPJ", "NOME_INSTITUICAO", "CODMUN_IBGE", "YEAR", "MONTH", "V160", "V399", "V400_401", "V420", "V432", "V712", "V899"]
NUM = USE[6:]
e = pd.read_csv(paths.ESTBAN_CSV, usecols=USE, dtype={"CNPJ": str}, low_memory=False, encoding="latin-1")
e = e[e.MONTH.isin([3, 6, 9, 12]) & e.YEAR.between(2015, 2024)].dropna(subset=["CODMUN_IBGE"])
e["CNPJ"] = e.CNPJ.str.strip().str.zfill(8)
for c in NUM:
    e[c] = pd.to_numeric(e[c], errors="coerce").fillna(0.0)
e["CODMUN_IBGE"] = e.CODMUN_IBGE.astype(int)
uf = e.groupby("CODMUN_IBGE").UF.first()
names = e.groupby("CNPJ").NOME_INSTITUICAO.last()
e = e.groupby(["CNPJ", "CODMUN_IBGE", "YEAR", "MONTH"], as_index=False)[NUM].sum()
e["q"] = e.MONTH // 3
e["t"] = e.YEAR * 4 + e.q - 1
ninst = e[e.V399 > 0].groupby(["CODMUN_IBGE", "t"]).CNPJ.nunique().rename("n_inst").reset_index()
e = e.merge(ninst, on=["CODMUN_IBGE", "t"], how="left")
e = e.sort_values(["CNPJ", "CODMUN_IBGE", "t"])
g = e.groupby(["CNPJ", "CODMUN_IBGE"])
for c in ["V160", "V400_401", "V420", "V432", "V899", "V712", "t"]:
    e[c + "_l"] = g[c].shift(1)
e = e[e.t_l == e.t - 1].copy()
# the account restarts in January and July: Q1 and Q3 are the balance, Q2 and Q4 the increase
e["flow"] = np.where(e.q.isin([1, 3]), e.V712.abs(), e.V712.abs() - e.V712_l.abs())
e = e[(e.YEAR >= 2016) & (e.flow > 0) & (e.V899_l > 0)].copy()

# national rates per quarter: demand deposits pay zero, so their spread is Selic
pan = pd.read_parquet(paths.PROCESSED / "market_panel.parquet", columns=["year", "quarter", "spread_a1", "spread_a2"])
nat = pan.groupby(["year", "quarter"]).agg(rf=("spread_a1", "median"), sp2=("spread_a2", "median")).reset_index()
nat["sav"] = nat.rf - nat.sp2
e = e.merge(nat.rename(columns={"year": "YEAR", "quarter": "q"})[["YEAR", "q", "rf", "sav"]], on=["YEAR", "q"], how="inner")
print(f"Selic per quarter: mean {e.rf.mean():.4f}, min {e.rf.min():.4f}, max {e.rf.max():.4f} | savings rate mean {e.sav.mean():.4f}")

REGION = {**{u: "North" for u in "AC AM AP PA RO RR TO".split()}, **{u: "Northeast" for u in "AL BA CE MA PB PE PI RN SE".split()},
          **{u: "Center-West" for u in "DF GO MS MT".split()}, **{u: "Southeast" for u in "ES MG RJ SP".split()},
          **{u: "South" for u in "PR RS SC".split()}}
e["region"] = e.CODMUN_IBGE.map(uf).map(REGION)
e["comp"] = pd.cut(e.n_inst.fillna(1), [0, 1, 2, 4, 1000], labels=["1 bank", "2 banks", "3-4 banks", "5+ banks"]).astype(str)


def demean2(df, cols, g1, g2, iters=100):
    out = df[cols].astype(float).copy()
    for _ in range(iters):
        prev = out.values.copy()
        out = out - out.groupby(df[g1]).transform("mean")
        out = out - out.groupby(df[g2]).transform("mean")
        if np.abs(out.values - prev).max() < 1e-10:
            break
    return out


def ols_cl(y, X, cl):
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ X.T @ y
    u = y - X @ b
    S = pd.DataFrame(X * u[:, None]).groupby(np.asarray(cl)).sum().values
    V = XtXi @ (S.T @ S) @ XtXi
    return b, V


def prep(b):
    b = b.copy()
    b = b[b.V899_l <= b.groupby("t").V899_l.transform(lambda x: x.quantile(.99))]      # drops head office and giants
    b["e"] = b.flow / b.V899_l
    for k, c in [("xD", "V400_401"), ("xS", "V420"), ("xT", "V432"), ("xL", "V160")]:
        b[k] = 0.5 * (b[c] + b[c + "_l"]) / b.V899_l
    lo, hi = b.e.quantile([.01, .99])
    b = b[(b.e >= lo) & (b.e <= hi)]
    for k in ("xD", "xS", "xT", "xL"):
        b = b[b[k] <= b[k].quantile(.995)]
    b["zS"] = b.sav * b.xS
    b["zT"] = b.rf * b.xT
    return b


def fit(b, by=None):
    cols = ["xD", "xS", "xT", "xL", "zS"]
    b = b.copy()
    if by is None:
        zc = ["zT"]
    else:
        zc = []
        for lev in sorted(b[by].dropna().unique()):
            nm = f"zT|{lev}"
            b[nm] = b.zT * (b[by] == lev)
            zc.append(nm)
    dm = demean2(b, ["e"] + cols + zc, "CODMUN_IBGE", "t")
    beta, V = ols_cl(dm["e"].values, dm[cols + zc].values, b.CODMUN_IBGE.values)
    return pd.Series(beta, index=cols + zc), pd.DataFrame(V, index=cols + zc, columns=cols + zc), len(b)


size = e.groupby(["CNPJ", "t"]).CODMUN_IBGE.nunique().groupby("CNPJ").median()
banks = size[size >= MIN_MUNIS].sort_values(ascending=False).index.tolist()
tot_T = e.V432.sum()
rows, keep = [], {}
for c in banks:
    b = prep(e[e.CNPJ == c])
    if len(b) < MIN_OBS:
        continue
    be, V, n = fit(b)
    se = np.sqrt(np.diag(V))
    rows.append({"bank": str(names[c])[:28], "cnpj": c, "obs": n, "munis": b.CODMUN_IBGE.nunique(),
                 "share of ESTBAN time dep %": 100 * e.V432[e.CNPJ == c].sum() / tot_T,
                 "time share of liab (median)": b.xT.median(),
                 "thS (target 1)": be["zS"], "se_S": se[4], "thT (rate/Selic)": be["zT"], "se_T": se[5]})
    keep[c] = b
r = pd.DataFrame(rows)
print("\n=== A. Validation by institution (municipality and quarter fixed effects, SE clustered by municipality) ===")
print(r.round(3).to_string(index=False))

print("\n=== B. Time-deposit rate as a fraction of Selic, by group of municipalities ===")
for c, b in keep.items():
    row = r[r.cnpj == c].iloc[0]
    passed = (abs(row["thS (target 1)"] - 1) < 0.5) and (row["thT (rate/Selic)"] > 0.2) and (row["se_T"] < 0.15)
    if not passed:
        continue
    print(f"\n{row.bank}  (pooled thT {row['thT (rate/Selic)']:.3f}, se {row.se_T:.3f})")
    for by in ("region", "comp"):
        be, V, n = fit(b, by)
        zc = [k for k in be.index if k.startswith("zT|")]
        est = be[zc].values
        se = np.sqrt(np.diag(V.loc[zc, zc].values))
        R = np.hstack([np.ones((len(zc) - 1, 1)), -np.eye(len(zc) - 1)])
        d = R @ est
        W = float(d @ np.linalg.pinv(R @ V.loc[zc, zc].values @ R.T) @ d)
        p = 1 - stats.chi2.cdf(W, len(zc) - 1)
        nobs = [int((b[by] == k.split("|")[1]).sum()) for k in zc]
        txt = "  ".join(f"{k.split('|')[1]}: {v:.2f} ({s:.2f}, n={m:,})" for k, v, s, m in zip(zc, est, se, nobs))
        print(f"  by {by:6s}: {txt}")
        print(f"             equality test p = {p:.4f}; range {est.max() - est.min():.2f} of Selic = {100 * (est.max() - est.min()) * ((1 + e.rf.mean()) ** 4 - 1):.2f} pp a year at the average Selic")

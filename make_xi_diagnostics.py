"""
make_xi_diagnostics.py
======================
Diagnostics on the demand residuals xi-hat of the random-coefficients model (specification 12,
the ext1 fits of the single-index routines E3 and E4): how the residual's variance splits across
quarter, bank, bank x type and market effects, how persistent it is, and how it lines up with two
variables outside the model (incumbency in the market, the bank's own branches there).

xi-hat is read from the saved fit (BLP_RESULTS/cluster_raw/blp_results_E{k}_spec_12_ext1.jls, the
vector "xi" = delta - X theta1, one entry per row of the demand parquet DEMAND_PREP/
demand_{k}_<tag>_spec_12.parquet, in the parquet's row order). Nothing is re-estimated.

Sample: B firms' conglomerate x deposit type x MCA x quarter cells of the demand sample (D firms
sit in one national market, so market effects and within-cell lags are not comparable). Two sets:
time deposits (type 4) and all deposit types. Weights: the cell's deposit balance.

  Panel A  order-free (Shapley) split of the weighted variance. Bank x type is the gain of the
           bank share when bank effects differ by deposit type: Shapley over (quarter, bank x
           type, market) gives the quarter, market and rest shares and the bank x type total; the
           bank share is the bank's Shapley share over (quarter, bank, market), and bank x type
           is the difference. The five shares sum to one, and bank x type is zero for one type.
  Panel B  slope of xi-hat on its own lag over consecutive quarters, within bank x type x market
           cells (cell-demeaned) and without demeaning; cells (14-B) truncates are not in the
           demand sample, so they break the lag.
  Panel C  xi-hat on one variable at a time with quarter effects: incumbency (years since the
           conglomerate's first quarter with a branch in the MCA, ESTBAN from 2013Q1, so first
           quarters at 2013Q1 are left-censored) and own branches (log of one plus the
           conglomerate's ESTBAN branch count in the MCA, lagged one quarter). Variables in x or in
           the instrument set are left out: the estimation makes their sample correlation with
           xi-hat zero.
Every slope: weighted least squares, conglomerate wild cluster bootstrap (Webb, 999 draws),
bias-corrected 95% interval (utils.sleep_links.linear_wild_cluster_bootstrap).

The truncated-cell shares come from the prep's own (14-B) step re-run on the stored sleep fit
(sleep_demand_prep_link.process_specification, summary "Active_truncated"); nothing is written by
that step.

Outputs
  Drafts/Deposit Competition/tab_xi_diagnostics.tex   (label tab:xi_diagnostics)
  Rout/xi_diagnostics.json                            unrounded values, read by make_paper_numbers.py
                                                      (demand.xi.*, demand.flows.trunc_share.*) and
                                                      make_slide_figures.py --only xi
  Rout/xi_cache/                                      xi-hat vectors, the ESTBAN branch panel, the
                                                      truncation counts (rebuilt when a source changes)

Usage:  python make_xi_diagnostics.py [--routines 3 4] [--refresh]
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import argparse
import datetime
import glob
import hashlib
import itertools
import json
import math
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from utils import paths as _paths
from utils import routines as _routines

SPEC = 12
STAGE = "ext1"
BOOT_B = 999
SETS = (("k4", "Time deposits"), ("all", "All types"))
FIRST_ESTBAN_T = 2013 * 4          # 2013Q1 on the quarter index t = 4 * year + quarter - 1
CACHE = _paths.rout_dir() / "xi_cache"
SIDECAR = _paths.rout_dir() / "xi_diagnostics.json"
TEX = _paths.drafts_dir() / "tab_xi_diagnostics.tex"
OWN_BRANCH_SIDECAR = _paths.PROCESSED / "PANEL_INTERMED" / "estban_own_branches.csv"
MCA_XWALK = _paths.IBGE_DIR / "muni_mca_regions_2010_2024_panel.csv"

_JL_DUMP = r'''using Serialization
r = deserialize(ARGS[1])
open(ARGS[2], "w") do io; write(io, Float64.(r["xi"])); end
t1 = Float64.(r["theta1"])
open(ARGS[3], "w") do io
    print(io, "{\"n\": ", length(r["xi"]), ", \"alpha\": ", t1[1], ", \"Q\": ", r["Q_value"],
          ", \"converged\": ", r["converged"], ", \"stage\": \"", r["stage"], "\"}")
end'''


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _stamp(p: Path) -> dict:
    st = p.stat()
    return dict(file=str(p), size=st.st_size,
                mtime=datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"))


# ── inputs ───────────────────────────────────────────────────────────────────
def load_xi(est: int, refresh: bool) -> tuple[np.ndarray, dict]:
    """xi-hat of routine `est` from the saved ext1 fit, through a cached Float64 dump."""
    jls = _paths.blp_results_dir() / "cluster_raw" / f"blp_results_E{est}_spec_{SPEC}_{STAGE}.jls"
    if not jls.is_file():
        sys.exit(f"no saved fit {jls}: xi-hat is not available, nothing to do")
    CACHE.mkdir(parents=True, exist_ok=True)
    vec, meta_p = CACHE / f"xi_E{est}_spec_{SPEC}_{STAGE}.f64", CACHE / f"xi_E{est}_spec_{SPEC}_{STAGE}.json"
    sha = _sha256(jls)
    meta = json.loads(meta_p.read_text()) if meta_p.is_file() else {}
    if refresh or not vec.is_file() or meta.get("sha256") != sha:
        tmp = CACHE / f"_dump_E{est}.json"
        subprocess.run(["julia", "--startup-file=no", "-e", _JL_DUMP, str(jls), str(vec), str(tmp)],
                       check=True)
        meta = json.loads(tmp.read_text())
        tmp.unlink()
        meta.update(sha256=sha, **_stamp(jls))
        meta_p.write_text(json.dumps(meta, indent=2))
    xi = np.fromfile(vec, dtype=np.float64)
    if len(xi) != meta["n"]:
        sys.exit(f"{vec.name}: {len(xi)} values, the fit holds {meta['n']}")
    js = jls.with_suffix(".json")
    if js.is_file():
        a = json.loads(js.read_text())["theta1"][0]
        if abs(a - meta["alpha"]) > 1e-7 * max(1.0, abs(a)):     # the JSON keeps 8 digits
            sys.exit(f"E{est}: alpha {meta['alpha']} in {jls.name} but {a} in {js.name}")
    return xi, meta


def load_frame(est: int, xi: np.ndarray) -> tuple[pd.DataFrame, Path]:
    """The demand parquet's B rows with xi-hat attached by row position."""
    pq = _paths.demand_parquet_dir() / f"{_routines.demand_prefix(est)}_spec_{SPEC}.parquet"
    cols = ["CodConglomeradoPrudencial", "deposit_type", "mca_code", "year", "quarter", "is_B",
            "deposit_balance"]
    d = pd.read_parquet(pq, columns=cols)
    if len(d) != len(xi):
        sys.exit(f"{pq.name} has {len(d):,} rows, the fit's xi {len(xi):,}: fit and parquet differ")
    d = d.reset_index(drop=True)
    d["xi"] = xi
    d = d[d["is_B"].astype(bool)].copy()
    d["bank"] = d["CodConglomeradoPrudencial"].astype(str)
    d["mkt"] = d["mca_code"].astype(str)
    d["k"] = d["deposit_type"].astype(int)
    d["t"] = d["year"].astype(int) * 4 + d["quarter"].astype(int) - 1
    d["w"] = d["deposit_balance"].astype(float)
    if not (np.isfinite(d["xi"]).all() and (d["w"] > 0).all()):
        sys.exit(f"E{est}: non-finite xi-hat or a non-positive balance in the B rows")
    return d[["bank", "mkt", "k", "t", "w", "xi"]].reset_index(drop=True), pq


def estban_branches(refresh: bool) -> pd.DataFrame:
    """Branches per (conglomerate, MCA, quarter) from ESTBAN, 2013Q1 on: the raw monthly files
    before 2016 and ESTBAN.csv after, CNPJ root to conglomerate and municipality to MCA exactly as
    panel_estban_instrument.py maps them. Cached; checked against that step's 2016+ sidecar."""
    out = CACHE / "estban_own_branches_from2013.parquet"
    srcs = [_paths.ESTBAN_CSV] + sorted(Path(p) for p in glob.glob(str(_paths.ESTBAN_RAW_MUN / "*")))
    key = [(_stamp(p)["size"], _stamp(p)["mtime"]) for p in srcs if p.is_file()]
    stamp_p = out.with_suffix(".json")
    if out.is_file() and not refresh and stamp_p.is_file() and json.loads(stamp_p.read_text()) == key:
        return pd.read_parquet(out)
    import panel_deposits as _pdp
    from utils import conglomerates as cc

    frames = []
    seen = set()
    for p in srcs[1:]:
        m = re.match(r"^(\d{4})(\d{2})_ESTBAN", p.name, re.IGNORECASE)
        if not m or int(m.group(1)) >= 2016 or int(m.group(2)) not in (3, 6, 9, 12):
            continue
        if m.group(0).upper() in seen:      # case-insensitive file system: one read per month
            continue
        seen.add(m.group(0).upper())
        r = _pdp.process_raw_estban_csv(str(p))
        frames.append(r[["CNPJ", "CODMUN_IBGE", "YEAR", "MONTH", "AGEN_PROCESSADAS"]])
    raw = pd.concat(frames, ignore_index=True)
    raw["CNPJ"] = pd.to_numeric(raw["CNPJ"], errors="coerce").astype("Int64").astype(str).str.zfill(8)
    new = pd.read_csv(_paths.ESTBAN_CSV, usecols=["CNPJ", "CODMUN_IBGE", "YEAR", "MONTH",
                                                   "AGEN_PROCESSADAS"],
                      dtype={"CNPJ": str}, low_memory=False, encoding="latin-1")
    new = new[new["MONTH"].isin([3, 6, 9, 12])]
    new["CNPJ"] = new["CNPJ"].str.strip().str.zfill(8)
    est = pd.concat([raw, new], ignore_index=True)
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
    # Before 2016 the as-of merge leaves many institutions under their own CNPJ root (Banco do
    # Brasil as 00000000, not its conglomerate code), so their history would not join the panel's
    # conglomerates. A CNPJ root takes the conglomerate it first belongs to from 2016 on.
    later = est[est["AnoMes"] >= 201601].sort_values("AnoMes").drop_duplicates("CNPJ")
    first_congl = later.set_index("CNPJ")["CodConglomeradoPrudencial"]
    pre = est["AnoMes"] < 201601
    est.loc[pre, "CodConglomeradoPrudencial"] = (est.loc[pre, "CNPJ"].map(first_congl)
                                                 .fillna(est.loc[pre, "CodConglomeradoPrudencial"]))
    xw = pd.read_csv(MCA_XWALK, usecols=["municipality_code", "year", "mca_code"])
    xw = xw.sort_values("year").drop_duplicates("municipality_code", keep="last")
    xw["mca_code"] = xw["mca_code"].astype(str)
    est = est.merge(xw[["municipality_code", "mca_code"]], left_on="CODMUN_IBGE",
                    right_on="municipality_code", how="inner")
    own = (est.groupby(["CodConglomeradoPrudencial", "mca_code", "YEAR", "quarter"], as_index=False)
           ["AGEN_PROCESSADAS"].sum()
           .rename(columns={"CodConglomeradoPrudencial": "bank", "mca_code": "mkt", "YEAR": "year",
                            "AGEN_PROCESSADAS": "own_br"}))
    own["bank"] = own["bank"].astype(str)
    own["t"] = own["year"].astype(int) * 4 + own["quarter"].astype(int) - 1
    own = own[["bank", "mkt", "t", "own_br"]]
    linked = own[own["t"] < 2016 * 4]["bank"].str.startswith("C").mean()
    print(f"  ESTBAN branches 2013-2015: {linked:.1%} of cells under a conglomerate code")
    if OWN_BRANCH_SIDECAR.is_file():
        sc = pd.read_csv(OWN_BRANCH_SIDECAR, dtype={"CodConglomeradoPrudencial": str, "mca_code": str})
        sc["t"] = sc["year"] * 4 + sc["quarter"] - 1
        sc = sc.rename(columns={"CodConglomeradoPrudencial": "bank", "mca_code": "mkt"})
        m = own[own["t"] >= 2016 * 4].merge(sc[["bank", "mkt", "t", "own_br"]], how="outer",
                                            on=["bank", "mkt", "t"], suffixes=("", "_sc"))
        agree = float((m["own_br"] == m["own_br_sc"]).mean())
        print(f"  ESTBAN branches 2016+: {agree:.4%} of {len(m):,} cells equal the "
              f"{OWN_BRANCH_SIDECAR.name} sidecar")
        if agree < 0.99:
            sys.exit("the ESTBAN branch panel disagrees with panel_estban_instrument.py's sidecar")
    own.to_parquet(out)
    stamp_p.write_text(json.dumps(key))
    return own


def truncation(est: int, refresh: bool) -> dict:
    """Cells (14-B) floors at zero in the spec-12 demand prep of routine `est`, by firm type and
    deposit type: [dropped, cells]. Re-runs the prep's own step on the stored sleep fit; cached
    on the fit and the market panel."""
    import sleep_demand_prep_link as P
    pkl = _paths.est_dir(est) / "estimation_results.pkl"
    key = dict(pkl=_stamp(pkl), panel=_stamp(Path(P.PANEL_CSV)))
    p = CACHE / f"trunc_E{est}_spec_{SPEC}.json"
    if p.is_file() and not refresh:
        c = json.loads(p.read_text())
        if c.get("key") == key:
            return c["counts"]
    link, _tag, time_block = P.DEMAND_CFG[est]
    df_base = P.build_base_panel(P.PANEL_CSV, time_block=time_block)
    with open(pkl, "rb") as f:
        rd = pickle.load(f)
    name = next(k for k in rd if P.SPEC_MAP[SPEC] in k)
    _df, summary, _ = P.process_specification(name, rd[name], df_base, link)
    counts = summary["Active_truncated"]
    p.write_text(json.dumps(dict(key=key, counts=counts), indent=2))
    return counts


# ── weighted fixed-effect algebra ────────────────────────────────────────────
def _codes(*cols) -> np.ndarray:
    if len(cols) == 1:
        return pd.factorize(cols[0])[0]
    return pd.MultiIndex.from_arrays(list(cols)).factorize()[0]


def _resid(Y: np.ndarray, w: np.ndarray, groups: list, tol=1e-10, maxit=10000) -> np.ndarray:
    """Weighted residuals of the columns of Y on the fixed effects in `groups` (alternating
    projections); with no groups, the weighted mean is removed."""
    Y = np.array(Y, dtype=float, copy=True)
    if Y.ndim == 1:
        Y = Y[:, None]
    if not groups:
        return Y - (w[:, None] * Y).sum(0) / w.sum()
    wsum = [np.bincount(g, weights=w) for g in groups]
    scale = np.sqrt((w[:, None] * Y ** 2).sum(0) / w.sum()) + 1e-300
    for it in range(maxit):
        step = 0.0
        for g, ws in zip(groups, wsum):
            for j in range(Y.shape[1]):
                m = np.bincount(g, weights=w * Y[:, j]) / ws
                Y[:, j] -= m[g]
                step = max(step, float(np.abs(m).max() / scale[j]))
        if len(groups) == 1 or step < tol:
            break
    else:
        sys.exit(f"fixed-effect residuals did not converge in {maxit} sweeps (last step {step:.2e})")
    return Y


def _wr2(y, w, groups) -> float:
    r = _resid(y, w, groups)[:, 0]
    yc = y - np.average(y, weights=w)
    return float(1.0 - np.sum(w * r ** 2) / np.sum(w * yc ** 2))


def _shapley(y, w, factors: dict) -> tuple[dict, float]:
    """Order-free shares of the weighted R2 of y on the fixed effects `factors` (name -> codes),
    and the R2 of all of them together."""
    names = list(factors)
    n = len(names)
    v = {(): 0.0}
    for r in range(1, n + 1):
        for S in itertools.combinations(names, r):
            v[S] = _wr2(y, w, [factors[f] for f in S])
    phi = {}
    for i in names:
        others = [f for f in names if f != i]
        tot = 0.0
        for r in range(n):
            for S in itertools.combinations(others, r):
                Si = tuple(f for f in names if f in S or f == i)
                wgt = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
                tot += wgt * (v[Si] - v[tuple(f for f in names if f in S)])
        phi[i] = tot
    return phi, v[tuple(names)]


def anova(d: pd.DataFrame) -> dict:
    y, w = d["xi"].to_numpy(), d["w"].to_numpy()
    q, b, m = _codes(d["t"]), _codes(d["bank"]), _codes(d["mkt"])
    bt = _codes(d["bank"], d["k"])
    phi_bt, r2 = _shapley(y, w, {"quarter": q, "bank_type": bt, "market": m})
    if d["k"].nunique() == 1:
        bank = phi_bt["bank_type"]
    else:
        bank = _shapley(y, w, {"quarter": q, "bank": b, "market": m})[0]["bank"]
    out = dict(quarter=phi_bt["quarter"], bank=bank, bank_type=phi_bt["bank_type"] - bank,
               market=phi_bt["market"], rest=1.0 - r2)
    assert abs(sum(out.values()) - 1.0) < 1e-9, out
    return out


def wls(yr: np.ndarray, xr: np.ndarray, w: np.ndarray, cl: np.ndarray, name: str) -> dict:
    """Slope of residualised y on residualised x, weighted, with the conglomerate wild cluster
    bootstrap interval."""
    import statsmodels.api as sm
    from utils.sleep_links import linear_wild_cluster_bootstrap
    sw = np.sqrt(w)
    res = sm.OLS(yr * sw, pd.DataFrame({name: xr * sw})).fit(cov_type="cluster",
                                                             cov_kwds={"groups": cl})
    band = {}
    linear_wild_cluster_bootstrap(res, B=BOOT_B, scheme="webb", seed=0, mode="normal",
                                  band_out=band)
    row = band["band"].set_index("name").loc[name]
    r2 = float(np.sum(w * xr * yr) ** 2 / (np.sum(w * xr ** 2) * np.sum(w * yr ** 2)))
    return dict(slope=float(res.params[name]), lo=float(row["lo_bc"]), hi=float(row["hi_bc"]),
                p=float(row["p_bc"]), stars=str(row["stars"]), r2=r2, n=int(len(yr)),
                banks=int(len(np.unique(cl))))


def persistence(d: pd.DataFrame) -> dict:
    d = d.sort_values(["bank", "k", "mkt", "t"]).reset_index(drop=True)
    cell = _codes(d["bank"], d["k"], d["mkt"])
    prev_ok = np.r_[False, (cell[1:] == cell[:-1]) & (d["t"].to_numpy()[1:] == d["t"].to_numpy()[:-1] + 1)]
    lag = np.r_[np.nan, d["xi"].to_numpy()[:-1]]
    p = d[prev_ok].assign(lag=lag[prev_ok], cell=cell[prev_ok])
    cl = p["bank"].to_numpy()
    w = p["w"].to_numpy()
    Y = p[["xi", "lag"]].to_numpy()
    within = _resid(Y, w, [_codes(p["cell"])])
    rw = wls(within[:, 0], within[:, 1], w, cl, "lag")
    raw = _resid(Y, w, [])
    rr = wls(raw[:, 0], raw[:, 1], w, cl, "lag")
    per_cell = p.groupby("cell").size()
    return dict(within=rw, raw=rr, n_pairs=int(len(p)), n_cells=int((per_cell >= 2).sum()),
                n_cells_any=int(len(per_cell)))


def correlates(d: pd.DataFrame, own: pd.DataFrame) -> dict:
    first = (own[own["own_br"] > 0].groupby(["bank", "mkt"])["t"].min().rename("first_t")
             .reset_index())
    x = d.merge(first, on=["bank", "mkt"], how="left")
    x["incumbency"] = np.where(x["first_t"] <= x["t"], (x["t"] - x["first_t"]) / 4.0, np.nan)
    lagged = own.assign(t=own["t"] + 1).rename(columns={"own_br": "own_br_lag"})
    x = x.merge(lagged, on=["bank", "mkt", "t"], how="left")
    present = own.groupby(["bank", "t"]).size().rename("n").reset_index().assign(t=lambda f: f["t"] + 1)
    x = x.merge(present[["bank", "t"]].assign(bank_in_estban=True), on=["bank", "t"], how="left")
    br = np.where(x["own_br_lag"].notna(), x["own_br_lag"],
                  np.where(x["bank_in_estban"].fillna(False), 0.0, np.nan))
    x["own_branches"] = np.log1p(br)
    out = {}
    for v in ("incumbency", "own_branches"):
        s = x[np.isfinite(x[v])]
        w = s["w"].to_numpy()
        R = _resid(s[["xi", v]].to_numpy(), w, [_codes(s["t"])])
        out[v] = wls(R[:, 0], R[:, 1], w, s["bank"].to_numpy(), v)
        out[v]["coverage"] = float(len(s) / len(x))
    inc = x[np.isfinite(x["incumbency"])]
    out["incumbency"]["censored_share"] = float(np.average(inc["first_t"] == FIRST_ESTBAN_T,
                                                           weights=inc["w"]))
    out["incumbency"]["censored_share_cells"] = float((inc["first_t"] == FIRST_ESTBAN_T).mean())
    return out


# ── table ────────────────────────────────────────────────────────────────────
def f3(v) -> str:
    s = f"{v:.3f}"
    return "0.000" if s == "-0.000" else s


def _cell_slope(r) -> str:
    """A slope in math mode (a true minus sign), stars as a superscript, as the other tables."""
    return f"${f3(r['slope'])}^{{{r['stars']}}}$" if r["stars"] else f"${f3(r['slope'])}$"


def _cell_ci(r) -> str:
    return f"[${f3(r['lo'])}$, ${f3(r['hi'])}$]"


def write_table(res: dict, ests: list) -> str:
    cols = [(e, s) for e in ests for s, _ in SETS]
    nc = 1 + len(cols)

    def row(label, fn):
        return f"\\quad {label} & " + " & ".join(fn(res[f'E{e}'][s]) for e, s in cols) + r" \\"

    def panel(title):
        return rf"\multicolumn{{{nc}}}{{@{{}}l}}{{\textit{{{title}}}}} \\"

    head1 = " & " + " & ".join(
        rf"\multicolumn{{2}}{{c}}{{{_routines.ROUTINES[e].tex_name} {_routines.ROUTINES[e].tex_ref}}}"
        for e in ests)
    rules = " ".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(ests)))
    head2 = " & " + " & ".join(lab for _e in ests for _s, lab in SETS)
    body = [
        panel(r"A. Share of the variance of $\hat{\xi}$"),
        row("Quarter", lambda r: f3(r["anova"]["quarter"])),
        row("Bank", lambda r: f3(r["anova"]["bank"])),
        row(r"Bank $\times$ type", lambda r: f3(r["anova"]["bank_type"])),
        row("Market", lambda r: f3(r["anova"]["market"])),
        row("Rest", lambda r: f3(r["anova"]["rest"])),
        r"\addlinespace",
        panel(r"B. Slope of $\hat{\xi}$ on its lag"),
        row(r"Within bank $\times$ type $\times$ market", lambda r: _cell_slope(r["ar1"]["within"])),
        row("", lambda r: _cell_ci(r["ar1"]["within"])),
        row("Without demeaning", lambda r: _cell_slope(r["ar1"]["raw"])),
        row("", lambda r: _cell_ci(r["ar1"]["raw"])),
        row("Pairs of quarters", lambda r: f"{r['ar1']['n_pairs']:,}"),
        row("Cells", lambda r: f"{r['ar1']['n_cells']:,}"),
        r"\addlinespace",
        panel(r"C. Slope of $\hat{\xi}$ on a variable outside the model"),
        row("Incumbency, years", lambda r: _cell_slope(r["corr"]["incumbency"])),
        row("", lambda r: _cell_ci(r["corr"]["incumbency"])),
        row(r"\quad Partial $R^2$", lambda r: f3(r["corr"]["incumbency"]["r2"])),
        row(r"\quad Share with first branch in 2013Q1",
            lambda r: f3(r["corr"]["incumbency"]["censored_share"])),
        row(r"Own branches, lagged log", lambda r: _cell_slope(r["corr"]["own_branches"])),
        row("", lambda r: _cell_ci(r["corr"]["own_branches"])),
        row(r"\quad Partial $R^2$", lambda r: f3(r["corr"]["own_branches"]["r2"])),
        r"\midrule",
        row("Observations", lambda r: f"{r['n_obs']:,}"),
        row("Conglomerates", lambda r: f"{r['n_banks']:,}"),
        row(r"Share of cells truncated at zero", lambda r: f3(r["trunc_share"])),
    ]
    note = (
        r"\textit{Notes:} $\hat{\xi}$ is the demand residual of the random-coefficients demand "
        r"model, specification 12. Sample: the "
        r"brick-and-mortar (B) firms' conglomerate $\times$ deposit type $\times$ minimum comparable "
        r"area (MCA) $\times$ quarter cells of the demand sample. Digital (D) firms are left out "
        r"because their market is national. Weights: the cell's deposit balance. Panel A: "
        r"order-free (Shapley) split of the weighted variance among quarter, conglomerate (bank) "
        r"and MCA (market) effects. Bank $\times$ type is what the bank share gains when bank "
        r"effects differ by deposit type, so it is zero for time deposits alone. Panel B: slope of "
        r"$\hat{\xi}$ on its value in the previous quarter, within bank $\times$ type $\times$ "
        r"market cells and without demeaning. Active balances come from the B-firm active-deposit "
        r"construction, the balance minus the carried-forward stock, so errors in the sleepy share "
        r"enter $\hat{\xi}$ and are themselves persistent. Panel B is therefore an upper bound on "
        r"the persistence of demand shocks. Cells that this construction truncates at zero (last "
        r"row) are not in the demand sample, so no panel uses them and they break the lag. Panel C: one "
        r"regression per variable, with quarter effects. Incumbency: years since the "
        r"conglomerate's first quarter with a branch in the MCA in the branch counts of the "
        r"Estat\'istica Banc\'aria Mensal (ESTBAN), which start in 2013Q1, so earlier entry is "
        r"recorded as 2013Q1. The weighted share of cells recorded at 2013Q1 is in the row below "
        r"the incumbency slope. Within a quarter the slope therefore compares conglomerates that "
        r"entered the MCA after 2013Q1 with those already there. Own branches: log of one plus the conglomerate's branches in the "
        r"MCA in the previous quarter. Variables in the model or in the demand instrument set are "
        r"left out of Panel C, since the estimation makes their sample correlation with "
        r"$\hat{\xi}$ zero by construction. Partial $R^2$: the share of the variance of "
        r"$\hat{\xi}$, net of quarter effects, that the variable accounts for. $\hat{\xi}$ "
        r"inherits the imprecision of the price coefficient. Intervals in brackets: 95\% wild "
        r"cluster bootstrap (Webb weights, 999 draws, bias-corrected), conglomerate clusters. "
        r"*** $p<0.01$, ** $p<0.05$, * $p<0.1$."
    )
    col_fmt = r"@{}l" + r">{\raggedleft\arraybackslash}X" * len(cols) + r"@{}"
    return "\n".join([
        "% tab_xi_diagnostics.tex -- GENERATED by make_xi_diagnostics.py. Do not edit.",
        r"\begin{spacing}{1.0}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\par\addvspace{\LTpre}\noindent\begin{minipage}{\textwidth}\captionsetup{position=top}",
        r"\captionof{table}{Demand Residuals: Decomposition, Persistence, and Correlates}"
        r"\label{tab:xi_diagnostics}",
        rf"\begin{{tabularx}}{{\textwidth}}{{{col_fmt}}}",
        r"\toprule",
        head1 + r" \\",
        rules,
        head2 + r" \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        rf"\multicolumn{{{nc}}}{{@{{}}p{{\dimexpr\textwidth-2\tabcolsep\relax}}@{{}}}}"
        rf"{{\footnotesize {note}}} \\",
        r"\end{tabularx}",
        r"\end{minipage}\par\prevdepth=0pt\addvspace{\LTpost}",
        r"\end{spacing}",
        "",
    ])


_STANDALONE = r"""\documentclass[12pt]{article}
\usepackage[letterpaper,margin=1in]{geometry}
\usepackage{amsmath,amssymb,booktabs,tabularx,array,longtable,setspace,caption}
\begin{document}
\input{tab_xi_diagnostics.tex}
\end{document}
"""


def compile_check(tex_dir: Path) -> bool:
    """Compile a standalone wrapper around the table (references to the paper print as ??);
    the wrapper and its by-products are deleted when the build is clean."""
    stem = "_xi_diag_standalone"
    (tex_dir / f"{stem}.tex").write_text(_STANDALONE, encoding="utf-8")
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", f"{stem}.tex"], cwd=str(tex_dir),
                       capture_output=True, text=True)
    log = (tex_dir / f"{stem}.log").read_text(encoding="latin-1", errors="replace")
    ok = (tex_dir / f"{stem}.pdf").is_file() and not re.search(r"^! ", r.stdout, re.M)
    overfull = len(re.findall(r"^Overfull \\hbox", log, re.M))
    pages = re.search(r"Output written on .*?\((\d+) page", log)
    print(f"  standalone build: {'clean' if ok else 'FAILED'}; overfull boxes {overfull}; "
          f"pages {pages.group(1) if pages else '?'}")
    if ok:
        for f in tex_dir.glob(f"{stem}.*"):
            f.unlink()
    return ok and overfull == 0


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--routines", type=int, nargs="+", default=list(_routines.LINK_ESTS))
    ap.add_argument("--refresh", action="store_true", help="rebuild every cached input")
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)

    own = estban_branches(a.refresh)
    res, meta = {}, {}
    for e in a.routines:
        xi, fit = load_xi(e, a.refresh)
        d, pq = load_frame(e, xi)
        tr = truncation(e, a.refresh)
        meta[f"E{e}"] = dict(fit=fit, parquet=_stamp(pq), truncated_counts=tr)
        res[f"E{e}"] = {}
        for s, _lab in SETS:
            ds = d[d["k"] == 4].reset_index(drop=True) if s == "k4" else d
            keys = [f"B_k4"] if s == "k4" else [k for k in tr if k.startswith("B_")]
            dropped, cells = (sum(tr[k][i] for k in keys) for i in (0, 1))
            print(f"E{e} {s}: {len(ds):,} B cells, {ds['bank'].nunique()} conglomerates")
            r = dict(anova=anova(ds), ar1=persistence(ds), corr=correlates(ds, own),
                     n_obs=int(len(ds)), n_banks=int(ds["bank"].nunique()),
                     trunc_share=dropped / cells, trunc_dropped=int(dropped), trunc_cells=int(cells))
            res[f"E{e}"][s] = r
            an = r["anova"]
            print("   anova " + ", ".join(f"{k} {v:.3f}" for k, v in an.items()))
            for k in ("within", "raw"):
                v = r["ar1"][k]
                print(f"   ar1 {k}: {v['slope']:.3f}{v['stars']} [{v['lo']:.3f}, {v['hi']:.3f}]")
            for k, v in r["corr"].items():
                print(f"   {k}: {v['slope']:.3f}{v['stars']} [{v['lo']:.3f}, {v['hi']:.3f}] "
                      f"R2 {v['r2']:.4f} coverage {v['coverage']:.3f}")
            print(f"   truncated: {dropped:,} of {cells:,} ({dropped / cells:.3f})")
        all_b = [k for k in tr if k.startswith("B_")]
        res[f"E{e}"]["trunc_share_B"] = (sum(tr[k][0] for k in all_b) / sum(tr[k][1] for k in all_b))

    res["_meta"] = dict(
        written=datetime.datetime.now().isoformat(timespec="seconds"), spec=SPEC, stage=STAGE,
        sample="B firms' conglomerate x deposit type x MCA x quarter cells of the demand sample",
        weights="deposit_balance of the cell", boot=f"Webb wild cluster bootstrap, B={BOOT_B}, "
        "conglomerate clusters, bias-corrected 95% interval", estban_first_quarter="2013Q1",
        inputs=meta)
    SIDECAR.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"  wrote {SIDECAR}")
    TEX.write_text(write_table(res, a.routines), encoding="utf-8")
    print(f"  wrote {TEX}")
    if not compile_check(TEX.parent):
        sys.exit("tab_xi_diagnostics.tex did not compile cleanly on its own")


if __name__ == "__main__":
    main()

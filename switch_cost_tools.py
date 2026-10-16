"""switch_cost_tools.py -- the local commands of the switch-cost investigation, in one CLI.

  locked    the locked share r_4 of time deposits by conglomerate and year, for the F3 arms
  fits      the spec-12 fits, pooled and type-shift, E3 and E4, one at a time
  prep      the demand prep of the four arms, with the identity checks C2-C4
  check     one identity check: `check fit --est N` or `check parquet --name X A B`
  tables    the tables of the local report
  bundle    the two cluster zips (scripts, data) with their manifests

Every output goes under ESTIMATION_OUTPUT/SWITCH_COST (utils/switch_cost_sandbox.py). Production
fits and parquets are read, never written. Run from the repo root with the project venv:

  python switch_cost_tools.py fits            # about 85 minutes for E3 and E4
  python switch_cost_tools.py prep            # about 40 minutes, checks included
  python switch_cost_tools.py tables
  python switch_cost_tools.py bundle --stamp 20261013_1946

The estimators and the demand prep run as child processes, one at a time: concurrent
estimators exhaust Windows' non-paged pool.

--- locked ---
r_4 is the share of a bank's time-deposit balances that cannot move within the quarter. Two
bounds, both reported (user decision of 2026-10-13):
  r4_hi  CONTRACTUAL: time deposits with remaining maturity beyond 90 days, from the statement
         notes (raw/DEPOSIT_MATURITY/deposit_maturity_summary.csv, `scrape_deposit_maturity.py summary`):
         1 - share_no_maturity - share_le90. An upper bound: a long CDB with daily liquidity
         counts here although it can be redeemed at any time.
  r4_lo  EARLIEST WITHDRAWAL: retail funding withdrawable only after six months or more, as a
         share of retail funding with a stated term, from the Pillar 3 NSFR tables
         (raw/PILAR3/liq2_retail_buckets.csv, row "retail funding", 5 deposit banks, 2020 on).
         A lower bound: three to six months of lock-in is not separated.
Fill rule, per bound: the bank's own year; else its nearest observed year; else the median over
observed banks in that year (the median of the nearest year with observations when a year has
none, i.e. 2016-2019 for r4_lo). The prep uses the `__median__` rows for banks outside the
sources (user decision of 2026-10-13).

--- check ---
  fit       the pooled re-estimate (switch_cost_pooled/est{N}) against the production fit
            (ESTIMATION_OUTPUT/DEMAND_PREP/est{N}), spec 12: index coefficients, the link grid,
            the first stage, and the national phi_t path.
  parquet   two demand parquets row by row on (entity_id, time_id): row sets, column sets and
            the largest absolute and relative gap of every common numeric column.
`prep` runs: C2 the pooled arm on the PRODUCTION fits vs DEMAND_PREP_20261013 (code identity);
C3 F1 fed the pooled fit as its type-shift fit vs the pooled arm; C4 F3hi with r_4 = 0 vs the
pooled arm. Reports go to SWITCH_COST/switch_cost_checks/.

--- tables ---
  T1  the type shifts: lambda_k, phi_k at the average market (the link at the index constant
      plus lambda_k), the type AMEs with the estimator's link-only standard errors, the pooled phi
  T1b the state AMEs, pooled against type-shift fit
  T2  the national phi_k,t path by type against the pooled phi_t (F1 parquet); T2s before and
      after Pix
  T3  the demand arms by type: mean phi, active deposits and shares against the pooled arm;
      T3b the pool factors
  T4  F3's implied passive outflow d_k = 1 - (beta_k - (1-phi) r_k) / phi at the average market
Writes CSVs and tables.md to SWITCH_COST/switch_cost_report/.

--- bundle ---
  scripts zip  every switch_cost_* file of the repo (root and utils/), plus README and
               MANIFEST.md5. Only NEW names, so unpacking into the project's scripts checkout
               cannot overwrite a file of the shared project. REQUIRES.md5 lists the shared
               modules the copies import (parsed recursively) with their LOCAL md5.
  data zip     the inputs of claude/switch_cost/data_list.tsv (local path, path inside the zip),
               plus MANIFEST.sha256. Nothing is uploaded or unpacked here.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import pickle
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from utils import switch_cost_sandbox  # noqa: F401  (output tree: import before any path is resolved)
from utils import paths
from utils.switch_cost_sleep_links import NonLinearResults  # noqa: F401 (unpickling)

REPO = Path(__file__).resolve().parent
BASE = switch_cost_sandbox.LOCAL_BASE
LOGS = BASE / "logs"
SPEC12 = "IV_RivalSpread x Tech"
TYPES = {1: "demand", 2: "savings", 4: "time", 5: "prepaid"}
ARMS = ["pooled", "F1", "F3hi", "F3lo"]
TAG = {3: "index", 4: "index_time"}
KEYS = ["entity_id", "time_id"]


def _out(name: str) -> Path:
    d = switch_cost_sandbox.check_dir(BASE / name, name)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _child_env(**extra) -> dict:
    """The environment of a child step: the parent's, without the output tree this process
    defaulted to, so each child's guard picks the tree of its own variant."""
    env = {k: v for k, v in os.environ.items() if k not in ("SLEEP_OUT_ROOT", "DEMAND_PREP_DIR")}
    env.setdefault("MPLBACKEND", "Agg")
    env.update(extra)
    return env


def _run(cmd: list[str], log: Path, **env) -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    print(f"[{datetime.now():%H:%M:%S}] {' '.join(cmd[1:])}  -> {log.name}", flush=True)
    with open(log, "w", encoding="utf-8") as fh:
        rc = subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT, env=_child_env(**env)).returncode
    print(f"[{datetime.now():%H:%M:%S}]   rc={rc}", flush=True)
    return rc


# ============================================================================== locked
MEDIAN_KEY = "__median__"
YEARS = list(range(2016, 2025))
# Statement-note firm keys and NSFR CNPJs -> prudential conglomerate of the market panel.
FIRM_TO_CONGL = {
    "abc": "C0080312", "banrisul": "C0080154", "bb": "C0080329", "bmg": "C0080178",
    "bradesco": "C0080075", "btg": "C0080336", "daycoval": "C0081744", "itau": "C0080099",
    "nubank": "C0084693", "pan": "C0080257", "santander_br": "C0080185",
}
# BNDES (33657248000189) takes no deposits and is left out of the median.
CNPJ_TO_CONGL = {
    191: "C0080329", 360305000104: "C0080738", 60701190000104: "C0080099",
    60746948000112: "C0080075", 90400888000142: "C0080185",
}


def _contractual_bound() -> pd.DataFrame:
    d = pd.read_csv(paths.RAW / "DEPOSIT_MATURITY" / "deposit_maturity_summary.csv")
    # one source per bank-year: the CVM filing when present (every bank but Nubank has it)
    d["_pref"] = (d["source"] != "cvm").astype(int)
    d = d.sort_values(["firm_key", "fiscal_year", "_pref"]).drop_duplicates(["firm_key", "fiscal_year"])
    d["value"] = 1.0 - d["share_no_maturity"].fillna(0.0) - d["share_le90"]
    d["CodConglomeradoPrudencial"] = d["firm_key"].map(FIRM_TO_CONGL)
    if d["CodConglomeradoPrudencial"].isna().any():
        raise SystemExit(f"unmapped firm keys: {sorted(d.loc[d['CodConglomeradoPrudencial'].isna(), 'firm_key'].unique())}")
    return d.rename(columns={"fiscal_year": "year"})[["CodConglomeradoPrudencial", "year", "value", "source"]]


def _withdrawal_bound() -> pd.DataFrame:
    n = pd.read_csv(paths.RAW / "PILAR3" / "liq2_retail_buckets.csv")
    n = n[(n["row_label"] == "retail funding") & n["cnpj"].isin(CNPJ_TO_CONGL)].copy()
    term = n["under_6m"] + n["6m_to_1y"] + n["1y_or_more"]
    n["value"] = (n["6m_to_1y"] + n["1y_or_more"]) / term.replace(0, np.nan)
    n["year"] = n["quarter"].astype(str).str[:4].astype(int)
    n["CodConglomeradoPrudencial"] = n["cnpj"].map(CNPJ_TO_CONGL)
    g = n.groupby(["CodConglomeradoPrudencial", "year"], as_index=False)["value"].mean()
    g["source"] = "pillar3_liq2"
    return g


def _fill(obs: pd.DataFrame, name: str) -> pd.DataFrame:
    obs = obs.dropna(subset=["value"])
    med = obs[obs["year"].isin(YEARS)].groupby("year")["value"].median()
    yrs_obs = sorted(med.index)
    if not yrs_obs:
        raise SystemExit(f"{name}: no observation inside {YEARS[0]}-{YEARS[-1]}")
    rows = []
    for y in YEARS:
        if y in med.index:
            rows.append((MEDIAN_KEY, y, float(med[y]), "year_median"))
        else:
            near = min(yrs_obs, key=lambda z: (abs(z - y), z))
            rows.append((MEDIAN_KEY, y, float(med[near]), f"year_median_of_{near}"))
    for c, g in obs.groupby("CodConglomeradoPrudencial"):
        own = g.set_index("year")["value"]
        for y in YEARS:
            if y in own.index:
                rows.append((c, y, float(own[y]), "own"))
            else:
                near = min(own.index, key=lambda z: (abs(z - y), z))
                rows.append((c, y, float(own[near]), f"own_{near}"))
    return pd.DataFrame(rows, columns=["CodConglomeradoPrudencial", "year", name, f"{name}_fill"])


def cmd_locked(_a) -> None:
    hi = _fill(_contractual_bound(), "r4_hi")
    lo = _fill(_withdrawal_bound(), "r4_lo")
    keys = pd.concat([hi[["CodConglomeradoPrudencial", "year"]], lo[["CodConglomeradoPrudencial", "year"]]]).drop_duplicates()
    out = keys.merge(hi, how="left").merge(lo, how="left")
    med = out[out["CodConglomeradoPrudencial"] == MEDIAN_KEY].set_index("year")
    for b in ("r4_hi", "r4_lo"):
        miss = out[b].isna()
        out.loc[miss, b] = out.loc[miss, "year"].map(med[b])
        out.loc[miss, f"{b}_fill"] = "year_median"
    out = out.sort_values(["CodConglomeradoPrudencial", "year"])
    if not ((out["r4_lo"] <= out["r4_hi"] + 1e-12) | out["r4_lo"].isna()).all():
        print("[note] r4_lo exceeds r4_hi in some bank-years (different sources and populations)")
    dest = _out("switch_cost_inputs") / "r4_locked_bank_year.csv"
    out.to_csv(dest, index=False, encoding="utf-8")
    pd.set_option("display.width", 200)
    print(med[["r4_hi", "r4_hi_fill", "r4_lo", "r4_lo_fill"]].round(3).to_string())
    print(f"\n{len(out)} rows, {out['CodConglomeradoPrudencial'].nunique() - 1} banks -> {dest}")


# ============================================================================== fits, prep
def cmd_fits(a) -> None:
    rcs = []
    for e in a.ests:
        for v in ("pooled", "typeshift"):
            rcs.append(_run([sys.executable, "-u", "switch_cost_sleep_est_single.py", "--est", str(e), "--spec12"],
                            LOGS / f"fit_{v}_est{e}.log", SWITCH_COST_TYPE_SHIFTS="1" if v == "typeshift" else "0"))
    if any(rcs):
        raise SystemExit(f"a fit failed (return codes {rcs}); see {LOGS}")


def _prep(e: int, name: str, *args: str) -> int:
    return _run([sys.executable, "-u", "switch_cost_sleep_demand_prep_link.py", "--est", str(e), "--spec", "12", *args],
                LOGS / f"prep_{name}_est{e}.log")


def cmd_prep(a) -> None:
    eo = paths.estimation_output()
    pooled = str(switch_cost_sandbox.local_fit_root("pooled"))
    bad = []
    for e in a.ests:
        f = f"demand_{e}_{TAG[e]}_spec_12.parquet"
        nobs = _fit_check(e)["second_stage"].get("nobs", [0, 1])
        print(f"   check fit_est{e}: N {nobs}")
        if nobs[0] != nobs[1]:
            bad.append(f"fit E{e}: N differs")
        # C2: the copied prep on the production fits reproduces the production parquet
        rc = _prep(e, "C2_prodfits", "--arm", "pooled", "--fits-dir", str(eo / "DEMAND_PREP"),
                   "--out-dir", str(BASE / "switch_cost_check_prodfits"))
        bad += _parquet_check(f"C2_est{e}", BASE / "switch_cost_check_prodfits" / f, eo / "DEMAND_PREP_20261013" / f, rc)
        # the pooled arm on the re-estimated pooled fit: the reference of every arm
        if _prep(e, "pooled", "--arm", "pooled"):
            bad.append(f"pooled E{e}")
        # C3 / C4: the F1 and F3 machinery collapse to the pooled arm
        rc = _prep(e, "C3_F1_identity", "--arm", "F1", "--fits-dir", pooled,
                   "--out-dir", str(BASE / "switch_cost_check_F1_identity"))
        bad += _parquet_check(f"C3_est{e}", BASE / "switch_cost_check_F1_identity" / f, BASE / "switch_cost_demand_pooled" / f, rc)
        rc = _prep(e, "C4_F3_zero", "--arm", "F3hi", "--zero-lock", "--fits-dir", pooled,
                   "--out-dir", str(BASE / "switch_cost_check_F3_zero"))
        bad += _parquet_check(f"C4_est{e}", BASE / "switch_cost_check_F3_zero" / f, BASE / "switch_cost_demand_pooled" / f, rc)
        for arm in ("F1", "F3hi", "F3lo"):
            if _prep(e, arm, "--arm", arm):
                bad.append(f"{arm} E{e}")
    print("all checks and arms passed" if not bad else f"FAILED: {bad}")
    if bad:
        raise SystemExit(1)


# ============================================================================== check
def _spec12(pkl: Path):
    with open(pkl, "rb") as f:
        d = pickle.load(f)
    return d[next(k for k in d if SPEC12 in k)]


def _gap(a, b) -> dict:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return {"shape": [list(a.shape), list(b.shape)]}
    d = np.abs(a - b)
    scale = np.maximum(np.abs(a), np.abs(b))
    rel = np.where(scale > 0, d / np.where(scale > 0, scale, 1.0), 0.0)
    return {"max_abs": float(np.nanmax(d)) if d.size else 0.0,
            "max_rel": float(np.nanmax(rel)) if d.size else 0.0}


def compare_fits(est: int, new_root: Path, prod_root: Path) -> dict:
    new = _spec12(new_root / f"est{est}" / "estimation_results.pkl")
    prod = _spec12(prod_root / f"est{est}" / "estimation_results.pkl")
    out = {}
    for stage in ("second_stage", "first_stage"):
        rn, rp = new.get(stage), prod.get(stage)
        if rn is None or rp is None:
            out[stage] = {"missing": [rn is None, rp is None]}
            continue
        pn = getattr(rn, "params_native", None)
        pn = rn.params if pn is None else pn
        pp = getattr(rp, "params_native", None)
        pp = rp.params if pp is None else pp
        common = [c for c in pp.index if c in pn.index]
        out[stage] = {"params": _gap(pn[common].values, pp[common].values),
                      "only_new": [c for c in pn.index if c not in pp.index],
                      "only_prod": [c for c in pp.index if c not in pn.index]}
        if stage == "second_stage":
            out[stage]["link"] = [getattr(rn, "link", None), getattr(rp, "link", None)]
            for g in ("si_vgrid", "si_ggrid"):
                if getattr(rn, g, None) is not None and getattr(rp, g, None) is not None:
                    out[stage][g] = _gap(getattr(rn, g), getattr(rp, g))
            out[stage]["nobs"] = [int(getattr(rn, "nobs", -1)), int(getattr(rp, "nobs", -1))]
    tn = pd.read_csv(new_root / f"est{est}" / "national_phi_t.csv")
    tp = pd.read_csv(prod_root / f"est{est}" / "national_phi_t.csv")
    col = next((c for c in tp.columns if SPEC12.replace(" ", "_") in c and c.startswith("phi_t_")), None)
    if col and col in tn.columns:
        m = tp[["year_quarter", col]].merge(tn[["year_quarter", col]], on="year_quarter", suffixes=("_prod", "_new"))
        out["national_phi_t"] = {"column": col, "quarters": len(m), **_gap(m[f"{col}_new"], m[f"{col}_prod"])}
    return out


def compare_parquets(a: Path, b: Path) -> dict:
    A, B = pd.read_parquet(a), pd.read_parquet(b)
    out = {"a": str(a), "b": str(b), "rows": [len(A), len(B)],
           "cols_only_a": sorted(set(A.columns) - set(B.columns)),
           "cols_only_b": sorted(set(B.columns) - set(A.columns))}
    m = A.merge(B, on=KEYS, how="outer", suffixes=("_a", "_b"), indicator=True)
    out["rows_only_a"] = int((m["_merge"] == "left_only").sum())
    out["rows_only_b"] = int((m["_merge"] == "right_only").sum())
    m = m[m["_merge"] == "both"]
    gaps = {}
    for c in sorted(set(A.columns) & set(B.columns) - set(KEYS)):
        ca, cb = m[f"{c}_a"], m[f"{c}_b"]
        if pd.api.types.is_numeric_dtype(ca) and pd.api.types.is_numeric_dtype(cb):
            na_a, na_b = ca.isna(), cb.isna()
            g = _gap(ca.where(~(na_a | na_b), 0.0), cb.where(~(na_a | na_b), 0.0))
            g["nan_mismatch"] = int((na_a != na_b).sum())
            gaps[c] = g
        else:
            gaps[c] = {"unequal": int((ca.astype(str) != cb.astype(str)).sum())}
    out["columns"] = gaps
    out["worst_rel"] = max((v.get("max_rel", 0.0) for v in gaps.values() if isinstance(v, dict)), default=0.0)
    return out


def _bad_columns(rep: dict, tol: float = 1e-10) -> dict:
    return {c: g for c, g in rep["columns"].items()
            if g.get("max_rel", 0) > tol or g.get("nan_mismatch", 0) or g.get("unequal", 0)}


def _save(name: str, rep: dict) -> Path:
    dest = _out("switch_cost_checks") / f"{name}.json"
    dest.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    return dest


def _fit_check(est: int) -> dict:
    rep = compare_fits(est, switch_cost_sandbox.local_fit_root("pooled"), paths.estimation_output() / "DEMAND_PREP")
    _save(f"fit_est{est}", rep)
    return rep


def _parquet_check(name: str, a: Path, b: Path, prep_rc: int) -> list[str]:
    """Run one parquet check inside `prep`; return the failures as labels."""
    if prep_rc:
        return [f"{name}: prep rc={prep_rc}"]
    rep = compare_parquets(a, b)
    _save(name, rep)
    bad = _bad_columns(rep)
    ok = not bad and not rep["rows_only_a"] and not rep["rows_only_b"]
    print(f"   check {name}: rows {rep['rows']}, worst relative gap {rep['worst_rel']:.1e} -> {'ok' if ok else 'FAIL'}")
    return [] if ok else [name]


def cmd_check(a) -> None:
    if a.what == "fit":
        rep = compare_fits(a.est, Path(a.new_root), Path(a.prod_root))
        dest = _save(f"fit_est{a.est}", rep)
        print(json.dumps(rep, indent=1))
    else:
        rep = compare_parquets(Path(a.a), Path(a.b))
        dest = _save(a.name, rep)
        print(json.dumps({k: v for k, v in rep.items() if k != "columns"}, indent=1))
        bad = _bad_columns(rep)
        print("columns with gaps above 1e-10 (relative):", json.dumps(bad, indent=1) if bad else "none")
    print(f"-> {dest}")


# ============================================================================== tables
def _fit(variant, est):
    p = switch_cost_sandbox.local_fit_root(variant) / f"est{est}" / "estimation_results.pkl"
    if not p.exists():
        return None
    return _spec12(p)["second_stage"]


def _link_at(res, v):
    return float(np.interp(v, res.si_vgrid, res.si_ggrid))


def _t1(est):
    r0, rt = _fit("pooled", est), _fit("typeshift", est)
    if r0 is None or rt is None:
        return None
    c0, ct = float(r0.params_native["nr_lagged_dep"]), float(rt.params_native["nr_lagged_dep"])
    rows = [{"est": est, "type": "pooled (all types)", "lambda": np.nan,
             "phi_avg_market": _link_at(r0, c0), "ame": np.nan, "se": np.nan, "p": np.nan,
             "se_quarter": np.nan}]
    for k, nm in TYPES.items():
        lam = 0.0 if k == 1 else float(rt.params_native.get(f"interaction_dtype_{k}", np.nan))
        a = f"interaction_dtype_{k}"
        rows.append({"est": est, "type": f"{nm} ({k})", "lambda": lam,
                     "phi_avg_market": _link_at(rt, ct + lam),
                     "ame": float(rt.params.get(a, np.nan)) if k != 1 else np.nan,
                     "se": float(rt.bse.get(a, np.nan)) if k != 1 else np.nan,
                     "p": float(rt.pvalues.get(a, np.nan)) if k != 1 else np.nan,
                     "se_quarter": float(getattr(rt, "bse_time", pd.Series(dtype=float)).get(a, np.nan)) if k != 1 else np.nan})
    df = pd.DataFrame(rows)
    df["nobs"] = int(rt.nobs)
    # the other states: does adding the shifts move them?
    states = list(r0.params.index)
    mv = pd.DataFrame({"pooled_ame": r0.params[states], "typeshift_ame": rt.params.reindex(states),
                       "pooled_se": r0.bse[states], "typeshift_se": rt.bse.reindex(states)})
    mv.insert(0, "est", est)
    return df, mv


def _parquet(arm, est):
    p = BASE / f"switch_cost_demand_{arm}" / f"demand_{est}_{TAG[est]}_spec_12.parquet"
    return pd.read_parquet(p) if p.exists() else None


def _t2(est):
    f1 = _parquet("F1", est)
    if f1 is None:
        return None
    g = f1.drop_duplicates(["time_id", "deposit_type"])[["time_id", "deposit_type", "phi_t", "phi0_t"]]
    w = g.pivot(index="time_id", columns="deposit_type", values="phi_t")
    w.columns = [f"phi_t_k{k}" for k in w.columns]
    w["phi0_t"] = g.groupby("time_id")["phi0_t"].first()
    w.insert(0, "est", est)
    return w.reset_index()


def _t2s(t2df):
    """T2 in two periods: before Pix (to 2020Q3) and from 2020Q4, and the change."""
    if t2df is None:
        return None
    d = t2df.copy()
    yq = d["time_id"].str.split("Q", expand=True).astype(int)
    d["post_pix"] = (yq[0] > 2020) | ((yq[0] == 2020) & (yq[1] == 4))
    cols = [c for c in d.columns if c.startswith("phi_t_k") or c == "phi0_t"]
    pre = d[~d["post_pix"]][cols].mean()
    post = d[d["post_pix"]][cols].mean()
    out = pd.DataFrame({"period": ["to 2020Q3", "from 2020Q4", "change"]})
    for c in cols:
        out[c] = [pre[c], post[c], post[c] - pre[c]]
    out.insert(0, "est", int(d["est"].iloc[0]))
    return out


def _t3(est):
    base = _parquet("pooled", est)
    if base is None:
        return None, None
    base = base.assign(share=np.where(base["is_B"], base["share_B_cond"], base["share_D"]))
    rows, pools = [], []
    for arm in ARMS:
        d = _parquet(arm, est)
        if d is None:
            continue
        d = d.assign(share=np.where(d["is_B"], d["share_B_cond"], d["share_D"]))
        m = d.merge(base[["entity_id", "time_id", "share", "Dep_Act"]], on=["entity_id", "time_id"],
                    how="inner", suffixes=("", "_pooled"))
        for k, nm in TYPES.items():
            dk, mk = d[d["deposit_type"] == k], m[m["deposit_type"] == k]
            rows.append({"est": est, "arm": arm, "type": f"{nm} ({k})", "rows": len(dk),
                         "phi_mean": float(dk["phi_mt"].mean()),
                         "dep_act_vs_pooled": float(mk["Dep_Act"].sum() / mk["Dep_Act_pooled"].sum()) if len(mk) else np.nan,
                         "mean_log_share_ratio": float(np.log(mk["share"] / mk["share_pooled"]).replace([np.inf, -np.inf], np.nan).mean()) if len(mk) else np.nan})
        if "pool_factor" in d.columns:
            pf = d.loc[d["is_B"], "pool_factor"]
            pools.append({"est": est, "arm": arm, "pool_factor_B_mean": float(pf.mean()),
                          "p05": float(pf.quantile(0.05)), "p50": float(pf.median()), "p95": float(pf.quantile(0.95)),
                          "pool_factor_national_mean": float(d.loc[~d["is_B"], "pool_factor"].mean())})
    return pd.DataFrame(rows), pd.DataFrame(pools)


def _t4(est, t1df):
    if t1df is None:
        return None
    lk = pd.read_csv(BASE / "switch_cost_inputs" / "r4_locked_bank_year.csv")
    med = lk[lk["CodConglomeradoPrudencial"] == MEDIAN_KEY]
    phi = float(t1df.loc[t1df["type"] == "pooled (all types)", "phi_avg_market"].iloc[0])
    rows = []
    for bound in ("hi", "lo"):
        r4 = float(med[f"r4_{bound}"].median())
        for k, nm in TYPES.items():
            beta = float(t1df.loc[t1df["type"] == f"{nm} ({k})", "phi_avg_market"].iloc[0])
            r = r4 if k == 4 else 0.0
            rows.append({"est": est, "bound": bound, "type": f"{nm} ({k})", "beta_k": beta, "phi": phi,
                         "r_k": r, "implied_d_k": 1.0 - (beta - (1.0 - phi) * r) / phi})
    return pd.DataFrame(rows)


def _md(df, floatfmt=3):
    if df is None or len(df) == 0:
        return "_not available_\n"
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda x: "" if pd.isna(x) else f"{x:.{floatfmt}f}")
    cols = list(d.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def cmd_tables(_a) -> None:
    out_dir = _out("switch_cost_report")
    parts = {"T1": [], "T1b": [], "T2": [], "T2s": [], "T3": [], "T3b": [], "T4": []}
    for est in (3, 4):
        r = _t1(est)
        t1df = None
        if r is not None:
            t1df, mv = r
            parts["T1"].append(t1df)
            parts["T1b"].append(mv.reset_index().rename(columns={"index": "state"}))
        t2df = _t2(est)
        parts["T2"].append(t2df)
        parts["T2s"].append(_t2s(t2df))
        a, b = _t3(est)
        parts["T3"].append(a)
        parts["T3b"].append(b)
        parts["T4"].append(_t4(est, t1df))
    text = ["# Switch-cost local results (spec 12)\n"]
    titles = {"T1": "T1. Type shifts and phi at the average market",
              "T1b": "T1b. State AMEs, pooled vs type-shift fit",
              "T2": "T2. National phi_k,t by type (F1) and pooled phi_t",
              "T2s": "T2s. National phi_k,t before and after Pix",
              "T3": "T3. Demand arms by deposit type",
              "T3b": "T3b. Pool factors (arm pool / pooled pool)",
              "T4": "T4. F3 implied passive outflow d_k at the average market"}
    for key, frames in parts.items():
        frames = [f for f in frames if f is not None and len(f)]
        df = pd.concat(frames, ignore_index=True) if frames else None
        if df is not None:
            df.to_csv(out_dir / f"{key}.csv", index=False)
        text.append(f"## {titles[key]}\n\n" + _md(df) + "\n")
    (out_dir / "tables.md").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))
    print(f"-> {out_dir}")


# ============================================================================== bundle
# utils/__init__.py defines load_panel_cached, which the copies import from the package;
# utils/routines.py reads config/routines.toml.
EXTRA_REQUIRES = ["config/routines.toml", "utils/__init__.py"]


def _digest(p: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _project_imports(py: Path) -> set[str]:
    """Repo modules imported by a file: utils.X, top-level repo modules."""
    try:
        tree = ast.parse(py.read_text(encoding="utf-8"))
    except SyntaxError:
        return set()
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "utils":
                mods |= {f"utils.{a.name}" for a in node.names}
            else:
                mods.add(node.module)
        elif isinstance(node, ast.Import):
            mods |= {a.name for a in node.names}
    out = set()
    for m in mods:
        rel = Path(*m.split("."))
        for cand in (REPO / rel.with_suffix(".py"), REPO / rel / "__init__.py"):
            if cand.exists():
                out.add(str(cand.relative_to(REPO)).replace("\\", "/"))
    return out


def cmd_bundle(a) -> None:
    out_dir = switch_cost_sandbox.check_dir(Path(a.out).resolve() if a.out else BASE / "switch_cost_bundle", "--out")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = a.stamp or datetime.now().strftime("%Y%m%d_%H%M")
    files = sorted({p for pat in ("switch_cost_*", "utils/switch_cost_*") for p in REPO.glob(pat)
                    if p.is_file() and p.suffix in (".py", ".sh")})
    names = [str(p.relative_to(REPO)).replace("\\", "/") for p in files]
    seen, todo = set(), [n for n in names if n.endswith(".py")]
    while todo:
        n = todo.pop()
        for dep in _project_imports(REPO / n):
            if dep not in seen and dep not in names:
                seen.add(dep)
                todo.append(dep)
    requires = sorted(seen | {r for r in EXTRA_REQUIRES if (REPO / r).exists()})
    readme = REPO / "claude" / "switch_cost" / "README_CLUSTER.md"
    zpath = out_dir / f"switch_cost_scripts_{stamp}.zip"
    zpath.unlink(missing_ok=True)   # rebuilding a stamp replaces this command's own zip
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p, n in zip(files, names):
            z.write(p, n)
        z.writestr("SWITCH_COST_MANIFEST.md5", "".join(f"{_digest(p, 'md5')}  {n}\n" for p, n in zip(files, names)))
        z.writestr("SWITCH_COST_REQUIRES.md5", "".join(f"{_digest(REPO / r, 'md5')}  {r}\n" for r in requires))
        if readme.exists():
            z.write(readme, "SWITCH_COST_README.md")
    print(f"scripts zip: {zpath} ({len(files)} files, {len(requires)} shared modules required)")
    for n in names:
        print("   ", n)
    print("  requires:", ", ".join(requires))
    if a.no_data:
        return
    data_list = Path(a.data_list) if a.data_list else REPO / "claude" / "switch_cost" / "data_list.tsv"
    rows = [ln.split("\t") for ln in data_list.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")]
    dpath = out_dir / f"switch_cost_data_{stamp}.zip"
    dpath.unlink(missing_ok=True)
    sha = []
    with zipfile.ZipFile(dpath, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for local, inner in rows:
            lp = Path(local)
            if not lp.exists():
                raise SystemExit(f"data file missing: {lp}")
            z.write(lp, inner)
            sha.append(f"{_digest(lp, 'sha256')}  {inner}\n")
        z.writestr("SWITCH_COST_DATA.sha256", "".join(sha))
    print(f"data zip: {dpath} ({len(rows)} files, {dpath.stat().st_size / 1e6:.1f} MB)")


# ============================================================================== CLI
def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="Local commands of the switch-cost investigation.",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("locked", help="build the locked-share table r4_locked_bank_year.csv")
    for nm in ("fits", "prep"):
        s = sub.add_parser(nm, help=f"run the {nm} for the routines given")
        s.add_argument("--ests", type=int, nargs="+", default=[3, 4], choices=[3, 4])
    c = sub.add_parser("check", help="one identity check")
    csub = c.add_subparsers(dest="what", required=True)
    cf = csub.add_parser("fit")
    cf.add_argument("--est", type=int, required=True)
    cf.add_argument("--new-root", default=str(switch_cost_sandbox.local_fit_root("pooled")))
    cf.add_argument("--prod-root", default=str(paths.estimation_output() / "DEMAND_PREP"))
    cp = csub.add_parser("parquet")
    cp.add_argument("--name", required=True, help="label of the check, e.g. C2_est3")
    cp.add_argument("a")
    cp.add_argument("b")
    sub.add_parser("tables", help="the tables of the local report")
    b = sub.add_parser("bundle", help="the two cluster zips")
    b.add_argument("--out", default=None, help="zip folder (name must contain switch_cost); default SWITCH_COST/switch_cost_bundle")
    b.add_argument("--stamp", default=None, help="name stamp; rebuilding a stamp replaces its zips")
    b.add_argument("--data-list", default=None, help="TSV of local path and path in the zip; default claude/switch_cost/data_list.tsv")
    b.add_argument("--no-data", action="store_true", help="build the scripts zip only")
    a = p.parse_args()
    {"locked": cmd_locked, "fits": cmd_fits, "prep": cmd_prep, "check": cmd_check,
     "tables": cmd_tables, "bundle": cmd_bundle}[a.cmd](a)


if __name__ == "__main__":
    main()

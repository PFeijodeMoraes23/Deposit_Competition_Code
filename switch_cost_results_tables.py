"""switch_cost_results_tables.py -- the tables of the switch-cost local report (read only).

From the spec-12 fits (switch_cost_pooled, switch_cost_typeshift) and the demand parquets of
the four arms, for E3 and E4:

  T1  the type shifts: lambda_k in index units, phi_k at the average market (the link at the
      index constant plus lambda_k), the type AMEs (discrete differences) with the estimator's
      link-only standard errors, and the pooled phi at the average market;
  T2  the national phi_k,t path by type against the pooled phi_t (F1 parquet);
  T3  the demand arms: by deposit type, the mean phi, the active-deposit total, and the mean log
      ratio of each arm's share to the pooled share; the pool factor and the floor counts;
  T4  F3's implied passive outflow d_k = 1 - (beta_k - (1-phi) r_k) / phi at the average market,
      beta_k the type-shift phi_k, phi the pooled phi, r_4 the median locked share.

Writes CSVs and tables.md to ESTIMATION_OUTPUT/SWITCH_COST/switch_cost_report/.
"""
from __future__ import annotations

import json
import pickle
import sys

import numpy as np
import pandas as pd

from utils import switch_cost_sandbox  # noqa: F401  (output tree: import before any path is resolved)
from utils.switch_cost_sleep_links import NonLinearResults  # noqa: F401 (unpickling)

BASE = switch_cost_sandbox.LOCAL_BASE
OUT = switch_cost_sandbox.check_dir(BASE / "switch_cost_report", "OUT")
SPEC12 = "IV_HausmanFull x Tech"
TYPES = {1: "demand", 2: "savings", 4: "time", 5: "prepaid"}
ARMS = ["pooled", "F1", "F3hi", "F3lo"]
TAG = {3: "index", 4: "index_time"}


def fit(variant, est):
    p = switch_cost_sandbox.local_fit_root(variant) / f"est{est}" / "estimation_results.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        d = pickle.load(f)
    return d[next(k for k in d if SPEC12 in k)]["second_stage"]


def link_at(res, v):
    return float(np.interp(v, res.si_vgrid, res.si_ggrid))


def t1(est):
    r0, rt = fit("pooled", est), fit("typeshift", est)
    if r0 is None or rt is None:
        return None
    c0, ct = float(r0.params_native["nr_lagged_dep"]), float(rt.params_native["nr_lagged_dep"])
    rows = [{"est": est, "type": "pooled (all types)", "lambda": np.nan,
             "phi_avg_market": link_at(r0, c0), "ame": np.nan, "se": np.nan, "p": np.nan,
             "se_quarter": np.nan}]
    for k, nm in TYPES.items():
        lam = 0.0 if k == 1 else float(rt.params_native.get(f"interaction_dtype_{k}", np.nan))
        a = f"interaction_dtype_{k}"
        rows.append({"est": est, "type": f"{nm} ({k})", "lambda": lam,
                     "phi_avg_market": link_at(rt, ct + lam),
                     "ame": float(rt.params.get(a, np.nan)) if k != 1 else np.nan,
                     "se": float(rt.bse.get(a, np.nan)) if k != 1 else np.nan,
                     "p": float(rt.pvalues.get(a, np.nan)) if k != 1 else np.nan,
                     "se_quarter": float(getattr(rt, "bse_time", pd.Series(dtype=float)).get(a, np.nan)) if k != 1 else np.nan})
    df = pd.DataFrame(rows)
    df["nobs"] = int(rt.nobs)
    # the other states: does adding the shifts move them?
    states = [c for c in r0.params.index]
    mv = pd.DataFrame({"pooled_ame": r0.params[states], "typeshift_ame": rt.params.reindex(states),
                       "pooled_se": r0.bse[states], "typeshift_se": rt.bse.reindex(states)})
    mv.insert(0, "est", est)
    return df, mv


def parquet(arm, est):
    p = BASE / f"switch_cost_demand_{arm}" / f"demand_{est}_{TAG[est]}_spec_12.parquet"
    return pd.read_parquet(p) if p.exists() else None


def t2(est):
    f1 = parquet("F1", est)
    if f1 is None:
        return None
    g = f1.drop_duplicates(["time_id", "deposit_type"])[["time_id", "deposit_type", "phi_t", "phi0_t"]]
    w = g.pivot(index="time_id", columns="deposit_type", values="phi_t")
    w.columns = [f"phi_t_k{k}" for k in w.columns]
    w["phi0_t"] = g.groupby("time_id")["phi0_t"].first()
    w.insert(0, "est", est)
    return w.reset_index()


def t2s(t2df):
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


def t3(est):
    base = parquet("pooled", est)
    if base is None:
        return None, None
    base = base.assign(share=np.where(base["is_B"], base["share_B_cond"], base["share_D"]))
    rows, pools = [], []
    for arm in ARMS:
        d = parquet(arm, est)
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


def t4(est, t1df):
    if t1df is None:
        return None
    lk = pd.read_csv(BASE / "switch_cost_inputs" / "r4_locked_bank_year.csv")
    med = lk[lk["CodConglomeradoPrudencial"] == "__median__"]
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


def md(df, floatfmt=3):
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


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    parts = {"T1": [], "T1b": [], "T2": [], "T2s": [], "T3": [], "T3b": [], "T4": []}
    for est in (3, 4):
        r = t1(est)
        t1df = None
        if r is not None:
            t1df, mv = r
            parts["T1"].append(t1df)
            parts["T1b"].append(mv.reset_index().rename(columns={"index": "state"}))
        _t2 = t2(est)
        parts["T2"].append(_t2)
        parts["T2s"].append(t2s(_t2))
        a, b = t3(est)
        parts["T3"].append(a)
        parts["T3b"].append(b)
        parts["T4"].append(t4(est, t1df))
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
            df.to_csv(OUT / f"{key}.csv", index=False)
        text.append(f"## {titles[key]}\n\n" + md(df) + "\n")
    (OUT / "tables.md").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()

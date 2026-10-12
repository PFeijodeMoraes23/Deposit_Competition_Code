"""diag_state_vector_selection.py -- which state vector S_mt does the sleepiness design support?

Author: Pedro Feijó de Moraes

The sleepiness function is phi(S_mt) (V_Main eq. 6 and Section "Depositor Sleepiness"). Its
state block is chosen by a rule that never looks at the fitted phi: every element must be a
depositor-side, exogenous market state with a law of motion, the block carries ONE market
moment per depositor characteristic (age, income, digital access) plus the aggregate states
(lagged Selic, Pix), and every element must vary inside the two-way fixed-effect design. This
script supplies the evidence for the last two conditions and reports, separately, how the
fitted level of phi moves with the block.

Two kinds of output, kept apart on purpose:

  (a) DESIGN -- computed from the regressors alone, never from deposits. For each candidate
      block, the two-way-demeaned interaction design [x, S_1 x, ..., S_K x, v_hat Dep_lag]
      (x = nr_lagged_dep, the regressor phi multiplies): condition number and effective
      dimension of its correlation matrix, the VIF of every column, and for every state the
      share of its variance, alone and interacted, that the entity and quarter fixed effects
      absorb. This is the selection evidence.

  (b) LEVEL -- reported sensitivity, not selection. Per block, the linear routine E2 and the
      single-index routine E3 at spec 12 (Hausman instruments): phi at the unweighted average
      market, phi at the lagged-deposit-weighted average market, the population-weighted
      national mean phi (the paper's "Mean phi-hat"), the minimum-variance point of the E2
      level and the shares of phi_mt above 1 and of phi * g at or above 1. E3 adds its AMEs and
      the gap between the phi D rows are estimated at, phi(S_t) at the national
      population-weighted state, and the population-weighted average of the B markets' phi_mt.

Gates, checked before anything else is written: the production block (S5) reproduces the
stored E2 coefficients and national mean phi, and the stored E3 national mean phi. A frame or
kernel that does not reproduce production stops the run.

Reads the market panel through sleep_est_e2.build_pooled_data and the production fits in
DEMAND_PREP/est2 and est3. Writes only to ESTIMATION_OUTPUT/DIAG_STATE_SELECTION.

    python diagnostics/diag_state_vector_selection.py              # design + E2 + E3
    python diagnostics/diag_state_vector_selection.py --skip-e3    # design + E2 only (minutes)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: utils/ and pipeline modules
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import os
# The single-index link bootstrap only produces standard errors, which this script does not
# report; a short run keeps every point estimate identical to production.
os.environ.setdefault("SLEEP_BOOT_B", "19")

import argparse
import json
import pickle
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from sleep_est_e2 import build_pooled_data, define_specifications, demean_variables_2way
from utils import paths as _paths
from utils import routines as R
from utils.sleep_links import fit_nlls_link, fit_single_index, phi_from_native

OUT = _paths.diag_dir("STATE_SELECTION")
SPEC12 = R.SPEC12
SPEC12_TAG = R.SPEC12_TAG
INSTRUMENTS = define_specifications()[1]["IV_HausmanFull"]

# Candidate states. Every block below is the constant plus a subset of these.
PROD = ["pix_exists", "cadunico_families_per1000", "fraction_65plus",
        "risk_free_qoq_lag", "connections_per100"]
EXTRA_MARKET = ["gdp_per_capita", "fraction_young", "frac_4g5g"]
NATIONAL = {"pix_exists", "risk_free_qoq_lag"}


def candidate_blocks(have_4g5g: bool) -> dict:
    """-> {block name: [state columns]} (constant excluded; it is always the first column)."""
    blocks = {
        "S5": PROD,
        "S7": PROD + ["gdp_per_capita", "fraction_young"],
        "S5+gdp_per_capita": PROD + ["gdp_per_capita"],
        "S5+fraction_young": PROD + ["fraction_young"],
    }
    if have_4g5g:
        blocks["S5+frac_4g5g"] = PROD + ["frac_4g5g"]
        blocks["S5 mobile->frac_4g5g"] = [c for c in PROD if c != "connections_per100"] + ["frac_4g5g"]
    for c in PROD:
        blocks[f"S5-{c}"] = [s for s in PROD if s != c]
    return blocks


# ── Frame ────────────────────────────────────────────────────────────────────────
def production_tech_block() -> list:
    """The live production Tech block, read with SLEEP_STATE_BLOCK unset."""
    saved = os.environ.pop("SLEEP_STATE_BLOCK", None)
    try:
        return define_specifications()[2]["Tech"]
    finally:
        if saved is not None:
            os.environ["SLEEP_STATE_BLOCK"] = saved


def build_frame() -> pd.DataFrame:
    """The production pooled frame with every candidate state filled and centred.

    build_pooled_data fills the D-type rows (CODMUN_IBGE == '0') of exactly the states in its
    block with the national population-weighted mean, then median-fills, then centres. Building
    under the seven-state block fills gdp_per_capita and fraction_young too. The fill acts on
    each column separately and no row is dropped on a state, so the five production states and
    the row set are identical to the production frame; the S5 gate verifies it."""
    saved = os.environ.get("SLEEP_STATE_BLOCK")
    os.environ["SLEEP_STATE_BLOCK"] = "legacy7"
    try:
        df = build_pooled_data(time_block=False)
    finally:
        if saved is None:
            os.environ.pop("SLEEP_STATE_BLOCK", None)
        else:
            os.environ["SLEEP_STATE_BLOCK"] = saved
    if "frac_4g5g" in df.columns:
        _fill_and_centre(df, "frac_4g5g")
    df["year_quarter"] = df["time_id"]
    df["is_D_row"] = df["CODMUN_IBGE"].astype(str) == "0"
    return df


def _fill_and_centre(df: pd.DataFrame, col: str) -> None:
    """The build_pooled_data treatment for a state outside the production registry: D rows get
    the national population-weighted mean of the B rows by quarter, the rest median-fill, and
    the column is centred at its frame mean (the registry centres the others at the persisted
    grand means of the same frame)."""
    b = df["CODMUN_IBGE"].astype(str) != "0"
    d = ~b
    med = df.loc[b, col].median()
    vals = df.loc[b, col].fillna(med)
    w = df.loc[b, "pop_total"].fillna(0)
    nat = (vals * w).groupby(df.loc[b, "time_id"]).sum() / w.groupby(df.loc[b, "time_id"]).sum().replace(0, np.nan)
    df.loc[d, col] = df.loc[d, "time_id"].map(nat.fillna(med)).values
    df[col] = df[col].fillna(df[col].median())
    df[col] = df[col] - df[col].mean()


# ── Control function and the E2 kernel ───────────────────────────────────────────
def first_stage(df: pd.DataFrame, s_cols: list) -> pd.DataFrame:
    """sleep_est_e2.run_pooled_first_stage without its standard errors: OLS of the spread on a
    constant, the instruments and the states, over k = 4, 5 rows with every regressor present.
    v_hat is 0 for k = 1, 2 and NaN for uninstrumented k = 4, 5 rows, as in production."""
    out = df.copy()
    endog = out["deposit_type"].isin([4, 5]).values
    regs = sorted(set(INSTRUMENTS + [c for c in s_cols if c != "constant"]))
    valid = endog & out[regs].notnull().all(axis=1).values
    X = np.column_stack([np.ones(valid.sum()), out.loc[valid, regs].to_numpy(float)])
    y = out.loc[valid, "spread_qoq"].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    v = np.where(endog, np.nan, 0.0)
    v[valid] = y - X @ beta
    out["v_hat"] = v
    out["v_hat_x_lagged_dep"] = out["v_hat"] * out["lagged_deposits"]
    return out


def e2_design(df_fs: pd.DataFrame, s_cols: list):
    """-> (estimation rows, raw design, demeaned design, demeaned y, column names)."""
    names = ["nr_lagged_dep"] + [f"interaction_{s}" for s in s_cols] + ["v_hat_x_lagged_dep"]
    work = df_fs.copy()
    for s in s_cols:
        work[f"interaction_{s}"] = work[s] * work["nr_lagged_dep"]
    work = work.dropna(subset=names + ["deposit_balance"])
    X_dm = demean_variables_2way(work, names, "entity_id", "time_id")
    y_dm = demean_variables_2way(work, ["deposit_balance"], "entity_id", "time_id")["deposit_balance"]
    return work, work[names], X_dm, y_dm, names


def crve(X: np.ndarray, u: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Conglomerate-clustered sandwich with the usual small-sample factor. Used only to locate
    the best-determined point of the level and to size its standard error; the paper's tables
    use the wild cluster bootstrap."""
    n, k = X.shape
    bread = np.linalg.pinv(X.T @ X)
    _, ginv = np.unique(groups, return_inverse=True)
    G = ginv.max() + 1
    S = np.zeros((G, k))
    np.add.at(S, ginv, X * u[:, None])
    meat = S.T @ S
    c = G / (G - 1) * (n - 1) / (n - k)
    return c * bread @ meat @ bread


# ── (a) Design ───────────────────────────────────────────────────────────────────
def design_metrics(work, X_raw, X_dm, names, s_cols):
    """Conditioning of the demeaned design and per-column VIF; FE-absorbed shares per state."""
    Z = X_dm.to_numpy(float)
    Z = (Z - Z.mean(0)) / Z.std(0)
    C = np.corrcoef(Z, rowvar=False)
    lam = np.clip(np.linalg.eigvalsh(C)[::-1], 1e-300, None)
    p = lam / lam.sum()
    vif = np.diag(np.linalg.inv(C))
    block = dict(n_cols=len(names), cond=float(np.sqrt(lam[0] / lam[-1])),
                 effdim_participation=float(lam.sum() ** 2 / (lam ** 2).sum()),
                 effdim_entropy=float(np.exp(-(p * np.log(p)).sum())),
                 max_vif=float(vif.max()), max_vif_col=names[int(vif.argmax())],
                 eigenvalues=";".join(f"{v:.4f}" for v in lam))
    per_var = []
    alone = demean_variables_2way(work, s_cols, "entity_id", "time_id") if s_cols else None
    for j, nm in enumerate(names):
        s = nm[len("interaction_"):] if nm.startswith("interaction_") else None
        row = dict(column=nm, state=s, vif=float(vif[j]))
        rv = float(X_raw[nm].var())
        row["absorbed_interacted"] = 1.0 - float(X_dm[nm].var()) / rv if rv > 0 else np.nan
        if s is not None:
            sv = float(work[s].var())
            row["absorbed_alone"] = 1.0 - float(alone[s].var()) / sv if sv > 0 else np.nan
            row["national"] = s in NATIONAL
        per_var.append(row)
    return block, per_var


# ── (b) Level ────────────────────────────────────────────────────────────────────
def national_mean_phi(df: pd.DataFrame, phi_mt: np.ndarray) -> tuple:
    """The paper's Mean phi-hat: population-weighted national phi_t by quarter over
    (quarter, MCA) cells, averaged over quarters -- sleep_est_e2.calculate_pooled_phis and
    sleep_est_single._calculate_phis. Also the same over B markets only."""
    tmp = pd.DataFrame({"yq": df["year_quarter"].values, "m": df["mca_code"].values,
                        "phi": phi_mt, "M": df["pop_total"].fillna(0).values,
                        "D": df["is_D_row"].values})

    def _nat(t):
        agg = t.groupby(["yq", "m"], observed=True).agg(phi=("phi", "mean"), M=("M", "mean")).reset_index()
        num = (agg["phi"] * agg["M"]).groupby(agg["yq"]).sum()
        den = agg["M"].groupby(agg["yq"]).sum().replace(0, np.nan)
        return (num / den).fillna(0), agg

    nat_all, agg_all = _nat(tmp)
    nat_b, _ = _nat(tmp[~tmp["D"]])
    return nat_all, nat_b, agg_all


def level_metrics(df, work, s_cols, phi_fn, phi_mt_all):
    """Level of phi under one fitted block. phi_fn maps a frame of states to phi."""
    w = work["lagged_deposits"].to_numpy(float)
    mean_u = {s: float(work[s].mean()) for s in s_cols}
    mean_w = {s: float(np.average(work[s], weights=w)) for s in s_cols}
    phi_u = float(phi_fn(pd.DataFrame({s: [v] for s, v in mean_u.items()}))[0])
    phi_w = float(phi_fn(pd.DataFrame({s: [v] for s, v in mean_w.items()}))[0])
    phi_rows = phi_fn(work[s_cols])
    g = (work["nr_lagged_dep"] / work["lagged_deposits"]).to_numpy(float)
    nat_all, nat_b, agg_all = national_mean_phi(df, phi_mt_all)
    return dict(phi_at_unweighted_mean=phi_u, phi_at_depweighted_mean=phi_w,
                depweighted_mean_phi=float(np.average(phi_rows, weights=w)),
                mean_phi_hat=float(nat_all.mean()), mean_phi_hat_B_markets=float(nat_b.mean()),
                share_cells_phi_gt_1=float((agg_all["phi"] > 1.0).mean()),
                share_rows_phi_g_ge_1=float((phi_rows * g >= 1.0).mean()))


def run_e2(df, s_cols):
    df_fs = first_stage(df, ["constant"] + s_cols)
    work, X_raw, X_dm, y_dm, names = e2_design(df_fs, s_cols)
    Xd = X_dm.to_numpy(float)
    b, *_ = np.linalg.lstsq(Xd, y_dm.to_numpy(float), rcond=None)
    params = pd.Series(b, index=names)
    u = y_dm.to_numpy(float) - Xd @ b
    V = crve(Xd, u, work["CodConglomeradoPrudencial"].astype(str).values)
    k = 1 + len(s_cols)                        # level + slopes; the CF column is excluded
    Vab, Vbb = V[1:k, 0], V[1:k, 1:k]
    s_star = -np.linalg.solve(Vbb, Vab) if len(s_cols) else np.zeros(0)
    a_vec = np.r_[1.0, s_star]
    sd = work[s_cols].std().to_numpy(float) if len(s_cols) else np.zeros(0)

    def phi_fn(S):
        return params["nr_lagged_dep"] + sum(params[f"interaction_{s}"] * S[s].to_numpy(float) for s in s_cols)

    Sall = df[s_cols].fillna(0.0)
    phi_mt_all = np.asarray(params["nr_lagged_dep"] + sum(params[f"interaction_{s}"] * Sall[s].to_numpy(float)
                                                       for s in s_cols), float)
    lev = level_metrics(df, work, s_cols, phi_fn, phi_mt_all)
    lev.update(constant=float(params["nr_lagged_dep"]), se_constant=float(np.sqrt(V[0, 0])),
               phi_at_min_var_point=float(a_vec @ b[:k]),
               se_at_min_var_point=float(np.sqrt(a_vec @ V[:k, :k] @ a_vec)),
               min_var_point_max_abs_sd=float(np.max(np.abs(s_star / sd))) if len(s_cols) else 0.0,
               min_var_point_norm_sd=float(np.linalg.norm(s_star / sd)) if len(s_cols) else 0.0,
               n_obs=int(len(work)), n_clusters=int(work["CodConglomeradoPrudencial"].nunique()))
    tvals = pd.Series(b / np.sqrt(np.diag(V)), index=names)
    return params, tvals, lev, (work, X_raw, X_dm, names), s_star


def run_e3(df, s_cols):
    """E3 at spec 12: the Cauchy NLLS logit direction (production starts) and the monotone
    I-spline link -- sleep_est_single._exec_spec for kind single_index."""
    cols = ["constant"] + s_cols
    df_fs = first_stage(df, cols)
    lg = fit_nlls_link(df_fs, cols, has_cf=True, link="logit", loss="cauchy",
                       fe_time_col=R.FE_TIME_COL, bootstrap=False,
                       n_starts=int(os.environ.get("SLEEP_NLLS_N_STARTS", "4")))
    si = fit_single_index(df_fs, cols, has_cf=True, logit_res=lg, degree=3,
                          fe_time_col=R.FE_TIME_COL, phi_band=False)
    link = getattr(si, "link", None) or "index_sieve"

    def phi_fn(S):
        S = S.copy()
        S["nr_lagged_dep"] = 1.0
        return phi_from_native(S, si, link)

    phi_mt_all = phi_from_native(df, si, link)
    work = df_fs.dropna(subset=cols[1:] + ["deposit_balance", "nr_lagged_dep", "entity_id",
                                           "v_hat_x_lagged_dep"])
    lev = level_metrics(df, work, s_cols, phi_fn, phi_mt_all)
    # D rows are estimated at phi(S_t), the national population-weighted state; the model's
    # national sleepiness is the population-weighted average of the market phi_mt.
    d_phi = pd.Series(phi_mt_all[df["is_D_row"].values],
                      index=df.loc[df["is_D_row"], "year_quarter"].values).groupby(level=0).mean()
    _, nat_b, _ = national_mean_phi(df, phi_mt_all)
    gap = (d_phi - nat_b.reindex(d_phi.index)).dropna()
    lev.update(d_row_gap_mean=float(gap.mean()), d_row_gap_max_abs=float(gap.abs().max()),
               saturated_pct=float(getattr(si, "si_qp", {}).get("saturated_pct", np.nan)),
               n_obs=int(si.nobs))
    return si, lev


# ── Gates against production ─────────────────────────────────────────────────────
def _stored(est):
    with open(_paths.est_dir(est) / "estimation_results.pkl", "rb") as f:
        res = pickle.load(f)[SPEC12]["second_stage"]
    nat = pd.read_csv(_paths.est_dir(est) / "national_phi_t.csv")
    return res, float(nat[f"phi_t_{SPEC12_TAG}"].mean())


def gate_e2(params, lev):
    res, mp = _stored(2)
    stored = res.params.reindex(params.index)
    rel = float(np.max(np.abs(params - stored) / np.maximum(np.abs(stored), 1e-12)))
    d_mp = abs(lev["mean_phi_hat"] - mp)
    ok = rel < 1e-6 and d_mp < 1e-6
    print(f"  [gate E2] max rel diff of coefficients {rel:.2e}; mean phi {lev['mean_phi_hat']:.6f} "
          f"vs stored {mp:.6f} -> {'PASS' if ok else 'FAIL'}")
    return ok, dict(e2_max_rel_coef_diff=rel, e2_mean_phi=lev["mean_phi_hat"], e2_mean_phi_stored=mp)


def gate_e3(si, lev):
    res, mp = _stored(3)
    d_mp = abs(lev["mean_phi_hat"] - mp)
    nat_new, nat_old = si.params_native, res.params_native.reindex(si.params_native.index)
    d_th = float(np.max(np.abs(nat_new - nat_old)))
    ok = d_mp < 1e-4
    print(f"  [gate E3] mean phi {lev['mean_phi_hat']:.6f} vs stored {mp:.6f}; max |d theta| "
          f"{d_th:.2e} -> {'PASS' if ok else 'FAIL'}")
    return ok, dict(e3_mean_phi=lev["mean_phi_hat"], e3_mean_phi_stored=mp, e3_max_abs_theta_diff=d_th)


# ── Driver ───────────────────────────────────────────────────────────────────────
def main(skip_e3: bool, e3_blocks: list | None) -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    prod = production_tech_block()
    if prod != ["constant"] + PROD:
        print(f"[!] production Tech block {prod} differs from this script's S5 {PROD}; update PROD")
        return 2
    df = build_frame()
    blocks = candidate_blocks("frac_4g5g" in df.columns)
    print(f"  frame: {len(df):,} rows; blocks: {', '.join(blocks)}")

    design_rows, var_rows, e2_rows, coef_rows, gates = [], [], [], [], {}
    for name, s_cols in blocks.items():
        params, tvals, lev, (work, X_raw, X_dm, names), s_star = run_e2(df, s_cols)
        if name == "S5":
            ok, g = gate_e2(params, lev)
            gates.update(g)
            if not ok:
                print("[!] S5 does not reproduce the stored E2 fit -- stopping before any output")
                return 3
        blk, per_var = design_metrics(work, X_raw, X_dm, names, s_cols)
        design_rows.append(dict(block=name, n_states=len(s_cols), **blk))
        var_rows += [dict(block=name, **r) for r in per_var]
        e2_rows.append(dict(block=name, routine="E2", **lev))
        coef_rows += [dict(block=name, routine="E2", term=t, estimate=float(params[t]),
                           t_crve=float(tvals[t])) for t in params.index]
        print(f"  {name:<28s} cond {blk['cond']:7.2f}  effdim {blk['effdim_participation']:.2f}  "
              f"maxVIF {blk['max_vif']:7.1f} | E2 const {lev['constant']:.4f}  "
              f"phi(dep-wtd) {lev['phi_at_depweighted_mean']:.4f}  mean phi {lev['mean_phi_hat']:.4f}  "
              f"[{time.time() - t0:.0f}s]", flush=True)

    # An E3 refit takes 8 to 17 minutes, so the outputs are rewritten after every block: an
    # interrupted run keeps what it finished, and meta.json names the E3 blocks it holds.
    e3_rows = []

    def _write():
        pd.DataFrame(design_rows).to_csv(OUT / "design_by_block.csv", index=False)
        pd.DataFrame(var_rows).to_csv(OUT / "design_by_variable.csv", index=False)
        pd.DataFrame(e2_rows + e3_rows).to_csv(OUT / "level_by_block.csv", index=False)
        pd.DataFrame(coef_rows).to_csv(OUT / "coefficients_by_block.csv", index=False)
        meta = dict(written_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    spec=SPEC12, blocks=blocks, gates=gates,
                    e3_blocks_done=[r["block"] for r in e3_rows],
                    e3_blocks_requested=[] if skip_e3 else (e3_blocks or list(blocks)),
                    sleep_boot_b=os.environ.get("SLEEP_BOOT_B"), seconds=round(time.time() - t0, 1))
        (OUT / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    _write()
    if not skip_e3:
        todo = e3_blocks or list(blocks)
        for name in todo:
            si, lev = run_e3(df, blocks[name])
            if name == "S5":
                ok, g = gate_e3(si, lev)
                gates.update(g)
                if not ok:
                    print("[!] S5 does not reproduce the stored E3 fit -- E3 rows are not comparable")
                    return 4
            e3_rows.append(dict(block=name, routine="E3", **lev))
            coef_rows += [dict(block=name, routine="E3", term=t, estimate=float(si.params[t]),
                               t_crve=np.nan) for t in si.params.index]
            _write()
            print(f"  E3 {name:<25s} mean phi {lev['mean_phi_hat']:.4f}  phi(avg) "
                  f"{lev['phi_at_unweighted_mean']:.4f}  phi(dep-wtd) {lev['phi_at_depweighted_mean']:.4f}  "
                  f"D gap {lev['d_row_gap_mean']:+.4f}  [{time.time() - t0:.0f}s]", flush=True)

    print(f"\nWrote {OUT} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    pd.options.mode.chained_assignment = None
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--skip-e3", action="store_true", help="design and E2 only")
    ap.add_argument("--e3-blocks", nargs="*", default=None,
                    help="restrict the E3 refits to these block names (S5 always gates first)")
    a = ap.parse_args()
    e3b = a.e3_blocks
    if e3b is not None and "S5" not in e3b:
        e3b = ["S5"] + e3b
    raise SystemExit(main(a.skip_e3, e3b))

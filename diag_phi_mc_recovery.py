"""diag_phi_mc_recovery.py -- can the sleep kernel recover phi when awake inflows persist?

Author: Pedro Feijo de Moraes

The map rho_xi -> bias(phi-hat) through the production estimator. Panels are simulated
from the model's own law of motion (foundation_deposit_sim.jl:192):

    Dep_jt = phi_true * accr_j * Dep_j,t-1 + A_jt
    ln A_jt = ln a_j + delta_t + xi_jt,     xi_jt = rho_xi * xi_j,t-1 + eps_jt

with the stationary sd of xi held FIXED across rho_xi (sigma_eps = sigma_j*sqrt(1-rho^2)),
so the bias curve isolates persistence, not variance. a_j, sigma_j, accr_j, Dep_j0 and the
conglomerate structure are calibrated from demand_2_spec_12.parquet. The estimator is the
byte-identical E2 point kernel (same demean_variables_2way + OLS; the WCB inference layer
is skipped per rep and cross-checked once against run_pooled_second_stage).

Readings:
  * rho_xi = 0 recovers phi_true up to the Nickell/within distortion (~ -(1+phi)/T);
    that row validates the harness AND quantifies a distortion never measured before.
  * The (phi_true, rho_xi) pairs whose E[phi-hat] matches the production estimate form
    the observational-equivalence set of the headline number.
  * --stats-under-null: per-rep D2b (lagged-A coef) and D3 (a_j-rank x carry) statistics
    give their finite-sample null (rho=0) and power (rho>0) distributions.

Modes:
  --mode grid   (default) the phi_true x rho_xi recovery grid
  --mode acf    D8: within-entity, quarter-partialled ACF(h=1..12) of deposits in the
                REAL panel vs the 2.5-97.5% band from panels simulated at the fitted
                phi with iid inflows (rho_xi=0). Sim and data get the identical
                transform, so the small-T distortion cancels. A data ACF that decays
                faster at h>=4 than the band says h=1 persistence is not long-run
                non-waking.
CLI: --smoke (2x2 grid, R=25), --reps, --n-entities, --seed, --stats-under-null.
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import argparse
import os
os.environ.setdefault("MPLBACKEND", "Agg")

from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

from utils import paths as _paths

PARQUET = _paths.PROCESSED / "ESTIMATION_OUTPUT" / "DEMAND_PREP" / "demand_2_spec_12.parquet"
OUT_DIR = _paths.PROCESSED / "ESTIMATION_OUTPUT" / "DIAG_PHI_SEPARATION"
OUT_DIR.mkdir(parents=True, exist_ok=True)

T_KEEP = 36      # 2016Q1-2024Q4
T_BURN = 20


# ==============================================================================
# CALIBRATION (from the real demand-prep parquet)
# ==============================================================================
def calibrate(n_entities, seed):
    pq = pd.read_parquet(PARQUET, columns=["entity_id", "CodConglomeradoPrudencial",
                                           "Dep_Act", "gross_return_lag",
                                           "lagged_deposits", "time_id"])
    pq = pq[pq["Dep_Act"] > 0].copy()
    pq["_ln_a"] = np.log(pq["Dep_Act"])
    g = pq.groupby("entity_id")
    ent = pd.DataFrame({
        "ln_a": g["_ln_a"].mean(),
        "sig": g["_ln_a"].std(),
        "accr": g["gross_return_lag"].median(),
        "dep0": g["lagged_deposits"].first(),
        "congl": g["CodConglomeradoPrudencial"].first(),
        "nobs": g.size(),
    }).dropna()
    ent = ent[(ent["nobs"] >= 8) & (ent["sig"] > 0)]
    rng = np.random.default_rng(seed)
    ent = ent.sample(n=min(n_entities, len(ent)), random_state=seed)
    # aggregate log-shock scale: sd over time of the cross-entity mean of ln A
    sd_t = pq.groupby("time_id")["_ln_a"].mean().std()
    print(f"  calibration: {len(ent):,} entities, {ent['congl'].nunique()} conglomerates, "
          f"median sigma_j={ent['sig'].median():.2f}, sigma_delta={sd_t:.2f}")
    return ent.reset_index(drop=True), float(sd_t)


# ==============================================================================
# DGP + the E2 point kernel (byte-identical demean + OLS, no per-rep WCB)
# ==============================================================================
def simulate_panel(ent, sd_t, phi, rho, rng):
    N = len(ent)
    T = T_BURN + T_KEEP
    a = ent["ln_a"].to_numpy()
    sig = ent["sig"].to_numpy()
    accr = ent["accr"].to_numpy()
    sig_eps = sig * np.sqrt(1.0 - rho ** 2)
    xi = rng.normal(0.0, sig, N)                     # stationary start
    delta = rng.normal(0.0, sd_t, T)
    dep = ent["dep0"].to_numpy().copy()
    rows_dep = np.empty((T_KEEP, N))
    rows_lag = np.empty((T_KEEP, N))
    for t in range(T):
        xi = rho * xi + rng.normal(0.0, sig_eps, N)
        A = np.exp(a + delta[t] + xi)
        dep_new = phi * accr * dep + A
        if t >= T_BURN:
            rows_lag[t - T_BURN] = accr * dep       # nr_lagged_dep
            rows_dep[t - T_BURN] = dep_new
        dep = dep_new
    return rows_dep, rows_lag


def within_2way(M, einv, ec, tinv, tc, n_iter=15, tol=1e-9):
    M = M.copy()
    for _ in range(n_iter):
        prev = M.copy()
        for j in range(M.shape[1]):
            M[:, j] -= (np.bincount(einv, M[:, j]) / ec)[einv]
            M[:, j] -= (np.bincount(tinv, M[:, j]) / tc)[tinv]
        if np.max(np.abs(M - prev)) < tol:
            break
    return M


def estimate_once(rows_dep, rows_lag, ent, stats_null=False, rho_for_lagA=None):
    T, N = rows_dep.shape
    y = rows_dep.reshape(-1)                        # t-major: rows 0..T-1 stacked
    z = rows_lag.reshape(-1)
    einv = np.tile(np.arange(N), T)
    tinv = np.repeat(np.arange(T), N)
    ec = np.bincount(einv).astype(float)
    tc = np.bincount(tinv).astype(float)
    y_dm = within_2way(y[:, None], einv, ec, tinv, tc)[:, 0]
    z_dm = within_2way(z[:, None], einv, ec, tinv, tc)[:, 0]
    phi_hat = float(z_dm @ y_dm / (z_dm @ z_dm))
    out = {"phi_hat": phi_hat}

    if stats_null:
        # D2b analog: augment with the lagged ACCOUNTING awake inflow at the estimated
        # phi (exactly what the data test can observe). Everything is re-demeaned on the
        # keep subsample so the augmented design is internally consistent.
        A_acc = y - phi_hat * z                       # Dep_t - phi_hat*(accr*Dep_{t-1})
        A_lag = np.full_like(A_acc, np.nan)
        A_lag[N:] = A_acc[:-N]
        keep = ~np.isnan(A_lag)
        einv_k, tinv_k = einv[keep], tinv[keep] - 1
        ec_k = np.bincount(einv_k).astype(float)
        tc_k = np.bincount(tinv_k).astype(float)
        Mk = within_2way(np.column_stack([y[keep], z[keep], A_lag[keep]]),
                         einv_k, ec_k, tinv_k, tc_k)
        yk, X = Mk[:, 0], Mk[:, 1:]
        b, *_ = np.linalg.lstsq(X, yk, rcond=None)
        e = yk - X @ b
        # cluster-by-conglomerate t on the A_lag coefficient
        congl = pd.factorize(ent["congl"])[0]
        cl = np.tile(congl, T)[keep]
        XtXi = np.linalg.inv(X.T @ X)
        G = np.zeros((cl.max() + 1, X.shape[1]))
        np.add.at(G, cl, X * e[:, None])
        V = XtXi @ (G.T @ G) @ XtXi
        out["d2b_coef"] = float(b[1])
        out["d2b_t"] = float(b[1] / np.sqrt(V[1, 1]))
        out["phi_hat_aug"] = float(b[0])
        # D3 analog: a_j-rank x carry
        rank = ent["ln_a"].rank(pct=True).to_numpy() - 0.5
        rz = z * np.tile(rank, T)
        rz_dm = within_2way(rz[:, None], einv, ec, tinv, tc)[:, 0]
        X3 = np.column_stack([z_dm, rz_dm])
        b3, *_ = np.linalg.lstsq(X3, y_dm, rcond=None)
        e3 = y_dm - X3 @ b3
        cl3 = np.tile(congl, T)
        X3tXi = np.linalg.inv(X3.T @ X3)
        G3 = np.zeros((cl3.max() + 1, 2))
        np.add.at(G3, cl3, X3 * e3[:, None])
        V3 = X3tXi @ (G3.T @ G3) @ X3tXi
        out["d3_coef"] = float(b3[1])
        out["d3_t"] = float(b3[1] / np.sqrt(V3[1, 1]))
    return out


def run_cell(args):
    ent_rec, sd_t, phi, rho, reps, seed, stats_null = args
    ent = pd.DataFrame(ent_rec)
    res = []
    for r in range(reps):
        rng = np.random.default_rng((seed, int(phi * 1000), int(rho * 100), r))
        rows_dep, rows_lag = simulate_panel(ent, sd_t, phi, rho, rng)
        out = estimate_once(rows_dep, rows_lag, ent, stats_null=stats_null)
        out.update({"phi_true": phi, "rho_xi": rho, "rep": r})
        res.append(out)
    return res


# ==============================================================================
# MODES
# ==============================================================================
def mode_grid(a):
    print("=== D1: Monte Carlo recovery of phi through the E2 kernel ===")
    ent, sd_t = calibrate(a.n_entities, a.seed)

    if a.smoke:
        phis, rhos, reps = [0.7, 0.985], [0.0, 0.9], 25
    else:
        phis, rhos, reps = [0.5, 0.7, 0.9, 0.985], [0.0, 0.3, 0.6, 0.9], a.reps

    # one-off cross-check: fast kernel == run_pooled_second_stage on the same panel
    rng = np.random.default_rng(a.seed)
    rows_dep, rows_lag = simulate_panel(ent, sd_t, 0.9, 0.0, rng)
    fast = estimate_once(rows_dep, rows_lag, ent)["phi_hat"]
    from estimation_2_sleep import run_pooled_second_stage
    T, N = rows_dep.shape
    df_chk = pd.DataFrame({
        "deposit_balance": rows_dep.reshape(-1), "nr_lagged_dep": rows_lag.reshape(-1),
        "entity_id": np.tile(ent.index.astype(str), T),
        "time_id": np.repeat([f"t{t}" for t in range(T)], N),
        "CodConglomeradoPrudencial": np.tile(ent["congl"].astype(str), T),
        "constant": 1.0})
    res_prod = run_pooled_second_stage(df_chk, ["constant"], has_cf=False)
    prod = float(res_prod.params["nr_lagged_dep"])
    print(f"  kernel cross-check: fast={fast:.10f}  production={prod:.10f}  "
          f"diff={abs(fast - prod):.2e}")
    assert abs(fast - prod) < 1e-8, "fast kernel diverges from run_pooled_second_stage"

    ent_rec = ent.to_dict("list")
    cells = [(ent_rec, sd_t, p, r, reps, a.seed, a.stats_under_null)
             for p in phis for r in rhos]
    rows = []
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
        for res in ex.map(run_cell, cells):
            rows += res
            c = res[0]
            m = np.mean([x["phi_hat"] for x in res])
            print(f"  phi_true={c['phi_true']:.3f} rho_xi={c['rho_xi']:.1f}: "
                  f"E[phi_hat]={m:.4f}  bias={m - c['phi_true']:+.4f}  (R={len(res)})")

    dfres = pd.DataFrame(rows)
    tag = "smoke" if a.smoke else "full"
    dfres.to_csv(OUT_DIR / f"d1_mc_recovery_{tag}.csv", index=False)

    # summary + money plot
    summ = dfres.groupby(["phi_true", "rho_xi"])["phi_hat"].agg(["mean", "std"]).reset_index()
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for p in sorted(summ["phi_true"].unique()):
        s = summ[summ["phi_true"] == p]
        ax1.plot(s["rho_xi"], s["mean"] - s["phi_true"], "o-", label=f"phi={p}")
    ax1.axhline(0, lw=0.8, color="0.5")
    ax1.set_xlabel("rho_xi (awake-inflow persistence)")
    ax1.set_ylabel("E[phi_hat] - phi_true")
    ax1.set_title("bias of the E2 kernel")
    ax1.legend()
    piv = summ.pivot(index="phi_true", columns="rho_xi", values="mean")
    im = ax2.imshow(piv.to_numpy(), origin="lower", aspect="auto", cmap="viridis")
    ax2.set_xticks(range(len(piv.columns)), [f"{c:g}" for c in piv.columns])
    ax2.set_yticks(range(len(piv.index)), [f"{i:g}" for i in piv.index])
    ax2.set_xlabel("rho_xi")
    ax2.set_ylabel("phi_true")
    ax2.set_title("E[phi_hat] (contour: production phi)")
    try:
        ax2.contour(piv.to_numpy(), levels=[a.phi_prod], colors="red")
    except Exception:
        pass
    fig.colorbar(im, ax=ax2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"d1_mc_recovery_{tag}.png", dpi=150)
    print(f"\n  outputs -> d1_mc_recovery_{tag}.csv / .png")

    base = summ[summ["rho_xi"] == 0.0]
    print("\n  VERDICT inputs:")
    for _, r in base.iterrows():
        print(f"    rho=0 check: phi_true={r['phi_true']:.3f} -> E[phi_hat]={r['mean']:.4f} "
              f"(Nickell-type distortion {r['mean'] - r['phi_true']:+.4f})")
    print("  Any cell whose E[phi_hat] reaches the production estimate with phi_true far")
    print("  below it is an observationally-equivalent DGP: the headline phi is not")
    print("  point-identified without a stand on rho_xi (which D2b/D3 estimate from data).")


def mode_acf(a):
    print("=== D8: ACF overidentification vs fitted pure-sleepiness model ===")
    from diag_phi_augmented_tests import load_sleep_frame
    from estimation_2_sleep import demean_variables_2way

    df, _ = load_sleep_frame()
    d = df.dropna(subset=["deposit_balance", "nr_lagged_dep"]).copy()
    d["_e"] = demean_variables_2way(d, ["deposit_balance"], "entity_id", "time_id")["deposit_balance"]

    def panel_acf(dd, col, hmax=12):
        out = []
        base = dd[["entity_id", "qidx", col]]
        for h in range(1, hmax + 1):
            lag = base.copy()
            lag["qidx"] = lag["qidx"] + h
            m = base.merge(lag, on=["entity_id", "qidx"], suffixes=("", "_l"))
            out.append(np.corrcoef(m[col], m[col + "_l"])[0, 1] if len(m) > 100 else np.nan)
        return np.array(out)

    acf_data = panel_acf(d, "_e")
    print("  data ACF(h=1..12):", np.round(acf_data, 3))

    ent, sd_t = calibrate(a.n_entities, a.seed)
    phi_fit = a.phi_prod
    bands = []
    for r in range(a.reps_acf):
        rng = np.random.default_rng((a.seed, 999, r))
        rows_dep, rows_lag = simulate_panel(ent, sd_t, phi_fit, 0.0, rng)
        T, N = rows_dep.shape
        sd = pd.DataFrame({"deposit_balance": rows_dep.reshape(-1),
                           "entity_id": np.tile(np.arange(N), T),
                           "time_id": np.repeat(np.arange(T), N),
                           "qidx": np.repeat(np.arange(T), N)})
        sd["_e"] = demean_variables_2way(sd, ["deposit_balance"], "entity_id", "time_id")["deposit_balance"]
        bands.append(panel_acf(sd, "_e"))
    B = np.vstack(bands)
    lo, hi = np.nanpercentile(B, 2.5, axis=0), np.nanpercentile(B, 97.5, axis=0)

    import matplotlib.pyplot as plt
    h = np.arange(1, 13)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.fill_between(h, lo, hi, alpha=0.3, label=f"fitted model band (phi={phi_fit}, iid inflows)")
    ax.plot(h, acf_data, "o-", color="crimson", label="data")
    ax.set_xlabel("horizon h (quarters)")
    ax.set_ylabel("within-entity, quarter-partialled ACF of deposits")
    ax.set_title("D8: medium-run persistence vs pure-sleepiness model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "d8_acf_overid.png", dpi=150)
    pd.DataFrame({"h": h, "acf_data": acf_data, "band_lo": lo, "band_hi": hi}
                 ).to_csv(OUT_DIR / "d8_acf_overid.csv", index=False)
    inside = np.mean((acf_data >= lo) & (acf_data <= hi))
    print(f"  share of horizons inside the band: {inside:.0%}  -> d8_acf_overid.png/.csv")
    print("\n  VERDICT: data ACF exiting the band DOWNWARD at h>=4 means the h=1")
    print("  persistence the kernel fits is not the same object as long-run non-waking;")
    print("  inside-the-band means geometric carry is consistent with medium-run dynamics.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["grid", "acf"], default="grid")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--reps-acf", type=int, default=200)
    ap.add_argument("--n-entities", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stats-under-null", action="store_true")
    ap.add_argument("--phi-prod", type=float, default=0.92,
                    help="production phi for the iso-contour / D8 band (E2 spec-12 median 0.92; "
                         "use 0.985 for the E7/E8 headline)")
    a = ap.parse_args()
    (mode_grid if a.mode == "grid" else mode_acf)(a)

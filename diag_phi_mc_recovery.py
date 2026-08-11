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
  --mode power  D1-power: read the ARCHIVED d1_mc_recovery_full.csv and turn the D2b/D3
                per-rep statistics into honest power curves. The per-rep statistic is a
                CRVE t on ~35 simulated conglomerates compared to 1.96, whose MEASURED
                size under rho_xi=0 is 37-48%, so raw rejection frequencies are not power
                at 5%. Two size-honest tables are emitted: (a) size-corrected t (critical
                value = 95th pct of |t| under rho_xi=0 AT THE SAME phi_true) and (b)
                coefficient-based (critical value = 95th pct of the coefficient under
                rho_xi=0). No re-simulation.

D8 sub-modes (all default OFF, so the archived d8_acf_overid.* are reproducible):
  --detrend     remove an entity-specific LINEAR TREND, after the two-way within
                transform, from the data AND from every simulated panel identically.
                Tests explanation (i) for the D8 gap: real conglomerate x type x MCA
                deposits trend (nominal growth, multi-year share gains); additive quarter
                FE remove the common component but not firm-specific slopes, and a
                trending level has a near-unit-root within-ACF. The DGP has no trend, so
                the sim side loses only degrees of freedom -- which is exactly why both
                sides must be detrended with the same estimator.
  --by type|size  data ACF within deposit-type groups / within size bins of entities
                (size = mean deposit_balance), each against a band simulated from THAT
                group's own calibration. Tests explanation (ii): a mixture of geometric
                decays is convex in h, so it decays more slowly at long horizons than its
                average component, and the pooled ACF is variance-weighted toward the
                largest, most inert conglomerates. With --fit-phi the run also simulates a
                MIXTURE panel (every entity carried at its own group's fitted phi) and
                confronts it with the POOLED data ACF -- the direct test of whether
                heterogeneity alone reproduces the pooled gap.
  --fit-phi     simulated-minimum-distance phi: grid phi, R panels per grid point, report
                argmin of SSE over h=1..12 against the mean simulated ACF, with a
                grid-profile (Monte-Carlo test inversion) interval, next to the h=1-only
                fit and next to D10's entry-dynamics phi read from d10_implied_phi.csv.

Grid sub-mode:
  --censor      apply the demand-step rule inside each simulated panel before the D2b
                statistic: Dep_Act = max(0, Dep_t - phi_hat*accr*Dep_{t-1}), rows with
                Dep_Act <= 1e-6 raw R$ dropped (estimation_2_demand_1_prep.py:314-322).
                The data test can only use surviving rows, so this is what D2b's power
                actually is.

CLI: --smoke (2x2 grid, R=25), --reps, --n-entities, --seed, --stats-under-null, --tag.
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
HMAX = 12        # ACF horizons; 12 quarters = 3 years

# The demand step's censoring rule, in RAW R$ (the simulated panels inherit the parquet's
# raw-R$ units through dep0=lagged_deposits): Dep_Act = max{0, Dep - phi*g*L} and rows with
# Dep_Act <= CENSOR_EPS are dropped -- estimation_2_demand_1_prep.py:314-322.
CENSOR_EPS = 1e-6

# desc_2.py palette
PAL_B, PAL_D, PAL_GRID = "#1565C0", "#E64A19", "#D5D5D0"

D10_CSV = OUT_DIR / "d10_implied_phi.csv"


def d10_reference(kind="B"):
    """D10's entry-dynamics phi as (phi, lo, hi), READ from disk -- never hard-coded, so
    that re-running diag_entry_dynamics.py updates this script's comparison automatically.
    Returns None if D10 has not been run."""
    try:
        t = pd.read_csv(D10_CSV)
        r = t[t["kind"] == kind].iloc[0]
        return float(r["phi_entry"]), float(r["lo"]), float(r["hi"])
    except Exception:
        return None


# ==============================================================================
# CALIBRATION (from the real demand-prep parquet)
# ==============================================================================
_ENT_CACHE = {}


def _entity_table(size_bins):
    """Per-entity calibration table + sigma_delta, cached within a process.

    `dtype` and `size_bin` are carried so the D8 group modes can calibrate a band from
    the SAME population the data group is drawn from. size_bin is cut on the FULL entity
    population (equal counts of entities, rank-based) BEFORE any sampling, so group
    membership never depends on the draw."""
    if size_bins in _ENT_CACHE:
        return _ENT_CACHE[size_bins]
    pq = pd.read_parquet(PARQUET, columns=["entity_id", "CodConglomeradoPrudencial",
                                           "Dep_Act", "gross_return_lag",
                                           "lagged_deposits", "time_id",
                                           "deposit_type", "deposit_balance"])
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
        "dtype": g["deposit_type"].first(),
        "mean_dep": g["deposit_balance"].mean(),
    })
    # dropna on the ORIGINAL columns only: `dtype`/`mean_dep` are additions and must not
    # be able to change the sample the archived runs were drawn from.
    ent = ent.dropna(subset=["ln_a", "sig", "accr", "dep0", "congl", "nobs"])
    ent = ent[(ent["nobs"] >= 8) & (ent["sig"] > 0)]
    # entity_id survives as a COLUMN: calibrate() resets the index, and the routine-band
    # mode needs the id to map each sampled entity to its routine-fitted phi.
    ent = ent.reset_index()
    ent["size_bin"] = pd.qcut(ent["mean_dep"].rank(method="first"),
                              size_bins, labels=False).astype(int)
    # aggregate log-shock scale: sd over time of the cross-entity mean of ln A
    sd_t = float(pq.groupby("time_id")["_ln_a"].mean().std())
    _ENT_CACHE[size_bins] = (ent, sd_t)
    return ent, sd_t


def calibrate(n_entities, seed, group=None, size_bins=3):
    """Draw the simulation's entities. group=None reproduces the archived calibration
    exactly; group=("type", k) / ("size", b) restricts to one group first and then draws
    up to n_entities WITHIN it, so every band has a comparable cross-section."""
    ent, sd_t = _entity_table(size_bins)
    tag = ""
    if group is not None:
        kind, val = group
        col = "dtype" if kind == "type" else "size_bin"
        ent = ent[ent[col] == val]
        tag = f"[{kind}={val}] "
    ent = ent.sample(n=min(n_entities, len(ent)), random_state=seed)
    print(f"  {tag}calibration: {len(ent):,} entities, {ent['congl'].nunique()} conglomerates, "
          f"median sigma_j={ent['sig'].median():.2f}, sigma_delta={sd_t:.2f}")
    return ent.reset_index(drop=True), float(sd_t)


# ==============================================================================
# DGP + the E2 point kernel (byte-identical demean + OLS, no per-rep WCB)
# ==============================================================================
def simulate_panel(ent, sd_t, phi, rho, rng):
    """phi may be a scalar OR a length-N vector (heterogeneous phi_j); the recursion is
    already elementwise in j, so the mixture DGP needs no separate code path."""
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


def detrend_by_entity(v, einv, tvals):
    """Residual of v on an entity-specific intercept AND linear trend in `tvals`.

    Used on BOTH sides of D8 --detrend. GOTCHA: this is not innocuous on the simulation
    side either -- with T=36 an entity-specific slope eats a real degree of freedom and
    mechanically bends the long-horizon ACF down. That is precisely why the band must be
    recomputed from detrended SIMULATED panels rather than compared to the old band."""
    ec = np.bincount(einv).astype(float)
    tbar = np.bincount(einv, tvals) / ec
    vbar = np.bincount(einv, v) / ec
    x = tvals - tbar[einv]
    y = v - vbar[einv]
    den = np.bincount(einv, x * x)
    num = np.bincount(einv, x * y)
    slope = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
    return y - slope[einv] * x


def estimate_once(rows_dep, rows_lag, ent, stats_null=False, censor=False):
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
        if censor:
            # The demand step stores Dep_Act = max{0, A} and DROPS Dep_Act <= 1e-6, so a
            # row whose t-1 inflow was censored has no regressor at all (the parquet merge
            # returns NaN and run_augmented's dropna removes it). Selection is therefore on
            # the LAGGED accounting inflow only -- deposit_balance and nr_lagged_dep come
            # from the uncensored sleep frame. max{0,.} itself is irrelevant on survivors
            # (there A > 1e-6 already), so only the sign selection bites.
            keep &= np.nan_to_num(A_lag, nan=-np.inf) > CENSOR_EPS
            out["censor_kept"] = float(keep[N:].mean())
        einv_k, tinv_k = einv[keep], tinv[keep] - 1
        # after censoring the surviving entities/quarters are no longer 0..N-1 / 0..T-2
        einv_k = np.unique(einv_k, return_inverse=True)[1]
        tinv_k = np.unique(tinv_k, return_inverse=True)[1]
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
        cl = np.unique(cl, return_inverse=True)[1]
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
        # NB --censor deliberately does NOT touch D3: the data-side D3 (attractiveness rank
        # x carry, diag_phi_interaction_tests.py --arm attractiveness) runs on the sleep
        # frame and never reads Dep_Act, so it never loses the censored rows.
        out["d3_coef"] = float(b3[1])
        out["d3_t"] = float(b3[1] / np.sqrt(V3[1, 1]))
    return out


def run_cell(args):
    ent_rec, sd_t, phi, rho, reps, seed, stats_null = args[:7]
    censor = args[7] if len(args) > 7 else False
    ent = pd.DataFrame(ent_rec)
    res = []
    for r in range(reps):
        rng = np.random.default_rng((seed, int(phi * 1000), int(rho * 100), r))
        rows_dep, rows_lag = simulate_panel(ent, sd_t, phi, rho, rng)
        out = estimate_once(rows_dep, rows_lag, ent, stats_null=stats_null, censor=censor)
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
    if a.phis:
        phis = a.phis
    if a.rhos:
        rhos = a.rhos

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
    cells = [(ent_rec, sd_t, p, r, reps, a.seed, a.stats_under_null, a.censor)
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
    tag = a.tag or ("smoke" if a.smoke else "full")
    if a.censor and not a.tag:
        tag += "_censored"
    dfres.to_csv(OUT_DIR / f"d1_mc_recovery_{tag}.csv", index=False)
    if "censor_kept" in dfres.columns:
        print(f"  censoring keeps {dfres['censor_kept'].mean():.1%} of the D2b rows "
              f"(data loses 33-40%)")

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


# ------------------------------------------------------------------------------
# D8 machinery
# ------------------------------------------------------------------------------
def _resid_2way(rows_dep, detrend=False):
    """(T,N) simulated deposits -> two-way within residual, optionally entity-detrended."""
    T, N = rows_dep.shape
    y = rows_dep.reshape(-1)
    einv = np.tile(np.arange(N), T)
    tinv = np.repeat(np.arange(T), N)
    e = within_2way(y[:, None], einv, np.bincount(einv).astype(float),
                    tinv, np.bincount(tinv).astype(float))[:, 0]
    if detrend:
        e = detrend_by_entity(e, einv, tinv.astype(float))
    return e.reshape(T, N)


def _acf_balanced(e, hmax=HMAX):
    return np.array([np.corrcoef(e[h:].ravel(), e[:-h].ravel())[0, 1]
                     for h in range(1, hmax + 1)])


SIM_KEYS = ("ln_a", "sig", "accr", "dep0")   # all simulate_panel needs for the ACF sims


def _acf_sim_block(job):
    """A BLOCK of fitted-model panels -> (len(reps), hmax) ACFs. Top-level for
    process-pool pickling. Balanced panel, so the lag alignment is a plain (T,N) reshape;
    `phi` is a scalar or a per-entity list (mixture DGP).

    GOTCHA (why blocks and not one job per replication): the calibration table has to
    travel to the worker with every job. At one job per replication the --fit-phi grid
    (60 phi x R) spent all of its wall clock pickling ~1.5 GB of entity tables and
    rebuilding a DataFrame 12,000 times. Blocking amortises both; the rng key is still
    (seed, 999, rep), so every replication draws exactly what it drew before."""
    ent = pd.DataFrame(job["ent"])
    phi = job["phi"]
    if isinstance(phi, (list, np.ndarray)):
        phi = np.asarray(phi, dtype=float)
    out = np.empty((len(job["reps"]), job["hmax"]))
    for i, r in enumerate(job["reps"]):
        rng = np.random.default_rng((job["seed"], 999, r))
        rows_dep, _ = simulate_panel(ent, job["sd_t"], phi, 0.0, rng)
        out[i] = _acf_balanced(_resid_2way(rows_dep, job["detrend"]), job["hmax"])
    return out


def _sim_grid(ex, ent_rec, sd_t, phis, seed, reps, detrend, hmax=HMAX, block=25):
    """(len(phis), reps, hmax) simulated ACFs. One map() over ALL (phi, block) jobs so the
    pool load-balances across the whole phi grid instead of syncing at each grid point."""
    ent_rec = {k: ent_rec[k] for k in SIM_KEYS}
    jobs = []
    for gi, phi in enumerate(phis):
        for s in range(0, reps, block):
            jobs.append({"ent": ent_rec, "sd_t": sd_t, "phi": phi, "seed": seed,
                         "reps": list(range(s, min(s + block, reps))),
                         "detrend": detrend, "hmax": hmax, "gi": gi, "s": s})
    A = np.empty((len(phis), reps, hmax))
    for job, res in zip(jobs, ex.map(_acf_sim_block, jobs, chunksize=1)):
        A[job["gi"], job["s"]:job["s"] + len(job["reps"])] = res
    return A


def _sim_band(ex, ent_rec, sd_t, phi, seed, reps, detrend, hmax=HMAX):
    """R simulated ACFs at `phi` -> (R,hmax) matrix."""
    return _sim_grid(ex, ent_rec, sd_t, [phi], seed, reps, detrend, hmax)[0]


def _data_acf(dd, col, hmax=HMAX):
    """Unbalanced-panel ACF by exact quarter alignment (merge, not shift)."""
    out = []
    base = dd[["entity_id", "qidx", col]]
    for h in range(1, hmax + 1):
        lag = base.copy()
        lag["qidx"] = lag["qidx"] + h
        m = base.merge(lag, on=["entity_id", "qidx"], suffixes=("", "_l"))
        out.append(np.corrcoef(m[col], m[col + "_l"])[0, 1] if len(m) > 100 else np.nan)
    return np.array(out)


def _transform_data(dd, detrend):
    """Two-way within residual of deposits (+ optional entity detrend) -- the DATA side of
    the identical transform applied to every simulated panel."""
    from estimation_2_sleep import demean_variables_2way
    e = demean_variables_2way(dd, ["deposit_balance"], "entity_id",
                              "time_id")["deposit_balance"]
    if detrend:
        e = pd.Series(detrend_by_entity(e.to_numpy(), pd.factorize(dd["entity_id"])[0],
                                        dd["qidx"].to_numpy(dtype=float)), index=dd.index)
    return e


def _fit_phi_md(ex, ent_rec, sd_t, acf_data, grid, reps, seed, detrend, hmax=HMAX):
    """Simulated minimum distance on the whole autocorrelogram.

    SSE(phi) = sum_h (acf_data_h - E_sim[acf_h | phi])^2 over h=1..hmax. The interval is a
    Monte-Carlo TEST INVERSION, not a delta-method CI: phi is retained when the data's SSE
    is below the 95th percentile of the SSE that the model's OWN replications produce at
    that phi (each replication scored against the leave-one-out mean, so the reference
    distribution is not shrunk by the replication's own contribution to the mean).
    Common random numbers across the grid (the rng key ignores phi) keep SSE(phi) smooth."""
    A = _sim_grid(ex, ent_rec, sd_t, [float(p) for p in grid], seed, reps,
                  detrend, hmax)                                    # (G,R,H)
    m = np.nanmean(A, axis=1)                                       # (G,H)
    sse_data = np.nansum((acf_data[None, :] - m) ** 2, axis=1)
    loo = (reps * m[:, None, :] - A) / (reps - 1)
    sse_sim = np.nansum((A - loo) ** 2, axis=2)                     # (G,R)
    crit = np.nanpercentile(sse_sim, 95, axis=1)
    ok = sse_data <= crit
    phi_md = float(grid[int(np.nanargmin(sse_data))])
    phi_h1 = float(grid[int(np.nanargmin(np.abs(m[:, 0] - acf_data[0])))])
    ci = (float(grid[ok].min()), float(grid[ok].max())) if ok.any() else (np.nan, np.nan)
    prof = pd.DataFrame({"phi": grid, "sse_data": sse_data, "sse_crit95": crit,
                         "accepted": ok, "sim_acf_h1": m[:, 0]})
    # an endpoint ON the grid boundary is a censored interval, not an estimate -- say so
    edge = ok.any() and (ok[0] or ok[-1])
    return {"phi_md": phi_md, "phi_h1": phi_h1, "ci_lo": ci[0], "ci_hi": ci[1],
            "sse_min": float(np.nanmin(sse_data)), "profile": prof, "at_grid_edge": bool(edge),
            "band_at_md": A[int(np.nanargmin(sse_data))]}


def _inside(acf, lo, hi):
    ins = (acf >= lo) & (acf <= hi)
    return ins, acf > hi, acf < lo


def _hlist(mask):
    """horizons flagged by a boolean mask, as plain ints (numpy 2 prints np.int64(...))"""
    return [int(x) for x in np.arange(1, len(mask) + 1)[mask]]


def mode_acf(a):
    print("=== D8: ACF overidentification vs fitted pure-sleepiness model ===")
    from diag_phi_augmented_tests import load_sleep_frame

    df, _ = load_sleep_frame()
    d = df.dropna(subset=["deposit_balance", "nr_lagged_dep"]).copy()

    detr = bool(a.detrend)
    suffix = ("" if a.by == "none" else f"_by{a.by}" + (f"{a.size_bins}" if a.by == "size" else "")) \
             + ("_detrend" if detr else "")
    base_name = "d8_acf_overid" + suffix
    grid = np.round(np.arange(a.fit_lo, a.fit_hi + 1e-9, a.fit_step), 4)

    # ---- pooled data ACF (always computed: it is the reference for every variant) ----
    d["_e"] = _transform_data(d, detr)
    acf_pool = _data_acf(d, "_e")
    print(f"  data ACF(h=1..{HMAX}){' [entity-detrended]' if detr else ''}:",
          np.round(acf_pool, 3))

    # ---- group definition (data side mirrors calibrate()'s bins) --------------------
    if a.by == "type":
        keys = sorted(int(k) for k in d["deposit_type"].dropna().unique())
        gsel = [(f"type{k}", d["deposit_type"] == k, ("type", k)) for k in keys]
    elif a.by == "size":
        # RANK-based bins on both sides. The sleep frame (43k entities, uncensored) and the
        # calibration parquet (38k entities, Dep_Act>0) are different populations and their
        # levels are in different units (R$ bn vs raw R$), so "tercile k" is defined as the
        # k-th tercile OF ITS OWN population -- comparable, but not the same entity list.
        msize = d.groupby("entity_id")["deposit_balance"].mean()
        bins = pd.qcut(msize.rank(method="first"), a.size_bins, labels=False).astype(int)
        b = d["entity_id"].map(bins)
        gsel = [(f"size{k}", b == k, ("size", k)) for k in range(a.size_bins)]
    else:
        gsel = [("pooled", pd.Series(True, index=d.index), None)]

    rows, fits, panels = [], {}, []
    ent_all = sd_all = None
    pooled_ref_inside = None
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
        # POOLED REFERENCE at the common phi. Always computed and always panels[0], so the
        # VERDICT is about the same object in every sub-mode and the mixture below is scored
        # against the right baseline. (An earlier version compared the mixture band with
        # panels[0] = the FIRST GROUP, and duly reported "heterogeneity does not close the
        # gap" while the pooled gap went 17% -> 100%: the comparison, not the finding.)
        if a.by != "none":
            ent_all, sd_all = calibrate(a.n_entities, a.seed, size_bins=a.size_bins)
            B0 = _sim_band(ex, ent_all.to_dict("list"), sd_all, a.phi_prod, a.seed,
                           a.reps_acf, detr)
            lo0, hi0 = np.nanpercentile(B0, 2.5, axis=0), np.nanpercentile(B0, 97.5, axis=0)
            ins0, above0, below0 = _inside(acf_pool, lo0, hi0)
            pooled_ref_inside = float(ins0.mean())
            print(f"  [pooled(common phi)] reference: inside band at phi={a.phi_prod}: "
                  f"{ins0.mean():.0%}  above: {_hlist(above0)}  below: {_hlist(below0)}")
            for i, h in enumerate(range(1, HMAX + 1)):
                rows.append({"group": "pooled(common phi)", "h": h, "acf_data": acf_pool[i],
                             "band_lo": lo0[i], "band_hi": hi0[i], "inside": bool(ins0[i])})
            panels.append(("pooled(common phi)", acf_pool, lo0, hi0))

        for label, mask, gspec in gsel:
            dd = d if gspec is None else d[mask].copy()
            if gspec is not None:
                dd["_e"] = _transform_data(dd, detr)   # re-demean WITHIN the group, so the
                                                       # data transform matches the sim's
            acf_data = acf_pool if gspec is None else _data_acf(dd, "_e")
            ent, sd_t = calibrate(a.n_entities, a.seed, group=gspec, size_bins=a.size_bins)
            ent_rec = ent.to_dict("list")
            B = _sim_band(ex, ent_rec, sd_t, a.phi_prod, a.seed, a.reps_acf, detr)
            lo, hi = np.nanpercentile(B, 2.5, axis=0), np.nanpercentile(B, 97.5, axis=0)
            ins, above, below = _inside(acf_data, lo, hi)
            print(f"  [{label}] n={len(dd):,} rows, {dd['entity_id'].nunique():,} entities | "
                  f"inside band at phi={a.phi_prod}: {ins.mean():.0%}  "
                  f"above: {_hlist(above)}  "
                  f"below: {_hlist(below)}")
            for i, h in enumerate(range(1, HMAX + 1)):
                rows.append({"group": label, "h": h, "acf_data": acf_data[i],
                             "band_lo": lo[i], "band_hi": hi[i], "inside": bool(ins[i])})
            panels.append((label, acf_data, lo, hi))

            if a.fit_phi:
                f = _fit_phi_md(ex, ent_rec, sd_t, acf_data, grid, a.reps_fit,
                                a.seed, detr)
                lo2 = np.nanpercentile(f["band_at_md"], 2.5, axis=0)
                hi2 = np.nanpercentile(f["band_at_md"], 97.5, axis=0)
                ins2, _, _ = _inside(acf_data, lo2, hi2)
                f["inside_at_md"] = float(ins2.mean())
                f["profile"].insert(0, "group", label)
                fits[label] = f
                print(f"    [{label}] SMD phi={f['phi_md']:.3f} "
                      f"[{f['ci_lo']:.3f}, {f['ci_hi']:.3f}]"
                      f"{' AT GRID EDGE' if f['at_grid_edge'] else ''} (inversion) | "
                      f"h=1-only phi={f['phi_h1']:.3f} | SSE={f['sse_min']:.4f} | "
                      f"inside its own band at the SMD phi: {ins2.mean():.0%}")

        # ---- mixture check: every entity carried at ITS group's fitted phi ----------
        mix = None
        if a.fit_phi and a.by != "none":
            sd_t = sd_all                      # same calibration as the pooled reference
            col = "dtype" if a.by == "type" else "size_bin"
            lab = (lambda v: f"type{int(v)}") if a.by == "type" else (lambda v: f"size{int(v)}")
            phi_vec = np.array([fits[lab(v)]["phi_md"] if lab(v) in fits else a.phi_prod
                                for v in ent_all[col]])
            Bm = _sim_band(ex, ent_all.to_dict("list"), sd_t, phi_vec.tolist(),
                           a.seed, a.reps_acf, detr)
            lo_m, hi_m = np.nanpercentile(Bm, 2.5, axis=0), np.nanpercentile(Bm, 97.5, axis=0)
            ins_m, above_m, below_m = _inside(acf_pool, lo_m, hi_m)
            mix = (phi_vec, lo_m, hi_m, ins_m, above_m, below_m)
            for i, h in enumerate(range(1, HMAX + 1)):
                rows.append({"group": "mixture(pooled data)", "h": h, "acf_data": acf_pool[i],
                             "band_lo": lo_m[i], "band_hi": hi_m[i], "inside": bool(ins_m[i])})
            panels.append(("mixture", acf_pool, lo_m, hi_m))
            print(f"  [mixture] pooled data ACF vs a panel whose entities carry their own "
                  f"group's SMD phi (mean {phi_vec.mean():.3f}, range "
                  f"{phi_vec.min():.3f}-{phi_vec.max():.3f}): inside {ins_m.mean():.0%}, "
                  f"above {_hlist(above_m)}")

        # ---- per-routine model bands (--routine-bands) -------------------------------
        # For each routine, the pooled data ACF is banded against that routine's OWN model:
        # a scalar band at its mean fitted phi (the LEVEL) and a vector band where every
        # entity carries its own routine-fitted phi_j = its entity-mean phi_mt (the
        # DISPERSION the routine's state index actually generates across markets). The
        # vector band is the theory-consistent object: phi_mt varies across markets in the
        # model, and is common across banks and types WITHIN one -- so this tests each
        # routine as estimated, not a strawman. Outputs are separate files; the archived
        # pooled csv/figure are never touched.
        if a.routine_bands:
            cf_dir = _paths.PROCESSED / "ESTIMATION_OUTPUT" / "CF_FOUNDATION"
            ent_r, sd_r = calibrate(a.n_entities, a.seed, size_bins=a.size_bins)
            r_rows, r_panels = [], []
            for e_id in a.routine_bands:
                fp = cf_dir / f"phi_nopix_E{e_id}_spec_12.parquet"
                if not fp.exists():
                    print(f"  [E{e_id}] MISSING {fp.name} -- run cf_4_upsilon_export.py "
                          f"--estim {e_id}; skipped")
                    continue
                pm = (pd.read_parquet(fp, columns=["entity_id", "phi_mt"])
                      .groupby("entity_id")["phi_mt"].mean())
                v = ent_r["entity_id"].map(pm)
                hit = float(v.notna().mean())
                v = v.fillna(v.median())
                phi_bar = float(v.mean())
                bands = {}
                for kind, phi_arg in (("level", phi_bar), ("fitted phi_j", v.tolist())):
                    B = _sim_band(ex, ent_r.to_dict("list"), sd_r, phi_arg,
                                  a.seed, a.reps_acf, detr)
                    lo, hi = (np.nanpercentile(B, 2.5, axis=0),
                              np.nanpercentile(B, 97.5, axis=0))
                    ins, above, below = _inside(acf_pool, lo, hi)
                    bands[kind] = (lo, hi)
                    print(f"  [E{e_id} | {kind:12s}] mean phi={phi_bar:.3f} "
                          f"p10-p90 [{v.quantile(.1):.3f},{v.quantile(.9):.3f}] "
                          f"map hit={hit:.0%} | inside: {ins.mean():.0%}  "
                          f"above: {_hlist(above)}  below: {_hlist(below)}")
                    for i, h in enumerate(range(1, HMAX + 1)):
                        r_rows.append({"estim": e_id, "kind": kind, "h": h,
                                       "acf_data": acf_pool[i], "band_lo": lo[i],
                                       "band_hi": hi[i], "inside": bool(ins[i]),
                                       "phi_mean": phi_bar, "phi_p10": float(v.quantile(.1)),
                                       "phi_p90": float(v.quantile(.9)), "map_hit": hit})
                r_panels.append((e_id, phi_bar, bands))
            if r_rows:
                pd.DataFrame(r_rows).to_csv(OUT_DIR / "d8_acf_routine_bands.csv",
                                            index=False)
                import matplotlib.pyplot as plt
                hgrid = np.arange(1, HMAX + 1)
                ncol_r = min(len(r_panels), 3)
                nrow_r = int(np.ceil(len(r_panels) / ncol_r))
                figr, axr = plt.subplots(nrow_r, ncol_r,
                                         figsize=(4.2 * ncol_r, 3.4 * nrow_r),
                                         squeeze=False, sharey=True)
                for ax, (e_id, phi_bar, bands) in zip(axr.ravel(), r_panels):
                    ax.grid(color=PAL_GRID, lw=0.6)
                    lo_v, hi_v = bands["fitted phi_j"]
                    lo_s, hi_s = bands["level"]
                    ax.fill_between(hgrid, lo_v, hi_v, alpha=0.30, color=PAL_B,
                                    label="fitted $\\phi_j$ band")
                    ax.plot(hgrid, lo_s, "--", color=PAL_B, lw=1.1,
                            label=f"level band ($\\bar\\phi$={phi_bar:.3f})")
                    ax.plot(hgrid, hi_s, "--", color=PAL_B, lw=1.1)
                    ax.plot(hgrid, acf_pool, "o-", color=PAL_D, ms=3.5, label="data")
                    ax.axhline(0, lw=0.8, color="0.5")
                    ax.set_title(f"E{e_id}", fontsize=10)
                    ax.set_xlabel("horizon h (quarters)")
                ax = axr.ravel()[0]
                ax.set_ylabel("within ACF" + (" (detrended)" if detr else ""))
                ax.legend(fontsize=7)
                for ax in axr.ravel()[len(r_panels):]:
                    ax.axis("off")
                figr.suptitle("D8 per routine: pooled data ACF vs each routine's own "
                              "fitted-model band", fontsize=11)
                figr.tight_layout()
                figr.savefig(OUT_DIR / "d8_acf_routine_bands.png", dpi=150)
                print(f"  -> d8_acf_routine_bands.csv / .png")

    out = pd.DataFrame(rows)
    # the archived default run keeps its exact 4-column layout (h, acf_data, band_lo,
    # band_hi); only the new variants get the long group/inside format.
    legacy = a.by == "none" and not detr
    (out[["h", "acf_data", "band_lo", "band_hi"]] if legacy else out
     ).to_csv(OUT_DIR / f"{base_name}.csv", index=False)
    if fits:
        prof = pd.concat([f["profile"] for f in fits.values()], ignore_index=True)
        prof.to_csv(OUT_DIR / f"d8_acf_fitphi{suffix}.csv", index=False)
        pd.DataFrame([{"group": k, "phi_md": f["phi_md"], "ci_lo": f["ci_lo"],
                       "ci_hi": f["ci_hi"], "phi_h1_only": f["phi_h1"],
                       "sse_min": f["sse_min"], "inside_at_md": f["inside_at_md"]}
                      for k, f in fits.items()]
                     ).to_csv(OUT_DIR / f"d8_acf_fitphi_summary{suffix}.csv", index=False)

    # ---- figure --------------------------------------------------------------------
    import matplotlib.pyplot as plt
    h = np.arange(1, HMAX + 1)
    ncol = min(len(panels), 3)
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.6 * nrow), squeeze=False)
    for ax, (label, acf_data, lo, hi) in zip(axes.ravel(), panels):
        ax.grid(color=PAL_GRID, lw=0.6)
        ax.fill_between(h, lo, hi, alpha=0.30, color=PAL_B,
                        label=f"model band (phi={a.phi_prod}, iid inflows)"
                              if label != "mixture" else "mixture band (phi_j by group)")
        ax.plot(h, acf_data, "o-", color=PAL_D, label="data", ms=3.5)
        ax.axhline(0, lw=0.8, color="0.5")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("horizon h (quarters)")
        ax.set_ylabel("within ACF" + (" (detrended)" if detr else ""))
        ax.legend(fontsize=7)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.suptitle("D8: medium-run persistence vs pure-sleepiness model"
                 + (" [entity-detrended]" if detr else ""), fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{base_name}.png", dpi=150)

    # ---- VERDICT: computed from this run's own output, never asserted --------------
    ins_all, above_all, below_all = _inside(panels[0][1], panels[0][2], panels[0][3])
    print(f"\n  share of horizons inside the band ({panels[0][0]}): {ins_all.mean():.0%}"
          f"  -> {base_name}.png/.csv")
    print(f"  horizons ABOVE the band: {_hlist(above_all)}")
    print(f"  horizons BELOW the band: {_hlist(below_all)}")

    # COMPUTE the direction; never assert it. An earlier version printed "exiting DOWNWARD"
    # unconditionally, and the archived reading of this diagnostic was written from that
    # boilerplate -- the data in fact sit ABOVE the band. A verdict that does not read its
    # own output is worse than no verdict.
    print("\n  VERDICT:", end=" ")
    if above_all.sum() > below_all.sum() and above_all.sum() >= 2:
        print("the data ACF sits ABOVE the band over most horizons.")
        print("  Deposits are MORE persistent at medium horizons than a geometric carry at")
        print(f"  phi={a.phi_prod} generates: the single h=1 moment the estimator fits implies too")
        print("  LITTLE medium-run stickiness. Read with care -- entity-specific drift and")
        print("  heterogeneous phi_j across firms both push the data ACF up without any extra")
        print("  sleepiness (--detrend and --by test exactly those two).")
    elif below_all.sum() > above_all.sum() and below_all.sum() >= 2:
        print("the data ACF sits BELOW the band over most horizons.")
        print("  Measured h=1 persistence dissipates faster than pure sleepiness at")
        print(f"  phi={a.phi_prod} allows, so it is not the same object as long-run non-waking.")
    else:
        print("the data ACF is largely inside the band -- a geometric carry at")
        print(f"  phi={a.phi_prod} is consistent with the medium-run dynamics.")

    if detr:
        print(f"  [--detrend] this run removed an entity-specific linear trend from BOTH "
              f"sides; compare the share inside ({ins_all.mean():.0%}) with the "
              f"un-detrended run's.")
    if a.by != "none":
        share = {lb: float(np.mean([r["inside"] for r in rows if r["group"] == lb]))
                 for lb in dict.fromkeys(r["group"] for r in rows)}
        print("  [--by " + a.by + "] inside-share by group at phi="
              f"{a.phi_prod}: " + ", ".join(f"{k}={v:.0%}" for k, v in share.items()))
        if fits:
            sp = ", ".join(f"{k}={f['phi_md']:.3f}" for k, f in fits.items())
            spread = max(f["phi_md"] for f in fits.values()) - min(f["phi_md"] for f in fits.values())
            print(f"  [--by {a.by}] SMD phi by group: {sp}  (spread {spread:.3f})")
            if mix is not None:
                print(f"  [mixture] pooled data inside the heterogeneous-phi band: "
                      f"{mix[3].mean():.0%}  vs {pooled_ref_inside:.0%} inside the "
                      f"COMMON-phi={a.phi_prod} band on the same data")
                if mix[3].mean() > pooled_ref_inside:
                    print("  Heterogeneity in phi_j moves the pooled ACF toward the data: the")
                    print("  pooled autocorrelogram is NOT clean evidence for a higher common phi.")
                elif mix[3].mean() < pooled_ref_inside:
                    print("  Heterogeneity in phi_j makes the pooled fit WORSE, not better.")
                else:
                    print("  Heterogeneity in phi_j leaves the pooled fit unchanged.")
    if fits:
        d10 = d10_reference()
        ref = (f"D10 entry dynamics phi={d10[0]:.3f} [{d10[1]:.3f}, {d10[2]:.3f}]"
               if d10 else "D10 not on disk")
        for k, f in fits.items():
            print(f"  [{k}] SMD(h=1..{HMAX}) phi={f['phi_md']:.3f} "
                  f"[{f['ci_lo']:.3f}, {f['ci_hi']:.3f}] vs h=1-only phi={f['phi_h1']:.3f} "
                  f"vs {ref}")


# ------------------------------------------------------------------------------
# POWER (no re-simulation: reads the archived grid CSV)
# ------------------------------------------------------------------------------
def _power_tables(df, stat):
    """Size-corrected-t and coefficient-based power for one statistic.

    The per-rep statistic is a CRVE t on ~35 simulated conglomerates against 1.96; its
    MEASURED null size is reported alongside, because it is nowhere near 5% and raw
    rejection frequencies have been read as power before. Both critical values are taken
    AT THE SAME phi_true (the null distribution moves a lot with phi_true), from the
    rho_xi=0 replications of this very file."""
    rows = []
    for phi, g in df.groupby("phi_true"):
        n0 = g[g["rho_xi"] == 0.0]
        R0 = len(n0)
        crit_t = float(np.nanpercentile(np.abs(n0[f"{stat}_t"]), 95))
        crit_c = float(np.nanpercentile(n0[f"{stat}_coef"], 95))
        for rho, gg in g.groupby("rho_xi"):
            R = len(gg)
            raw = float((np.abs(gg[f"{stat}_t"]) > 1.96).mean())
            sct = float((np.abs(gg[f"{stat}_t"]) > crit_t).mean())
            cbp = float((gg[f"{stat}_coef"] > crit_c).mean())
            rows.append({"stat": stat, "phi_true": phi, "rho_xi": rho, "R": R, "R_null": R0,
                         "mean_coef": float(gg[f"{stat}_coef"].mean()),
                         "crit_t95": crit_t, "crit_coef95": crit_c,
                         "size_raw_t196": raw,
                         "power_t_sizecorr": sct, "se_t_sizecorr": np.sqrt(sct * (1 - sct) / R),
                         "power_coef": cbp, "se_coef": np.sqrt(cbp * (1 - cbp) / R)})
    return pd.DataFrame(rows)


def _print_power(tab, stat, title):
    print(f"\n  --- {title} ---")
    print("   phi_true  rho_xi   E[coef]   raw |t|>1.96   size-corr t power   coef power (MC se)")
    for _, r in tab[tab["stat"] == stat].iterrows():
        flag = "  <- null (size)" if r["rho_xi"] == 0 else ""
        print(f"   {r['phi_true']:>8.3f} {r['rho_xi']:>6.1f} {r['mean_coef']:>+9.4f} "
              f"{r['size_raw_t196']:>13.3f} {r['power_t_sizecorr']:>19.3f} "
              f"{r['power_coef']:>13.3f} ({r['se_coef']:.3f}){flag}")


def mode_power(a):
    print("=== D1-power: size-honest power of the D2b / D3 statistics ===")
    src = OUT_DIR / a.power_csv
    # tag the outputs with the source grid, so a --power-csv run on an alternative grid
    # (e.g. the R=1000 one) cannot silently overwrite the default R=200 tables.
    tag = "" if src.name == "d1_mc_recovery_full.csv" else \
        "_" + src.stem.replace("d1_mc_recovery_", "")
    df = pd.read_csv(src)
    print(f"  read {src.name}: {len(df):,} replications, "
          f"{df.groupby(['phi_true','rho_xi']).ngroups} cells "
          f"(R={df.groupby(['phi_true','rho_xi']).size().min()}-"
          f"{df.groupby(['phi_true','rho_xi']).size().max()} per cell) -- NOT re-simulated")
    need = {"d2b_t", "d2b_coef", "d3_t", "d3_coef"}
    if not need <= set(df.columns):
        raise SystemExit(f"{src} lacks {sorted(need - set(df.columns))}: re-run "
                         "--mode grid --stats-under-null")

    tab = pd.concat([_power_tables(df, "d2b"), _power_tables(df, "d3")], ignore_index=True)
    _print_power(tab, "d2b", "D2b (lagged awake-inflow coefficient)")
    _print_power(tab, "d3", "D3 (attractiveness-rank x carry)")

    cmp_tab = None
    if a.power_compare:
        df2 = pd.read_csv(OUT_DIR / a.power_compare)
        t2 = pd.concat([_power_tables(df2, "d2b"), _power_tables(df2, "d3")],
                       ignore_index=True)
        cmp_tab = tab.merge(t2, on=["stat", "phi_true", "rho_xi"], suffixes=("", "_cmp"))
        cmp_tab["d_power_coef"] = cmp_tab["power_coef_cmp"] - cmp_tab["power_coef"]
        if "censor_kept" in df2.columns:
            kk = df2.groupby(["phi_true", "rho_xi"])["censor_kept"].mean()
            print("\n  rows surviving the demand-step censoring in the compared file:")
            print("   " + ", ".join(f"phi={i[0]:g}/rho={i[1]:g}: {v:.1%}"
                                    for i, v in kk.items()))
            print("   (the DATA lose 33-40% of D2b rows; the DGP's inflow A is lognormal and")
            print("    therefore never negative, so simulated censoring bites only through")
            print("    phi_hat estimation error -- the power cost below is a LOWER bound.)")
        print(f"\n  --- power cost of {a.power_compare} vs {a.power_csv} "
              f"(coefficient-based) ---")
        print("   stat  phi_true  rho_xi   base   compared   delta")
        for _, r in cmp_tab[cmp_tab["rho_xi"] > 0].iterrows():
            print(f"   {r['stat']:<5s} {r['phi_true']:>8.3f} {r['rho_xi']:>6.1f} "
                  f"{r['power_coef']:>6.3f} {r['power_coef_cmp']:>10.3f} "
                  f"{r['d_power_coef']:>+7.3f}")
        cmp_tab.to_csv(OUT_DIR / f"d1_power_compare{tag}.csv", index=False)

    tab.to_csv(OUT_DIR / f"d1_power_tables{tag}.csv", index=False)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), sharey=True)
    for ax, stat, name in zip(axes, ("d2b", "d3"), ("D2b", "D3")):
        t = tab[tab["stat"] == stat]
        for i, (phi, g) in enumerate(t.groupby("phi_true")):
            ax.plot(g["rho_xi"], g["power_coef"], "o-", label=f"phi_true={phi:g}",
                    color=[PAL_B, PAL_D, "#2E7D32", "#6A1B9A"][i % 4])
        ax.axhline(0.05, color="0.4", lw=0.8, ls="--")
        ax.grid(color=PAL_GRID, lw=0.6)
        ax.set_xlabel("rho_xi"); ax.set_title(f"{name}: coefficient-based power (5% size)")
    axes[0].set_ylabel("rejection frequency"); axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"d1_power_tables{tag}.png", dpi=150)

    # ---- VERDICT: computed ---------------------------------------------------------
    d2 = tab[(tab["stat"] == "d2b")]
    sizes = d2[d2["rho_xi"] == 0]["size_raw_t196"]
    p9 = d2[(d2["phi_true"] == 0.9) & (d2["rho_xi"] > 0)].sort_values("rho_xi")
    d3 = tab[(tab["stat"] == "d3") & (tab["rho_xi"] > 0)]
    print(f"\n  outputs -> d1_power_tables{tag}.csv / .png")
    print("\n  VERDICT:")
    print(f"  The raw |t|>1.96 rule rejects {sizes.min():.0%}-{sizes.max():.0%} of the time "
          f"UNDER THE NULL, so raw")
    print("  rejection frequencies in this file are not power at 5% and must not be read as such.")
    if len(p9):
        print(f"  Size-corrected: at phi_true=0.9 the D2b COEFFICIENT test has power "
              f"{', '.join(f'{r.power_coef:.2f} at rho={r.rho_xi:g}' for r in p9.itertuples())};")
        tp = d2[(d2["phi_true"] == 0.9) & (d2["rho_xi"] > 0)]["power_t_sizecorr"]
        print(f"  the size-corrected t on the same replications reaches only "
              f"{tp.min():.2f}-{tp.max():.2f}, i.e. the CRVE")
        print("  standard error, not the coefficient, is what destroys the test on ~35 clusters.")
    print(f"  D3's coefficient test stays at {d3['power_coef'].min():.2f}-"
          f"{d3['power_coef'].max():.2f} across the grid: as a check on awake-flow")
    print("  persistence it is close to uninformative, and a null D3 in the data is weak evidence.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["grid", "acf", "power"], default="grid")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--reps-acf", type=int, default=200)
    ap.add_argument("--reps-fit", type=int, default=200)
    ap.add_argument("--n-entities", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stats-under-null", action="store_true")
    ap.add_argument("--censor", action="store_true",
                    help="grid: apply the demand step's max{0,.} / >1e-6 rule to the "
                         "simulated D2b sample (what the data test actually sees)")
    ap.add_argument("--tag", default=None, help="grid: output filename tag")
    ap.add_argument("--phis", type=float, nargs="+", default=None,
                    help="grid: override the phi_true grid")
    ap.add_argument("--rhos", type=float, nargs="+", default=None,
                    help="grid: override the rho_xi grid")
    ap.add_argument("--detrend", action="store_true",
                    help="acf: remove an entity-specific linear trend from data AND sims")
    ap.add_argument("--by", choices=["none", "type", "size"], default="none",
                    help="acf: group the ACF (and its band) by deposit type or size bin")
    ap.add_argument("--size-bins", type=int, default=3,
                    help="acf --by size: 3 = terciles, 10 = deciles")
    ap.add_argument("--fit-phi", action="store_true",
                    help="acf: simulated-minimum-distance phi over h=1..12")
    ap.add_argument("--fit-lo", type=float, default=0.70)
    ap.add_argument("--fit-hi", type=float, default=0.995)
    ap.add_argument("--fit-step", type=float, default=0.005)
    ap.add_argument("--power-csv", default="d1_mc_recovery_full.csv",
                    help="power: archived grid CSV to read (never re-simulated)")
    ap.add_argument("--power-compare", default=None,
                    help="power: second grid CSV (e.g. the --censor run) to difference")
    # default=None so an unspecified --phi-prod RESOLVES from the estimation output rather
    # than freezing a literal; an explicit value on the command line still wins.
    ap.add_argument("--routine-bands", nargs="+", type=int, default=None,
                    help="D8 per routine: for each estimation routine N, band the POOLED data "
                         "ACF against (a) a scalar band at that routine's mean fitted phi and "
                         "(b) a vector band where each entity carries its own routine-fitted "
                         "phi_j (entity mean of phi_mt from CF_FOUNDATION/phi_nopix_E{N}). "
                         "Writes d8_acf_routine_bands.{csv,png}; the archived pooled outputs "
                         "are untouched.")
    ap.add_argument("--phi-prod", type=float, default=None,
                    help="production phi for the iso-contour / D8 band. Default: read from "
                         "the est2 spec-12 fit via utils.phi_reference (pass 0.985 to use "
                         "the E7/E8 headline instead)")
    a = ap.parse_args()
    if a.phi_prod is None:
        from utils import phi_reference as _pr
        a.phi_prod = round(_pr.phi_e2_avg(), 3)
    {"grid": mode_grid, "acf": mode_acf, "power": mode_power}[a.mode](a)

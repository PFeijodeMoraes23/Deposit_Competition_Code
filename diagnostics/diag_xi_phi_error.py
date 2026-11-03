"""diag_xi_phi_error.py -- how persistent is the error that phi-hat passes into the demand residual
xi-hat, asked in the handoff of 2026-11-02 (XI2, Update 11-02g). Descriptive: no paper numbers.

xi-hat = xi + e, with e the error phi-hat passes into xi-hat through the active balances (14-B).
Panel B of tab_xi_diagnostics regresses xi-hat on its lag; a persistent e raises those slopes, a
noise-like e lowers them. Three parts, specification 12, routine E3 by default:

  (a) phi-hat itself: slope of phi-hat_mt on its lag over consecutive quarters, within market
      (market-demeaned), within market and quarter, and pooled; population-weighted, wild cluster
      bootstrap over markets (MCAs). And the between-market share of phi-hat's weighted variance.
  (b) truncation: within B bank x type x market cells, P(truncated at t | truncated at t-1) and
      P(truncated at t | not truncated at t-1), and the shares of cells ever and always
      truncated; unweighted and weighted by the deposit balance (pairs: the balance at t; cells:
      the balance summed over quarters). Truncated: (14-B) floors the active balance at zero.
  (c) error propagation: phi-hat perturbed; the prep's (14-B) step, market size and shares rebuilt
      (sleep_demand_prep_link.process_specification and demand_chars.finalize, unchanged); the
      logit benchmark's xi rebuilt by its closed-form Berry inversion and linear IV (blp_logit.jl:
      berry_inversion, build_matrices, project_spreads, estimate_theta1_logit; sub-model
      full_dtype); e = xi(perturbed) - xi(baseline) on the B cells present in both. For e: Panel
      B's slopes (its cells, pair rule, weights and bootstrap), var(e)/var(xi-hat) and
      corr(e, xi-hat) weighted by the balance, and the truncated share.

The perturbation. The sleep outputs store no bootstrap draws of the sleepiness parameters (the
two-stage bootstrap pickles keep draws of the AMEs only), so phi-hat is moved by the stored
national band, Rout/ts_link_band_est<k>.pkl: the link's wild cluster bootstrap with the index
direction held fixed (utils/sleep_links.py, phi_t_boot). Three kinds of shift, each common to every
market in a quarter and clipped to [0, 1]:
  lo, hi   phi_mt + (lo_t - phi_t) and phi_mt + (hi_t - phi_t), the band's raw 95% ends: bounds;
  draws    phi_mt + (phi_t^(b) - phi_t) for a seeded subset of the band's 400 stored draws
           (attrs["draws"]): samples of the link channel's error, aggregated to the quarter.
2016Q1, before the band's first quarter, takes 2016Q2's shift. Left out: the error of the index
direction, and any part of the link's error that differs across markets.

The logit, not the random-coefficients xi-hat: closed form, one linear IV per perturbation, and
independent of the RC refit. It approximates how phi-hat errors enter the RC xi-hat; the report
gives the logit xi-hat's own Panel B slopes beside the RC ones.

Gates (the run stops on a failure): the rebuilt baseline frame equals the stored demand parquet
(rows, keys, phi, shares); the Berry delta of the stored parquet equals the logit's saved delta
(BLP_RESULTS/logit/logit_delta_E<k>_spec_12.bin); theta1 equals the logit summary's full_dtype
theta1.

Writes rout_dir() / xi_phi_error.md and xi_phi_error.json (unrounded).

    python diagnostics/diag_xi_phi_error.py [--routines 3 4] [--draws 20]
"""
import argparse
import datetime
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: utils/, the prep, the xi table

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import make_xi_diagnostics as XD  # noqa: E402  (Panel B: _codes, _resid, wls, persistence)
import sleep_demand_prep_link as P  # noqa: E402
from utils import demand_chars, paths, routines as R  # noqa: E402
from utils.sleep_links import NonLinearResults  # noqa: E402,F401  (unpickling)

SPEC = 12
OUT_MD = paths.rout_dir() / "xi_phi_error.md"
OUT_JSON = paths.rout_dir() / "xi_phi_error.json"
SETS = (("k4", "Time deposits"), ("all", "All types"))

# blp_logit.jl, the full_dtype sub-model: X_COLS then the D-type dummy; the instruments in order.
LOGIT_X = ["fgc_covered", "has_ip", "seg_S2", "seg_S3", "seg_S4", "seg_S5", "log_total_assets_lag",
           "is_state_owned"]
LOGIT_Z = ["loo_log_assets", "mean_loo_log_assets", "loo_equity_ratio", "mean_loo_equity_ratio",
           "loo_basileia", "mean_loo_basileia", "loo_borrow_assets", "mean_loo_borrow_assets",
           "loo_npl_provision", "mean_loo_npl_provision", "n_rivals", "estban_rival_branches_lag",
           "personnel_cost_ratio_lag", "admin_cost_ratio_lag", "tax_cost_ratio_lag",
           "indice_basileia_lag"]

_LINK = P._apply_link


def _shifted_link(shift_row):
    """The prep's link with a per-row additive shift of phi, clipped to [0, 1]; None restores it."""
    if shift_row is None:
        P._apply_link = _LINK
        return

    def link(index_series, link_name, res_ss):
        phi = _LINK(index_series, link_name, res_ss)
        return (phi + shift_row.reindex(phi.index).to_numpy()).clip(0.0, 1.0)

    P._apply_link = link


# ── the logit benchmark (blp_logit.jl), in numpy ─────────────────────────────────────────────
def berry_delta(df: pd.DataFrame) -> np.ndarray:
    """blp_logit.jl berry_inversion: delta = ln s - ln s0 in the engine's market structure."""
    is_b = df["is_B"].fillna(False).astype(bool).to_numpy()
    tid = df["time_id"].astype(str).to_numpy()
    s_b = pd.to_numeric(df["share_B_cond"], errors="coerce").to_numpy(float)
    s_d = pd.to_numeric(df["share_D"], errors="coerce").to_numpy(float)
    pop = pd.to_numeric(df["pop_total"], errors="coerce").fillna(0.0).to_numpy(float)
    bc = pd.to_numeric(df["banked_correction"], errors="coerce").fillna(1.1).to_numpy(float)
    b = pd.DataFrame({"mca": df["mca_code"].astype(str).to_numpy()[is_b], "t": tid[is_b],
                      "sb": np.where(np.isfinite(s_b[is_b]), s_b[is_b], 0.0),
                      "bcpop": bc[is_b] * pop[is_b]})
    pr = b.groupby(["mca", "t"], sort=False).agg(SB=("sb", "sum"), bcpop=("bcpop", "mean"))
    pr["w"] = pr["bcpop"] / pr.groupby(level="t")["bcpop"].transform("sum").clip(lower=1e-30)
    K = (pr["w"] * (1.0 - pr["SB"])).groupby(level="t").sum()
    d_rows = ~is_b & np.isfinite(s_d)
    SD = pd.Series(s_d[d_rows]).groupby(tid[d_rows]).sum().reindex(K.index).fillna(0.0)
    G = K - SD
    if not (G > 0).all():
        sys.exit(f"berry: non-positive outside share in {int((G <= 0).sum())} quarter(s)")
    t_of_pair = pr.index.get_level_values("t")
    g = ((1.0 - pr["SB"]) * G.reindex(t_of_pair).to_numpy() / K.reindex(t_of_pair).to_numpy()).clip(1e-12, 1.0)
    g_row = pd.Series(g.to_numpy(), index=pr.index).reindex(pd.MultiIndex.from_arrays([b["mca"], b["t"]])).to_numpy()
    delta = np.empty(len(df))
    delta[is_b] = np.log(np.clip(s_b[is_b], 1e-15, None)) - np.log(g_row)
    delta[~is_b] = np.log(np.clip(s_d[~is_b], 1e-15, None)) - np.log(G.reindex(tid[~is_b]).to_numpy())
    return delta


def logit_xi(df: pd.DataFrame, delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """blp_logit.jl build_matrices + project_spreads + estimate_theta1_logit (full_dtype)."""
    for c in ["spread_ann"] + LOGIT_X + LOGIT_Z:
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        if not np.isfinite(v).all():
            sys.exit(f"logit inputs: {c} has {int((~np.isfinite(v)).sum())} non-finite values")
    spread = df["spread_ann"].to_numpy(float) / 100.0
    X = np.column_stack([df[LOGIT_X].to_numpy(float), (~df["is_B"].astype(bool)).to_numpy(float)])
    Z = df[LOGIT_Z].to_numpy(float)
    Z = Z[:, Z.std(axis=0, ddof=1) > 1e-10]
    X_full = np.column_stack([spread, X])
    spread_hat = spread.copy()
    H = np.column_stack([X, Z])
    k = pd.to_numeric(df["deposit_type"], errors="coerce").fillna(0).astype(int).to_numpy()
    for kv in (4, 5):
        m = k == kv
        if m.sum() > H.shape[1]:
            beta = np.linalg.lstsq(H[m], spread[m], rcond=None)[0]
            spread_hat[m] = H[m] @ beta
    X_hat = np.column_stack([spread_hat, X])
    ok = np.isfinite(X_hat).all(1) & np.isfinite(delta) & np.isfinite(Z).all(1)
    theta1 = np.linalg.solve(X_hat[ok].T @ X_hat[ok], X_hat[ok].T @ delta[ok])
    return delta - X_full @ theta1, theta1


# ── frames ───────────────────────────────────────────────────────────────────────────────────
def b_cells(df: pd.DataFrame, xi: np.ndarray) -> pd.DataFrame:
    """Panel B's frame (make_xi_diagnostics.load_frame): B rows, bank x type x market x quarter."""
    d = df[["CodConglomeradoPrudencial", "deposit_type", "mca_code", "year", "quarter", "is_B",
            "deposit_balance"]].reset_index(drop=True).assign(xi=xi)
    d = d[d["is_B"].astype(bool)]
    return pd.DataFrame({"bank": d["CodConglomeradoPrudencial"].astype(str), "mkt": d["mca_code"].astype(str),
                         "k": d["deposit_type"].astype(int),
                         "t": d["year"].astype(int) * 4 + d["quarter"].astype(int) - 1,
                         "w": d["deposit_balance"].astype(float), "xi": d["xi"]}).reset_index(drop=True)


def point_slopes(d: pd.DataFrame) -> dict:
    """Panel B's two slopes (within cell, without demeaning), point estimates only."""
    d = d.sort_values(["bank", "k", "mkt", "t"]).reset_index(drop=True)
    cell = XD._codes(d["bank"], d["k"], d["mkt"])
    t = d["t"].to_numpy()
    ok = np.r_[False, (cell[1:] == cell[:-1]) & (t[1:] == t[:-1] + 1)]
    y, x = d["xi"].to_numpy()[ok], d["xi"].to_numpy()[np.r_[ok[1:], False]]
    w, c = d["w"].to_numpy()[ok], cell[ok]
    out = {}
    for name, groups in (("within", [pd.factorize(c)[0]]), ("raw", [])):
        r = XD._resid(np.column_stack([y, x]), w, groups)
        out[name] = float(np.sum(w * r[:, 0] * r[:, 1]) / np.sum(w * r[:, 1] ** 2))
    out["n_pairs"] = int(ok.sum())
    return out


def wmoments(e: np.ndarray, xi: np.ndarray, w: np.ndarray) -> dict:
    ec, xc = e - np.average(e, weights=w), xi - np.average(xi, weights=w)
    ve, vx, cv = (np.sum(w * ec ** 2), np.sum(w * xc ** 2), np.sum(w * ec * xc))
    return dict(var_ratio=float(ve / vx), corr=float(cv / np.sqrt(ve * vx)),
                mean_e=float(np.average(e, weights=w)), sd_e=float(np.sqrt(ve / w.sum())))


def trunc_share(summary: dict, which: str) -> float:
    keys = ["B_k4"] if which == "k4" else [k for k in summary["Active_truncated"] if k.startswith("B_")]
    tr = summary["Active_truncated"]
    return sum(tr[k][0] for k in keys) / sum(tr[k][1] for k in keys)


# ── the three parts ──────────────────────────────────────────────────────────────────────────
def part_a(pre: pd.DataFrame) -> dict:
    """phi-hat_mt by market and quarter (B rows: phi is one value per MCA-quarter)."""
    b = pre[pre["is_B"].astype(bool)]
    m = (b.groupby(["mca_code", "year", "quarter"], sort=False)
         .agg(phi=("phi_mt", "median"), lo=("phi_mt", "min"), hi=("phi_mt", "max"),
              pop=("pop_total", "first")).reset_index())
    # E4's index carries gdp_growth_yoy, built per entity (bank x type x market) from the entity's
    # own row four quarters back and 0 without one, so an entrant's phi differs from its market's
    # (type 5 in 2020-21). The market's phi is then the median over its rows; the count is kept.
    varies = (m["hi"] - m["lo"]) > 1e-12
    n_vary, max_spread = int(varies.sum()), float((m["hi"] - m["lo"]).max())
    m = m.dropna(subset=["pop"])
    m = m[m["pop"] > 0]
    m["t"] = m["year"].astype(int) * 4 + m["quarter"].astype(int) - 1
    m = m.sort_values(["mca_code", "t"]).reset_index(drop=True)
    mk = pd.factorize(m["mca_code"])[0]
    t = m["t"].to_numpy()
    ok = np.r_[False, (mk[1:] == mk[:-1]) & (t[1:] == t[:-1] + 1)]
    y = m["phi"].to_numpy()[ok]
    x = m["phi"].to_numpy()[np.r_[ok[1:], False]]
    # Codes re-taken on the pairs: a gap (a market with no pair) is an empty group, whose 0/0 mean
    # would hide the market sweep from the two-way demeaning's convergence test.
    w, cl, tq = m["pop"].to_numpy()[ok], pd.factorize(mk[ok])[0], t[ok]
    Y = np.column_stack([y, x])
    out = {}
    for name, groups in (("within_market", [cl]), ("within_market_quarter", [cl, pd.factorize(tq)[0]]),
                         ("pooled", [])):
        r = XD._resid(Y, w, groups)
        out[name] = XD.wls(r[:, 0], r[:, 1], w, cl, "lag")
    wa, ph, mca = m["pop"].to_numpy(), m["phi"].to_numpy(), pd.factorize(m["mca_code"])[0]
    within = XD._resid(ph, wa, [mca])[:, 0]
    total = ph - np.average(ph, weights=wa)
    out.update(between_share=float(1.0 - np.sum(wa * within ** 2) / np.sum(wa * total ** 2)),
               mean=float(np.average(ph, weights=wa)), sd=float(np.sqrt(np.average(total ** 2, weights=wa))),
               sd_within=float(np.sqrt(np.average(within ** 2, weights=wa))),
               n_pairs=int(ok.sum()), n_markets=int(len(np.unique(cl))), n_market_quarters=int(len(m)),
               n_mq_phi_varies=n_vary, max_within_mq_spread=max_spread)
    return out


def part_b(pre: pd.DataFrame) -> dict:
    """Truncation transitions within B bank x type x market cells, before the (14-B) cut."""
    b = pre[pre["is_B"].astype(bool)]
    d = pd.DataFrame({"bank": b["CodConglomeradoPrudencial"].astype(str).to_numpy(),
                      "k": b["deposit_type"].astype(int).to_numpy(), "mkt": b["mca_code"].astype(str).to_numpy(),
                      "t": (b["year"].astype(int) * 4 + b["quarter"].astype(int) - 1).to_numpy(),
                      "w": b["deposit_balance"].astype(float).to_numpy(),
                      "tr": (~(b["active_raw"].astype(float) > 1e-6)).to_numpy()})
    out = {}
    for s, _ in SETS:
        x = (d[d["k"] == 4] if s == "k4" else d).sort_values(["bank", "k", "mkt", "t"]).reset_index(drop=True)
        cell = XD._codes(x["bank"], x["k"], x["mkt"])
        t, tr, w = x["t"].to_numpy(), x["tr"].to_numpy(), x["w"].to_numpy()
        ok = np.r_[False, (cell[1:] == cell[:-1]) & (t[1:] == t[:-1] + 1)]
        prev = np.r_[False, tr[:-1]]
        r = {"n_rows": int(len(x)), "n_pairs": int(ok.sum()), "share_rows": float(tr.mean()),
             "share_rows_w": float(np.average(tr, weights=w))}
        for cond, lab in ((True, "p_tt"), (False, "p_tn")):
            sel = ok & (prev == cond)
            r[lab] = float(tr[sel].mean())
            r[lab + "_w"] = float(np.average(tr[sel], weights=w[sel]))
            r["n_" + lab] = int(sel.sum())
        c = pd.DataFrame({"cell": cell, "tr": tr, "w": w}).groupby("cell").agg(
            ever=("tr", "max"), always=("tr", "min"), w=("w", "sum"), n=("tr", "size"))
        r.update(n_cells=int(len(c)), ever=float(c["ever"].mean()), always=float(c["always"].mean()),
                 ever_w=float(np.average(c["ever"], weights=c["w"])),
                 always_w=float(np.average(c["always"], weights=c["w"])),
                 n_cells_2plus=int((c["n"] >= 2).sum()))
        out[s] = r
    return out


def run_routine(est: int, n_draws: int, seed: int) -> dict:
    t0 = time.time()
    link, _tag, time_block = P.DEMAND_CFG[est]
    sp = pickle.load(open(paths.est_dir(est) / "_specs" / f"spec_{SPEC}.pkl", "rb"))
    name, spec_res = sp["spec_name"], {"second_stage": sp["result"][0]}
    if name != P.SPEC_MAP[SPEC]:
        sys.exit(f"E{est}: spec_{SPEC}.pkl holds {name!r}, not {P.SPEC_MAP[SPEC]!r}")
    print(f"E{est}: building the base panel ...", flush=True)
    df_base = P.build_base_panel(P.PANEL_CSV, time_block=time_block)

    def prep(shift_t=None):
        if shift_t is None:
            _shifted_link(None)
        else:
            _shifted_link(df_base["time_id"].astype(str).map(shift_t).fillna(0.0))
        try:
            df, summary, _ = P.process_specification(name, spec_res, df_base, link)
            df = demand_chars.finalize(df, None, summary)
        finally:
            _shifted_link(None)
        return df.reset_index(drop=True), summary

    # ── baseline and gates ──
    base, base_sum = prep()
    pq = paths.demand_parquet_dir() / f"{R.demand_prefix(est)}_spec_{SPEC}.parquet"
    stored = pd.read_parquet(pq)
    gates = {}
    keys = ["entity_id", "deposit_type", "mca_code", "time_id"]
    same_keys = len(base) == len(stored) and all(
        (base[c].astype(str).to_numpy() == stored[c].astype(str).to_numpy()).all() for c in keys)
    gates["frame_rows_keys"] = dict(ok=bool(same_keys), rebuilt=len(base), stored=len(stored))
    if not same_keys:
        sys.exit(f"E{est} gate: the rebuilt frame's rows differ from {pq.name}: {gates}")
    for c in ("phi_mt", "phi_t", "share_B_cond", "share_D", "Dep_Act"):
        a, s = base[c].to_numpy(float), stored[c].to_numpy(float)
        both = np.isfinite(a) & np.isfinite(s)
        diff = float(np.max(np.abs(a[both] - s[both]) / np.maximum(np.abs(s[both]), 1e-300))) if both.any() else 0.0
        ok = bool((np.isfinite(a) == np.isfinite(s)).all() and diff < 1e-9)
        gates[f"frame_{c}"] = dict(ok=ok, max_rel_diff=diff)
        if not ok:
            sys.exit(f"E{est} gate: {c} of the rebuilt frame differs from {pq.name}: {gates[f'frame_{c}']}")
    d_stored = berry_delta(stored)
    binp = paths.blp_results_dir() / "logit" / f"logit_delta_E{est}_spec_{SPEC}.bin"
    raw = np.fromfile(binp, dtype=np.float64, offset=8)
    n_bin = int(np.fromfile(binp, dtype=np.int64, count=1)[0])
    dd = float(np.max(np.abs(d_stored - raw))) if n_bin == len(d_stored) == len(raw) else np.inf
    gates["logit_delta"] = dict(ok=bool(dd < 1e-9), max_abs_diff=dd, file=binp.name)
    if not gates["logit_delta"]["ok"]:
        sys.exit(f"E{est} gate: Berry delta differs from {binp.name}: {gates['logit_delta']}")
    xi_s, th_s = logit_xi(stored, d_stored)
    summ = json.loads((paths.blp_results_dir() / "logit" / f"logit_summary_spec_{SPEC}.json").read_text())
    th_jl = np.asarray(summ[f"E{est}_full_dtype"]["theta1"], float)
    dt = float(np.max(np.abs(th_s - th_jl) / np.maximum(np.abs(th_jl), 1e-12)))
    gates["logit_theta1"] = dict(ok=bool(dt < 1e-6), max_rel_diff=dt, alpha=float(th_s[0]))
    if not gates["logit_theta1"]["ok"]:
        sys.exit(f"E{est} gate: theta1 differs from the logit summary: {gates['logit_theta1']}")
    print(f"E{est}: gates pass ({time.time() - t0:.0f} s)", flush=True)

    d_base = berry_delta(base)
    xi_b, th_b = logit_xi(base, d_base)
    cb = b_cells(base, xi_b)
    res = dict(gates=gates, baseline=dict(alpha=float(th_b[0]), trunc={s: trunc_share(base_sum, s) for s, _ in SETS}))
    res["baseline"]["ar1"] = {s: XD.persistence(cb[cb["k"] == 4].reset_index(drop=True) if s == "k4" else cb)
                              for s, _ in SETS}

    # ── (a) and (b) on the pre-cut frame ──
    pre, _, _ = P.process_specification(name, spec_res, df_base, link, pre_cut=True)
    res["a"] = part_a(pre)
    res["b"] = part_b(pre)
    del pre
    print(f"E{est}: parts (a), (b) done ({time.time() - t0:.0f} s)", flush=True)

    # ── (c) perturbations ──
    band = pickle.load(open(paths.rout_dir() / f"ts_link_band_est{est}.pkl", "rb"))
    q = band["time_id"].astype(str).tolist()
    first_q = sorted(set(base["time_id"].astype(str)))[0]

    def shift_map(v):
        s = dict(zip(q, np.asarray(v, float) - band["phi_t"].to_numpy(float)))
        if first_q not in s:
            s[first_q] = s[q[0]]
        return s

    draws = np.asarray(band.attrs["draws"], float)
    pick = np.sort(np.random.default_rng(seed).choice(len(draws), size=min(n_draws, len(draws)), replace=False))
    perts = [("lo", shift_map(band["lo"]), True), ("hi", shift_map(band["hi"]), True)]
    perts += [(f"draw{int(i)}", shift_map(draws[i]), False) for i in pick]
    res["c"] = {}
    for lab, smap, boot in perts:
        df_p, sum_p = prep(smap)
        xi_p, th_p = logit_xi(df_p, berry_delta(df_p))
        cp = b_cells(df_p, xi_p)
        m = cb.merge(cp[["bank", "mkt", "k", "t", "xi"]], on=["bank", "mkt", "k", "t"], suffixes=("", "_p"))
        m["e"] = m["xi_p"] - m["xi"]
        r = dict(alpha=float(th_p[0]), mean_shift=float(np.mean(list(smap.values()))),
                 min_shift=float(min(smap.values())), max_shift=float(max(smap.values())))
        for s, _ in SETS:
            ms = (m[m["k"] == 4] if s == "k4" else m).reset_index(drop=True)
            cps = (cp[cp["k"] == 4] if s == "k4" else cp).reset_index(drop=True)
            nb = int((cb["k"] == 4).sum()) if s == "k4" else len(cb)
            de = ms[["bank", "mkt", "k", "t", "w"]].assign(xi=ms["e"].to_numpy())
            rs = dict(wmoments(ms["e"].to_numpy(), ms["xi"].to_numpy(), ms["w"].to_numpy()),
                      trunc=trunc_share(sum_p, s), n_both=int(len(ms)), n_base=nb, n_pert=int(len(cps)))
            rs["ar1"] = XD.persistence(de) if boot else point_slopes(de)
            # Panel B recomputed on the perturbed xi, on its own cells: the table under this phi
            rs["xi_p_ar1"] = point_slopes(cps)
            r[s] = rs
        res["c"][lab] = r
        k4 = r["k4"]
        sl = k4["ar1"]["within"]["slope"] if boot else k4["ar1"]["within"]
        print(f"E{est} {lab}: alpha {th_p[0]:+.4f}; k4 e within slope {sl:.3f}, var ratio "
              f"{k4['var_ratio']:.3f}, trunc {k4['trunc']:.3f} ({time.time() - t0:.0f} s)", flush=True)
    res["meta"] = dict(spec_pkl=str(paths.est_dir(est) / "_specs" / f"spec_{SPEC}.pkl"),
                       spec_written=sp.get("written_utc"), parquet=str(pq), band=f"ts_link_band_est{est}.pkl",
                       draws_used=[int(i) for i in pick], n_draws_stored=int(len(draws)), seed=seed,
                       runtime_s=round(time.time() - t0, 1))
    return res


# ── report ───────────────────────────────────────────────────────────────────────────────────
def f3(v) -> str:
    return "n/a" if v is None or not np.isfinite(v) else f"{v:.3f}"


def ci(r: dict) -> str:
    return f"{f3(r['slope'])}{r['stars']} [{f3(r['lo'])}, {f3(r['hi'])}]"


def report(all_res: dict, rc: dict) -> str:
    L = ["# XI2: persistence of the error phi-hat passes into xi-hat", "",
         f"Written {datetime.datetime.now():%Y-%m-%d %H:%M} by diagnostics/diag_xi_phi_error.py. "
         "Specification 12. Descriptive only: no paper keys.", "",
         "**Perturbation used: option 2, the band.** The sleep outputs store no bootstrap draws of the "
         "sleepiness parameters (the two-stage bootstrap pickles keep draws of the AMEs only). phi-hat_mt is "
         "moved by the stored national band of the link (`ts_link_band_est<k>.pkl`; link wild cluster "
         "bootstrap, index direction held fixed): to its raw 95% ends (lo, hi: bounds), and by a seeded subset "
         "of its 400 stored draws (samples of the link channel's error). Each shift is common to all markets in "
         "a quarter. Left out: the index direction's error, and link error that differs across markets.", "",
         "**xi is the logit benchmark's** (closed-form Berry inversion and linear IV, full_dtype), not the "
         "random-coefficients xi-hat of the table: an approximation to how phi-hat errors enter xi-hat.", ""]
    for est, r in all_res.items():
        g = r["gates"]
        L += [f"## {est}", "",
              "Gates: rebuilt frame = stored parquet (" + ", ".join(
                  f"{k.replace('frame_', '')} {v['max_rel_diff']:.1e}" for k, v in g.items() if k.startswith("frame_") and k != "frame_rows_keys")
              + f"); Berry delta = logit's saved delta (max abs diff {g['logit_delta']['max_abs_diff']:.1e}); "
              f"theta1 = logit summary (max rel diff {g['logit_theta1']['max_rel_diff']:.1e}). All pass.", ""]
        a = r["a"]
        L += ["### (a) phi-hat_mt itself", "",
              f"{a['n_market_quarters']:,} market-quarters, {a['n_markets']:,} markets, {a['n_pairs']:,} pairs of "
              f"consecutive quarters; population-weighted; wild cluster bootstrap over markets (Webb, 999).", "",
              "| slope of phi-hat on its lag | estimate [95% BC interval] |", "|---|---|",
              f"| within market | {ci(a['within_market'])} |",
              f"| within market and quarter | {ci(a['within_market_quarter'])} |",
              f"| pooled | {ci(a['pooled'])} |", "",
              f"Mean {a['mean']:.4f}, SD {a['sd']:.4f}, SD within market {a['sd_within']:.4f}; between-market "
              f"share of the weighted variance **{a['between_share']:.3f}**.", "",
              f"Market-quarters where phi-hat differs across B rows: {a['n_mq_phi_varies']:,} (largest spread "
              f"{a['max_within_mq_spread']:.4f}); the market's phi-hat is the median over its rows. In E4 this comes "
              "from gdp_growth_yoy, built per entity from its own row four quarters back and 0 without one "
              "(sleep_est_e2.add_time_variables), so entrants (type 5 in 2020-21) get zero GDP growth.", ""]
        L += ["### (b) truncation in (14-B), B bank x type x market cells", "",
              "| | time deposits | | all types | |", "|---|---|---|---|---|",
              "| | unweighted | balance-weighted | unweighted | balance-weighted |"]
        b = r["b"]
        for lab, key in (("share of cell-quarters truncated", "share_rows"),
                         ("P(truncated at t / truncated at t-1)", "p_tt"),
                         ("P(truncated at t / not truncated at t-1)", "p_tn"),
                         ("share of cells ever truncated", "ever"), ("share of cells always truncated", "always")):
            L.append(f"| {lab} | {f3(b['k4'][key])} | {f3(b['k4'][key + '_w'])} | {f3(b['all'][key])} | "
                     f"{f3(b['all'][key + '_w'])} |")
        L += ["", f"Cells: {b['k4']['n_cells']:,} (time deposits), {b['all']['n_cells']:,} (all types); pairs of "
              f"consecutive quarters: {b['k4']['n_pairs']:,} and {b['all']['n_pairs']:,}. Pairs weighted by the "
              "balance at t, cells by the balance summed over quarters.", ""]
        base = r["baseline"]
        L += ["### (c) error propagation", "",
              "Reference: Panel B's slopes of xi-hat on its lag.", "",
              "| | time deposits, within | time deposits, pooled | all types, within | all types, pooled |",
              "|---|---|---|---|---|"]
        if est in rc:
            L.append("| RC xi-hat (the table) | " + " | ".join(
                f3(rc[est][s]["ar1"][w]["slope"]) for s in ("k4", "all") for w in ("within", "raw")) + " |")
        L.append("| logit xi-hat (baseline here) | " + " | ".join(
            ci(base["ar1"][s][w]) for s in ("k4", "all") for w in ("within", "raw")) + " |")
        L += ["", f"Logit alpha at baseline {base['alpha']:+.4f}; truncated share of B cells {base['trunc']['k4']:.3f} "
              f"(time deposits), {base['trunc']['all']:.3f} (all types).", "",
              "**Bounds (the band's ends).** e = xi(perturbed) - xi(baseline), on the B cells present in both. "
              "\"Panel B on xi(perturbed)\": the table's slopes recomputed on the perturbed xi and its own cells.", ""]
        c = r["c"]
        L += [f"phi shift, mean over quarters [min, max]: lo {c['lo']['mean_shift']:+.4f} "
              f"[{c['lo']['min_shift']:+.4f}, {c['lo']['max_shift']:+.4f}], hi {c['hi']['mean_shift']:+.4f} "
              f"[{c['hi']['min_shift']:+.4f}, {c['hi']['max_shift']:+.4f}]; logit alpha: lo {c['lo']['alpha']:+.4f}, "
              f"hi {c['hi']['alpha']:+.4f} (baseline {base['alpha']:+.4f}).", "",
              "| | time deposits, lo | time deposits, hi | all types, lo | all types, hi |", "|---|---|---|---|---|"]
        rows = (("e slope, within cell", lambda x, s: ci(x[s]["ar1"]["within"])),
                ("e slope, pooled", lambda x, s: ci(x[s]["ar1"]["raw"])),
                ("var(e) / var(xi-hat)", lambda x, s: f3(x[s]["var_ratio"])),
                ("corr(e, xi-hat)", lambda x, s: f3(x[s]["corr"])),
                ("Panel B on xi(perturbed), within", lambda x, s: f3(x[s]["xi_p_ar1"]["within"])),
                ("Panel B on xi(perturbed), pooled", lambda x, s: f3(x[s]["xi_p_ar1"]["raw"])),
                ("truncated share of B cells", lambda x, s: f3(x[s]["trunc"])),
                ("B cells: baseline / perturbed / both",
                 lambda x, s: f"{x[s]['n_base']:,} / {x[s]['n_pert']:,} / {x[s]['n_both']:,}"))
        for lab, fn in rows:
            L.append(f"| {lab} | " + " | ".join(fn(c[p], s) for s, _ in SETS for p in ("lo", "hi")) + " |")
        dr = [v for k, v in c.items() if k.startswith("draw")]
        if dr:
            L += ["", f"**Samples ({len(dr)} of the band's stored draws).** Median [5th, 95th percentile] across draws; "
                  "slopes are point estimates.", "", "| | time deposits | all types |", "|---|---|---|"]

            def q3(vals):
                v = np.asarray(vals, float)
                return f"{np.median(v):.3f} [{np.quantile(v, 0.05):.3f}, {np.quantile(v, 0.95):.3f}]"
            L.append("| phi shift, mean over quarters | " + q3([x["mean_shift"] for x in dr]) + " | (same) |")
            for lab, fn in (("e slope, within cell", lambda x, s: x[s]["ar1"]["within"]),
                            ("e slope, pooled", lambda x, s: x[s]["ar1"]["raw"]),
                            ("var(e) / var(xi-hat)", lambda x, s: x[s]["var_ratio"]),
                            ("corr(e, xi-hat)", lambda x, s: x[s]["corr"]),
                            ("Panel B on xi(perturbed), within", lambda x, s: x[s]["xi_p_ar1"]["within"]),
                            ("Panel B on xi(perturbed), pooled", lambda x, s: x[s]["xi_p_ar1"]["raw"]),
                            ("truncated share of B cells", lambda x, s: x[s]["trunc"]),
                            ("share of baseline B cells kept", lambda x, s: x[s]["n_both"] / x[s]["n_base"])):
                L.append(f"| {lab} | " + " | ".join(q3([fn(x, s) for x in dr]) for s, _ in SETS) + " |")
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--routines", type=int, nargs="+", default=[3])
    ap.add_argument("--draws", type=int, default=20, help="band draws to propagate (point slopes)")
    ap.add_argument("--seed", type=int, default=20261102)
    a = ap.parse_args()
    pd.options.mode.chained_assignment = None
    all_res = {f"E{e}": run_routine(e, a.draws, a.seed) for e in a.routines}
    rc = {}
    if XD.SIDECAR.is_file():
        side = json.loads(XD.SIDECAR.read_text())
        rc = {k: v for k, v in side.get("results", side).items() if k in all_res}
    OUT_JSON.write_text(json.dumps(all_res, indent=1, default=float), encoding="utf-8")
    OUT_MD.write_text(report(all_res, rc), encoding="utf-8")
    print(f"wrote {OUT_MD}\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()

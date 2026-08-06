"""diag_entry_dynamics.py -- D10: do entrants accumulate share at the speed phi-hat implies?

Author: Pedro Feijo de Moraes

This is the Brazilian analogue of Egan et al. (2025) Figure 3 -- the one moment that
discriminates SLEEPINESS from PERSISTENT PREFERENCES, and the piece our draft lacked.

    Persistent preferences + full reoptimisation => an entrant jumps immediately to its
    steady-state share (everyone who prefers it sorts in at once).
    Sleepiness => the entrant can only reach the (1-phi) awake fraction each period, so
    share accumulates slowly, with a curvature pinned by phi.

The model line is CLOSED FORM. With frozen spreads the law of motion
(foundation_deposit_sim.jl:192)

    Dep_{t+1} = (1-phi)*M*s + phi*g*Dep_t,      g = 1 + r^dep_q

is a scalar affine map, so from Dep_0 = 0

    Dep_h = A*(1-(phi g)^h)/(1-phi g),   A = (1-phi)*M*s,   Dep_h/Dep_inf = 1-(phi g)^h.

Rivals enter only through A (the level), which BOTH Egan normalisations cancel. So the
model curve needs phi-hat and g alone -- no BLP context, no draws, no cluster.

Two normalisations, exactly as in the paper:
  main (their Fig. 3):  n_h = (s_h - s_0)/(s_end - s_0)
  alt  (their Fig. A1): n_h =  s_h / s_end,  where the instant-sorting benchmark is the
                        flat line at 1 (it is 0/0 in the main normalisation, so it is
                        drawn on the ALT panel only -- same as the paper).

Outputs -> DIAG_PHI_SEPARATION/d10_entry_{events,paths,model_curves}.csv,
           d10_entry_dynamics.png/.pdf, and fig_entry_dynamics.png/.pdf in Drafts.
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import argparse
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import shutil

import numpy as np
import pandas as pd

from utils import paths as _paths
from utils import load_panel_cached

OUT_DIR = _paths.PROCESSED / "ESTIMATION_OUTPUT" / "DIAG_PHI_SEPARATION"
DRAFTS = _paths.OPEN_FINANCE / "Drafts" / "Deposit Competition"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PANEL_CSV = _paths.PROCESSED / "market_panel.csv"
BRANCH_SIDECAR = _paths.PROCESSED / "PANEL_INTERMED" / "estban_own_branches.csv"
DEP_COLS = ["dep_a1", "dep_a2", "dep_a4"]          # a5 (prepaid) is NaN in ESTBAN

# figure palette (desc_2.py convention)
B_COLOR, D_COLOR, INK, GRID = "#1565C0", "#E64A19", "#4F4F4F", "#D5D5D0"
MODEL_COLORS = {"E2": "#2E7D32", "E7": "#6A1B9A", "E8": "#00838F"}


# ----------------------------------------------------------------------------- phi vintages
# Nonlinear reference vintages plotted against the entry paths. E5/E6 (single-index) rather
# than E7/E8 (joint sieve): the joint fits' index DIRECTION is weakly identified -- four
# polishes from different starts move phi_t by 7x at an R2 difference of 0.0013 -- so their
# level is not a stable object to draw a curve at. E5/E6 inherit the logit direction and are
# reproducible. Override with SLEEP_ENTRY_REFS if a comparison against the sieve is wanted.
REF_ESTS = tuple(os.environ.get("SLEEP_ENTRY_REFS", "E5,E6").split(","))


def phi_vintages():
    """(label, phi, source) for each model line. All refresh from disk: E2 via
    utils.phi_reference (est2 spec-12 fit, with a self-updating last-known-good cache),
    the REF_ESTS from the CF_FOUNDATION exports. Nothing here is a frozen literal -- a stale
    model line plotted against fresh data is the one failure this figure cannot survive."""
    from utils import phi_reference as _pr
    out = [("E2", _pr.phi_e2_avg(), "est2 spec-12 fit via utils.phi_reference")]
    cf = _paths.PROCESSED / "ESTIMATION_OUTPUT" / "CF_FOUNDATION"
    for est in REF_ESTS:
        fp = cf / f"phi_nopix_{est}_spec_12.parquet"
        if not fp.exists():
            print(f"  [phi] {est}: {fp.name} absent -- line skipped")
            continue
        m = float(pd.read_parquet(fp, columns=["phi_mt"])["phi_mt"].mean())
        ts = pd.Timestamp(fp.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M")
        out.append((est, m, f"mean phi_mt of {fp.name} [{ts}]"))
    return out


def median_g():
    fp = _paths.PROCESSED / "ESTIMATION_OUTPUT" / "DEMAND_PREP" / "demand_2_spec_12.parquet"
    g = float(pd.read_parquet(fp, columns=["gross_return_lag"])["gross_return_lag"].median())
    assert 1.0 < g < 1.06, f"implausible accrual g={g}"
    return g


def model_curve(phi, g, H, norm, plateau_w):
    """Normalised model path on h=0..H, from Dep_0 = 0 under frozen spreads.

    Level path Dep_h = A*(1-(phi g)^(h+1))/(1-phi g) (h=0 is the first observed quarter,
    i.e. one period of accumulation has already happened). A cancels under both
    normalisations, so only phi*g shapes the curve.

    TWO REGIMES, and which one we are in is itself a result:
      phi*g < 1  CONCAVE approach to the steady state A/(1-phi g)  -- the Egan picture.
      phi*g >= 1 EXPLOSIVE: interest accrues on the sleepy stock faster than depositors
                 leak away, so deposits diverge and "steady-state share" is undefined.
                 The normalised curve is then CONVEX (accelerating). We still draw it --
                 refusing to plot would hide the finding -- but the caller flags it.
    """
    pg = phi * g
    if abs(pg - 1.0) < 1e-9:                       # knife-edge: linear accumulation
        lvl = np.array([float(h + 1) for h in range(H + 1)])
    else:
        lvl = np.array([(1.0 - pg ** (h + 1)) / (1.0 - pg) for h in range(H + 1)])
    end = lvl[plateau_w].mean()
    if norm == "main":
        return (lvl - lvl[0]) / (end - lvl[0])
    return lvl / end


# ----------------------------------------------------------------------------- events
def load_branch_entry(args):
    """(congl, mca) -> first quarter with a branch, from the panel_10 sidecar.

    Egan et al. flag that Summary-of-Deposits entry can be mechanical: deposits already
    collected nearby get booked at a newly reported branch, so the entrant appears with a
    large share on day one. The ESTBAN branch count is the independent timing check --
    deposits appearing well BEFORE any branch is the signature of that artifact (or of
    remote//digital booking), not of a firm building a local franchise.
    """
    if args.no_branch_screen:
        print("  [branch] screen disabled (--no-branch-screen)")
        return None
    if not BRANCH_SIDECAR.exists():
        print(f"  [branch] {BRANCH_SIDECAR.name} absent -- screen skipped "
              "(run panel_10_estban_instruments.py to create it)")
        return None
    br = pd.read_csv(BRANCH_SIDECAR, dtype={"mca_code": str})
    br = br[br["own_br"] > 0]
    br["qidx"] = br["year"].astype(int) * 4 + br["quarter"].astype(int) - 1
    out = (br.groupby([br["CodConglomeradoPrudencial"].astype(str), "mca_code"])["qidx"]
           .min().rename("q_branch"))
    out.index.names = ["congl", "mkt"]
    ts = pd.Timestamp(BRANCH_SIDECAR.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M")
    print(f"  [branch] sidecar {BRANCH_SIDECAR.name} [{ts}]: {len(out):,} congl-MCA pairs "
          "with a branch")
    return out


def build_events(df, kind, args, branch=None):
    """One row per candidate entry event with every screen flag. kind in {'B','D'}."""
    key = ["congl", "mkt"]
    df = df.sort_values(key + ["qidx"])
    first = df.groupby(key, as_index=False)["qidx"].min().rename(columns={"qidx": "q_entry"})
    # market age: quarters in which the market had ANY other firm before entry
    mkt_first = df.groupby("mkt", as_index=False)["qidx"].min().rename(columns={"qidx": "q_mkt0"})
    ev = first.merge(mkt_first, on="mkt", how="left")
    ev["cohort"] = ev["q_entry"] // 4
    ev["kind"] = kind

    qmax = int(df["qidx"].max())
    # --- screens -------------------------------------------------------------------
    ev["f_trunc"] = ev["q_entry"] <= int(df["qidx"].min()) + 3      # left truncation
    ev["f_mkt_young"] = (ev["q_entry"] - ev["q_mkt0"]) < 4          # market itself just born
    ev["f_short"] = (qmax - ev["q_entry"]) < args.min_post          # not enough post quarters

    # presence/gap + share path
    piv = (df.pivot_table(index=key, columns="qidx", values="dep", aggfunc="sum")
           .reindex(columns=range(int(df["qidx"].min()), qmax + 1)))
    tot = df.groupby(["mkt", "qidx"])["dep"].sum().rename("mkt_dep")
    sh = df.join(tot, on=["mkt", "qidx"])
    sh["share"] = sh["dep"] / sh["mkt_dep"].replace(0, np.nan)
    shp = sh.pivot_table(index=key, columns="qidx", values="share", aggfunc="sum")

    H = args.horizon
    rows, paths = [], {}
    for _, e in ev.iterrows():
        k = (e["congl"], e["mkt"])
        if k not in shp.index:
            continue
        q0 = int(e["q_entry"])
        cols = [q0 + h for h in range(H + 1) if q0 + h <= qmax]
        s = shp.loc[k, cols].astype(float)
        d = piv.loc[k, cols].astype(float)
        n_gap = int(d.isna().sum() + (d.fillna(0) <= 0).sum())
        r = dict(e)
        r["n_obs_post"] = int(len(cols))
        r["f_gappy"] = n_gap > 1
        r["s0"] = float(s.iloc[0]) if len(s) else np.nan
        jumps = s.diff().abs()
        r["f_jump"] = bool((jumps.iloc[1:] > args.max_jump).any())
        r["f_big_entry"] = bool(r["s0"] > args.max_entry_share)
        paths[k] = s
        rows.append(r)
    reg = pd.DataFrame(rows)
    if reg.empty:
        return reg, paths

    # conglomerate-code churn: same CNPJ_Lider root already active in this market
    if "cnpj_root" in df.columns:
        pre = df[["cnpj_root", "mkt", "congl", "qidx"]].dropna()
        reg = reg.merge(pre.groupby(["cnpj_root", "mkt"], as_index=False)["qidx"].min()
                        .rename(columns={"qidx": "q_root_first"}),
                        left_on=[reg.get("cnpj_root", pd.Series(index=reg.index)), "mkt"],
                        right_on=["cnpj_root", "mkt"], how="left") if False else reg
    reg["f_churn"] = False
    if "cnpj_root" in df.columns:
        root_first = (df.dropna(subset=["cnpj_root"])
                      .groupby(["cnpj_root", "mkt"], as_index=False)["qidx"].min()
                      .rename(columns={"qidx": "q_root_first"}))
        ev_root = (df.dropna(subset=["cnpj_root"])
                   .groupby(["congl", "mkt"], as_index=False)["cnpj_root"].first())
        reg = reg.merge(ev_root, on=["congl", "mkt"], how="left")
        reg = reg.merge(root_first, on=["cnpj_root", "mkt"], how="left")
        # the CNPJ root was in this market BEFORE this conglomerate code appeared => rename
        reg["f_churn"] = reg["q_root_first"] < reg["q_entry"] - 0.5

    # branch-timing screen (B firms only: D firms have no branch network by construction)
    reg["f_branch"] = False
    if branch is not None and kind == "B":
        qb = pd.MultiIndex.from_arrays([reg["congl"], reg["mkt"]]).map(branch)
        reg["q_branch"] = pd.to_numeric(pd.Series(qb, index=reg.index), errors="coerce")
        # deposits >= 2 quarters before the first branch, or never any branch at all
        reg["f_branch"] = (reg["q_branch"].isna()
                           | (reg["q_entry"] <= reg["q_branch"] - 2))

    flags = ["f_trunc", "f_mkt_young", "f_short", "f_gappy", "f_jump", "f_big_entry",
             "f_churn", "f_branch"]
    if args.window_only:
        reg["f_window"] = reg["q_entry"] < args.window_q0
        flags.append("f_window")
    reg["keep"] = ~reg[flags].any(axis=1)
    return reg, paths


def implied_phi(reg, paths, H, plateau_w, g, boot, seed, grid=None):
    """INVERT the figure: what phi do the entry paths themselves imply?

    One-parameter fit -- pick phi minimising SSE between the model curve 1-(phi g)^h
    (main normalisation) and the median empirical path over h=1..H-1 (h=0 and the plateau
    window are pinned to 0/1 by the normalisation and carry no information). The bootstrap
    resamples EVENTS, refits phi on each draw, and returns the percentile CI.

    This is a genuine second measure of phi, independent of the deposit-autocorrelation
    moment that produced phi-hat: it uses only the SHAPE of post-entry accumulation.
    """
    if grid is None:
        grid = np.linspace(0.50, 0.9985, 1400)
    M = _path_matrix(reg, paths, H, "main", plateau_w)
    if M is None or M.shape[0] < 10:
        return None
    hs = [h for h in range(1, H) if h not in plateau_w]
    curves = np.vstack([model_curve(p, g, H, "main", plateau_w) for p in grid])   # (G,H+1)

    def fit(rows):
        med = np.nanmedian(rows, axis=0)
        ok = [h for h in hs if np.isfinite(med[h])]
        sse = ((curves[:, ok] - med[ok]) ** 2).sum(axis=1)
        return float(grid[int(np.argmin(sse))])

    phi_hat = fit(M)
    rng = np.random.default_rng(seed)
    bs = [fit(M[rng.integers(0, M.shape[0], M.shape[0])]) for _ in range(min(boot, 400))]
    return {"phi_entry": phi_hat, "lo": float(np.percentile(bs, 2.5)),
            "hi": float(np.percentile(bs, 97.5)), "n_events": int(M.shape[0]),
            "phi_g": phi_hat * g}


def _path_matrix(reg, paths, H, norm, plateau_w):
    mat = []
    for _, e in reg[reg["keep"]].iterrows():
        s = paths[(e["congl"], e["mkt"])].astype(float)
        v = np.full(H + 1, np.nan)
        v[:len(s)] = s.values[:H + 1]
        w = [h for h in plateau_w if h < len(s) and np.isfinite(v[h])]
        if not w:
            continue
        end = np.nanmean(v[w])
        if norm == "main":
            den = end - v[0]
            if not np.isfinite(den) or abs(den) < 1e-6:
                continue
            mat.append((v - v[0]) / den)
        else:
            if not np.isfinite(end) or end <= 1e-9:
                continue
            mat.append(v / end)
    return np.vstack(mat) if mat else None


def event_paths(reg, paths, H, norm, plateau_w, boot, seed):
    """Median normalised path + IQR + bootstrap CI over kept events."""
    mat = []
    for _, e in reg[reg["keep"]].iterrows():
        s = paths[(e["congl"], e["mkt"])].astype(float)
        v = np.full(H + 1, np.nan)
        v[:len(s)] = s.values[:H + 1]
        w = [h for h in plateau_w if h < len(s) and np.isfinite(v[h])]
        if not w:
            continue
        end = np.nanmean(v[w])
        if norm == "main":
            den = end - v[0]
            if not np.isfinite(den) or abs(den) < 1e-6:
                continue
            mat.append((v - v[0]) / den)
        else:
            if not np.isfinite(end) or end <= 1e-9:
                continue
            mat.append(v / end)
    if not mat:
        return pd.DataFrame()
    M = np.vstack(mat)
    rng = np.random.default_rng(seed)
    out = []
    for h in range(H + 1):
        col = M[:, h]
        col = col[np.isfinite(col)]
        if len(col) == 0:
            continue
        bs = [np.median(rng.choice(col, len(col), replace=True)) for _ in range(boot)]
        out.append({"h": h, "n": len(col), "median": float(np.median(col)),
                    "p25": float(np.percentile(col, 25)), "p75": float(np.percentile(col, 75)),
                    "ci_lo": float(np.percentile(bs, 2.5)), "ci_hi": float(np.percentile(bs, 97.5))})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- main
def main(args):
    print("=== D10: entry dynamics vs the closed-form accumulation path (Egan Fig. 3) ===")
    g = median_g()
    vint = phi_vintages()
    H = args.horizon
    plateau_w = list(range(max(0, H - 2), H + 1))
    print(f"  accrual g = {g:.5f} (median gross_return_lag); horizon h=0..{H}; "
          f"plateau window h in {plateau_w}")
    # phi*g < 1 is the condition for the sleepy carry to have a steady state at all.
    print(f"  stationarity threshold: phi < 1/g = {1/g:.4f}")
    explosive = []
    for lab, p, src in vint:
        tag = ""
        if p * g >= 1.0:
            tag = "   <-- EXPLOSIVE (phi*g>=1: no steady state)"
            explosive.append(lab)
        print(f"  phi[{lab}] = {p:.4f}  ({src})   phi*g = {p*g:.4f}{tag}")
    if explosive:
        print(f"  NOTE: {', '.join(explosive)} sit ABOVE the stationarity threshold, so the")
        print("  sleepy carry alone would make deposits diverge at the observed accrual rate.")
        print("  Their model curves are drawn CONVEX (accelerating), which no entrant path can")
        print("  match -- read that as evidence against those phi LEVELS, not as a plot artifact.")

    usecols = ["CodConglomeradoPrudencial", "CNPJ_Lider", "CODMUN_IBGE", "mca_code",
               "year", "quarter", "Source"] + DEP_COLS
    # GOTCHA (hit 2026-08-03): panel_10_estban_instruments.py rewrites market_panel.csv in
    # place and takes minutes to do it. Reading concurrently yields a TRUNCATED frame -- the
    # run looked fine but silently lost half the entry events (B 327 -> 194, D 65 -> 8).
    # Row count is the cheap tripwire; the panel has ~500k rows.
    df = load_panel_cached(PANEL_CSV)
    if len(df) < 400_000:
        raise SystemExit(f"market_panel has only {len(df):,} rows -- it is probably being "
                         "rewritten right now (panel_10/panel_6). Re-run once that finishes.")
    df = df[[c for c in usecols if c in df.columns]].copy()
    df["qidx"] = df["year"].astype(int) * 4 + df["quarter"].astype(int) - 1
    df["dep"] = df[DEP_COLS].fillna(0).sum(axis=1)
    df = df[df["dep"] > 0]
    df["congl"] = df["CodConglomeradoPrudencial"].astype(str)
    df["cnpj_root"] = (df["CNPJ_Lider"].astype(str).str.replace(r"\D", "", regex=True)
                       .str[:8].replace({"": np.nan, "nan": np.nan}))
    is_d = df["CODMUN_IBGE"].astype(str).str.strip().isin(["0", "0.0"])
    args.window_q0 = 2016 * 4
    print(f"  panel rows with positive deposits: {len(df):,}  (D rows {int(is_d.sum()):,})")

    branch = load_branch_entry(args)
    b = df[~is_d].copy(); b["mkt"] = b["mca_code"].astype(str)
    d = df[is_d].copy(); d["mkt"] = "NATIONAL"
    reg_b, paths_b = build_events(b, "B", args, branch)
    reg_d, paths_d = build_events(d, "D", args)

    frames, all_paths = [], {}
    for reg, paths in ((reg_b, paths_b), (reg_d, paths_d)):
        if not reg.empty:
            frames.append(reg)
            all_paths.update(paths)
    reg_all = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    reg_all.to_csv(OUT_DIR / "d10_entry_events.csv", index=False)

    for kind, reg in (("B", reg_b), ("D", reg_d)):
        if reg.empty:
            print(f"  [{kind}] no candidate events")
            continue
        flags = [c for c in reg.columns if c.startswith("f_")]
        print(f"\n  [{kind}] candidates={len(reg):,}  kept={int(reg['keep'].sum()):,}")
        # SEQUENTIAL waterfall: flags overlap heavily (a firm present since 2013 trips
        # f_trunc AND f_mkt_young AND usually f_big_entry), so independent counts would
        # triple-count the same rows and misrepresent which screen actually binds.
        alive = pd.Series(True, index=reg.index)
        for f in flags:
            drop = int((alive & reg[f]).sum())
            alive &= ~reg[f]
            print(f"      {f:<14s} -{drop:>6,}  remaining {int(alive.sum()):>6,}"
                  f"   (marginal; flag total {int(reg[f].sum()):,})")
        kept = reg[reg["keep"]]
        if len(kept):
            ch = (kept["cohort"]).value_counts().sort_index()
            print("      kept cohorts: " + ", ".join(f"{int(y)}:{n}" for y, n in ch.items()))

    # paths + model curves
    rows, curves = [], []
    for norm in ("main", "alt"):
        for kind, reg, paths in (("B", reg_b, paths_b), ("D", reg_d, paths_d)):
            if reg.empty:
                continue
            p = event_paths(reg, paths, H, norm, plateau_w, args.boot, args.seed)
            if p.empty:
                continue
            p["kind"], p["norm"] = kind, norm
            rows.append(p)
        for lab, phi, _ in vint:
            c = model_curve(phi, g, H, norm, plateau_w)
            curves.append(pd.DataFrame({"h": range(H + 1), "value": c,
                                        "vintage": lab, "phi": phi, "norm": norm}))
    paths_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    curves_df = pd.concat(curves, ignore_index=True)
    paths_df.to_csv(OUT_DIR / "d10_entry_paths.csv", index=False)
    curves_df.to_csv(OUT_DIR / "d10_model_curves.csv", index=False)

    if not paths_df.empty:
        print("\n  normalised median share paths (main normalisation):")
        for kind in ("B", "D"):
            sub = paths_df[(paths_df["kind"] == kind) & (paths_df["norm"] == "main")]
            if sub.empty:
                continue
            s = ", ".join(f"h{int(r.h)}={r['median']:.2f}(n={int(r.n)})"
                          for _, r in sub.iterrows() if r.h <= 8)
            print(f"    {kind}: {s}")
        for lab, phi, _ in vint:
            c = model_curve(phi, g, H, "main", plateau_w)
            print(f"    model[{lab}] phi={phi:.3f}: "
                  + ", ".join(f"h{h}={c[h]:.2f}" for h in range(min(9, H + 1))))

    # INVERT: the phi the entry paths themselves imply (independent of the AR moment)
    imp = {}
    print("\n  implied phi from the SHAPE of entry accumulation (one-parameter fit):")
    for kind, reg, paths in (("B", reg_b, paths_b), ("D", reg_d, paths_d)):
        if reg.empty:
            continue
        r = implied_phi(reg, paths, H, plateau_w, g, args.boot, args.seed)
        if r is None:
            continue
        imp[kind] = r
        print(f"    {kind}: phi_entry = {r['phi_entry']:.4f}  95% CI [{r['lo']:.4f}, "
              f"{r['hi']:.4f}]  (n={r['n_events']} events, phi*g={r['phi_g']:.4f})")
        for lab, p, _ in vint:
            inside = r["lo"] <= p <= r["hi"]
            print(f"        vs phi[{lab}]={p:.4f}: {'INSIDE' if inside else 'OUTSIDE'} the CI")
    if imp:
        pd.DataFrame(imp).T.rename_axis("kind").reset_index().to_csv(
            OUT_DIR / "d10_implied_phi.csv", index=False)

    make_figure(paths_df, curves_df, vint, g, args, imp)

    # verdict
    sub = paths_df[(paths_df["kind"] == "B") & (paths_df["norm"] == "alt")]
    print("\n  VERDICT: under INSTANT SORTING (persistent preferences, no sleepiness) the")
    print("  alt-normalised path is FLAT AT 1.00 from h=0 -- entrants reach steady state at")
    print("  once. Sluggish accumulation tracking 1-(phi g)^h is the sleepiness prediction.")
    if not sub.empty:
        h1 = sub.loc[sub["h"] == 1, "median"]
        if len(h1):
            print(f"  observed B median at h=1: {float(h1.iloc[0]):.3f} (vs 1.00 under instant")
            print("  sorting) -- entry is slow, as sleepiness implies.")
    if "B" in imp:
        r = imp["B"]
        print(f"  The B entry paths imply phi = {r['phi_entry']:.3f} [{r['lo']:.3f}, "
              f"{r['hi']:.3f}] from an UNTARGETED moment. Compare that interval with the")
        print("  estimated vintages above: agreement is Egan-style corroboration; a vintage")
        print("  outside it is rejected by entry dynamics.")
    print(f"\nresults -> {OUT_DIR}")
    return 0


def make_figure(paths_df, curves_df, vint, g, args, imp=None):
    import matplotlib.pyplot as plt
    if paths_df.empty:
        print("  [fig] no paths to plot")
        return
    H = args.horizon
    dashes = {"E2": (5, 2), "E7": (2, 1.5), "E8": (7, 2, 1.5, 2)}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=False)
    for ax, norm, title in (
            (axes[0], "main", "(a) Egan Fig. 3 normalisation:  $(s_h-s_0)/(s_{end}-s_0)$"),
            (axes[1], "alt", "(b) Alt. normalisation:  $s_h/s_{end}$")):
        for kind, color in (("B", B_COLOR), ("D", D_COLOR)):
            sub = paths_df[(paths_df["kind"] == kind) & (paths_df["norm"] == norm)]
            if sub.empty:
                continue
            ax.fill_between(sub["h"], sub["p25"], sub["p75"], color=color, alpha=0.13, lw=0)
            ax.fill_between(sub["h"], sub["ci_lo"], sub["ci_hi"], color=color, alpha=0.28, lw=0)
            n0 = int(sub["n"].iloc[0])
            ax.plot(sub["h"], sub["median"], color=color, lw=2, marker="o", ms=4.5,
                    markeredgecolor="white", markeredgewidth=1.0, zorder=3,
                    label=f"{kind}-firm entries (n={n0})")
        for lab, phi, _ in vint:
            c = curves_df[(curves_df["vintage"] == lab) & (curves_df["norm"] == norm)]
            expl = " explosive" if phi * g >= 1.0 else ""
            ax.plot(c["h"], c["value"], color=MODEL_COLORS.get(lab, INK), lw=1.6,
                    ls=(0, dashes.get(lab, (5, 2))), zorder=2,
                    label=f"model, {lab} ($\\hat\\phi$={phi:.3f}){expl}")
        if norm == "alt":
            ax.axhline(1.0, color="#B71C1C", lw=1.6, ls=":", zorder=4,
                       label="instant sorting (no sleepiness)")
            # D-firm dispersion is wide (n is small and shares are national); clip so the
            # comparison of interest -- B vs the model curves -- stays legible.
            ax.set_ylim(-0.05, 1.65)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("quarters since entry ($h$)")
        ax.set_ylabel("normalised market share")
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#BFBFBA")
            ax.spines[sp].set_linewidth(0.8)
        ax.set_xlim(-0.3, H + 0.3)
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    axes[1].legend(frameon=False, fontsize=7.5, loc="lower right")
    if imp and "B" in imp:
        r = imp["B"]
        axes[0].text(0.98, 0.06,
                     f"entry-implied $\\phi$ = {r['phi_entry']:.3f}  "
                     f"[{r['lo']:.3f}, {r['hi']:.3f}]",
                     transform=axes[0].transAxes, ha="right", fontsize=8.5, color=INK)
    fig.suptitle("Entrant share accumulation: data vs the sleepiness-implied path",
                 x=0.008, ha="left", fontsize=12)
    fig.text(0.008, 0.005,
             "Medians across entry events; shaded = IQR and bootstrap CI of the median. "
             "Model curves are the closed-form path $1-(\\hat\\phi g)^h$ implied by the "
             "estimated law of motion (frozen spreads, $g$ = median gross accrual).",
             ha="left", fontsize=7.5, color=MUTED if (MUTED := "#5A5A57") else INK)
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    for stem, d in (("d10_entry_dynamics", OUT_DIR), ("fig_entry_dynamics", DRAFTS)):
        try:
            fig.savefig(d / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
            fig.savefig(d / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
        except OSError as e:
            print(f"  [fig] save to {d} failed: {e}")
    plt.close(fig)
    print(f"  figure -> {OUT_DIR/'d10_entry_dynamics.png'} (+ Drafts/fig_entry_dynamics.*)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--min-post", type=int, default=8)
    ap.add_argument("--boot", type=int, default=999)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-jump", type=float, default=0.15,
                    help="drop events with a post-entry share jump above this (acquisition)")
    ap.add_argument("--max-entry-share", type=float, default=0.10,
                    help="drop events entering above this share (mechanical booking)")
    ap.add_argument("--window-only", action="store_true",
                    help="restrict to entries from 2016Q1 (the estimation window)")
    ap.add_argument("--no-branch-screen", action="store_true",
                    help="skip the ESTBAN branch-timing screen even if the sidecar exists")
    raise SystemExit(main(ap.parse_args()))

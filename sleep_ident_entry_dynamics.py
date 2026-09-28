"""sleep_ident_entry_dynamics.py -- D6: do entrants accumulate share at the speed phi-hat implies?

Author: Pedro Feijo de Moraes

This is the Brazilian analogue of Egan et al. (2025) Figure 3 -- the one moment that
discriminates SLEEPINESS from PERSISTENT PREFERENCES, and the piece our draft lacked.

    Persistent preferences + full reoptimisation => an entrant jumps immediately to its
    steady-state share (everyone who prefers it sorts in at once).
    Sleepiness => the entrant can only reach the (1-phi) awake fraction each period, so
    share accumulates slowly, with a curvature pinned by phi.

The model line is CLOSED FORM. With frozen spreads the law of motion
(cf_deposit_sim.jl:192)

    Dep_{t+1} = (1-phi)*M*s + phi*g*Dep_t,      g = 1 + r^dep_q

is a scalar affine map, so from Dep_0 = 0

    Dep_h = A*(1-(phi g)^h)/(1-phi g),   A = (1-phi)*M*s,   Dep_h/Dep_inf = 1-(phi g)^h.

Rivals enter only through A (the level), which BOTH Egan normalisations cancel. So the
model curve needs phi-hat and g alone -- no BLP context and no simulation draws. What DOES
cost is everything around it: the full market panel, every routine's phi_nopix export, and
a 999-draw resample of the entry events behind the implied-phi interval.

Two normalisations, exactly as in the paper:
  main (their Fig. 3):  n_h = (s_h - s_0)/(s_end - s_0)
  alt  (their Fig. A1): n_h =  s_h / s_end,  where the instant-sorting benchmark is the
                        flat line at 1 (it is 0/0 in the main normalisation, so it is
                        drawn on the ALT panel only -- same as the paper).

Outputs -> DIAG_PHI_SEPARATION/d6_entry_{events,paths,model_curves}.csv,
           d6_implied_phi.csv, d6_routine_{curves,panel}*.csv, d6_meta.json,
           d6_entry_dynamics.png/.pdf, and fig_entry_dynamics.png/.pdf in Drafts.

TWO HALVES, ONE PASS BY DEFAULT.
  --compute-only   every moment, every bootstrap, every CSV + d6_meta.json; no figure.
                   This is the cluster half (sleep_job.sh SLEEP_STEP=entry).
  --figures-only   every exhibit, drawn from those CSVs; nothing recomputed. The local half.
  neither          compute and draw in one pass, which is what it has always done.
Nothing about the split changes a number: the figure functions already took only the frames
the compute half writes, so the seam is where the data already was. d6_meta.json carries the
handful of scalars no CSV holds -- g, the horizon, the phi vintages and the settings the run
used -- and --figures-only adopts those settings rather than its own, so a render cannot
caption a figure with a construction the CSVs were not built under.
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import argparse
import json
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import shutil

import numpy as np
import pandas as pd

from utils import paths as _paths
from utils import routines as _routines
from utils import load_panel_cached

OUT_DIR = _paths.PROCESSED / "ESTIMATION_OUTPUT" / "DIAG_PHI_SEPARATION"
DRAFTS = _paths.OPEN_FINANCE / "Drafts" / "Deposit Competition"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PANEL_CSV = _paths.PROCESSED / "market_panel.csv"
BRANCH_SIDECAR = _paths.PROCESSED / "PANEL_INTERMED" / "estban_own_branches.csv"
DEP_COLS = ["dep_a1", "dep_a2", "dep_a4"]          # a5 (prepaid) is NaN in ESTBAN

# figure palette (make_desc_compressed_tables.py convention)
B_COLOR, D_COLOR, INK, GRID = "#1565C0", "#E64A19", "#4F4F4F", "#D5D5D0"
# Every routine that can appear in REF_ESTS needs its own hue: an unkeyed line falls back to
# INK, and two INK lines are indistinguishable in the legend.
MODEL_COLORS = {"E1": "#455A64", "E2": "#2E7D32",
                "E3": "#6A1B9A", "E4": "#00838F"}


# ----------------------------------------------------------------------------- phi vintages
# The reference lines are the reported lineup, so the figure shows every level the entry
# moment has to adjudicate. Restrict with SLEEP_ENTRY_REFS (comma-separated) when a panel
# should carry fewer lines -- e.g. "E3,E4" for the single-index pair alone, which keeps the
# linear routines' spec-12 averages off a panel whose companion table reports their
# market-means, two different objects that read as a discrepancy side by side.
REF_ESTS = tuple(x.strip() for x in
                 os.environ.get("SLEEP_ENTRY_REFS",
                                ",".join(f"E{e}" for e in _routines.LINK_ESTS)
                                ).split(",") if x.strip())


def phi_vintages():
    """(label, phi, source) for each model line, all from the CF_FOUNDATION phi_nopix
    exports so every line is the same object measured the same way. Nothing here is a frozen
    literal -- a stale model line plotted against fresh data is the one failure this figure
    cannot survive."""
    cf = _paths.cf_foundation_dir()
    out = []
    for est in REF_ESTS:
        fp = cf / f"phi_nopix_{est}_spec_12.parquet"
        if not fp.exists():
            print(f"  [phi] {est}: {fp.name} absent -- line skipped")
            continue
        m = float(pd.read_parquet(fp, columns=["phi_mt"])["phi_mt"].mean())
        ts = pd.Timestamp(fp.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M")
        out.append((est, m, f"mean phi_mt of {fp.name} [{ts}]"))
    return out


# WHY THE ACCESSORS AND NOT PROCESSED/"ESTIMATION_OUTPUT"/... . Both inputs move on the cluster
# and only there: cl_export_step_dirs exports DEMAND_PREP_DIR -> data/output/demand_prep and
# CF_FOUNDATION_DIR -> data/output/counterfactuals (cluster_lib.sh:79-80), and
# demand_parquet_dir()/cf_foundation_dir() are the accessors that honour them. Spelled as the
# literal, this script reads data/output/DEMAND_PREP and data/output/CF_FOUNDATION -- folders
# nothing writes -- so SLEEP_STEP=entry dies on median_g()'s first read, and the phi_nopix lookups
# below degrade to a printed "absent -- skipped" that leaves the exhibit empty while the job still
# exits 0. OUT_DIR above is deliberately NOT converted: estimation_output() is the one anchor
# cl_export_step_dirs does not redirect, so DIAG_PHI_SEPARATION resolves to data/output there,
# which is exactly where cluster_archive.sh --set sleep globs it from.
def median_g():
    fp = _paths.demand_parquet_dir() / "demand_2_spec_12.parquet"
    g = float(pd.read_parquet(fp, columns=["gross_return_lag"])["gross_return_lag"].median())
    assert 1.0 < g < 1.06, f"implausible accrual g={g}"
    return g


# ------------------------------------------------------------------------- per-event accrual
# WHY THIS EXISTS. The moment identifies the PRODUCT phi*g, not phi: phi is recovered only by
# dividing the fitted phi*g by an assumed g. A single scalar g therefore transmits one-for-one
# into the reported level -- measured on the spec-12 frame, holding phi*g at its fitted 0.9270,
# the implied phi runs 0.9285 at p10 g to 0.9059 at p90 g, a 0.023 span against a bootstrap CI
# width of 0.045. The variance decomposition says where to get it back: type x quarter carries
# 89.6% of the between-group variance and entity 76.2%, while MARKET carries 3.7% -- so the cut
# is (entity, type, quarter) and a per-market g would add noise, not signal.
#
# The phi side of this test is already per-event (each event uses its own market's fitted
# phi_m); g was the only input still collapsed, which made the model path a median over events
# in phi but a constant in g while the data path is a median over events in everything.
GMODE_CHOICES = ("path", "event", "scalar")
# One suffix map for every phi-resolution artifact -- CSVs and figures alike, so a cell
# of the aggregation 2x2 can never write over another.
PHI_SFX = {"path": "_phipath", "market": "", "time": "_phitime",
           "scalar": "_phiscalar"}
# Raw strings: "$\bar\phi$" in a normal literal puts an actual backspace in the label, and
# mathtext then fails to parse it at savefig time.
PHI_LABEL = {"path": r"$\phi_{m,t}$ (no averaging)",
             "market": r"$\phi_m$ (averaged over quarters)",
             "time": r"$\phi_t$ (averaged over markets)",
             "scalar": r"$\bar{\phi}$ (averaged over both)"}
G_TYPES = (1, 2, 4)          # event deposits are dep_a1+dep_a2+dep_a4; type 5 (g ~ 1.025)
                             # is in the parquet but NEVER entered the path being modelled.


def build_g_paths(reg, H, mode, scalar_g):
    """(g_matrix, diagnostics) for the kept events, aligned to reg[reg.keep] row order.

    g_matrix is (n_kept, H+1): the gross accrual each event faces at horizon h, i.e. at
    calendar quarter q_entry + h. Sources, in order, each a deposit-weighted mean of
    `gross_return_lag` over types 1/2/4:

      own      the event's own (congl, mkt, quarter) cells -- the deposits actually modelled
      entity   the conglomerate's cells across all its markets, same quarter
      national all cells that quarter

    Quarters outside the parquet's 2016Q1-2024Q4 window (late entrants' tails, and the
    2015Q4 cohort's first quarters) take the event's nearest observed value, carried flat.
    That is a horizon-edge convention, not an estimate: those h feed the plateau or the
    pinned h=0 rather than the fitted horizons.
    """
    kept = reg[reg["keep"]]
    n = len(kept)
    if mode == "scalar" or n == 0:
        return np.full((n, H + 1), float(scalar_g)), None

    fp = _paths.demand_parquet_dir() / "demand_2_spec_12.parquet"
    d = pd.read_parquet(fp, columns=["CodConglomeradoPrudencial", "mca_code", "deposit_type",
                                     "year", "quarter", "gross_return_lag", "deposit_balance"])
    d = d[d["deposit_type"].astype(int).isin(G_TYPES)].copy()
    d["congl"] = d["CodConglomeradoPrudencial"].astype(str)
    d["mkt"] = d["mca_code"].astype(str)
    d["q"] = d["year"].astype(int) * 4 + d["quarter"].astype(int) - 1
    d["w"] = d["deposit_balance"].astype(float).clip(lower=0.0)
    d["wg"] = d["w"] * d["gross_return_lag"].astype(float)

    def _wmean(keys):
        a = d.groupby(keys, observed=True)[["wg", "w"]].sum()
        return (a["wg"] / a["w"].replace(0.0, np.nan)).dropna()

    own = _wmean(["congl", "mkt", "q"])
    ent = _wmean(["congl", "q"])
    nat = d.groupby("q", observed=True)["gross_return_lag"].median()

    congl = kept["congl"].astype(str).to_numpy()
    mkt = kept["mkt"].astype(str).to_numpy()
    q0 = kept["q_entry"].astype(int).to_numpy()

    G = np.full((n, H + 1), np.nan)
    src = np.zeros((n, H + 1), dtype=np.int8)          # 1 own, 2 entity, 3 national, 0 none
    for h in range(H + 1):
        q = q0 + h
        # copy=True: a reindex that misses nothing hands back a read-only view of the
        # source, and the fallback assignments below write in place.
        v = own.reindex(pd.MultiIndex.from_arrays([congl, mkt, q])).to_numpy(float, copy=True)
        src[:, h] = np.where(np.isfinite(v), 1, 0)
        m = ~np.isfinite(v)
        if m.any():
            e = ent.reindex(pd.MultiIndex.from_arrays([congl[m], q[m]])).to_numpy(float)
            v[m] = e
            src[m, h] = np.where(np.isfinite(e), 2, 0)
        m = ~np.isfinite(v)
        if m.any():
            nv = nat.reindex(q[m]).to_numpy(float)
            v[m] = nv
            src[m, h] = np.where(np.isfinite(nv), 3, 0)
        G[:, h] = v

    # flat carry across the panel edges, per event, then a global floor for any event with no
    # observed cell at all (cannot happen with the national fallback, but keep it total).
    Gf = pd.DataFrame(G).ffill(axis=1).bfill(axis=1).to_numpy(float)
    n_extrap = int((~np.isfinite(G) & np.isfinite(Gf)).sum())
    Gf = np.where(np.isfinite(Gf), Gf, float(scalar_g))

    if mode == "event":
        obs = src > 0
        num = np.where(obs, Gf, 0.0).sum(axis=1)
        den = obs.sum(axis=1)
        ge = np.where(den > 0, num / np.maximum(den, 1), float(scalar_g))
        Gf = np.repeat(ge[:, None], H + 1, axis=1)

    diag = pd.DataFrame({
        "congl": congl, "mkt": mkt, "q_entry": q0,
        "g_e_mean": Gf.mean(axis=1),
        "g_e_min": Gf.min(axis=1), "g_e_max": Gf.max(axis=1),
        "n_g_cells_own": (src == 1).sum(axis=1),
        "n_g_fallback_entity": (src == 2).sum(axis=1),
        "n_g_fallback_national": (src == 3).sum(axis=1),
        "n_g_extrapolated": (src == 0).sum(axis=1),
    })
    tot = src.size
    print(f"  [g:{mode}] {n} events x {H+1} horizons: own {100*(src==1).sum()/tot:.1f}%, "
          f"entity {100*(src==2).sum()/tot:.1f}%, national {100*(src==3).sum()/tot:.1f}%, "
          f"carried {100*n_extrap/tot:.1f}%")
    print(f"  [g:{mode}] g_e mean over events {Gf.mean():.5f}  "
          f"p10 {np.percentile(Gf.mean(axis=1), 10):.5f}  "
          f"p90 {np.percentile(Gf.mean(axis=1), 90):.5f}  "
          f"(scalar reference {scalar_g:.5f})")
    assert np.isfinite(Gf).all(), "per-event g has non-finite entries"
    return Gf, diag


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
    gv = np.asarray(g, dtype=float)
    if gv.ndim == 0:
        pg = phi * float(gv)
        if abs(pg - 1.0) < 1e-9:                   # knife-edge: linear accumulation
            lvl = np.array([float(h + 1) for h in range(H + 1)])
        else:
            lvl = np.array([(1.0 - pg ** (h + 1)) / (1.0 - pg) for h in range(H + 1)])
    else:
        # g varies along the path, so the geometric sum has no closed form: iterate the same
        # affine map the closed form solves, Dep_h = phi*g_h*Dep_{h-1} + A, at A = 1. With g
        # constant this returns the branch above to floating-point tolerance -- verified by
        # --g-selftest -- and it handles phi*g crossing 1 mid-path without a special case.
        lvl = np.empty(H + 1)
        lvl[0] = 1.0
        for h in range(1, H + 1):
            lvl[h] = phi * gv[h] * lvl[h - 1] + 1.0
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
              "(run panel_estban_instrument.py to create it)")
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


def implied_phi(reg, paths, H, plateau_w, g, boot, seed, grid=None, loss="median"):
    """INVERT the figure: what phi do the entry paths themselves imply?

    One-parameter fit over h=1..H-1 (h=0 and the plateau window are pinned to 0/1 by the
    normalisation and carry no information). The bootstrap resamples EVENTS, refits phi on
    each draw, and returns the percentile CI.

    This is a genuine second measure of phi, independent of the deposit-autocorrelation
    moment that produced phi-hat: it uses only the SHAPE of post-entry accumulation.

    TWO LOSSES, both reported, because the choice is not innocuous:

      median  minimise the squared distance between the MEDIAN model path and the MEDIAN
              data path. Robust, and the presentation Egan et al. use -- but it takes the
              two medians separately, so the median data path is not the one the median
              accrual generates: the event-to-accrual pairing is broken by the aggregation.
      pooled  minimise summed ABSOLUTE deviations over every (event, horizon) cell, each
              event against its own accrual. Keeps the pairing and uses every event.

    Squared loss on the pooled cells is NOT offered, and deliberately: each path is divided
    by its own three-quarter plateau mean, so it is a ratio with a noisy denominator, and the
    individual paths run from -2.4 to +12.1 against a median path inside [0.11, 0.90]. Pooled
    least squares returns 0.852 against 0.9165 here -- a shift of 1.4 CI widths driven by a
    handful of small-denominator events. Pooled ABSOLUTE deviation on the same cells returns
    0.914, i.e. the movement was the loss function meeting heavy tails, not extra information.
    """
    if grid is None:
        grid = np.linspace(0.50, 0.9985, 1400)
    M, keep_rows = _path_matrix(reg, paths, H, "main", plateau_w, want_rows=True)
    if M is None or M.shape[0] < 10:
        return None
    hs = [h for h in range(1, H) if h not in plateau_w]
    Gv = np.asarray(g, dtype=float)
    per_event = Gv.ndim == 2

    if per_event:
        # One model curve PER EVENT at each phi, medianed the way the data is. The bootstrap
        # then resamples events and their g paths JOINTLY, so cross-event g dispersion enters
        # the CI instead of being assumed away.
        T = model_tensor(grid, Gv[keep_rows], H, plateau_w)          # (G,E,H+1)

        def fit(idx):
            med = np.nanmedian(M[idx], axis=0)
            ok = [h for h in hs if np.isfinite(med[h])]
            mod = np.median(T[:, idx, :][:, :, ok], axis=1)          # model has no NaNs
            return float(grid[int(np.argmin(((mod - med[ok]) ** 2).sum(axis=1)))])
    else:
        curves = np.vstack([model_curve(p, float(Gv), H, "main", plateau_w) for p in grid])

        def fit(idx):
            med = np.nanmedian(M[idx], axis=0)
            ok = [h for h in hs if np.isfinite(med[h])]
            sse = ((curves[:, ok] - med[ok]) ** 2).sum(axis=1)
            return float(grid[int(np.argmin(sse))])

    def fit_pooled(idx):
        """Every (event, horizon) cell against that event's own accrual, absolute loss."""
        A = M[idx][:, hs]
        ok = np.isfinite(A)
        if per_event:
            Bm = T[:, idx, :][:, :, hs]
        else:
            Bm = np.repeat(curves[:, hs][:, None, :], len(idx), axis=1)
        r = np.where(ok[None, :, :], np.abs(Bm - A[None, :, :]), 0.0)
        return float(grid[int(np.argmin(r.sum(axis=(1, 2))))])

    n = M.shape[0]
    rng = np.random.default_rng(seed)
    # One set of resampled indices drives both estimators, so the two intervals are paired
    # and any difference between them is the loss function rather than the draws.
    draws = [rng.integers(0, n, n) for _ in range(min(boot, 400))]
    phi_med = fit(np.arange(n))
    bs_med = [fit(d_) for d_ in draws]
    # The pooled fit touches grid x events x horizons per draw; cap it so a full run stays
    # in the same order of magnitude as before.
    d_pool = draws[:min(len(draws), 200)]
    phi_pool = fit_pooled(np.arange(n))
    bs_pool = [fit_pooled(d_) for d_ in d_pool]

    head = phi_pool if loss == "pooled" else phi_med
    head_bs = bs_pool if loss == "pooled" else bs_med
    g_row = Gv[keep_rows].mean(axis=1) if per_event else np.full(n, float(Gv))
    return {"phi_entry": head, "lo": float(np.percentile(head_bs, 2.5)),
            "hi": float(np.percentile(head_bs, 97.5)), "n_events": int(n),
            "loss": loss,
            "phi_median_loss": phi_med, "lo_median_loss": float(np.percentile(bs_med, 2.5)),
            "hi_median_loss": float(np.percentile(bs_med, 97.5)),
            "phi_pooled_abs": phi_pool, "lo_pooled_abs": float(np.percentile(bs_pool, 2.5)),
            "hi_pooled_abs": float(np.percentile(bs_pool, 97.5)),
            "phi_g": head * float(np.median(g_row)),
            "g_e_median": float(np.median(g_row)),
            "g_e_p10": float(np.percentile(g_row, 10)),
            "g_e_p90": float(np.percentile(g_row, 90))}


def _path_matrix(reg, paths, H, norm, plateau_w, want_rows=False):
    """Normalised event paths, one row per usable event.

    Events without an observed plateau horizon, or with a degenerate denominator, are
    dropped. `want_rows` also returns their positions within reg[reg.keep] -- the per-event
    g matrix is built in that order, so the caller needs them to keep g aligned with the
    surviving data rows.
    """
    mat, rows = [], []
    for i, (_, e) in enumerate(reg[reg["keep"]].iterrows()):
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
        rows.append(i)
    M = np.vstack(mat) if mat else None
    return (M, np.asarray(rows, dtype=int)) if want_rows else M


def model_curve_phipath(phi, g, H, norm, plateau_w):
    """Model path when BOTH phi and g vary along the horizon. Same affine map as
    model_curve; the only difference is that the multiplier is phi_h*g_h rather than
    phi*g_h, so nothing about the entrant's environment is held at an average."""
    ph = np.asarray(phi, float)
    gv = np.asarray(g, float)
    if ph.ndim == 0:
        return model_curve(float(ph), gv, H, norm, plateau_w)
    lvl = np.empty(H + 1)
    lvl[0] = 1.0
    for h in range(1, H + 1):
        lvl[h] = ph[h] * gv[h] * lvl[h - 1] + 1.0
    end = lvl[plateau_w].mean()
    return (lvl - lvl[0]) / (end - lvl[0]) if norm == "main" else lvl / end


def vintage_curve(phi, g, H, norm, plateau_w):
    """One model line for a single phi. With a per-event g this is the MEDIAN over the
    events' own curves -- the same aggregation the data median uses, so the comparison is
    like-for-like rather than a curve at an average accrual."""
    gv = np.asarray(g, dtype=float)
    if gv.ndim < 2:
        return model_curve(phi, gv if gv.ndim else float(gv), H, norm, plateau_w)
    C = np.vstack([model_curve(phi, gv[i], H, norm, plateau_w) for i in range(gv.shape[0])])
    return np.nanmedian(C, axis=0)


def model_tensor(grid, Ge, H, plateau_w):
    """(n_phi, n_events, H+1) main-normalised model curves, vectorised over the phi grid.

    Same recursion as model_curve's array branch, evaluated for every (phi, event) pair at
    once so the bootstrap can re-median without rebuilding curves.
    """
    P = np.asarray(grid, float)[:, None]
    Gm = np.asarray(Ge, float)[None, :, :]
    lvl = np.empty((P.shape[0], Gm.shape[1], H + 1))
    lvl[:, :, 0] = 1.0
    for h in range(1, H + 1):
        lvl[:, :, h] = P * Gm[:, :, h] * lvl[:, :, h - 1] + 1.0
    end = lvl[:, :, plateau_w].mean(axis=2)
    return (lvl - lvl[:, :, [0]]) / (end - lvl[:, :, 0])[:, :, None]


_POP_G_CACHE = {}


def phi_in_interval(ests, lo, hi):
    """How much of each routine's fitted phi lies inside the entry-implied interval.

    The level comparison sets ONE number per routine against [lo, hi]. That is silent on how
    much of the fitted distribution the interval covers -- a routine can have the right mean
    with almost no mass inside, or the wrong mean with substantial mass inside. Reported at
    the three resolutions the aggregation 2x2 uses, plus deposit-weighted, since the
    counterfactuals weight cells by deposits rather than counting them.

    READ IT AS A COMPARISON ACROSS ROUTINES, NOT AS PASS/FAIL. [lo, hi] is a confidence
    interval for a single central phi, not a tolerance region for a distribution: a routine
    whose phi genuinely varies across markets SHOULD place mass outside it. What is
    interpretable is the same yardstick applied to every routine -- and in particular a share
    of exactly zero, which says a routine has no admissible cell anywhere.
    """
    cf = _paths.cf_foundation_dir()
    dp = _paths.demand_parquet_dir() / "demand_2_spec_12.parquet"
    w = pd.read_parquet(dp, columns=["mca_code", "time_id", "deposit_type",
                                     "deposit_balance"])
    w = w[w["deposit_type"].astype(int).isin(G_TYPES)]
    wmt = w.groupby(["mca_code", "time_id"], observed=True)["deposit_balance"].sum()

    def _sh(v):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        if not v.size:
            return np.nan, np.nan, np.nan, 0
        return (float(np.mean(v < lo)), float(np.mean((v >= lo) & (v <= hi))),
                float(np.mean(v > hi)), int(v.size))

    rows = []
    for est in ests:
        fp = cf / f"phi_nopix_E{est}_spec_12.parquet"
        if not fp.exists():
            continue
        d = pd.read_parquet(fp, columns=["mca_code", "time_id", "deposit_type", "phi_mt"])
        d = d[d["deposit_type"].astype(int).isin(G_TYPES)]
        mt = d.groupby(["mca_code", "time_id"], observed=True)["phi_mt"].mean()
        r = {"estim": est, "lo": lo, "hi": hi, "phi_mean": float(d["phi_mt"].mean())}
        for tag, v in (("mq", mt),
                       ("mkt", d.groupby("mca_code", observed=True)["phi_mt"].mean()),
                       ("qtr", d.groupby("time_id", observed=True)["phi_mt"].mean())):
            b, i, a, n = _sh(v)
            r |= {f"{tag}_below": b, f"{tag}_in": i, f"{tag}_above": a, f"{tag}_n": n}
        j = pd.concat([mt.rename("phi"), wmt.rename("w")], axis=1).dropna()
        ins = (j["phi"] >= lo) & (j["phi"] <= hi)
        r["mq_in_depwt"] = float(j.loc[ins, "w"].sum() / j["w"].sum()) if len(j) else np.nan
        rows.append(r)
        print(f"  [phi-in-CI] E{est}: market-quarters {100*r['mq_in']:.1f}% in "
              f"({100*r['mq_below']:.1f}% below, {100*r['mq_above']:.1f}% above) | "
              f"markets {100*r['mkt_in']:.1f}% | quarters {100*r['qtr_in']:.1f}% | "
              f"deposit-weighted {100*r['mq_in_depwt']:.1f}%")
    if rows:
        pd.DataFrame(rows).to_csv(OUT_DIR / "d6_phi_in_interval.csv", index=False)
        print("  -> d6_phi_in_interval.csv")
    return rows


def population_stationarity(est):
    """Share of ALL (market, quarter) cells whose fitted carry breaches phi*g >= 1.

    The entry-event diagnostics speak only for the markets an entrant happened to enter.
    This is the same bound evaluated over every market-quarter the routine fits, which is
    the population the counterfactuals actually simulate.
    """
    cf = _paths.cf_foundation_dir()
    fp = cf / f"phi_nopix_E{est}_spec_12.parquet"
    if not fp.exists():
        return {}
    if "g" not in _POP_G_CACHE:
        dp = _paths.demand_parquet_dir() / "demand_2_spec_12.parquet"
        q = pd.read_parquet(dp, columns=["mca_code", "time_id", "deposit_type",
                                         "gross_return_lag", "deposit_balance"])
        q = q[q["deposit_type"].astype(int).isin(G_TYPES)].copy()
        q["w"] = q["deposit_balance"].astype(float).clip(lower=0.0)
        q["wg"] = q["w"] * q["gross_return_lag"].astype(float)
        a = q.groupby(["mca_code", "time_id"], observed=True)[["wg", "w"]].sum()
        _POP_G_CACHE["g"] = (a["wg"] / a["w"].replace(0.0, np.nan)).rename("g")
    gmt = _POP_G_CACHE["g"]
    d = pd.read_parquet(fp, columns=["mca_code", "time_id", "deposit_type", "phi_mt"])
    d = d[d["deposit_type"].astype(int).isin(G_TYPES)]
    d = (d.groupby(["mca_code", "time_id"], observed=True)["phi_mt"].mean()
         .rename("phi").to_frame().join(gmt, how="inner"))
    if d.empty:
        return {}
    pg = d["phi"] * d["g"]
    return {"pop_cells": int(len(d)), "pop_share_expl": float((pg >= 1.0).mean()),
            "pop_phi_max": float(d["phi"].max()), "pop_pg_max": float(pg.max())}


def explosive_diagnostics(est, Pm, Gm):
    """Depth, persistence and breadth of the stationarity breach for one routine.

    The headline shares say how OFTEN phi*g >= 1. They do not say how far past the bound it
    goes, nor whether it is a fixed property of a market or an episode the rate cycle drives
    -- and those change the reading, so they are measured rather than asserted.
    """
    PG = Pm * Gm
    expl = PG >= 1.0
    exc = PG[expl] - 1.0
    out = {"estim": est, "phi_mean": float(Pm.mean()),
           "phi_g_median": float(np.median(PG)),
           "phi_g_p90": float(np.percentile(PG, 90)), "phi_g_max": float(PG.max()),
           "share_cells_expl": float(expl.mean()),
           "share_events_ever": float(expl.any(axis=1).mean()),
           "share_events_always": float(expl.all(axis=1).mean()),
           "mean_excess": float(exc.mean()) if exc.size else 0.0,
           "n_events": int(Pm.shape[0])}
    out.update(population_stationarity(est))
    return out


def _qidx_to_time_id(q):
    """qidx = year*4 + quarter - 1  ->  the parquet's 'YYYYQq' label."""
    return f"{int(q) // 4}Q{int(q) % 4 + 1}"


def build_phi_paths(kept, H, est, mode):
    """(n_kept, H+1) fitted phi for each event at each horizon, plus a source tally.

    phi_mt carries exactly two indices, so there are four ways to feed it to the entry
    recursion -- a 2x2 over "keep the market variation?" and "keep the time variation?":

        path    m and t both kept. The resolution phi_mt is estimated at, and the resolution
                the demand parquet and the counterfactuals consume. Collapses nothing.
        market  average over t. Isolates the cross-market level assignment.
        time    average over m. Isolates the fitted time path.
        scalar  average over both. The pure level test, closest to Egan's own.

    They are not interchangeable, and which one flatters a routine depends on where that
    routine put its variance: E4's phi is 3.1% between-quarter and 60.5% between-market, so
    averaging over t costs it almost nothing, while a routine that loads on the time index
    instead loses most of what it fitted to the same average. An aggregation choice is a
    choice about whose estimate to amputate, which is why all four are reported rather than
    one.

    Deposit types are restricted to 1/2/4, matching the deposits an entry path is built
    from. That also removes a boundary artifact: within a (market, quarter) cell phi is
    exactly constant for 90% of E4's cells, and the exceptions are rows the single-index
    link clipped to zero -- in the worst cell, types 1/2/4 sit at 0.0 while type 5 sits at
    0.984. Averaging across types would mix a clipped value with a live one; averaging
    within the modelled types does not.
    """
    cf = _paths.cf_foundation_dir()
    fp = cf / f"phi_nopix_E{est}_spec_12.parquet"
    if not fp.exists():
        return None, None
    pm = pd.read_parquet(fp, columns=["mca_code", "time_id", "deposit_type", "phi_mt"])
    pm = pm[pm["deposit_type"].astype(int).isin(G_TYPES)]
    pm["mca_code"] = pm["mca_code"].astype(str)
    by_mt = pm.groupby(["mca_code", "time_id"], observed=True)["phi_mt"].mean()
    by_m = pm.groupby("mca_code", observed=True)["phi_mt"].mean()
    by_t = pm.groupby("time_id", observed=True)["phi_mt"].mean()
    glob = float(pm["phi_mt"].mean())

    mkt = kept["mkt"].astype(str).to_numpy()
    q0 = kept["q_entry"].astype(int).to_numpy()
    n = len(kept)
    if mode == "scalar":
        return np.full((n, H + 1), glob), {"hit": 1.0, "exact": 0.0}
    if mode == "market":
        v = by_m.reindex(mkt).to_numpy(float, copy=True)
        hit = float(np.isfinite(v).mean())
        v = np.where(np.isfinite(v), v, glob)
        return np.repeat(v[:, None], H + 1, axis=1), {"hit": hit, "exact": 0.0}
    if mode == "time":
        # Market variation averaged away; every event at the national phi of its own quarter.
        P_ = np.full((n, H + 1), np.nan)
        for h in range(H + 1):
            tid = [_qidx_to_time_id(q) for q in q0 + h]
            P_[:, h] = by_t.reindex(tid).to_numpy(float, copy=True)
        P_ = np.where(np.isfinite(P_), P_, glob)
        return P_, {"hit": 1.0, "exact": 0.0}

    P_ = np.full((n, H + 1), np.nan)
    exact = 0
    for h in range(H + 1):
        tid = [_qidx_to_time_id(q) for q in q0 + h]
        v = by_mt.reindex(pd.MultiIndex.from_arrays([mkt, tid])).to_numpy(float, copy=True)
        exact += int(np.isfinite(v).sum())
        P_[:, h] = v
    # market mean where that quarter is outside the estimation window, then flat carry
    fill = by_m.reindex(mkt).to_numpy(float, copy=True)
    fill = np.where(np.isfinite(fill), fill, glob)
    P_ = np.where(np.isfinite(P_), P_, fill[:, None])
    hit = float(np.isfinite(by_m.reindex(mkt).to_numpy(float)).mean())
    return P_, {"hit": hit, "exact": exact / max(n * (H + 1), 1)}


def routine_event_curves(reg, paths, H, plateau_w, g, ests, phi_mode="market"):
    """Per-routine model paths built from each routine's OWN fitted dispersion.

    The scalar model line collapses a routine's fitted phi_mt to one number; this is the
    D4c '--routine-bands' idea transported to entry dynamics. For each routine N, every kept
    B event gets the model curve at ITS market's mean fitted phi_m (from the CF_FOUNDATION
    phi_nopix export, which carries phi_mt through the routine's own link), and the
    routine's prediction is the MEDIAN of those per-event curves -- the routine as
    estimated, phi common within a market and varying across markets, exactly what the
    depositor-attention theory permits. Per routine it also reports the share of events in
    the explosive regime phi_m*g >= 1 (where no steady-state share exists and the curve is
    convex -- see model_curve), and the SSE against the empirical median path on the same
    horizons the implied-phi fit uses.
    """
    cf = _paths.cf_foundation_dir()
    M = _path_matrix(reg, paths, H, "main", plateau_w)
    if M is None:
        print("  [routine curves] no usable events"); return None
    emp = np.nanmedian(M, axis=0)
    hs = [h for h in range(1, H) if h not in plateau_w and np.isfinite(emp[h])]
    kept = reg[reg["keep"]].copy()
    kept["_mkt"] = kept["mkt"].astype(str)
    rows, curves, panel_rows, expl_rows = [], {}, [], []
    for est in ests:
        fp = cf / f"phi_nopix_E{est}_spec_12.parquet"
        if not fp.exists():
            print(f"  [routine curves] E{est}: {fp.name} absent -- skipped"); continue
        Pm, pinfo = build_phi_paths(kept, H, est, phi_mode)
        if Pm is None:
            print(f"  [routine curves] E{est}: {fp.name} absent -- skipped"); continue
        hit = pinfo["hit"]
        if hit < 0.90:
            print(f"  [routine curves] E{est}: market map hit-rate {hit:.0%} < 90% -- "
                  f"skipped (check mca key compatibility)"); continue
        v = Pm.mean(axis=1)                     # per-event summary, for the reported moments
        Gv = np.asarray(g, dtype=float)
        Gm = Gv if Gv.ndim == 2 else np.full_like(Pm, float(Gv))
        # phi and g both enter the recursion at (event, horizon) resolution; with
        # phi_mode='market' the phi rows are constant along h and this reduces to the
        # single-phi-per-market construction.
        C = np.vstack([model_curve_phipath(Pm[i], Gm[i], H, "main", plateau_w)
                       for i in range(len(Pm))])
        # The alt normalisation is the same per-event construction; computing it here means
        # the two-panel exhibit below draws the identical objects the SSE is scored on.
        C_alt = np.vstack([model_curve_phipath(Pm[i], Gm[i], H, "alt", plateau_w)
                           for i in range(len(Pm))])
        for _nm, _C in (("main", C), ("alt", C_alt)):
            _m = np.nanmedian(_C, axis=0)
            for h in range(H + 1):
                panel_rows.append({"h": h, "value": float(_m[h]), "vintage": f"E{est}",
                                   "phi": float(Pm.mean()), "norm": _nm})
        g_row = Gm.mean(axis=1)
        expl_q = float((Pm * Gm >= 1.0).mean())
        expl_rows.append(explosive_diagnostics(est, Pm, Gm))
        med = np.nanmedian(C, axis=0)
        curves[est] = (med, np.nanpercentile(C, 25, axis=0), np.nanpercentile(C, 75, axis=0))
        sse = float(((med[hs] - emp[hs]) ** 2).sum())
        expl = float((v * g_row >= 1.0).mean())
        rows.append({"estim": est, "phi_mean": float(v.mean()),
                     "phi_p10": float(np.percentile(v, 10)),
                     "phi_p90": float(np.percentile(v, 90)),
                     "map_hit": hit, "share_explosive": expl,
                     "share_explosive_qtrs": expl_q, "sse_vs_data": sse,
                     "n_events": len(v)})
        print(f"  [routine curves] E{est}: mean phi_m={v.mean():.3f} "
              f"p10-p90 [{np.percentile(v,10):.3f},{np.percentile(v,90):.3f}] "
              f"map hit={hit:.0%} | explosive share={expl:.0%} "
              f"(quarters {expl_q:.0%}) | SSE vs data={sse:.4f}")
        for h in range(H + 1):
            rows.append({"estim": est, "h": h, "model_median": float(med[h]),
                         "model_q25": float(curves[est][1][h]),
                         "model_q75": float(curves[est][2][h]),
                         "data_median": float(emp[h]) if np.isfinite(emp[h]) else np.nan})
    if not rows:
        return None
    # The frozen-phi exhibit and the full-resolution one are different exercises, not
    # successive versions of one -- they are written side by side.
    sfx = PHI_SFX[phi_mode]
    pd.DataFrame(rows).to_csv(OUT_DIR / f"d6_routine_curves{sfx}.csv", index=False)
    # The two-panel exhibit's own input. It used to reach the figure only as a return value,
    # which meant the panel could not be redrawn without re-running the whole construction --
    # every phi_nopix parquet, every per-event recursion. Persisted, the figure is a read.
    panel_df = pd.DataFrame(panel_rows)
    panel_df.to_csv(OUT_DIR / f"d6_routine_panel{sfx}.csv", index=False)
    if expl_rows:
        pd.DataFrame(expl_rows).to_csv(OUT_DIR / f"d6_explosive_diag{sfx}.csv",
                                       index=False)
        print(f"  -> d6_explosive_diag{sfx}.csv")
    print(f"  -> d6_routine_curves{sfx}.csv / d6_routine_panel{sfx}.csv "
          f"(phi resolution: {phi_mode})")
    return rows, panel_df


def _draw_routine_curves(rows_df, H, phi_mode):
    """The standalone per-routine exhibit.

    Drawn from d6_routine_curves{sfx}.csv and nothing else: its per-horizon rows carry
    `data_median` and each routine's `model_median` / `model_q25` / `model_q75`, which is
    exactly what this panel plots. Keeping it out of routine_event_curves is what lets the
    curve construction run as a cluster job and the figure render locally from the CSV.
    """
    import matplotlib.pyplot as plt
    per_h = rows_df[rows_df["h"].notna()] if "h" in rows_df.columns else rows_df.iloc[0:0]
    if per_h.empty:
        print("  [routine curves] no per-horizon rows to draw"); return
    sfx = PHI_SFX[phi_mode]
    hgrid = np.arange(H + 1)
    # data_median is the same empirical path repeated under every routine, so one row per h.
    _e = per_h.drop_duplicates("h")
    emp = (_e.set_index(_e["h"].astype(int))["data_median"]
             .reindex(hgrid).to_numpy(dtype=float))

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.grid(color=GRID, lw=0.6)
    ax.plot(hgrid, emp, "o-", color=D_COLOR, lw=2.0, ms=4, label="data (median B path)",
            zorder=5)
    # Drawn set follows REF_ESTS, the same switch that selects Figure 4's reference lines, so
    # the two exhibits cannot end up showing different line-ups. Every routine passed in is
    # still computed and written to the CSV -- the filter is presentational.
    ests = [int(e) for e in sorted(per_h["estim"].dropna().unique())]
    shown = [e for e in ests if f"E{e}" in REF_ESTS] or ests
    # Colour and label come from the same registries Figure 4 uses, keyed by routine rather
    # than by position, so a routine keeps its identity across every exhibit.
    for est in shown:
        s = per_h[per_h["estim"] == est].sort_values("h")
        c = MODEL_COLORS.get(f"E{est}", INK)
        ax.plot(s["h"], s["model_median"], "-", color=c, lw=1.4,
                label=ROMAN.get(f"E{est}", f"E{est}"))
        ax.fill_between(s["h"], s["model_q25"], s["model_q75"], color=c, alpha=0.10)
    ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("Quarters Since Entry ($h$)")
    ax.set_ylabel("Normalized Entrant Deposits")
    # Kept short enough to fit the canvas: the previous one-line title ran past the right
    # edge and was clipped in the saved figure.
    ax.set_title("Entry Paths at Each Routine's Fitted "
                 + ("$\\phi_{m,t}$ (Market $\\times$ Quarter)"
                    if phi_mode == "path" else "$\\phi_m$ (Market Average)"),
                 fontsize=10.5)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"d6_routine_curves{sfx}.png", dpi=150)
    try:
        fig.savefig(DRAFTS / f"fig_routine_curves{sfx}.pdf", bbox_inches="tight",
                    facecolor="white")
    except OSError as e:
        print(f"  [fig] Drafts copy failed: {e}")
    plt.close(fig)
    print(f"  -> d6_routine_curves{sfx}.png")


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
    print("=== D6: entry dynamics vs the closed-form accumulation path (Egan Fig. 3) ===")
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
    # GOTCHA (hit 2026-08-03): panel_estban_instrument.py rewrites market_panel.csv in
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

    # Per-event accrual paths, aligned to reg[reg.keep] row order for each kind.
    gmode = args.g_mode
    G_b, gdiag_b = build_g_paths(reg_b, H, gmode, g)
    G_d, gdiag_d = build_g_paths(reg_d, H, gmode, g)
    if args.g_selftest and gmode != "scalar":
        G_b = np.full_like(G_b, g)
        G_d = np.full_like(G_d, g)
        print(f"  [g:selftest] every g_(e,h) forced to the scalar {g:.5f} -- results must "
              f"match --g-mode scalar")
    gB = G_b if gmode != "scalar" else g
    gD = G_d if gmode != "scalar" else g

    frames, all_paths = [], {}
    for reg, paths in ((reg_b, paths_b), (reg_d, paths_d)):
        if not reg.empty:
            frames.append(reg)
            all_paths.update(paths)
    reg_all = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    gdiag = [x for x in (gdiag_b, gdiag_d) if x is not None]
    if gdiag:
        gd = pd.concat(gdiag, ignore_index=True)
        reg_all = reg_all.merge(gd, on=["congl", "mkt", "q_entry"], how="left")
    reg_all.to_csv(OUT_DIR / "d6_entry_events.csv", index=False)

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
            c = vintage_curve(phi, gB, H, norm, plateau_w)
            curves.append(pd.DataFrame({"h": range(H + 1), "value": c,
                                        "vintage": lab, "phi": phi, "norm": norm}))
    paths_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    curves_df = pd.concat(curves, ignore_index=True)
    paths_df.to_csv(OUT_DIR / "d6_entry_paths.csv", index=False)
    curves_df.to_csv(OUT_DIR / "d6_model_curves.csv", index=False)

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
            c = vintage_curve(phi, gB, H, "main", plateau_w)
            print(f"    model[{lab}] phi={phi:.3f}: "
                  + ", ".join(f"h{h}={c[h]:.2f}" for h in range(min(9, H + 1))))

    # INVERT: the phi the entry paths themselves imply (independent of the AR moment)
    imp = {}
    print("\n  implied phi from the SHAPE of entry accumulation (one-parameter fit):")
    for kind, reg, paths, gk in (("B", reg_b, paths_b, gB), ("D", reg_d, paths_d, gD)):
        if reg.empty:
            continue
        r = implied_phi(reg, paths, H, plateau_w, gk, args.boot, args.seed, loss=args.inversion_loss)
        if r is None:
            continue
        imp[kind] = r
        print(f"    {kind}: phi_entry = {r['phi_entry']:.4f}  95% CI [{r['lo']:.4f}, "
              f"{r['hi']:.4f}]  (n={r['n_events']} events, phi*g={r['phi_g']:.4f} "
              f"at median g_e={r['g_e_median']:.5f})")
        for lab, p, _ in vint:
            inside = r["lo"] <= p <= r["hi"]
            print(f"        vs phi[{lab}]={p:.4f}: {'INSIDE' if inside else 'OUTSIDE'} the CI")
    if imp:
        pd.DataFrame(imp).T.rename_axis("kind").reset_index().to_csv(
            OUT_DIR / "d6_implied_phi.csv", index=False)
        # How much of each routine's fitted phi the B interval actually covers -- the
        # distributional companion to the level comparison printed above.
        if "B" in imp and args.routine_curves:
            print("\n  fitted phi inside the entry-implied interval:")
            phi_in_interval([int(x) for x in args.routine_curves],
                            float(imp["B"]["lo"]), float(imp["B"]["hi"]))

    # ── the compute/render seam ──────────────────────────────────────────────────────────
    # Everything above is computation and has landed in OUT_DIR as CSV. This records the few
    # scalars no CSV carries, so --figures-only is a pure read.
    _write_meta(g, H, plateau_w, vint, imp, args)

    if args.routine_curves:
        print("\n  per-routine model paths from each routine's own fitted phi_m dispersion:")
        _rc = routine_event_curves(reg_b, paths_b, H, plateau_w, gB,
                                   [int(x) for x in args.routine_curves], args.phi_mode)
        if _rc is not None and not args.compute_only:
            _rows, _panel = _rc
            _draw_routine_curves(pd.DataFrame(_rows), H, args.phi_mode)
            # Same two-panel treatment as the reference-curve exhibit, but every line is a
            # routine evaluated at ITS OWN fitted dispersion rather than one level.
            _routine_panel_figure(paths_df, _panel, g, args, imp)

    if args.compute_only:
        print("\n[compute-only] every result CSV and " + META_NAME + " written; no figure "
              "drawn. Render them with --figures-only.")
    else:
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


PANEL_TITLE = {"main": "(a) Egan Fig. 3 normalisation:  $(s_h-s_0)/(s_{end}-s_0)$",
               "alt": "(b) Alt. normalisation:  $s_h/s_{end}$"}
PANEL_STEM = {"main": "a", "alt": "b"}
DASHES = {"E2": (5, 2), "E3": (2, 1.5), "E4": (7, 2, 1.5, 2)}
# The paper enumerates the estimation strategies I-IV in the order E1, E2, E3, E4 (see the
# stage-2 comparison table's column refs). Figures carry those numerals so a reader moves
# between a figure legend and the specification tables without a translation step.
ROMAN = {"E1": "(I)", "E2": "(II)", "E3": "(III)", "E4": "(IV)"}


# Legend text for a model line's phi. Every vintage in phi_vintages() is the UNWEIGHTED mean
# of the routine's fitted phi_{m,t} over its spec-12 cells (conglomerate x type x market x
# quarter), which is not the Mean phi-hat of the stage-2 comparison table (population-weighted
# national phi_t averaged over quarters). The label names the average so the two are not read
# as the same number.
PHI_VINTAGE_LABEL = r"mean $\hat\phi_{m,t}$"


def _draw_entry_panel(ax, norm, paths_df, curves_df, vint, g, H, imp=None, title=True,
                      phi_label=PHI_VINTAGE_LABEL):
    """One normalisation's panel. Shared by the two-panel exhibit and the standalone
    per-panel figures the paper inserts use, so the two can never drift apart."""
    from matplotlib.patches import Patch
    for kind, color in (("B", B_COLOR), ("D", D_COLOR)):
        sub = paths_df[(paths_df["kind"] == kind) & (paths_df["norm"] == norm)]
        if sub.empty:
            continue
        # Bootstrap CI of the median only. The IQR describes how much entrants differ from
        # one another, which is not what this exhibit is asking -- the question is where the
        # median path sits relative to each model level, and two overlapping bands made that
        # harder to read. The narrow B band sits above the wide D band so the overlap does
        # not hide the comparison the model curves are drawn for.
        ax.fill_between(sub["h"], sub["ci_lo"], sub["ci_hi"], color=color, alpha=0.28, lw=0,
                        zorder=1.2 if kind == "B" else 1.0)
        n0 = int(sub["n"].iloc[0])
        ax.plot(sub["h"], sub["median"], color=color, lw=2, marker="o", ms=4.5,
                markeredgecolor="white", markeredgewidth=1.0, zorder=3,
                label=f"{kind}-firm entries (n={n0})")
    for lab, phi, _ in vint:
        c = curves_df[(curves_df["vintage"] == lab) & (curves_df["norm"] == norm)]
        ax.plot(c["h"], c["value"], color=MODEL_COLORS.get(lab, INK), lw=1.6,
                ls=(0, DASHES.get(lab, (5, 2))), zorder=2,
                label=f"{ROMAN.get(lab, lab)} {phi_label} = {phi:.3f}")
    # The shading is named in the legend as well as in the caption: the model curves carry
    # no band, and a reader should not have to guess which lines the shading belongs to.
    band_key = Patch(facecolor="#9E9E9E", alpha=0.45, lw=0,
                     label="95% bootstrap CI of the median")
    if norm == "alt":
        # The benchmark stays as a rule; naming it in the legend spent a row on a
        # line the caption already explains.
        ax.axhline(1.0, color="#B71C1C", lw=1.6, ls=":", zorder=4, label="_nolegend_")
        # D-firm dispersion is wide (n is small and shares are national); clip so the
        # comparison of interest -- B vs the model curves -- stays legible.
        ax.set_ylim(-0.05, 1.65)
    if title:
        ax.set_title(PANEL_TITLE[norm], loc="left", fontsize=10)
    ax.set_xlabel("Quarters Since Entry ($h$)")
    ax.set_ylabel("Normalized Market Share")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#BFBFBA")
        ax.spines[sp].set_linewidth(0.8)
    ax.set_xlim(-0.3, H + 0.3)
    # Opaque box: the curves converge into the lower-right corner at the short horizons, so
    # a transparent legend would sit on top of them.
    handles, labels = ax.get_legend_handles_labels()
    n_data = sum(1 for lb in labels if "entries" in lb)
    handles.insert(n_data, band_key)
    labels.insert(n_data, band_key.get_label())
    leg = ax.legend(handles, labels, loc="lower right", fontsize=7.5, frameon=True,
                    framealpha=1.0, facecolor="white", edgecolor="#BFBFBA", borderpad=0.6)
    leg.set_zorder(10)
    leg.get_frame().set_linewidth(0.8)


def _panel_tex(norm, vint, imp, gmode, n_b, stem="fig_entry_dynamics", per_routine=False,
               n_by=None):
    r"""A standalone \input-able float for one panel. The image carries no title or footnote
    -- the caption states the content, which is what a paper float wants.

    `n_by` maps (kind, norm) to the number of events the panel's median is taken over, so the
    caption can say when the two normalisations do not keep the same events."""
    phi_desc = (r"mean over the events of their fitted $\hat\phi_m$" if per_routine
                else r"unweighted mean of the fitted $\hat\phi_{m,t}$")
    labs = ", ".join(f"{ROMAN.get(lab, lab)} ({phi_desc} = {phi:.3f})" for lab, phi, _ in vint)
    accrual = ("each entrant's own deposit-weighted accrual path" if gmode != "scalar"
               else "the median gross accrual across the panel")
    if norm == "main":
        what = (r"Median normalised share path of the " + str(n_b) + r" conglomerate--market "
                r"entry events (markers), with a bootstrap "
                r"confidence interval for the median (shading), against the accumulation path "
                r"implied by the estimated law of motion at " + labs + r". Each model curve is "
                + (r"the median over the events' own paths, each evaluated at ITS OWN "
                   r"market-and-quarter fitted $\phi_{m,t}$ and its own accrual, so no "
                   r"dimension of the estimate is averaged away" if per_routine else
                   r"the median over the events' own paths, evaluated at " + accrual +
                   r", so the curve and the data are aggregated identically and a curve "
                   r"asserts a level of $\phi$ only")
                + r". Shares are normalised as in \textcite{egan2025dynamic} Figure~3, "
                r"$(s_h-s_0)/(s_{\mathrm{end}}-s_0)$, with $s_{\mathrm{end}}$ the average over "
                r"$h\in\{10,11,12\}$; the normalisation cancels the level of the awake inflow, "
                r"so the path depends on $\phi$ and the accrual alone.")
        if not per_routine:
            what += (r" The mean is taken over the routine's specification-(12) cells "
                     r"(conglomerate $\times$ deposit type $\times$ market $\times$ quarter); it "
                     r"is not the Mean $\hat{\phi}$ of the second-stage comparison table, which "
                     r"averages the population-weighted national $\hat{\phi}_t$ over quarters.")
        if imp and "B" in imp:
            r_ = imp["B"]
            what += (r" Inverting the comparison, the paths alone imply $\phi = "
                     f"{r_['phi_entry']:.3f}$ (95\\% CI $[{r_['lo']:.3f}, {r_['hi']:.3f}]$, "
                     r"event bootstrap).")
        cap = ("Entrant accumulation against each routine's own fitted dispersion"
               if per_routine else
               "Entrant share accumulation against the sleepiness-implied path")
        lab = stem.replace("fig_", "") + "_main"
    else:
        what = (r"The entry events under the alternative normalisation $s_h/s_{\mathrm{end}}$, "
                r"in which \emph{instant sorting} --- the persistent-preferences benchmark with no "
                r"sleepiness, under which an entrant reaches its steady-state share immediately --- "
                r"is the flat line at one. The observed median one quarter after entry is far below "
                r"it, which is the qualitative content of the test: entry is gradual. Model curves "
                r"as in the preceding figure. The vertical range is clipped for legibility; D-firm "
                r"dispersion is wide because those shares are national and few.")
        for kind in ("B", "D"):
            nm, na = (n_by or {}).get((kind, "main")), (n_by or {}).get((kind, "alt"))
            if nm is not None and na is not None and nm != na:
                what += (f" The {kind}-firm median is over {na} events here and {nm} under "
                         r"the plateau normalisation, which drops an event whose plateau "
                         r"share is within $10^{-6}$ of its entry share (a zero denominator "
                         r"for $(s_h-s_0)/(s_{\mathrm{end}}-s_0)$).")
        cap = "Entrant share accumulation against the instant-sorting benchmark"
        lab = stem.replace("fig_", "") + "_alt"
    return ("\\begin{figure}[htbp]\n"
            "  \\centering\n"
            f"  \\includegraphics[width=\\linewidth]{{{stem}_{PANEL_STEM[norm]}.pdf}}\n"
            f"  \\caption[{cap}]{{\\textbf{{{cap}.}} {what}}}\n"
            f"  \\label{{fig:{lab}}}\n"
            "\\end{figure}\n")


def make_figure(paths_df, curves_df, vint, g, args, imp=None):
    """The reference-curve exhibit: one scalar phi per routine."""
    make_panel_figures(paths_df, curves_df, vint, g, args, imp,
                       stem="fig_entry_dynamics", out_stem="d6_entry_dynamics",
                       suptitle="Entrant share accumulation: data vs the "
                                "sleepiness-implied path")


def make_panel_figures(paths_df, curves_df, vint, g, args, imp=None,
                       stem="fig_entry_dynamics", out_stem=None, suptitle="",
                       per_routine=False):
    import matplotlib.pyplot as plt
    if paths_df.empty:
        print("  [fig] no paths to plot")
        return
    H = args.horizon
    phi_label = r"event-mean $\hat\phi_m$" if per_routine else PHI_VINTAGE_LABEL
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=False)
    for ax, norm in ((axes[0], "main"), (axes[1], "alt")):
        _draw_entry_panel(ax, norm, paths_df, curves_df, vint, g, H, imp, phi_label=phi_label)
    out_stem = out_stem or stem.replace("fig_", "d6_")
    fig.suptitle(suptitle, x=0.008, ha="left", fontsize=12)
    accrual = ("each event's own accrual path" if args.g_mode != "scalar"
               else "$g$ = median gross accrual")
    fig.text(0.008, 0.005,
             "Medians across entry events; shaded = 95% bootstrap CI of the median. "
             "Model curves iterate the estimated law of motion under frozen spreads "
             f"($\\mathrm{{lvl}}_h=\\hat\\phi g_h\\mathrm{{lvl}}_{{h-1}}+1$, {accrual}), "
             "medianed over events exactly as the data is.",
             ha="left", fontsize=7.5, color="#5A5A57")
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    for _s, d in ((out_stem, OUT_DIR), (stem, DRAFTS)):
        try:
            fig.savefig(d / f"{_s}.png", dpi=300, bbox_inches="tight", facecolor="white")
            fig.savefig(d / f"{_s}.pdf", bbox_inches="tight", facecolor="white")
        except OSError as e:
            print(f"  [fig] save to {d} failed: {e}")
    plt.close(fig)

    # Standalone panels + their \input-able floats, for V_Main. Drawn from the same helper
    # as the combined exhibit, so the paper and the notes cannot show different pictures.
    n_b = int(paths_df.loc[paths_df["kind"] == "B", "n"].max()) if (paths_df["kind"] == "B").any() else 0
    # The legend's n: the events at the first horizon of each (kind, normalisation) median.
    n_by = {(k, nm): int(s["n"].iloc[0])
            for (k, nm), s in paths_df.groupby(["kind", "norm"], sort=False) if len(s)}
    for norm in ("main", "alt"):
        f1, a1 = plt.subplots(figsize=(6.4, 4.6))
        _draw_entry_panel(a1, norm, paths_df, curves_df, vint, g, H, imp, title=False,
                          phi_label=phi_label)
        f1.tight_layout()
        pstem = f"{stem}_{PANEL_STEM[norm]}"
        try:
            for ext in ("png", "pdf"):
                f1.savefig(DRAFTS / f"{pstem}.{ext}", dpi=300, bbox_inches="tight",
                           facecolor="white")
            (DRAFTS / f"{pstem}.tex").write_text(
                _panel_tex(norm, vint, imp, args.g_mode, n_b, stem, per_routine, n_by),
                encoding="utf-8")
        except OSError as e:
            print(f"  [fig] panel save failed: {e}")
        plt.close(f1)
    print(f"  figure -> {out_stem}.png (+ Drafts/{stem}.*, "
          f"per-panel {stem}_{{a,b}}.{{png,pdf,tex}})")


def _routine_panel_figure(paths_df, panel_df, g, args, imp):
    """The two-panel exhibit with every line at its routine's own fitted dispersion.

    Split out of main() so the compute run and the --figures-only run derive the vintage
    list and the file stem the same way rather than from two copies of the same expression.
    """
    if panel_df is None or panel_df.empty:
        return
    _shown = [e for e in REF_ESTS if e in set(panel_df["vintage"].astype(str))]
    _vint = [(e, float(panel_df.loc[panel_df["vintage"] == e, "phi"].iloc[0]),
              f"per-event phi ({args.phi_mode})") for e in _shown]
    # One figure family per cell of the aggregation 2x2, so the four can be compared side by
    # side instead of overwriting each other.
    _stem = f"fig_routine_dynamics{PHI_SFX[args.phi_mode]}"
    make_panel_figures(paths_df, panel_df, _vint, g, args, imp, _stem,
                       suptitle=("Entrant accumulation vs each routine's own "
                                 + PHI_LABEL[args.phi_mode]),
                       per_routine=True)


# The few things the figures need that no result CSV carries: the accrual scalar, the horizon,
# and the phi vintages with their labels and provenance. Written beside the CSVs so a
# --figures-only run is a pure read and cannot silently re-derive g from a different panel
# than the one the numbers came from.
META_NAME = "d6_meta.json"


def _write_meta(g, H, plateau_w, vint, imp, args):
    meta = {
        "g": float(g),
        "horizon": int(H),
        "plateau_w": [int(h) for h in plateau_w],
        "vintages": [[str(lab), float(p), str(src)] for lab, p, src in vint],
        "ref_ests": list(REF_ESTS),
        # settings that shaped the numbers; the render adopts these rather than its own
        "g_mode": args.g_mode,
        "phi_mode": args.phi_mode,
        "boot": int(args.boot),
        "seed": int(args.seed),
        "inversion_loss": args.inversion_loss,
        "window_only": bool(args.window_only),
        "routine_curves": list(args.routine_curves) if args.routine_curves else None,
        "implied_phi_kinds": sorted(imp.keys()) if imp else [],
    }
    (OUT_DIR / META_NAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  -> {META_NAME}")


def render_figures(args):
    """Draw every D6 exhibit from what a --compute-only run left in OUT_DIR.

    This is the local half of the split: no panel read, no bootstrap, no phi_nopix parquet,
    no refit. Everything it plots was computed on the cluster and written to
    DIAG_PHI_SEPARATION, so the figure in the paper and the numbers in the notes come from
    one run by construction.
    """
    mp = OUT_DIR / META_NAME
    if not mp.exists():
        raise SystemExit(f"{META_NAME} not found in {OUT_DIR} -- run the compute half first "
                         f"(sleep_job.sh SLEEP_STEP=entry, or --compute-only locally).")
    meta = json.loads(mp.read_text(encoding="utf-8"))
    g, H = float(meta["g"]), int(meta["horizon"])
    vint = [(str(lab), float(p), str(src)) for lab, p, src in meta["vintages"]]
    # Anything that shaped the stored numbers is taken from the computing run, not from this
    # invocation: a --figures-only call with a different --g-mode would otherwise caption the
    # figure with a construction the CSVs were not built under.
    args.horizon, args.g_mode, args.phi_mode = H, meta["g_mode"], meta["phi_mode"]
    args.boot, args.seed = int(meta["boot"]), int(meta["seed"])
    args.window_only = bool(meta["window_only"])
    print(f"=== D6 figures from {OUT_DIR} ===")
    print(f"  g={g:.5f}  horizon={H}  g-mode={args.g_mode}  phi-mode={args.phi_mode}")

    paths_df = pd.read_csv(OUT_DIR / "d6_entry_paths.csv")
    curves_df = pd.read_csv(OUT_DIR / "d6_model_curves.csv")

    imp = {}
    ip = OUT_DIR / "d6_implied_phi.csv"
    if ip.exists():
        for _, r in pd.read_csv(ip).iterrows():
            imp[str(r["kind"])] = {k: r[k] for k in r.index if k != "kind"}

    sfx = PHI_SFX[args.phi_mode]
    rc, pn = OUT_DIR / f"d6_routine_curves{sfx}.csv", OUT_DIR / f"d6_routine_panel{sfx}.csv"
    if rc.exists():
        _draw_routine_curves(pd.read_csv(rc), H, args.phi_mode)
    if pn.exists():
        _routine_panel_figure(paths_df, pd.read_csv(pn), g, args, imp)
    make_figure(paths_df, curves_df, vint, g, args, imp)
    print(f"\nfigures -> {OUT_DIR}  (+ Drafts)")
    return 0


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
    ap.add_argument("--routine-curves", nargs="+", default=None,
                    help="per-routine model paths from each routine's own fitted phi_m (D4c --routine-bands analogue), e.g. --routine-curves 1 2 3 4")
    ap.add_argument("--no-branch-screen", action="store_true",
                    help="skip the ESTBAN branch-timing screen even if the sidecar exists")
    ap.add_argument("--g-mode", choices=GMODE_CHOICES, default="path",
                    help="accrual factor: 'path' = per-event, per-quarter g (default); "
                         "'event' = one g per event; 'scalar' = the single median g")
    ap.add_argument("--phi-mode", choices=("path", "market", "time", "scalar"),
                    default="market",
                    help="fitted phi resolution in the per-routine curves: 'market' "
                         "(default) = one phi per market, the frozen-phi exhibit; 'path' = "
                         "phi at the event's own market AND quarter, which writes SEPARATE "
                         "*_phipath artifacts rather than replacing the frozen-phi ones")
    ap.add_argument("--inversion-loss", choices=("median", "pooled"), default="median",
                    help="which loss carries the headline inverted phi: 'median' "
                         "(default, Egan-comparable) or 'pooled' = every event against "
                         "its own accrual under absolute loss. BOTH are always computed "
                         "and written to d6_implied_phi.csv")
    ap.add_argument("--g-selftest", action="store_true",
                    help="force every per-event g to the scalar median; 'path' must then "
                         "reproduce '--g-mode scalar'")
    # The compute/render split. The moment and its bootstrap read the full market panel and
    # every routine's phi_nopix export and resample events 999 times, so they run on the
    # cluster (sleep_job.sh SLEEP_STEP=entry); the figures are a read of the CSVs that run
    # leaves in DIAG_PHI_SEPARATION. Neither flag changes a number: with both omitted the
    # script computes and draws in one pass exactly as before.
    g_split = ap.add_mutually_exclusive_group()
    g_split.add_argument("--compute-only", action="store_true", dest="compute_only",
                         help="compute every moment and write the result CSVs + "
                              "d6_meta.json; draw nothing (the cluster half)")
    g_split.add_argument("--figures-only", action="store_true", dest="figures_only",
                         help="draw every exhibit from the CSVs a --compute-only run left "
                              "in DIAG_PHI_SEPARATION; compute nothing (the local half)")
    _args = ap.parse_args()
    raise SystemExit(render_figures(_args) if _args.figures_only else main(_args))

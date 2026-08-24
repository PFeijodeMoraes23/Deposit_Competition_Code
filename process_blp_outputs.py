#!/usr/bin/env python3
"""
process_blp_outputs.py — organize BLP_RESULTS and build per-routine CF inputs.

Reorganizes the BLP_RESULTS folder into a clean layout and turns the raw cluster
zip (blp_outputs.zip, downloaded from Bouchet) into one consolidated artifact per
estimation routine that the counterfactual scripts can read.

Layout produced (under BLP_RESULTS/):
    logit/              local non-RC logit outputs (warm-starts, summaries, sub-model JLS)
    cluster_raw/        blp_outputs.zip + everything extracted from it (per-stage, both engines)
    cluster_processed/  ONE artifact per routine: blp_E{k}_spec_12.jls (+ .json metadata)
    legacy/             archived old/loose files that were cluttering the root (non-destructive)

For each routine k in {3,4} the processed artifact is the FINAL (extended-stage)
IFT result:
    cluster_processed/blp_E{k}_spec_12.jls   <- copy of blp_results_E{k}_spec_12_extended.jls
                                                 (Julia-serialized Dict: delta-hat, theta1, theta2, Q)
    cluster_processed/blp_E{k}_spec_12.json  <- scalar/vector summary (Q, alpha, sigma, converged,
                                                 numerical-engine cross-check Q) for inspection/Python

Idempotent and re-runnable: drop a freshly downloaded blp_outputs.zip into BLP_RESULTS
(or cluster_raw/) and run again.

Usage:
    python process_blp_outputs.py [BLP_RESULTS_dir] [--stage extended] [--routines 3,4]
"""
import os, sys, glob, json, zipfile, shutil, argparse, datetime, math

from utils import paths as _paths
from utils import routines as _routines   # aliased: `routines` is a local in main()

ROUTINE_LABEL = {e: _routines.est_label(e) for e in _routines.ACTIVE}
SUBDIRS = ["logit", "cluster_raw", "cluster_processed", "legacy"]
# Increasing-complexity RC sequence: each stage frees one more random coefficient
# (sigma = 1 σ ... extended = 8). The per-routine progression of Q across these stages is
# recorded in each processed JSON; the per-stage result .jls stay in cluster_raw/.
def _flatten_into(root):
    """Move every nested file up to ``root``, then drop the emptied directories.

    Archives written by cluster_archive.sh keep their domain prefix, so members arrive as
    ``output/blp_results_E3_spec_12_extended.jls``. Everything below addresses files as
    ``cluster_raw/<name>``, so without this a dir-prefixed archive yields
    "MISSING ... skipped" for every routine while the files sit one level down.
    """
    moved = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        if os.path.abspath(dirpath) == os.path.abspath(root):
            continue
        for name in filenames:
            src = os.path.join(dirpath, name)
            dest = os.path.join(root, name)
            if os.path.exists(dest):
                # same file by size: the flat copy already won, drop the nested duplicate
                if os.path.getsize(dest) == os.path.getsize(src):
                    os.remove(src)
                    continue
                dest = os.path.join(root, f"{os.path.basename(dirpath)}__{name}")
            shutil.move(src, dest)
            moved += 1
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        if os.path.abspath(dirpath) == os.path.abspath(root):
            continue
        if not os.listdir(dirpath):
            # Best-effort: an emptied dir under OneDrive can still be locked by the sync
            # client. The files are already where the readers look, so failing to remove
            # the husk must not abort the ingest.
            try:
                os.rmdir(dirpath)
            except OSError:
                pass
    if moved:
        print(f"[zip] flattened {moved} nested member(s) -> cluster_raw/")
    return moved


STAGE_SEQUENCE = ["sigma", "rc2", "rc3", "rc4", "full", "ext1", "ext2", "extended"]

# SUMMARY.md is written next to the paper draft (V_Main.tex) so the results writeup travels
# with the manuscript. Falls back to cluster_processed/ if that folder is unavailable.
SUMMARY_DIR = str(_paths.drafts_dir())

# Variable names — must mirror blp_1_estimation.jl (X_COLS / D_COLS) so θ₂ indices decode.
X_COLS = ["fgc_covered", "has_ip", "seg_S2", "seg_S3", "seg_S4", "seg_S5", "log_total_assets_lag", "is_state_owned"]
COEF_NAMES = ["spread"] + X_COLS                       # θ₁/random-coef characteristics (1-based)
D_COLS = ["gdp_per_capita", "fraction_65plus", "fraction_young", "pix_users_pf_per1000",
          "connections_per100", "frac_4g5g", "branches_per1000", "cadunico_families_per1000"]
THETA1_PRETTY = {"alpha": "α (spread)"}

def _name(lst, idx):
    return lst[idx - 1] if isinstance(idx, int) and 1 <= idx <= len(lst) else f"#{idx}"

def _theta2_labels(sigma_indices, pi_interactions):
    """σ first (the freed random-coefficient std devs), then the π demographic interactions —
    same order the engine packs θ₂."""
    labs = [f"σ({_name(COEF_NAMES, s)})" for s in (sigma_indices or [])]
    labs += [f"π({_name(COEF_NAMES, ci)} × {_name(D_COLS, di)})" for ci, di in (pi_interactions or [])]
    return labs

_STAGE_HEAD = {"sigma": "Sigma", "rc2": "RC2", "rc3": "RC3", "rc4": "RC4",
               "full": "Full", "ext1": "Ext1", "ext2": "Ext2", "extended": "Extended"}

def md_compare_table(k, raw_dir, logit_e):
    """Markdown version of the per-routine Logit-vs-RC-stages compare table (same content as
    Rout/blp_compare_E{k}_spec12.tex): Panel A θ₁ (coef (SE)), Panel B θ₂ (point estimates),
    footer Q / converged / N / G*. Reads the small per-stage result JSONs."""
    sdata, stages = {}, []
    for st in STAGE_SEQUENCE:
        try:
            d = json.load(open(os.path.join(raw_dir, f"blp_results_E{k}_spec_12_{st}.json")))
        except Exception:
            d = None
        if d:
            sdata[st] = d; stages.append(st)
    if not stages:
        return []
    cols = (["logit"] if logit_e else []) + stages
    head = ["Logit" if c == "logit" else _STAGE_HEAD[c] for c in cols]
    def cell(v, se):
        return "—" if v is None else (f"{v:.3f}" + (f" ({se:.3f})" if se else ""))
    def src(c): return logit_e if c == "logit" else sdata[c]
    L = [f"### E{k} — Logit vs RC-BLP stages", "",
         "| Parameter | " + " | ".join(head) + " |",
         "|:--" + "|--:" * len(cols) + "|"]
    ref = sdata[stages[-1]]
    for nm in (ref.get("param_names_theta1") or []):              # Panel A — θ₁
        row = [THETA1_PRETTY.get(nm, nm.replace("_", " "))]
        for c in cols:
            d = src(c)
            nms = d.get("param_names_theta1") or d.get("param_names") or []
            ses = d.get("theta1_se") or d.get("se") or []
            if nm in nms and nms.index(nm) < len(d.get("theta1", [])):
                i = nms.index(nm); row.append(cell(d["theta1"][i], ses[i] if i < len(ses) else None))
            else:
                row.append("—")
        L.append("| " + " | ".join(row) + " |")
    labs = []                                                     # Panel B — θ₂
    for c in stages:
        for lb in _theta2_labels(sdata[c].get("sigma_indices"), sdata[c].get("pi_interactions")):
            if lb not in labs: labs.append(lb)
    for lb in labs:
        row = [lb]
        for c in cols:
            if c == "logit":
                row.append("—"); continue
            d = sdata[c]; dl = _theta2_labels(d.get("sigma_indices"), d.get("pi_interactions"))
            t2 = d.get("theta2") or []
            row.append(f"{t2[dl.index(lb)]:.4f}" if lb in dl and dl.index(lb) < len(t2) else "—")
        L.append("| " + " | ".join(row) + " |")
    def foot(label, fn):                                          # footer
        return "| " + label + " | " + " | ".join(fn(c) for c in cols) + " |"
    L.append(foot("**Q (GMM)**", lambda c: f"{src(c).get('Q_value'):.4f}" if src(c).get('Q_value') is not None else "—"))
    L.append(foot("**converged**", lambda c: "yes" if src(c).get("converged") else "no"))
    L.append(foot("**N**", lambda c: f"{src(c).get('n_obs'):,}" if src(c).get("n_obs") else "—"))
    L.append(foot("**G\\***", lambda c: f"{src(c).get('G_star'):.2f}" if src(c).get("G_star") is not None else "—"))
    L.append("")
    return L

def compute_coef_heterogeneity(index, res_dir):
    """Per-market effective coefficient β_i = θ₁[char] + Σ_d π(char,d)·((D_dm − D̄_d)/σ_d) for
    every characteristic with a demographic interaction. Demographics are now CENTERED in the
    engine (load_precomputed_draws: D̃ = (D − D̄)/σ), so the reported θ₁ is the AVERAGE-market
    coefficient — `mean(β_i) ≈ θ₁` here is the reparametrization sanity check, and the spread of
    β_i across markets is genuine random-coefficient heterogeneity (an economic result, not the
    old centering artifact). Demographics are centered+scaled with per-market parquet moments to
    mirror the engine convention (an illustration; the exact per-market coefficients live in the
    engine's demo_draws). Reads the demand parquets; writes cluster_processed/coef_heterogeneity.json
    and returns {routine: rec}. Returns None if pandas/parquets are unavailable (graceful)."""
    try:
        import pandas as pd, numpy as np
    except Exception:
        print("[heterogeneity] pandas unavailable — skipped"); return None
    dp = os.path.join(os.path.dirname(res_dir), "DEMAND_PREP")
    out = {}
    for m in index:
        k = m["routine"]
        fs = glob.glob(os.path.join(dp, f"demand_{k}_*spec_12.parquet"))
        t1 = m.get("theta1") or m.get("theta1_alpha") or []
        if not fs or not t1:
            print(f"[heterogeneity] E{k}: missing parquet/θ₁ — skipped"); continue
        sidx, pis, t2 = m.get("sigma_indices") or [], m.get("pi_interactions") or [], m.get("theta2_sigma_pi") or []
        ns = len(sidx)
        all_demos = list(dict.fromkeys(D_COLS[di-1] for ci, di in pis if 0 < di <= len(D_COLS)))
        need = list(dict.fromkeys(["spread_ann","deposit_balance","mca_code","time_id"] + X_COLS + all_demos))
        try:
            # X_COLS enter `need` for completeness, but the β_i loop below only reads demographics; drop
            # any column the parquet lacks (e.g. is_state_owned in a pre-registration parquet) so a newly
            # added θ₁ characteristic never silently kills the entire heterogeneity block.
            import pyarrow.parquet as _pq
            avail = set(_pq.ParquetFile(fs[0]).schema.names)
            df = pd.read_parquet(fs[0], columns=[c for c in need if c in avail])
        except Exception as e:
            print(f"[heterogeneity] E{k}: parquet read failed ({e}) — skipped"); continue
        mk = df.drop_duplicates(["mca_code", "time_id"])
        sd_cache  = {n: float(mk[n].std())  for n in all_demos}    # σ_d
        avg_cache = {n: float(mk[n].mean()) for n in all_demos}    # D̄_d (subtracted → centered)
        # β_i = θ₁[char] + Σ_d π(char,d)·((D − D̄)/σ_d) for EVERY characteristic with a demographic
        # interaction. Centering matches the engine, so mean(β_i) ≈ θ₁ (the average-market value).
        coefs = {}
        for ci in sorted(set(sidx) | {ci for ci, di in pis}):
            if not (0 < ci <= len(COEF_NAMES)) or ci-1 >= len(t1):
                continue
            beta = np.full(len(df), float(t1[ci-1]), float)
            for j, (c2, di) in enumerate(pis):
                if c2 == ci and 0 < di <= len(D_COLS) and ns+j < len(t2):
                    nm = D_COLS[di-1]; sd = sd_cache.get(nm, 0.0)
                    if sd > 0:
                        beta += t2[ns+j] * ((df[nm].to_numpy(float) - avg_cache[nm]) / sd)
            entry = {"reported": float(t1[ci-1]), "mean_eff": float(np.nanmean(beta)),
                     "median_eff": float(np.nanmedian(beta))}
            if ci == 1:    # spread (α): extra distribution detail
                w = np.nan_to_num(df["deposit_balance"].to_numpy(float), nan=0.0)
                negv = beta < 0
                entry.update({"p5": float(np.nanpercentile(beta, 5)), "p95": float(np.nanpercentile(beta, 95)),
                              "frac_obs_neg": float(negv.mean()),
                              "frac_dep_neg": float(w[negv].sum()/w.sum()) if w.sum() else None})
            coefs[COEF_NAMES[ci-1]] = entry
        sp = coefs.get("spread", {})
        rec = {"alpha_bar": sp.get("reported"), "alpha_i_mean": sp.get("mean_eff"),
               "alpha_i_median": sp.get("median_eff"), "alpha_i_p5": sp.get("p5"),
               "alpha_i_p95": sp.get("p95"), "frac_obs_neg": sp.get("frac_obs_neg"),
               "frac_dep_neg": sp.get("frac_dep_neg"), "coefs": coefs}
        out[str(k)] = rec
        a = sp.get("reported"); mb = rec["alpha_i_mean"]
        print(f"[heterogeneity] E{k}: α (avg-mkt) {a:+.3f} | mean β_i {mb:+.3f} "
              f"({(rec.get('frac_dep_neg') or 0):.0%} of deposits β_i<0)"
              if (a is not None and mb is not None) else f"[heterogeneity] E{k}: computed")
    if out:
        with open(os.path.join(res_dir, "cluster_processed", "coef_heterogeneity.json"), "w") as f:
            json.dump(out, f, indent=2)
    return out

def default_results_dir():
    return str(_paths.blp_results_dir())

def move_into(path, dest_dir):
    """Move a file into dest_dir, overwriting any same-named file there."""
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, os.path.basename(path))
    if os.path.abspath(path) == os.path.abspath(dest):
        return
    if os.path.exists(dest):
        os.remove(dest)
    shutil.move(path, dest)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?", default=default_results_dir())
    ap.add_argument("--stage", default="extended", help="cluster stage to treat as final")
    ap.add_argument("--routines", default=_routines.csv(_routines.LINK_ESTS))
    args = ap.parse_args()

    RES = os.path.abspath(args.results_dir)
    if not os.path.isdir(RES):
        sys.exit(f"ERROR: BLP_RESULTS not found: {RES}")
    routines = [int(x) for x in args.routines.split(",") if x.strip()]
    sub = {s: os.path.join(RES, s) for s in SUBDIRS}
    for s in ("logit", "cluster_raw", "cluster_processed"):
        os.makedirs(sub[s], exist_ok=True)  # legacy/ created lazily, only if it archives files
    print(f"BLP_RESULTS = {RES}")
    print(f"subfolders  = {', '.join(SUBDIRS)}\n")

    # ── 1. zip -> cluster_raw/ + extract ──────────────────────────────────────
    # Accept either the legacy literal `blp_outputs.zip` or the auto-named
    # `blp_outputs_<jobid>.zip` produced by submit_blp_rc_all.sh — newest wins, no rename needed.
    def _newest_zip(d):
        zs = glob.glob(os.path.join(d, "blp_outputs*.zip"))
        return max(zs, key=os.path.getmtime) if zs else None
    zip_root = _newest_zip(RES)
    if zip_root:
        move_into(zip_root, sub["cluster_raw"])
        print(f"[zip] moved {os.path.basename(zip_root)} -> cluster_raw/")
    zip_raw = _newest_zip(sub["cluster_raw"])
    if not zip_raw:
        sys.exit(f"ERROR: no blp_outputs*.zip found in {RES} or cluster_raw/.")
    with zipfile.ZipFile(zip_raw) as z:
        members = [m for m in z.namelist() if not m.endswith("/")]
        z.extractall(sub["cluster_raw"])
    print(f"[zip] extracted {len(members)} files from {os.path.basename(zip_raw)} -> cluster_raw/")
    _flatten_into(sub["cluster_raw"])

    # ── 2. current logit_* (un-suffixed) -> logit/ ────────────────────────────
    n_logit = 0
    for p in glob.glob(os.path.join(RES, "logit_*")):
        if os.path.isfile(p) and "coherence" not in os.path.basename(p):
            move_into(p, sub["logit"]); n_logit += 1
    print(f"[logit] moved {n_logit} current logit_* files -> logit/")

    # ── 3. remaining loose root files -> legacy/ ──────────────────────────────
    n_legacy = 0
    for name in os.listdir(RES):
        p = os.path.join(RES, name)
        if os.path.isfile(p):  # only loose files; the 4 subdirs are skipped
            move_into(p, sub["legacy"]); n_legacy += 1
    print(f"[legacy] archived {n_legacy} old/loose root files -> legacy/")

    # ── 4. per-routine processed artifact (final stage, IFT) ──────────────────
    raw = sub["cluster_raw"]
    def load_summary(path):
        try:
            with open(path) as f:
                d = json.load(f)
            return d.get("12", d)  # summaries are keyed by spec id "12"
        except Exception:
            return None

    index = []
    print(f"\n[process] stage={args.stage} | routines={routines}")
    for k in routines:
        res_jls = os.path.join(raw, f"blp_results_E{k}_spec_12_{args.stage}.jls")
        if not os.path.exists(res_jls):
            print(f"  E{k}: MISSING {os.path.basename(res_jls)} — skipped"); continue
        out_jls = os.path.join(sub["cluster_processed"], f"blp_E{k}_spec_12.jls")
        shutil.copyfile(res_jls, out_jls)

        s_ift = load_summary(os.path.join(raw, f"blp_summary_E{k}_{args.stage}_gpu_ift.json")) or {}
        s_num = load_summary(os.path.join(raw, f"blp_summary_E{k}_{args.stage}_gpu_num.json")) or {}
        rj    = load_summary(os.path.join(raw, f"blp_results_E{k}_spec_12_{args.stage}.json")) or {}
        # Increasing-complexity progression: Q and θ̂₂ at each stage (1→8 free random
        # coefficients), from the per-stage IFT summaries in cluster_raw/.
        progression = []
        for st in STAGE_SEQUENCE:
            ss = load_summary(os.path.join(raw, f"blp_summary_E{k}_{st}_gpu_ift.json"))
            if ss is None:
                continue
            t2 = ss.get("theta2") or []
            progression.append({"stage": st, "n_theta2": len(t2), "Q": ss.get("Q_value"),
                                "converged": ss.get("converged"),
                                "theta1_alpha": ss.get("theta1_alpha"), "theta2": t2})
        meta = {
            "routine": k,
            "label": ROUTINE_LABEL.get(k, f"E{k}"),
            "spec": 12,
            "stage": args.stage,
            "engine": "ift",
            "converged": s_ift.get("converged"),
            "Q_ift": s_ift.get("Q_value"),
            "Q_num_crosscheck": s_num.get("Q_value"),
            "theta1_alpha": s_ift.get("theta1_alpha"),
            "theta2_sigma_pi": s_ift.get("theta2"),
            "n_theta2": len(s_ift.get("theta2") or []),
            "n_obs": rj.get("n_obs"),
            "G_star": rj.get("G_star"),
            "theta1": rj.get("theta1"),
            "theta1_se": rj.get("theta1_se"),
            "theta1_pval": rj.get("theta1_pval"),
            # θ₂ SEs exist since the on-bound-σ profiling fix in se_common.jl (gmm_cluster_ses):
            # on-bound σ's are profiled out of the SE computation and come back as 0/NaN, rendered
            # as a dagger in the summary. se_method records the scheme (wcb = wild cluster bootstrap).
            "theta2_se": rj.get("theta2_se"),
            "theta2_pval": rj.get("theta2_pval"),
            "se_method": rj.get("se_method"),
            "param_names_theta1": rj.get("param_names_theta1"),
            "sigma_indices": rj.get("sigma_indices"),
            "pi_interactions": rj.get("pi_interactions"),
            "stage_progression": progression,
            "result_jls": os.path.basename(out_jls),
            "source": f"blp_results_E{k}_spec_12_{args.stage}.jls (from blp_outputs.zip)",
            "note": ("delta-hat (mean utilities) + theta1/theta2 are inside result_jls "
                     "(Julia Serialization, Dict). This JSON holds the scalar summary."),
            "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        with open(os.path.join(sub["cluster_processed"], f"blp_E{k}_spec_12.json"), "w") as f:
            json.dump(meta, f, indent=2)
        index.append(meta)
        q = meta["Q_ift"]; qn = meta["Q_num_crosscheck"]
        dq = f" (num xcheck {qn:.5f}, Δ={abs(q-qn):.1e})" if (q is not None and qn is not None) else ""
        print(f"  E{k} [{meta['label']}]: Q={q}{dq} | converged={meta['converged']} "
              f"| {meta['n_theta2']} σ/π params -> blp_E{k}_spec_12.jls (+.json)")
        qpath = " -> ".join(f"{p['Q']:.4f}" for p in progression if p["Q"] is not None)
        print(f"       complexity Q-path ({len(progression)} stages, 1->{progression[-1]['n_theta2'] if progression else 0} RC): {qpath}")

    with open(os.path.join(sub["cluster_processed"], "INDEX.json"), "w") as f:
        json.dump({"routines": index, "stage": args.stage}, f, indent=2)
    het = compute_coef_heterogeneity(index, RES)
    write_summary_md(sub, index, args.stage, het)
    print(f"\n[done] {len(index)} routines processed -> cluster_processed/  (+ INDEX.json + SUMMARY.md)")


def write_summary_md(sub, index, stage, het=None):
    """Emit a human-readable SUMMARY.md alongside the processed artifacts (auto-regenerated
    each run, so it never goes stale). `het` = compute_coef_heterogeneity() output (or None)."""
    ts = datetime.datetime.now().isoformat(timespec="minutes")
    BOUND = 5.0  # current θ₂ box half-width (blp_2_rc.jl sets BLP_SIGMA_UB / BLP_PI_BOUND)
    # MD013 (line-length) and MD060 (table-pipe spacing) are both unfixable for the wide compact
    # tables this report uses, so disable them file-wide (the only markdownlint rules it trips).
    # Recognised by the markdownlint VS Code extension / markdownlint-cli2.
    L = ["# BLP RC-BLP — results summary", "",
         "<!-- markdownlint-disable-file MD013 MD060 -->", ""]
    L.append(f"_Generated {ts} · stage = **{stage}** · IFT engine · "
             f"θ₂ box: σ∈[0,{BOUND:g}], π∈[−{BOUND:g},{BOUND:g}]._")

    # ── what the estimation actually recovers ────────────────────────────────────────────────
    # Derived from the engine's own theta2 structure (sigma_indices / pi_interactions), never
    # hardcoded, so this section cannot drift from the specification that was estimated.
    _r0 = index[0] if index else {}
    _t1 = _r0.get("param_names_theta1") or []
    _sidx = _r0.get("sigma_indices") or []
    _pis = _r0.get("pi_interactions") or []
    if _t1 or _sidx or _pis:
        _has_sigma = set(_sidx)
        _has_pi = {ci for ci, _ in _pis}
        _flat = [c for c in range(1, len(COEF_NAMES) + 1)
                 if c not in _has_sigma and c not in _has_pi]
        L += ["", "## What the estimation recovers", "",
              "Two blocks of parameters, estimated jointly by GMM on the moment condition "
              "E[Z'ξ]=0. Individual `i` in market `m` has coefficient "
              "`β_i = θ₁ + Σ_d π·(D_dm − D̄_d)/σ_d + σ·ν_i` on each characteristic.", ""]
        L += [f"**θ₁ — mean utility ({len(_t1)} coefficients).** Linear; recovered by IV-GMM "
              "given the mean utilities δ, then re-estimated at each θ₂ trial. Demographics enter "
              "centered, so θ₁ is the **average-market** coefficient:", ""]
        L.append("| # | Parameter | Interpretation |")
        L.append("|--:|:--|:--|")
        for i, nm in enumerate(_t1, start=1):
            role = ("**price**: the deposit spread (markdown rf − dep. rate); the structural object "
                    "the counterfactuals need" if nm == "alpha" else "product characteristic")
            L.append(f"| {i} | {THETA1_PRETTY.get(nm, nm.replace('_', ' '))} | {role} |")
        L += ["", f"**θ₂ — heterogeneity ({len(_sidx)} σ + {len(_pis)} π = "
              f"{len(_sidx) + len(_pis)} parameters).** Nonlinear; these are what the outer GMM "
              "search optimises over. Everything else is profiled out analytically.", ""]
        L.append("| Parameter | What it lets vary |")
        L.append("|:--|:--|")
        for s in _sidx:
            L.append(f"| σ({_name(COEF_NAMES, s)}) | **unobserved** dispersion in the "
                     f"{_name(COEF_NAMES, s)} coefficient across individuals |")
        for ci, di in _pis:
            L.append(f"| π({_name(COEF_NAMES, ci)} × {_name(D_COLS, di)}) | **observed** variation: "
                     f"how the {_name(COEF_NAMES, ci)} coefficient shifts with market "
                     f"{_name(D_COLS, di).replace('_', ' ')} |")
        if _flat:
            L += ["", "**Not freed** (no σ and no π, so β_i = θ₁ for every market — no "
                  "heterogeneity is estimated for these): "
                  + ", ".join(f"`{_name(COEF_NAMES, c)}`" for c in _flat) + ".", ""]
        L += ["A dagger (†) on a θ₂ SE below means that coefficient was **profiled out of the "
              "covariance** as a degenerate direction (an on-bound σ, or a π whose moment-Jacobian "
              "column is numerically flat): its point estimate is still reported, but it is locally "
              "unidentified so no SE is defined for it. σ(ln assets) is absent by construction — it "
              "sat on the σ≥0 bound in every routine and was dropped from the specification.", ""]

    L += ["", "## Headline estimates", "",
          "| Routine | Model | Q (IFT) | conv. | dim θ₂ | N obs | G\\* | num Q | Δ(IFT,num) | θ₂ near bound |",
          "|---|---|---:|:---:|---:|---:|---:|---:|---:|---|"]
    for m in index:
        q, qn = m.get("Q_ift"), m.get("Q_num_crosscheck")
        dq = f"{abs(q - qn):.1e}" if (q is not None and qn is not None) else "—"
        t2 = m.get("theta2_sigma_pi") or []
        pinned = [f"θ₂[{i+1}]={v:.2f}" for i, v in enumerate(t2) if abs(abs(v) - BOUND) < 1e-2]
        nb = f"{m['n_obs']:,}" if m.get("n_obs") else "—"
        g = f"{m['G_star']:.2f}" if m.get("G_star") is not None else "—"
        # Q_num_crosscheck is absent whenever the run was IFT-only (ENGINES=ift, the default since
        # the numerical cross-check was turned off), so both Q cells need the same "—" fallback.
        qs = f"{q:.4f}" if q is not None else "—"
        qns = f"{qn:.4f}" if qn is not None else "—"
        L.append(f"| E{m['routine']} | {m['label']} | {qs} | "
                 f"{'yes' if m.get('converged') else 'no'} | {m.get('n_theta2')} | {nb} | {g} | "
                 f"{qns} | {dq} | {', '.join(pinned) if pinned else 'none'} |")
    L += ["", "## Increasing-complexity Q-path (1→8 random coefficients)", ""]
    for m in index:
        path = " → ".join(f"{p['Q']:.4f}" for p in m.get("stage_progression", []) if p.get("Q") is not None)
        L.append(f"- **E{m['routine']}** ({m['label']}): {path}")
    # ── parameter estimates (final stage): θ₁ and θ₂ across routines ──
    L += ["", "## Parameter estimates (final stage)", ""]
    ref = index[0] if index else {}
    t1_names = ref.get("param_names_theta1") or []
    if t1_names:
        L += ["**Mean utility (θ₁)** — coefficient (SE):", ""]
        L.append("| Parameter | " + " | ".join(f"E{m['routine']}" for m in index) + " |")
        L.append("|:--" + "|--:" * len(index) + "|")
        for nm in t1_names:
            cells = [THETA1_PRETTY.get(nm, nm.replace("_", " "))]
            for m in index:
                names, t1, se = (m.get("param_names_theta1") or []), (m.get("theta1") or []), (m.get("theta1_se") or [])
                if nm in names and names.index(nm) < len(t1):
                    i = names.index(nm)
                    cells.append(f"{t1[i]:.3f}" + (f" ({se[i]:.3f})" if i < len(se) else ""))
                else:
                    cells.append("—")
            L.append("| " + " | ".join(cells) + " |")
    labels = _theta2_labels(ref.get("sigma_indices"), ref.get("pi_interactions"))
    if labels:
        se_meth = next((m.get("se_method") for m in index if m.get("se_method")), None)
        has_t2se = any(m.get("theta2_se") for m in index)
        head = (f"point estimate (SE), SEs by {se_meth or 'cluster'}. A dagger (†) marks a σ "
                "**profiled out** of the SE computation because it sits on the σ ≥ 0 boundary, where "
                "∂δ/∂σ ≈ 0 leaves it locally unidentified and would otherwise inflate every SE"
                if has_t2se else "point estimates (θ₂ SEs not computed)")
        L += ["", f"**Random coefficients (θ₂)** — {head}:", ""]
        L.append("| Parameter | " + " | ".join(f"E{m['routine']}" for m in index) + " |")
        L.append("|:--" + "|--:" * len(index) + "|")
        for k, lbl in enumerate(labels):
            cells = [lbl]
            for m in index:
                t2, se2 = (m.get("theta2_sigma_pi") or []), (m.get("theta2_se") or [])
                if k >= len(t2):
                    cells.append("—")
                    continue
                cell = f"{t2[k]:.4f}"
                if k < len(se2):
                    v = se2[k]
                    bad = v is None or (isinstance(v, float) and math.isnan(v)) or v == 0
                    cell += " †" if bad else f" ({v:.4f})"
                cells.append(cell)
            L.append("| " + " | ".join(cells) + " |")

    if het:
        L += ["", "## Coefficient heterogeneity across markets", "",
              "Demographics enter μ **centered** (D̃ = (D − D̄)/σ in `load_precomputed_draws`), so the "
              "reported θ₁ above is the **average-market** coefficient — spread α is negative "
              "(downward-sloping demand; spread is the deposit markdown, rf − dep_rate). Random "
              "coefficients make the coefficient an individual market faces, "
              "`β_i = θ₁ + Σ_d π·(D_dm − D̄_d)/σ_d`, vary around that average. The table illustrates "
              "the spread distribution (parquet-based; the exact per-market coefficients live in the "
              "engine's demo_draws):", ""]
        def _f(x, p="+.3f"):
            return format(x, p) if x is not None else "—"
        L += ["| Routine | α (avg-mkt, θ₁) | mean β_i | median β_i | p5 | p95 | % deposits β_i<0 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
        for m in index:
            r = het.get(str(m["routine"]))
            if not r: continue
            fd = f"{r['frac_dep_neg']:.1%}" if r.get("frac_dep_neg") is not None else "—"
            L.append(f"| E{m['routine']} | {_f(r['alpha_bar'])} | {_f(r['alpha_i_mean'])} | "
                     f"{_f(r['alpha_i_median'])} | {_f(r['alpha_i_p5'])} | {_f(r['alpha_i_p95'])} | {fd} |")
        onames = []
        for m in index:
            for nm in ((het.get(str(m["routine"])) or {}).get("coefs") or {}):
                if nm != "spread" and nm not in onames:
                    onames.append(nm)
        if onames:
            L += ["", "**Other interacted coefficients** — average-market θ₁ → mean β_i over markets "
                  "(they should roughly coincide; the gap is parquet-proxy / sampling noise):", ""]
            L.append("| Coefficient | " + " | ".join(f"E{m['routine']}" for m in index) + " |")
            L.append("|:--" + "|:--:" * len(index) + "|")
            for nm in onames:
                cells = [nm.replace("_", " ")]
                for m in index:
                    c = ((het.get(str(m["routine"])) or {}).get("coefs") or {}).get(nm)
                    cells.append(f"{c['reported']:+.2f} → {c['mean_eff']:+.2f}" if c else "—")
                L.append("| " + " | ".join(cells) + " |")
        L += ["",
              "- **`mean β_i ≈ θ₁`** is the sanity check that demographic centering is a pure "
              "reparametrization: the average-market coefficient equals θ₁, and the dispersion "
              "(p5–p95) is real preference heterogeneity, not a centering artifact.",
              "- **`fgc_covered`** carries the largest interaction (`π(fgc×65+)`), so its β_i spreads "
              "most around θ₁; `has_ip` and the segment dummies have no interaction (β_i = θ₁)."]

    # ── weak-instruments diagnostics (from weak_iv_analysis.py → cluster_processed/weak_iv.json) ──
    try:
        wiv = json.load(open(os.path.join(sub["cluster_processed"], "weak_iv.json")))
    except Exception:
        wiv = {}
    if wiv:
        L += ["", "## Weak-instruments diagnostics", "",
              "First stage of the demand model: the deposit **spread** (the only endogenous "
              "regressor) on the 16 excluded instruments (leave-one-out rival characteristics, "
              "`n_rivals`, cost ratios, capital ratio, plus the ESTBAN branch-competition "
              "instrument `estban_rival_branches_lag` — the one instrument with within-conglomerate "
              "variation), partialling out the product controls and "
              "clustering by conglomerate. The engine instruments spread for deposit **types 4 and "
              "5** (`project_spreads`), so the battery is reported on those subsamples (and pooled). "
              "`KP-F` = cluster-robust first-stage / Kleibergen-Paap rk Wald F [@kleibergenpaap2006]; "
              "`CD-F` = Cragg-Donald (homoskedastic) [@craggdonald1993]; `eff-F` = Montiel-Olea–"
              "Pflueger effective F [@oleapflueger2013]; `LM 95%` = Kleibergen LM/K weak-IV-robust CI "
              "for α (wild-cluster-bootstrap criticals) [@kleibergen2005]; `J (p)` = Hansen overid test [@hansen1982]. Staiger–Stock "
              "rule of thumb is F ≈ 10 [@staigerstock1997].", ""]
        L += ["| Routine | sample | N | K | 2SLS α̂ (SE) | LIML α̂ | partial R² | KP-F | eff-F | LM 95% CI | tF 95% CI | J (p) |",
              "|---|:--:|---:|---:|---:|---:|---:|---:|---:|:--:|:--:|---:|"]
        def _F(x):
            return f"{x:,.0f}" if isinstance(x, (int, float)) else "—"
        def _s(x, d=3):
            return f"{x:+.{d}f}" if isinstance(x, (int, float)) and math.isfinite(x) else "—"
        def _tfstr(r):
            tf = r.get("tf") or {}
            if tf.get("tf_defined") and tf.get("tf_ci_low") is not None:
                return f"[{tf['tf_ci_low']:+.2f}, {tf['tf_ci_high']:+.2f}]"
            return "(undef, F<3.84)" if tf.get("tf_defined") is False else "—"
        for m in index:
            rr = wiv.get(str(m["routine"]))
            if not rr: continue
            for key, lbl in (("type45", "4+5"), ("type4", "4"), ("type5", "5")):
                r = rr.get(key)
                if not r: continue
                a, se = r.get("alpha_2sls"), r.get("alpha_se")
                astr = f"{a:+.3f} ({se:.3f})" if (a is not None and se is not None) else "—"
                pr = f"{r['partial_R2']:.3f}" if r.get("partial_R2") is not None else "—"
                lm = (f"[{r['lm_ci_low']:+.3f}, {r['lm_ci_high']:+.3f}]"
                      + ("" if r.get("lm_ci_bounded", True) else " (open)")
                      + (" (disc.)" if r.get("lm_ci_disconnected") else "")) \
                    if r.get("lm_ci_low") is not None else "∅"
                jp = f"{r['hansen_J_p']:.2f}" if r.get("hansen_J_p") is not None else "—"
                L.append(f"| E{m['routine']} | {lbl} | {r['n_obs']:,} | {r['n_iv']} | {astr} | "
                         f"{_s(r.get('alpha_liml'))} | {pr} | {_F(r.get('kp_first_stage_F'))} | "
                         f"{_F(r.get('effective_F'))} | {lm} | {_tfstr(r)} | {jp} |")

        # ── Estimator ladder (OLS → 2SLS → LIML → Fuller): the weak-ID signature ──
        L += ["", "### Estimator ladder (α̂)", "",
              "OLS is pulled toward the endogeneity-biased estimate; LIML is ~median-unbiased under weak "
              "identification [@andersonrubin1949] and Fuller(1) [@fuller1977] restores finite moments. "
              "A LIML sitting well away from a 2SLS that is itself pulled toward OLS is the "
              "weak-instrument signature; `DWH p` is the control-function Durbin–Wu–Hausman endogeneity "
              "test [@hausman1978] (low power under weak ID).", "",
              "| Routine | sample | OLS α̂ | 2SLS α̂ | LIML α̂ | Fuller α̂ | LIML−2SLS | 2SLS−OLS | DWH p |",
              "|---|:--:|---:|---:|---:|---:|---:|---:|---:|"]
        for m in index:
            rr = wiv.get(str(m["routine"]))
            if not rr: continue
            for key, lbl in (("type45", "4+5"), ("type4", "4"), ("type5", "5")):
                r = rr.get(key)
                if not r: continue
                a2, al, ao = r.get("alpha_2sls"), r.get("alpha_liml"), r.get("alpha_ols")
                lg = (al - a2) if (isinstance(al, (int, float)) and isinstance(a2, (int, float))) else None
                dp = f"{r['dwh_cf_p']:.2f}" if r.get("dwh_cf_p") is not None else "—"
                L.append(f"| E{m['routine']} | {lbl} | {_s(ao)} | {_s(a2)} | {_s(al)} | "
                         f"{_s(r.get('alpha_fuller'))} | {_s(lg)} | {_s(r.get('dwh_gap'))} | {dp} |")

        # ── First-stage instrument diagnostics: collinearity + leave-group-out ──
        L += ["", "### First-stage instrument diagnostics", "",
              "Collinearity of the 15 instruments (condition number and max VIF of the partialled Z) and "
              "leave-one-**group**-out sensitivity: the `mean_loo_*` block is a set of market-level rival "
              "averages that are near-collinear (the load-bearing case for the differentiation-IV "
              "construction [@gandhihoude2019]). Per-instrument _cluster-robust_ t's are rank-deficient "
              "here (K=15 > effective clusters ≈ 7) and are omitted; `neg π̂` counts negative first-stage "
              "coefficients (sign expectation for market-level generated instruments is ambiguous).", "",
              "| Routine | sample | cond # | max VIF | neg π̂ / K | eff-F full | eff-F drop mean_loo | Δα drop mean_loo |",
              "|---|:--:|---:|---:|:--:|---:|---:|---:|"]
        for m in index:
            rr = wiv.get(str(m["routine"]))
            if not rr: continue
            for key, lbl in (("type45", "4+5"), ("type4", "4"), ("type5", "5")):
                r = rr.get(key)
                if not r: continue
                zc = r.get("z_collinearity") or {}; pv = r.get("per_iv") or {}
                dm = ((r.get("loo") or {}).get("drop_group") or {}).get("mean_loo") or {}
                negk = (f"{pv['n_neg']}/{r['n_iv']}" if pv.get("n_neg") is not None else "—")
                L.append(f"| E{m['routine']} | {lbl} | {_F(zc.get('cond_number'))} | "
                         f"{_F(zc.get('max_vif'))} | {negk} | {_F(r.get('effective_F'))} | "
                         f"{_F(dm.get('eff_F'))} | {_s(dm.get('dAlpha'))} |")

        L += ["",
              "- **Clustering matters a lot here.** The homoskedastic Cragg-Donald F is in the "
              "hundreds, but the cluster-robust effective F [@oleapflueger2013] and KP rk Wald F "
              "[@kleibergenpaap2006] are an order of magnitude smaller — the leave-one-out / rival "
              "instruments are highly correlated **within conglomerate** (see the max VIF), so once SEs "
              "are clustered the spread is only weakly identified. The per-subsample pattern (type 4 vs "
              "5 vs pooled) is in the table above; the naive first-stage F badly overstated strength.",
              "- **The estimator ladder is the verdict on the price sign.** OLS → 2SLS → LIML: LIML "
              "moves α̂ further in the economically-sensible (negative) direction than 2SLS, which is the "
              "classic weak-IV signature (2SLS biased toward OLS). But the **tF honest CI** "
              "[@leemccrarymoreira2022] — which converts the effective F into an F-adjusted critical "
              "value (at eff-F ≈ 4 the multiplier is ≈ 13–14, i.e. the SE is inflated ~7×) — is "
              "near-uninformative and does not exclude zero. **So the sign of the price coefficient is "
              "not identified from the excluded instruments**; the naive ±1.96·SE Wald interval overstates "
              "precision. This is the direct evidence that the positive/insignificant logit α is a "
              "weak-instrument artifact, and that structural price sensitivity must lean on the BLP "
              "(RC + demographics), not the instruments alone.",
              "- **Weak-IV-robust inference (LM/K).** The `LM 95% CI` is the Kleibergen LM/K CI "
              "[@kleibergen2005], inverted against **wild-cluster-bootstrap** criticals (few-cluster "
              "valid; isolates α from the overid direction), which stays informative even when the "
              "**Anderson-Rubin** set [@andersonrubin1949] is empty (∅). Because min over α of AR equals "
              "the **Hansen J** [@hansen1982] and J rejects at these n, the AR set collapses to ∅; the "
              "LM/K CI does not. The optimal conditional-LR refinement [@moreira2003] is available if "
              "needed. With a single endogenous regressor the Sanderson–Windmeijer conditional F equals "
              "the first-stage F, so it adds nothing here.",
              "- **Caveats.** The DWH endogeneity test and the overid/AR sets are low-power at "
              "effective clusters ≈ 7 (descriptive, not decisive). Per-instrument cluster-robust t's are "
              "rank-deficient (K > G) and omitted. These α̂ use the engine's ln(share) δ (no market/time "
              "FE), so they are a first-stage-quality diagnostic, not the headline structural α."]

    # ── Limitations and paths forward (interpretation; the weak-α / concentration story) ──────────
    L += ["", "## Limitations and paths forward", "",
          "The price coefficient α is correctly signed (downward-sloping demand) and qualitatively "
          "robust across strategies and stages, but it is _imprecisely estimated_: the excluded "
          "instruments are weak (effective F ≈ 4, the tF honest CI spans zero) and, because deposits "
          "are highly concentrated, conglomerate-clustered inference has only G\\* ≈ 5–7 effective "
          "clusters — so cluster-robust SEs are large almost regardless of the point estimate. This "
          "is a feature of the data (weak instruments + a concentrated market), not of the modeling "
          "choices, which remain the honest ones (a finer clustering would manufacture significance "
          "by ignoring within-conglomerate price co-movement). The realistic paths, in order of "
          "weight:", "",
          "1. **Reframe the demand results around weak-IV-robust inference** — report the AR / tF / "
          "LM confidence set for α as the headline object, not the ±SE. \"Demand slopes down; the "
          "level lies in [weak-IV CI]\" is a legitimate, publishable finding. Many IO papers do "
          "exactly this when instruments are weak. This is the honest move and it is available now.",
          "2. **Carry the imprecision into the counterfactuals** — present them as ranges over the α "
          "CI, or lead with the ones that do not hinge on α's exact level. Do not report a "
          "single-number counterfactual from this α.",
          "3. **Instrument search is exhausted** — Gandhi-Houde moments, funding-cost shifters, "
          "reserve requirements (strong but macro-endogenous), and a branch-overlap-weighted rival "
          "characteristic (tested: dilutes eff-F 4.3→3.0, own partial-F≈2.6, VIF≈1,500, α "
          "wrong-signed) all fail relevance or exclusion. The blocker is structural: the spread is "
          "national (conglomerate×type×quarter) so no MCA-level instrument has matching price "
          "variation, and G\\* ≈ 5–7 caps cross-sectional power. Not a design problem.",
          "4. **State the concentration limitation explicitly** as a scope condition. It is a feature "
          "of the Brazilian deposit market, and naming it is more credible than hoping a reader does "
          "not notice G\\* ≈ 5.", ""]

    # per-routine Logit-vs-RC-stages compare tables (markdown mirror of Rout/blp_compare_*.tex)
    raw_dir = sub["cluster_raw"]
    try:
        logit_summary = json.load(open(os.path.join(sub["logit"], "logit_summary_spec_12.json")))
    except Exception:
        logit_summary = {}
    L += ["", "## Logit vs RC-BLP stages (per routine)", "",
          "Full comparison tables (same content as `Rout/blp_compare_E{k}_spec12.tex`): the non-RC "
          "logit (`full` sub-model) then each RC-BLP stage. Panel-A cells are coefficient (SE); θ₂ "
          "are point estimates (no SE). With demographics centered, θ₁ (incl. spread α) is the "
          "average-market coefficient, directly comparable to the logit α.", ""]
    for m in index:
        kk = m["routine"]
        le = next((logit_summary.get(f"E{kk}_{sm_}") for sm_ in ("full", "full_dtype", "core", "priceonly")
                   if logit_summary.get(f"E{kk}_{sm_}")), None)
        L += md_compare_table(kk, raw_dir, le or {})

    L += ["", "## Notes", "",
          "- **Stages** free one random coefficient at a time: `sigma`(1) → `rc2`(2) → `rc3`(3) → "
          "`rc4`(4) → `full`(5) → `ext1`(6) → `ext2`(7) → `extended`(8). The processed `.jls` is the "
          "final `extended` model (lower Q = better GMM fit).",
          "- **IFT** (analytical gradient) is the production engine; the **numerical** column is a "
          "cross-check only. A large Δ usually means the numerical engine did not converge (high "
          "picard-restart count), not an IFT problem.",
          "- **θ₂ box** is ±5 (widened from ±2, uniform across routines). A parameter flagged at the "
          "bound above is weakly identified.",
          "- **Artifacts:** `cluster_processed/blp_E{k}_spec_12.jls` (CF input — full δ̂/θ̂₁/θ̂₂) + "
          "`.json` (scalars + `stage_progression`); labeled parameter tables in "
          "`Rout/blp_compare_E{k}_spec12.tex`."]
    # References — pandoc-citeproc resolves the [@key] citations above against Drafts/References.bib
    # and renders the bibliography under this heading in the PDF (build_summary_pdf.py).
    L += ["", "## References"]
    L = [ln for i, ln in enumerate(L) if not (ln == "" and i and L[i-1] == "")]  # collapse blank runs (MD012/MD022)
    while L and L[-1] == "":                       # no trailing blank (MD012 on final newline)
        L.pop()
    out_dir = SUMMARY_DIR if os.path.isdir(SUMMARY_DIR) else sub["cluster_processed"]
    path = os.path.join(out_dir, "SUMMARY.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"[summary] wrote {path}")


if __name__ == "__main__":
    main()

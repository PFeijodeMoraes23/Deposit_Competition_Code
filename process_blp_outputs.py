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

For each routine k in {5,6,7,8} the processed artifact is the FINAL (extended-stage)
IFT result:
    cluster_processed/blp_E{k}_spec_12.jls   <- copy of blp_results_E{k}_spec_12_extended.jls
                                                 (Julia-serialized Dict: delta-hat, theta1, theta2, Q)
    cluster_processed/blp_E{k}_spec_12.json  <- scalar/vector summary (Q, alpha, sigma, converged,
                                                 numerical-engine cross-check Q) for inspection/Python

Idempotent and re-runnable: drop a freshly downloaded blp_outputs.zip into BLP_RESULTS
(or cluster_raw/) and run again.

Usage:
    python process_blp_outputs.py [BLP_RESULTS_dir] [--stage extended] [--routines 5,6,7,8]
"""
import os, sys, glob, json, zipfile, shutil, argparse, datetime

ROUTINE_LABEL = {
    1: "Local B-type", 2: "Pooled Linear",
    3: "Pooled Logistic", 4: "Pooled Logistic + Time",
    5: "Pooled Single-Index", 6: "Pooled Single-Index + Time",
    7: "Pooled Joint Single-Index", 8: "Pooled Joint Single-Index + Time",
}
SUBDIRS = ["logit", "cluster_raw", "cluster_processed", "legacy"]
# Increasing-complexity RC sequence: each stage frees one more random coefficient
# (sigma = 1 σ ... extended = 8). The per-routine progression of Q across these stages is
# recorded in each processed JSON; the per-stage result .jls stay in cluster_raw/.
STAGE_SEQUENCE = ["sigma", "rc2", "rc3", "rc4", "full", "ext1", "ext2", "extended"]

# SUMMARY.md is written next to the paper draft (V_Main.tex) so the results writeup travels
# with the manuscript. Falls back to cluster_processed/ if that folder is unavailable.
SUMMARY_DIR = r"C:\Users\pedro\OneDrive\Documentos\Yale\Year 3 (2024 - 2025)\Open Finance\Open-Finance\Drafts\Deposit Competition"

# Variable names — must mirror blp_1_estimation.jl (X_COLS / D_COLS) so θ₂ indices decode.
X_COLS = ["fgc_covered", "has_ip", "seg_S2", "seg_S3", "seg_S4", "seg_S5", "log_total_assets_lag"]
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

def default_results_dir():
    repo = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(repo))  # .../Open-Finance
    return os.path.join(root, "BCB", "Egan_et_al_2025_Rep", "processed",
                        "ESTIMATION_OUTPUT", "BLP_RESULTS")

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
    ap.add_argument("--routines", default="5,6,7,8")
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
    zip_root = os.path.join(RES, "blp_outputs.zip")
    zip_raw  = os.path.join(sub["cluster_raw"], "blp_outputs.zip")
    if os.path.exists(zip_root):
        move_into(zip_root, sub["cluster_raw"])
        print("[zip] moved blp_outputs.zip -> cluster_raw/")
    if not os.path.exists(zip_raw):
        sys.exit(f"ERROR: blp_outputs.zip not found in {RES} or cluster_raw/.")
    with zipfile.ZipFile(zip_raw) as z:
        members = [m for m in z.namelist() if not m.endswith("/")]
        z.extractall(sub["cluster_raw"])
    print(f"[zip] extracted {len(members)} files -> cluster_raw/")

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
    write_summary_md(sub, index, args.stage)
    print(f"\n[done] {len(index)} routines processed -> cluster_processed/  (+ INDEX.json + SUMMARY.md)")


def write_summary_md(sub, index, stage):
    """Emit a human-readable SUMMARY.md alongside the processed artifacts (auto-regenerated
    each run, so it never goes stale)."""
    ts = datetime.datetime.now().isoformat(timespec="minutes")
    BOUND = 5.0  # current θ₂ box half-width (blp_2_rc.jl sets BLP_SIGMA_UB / BLP_PI_BOUND)
    L = ["# BLP RC-BLP — results summary", ""]
    L.append(f"_Generated {ts} · stage = **{stage}** · IFT engine · "
             f"θ₂ box: σ∈[0,{BOUND:g}], π∈[−{BOUND:g},{BOUND:g}]._")
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
        L.append(f"| E{m['routine']} | {m['label']} | {q:.4f} | "
                 f"{'yes' if m.get('converged') else 'no'} | {m.get('n_theta2')} | {nb} | {g} | "
                 f"{qn:.4f} | {dq} | {', '.join(pinned) if pinned else 'none'} |")
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
        L += ["", "**Random coefficients (θ₂)** — point estimates (θ₂ SEs not computed):", ""]
        L.append("| Parameter | " + " | ".join(f"E{m['routine']}" for m in index) + " |")
        L.append("|:--" + "|--:" * len(index) + "|")
        for k, lbl in enumerate(labels):
            cells = [lbl] + [f"{(m.get('theta2_sigma_pi') or [])[k]:.4f}"
                             if k < len(m.get('theta2_sigma_pi') or []) else "—" for m in index]
            L.append("| " + " | ".join(cells) + " |")

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
    out_dir = SUMMARY_DIR if os.path.isdir(SUMMARY_DIR) else sub["cluster_processed"]
    path = os.path.join(out_dir, "SUMMARY.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"[summary] wrote {path}")


if __name__ == "__main__":
    main()

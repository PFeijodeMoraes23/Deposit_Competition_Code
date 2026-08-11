#!/usr/bin/env python
"""
diag_phi_pix_gradient.py — D6c: is each routine's structural Pix effect inside the reduced-form
bound?

WHY THIS EXISTS. D6/D6b are routine-INDEPENDENT: both re-estimate the carry with E2's kernel from
`load_sleep_frame()` and never read any routine's fitted φ, so `--estim` cannot make them
routine-specific (running them per routine yields identical files — same trap as D3/D5). The
routine-specific object is the MODEL-IMPLIED Pix effect: CF_FOUNDATION/phi_nopix_E{N}_spec_12
materialises φ_mt and φ_mt^noPix through each routine's OWN link (cf_4_upsilon_export.py), so

    Δφ^Pix_it = φ_mt(i,t) − φ^noPix_mt(i,t)

is that routine's structural claim about what Pix did to the carry, row by row. D6b bounds the
same object in reduced form: the post × carry × connectivity triple says any Pix break in the
carry bigger than MDE(80%) per SD of exposure would have been detected. Comparing the two closes
the loop:

    |implied gradient| > MDE   →  the reduced form would have SEEN this routine's Pix effect,
                                  and it did not — tension with that routine's Pix loading;
    |implied gradient| ≤ MDE   →  the routine's Pix channel is consistent with the data's
                                  silence — unfalsified, but also unvalidated (CF4 caveat).

UNITS. φ IS the carry coefficient (Dep = φ·g·Dep_{t−1} + A), so a Δ in φ_mt is unit-identical to
the carry-coefficient shift D6b estimates. Two-way FE absorb the national level, so the reduced
form identifies only the GRADIENT in exposure — the implied national mean break is reported as
descriptive, never tested.

EXPOSURE is rebuilt exactly as in arm_pixpooled: the entity's pre-2020 mean of
connections_per100, standardised over rows (not entities), so "per SD" means the same thing in
both columns of the comparison.

Outputs (OUT_DIR = .../DIAG_PHI_SEPARATION):
    d6c_pix_gradient.csv          long results
    tab_d6c_pix_gradient.md/.tex  comparison table (.tex also copied next to V_Main.tex)
    fig_d6c_pix_gradient.png      implied gradients vs the D6b band

Usage:
    python diag_phi_pix_gradient.py                  # all of SLEEP_ACTIVE_ESTS
    python diag_phi_pix_gradient.py --ests 5 6 7 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
import utils.paths as _paths  # noqa: E402
from diag_phi_augmented_tests import OUT_DIR, load_sleep_frame  # noqa: E402

CF_DIR = _paths.PROCESSED / "ESTIMATION_OUTPUT" / "CF_FOUNDATION"
DRAFTS = Path(r"c:\Users\pedro\OneDrive\Documentos\Yale\Year 3 (2024 - 2025)\Open Finance"
              r"\Open-Finance\Drafts\Deposit Competition")

LABEL = {1: "E1 identity", 2: "E2 pooled linear", 5: "E5 single-index",
         6: "E6 single-index + time", 7: "E7 joint sieve", 8: "E8 joint sieve + time"}
LAUNCH_QIDX = 2020 * 4 + 3          # 2020Q4, matching arm_pix / arm_pixpooled

# The two samples D6b reports, keyed by how the parquet rows are filtered. spec12 keeps the CF
# sample's deposit types; OLSxTech keeps all four.
SAMPLES = [("spec12(k=4,5)", lambda d: d["deposit_type"].isin([4, 5])),
           ("OLSxTech(all k)", lambda d: pd.Series(True, index=d.index))]


def qidx_of(time_id: pd.Series) -> pd.Series:
    """'2016Q2' → 2016*4+1, matching the sleep frame's qidx = year*4 + (quarter-1)."""
    s = time_id.astype(str)
    return s.str[:4].astype(int) * 4 + s.str[-1].astype(int) - 1


def load_d6b_reference() -> pd.DataFrame:
    """The postZ_x rows of the pooled variant: coefficient, WCB MDE and the randomization MDE.
    Read from disk rather than hardcoded so a D6b re-run propagates here automatically."""
    f = OUT_DIR / "d6b_pix_pooled.csv"
    if not f.exists():
        raise SystemExit(f"missing {f} — run: python diag_phi_augmented_tests.py --arm pixpooled")
    d = pd.read_csv(f)
    d = d[(d["param"] == "postZ_x") & (d["variant"] == "pooled")]
    out = []
    for spec, g in d.groupby("spec"):
        wcb = g[g["coef"].notna()].iloc[0]
        perm = g[g.get("perm_mde80", pd.Series(dtype=float)).notna()]
        out.append({"spec": spec, "coef": float(wcb["coef"]), "p_wcb": float(wcb["p_wcb"]),
                    "mde80_wcb": float(wcb["mde80"]),
                    "mde80_perm": float(perm["perm_mde80"].mean()) if len(perm) else np.nan})
    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ests", nargs="+", type=int,
                    default=[int(x) for x in
                             os.environ.get("SLEEP_ACTIVE_ESTS", "1 2 5 6 7 8").split()])
    a = ap.parse_args()

    ref = load_d6b_reference()
    print("D6b reference (from d6b_pix_pooled.csv):")
    print(ref.to_string(index=False))

    # ---- exposure, exactly as arm_pixpooled builds it --------------------------------------
    print("\nbuilding exposure from the sleep frame (pre-2020 entity mean connectivity) ...")
    df, _ = load_sleep_frame()
    pre = df[df["year"] < 2020].groupby("entity_id")["connections_per100"].mean()
    e = df["entity_id"].map(pre)
    df["_expo"] = ((e - e.mean()) / e.std(ddof=0)).fillna(0.0)
    expo = df[["entity_id", "time_id", "_expo"]].drop_duplicates(["entity_id", "time_id"])
    print(f"  {len(expo):,} (entity, quarter) exposure rows; "
          f"{pre.notna().sum():,} entities with a pre-2020 mean")

    rows = []
    for est in a.ests:
        fp = CF_DIR / f"phi_nopix_E{est}_spec_12.parquet"
        if not fp.exists():
            print(f"  [E{est}] MISSING {fp.name} — run cf_4_upsilon_export.py --estim {est}")
            continue
        p = pd.read_parquet(fp)
        ups = json.loads((CF_DIR / f"upsilon_pix_E{est}_spec_12.json").read_text())
        p["dphi"] = p["phi_mt"] - p["phi_mt_nopix"]
        p["qidx"] = qidx_of(p["time_id"])

        m = p.merge(expo, on=["entity_id", "time_id"], how="left", validate="m:1")
        hit = float(m["_expo"].notna().mean())
        # Below 90% the gradient is computed on a nonrandom subset (the MCA-crosswalk trap made
        # exactly this failure quiet once) — refuse to report rather than report on it.
        if hit < 0.90:
            print(f"  [E{est}] exposure merge hit-rate {hit:.1%} < 90% — SKIPPING; "
                  f"check entity_id key compatibility / MCA coverage")
            continue
        post = m[(m["qidx"] >= LAUNCH_QIDX) & m["_expo"].notna()]

        for sname, filt in SAMPLES:
            s = post[filt(post)]
            x = s["_expo"].to_numpy(float)
            y = s["dphi"].to_numpy(float)
            vx = float(np.var(x))
            grad = float(np.cov(x, y, bias=True)[0, 1] / vx) if vx > 0 else np.nan
            w = s["dphi"].abs()  # descriptive spread of the implied effect
            r = ref[ref["spec"] == sname].iloc[0]
            rows.append({
                "estim": est, "label": LABEL.get(est, f"E{est}"), "spec": sname,
                "upsilon_pix": float(ups["upsilon_pix"]), "link": ups["link"],
                "n_post": len(s), "expo_hit": hit,
                "implied_gradient": grad,
                "implied_mean_break": float(y.mean()),
                "dphi_p10": float(np.percentile(y, 10)),
                "dphi_p90": float(np.percentile(y, 90)),
                "d6b_coef": r["coef"], "d6b_mde_wcb": r["mde80_wcb"],
                "d6b_mde_perm": r["mde80_perm"],
                "inside_bound": bool(abs(grad) <= r["mde80_wcb"]) if np.isfinite(grad) else None,
                "sign_matches_d6b": bool(np.sign(grad) == np.sign(r["coef"]))
                if np.isfinite(grad) and grad != 0 else None,
            })
            print(f"  [E{est} | {sname}] implied gradient {grad:+.5f}/SD, mean break "
                  f"{y.mean():+.5f}, dphi p10..p90 [{np.percentile(y,10):+.4f},"
                  f"{np.percentile(y,90):+.4f}]  vs D6b {r['coef']:+.4f} (MDE {r['mde80_wcb']:.4f})"
                  f"  n={len(s):,}")

    if not rows:
        print("nothing computed"); return 1
    T = pd.DataFrame(rows)
    T.to_csv(OUT_DIR / "d6c_pix_gradient.csv", index=False)

    # ---- table ---------------------------------------------------------------------------
    md = []
    for sname, _ in SAMPLES:
        S = T[T["spec"] == sname]
        if not len(S):
            continue
        r = ref[ref["spec"] == sname].iloc[0]
        md.append(f"**{sname}** — D6b reduced form: {r['coef']:+.4f} per SD "
                  f"(WCB p={r['p_wcb']:.2f}), MDE(80%) {r['mde80_wcb']:.4f} WCB / "
                  f"{r['mde80_perm']:.4f} randomization\n")
        md.append("| routine | link | Υ_pix (AME) | implied gradient /SD | implied mean break | "
                  "Δφ p10…p90 | inside MDE? | sign as D6b? |")
        md.append("|---|---|---|---|---|---|---|---|")
        for _, q in S.sort_values("estim").iterrows():
            md.append(f"| {q['label']} | {q['link']} | {q['upsilon_pix']:+.4f} | "
                      f"{q['implied_gradient']:+.5f} | {q['implied_mean_break']:+.5f} | "
                      f"[{q['dphi_p10']:+.4f}, {q['dphi_p90']:+.4f}] | "
                      f"{'yes' if q['inside_bound'] else '**NO**'} | "
                      f"{'yes' if q['sign_matches_d6b'] else 'no'} |")
        md.append("")
    (OUT_DIR / "tab_d6c_pix_gradient.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    tex = [r"\begin{tabular}{llccccc}", r"\toprule",
           r"Routine & Link & $\Upsilon_{pix}$ & Implied grad./SD & Mean break & "
           r"Inside MDE & Sign \\", r"\midrule"]
    for sname, _ in SAMPLES:
        S = T[T["spec"] == sname]
        if not len(S):
            continue
        r = ref[ref["spec"] == sname].iloc[0]
        tex.append(rf"\multicolumn{{7}}{{l}}{{\textit{{{sname}}}: D6b ${r['coef']:+.4f}$/SD "
                   rf"($p={r['p_wcb']:.2f}$), MDE ${r['mde80_wcb']:.4f}$}} \\")
        for _, q in S.sort_values("estim").iterrows():
            tex.append(f"\\quad {q['label']} & {q['link']} & {q['upsilon_pix']:+.4f} & "
                       f"{q['implied_gradient']:+.5f} & {q['implied_mean_break']:+.5f} & "
                       f"{'yes' if q['inside_bound'] else 'NO'} & "
                       f"{'$-$' if q['sign_matches_d6b'] else '$+$'} \\\\")
        tex.append(r"\addlinespace")
    tex += [r"\bottomrule", r"\end{tabular}"]
    txt = "\n".join(tex) + "\n"
    (OUT_DIR / "tab_d6c_pix_gradient.tex").write_text(txt, encoding="utf-8")
    (DRAFTS / "tab_d6c_pix_gradient.tex").write_text(txt, encoding="utf-8")

    # ---- figure --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), sharey=True)
    for ax, (sname, _) in zip(axes, SAMPLES):
        S = T[T["spec"] == sname].sort_values("estim").reset_index(drop=True)
        r = ref[ref["spec"] == sname].iloc[0]
        yy = np.arange(len(S))
        ax.axvspan(-r["mde80_wcb"], r["mde80_wcb"], color="0.88", zorder=0,
                   label="D6b MDE(80%) band")
        ax.axvline(r["coef"], color="#a63603", lw=1.6, ls="--", zorder=2,
                   label=f"D6b point ({r['coef']:+.3f})")
        ax.axvline(0, color="0.4", lw=0.8, zorder=1)
        ax.scatter(S["implied_gradient"], yy, s=70, color="#1f4e79", zorder=3,
                   edgecolor="white", linewidth=0.9, label="routine-implied gradient")
        for yi, (_, q) in zip(yy, S.iterrows()):
            ax.plot([q["dphi_p10"], q["dphi_p90"]], [yi, yi], color="#1f4e79",
                    lw=1.2, alpha=0.45, zorder=2)
        ax.set_yticks(yy); ax.set_yticklabels(S["label"], fontsize=9)
        ax.set_title(sname, fontsize=10.5, weight="bold")
        ax.set_xlabel("carry shift per SD of connectivity / Δφ range", fontsize=9)
        ax.grid(axis="x", alpha=0.25)
    # Shared y-axis: invert ONCE — inverting per-axes flips it back on the second call.
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=8, loc="lower left", framealpha=0.92)
    fig.suptitle("D6c — each routine's model-implied Pix effect vs the D6b reduced-form bound "
                 "(thin bars: p10–p90 of Δφ$^{Pix}$, descriptive)", fontsize=10.5, y=1.04)
    fig.tight_layout()
    fp = OUT_DIR / "fig_d6c_pix_gradient.png"
    fig.savefig(fp, dpi=170, bbox_inches="tight")
    plt.close(fig)

    print(f"\n  -> {OUT_DIR / 'd6c_pix_gradient.csv'}")
    print(f"  -> {OUT_DIR / 'tab_d6c_pix_gradient.md'} (+ .tex here and next to V_Main.tex)")
    print(f"  -> {fp}")
    print("\n" + "\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())

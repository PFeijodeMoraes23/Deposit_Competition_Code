"""
diag_entry_dynamics_estban.py
=============================
READ-ONLY diagnostic.  Changes no panel and no paper exhibit; writes only to
ESTIMATION_OUTPUT/DIAG_ENTRY_ESTBAN.

QUESTION.  Figure 2 of V_Main (fig_entry_dynamics_a/b, sleep_ident_entry_dynamics.py) builds
the B-firm entry paths from market_panel.  Built straight from the raw ESTBAN monthly files,
do the same paths give the same picture and the same entry-implied phi?

WHAT IS THE SAME.  The share is the entrant's ESTBAN deposits (V401-V419 demand + V420
savings + V432 time) over the ESTBAN deposits of every institution in the MCA, quarter-end
months.  It is the share conditional on banking with a branch-network institution, as in
Figure 2.  The sample rule is Figure 2's: entries dated inside the estimation window
(2016Q1-2024Q4), paths read through its last quarter, earlier raw files (from 2013Q1) used only
to date the entry.  Screens, normalisations, bootstrap and inversion are imported from
sleep_ident_entry_dynamics.py and run unchanged.

WHAT DIFFERS FROM FIGURE 2.
  firm           the ESTBAN institution (8-digit CNPJ), not the prudential conglomerate.
  denominator    every ESTBAN reporter in the MCA, including the institutions the panel books
                 as national (D) and so removes from local markets.
  branch screen  first quarter with AGEN_PROCESSADAS > 0, from the same raw files, from 2013Q1.
                 Figure 2 reads estban_own_branches.csv, which starts in 2016Q1 and so covers
                 every entry inside the window.
  accrual        one scalar g, Figure 2's median gross_return_lag (d6_meta.json).  The moment
                 identifies phi*g; phi is phi*g divided by that g.

A control row, "panel_fixed_g", rebuilds Figure 2's own events from the market panel with its
own loader (checked against the published d6_entry_paths.csv) and fits them at the same fixed g.

Outputs (DIAG_ENTRY_ESTBAN): estban_entry_events_window.csv, estban_entry_paths.csv,
estban_implied_phi.csv, estban_continuity.csv, estban_nationally_booked.csv, estban_meta.json,
fig_entry_estban_window.{png,pdf}.  diagnostics/report_entry_paths_estban.py turns them into
the report next to V_Main.tex.

Usage:  python diagnostics/diag_entry_dynamics_estban.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: utils/ and pipeline modules
try:
    from utils.venv_guard import ensure_project_venv
    ensure_project_venv(__file__)
except Exception:
    pass

try:                                          # Windows console defaults to cp1252
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import json
import os
import re
from types import SimpleNamespace

os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
import pandas as pd

from utils import paths
import sleep_ident_entry_dynamics as d6

OUT = paths.PROCESSED / "ESTIMATION_OUTPUT" / "DIAG_ENTRY_ESTBAN"
OUT.mkdir(parents=True, exist_ok=True)
D6_DIR = d6.OUT_DIR

Q0, Q1 = 2013 * 4, d6.WINDOW_Q1                 # first raw quarter .. the window's last
DEP_PREFIX = ("VERBETE_401", "VERBETE_420", "VERBETE_432")
KEEP_PREFIX = ("#DATA_BASE", "CNPJ", "CODMUN_IBGE", "AGEN_PROCESSADAS") + DEP_PREFIX


def read_month(fp: Path) -> pd.DataFrame:
    """One raw monthly file -> CNPJ, CODMUN_IBGE, AGEN_PROCESSADAS, dep.

    Columns are matched by prefix: the demand-deposit column is one compound header
    (VERBETE_401 + ... + VERBETE_419), and the long names differ slightly across vintages."""
    df = pd.read_csv(fp, sep=";", decimal=",", encoding="latin1", skiprows=2, dtype=str,
                     usecols=lambda c: c.strip().upper().startswith(KEEP_PREFIX))
    df.columns = [c.strip().upper() for c in df.columns]
    cols = {p: [c for c in df.columns if c.startswith(p)] for p in DEP_PREFIX}
    for p, found in cols.items():
        if len(found) != 1:
            raise SystemExit(f"{fp.name}: expected one column starting {p}, found {found}")
    num = lambda s: pd.to_numeric(s.str.strip(), errors="coerce").fillna(0.0)
    out = pd.DataFrame({
        "CNPJ": df["CNPJ"].str.strip().str.zfill(8),
        "CODMUN_IBGE": pd.to_numeric(df["CODMUN_IBGE"], errors="coerce"),
        "agen": num(df["AGEN_PROCESSADAS"]),
        "dep": sum(num(df[cols[p][0]]) for p in DEP_PREFIX),
    })
    return out


def load_estban() -> pd.DataFrame:
    """Every quarter-end month of the raw folder, each file read once.

    os.listdir, not glob("*.CSV") + glob("*.csv"): Windows globbing is case-insensitive, so
    that pair returns every file twice."""
    rows = []
    for name in sorted(os.listdir(paths.ESTBAN_RAW_MUN)):
        m = re.match(r"^(\d{4})(\d{2})_ESTBAN\.csv$", name, re.IGNORECASE)
        if not m:
            continue
        y, mo = int(m.group(1)), int(m.group(2))
        if mo not in (3, 6, 9, 12):
            continue
        q = y * 4 + mo // 3 - 1
        if not Q0 <= q <= Q1:
            continue
        f = read_month(Path(paths.ESTBAN_RAW_MUN) / name)
        f["qidx"] = q
        rows.append(f)
    e = pd.concat(rows, ignore_index=True)
    n_q = e["qidx"].nunique()
    if n_q != Q1 - Q0 + 1:
        raise SystemExit(f"raw ESTBAN covers {n_q} quarter-end months, expected {Q1 - Q0 + 1}")
    return e


def mca_crosswalk() -> pd.Series:
    """Municipality -> MCA, read from the market panel's ESTBAN rows (the one geography input
    that is not ESTBAN itself)."""
    mp = pd.read_parquet(paths.PROCESSED / "market_panel.parquet",
                         columns=["CODMUN_IBGE", "mca_code", "Source"])
    x = mp[(mp["Source"] == "ESTBAN") & ~mp["CODMUN_IBGE"].isin(["0", "0.0"])]
    x = x[["CODMUN_IBGE", "mca_code"]].dropna().drop_duplicates()
    x["CODMUN_IBGE"] = pd.to_numeric(x["CODMUN_IBGE"], errors="coerce")
    if x["CODMUN_IBGE"].duplicated().any():
        raise SystemExit("municipality maps to more than one MCA in market_panel")
    x = x.dropna(subset=["CODMUN_IBGE"])
    return x.set_index(x["CODMUN_IBGE"].astype("int64"))["mca_code"].astype(str)


def continuity_check(e: pd.DataFrame) -> pd.DataFrame:
    """National ESTBAN deposits around 2016Q1, raw vs market panel.  The raw series should be
    continuous; the panel's pre-2016 quarters are on twice the scale.  Raw rows without a
    municipality code are shown separately because the panel drops them."""
    mp = pd.read_parquet(paths.PROCESSED / "market_panel.parquet",
                         columns=["year", "quarter", "Source", "dep_a1", "dep_a2", "dep_a4"])
    mp = mp[mp["Source"] == "ESTBAN"]
    mp["qidx"] = mp["year"] * 4 + mp["quarter"] - 1
    mp["dep"] = mp[["dep_a1", "dep_a2", "dep_a4"]].fillna(0).sum(axis=1)
    win = range(2015 * 4 + 1, 2016 * 4 + 2)
    out = pd.DataFrame({
        "raw_all": e.groupby("qidx")["dep"].sum(),
        "raw_with_mun": e.dropna(subset=["CODMUN_IBGE"]).groupby("qidx")["dep"].sum(),
        "panel": mp.groupby("qidx")["dep"].sum(),
    }).reindex(win)
    out["panel_over_raw_with_mun"] = out["panel"] / out["raw_with_mun"]
    out.index.name = "qidx"
    out = out.reset_index()
    out["quarter"] = out["qidx"].map(lambda q: f"{q // 4}Q{q % 4 + 1}")
    print("\n  national ESTBAN deposits (R$ bn), raw files vs market panel:")
    for _, r in out.iterrows():
        print(f"    {r['quarter']}  raw {r['raw_all'] / 1e9:9.1f}  raw with municipality "
              f"{r['raw_with_mun'] / 1e9:9.1f}   panel {r['panel'] / 1e9:9.1f}"
              f"   panel/raw {r['panel_over_raw_with_mun']:.4f}")
    return out


def nationally_booked() -> pd.DataFrame:
    """Conglomerates whose ESTBAN rows the market panel moves to national (D) booking, and so
    out of every local denominator.  Raw ESTBAN keeps them in their municipalities."""
    mp = pd.read_parquet(paths.PROCESSED / "market_panel.parquet",
                         columns=["CodConglomeradoPrudencial", "NomeInstituicao", "CODMUN_IBGE",
                                  "year", "Source"])
    x = mp[(mp["Source"] == "ESTBAN") & mp["CODMUN_IBGE"].isin(["0", "0.0"])]
    return (x.groupby(["CodConglomeradoPrudencial", "NomeInstituicao"], as_index=False)
            .agg(quarters=("year", "size"), first_year=("year", "min"),
                 last_year=("year", "max"))
            .sort_values("NomeInstituicao"))


def waterfall(reg: pd.DataFrame, label: str) -> None:
    flags = [c for c in reg.columns if c.startswith("f_")]
    print(f"\n  [{label}] candidates={len(reg):,}  kept={int(reg['keep'].sum()):,}")
    alive = pd.Series(True, index=reg.index)
    for f in flags:
        drop = int((alive & reg[f]).sum())
        alive &= ~reg[f]
        print(f"      {f:<14s} -{drop:>6,}  remaining {int(alive.sum()):>6,}")
    kept = reg[reg["keep"]]
    if len(kept):
        ch = (kept["q_entry"] // 4).value_counts().sort_index()
        print("      kept cohorts: " + ", ".join(f"{int(y)}:{n}" for y, n in ch.items()))


PANEL_TITLE = {"main": r"(a) $(s_h - s_0)/(s_{\mathrm{end}} - s_0)$",
               "alt": r"(b) $s_h/s_{\mathrm{end}}$"}


def draw(variant, paths_df, fig2, phi, g, H, plateau_w):
    """Two panels at print width (6.5 in), titled by normalization; the caption of the
    document that includes the figure names the sample."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.9))
    for ax, norm in zip(axes, ("main", "alt")):
        sub = paths_df[paths_df["norm"] == norm]
        ref = fig2[fig2["norm"] == norm]
        ax.fill_between(sub["h"], sub["ci_lo"], sub["ci_hi"], color=d6.B_COLOR, alpha=0.25, lw=0,
                        label="95% CI of median")
        ax.plot(sub["h"], sub["median"], color=d6.B_COLOR, lw=1.8, marker="o", ms=4,
                markeredgecolor="white", markeredgewidth=0.8, zorder=3,
                label=f"Raw ESTBAN (n = {int(sub['n'].iloc[0])})")
        ax.plot(ref["h"], ref["median"], color=d6.INK, lw=1.0, ls="--", marker="o", ms=4,
                markerfacecolor="white", markeredgecolor=d6.INK, zorder=4,
                label=f"Paper, B firms (n = {int(ref['n'].iloc[0])})")
        fit = d6.model_curve(phi, g, H, norm, plateau_w)
        ax.plot(range(H + 1), fit, color="#212121", lw=1.2, zorder=2,
                label=f"Model, $\\phi$ = {phi:.3f}")
        if norm == "alt":
            ax.axhline(1.0, color=d6.INK, lw=1.2, ls=":", label="No sleepiness")
        ax.set_title(PANEL_TITLE[norm], loc="left", fontsize=9)
        ax.set_xlabel("Quarters Since Entry ($h$)", fontsize=8.5)
        ax.set_ylabel("Normalized Market Share", fontsize=8.5)
        ax.tick_params(axis="both", labelsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis="y", color=d6.GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_xlim(-0.3, H + 0.3)
        ax.legend(loc="lower right", fontsize=6.8, frameon=True, framealpha=0.9,
                  handlelength=1.6, borderpad=0.3, labelspacing=0.25)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_entry_estban_{variant}.{ext}", dpi=200, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)


def main():
    meta = json.loads((D6_DIR / "d6_meta.json").read_text(encoding="utf-8"))
    g, H, plateau_w = float(meta["g"]), int(meta["horizon"]), list(meta["plateau_w"])
    boot, seed = int(meta["boot"]), int(meta["seed"])
    print(f"=== entry paths from raw ESTBAN: g={g:.5f} (Figure 2's median), H={H}, "
          f"plateau {plateau_w}, boot={boot}, seed={seed} ===")

    e = load_estban()
    continuity_check(e).to_csv(OUT / "estban_continuity.csv", index=False)
    nationally_booked().to_csv(OUT / "estban_nationally_booked.csv", index=False)
    cw = mca_crosswalk()
    n_raw = len(e)
    no_mun = e["CODMUN_IBGE"].isna()
    e = e[~no_mun].copy()
    e["CODMUN_IBGE"] = e["CODMUN_IBGE"].astype("int64")
    e["mkt"] = e["CODMUN_IBGE"].map(cw)
    lost = e["mkt"].isna()
    meta_out = {"n_files": int(e["qidx"].nunique()), "q_first": Q0, "q_last": Q1,
                "n_rows": int(n_raw), "n_rows_no_municipality": int(no_mun.sum()),
                "n_rows_no_mca": int(lost.sum()),
                "share_deposits_no_mca": float(e.loc[lost, "dep"].sum() / e["dep"].sum()),
                "n_mca": int(e.loc[~lost, "mkt"].nunique()),
                "n_institutions": int(e.loc[~lost, "CNPJ"].nunique()),
                "g": g, "window_q": [d6.WINDOW_Q0, d6.WINDOW_Q1]}
    print(f"\n  raw rows {n_raw:,}; without a municipality {int(no_mun.sum()):,}; without an MCA "
          f"{int(lost.sum()):,} ({100 * meta_out['share_deposits_no_mca']:.3f}% of deposits)")
    (OUT / "estban_meta.json").write_text(json.dumps(meta_out, indent=2), encoding="utf-8")
    e = e[~lost]

    df = (e.groupby(["CNPJ", "mkt", "qidx"], as_index=False)[["dep", "agen"]].sum())
    branch = (df[df["agen"] > 0].groupby(["CNPJ", "mkt"])["qidx"].min()
              .rename("q_branch"))
    branch.index.names = ["congl", "mkt"]
    df = df[df["dep"] > 0].rename(columns={"CNPJ": "congl"})
    df["cnpj_root"] = df["congl"]

    fig2 = pd.read_csv(D6_DIR / "d6_entry_paths.csv")
    fig2 = fig2[fig2["kind"] == "B"]
    fig2_phi = pd.read_csv(D6_DIR / "d6_implied_phi.csv").set_index("kind").loc["B"]

    rows_paths, rows_phi = [], []
    for variant in ("window",):
        args = SimpleNamespace(horizon=H, min_post=8, max_jump=0.15, max_entry_share=0.10,
                               window_only=True, window_q0=d6.WINDOW_Q0)
        reg, ev_paths = d6.build_events(df, "B", args, branch)
        waterfall(reg, variant)
        reg.to_csv(OUT / f"estban_entry_events_{variant}.csv", index=False)

        frames = []
        for norm in ("main", "alt"):
            p = d6.event_paths(reg, ev_paths, H, norm, plateau_w, boot, seed)
            p["norm"], p["variant"] = norm, variant
            frames.append(p)
        pdf = pd.concat(frames, ignore_index=True)
        rows_paths.append(pdf)

        r = d6.implied_phi(reg, ev_paths, H, plateau_w, g, boot, seed, loss="median")
        r["variant"] = variant
        rows_phi.append(r)
        print(f"    implied phi = {r['phi_entry']:.4f}  95% CI [{r['lo']:.4f}, {r['hi']:.4f}]"
              f"  phi*g = {r['phi_g']:.4f}  (n = {r['n_events']}; pooled-abs "
              f"{r['phi_pooled_abs']:.4f})")
        draw(variant, pdf, fig2, r["phi_entry"], g, H, plateau_w)

    # Figure 2's own B events, rebuilt from the market panel by its own loader, fitted at the
    # same fixed g as the raw sample: the difference from the published phi is the accrual path
    # alone, and the difference from the raw sample is the data source alone.
    args = SimpleNamespace(horizon=H, min_post=8, max_jump=0.15, max_entry_share=0.10,
                           window_only=bool(meta["window_only"]), no_branch_screen=False)
    b, _ = d6.load_event_frames(args)
    reg_p, paths_p = d6.build_events(b, "B", args, d6.load_branch_entry(args))
    chk = d6.event_paths(reg_p, paths_p, H, "main", plateau_w, boot, seed)
    pub = fig2[fig2["norm"] == "main"].set_index("h")[["median", "ci_lo", "ci_hi"]]
    dev = float((chk.set_index("h")[["median", "ci_lo", "ci_hi"]] - pub).abs().max().max())
    print(f"\n  [panel_fixed_g] kept={int(reg_p['keep'].sum())}; rebuilt median/CI vs "
          f"published d6_entry_paths.csv: max |diff| = {dev:.2e}")
    r = d6.implied_phi(reg_p, paths_p, H, plateau_w, g, boot, seed, loss="median")
    r["variant"] = "panel_fixed_g"
    r["max_abs_diff_vs_published_path"] = dev
    rows_phi.append(r)
    print(f"    implied phi = {r['phi_entry']:.4f}  95% CI [{r['lo']:.4f}, {r['hi']:.4f}]"
          f"  (n = {r['n_events']}; published, per-entrant g: {fig2_phi['phi_entry']:.4f})")

    pd.concat(rows_paths, ignore_index=True).to_csv(OUT / "estban_entry_paths.csv", index=False)
    pd.DataFrame(rows_phi).to_csv(OUT / "estban_implied_phi.csv", index=False)

    print(f"\n  Figure 2 (market panel, path g): phi = {fig2_phi['phi_entry']:.4f} "
          f"[{fig2_phi['lo']:.4f}, {fig2_phi['hi']:.4f}], n = {int(fig2_phi['n_events'])}")
    print("\n  median path, plateau normalisation:")
    print("    h      " + " ".join(f"{h:>6d}" for h in range(H + 1)))
    f2 = fig2[fig2["norm"] == "main"].set_index("h")["median"]
    print("    Fig. 2 " + " ".join(f"{f2[h]:6.3f}" for h in range(H + 1)))
    for pdf in rows_paths:
        s = pdf[pdf["norm"] == "main"].set_index("h")["median"]
        print(f"    {pdf['variant'].iloc[0][:7]:<7s}" + " ".join(f"{s[h]:6.3f}" for h in range(H + 1)))
    print(f"\nresults -> {OUT}")


if __name__ == "__main__":
    main()

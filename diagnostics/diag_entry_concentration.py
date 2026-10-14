"""
diag_entry_concentration.py -- READ-ONLY check of Figure 2's entry-implied sleeping share against
the concentration of the B entry events in a few banks.

Under the window rule the kept B entries come from 17 conglomerates, and Santander (C0080185)
makes 122 of the 201. Figure 2's interval resamples ENTRIES as if independent. This script
rebuilds the B events, share paths and per-event accrual exactly as sleep_ident_entry_dynamics
does (its own functions, the paper's settings from d6_meta.json), reproduces the paper's B row of
d6_implied_phi.csv, and then reports:

  1. bank bootstrap  the same fit, resampling conglomerates with all their entries;
  2. no Santander    the fit without Santander's entries, with both intervals;
  3. one vote/bank   weighted medians (each entry weighted 1 / its bank's number of entries) on
                     the data side and on the model side, bank-bootstrap interval;
plus the fit on Santander's entries alone, for context.

Writes DIAG_PHI_SEPARATION/diag_entry_concentration.{csv,json}. Touches no other file.
Run: C:/venvs/egan/Scripts/python.exe diagnostics/diag_entry_concentration.py
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
from types import SimpleNamespace

import numpy as np
import pandas as pd

import sleep_ident_entry_dynamics as sed

SANTANDER = "C0080185"
N_DRAWS = 400            # implied_phi caps its bootstrap at 400 draws; the same count here
GRID = np.linspace(0.50, 0.9985, 1400)   # implied_phi's grid


def wmedian(x: np.ndarray, w: np.ndarray, axis: int) -> np.ndarray:
    """Weighted median along `axis`: the smallest value whose cumulative weight reaches half the
    total, averaged with the next value when the half is reached exactly (so equal weights give
    np.median). NaNs carry no weight."""
    x = np.asarray(x, float)
    w = np.broadcast_to(np.asarray(w, float), x.shape)
    x, w = np.moveaxis(x, axis, -1), np.moveaxis(w, axis, -1)
    w = np.where(np.isfinite(x), w, 0.0)
    order = np.argsort(np.where(np.isfinite(x), x, np.inf), axis=-1)
    xs = np.take_along_axis(x, order, axis=-1)
    cw = np.cumsum(np.take_along_axis(w, order, axis=-1), axis=-1)
    tot = cw[..., -1:]
    half = 0.5 * tot
    k = np.argmax(cw >= half * (1 - 1e-12), axis=-1)[..., None]
    out = np.take_along_axis(xs, k, axis=-1)[..., 0]
    exact = np.abs(np.take_along_axis(cw, k, axis=-1) - half)[..., 0] <= 1e-12 * tot[..., 0]
    k2 = np.minimum(k + 1, x.shape[-1] - 1)
    nxt = np.take_along_axis(xs, k2, axis=-1)[..., 0]
    out = np.where(exact & np.isfinite(nxt), 0.5 * (out + nxt), out)
    return np.where(tot[..., 0] > 0, out, np.nan)


def main():
    meta = json.loads((sed.OUT_DIR / "d6_meta.json").read_text(encoding="utf-8"))
    H, plateau_w = int(meta["horizon"]), list(meta["plateau_w"])
    args = SimpleNamespace(horizon=H, min_post=8, boot=int(meta["boot"]), seed=int(meta["seed"]),
                           max_jump=0.15, max_entry_share=0.10,
                           window_only=bool(meta.get("window_q")), no_branch_screen=False,
                           g_mode=meta["g_mode"], inversion_loss=meta["inversion_loss"])
    print(f"settings from d6_meta.json: H={H}, plateau {plateau_w}, g_mode {args.g_mode}, "
          f"boot {args.boot}, seed {args.seed}, window {args.window_only}")

    g = sed.median_g()
    b, _ = sed.load_event_frames(args)
    branch = sed.load_branch_entry(args)
    reg, paths = sed.build_events(b, "B", args, branch)
    G, _ = sed.build_g_paths(reg, H, args.g_mode, g)

    # Fidelity: the paper's own function on the rebuilt inputs must give the stored B row.
    paper = sed.implied_phi(reg, paths, H, plateau_w, G, args.boot, args.seed,
                            loss=args.inversion_loss)
    stored = pd.read_csv(sed.OUT_DIR / "d6_implied_phi.csv").set_index("kind").loc["B"]
    gap = max(abs(paper[k] - float(stored[k])) for k in ("phi_entry", "lo", "hi"))
    if not gap < 1e-12:
        sys.exit(f"rebuilt B fit differs from d6_implied_phi.csv by {gap:.3e}; stop")
    print(f"paper B row reproduced: {paper['phi_entry']:.4f} [{paper['lo']:.4f}, "
          f"{paper['hi']:.4f}], n = {paper['n_events']} (max |diff| {gap:.1e})")

    M, rows = sed._path_matrix(reg, paths, H, "main", plateau_w, want_rows=True)
    ev = reg[reg["keep"]].iloc[rows].reset_index(drop=True)
    banks = ev["congl"].to_numpy()
    T = sed.model_tensor(GRID, np.asarray(G, float)[rows], H, plateau_w)      # (grid, E, H+1)
    hs = [h for h in range(1, H) if h not in plateau_w]
    n_per_bank = pd.Series(banks).map(pd.Series(banks).value_counts()).to_numpy(float)

    def fit(idx, weighted=False):
        """implied_phi's median-loss fit on the entries idx; weighted: one vote per bank."""
        if weighted:
            w = 1.0 / n_per_bank[idx]
            med = wmedian(M[idx], w[:, None], axis=0)
            mod = wmedian(T[:, idx, :][:, :, hs], w[None, :, None], axis=1)
        else:
            med = np.nanmedian(M[idx], axis=0)
            mod = np.median(T[:, idx, :][:, :, hs], axis=1)
        ok = [i for i, h in enumerate(hs) if np.isfinite(med[h])]
        dat = med[hs][ok]
        return float(GRID[int(np.argmin(((mod[:, ok] - dat) ** 2).sum(axis=1)))])

    all_idx = np.arange(len(ev))
    p0 = fit(all_idx)
    if not abs(p0 - paper["phi_median_loss"]) < 1e-12:
        sys.exit(f"local fit {p0} differs from implied_phi's {paper['phi_median_loss']}; stop")

    def event_boot(idx, seed=0):
        rng = np.random.default_rng(seed)
        return [fit(idx[rng.integers(0, len(idx), len(idx))]) for _ in range(N_DRAWS)]

    def bank_boot(idx, weighted=False, seed=0):
        rng = np.random.default_rng(seed)
        ids = np.unique(banks[idx])
        rows_of = {c: idx[banks[idx] == c] for c in ids}
        out = []
        for _ in range(N_DRAWS):
            pick = rng.choice(ids, len(ids), replace=True)
            out.append(fit(np.concatenate([rows_of[c] for c in pick]), weighted))
        return out

    def ci(draws):
        return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))

    no_s = np.flatnonzero(banks != SANTANDER)
    only_s = np.flatnonzero(banks == SANTANDER)
    res = []

    def row(spec, phi, lo, hi, interval, idx):
        res.append(dict(spec=spec, phi=phi, lo=lo, hi=hi, interval=interval,
                        n_events=int(len(idx)), n_banks=int(len(np.unique(banks[idx])))))
        print(f"  {spec:<34s} {phi:.4f} [{lo:.4f}, {hi:.4f}]  {interval:<16s} "
              f"n={len(idx)}, banks={len(np.unique(banks[idx]))}")

    print("\nresults (median loss, per-event accrual, 400 draws):")
    row("paper (all entries)", paper["phi_entry"], paper["lo"], paper["hi"], "events", all_idx)
    row("all entries", p0, *ci(bank_boot(all_idx)), "banks", all_idx)
    p_ns = fit(no_s)
    row("without Santander", p_ns, *ci(event_boot(no_s)), "events", no_s)
    row("without Santander", p_ns, *ci(bank_boot(no_s)), "banks", no_s)
    p_w = fit(all_idx, weighted=True)
    row("one vote per bank", p_w, *ci(bank_boot(all_idx, weighted=True)), "banks", all_idx)
    p_s = fit(only_s)
    row("Santander only", p_s, *ci(event_boot(only_s)), "events", only_s)

    counts = pd.Series(banks).value_counts()
    out = dict(generator="diagnostics/diag_entry_concentration.py",
               settings=dict(horizon=H, plateau_w=plateau_w, g_mode=args.g_mode, draws=N_DRAWS,
                             seed=0, loss="median", window_only=args.window_only),
               plotted_events=int(len(ev)), banks=int(counts.size),
               santander_events=int(counts.get(SANTANDER, 0)),
               events_by_bank={k: int(v) for k, v in counts.items()}, results=res)
    pd.DataFrame(res).to_csv(sed.OUT_DIR / "diag_entry_concentration.csv", index=False)
    (sed.OUT_DIR / "diag_entry_concentration.json").write_text(json.dumps(out, indent=2) + "\n",
                                                               encoding="utf-8")
    print(f"\nwrote {sed.OUT_DIR / 'diag_entry_concentration.csv'} and .json")


if __name__ == "__main__":
    main()

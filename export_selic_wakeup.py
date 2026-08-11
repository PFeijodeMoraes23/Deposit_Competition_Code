"""export_selic_wakeup.py -- D7: the Selic wake-up comovement row (Egan alignment).

Author: Pedro Feijo de Moraes

Egan et al. (2025) validate their sleepiness estimates with the comovement of inertia
and the policy rate ("people wake up when interest rates increase": their Table 3,
Upsilon_{1,R^F} < 0). Our state vector carries the same object: `risk_free_qoq_lag`
(lagged Selic, quarterly decimal, grand-mean centred). This export pulls that loading
from every estimator's saved spec-12 fit and reports it with the HONEST inference for a
national regressor: the conglomerate WCB is anti-conservative here (Selic is constant
within a quarter), so the reported cell is quarter-clustered WCB / Driscoll-Kraay via
utils.se_national.select_se -- the same dagger convention as the paper tables.

Units: E1/E2 are linear, so the coefficient IS d(phi)/d(Selic) per unit of the raw
regressor. E3-E8 report the INDEX loading (sign-interpretable; magnitude runs through
the link); the AME row is included when the saved fit carries one.

Outputs: DIAG_PHI_SEPARATION/d7_selic_comovement.csv + tab_selic_wakeup.tex (Rout +
copied to Drafts). No estimation -- reads saved pickles only.
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import gc
import pickle
import shutil

import numpy as np
import pandas as pd

from utils import paths as _paths
from utils import se_national as _sen
import utils.sleep_links  # noqa: F401  (class defs needed to unpickle nonlinear fits)

OUT_DIR = _paths.PROCESSED / "ESTIMATION_OUTPUT" / "DIAG_PHI_SEPARATION"
TEX_OUT = _paths.PROCESSED / "ESTIMATION_OUTPUT" / "Rout"
DRAFTS = _paths.OPEN_FINANCE / "Drafts" / "Deposit Competition"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TEX_OUT.mkdir(parents=True, exist_ok=True)

SPEC = "IV_HausmanFull x Tech"      # spec 12
VAR = "interaction_risk_free_qoq_lag"
KIND = {1: "linear", 2: "linear", 3: "logit", 4: "logit+time",
        5: "single-index", 6: "single-index+time", 7: "joint sieve", 8: "joint sieve+time"}


def _series_get(obj, attr, key):
    s = getattr(obj, attr, None)
    if s is None:
        return None
    try:
        v = float(pd.Series(s)[key])
        return v if np.isfinite(v) else None
    except Exception:
        return None


def one_row(est):
    pkl = _paths.PROCESSED / "ESTIMATION_OUTPUT" / "DEMAND_PREP" / f"est{est}" / "estimation_results.pkl"
    if not pkl.exists():
        return {"est": f"E{est}", "kind": KIND[est], "status": "no pickle"}
    with open(pkl, "rb") as fh:
        d = pickle.load(fh)
    entry = d.get(SPEC)
    if entry is None:
        return {"est": f"E{est}", "kind": KIND[est], "status": f"spec 12 absent ({list(d)[:3]}...)"}
    res = entry.get("second_stage") if isinstance(entry, dict) else entry
    coef = _series_get(res, "params", VAR)
    if coef is None:
        coef = _series_get(res, "params_native", VAR)
    if coef is None:
        return {"est": f"E{est}", "kind": KIND[est], "status": "Selic row absent"}
    se_c = _series_get(res, "bse", VAR)
    se_hon, p_hon, scheme = _sen.select_se(res, VAR)
    row = {"est": f"E{est}", "kind": KIND[est], "status": "ok",
           "coef": coef, "se_congl": se_c,
           "se_honest": se_hon, "p_honest": p_hon, "scheme": scheme,
           "se_dk": _sen.dk_se(res, VAR),
           "ame": _series_get(res, "ames", "risk_free_qoq_lag"),
           "pkl_mtime": pd.Timestamp(pkl.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M")}
    del d, entry, res
    gc.collect()
    return row


def main():
    print("=== D7: Selic wake-up comovement (spec 12, all estimators) ===")
    rows = [one_row(e) for e in range(1, 9)]
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "d7_selic_comovement.csv", index=False)
    ok = df[df["status"] == "ok"]
    for _, r in df.iterrows():
        if r["status"] != "ok":
            print(f"  {r['est']:>3s} ({r['kind']}): {r['status']}")
            continue
        star = "" if not np.isfinite(r.get("p_honest", np.nan)) else (
            "***" if r["p_honest"] < 0.01 else "**" if r["p_honest"] < 0.05
            else "*" if r["p_honest"] < 0.10 else "")
        print(f"  {r['est']:>3s} ({r['kind']:<17s}) coef={r['coef']:+.4f}{star}  "
              f"[{r['scheme']}] se={r['se_honest']:.4f} p={r['p_honest']:.3f}  "
              f"(congl se={r['se_congl'] if r['se_congl'] is not None else float('nan'):.4f})"
              + (f"  AME={r['ame']:+.4f}" if r.get("ame") is not None and np.isfinite(r["ame"] or np.nan) else "")
              + f"  [pkl {r['pkl_mtime']}]")

    # minimal LaTeX fragment (same dagger convention as the paper tables)
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"Estimator & Coef.\ on lagged Selic$^\dagger$ & SE & Scheme & $p$ \\",
             r"\midrule"]
    for _, r in ok.iterrows():
        lines.append(f"  {r['est']} ({r['kind']}) & {r['coef']:+.4f} & {r['se_honest']:.4f} & "
                     f"{r['scheme']} & {r['p_honest']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    frag = "\n".join(lines) + "\n"
    (TEX_OUT / "tab_selic_wakeup.tex").write_text(frag, encoding="utf-8")
    try:
        shutil.copy(TEX_OUT / "tab_selic_wakeup.tex", DRAFTS / "tab_selic_wakeup.tex")
    except OSError as e:
        print(f"  [warn] Drafts copy failed: {e}")

    if len(ok):
        n_neg = int((ok["coef"] < 0).sum())
        sig = ok[ok["p_honest"] < 0.05]
        print(f"\n  VERDICT inputs: {n_neg}/{len(ok)} estimators have the Egan wake-up sign")
        print("  (NEGATIVE loading: higher Selic -> lower index -> lower phi, since the sieve/")
        print("  logit link G is monotone increasing; for E1/E2 it is phi directly);")
        print(f"  {len(sig)}/{len(ok)} significant at 5% under time-robust (quarter/DK) inference.")
        if len(sig):
            print("  significant rows: " + ", ".join(
                f"{r['est']} ({r['coef']:+.3f})" for _, r in sig.iterrows()))
        # E7/E8 caveat: diag_index_identification.py found E7 spec 12 at a corner loading
        # 0.988 of a unit-norm theta on THIS regressor, so its Selic row is the artifact the
        # direction diagnostic warned about, not independent evidence.
        if "E7" in set(sig["est"]) or "E8" in set(sig["est"]):
            print("  NOTE: E7/E8 are the joint-sieve fits whose index DIRECTION is weakly")
            print("  identified (diag_index_identification.py: E7 spec 12 sits at a corner with")
            print("  ~0.99 of unit-norm theta on risk_free_qoq_lag). A significant Selic loading")
            print("  there reflects that corner; do not read it as wake-up evidence either way.")
        print("  Sign agreement with weak time-robust significance = qualitative corroboration;")
        print("  do NOT lean on magnitudes (T=35 quarters of identifying variation).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

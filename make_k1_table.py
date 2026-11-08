"""
make_k1_table.py
================
The net-flow equation on the observed net flows (K1, kappa/k1_netflow_nls.py) as the paper's table:
the switching cost kappa-tilde by source pair, its Pix change (the headline), its level (an upper
bound on kappa) and its dose, with the 95 percent intervals of K1's score wild cluster bootstrap.

  Panel A  the headline: statement lambda; the change and the level from the before_after fit, the
           dose from the dose fit; savings (the headline pair), checking (with matured time
           deposits), time deposits (daily liquidity; not estimable when K1 says so); the matured
           exit coefficient theta_w as a check
  Panel B  lambda = 1 for every bank (K1's --lambda-one run): the three pairs and ln E_c
  Panel C  the savings pair on K1's robustness samples (carried_kept, zero_premium, no_negative_adv)

Inputs: k1_estimates.csv and k1_numbers.json of K1's run folders, by default Rout/k1 (the
invocation with before_after, dose and the robustness samples) and Rout/k1_lambda_one (the
--lambda-one invocation). K1 rewrites k1_estimates.csv at every invocation, so each invocation needs
its own folder.

Guard: every run folder must record a real-data fit on the final panel (guard.final_inputs, the
context, frame and K0 panels and inputs.panel_md5 on FINAL_PANEL; no --allow-interim, no market
subset, no --test; every spec of the CSV fitted by the recorded run). Otherwise the script prints the
reason and writes nothing. --allow-smoke renders a smoke run, and only into an --out-dir outside the
paper folder. make_paper_numbers.py (collect_k1) reads the runs through load_runs, the same guard.

Outputs (paper folder)
  tab_k1_netflow.tex       tabular only (booktabs); the paper wraps it
  tab_k1_netflow_note.tex  the table's note, \\footnotesize
  tab_k1_netflow.md        the same table in markdown
Keys: \\pn{kappa.k1.*} (make_paper_numbers.py, collect_k1).

Usage:  python make_k1_table.py
        python make_k1_table.py --run <folder> --run <folder> --allow-smoke --out-dir <scratch folder>
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from utils import paths as _paths

FINAL_PANEL = "fa38794d"
RUN_DIRS = ("k1", "k1_lambda_one")            # under Rout
ROBUSTNESS = (("carried_kept", "Carried-rate rows kept"),
              ("zero_premium", "Zero-premium quarters only"),
              ("no_negative_adv", "Negative-advertising quarters dropped"))
PAIRS = (("sav", "", "Savings"), ("chk", "_chk", "Checking"), ("td", "_td", "Time deposits"))
TEX = "tab_k1_netflow.tex"
NOTE = "tab_k1_netflow_note.tex"
MD = "tab_k1_netflow.md"
MINUS = "$-$"


class K1Refused(Exception):
    """A K1 run that is not a real-data fit on the final panel, or an inconsistent set of runs."""


def default_dirs() -> list:
    return [_paths.rout_dir() / d for d in RUN_DIRS]


def check_run(d: Path) -> list:
    """Reasons the run in folder d is not a real-data fit on FINAL_PANEL (empty when it is)."""
    est, num = d / "k1_estimates.csv", d / "k1_numbers.json"
    if not (est.is_file() and num.is_file()):
        return [f"k1_estimates.csv or k1_numbers.json missing in {d}"]
    n = json.loads(num.read_text(encoding="utf-8"))
    g, a = n.get("guard") or {}, n.get("args") or {}
    out = []
    if not g.get("final_inputs"):
        out.append("guard.final_inputs is not true")
    for k in ("context_panel", "frame_panel", "k0_numbers_panel"):
        if not str(g.get(k, "")).startswith(FINAL_PANEL):
            out.append(f"guard.{k} = {g.get(k)}, not {FINAL_PANEL}")
    if not str((n.get("inputs") or {}).get("panel_md5", "")).startswith(FINAL_PANEL):
        out.append(f"inputs.panel_md5 = {(n.get('inputs') or {}).get('panel_md5')}, not {FINAL_PANEL}")
    if a.get("allow_interim"):
        out.append("run with --allow-interim (smoke test)")
    if a.get("markets"):
        out.append(f"run on {a.get('markets')} markets (smoke test)")
    if a.get("test"):
        out.append(f"a --test {a.get('test')} run")
    fitted = set(str(a.get("spec") or "").split(","))
    extra = sorted(set(pd.read_csv(est)["spec"].astype(str)) - fitted)
    if extra:
        out.append(f"k1_estimates.csv holds specs {extra} the recorded run did not fit")
    return out


def load_runs(dirs=None, allow_smoke=False) -> dict:
    """{(spec, lambda_one, exit_by_cell): row} over the run folders, each row with _run, _draws,
    _panel, _smoke and _folder. Raises K1Refused with the reason unless every folder passes the guard
    (allow_smoke: render a smoke run anyway, marked _smoke)."""
    rows = {}
    for d in [Path(x) for x in (dirs or default_dirs())]:
        reasons = check_run(d)
        if reasons and (not allow_smoke or not (d / "k1_estimates.csv").is_file()):
            raise K1Refused(f"{d}: " + "; ".join(reasons))
        n = json.loads((d / "k1_numbers.json").read_text(encoding="utf-8"))
        a = n.get("args") or {}
        key2 = (bool(a.get("lambda_one")), bool(a.get("exit_by_cell")))
        for r in pd.read_csv(d / "k1_estimates.csv").to_dict("records"):
            key = (str(r["spec"]),) + key2
            if key in rows:
                raise K1Refused(f"spec {key} appears in two run folders ({rows[key]['_folder']}, {d})")
            rows[key] = dict(r, _run=n.get("run"), _draws=a.get("draws"),
                             _panel=str((n.get("guard") or {}).get("context_panel")),
                             _smoke=bool(reasons), _reasons=reasons, _folder=str(d))
    if ("before_after", False, False) not in rows:
        raise K1Refused("no headline row (before_after, statement lambda) in " + ", ".join(map(str, dirs or default_dirs())))
    return rows


def pick(rows, spec, lambda_one=False):
    return rows.get((spec, lambda_one, False))


def est(r, p):
    """(value, lo, hi) of parameter p in row r, or None when the row or the column is absent."""
    if r is None or p not in r or pd.isna(r[p]):
        return None
    return float(r[p]), float(r[f"ci_lo_{p}"]), float(r[f"ci_hi_{p}"])


# ------------------------------------------------------------------------------- rendering
def num(x) -> str:
    """Three decimals, thousands separators, a math minus, never -0.000."""
    s = f"{float(x):,.3f}"
    if s.startswith("-"):
        s = s[1:] if float(s.replace(",", "")) == 0.0 else MINUS + s[1:]
    return s.replace(",", "{,}")


def cell(e):
    return ("", "") if e is None else (num(e[0]), f"[{num(e[1])}, {num(e[2])}]")


def dagger(r) -> str:
    return "" if r is None or bool(r.get("converged")) else r"$^{\dagger}$"


def build(rows) -> dict:
    head, dose, l1 = pick(rows, "before_after"), pick(rows, "dose"), pick(rows, "before_after", True)
    td_note = str(head.get("kappa_td_note")) if pd.notna(head.get("kappa_td_note")) else None
    A = []
    for code, suf, label in PAIRS:
        if code == "td" and est(head, "kappa_td") is None:
            A.append((label, None, "not estimable"))
            continue
        A.append((label + (" (headline)" if code == "sav" else ""),
                  [est(head, f"dkappa{suf}"), est(head, f"kappa{suf}"), est(dose, f"kappa_U{suf}")],
                  dagger(head)))
    A.append((r"Matured exit, $\vartheta^{\omega}$ (check)", [None, est(head, "theta_w"), None], ""))
    B = []
    if l1 is not None:
        for code, suf, label in PAIRS:
            B.append((label, [est(l1, f"dkappa{suf}"), est(l1, f"kappa{suf}"), None], dagger(l1)))
        B.append((r"$\ln E_c=\tilde\kappa_{\mathrm{sav}}-\tilde\kappa_{\mathrm{td}}$", [None, est(l1, "ln_Ec"), None], ""))
    C = []
    for sample, label in ROBUSTNESS:
        r = pick(rows, f"before_after+{sample}")
        if r is not None:
            C.append((label, [est(r, "dkappa"), est(r, "kappa"), None], dagger(r)))
    return dict(A=A, B=B, C=C, head=head, dose=dose, l1=l1, td_note=td_note,
                smoke=any(r["_smoke"] for r in rows.values()))


def render_tex(b) -> str:
    L = ["% tab_k1_netflow.tex -- GENERATED by make_k1_table.py. Do not edit.",
         "% Tabular only (needs booktabs): the paper wraps it; the note is tab_k1_netflow_note.tex.",
         f"% Source: K1 runs on panel {b['head']['_panel']} ({b['head']['_run']}). Keys: \\pn{{kappa.k1.*}}.",
         r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
         r" & Pix change & Level (bound) & Dose \\", r"\midrule"]

    def panel(title, rows_):
        L.append(r"\multicolumn{4}{@{}l}{\textit{" + title + r"}} \\")
        for label, cells, flag in rows_:
            if cells is None:
                L.append(f"{label} & " + r"\multicolumn{3}{c}{" + flag + r"} \\")
                continue
            c = [cell(e) for e in cells]
            L.append(f"{label} & {c[0][0]}{flag if c[0][0] else ''} & {c[1][0]} & {c[2][0]} " + r"\\")
            if any(x[1] for x in c):
                L.append(f" & {c[0][1]} & {c[1][1]} & {c[2][1]} " + r"\\")
    panel(r"Panel A: headline, statement $\lambda$", b["A"])
    if b["B"]:
        L.append(r"\midrule")
        panel(r"Panel B: $\lambda = 1$ for every bank", b["B"])
    if b["C"]:
        L.append(r"\midrule")
        panel("Panel C: savings pair, robustness samples", b["C"])
    L += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(L)


def render_note(b) -> str:
    h = b["head"]
    draws = h.get("_draws")
    parts = [
        r"\footnotesize",
        "Net-flow equation on the observed net flows, nonlinear least squares with bank $\\times$ market "
        "and market $\\times$ quarter effects, one switching cost $\\tilde\\kappa$ per source pair: savings "
        "(the headline pair), checking with matured time deposits, and daily-liquidity time deposits. "
        "Pix change: $\\Delta\\tilde\\kappa$ from 2020Q4, the headline estimate. Level: $\\tilde\\kappa$ "
        "before Pix, an upper bound on $\\kappa$. Dose: the change per unit of fitted Pix use.",
        "Brackets: 95 percent intervals from the score wild cluster bootstrap by destination "
        "conglomerate (Webb weights, " + (f"{int(draws):,} draws, " if draws is not None else "")
        + "symmetric percentile-$t$).",
        f"Panel A: {int(h['n']):,} observations, {int(h['markets']):,} markets, {int(h['G']):,} clusters"
        + (f" (effective number $G^*$ {num(h['Gstar'])})" if pd.notna(h.get("Gstar")) else "") + "."]
    if b["td_note"]:
        parts.append("Time deposits are not estimable in the headline: the daily-liquidity share of the "
                     "time-deposit pools from the statements is about zero, so those pools drop out.")
    if b["l1"] is not None:
        parts.append("Panel B sets $\\lambda = 1$ for every bank; there $\\ln E_c = \\tilde\\kappa_{\\mathrm{sav}}"
                     " - \\tilde\\kappa_{\\mathrm{td}}$, the clock under one switching cost across pairs.")
    parts.append("$\\vartheta^{\\omega}$: the matured exit coefficient, one constant, 1 in the model, "
                 "reported as a check.")
    if any(r[2] for r in b["A"] + b["B"] + b["C"] if isinstance(r[2], str) and "dagger" in r[2]):
        parts.append("$^{\\dagger}$ The fit stopped at its iteration limit.")
    if b["smoke"]:
        parts.append("SMOKE TEST, NOT FOR THE PAPER.")
    return " ".join(parts) + "\n"


def render_md(b) -> str:
    out = ["<!-- GENERATED by make_k1_table.py. Do not edit. -->", "",
           "| Row | Pix change | Level (bound) | Dose |", "|---|---|---|---|"]
    for title, rows_ in (("Panel A: headline, statement lambda", b["A"]), ("Panel B: lambda = 1", b["B"]),
                         ("Panel C: savings pair, robustness", b["C"])):
        if not rows_:
            continue
        out.append(f"| *{title}* | | | |")
        for label, cells, flag in rows_:
            if cells is None:
                out.append(f"| {label} | {flag} | {flag} | {flag} |")
                continue
            c = [cell(e) for e in cells]
            fix = lambda s: s.replace(MINUS, "−").replace("{,}", ",").replace("$^{\\dagger}$", "†")
            out.append(f"| {label} | {fix(c[0][0] + (flag if c[0][0] else ''))} {fix(c[0][1])} | "
                       f"{fix(c[1][0])} {fix(c[1][1])} | {fix(c[2][0])} {fix(c[2][1])} |")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=None, help="a K1 run folder (repeat); default Rout/k1, Rout/k1_lambda_one")
    ap.add_argument("--allow-smoke", action="store_true", help="render a smoke run; needs --out-dir outside the paper folder")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    drafts = _paths.drafts_dir().resolve()
    out = Path(args.out_dir).resolve() if args.out_dir else drafts
    if args.allow_smoke and (out == drafts or drafts in out.parents):
        print("refused: --allow-smoke writes only outside the paper folder (give --out-dir)")
        return 1
    try:
        rows = load_runs(args.run, allow_smoke=args.allow_smoke)
    except K1Refused as e:
        print(f"refused: {e}")
        return 1
    b = build(rows)
    out.mkdir(parents=True, exist_ok=True)
    for name, text in ((TEX, render_tex(b)), (NOTE, render_note(b)), (MD, render_md(b))):
        (out / name).write_text(text, encoding="utf-8", newline="\n")
        print(f"  wrote {out / name}")
    if b["smoke"]:
        print("  smoke render: " + "; ".join(sorted({x for r in rows.values() for x in r['_reasons']})))
    return 0


if __name__ == "__main__":
    sys.exit(main())

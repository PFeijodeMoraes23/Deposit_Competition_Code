"""
make_tax_clock_table.py
=======================
The cost of moving a time deposit (CDB) before maturity: the "tax clock". Brazilian time-deposit
income is taxed at source at a rate that falls with the holding period (Lei 11.033/2004, art. 1),
and redemptions within 30 days also pay the tax on financial operations (IOF) on a regressive
daily table (Decreto 6.306/2007, art. 32 and its annex). Moving an application to another bank
restarts both schedules, so the depositor gives up the lower rate the old application had reached.

For an application paying the gross annual rate r (the CDI), aged a days at the move, whose money
then stays invested h more days:

  stay      W_s = 1 + [1 - tau(a + h)] [(1 + r)^((a + h)/365) - 1]
  move      P   = 1 + [1 - tau(a)] [1 - iota(a)] [(1 + r)^(a/365) - 1]
            W_m(r') = P {1 + [1 - tau(h)] [(1 + r')^(h/365) - 1]}
  cost per R$100 of principal        100 [W_s - W_m(r)]
  break-even premium (pp a year)     100 (r* - r), W_m(r*) = W_s, by bisection
  premium as a percent of the CDI    100 (r* - r) / r

tau(d) is the withholding rate on income at age d days and iota(d) the IOF share of income, both
entered below as data. iota does not bind in any cell: every age is at least 90 days. The cells
are the six (age, horizon) pairs of deposit_types_report_2026-10-15.md, Table 3, at a CDI of 10
percent a year, and the same cells at 13.75 and 6.5 percent. The script checks that the CDI-10
cells reproduce that table and exits with status 1 if they do not.

Rounding: printed values round the unrounded number once, to 2 decimals (cost, premium in pp) or
1 (percent of the CDI), as Python's format does. The 800-day, 180-day cost is 0.5748, printed
0.57; the report printed 0.58 by rounding a 3-decimal 0.575 a second time.

Outputs
  Drafts/Deposit Competition/tab_tax_clock.tex   tabular only (booktabs); the paper wraps it
  Drafts/Deposit Competition/tab_tax_clock.md    the same table in markdown
  Rout/tax_clock_cost.csv                        18 rows, unrounded
  Rout/tax_clock.json                            the sidecar make_paper_numbers.py reads
                                                 (keys clock.t3.*)

Usage:  python make_tax_clock_table.py            (Panel A and Panel B)
        python make_tax_clock_table.py --no-panel-b
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import argparse
import datetime
import json
import sys

import pandas as pd

from utils import paths as _paths

# Withholding tax on time-deposit income, Lei 11.033/2004, art. 1: (last day of the bracket,
# rate). Ages beyond the last bracket pay the last rate.
WITHHOLDING = ((180, 0.225), (360, 0.200), (720, 0.175), (None, 0.150))
# IOF on redemptions, Decreto 6.306/2007, annex: percent of the income due on day 1, 2, ..., 29
# after the application; zero from day 30 on.
IOF_PCT = (96, 93, 90, 86, 83, 80, 76, 73, 70, 66, 63, 60, 56, 53, 50, 46, 43, 40, 36, 33, 30,
           26, 23, 20, 16, 13, 10, 6, 3)

CELLS = ((90, 365), (200, 180), (200, 730), (400, 365), (800, 180), (800, 730))
CDI_MAIN = 10.0
CDI_LEVELS = (13.75, 10.0, 6.5)          # percent a year, Panel B order

# The six CDI-10 cells of deposit_types_report_2026-10-15.md, Table 3:
# (age, horizon) -> (cost per R$100, premium in pp a year, premium in percent of the CDI).
REPORT = {(90, 365): (0.16, 0.19, 1.9), (200, 180): (0.43, 1.13, 11.3),
          (200, 730): (0.46, 0.24, 2.4), (400, 365): (0.71, 0.79, 7.9),
          (800, 180): (0.58, 1.32, 13.2), (800, 730): (0.62, 0.28, 2.8)}
TOLERANCE = (0.01, 0.01, 0.1)

DEC = dict(cost=2, premium_pp=2, premium_pct_cdi=1)     # printed decimals

DRAFTS = _paths.drafts_dir()
CSV = _paths.rout_dir() / "tax_clock_cost.csv"
SIDECAR = _paths.rout_dir() / "tax_clock.json"
TEX = DRAFTS / "tab_tax_clock.tex"
MD = DRAFTS / "tab_tax_clock.md"


def tau(days: int) -> float:
    """Withholding rate on income held `days` days."""
    for last, rate in WITHHOLDING:
        if last is None or days <= last:
            return rate
    raise ValueError(days)


def iota(days: int) -> float:
    """IOF as a share of income for a redemption `days` days after the application."""
    return IOF_PCT[days - 1] / 100.0 if 1 <= days <= len(IOF_PCT) else 0.0


def wealth_stay(r: float, a: int, h: int) -> float:
    return 1.0 + (1.0 - tau(a + h)) * ((1.0 + r) ** ((a + h) / 365.0) - 1.0)


def wealth_move(r: float, r_new: float, a: int, h: int) -> float:
    p = 1.0 + (1.0 - tau(a)) * (1.0 - iota(a)) * ((1.0 + r) ** (a / 365.0) - 1.0)
    return p * (1.0 + (1.0 - tau(h)) * ((1.0 + r_new) ** (h / 365.0) - 1.0))


def break_even(r: float, a: int, h: int, tol: float = 1e-13) -> float:
    """r* - r, the extra annual rate the new bank must pay for moving to match staying."""
    target = wealth_stay(r, a, h)
    lo, hi = r, r + 1.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if wealth_move(r, mid, a, h) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi) - r


def compute() -> pd.DataFrame:
    rows = []
    for cdi in CDI_LEVELS:
        r = cdi / 100.0
        for a, h in CELLS:
            prem = break_even(r, a, h)
            rows.append(dict(cdi_pct=cdi, age_days=a, horizon_days=h,
                             cost_per_100=100.0 * (wealth_stay(r, a, h) - wealth_move(r, r, a, h)),
                             premium_pp=100.0 * prem, premium_pct_cdi=100.0 * prem / r))
    return pd.DataFrame(rows)


def validate(df: pd.DataFrame) -> list:
    """Differences between the CDI-10 cells and the report's Table 3 beyond TOLERANCE."""
    bad = []
    main = df[df["cdi_pct"] == CDI_MAIN].set_index(["age_days", "horizon_days"])
    for cell, ref in REPORT.items():
        got = main.loc[cell, ["cost_per_100", "premium_pp", "premium_pct_cdi"]].to_numpy()
        for name, g, x, tol in zip(("cost", "premium", "percent of CDI"), got, ref, TOLERANCE):
            if abs(g - x) > tol + 1e-12:
                bad.append(f"age {cell[0]}, horizon {cell[1]}: {name} {g:.4f} vs report {x}")
    return bad


def fmt(x: float, d: int) -> str:
    s = f"{x:.{d}f}"
    return s[1:] if s.startswith("-") and float(s) == 0.0 else s


def ranges(df: pd.DataFrame) -> dict:
    """{cdi: {measure: (min, max)}} over the six cells."""
    out = {}
    for cdi, g in df.groupby("cdi_pct", sort=False):
        out[cdi] = {k: (float(g[k].min()), float(g[k].max()))
                    for k in ("cost_per_100", "premium_pp", "premium_pct_cdi")}
    return out


def cdi_label(cdi: float) -> str:
    return f"{cdi:g}"


def render_tex(df: pd.DataFrame, panel_b: bool) -> str:
    main = df[df["cdi_pct"] == CDI_MAIN]
    rng = ranges(df)
    lines = [
        "% tab_tax_clock.tex -- GENERATED by make_tax_clock_table.py. Do not edit.",
        "% Tabular only (needs booktabs): the paper wraps it in its own table, caption and notes.",
        "% Cost of moving a time deposit before maturity, per R$100 of principal, and the break-even",
        "% premium, from the withholding schedule of Lei 11.033/2004 and the IOF of Decreto",
        "% 6.306/2007. Unrounded values: Rout/tax_clock_cost.csv. Keys: \\pn{clock.t3.*}.",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r" & & & \multicolumn{2}{c}{Break-even premium} \\",
        r"\cmidrule(l){4-5}",
        r"Age at move (days) & Horizon (days) & Cost per R\$100 & pp a year & \% of CDI \\",
        r"\midrule",
        rf"\multicolumn{{5}}{{@{{}}l}}{{\textit{{Panel A: CDI at {cdi_label(CDI_MAIN)} percent a year}}}} \\",
    ]
    for _, r in main.iterrows():
        lines.append(f"{int(r['age_days'])} & {int(r['horizon_days'])} & "
                     f"{fmt(r['cost_per_100'], DEC['cost'])} & "
                     f"{fmt(r['premium_pp'], DEC['premium_pp'])} & "
                     f"{fmt(r['premium_pct_cdi'], DEC['premium_pct_cdi'])} \\\\")
    if panel_b:
        lines += [r"\midrule",
                  r"\multicolumn{5}{@{}l}{\textit{Panel B: smallest and largest value over the six cells "
                  r"of Panel A, by level of the CDI}} \\"]
        for cdi in CDI_LEVELS:
            g = rng[cdi]
            cells = [f"{fmt(g[k][0], d)} to {fmt(g[k][1], d)}" for k, d in
                     (("cost_per_100", DEC["cost"]), ("premium_pp", DEC["premium_pp"]),
                      ("premium_pct_cdi", DEC["premium_pct_cdi"]))]
            lines.append(f"CDI {cdi_label(cdi)} percent & & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def render_md(df: pd.DataFrame, panel_b: bool) -> str:
    main = df[df["cdi_pct"] == CDI_MAIN]
    rng = ranges(df)
    out = ["<!-- GENERATED by make_tax_clock_table.py. Do not edit. -->", "",
           f"Panel A: CDI at {cdi_label(CDI_MAIN)} percent a year", "",
           "| Age at move (days) | Horizon (days) | Cost per R$100 | Break-even premium "
           "(pp a year) | Break-even premium (% of CDI) |",
           "|---|---|---|---|---|"]
    for _, r in main.iterrows():
        out.append(f"| {int(r['age_days'])} | {int(r['horizon_days'])} | "
                   f"{fmt(r['cost_per_100'], DEC['cost'])} | "
                   f"{fmt(r['premium_pp'], DEC['premium_pp'])} | "
                   f"{fmt(r['premium_pct_cdi'], DEC['premium_pct_cdi'])} |")
    if panel_b:
        out += ["", "Panel B: smallest and largest value over the six cells of Panel A, by level of "
                    "the CDI", "",
                "| CDI (percent a year) | Cost per R$100 | Break-even premium (pp a year) | "
                "Break-even premium (% of CDI) |", "|---|---|---|---|"]
        for cdi in CDI_LEVELS:
            g = rng[cdi]
            out.append(f"| {cdi_label(cdi)} | "
                       + " | ".join(f"{fmt(g[k][0], d)} to {fmt(g[k][1], d)}" for k, d in
                                    (("cost_per_100", DEC["cost"]),
                                     ("premium_pp", DEC["premium_pp"]),
                                     ("premium_pct_cdi", DEC["premium_pct_cdi"]))) + " |")
    return "\n".join(out) + "\n"


def sidecar(df: pd.DataFrame, panel_b: bool) -> dict:
    cells = {}
    for cdi in CDI_LEVELS:
        g = df[df["cdi_pct"] == cdi]
        cells[cdi_label(cdi)] = [dict(age=int(r["age_days"]), horizon=int(r["horizon_days"]),
                                      cost=float(r["cost_per_100"]),
                                      premium_pp=float(r["premium_pp"]),
                                      premium_pct_cdi=float(r["premium_pct_cdi"]))
                                 for _, r in g.iterrows()]
    return dict(_meta=dict(generator="make_tax_clock_table.py",
                           generated=datetime.datetime.now().isoformat(timespec="seconds"),
                           tex_file=TEX.name, main_cdi=cdi_label(CDI_MAIN), panel_b=panel_b,
                           decimals=DEC,
                           iof="legal daily table of Decreto 6.306/2007 (does not bind: ages >= 90)",
                           rate="the application pays the CDI (100 percent of the CDI)"),
                cells=cells)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--no-panel-b", action="store_true",
                    help="omit Panel B (the CDI variants); the keys are written either way")
    args = ap.parse_args()
    panel_b = not args.no_panel_b

    df = compute()
    bad = validate(df)
    if bad:
        print("Validation FAILED against the report's Table 3:\n  " + "\n  ".join(bad))
        return 1
    print("Validation passed: the six CDI-10 cells reproduce the report's Table 3.")

    CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV, index=False, float_format="%.10g")
    SIDECAR.write_text(json.dumps(sidecar(df, panel_b), indent=2) + "\n", encoding="utf-8")
    TEX.write_text(render_tex(df, panel_b), encoding="utf-8", newline="\n")
    MD.write_text(render_md(df, panel_b), encoding="utf-8", newline="\n")
    for p in (CSV, SIDECAR, TEX, MD):
        print(f"  wrote {p}")
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 120):
        print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

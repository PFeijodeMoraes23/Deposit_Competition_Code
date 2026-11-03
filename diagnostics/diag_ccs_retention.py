"""diag_ccs_retention.py -- three checks on the CCS relationship flows (LAI 18810.025400/2026-04),
PF only, asked in the handoff of 2026-10-31 (G5). Descriptive: no paper numbers.

1. Cohort retention (opened and still active / gross openings) by opening year and firm type, in
   three versions: every institution; without the institutions whose base was closed by a sourced
   decision of the firm (SHUTDOWNS); and without those and the institutions with no active
   relationship left (every relationship opened since 2016-01 closed by the extraction date).
   Retention is a property of the cohort, so an event month cannot be removed from it: whole
   institutions are.
2. The 2021 B cohort by conglomerate, with the B conglomerates' retention in the 2019-2022 cohorts.
3. CC4's comparison with the complaints ranking (diag_ccs_vs_ranking.py), the share of quarter pairs
   whose gap at the quarter's second month is within 1% of the stock, for banks, payment
   institutions and the rest separately. Segment = the BCB file's segment of the institution.

Reads (utils.paths.rout_dir()): ccs_flows_institution_month_mapped.csv (make_ccs_flows_conglomerate.py)
and ccs_vs_ranking.csv (diag_ccs_vs_ranking.py); the segments from the compiled flows file.
Writes rout_dir() / ccs_retention_g5.md.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: utils/
from utils import paths, raw_paths  # noqa: E402

ROUT = paths.rout_dir()
MONTHLY = ROUT / "ccs_flows_institution_month_mapped.csv"
RANKING_PAIRS = ROUT / "ccs_vs_ranking.csv"
FLOWS = raw_paths.BCB_CCS_LAI.parent / "compiled" / "ccs_flows_institution_month.csv"
OUT = ROUT / "ccs_retention_g5.md"
# Sourced closures of a whole client base by the firm (config/ccs_flow_events.csv): Ame Digital
# (2024), PayPal's PF payment accounts (2022), Superdigital (2024). HSBC to Bradesco (Kirton, 2016)
# is a conversion inside a B firm, not a shutdown, and stays in.
SHUTDOWNS = {"32778350": "Ame Digital", "10878448": "PayPal", "09554480": "Superdigital"}
GROUPS = ["B", "D", "not in panel"]


def group(d: pd.DataFrame) -> np.ndarray:
    return np.select([(d["in_panel"] == 1) & (d["is_B"] == 1), (d["in_panel"] == 1) & (d["is_B"] == 0)],
                     ["B", "D"], "not in panel")


def md(df: pd.DataFrame, fmt="{:.3f}") -> str:
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join([str(df.index.name or "")] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for i, r in df.iterrows():
        out.append("| " + " | ".join([str(i)] + [fmt.format(v) if isinstance(v, (float, np.floating))
                                                  and np.isfinite(v) else ("" if isinstance(v, float) else str(v))
                                                  for v in r]) + " |")
    return "\n".join(out)


def retention(d: pd.DataFrame) -> pd.DataFrame:
    r = d.groupby(["year", "group"])[["opened_active", "opened_gross"]].sum()
    return (r["opened_active"] / r["opened_gross"]).unstack("group").reindex(columns=GROUPS)


def main():
    d = pd.read_csv(MONTHLY, dtype={"cnpj8": str})
    d = d[d["type"] == "PF"].copy()
    d["group"] = group(d)
    d["year"] = d["ym"].str[:4].astype(int)
    d = d[d["year"] <= 2024]
    out = ["# CCS checks for the handoff of 2026-10-31 (G5), PF", ""]

    # ── 1. retention, three versions ─────────────────────────────────────────────────────────────
    inst = d.groupby("cnpj8").agg(gross=("opened_gross", "sum"), active=("opened_active", "sum"),
                                  group=("group", "first"), name=("name", "first"))
    zero = inst[(inst["gross"] > 0) & (inst["active"] == 0)]
    shut = d["cnpj8"].isin(SHUTDOWNS)
    no_left = d["cnpj8"].isin(zero.index)
    v_all, v_shut, v_both = retention(d), retention(d[~shut]), retention(d[~shut & ~no_left])
    t = pd.concat({"all": v_all, "excl. shutdowns": v_shut, "excl. shutdowns and none left": v_both}, axis=1)
    t.columns = [f"{g}: {v}" for v, g in t.columns]
    t = t[[f"{g}: {v}" for g in GROUPS for v in ("all", "excl. shutdowns", "excl. shutdowns and none left")]]
    t.index.name = "opening year"
    gross = d.groupby("group")["opened_gross"].sum()
    excl = pd.DataFrame({
        "shutdown institutions": d[shut].groupby("group")["cnpj8"].nunique(),
        "their gross (m)": d[shut].groupby("group")["opened_gross"].sum() / 1e6,
        "none-left institutions": zero.groupby("group").size(),
        "their gross (m)_": zero.groupby("group")["gross"].sum() / 1e6,
        "share of the group's gross": (d[shut | no_left].groupby("group")["opened_gross"].sum() / gross),
    }).reindex(GROUPS).fillna(0)
    for k in ("shutdown institutions", "none-left institutions"):
        excl[k] = excl[k].astype(int).astype(str)
    excl.index.name = "group"
    top_zero = zero.sort_values("gross", ascending=False).head(8)
    out += ["## 1. Cohort retention by opening year and firm type", "",
            "Opened and still active at the BCB's unstated extraction date / gross openings. "
            "\"Shutdowns\": " + ", ".join(SHUTDOWNS.values()) + " (sourced in config/ccs_flow_events.csv). "
            "\"None left\": institutions whose every relationship opened since 2016-01 was closed by the "
            "extraction date.", "", md(t), "", "What is removed, 2016-2024 openings:", "",
            md(excl.rename(columns={"their gross (m)_": "their gross (m) "})), "",
            "Largest none-left institutions: " + "; ".join(
                f"{r.name} ({r.group}, {r.gross / 1e6:.2f} m)" for r in top_zero.itertuples()), ""]

    # ── 2. the 2021 B cohort by conglomerate ─────────────────────────────────────────────────────
    b = d[d["group"] == "B"]
    names = (b.groupby(["conglomerate", "name"])["opened_gross"].sum().reset_index()
             .sort_values("opened_gross", ascending=False).drop_duplicates("conglomerate")
             .set_index("conglomerate")["name"])
    c = b.groupby(["conglomerate", "year"])[["opened_gross", "opened_active", "opened_closed_later"]].sum()
    c["retention"] = c["opened_active"] / c["opened_gross"]
    y21 = c.xs(2021, level="year").sort_values("opened_gross", ascending=False)
    tot21 = y21[["opened_gross", "opened_active", "opened_closed_later"]].sum()
    ret_ref = float(v_all.loc[2016:2019, "B"].mean())
    t2 = pd.DataFrame({
        "largest institution": names.reindex(y21.index),
        "gross (m)": y21["opened_gross"] / 1e6,
        "retention": y21["retention"],
        "share of B gross": y21["opened_gross"] / tot21["opened_gross"],
        "share of B closed-later": y21["opened_closed_later"] / tot21["opened_closed_later"],
        "excess closed (m)": (y21["opened_closed_later"] - (1 - ret_ref) * y21["opened_gross"]) / 1e6,
    }).head(10)
    t2.index.name = "conglomerate"
    ret = c["retention"].unstack("year").reindex(y21.index[:6])[[2019, 2020, 2021, 2022]]
    ret.insert(0, "largest institution", names.reindex(ret.index))
    ret.index.name = "conglomerate"
    top_c = t2["excess closed (m)"].idxmax()
    inst21 = (b[(b["conglomerate"] == top_c) & (b["year"] == 2021)]
              .groupby("name")[["opened_gross", "opened_active"]].sum())
    inst21 = inst21[inst21["opened_gross"] >= 1e4].sort_values("opened_gross", ascending=False)
    inst21 = pd.DataFrame({"gross (m)": inst21["opened_gross"] / 1e6,
                           "retention": inst21["opened_active"] / inst21["opened_gross"]})
    inst21.index.name = "institution"
    out += ["## 2. The 2021 B cohort by conglomerate", "",
            f"B's 2021 cohort: {tot21['opened_gross'] / 1e6:.2f} m gross, retention "
            f"{tot21['opened_active'] / tot21['opened_gross']:.3f}. \"Excess closed\" = closed-later minus "
            f"what B's mean 2016-2019 retention ({ret_ref:.3f}) implies; the column sums to the cohort's "
            "shortfall.", "", md(t2), "", "Retention of the six largest, by cohort year:", "", md(ret), "",
            f"The 2021 cohort of {top_c} ({names.get(top_c)}), the largest excess, by institution "
            "(10,000 openings or more):", "", md(inst21), "",
            "Banco Votorantim (BV, C0080484) is a D firm in the panel (is_B = 0), so its 2022-05 closures "
            "are in D, not in this cohort.", ""]

    # ── 3. CC4 within-1% by segment ──────────────────────────────────────────────────────────────
    seg = (pd.read_csv(FLOWS, dtype={"cnpj8": str}, usecols=["cnpj8", "segment"]).dropna()
           .drop_duplicates("cnpj8").set_index("cnpj8")["segment"])
    v = pd.read_csv(RANKING_PAIRS, dtype={"cnpj8": str})
    # CC4's rule: every pair with a month-2 net flow; a pair whose previous stock is 0 (an
    # institution's first ranking quarter) counts as outside 1%.
    v = v[v["net_to_month2"].notna()].copy()
    s = v["cnpj8"].map(seg).fillna("")
    v["class"] = np.select([s.str.startswith("Banco") | s.str.startswith("Caixa"),
                            s.str.startswith("Instituição de Pagamento")], ["banks", "payment institutions"], "other")
    prev = v["n_clients_ccs_prev"].where(v["n_clients_ccs_prev"] > 0)
    v["within"] = (v["gap_to_month2"].abs() / prev) <= 0.01
    v["within_cal"] = (v["gap"].abs() / prev) <= 0.01
    rows = []
    for k, g in [("all", v)] + list(v.groupby("class")):
        rows.append(dict(cls=k, institutions=g["cnpj8"].nunique(), pairs=len(g),
                         within_1pct_month2=g["within"].mean(), within_1pct_calendar=g["within_cal"].mean(),
                         median_abs_gap_month2=float(g["gap_to_month2"].abs().median())))
    t3 = pd.DataFrame(rows).set_index("cls")
    t3.index.name = "segment"
    t3["institutions"] = t3["institutions"].astype(int).astype(str)
    t3["pairs"] = t3["pairs"].astype(int).astype(str)
    worst = (v[v["class"] == "payment institutions"].assign(miss=~v["within"])
             .groupby(["cnpj8", "name"])["miss"].agg(["sum", "size"]).sort_values("sum", ascending=False).head(8))
    out += ["## 3. The ranking check (CC4) by segment", "",
            "Share of quarter pairs whose gap is within 1% of the previous stock: the net flow over the three "
            "months ending at the quarter's second month (the ranking's reference date) and, for comparison, "
            "over the calendar quarter. CC4's rule: a pair whose previous stock is 0 counts as outside.", "",
            md(t3), "", "Payment institutions with the most pairs outside 1%: " + "; ".join(
                f"{n} ({int(r['sum'])} of {int(r['size'])})" for (_, n), r in worst.iterrows()), ""]

    OUT.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

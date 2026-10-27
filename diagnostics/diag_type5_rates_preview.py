"""
diag_type5_rates_preview.py
===========================
Preview of the posted type-5 balance rules (config/type5_balance_rates.csv) on the current market
panel, without writing any pipeline file. It applies utils/type5_rates.apply_rules, the function
step 16 (panel_deposit_rates.py) calls, to each conglomerate-quarter's stored rate_a5, and reports:

  - the diff by conglomerate x quarter: the stored and the posted rate and spread (annualised by
    compounding, in percent) and the type-5 balance it applies to;
  - the share of 2020-2024 type-5 balances whose rate changes, and the share at a zero rate;
  - checks: only conglomerates named in the rules change; per conglomerate, the quarters changed.

The CDI and Selic come from PANEL_INTERMED/quarterly_macro_rates.csv (percent; step 16's output).
Writes Rout/type5_rates_preview.csv.

Usage:  python diagnostics/diag_type5_rates_preview.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.venv_guard import ensure_project_venv  # noqa: E402

ensure_project_venv(__file__)

import pandas as pd  # noqa: E402

from utils import paths as _paths  # noqa: E402
from utils import type5_rates  # noqa: E402

KEY = "CodConglomeradoPrudencial"


def main():
    panel = Path(_paths.PROCESSED) / "market_panel.parquet"
    macro = Path(_paths.PROCESSED) / "PANEL_INTERMED" / "quarterly_macro_rates.csv"
    p = pd.read_parquet(panel, columns=[KEY, "NomeInstituicao", "year", "quarter", "dep_a5", "rate_a5"])
    p["AnoMes"] = p["year"] * 100 + 3 * p["quarter"]
    m = pd.read_csv(macro)
    for c in ("cdi_qoq", "selic_qoq"):
        m[c] = m[c] / 100.0

    g = p.groupby([KEY, "AnoMes"]).agg(name=("NomeInstituicao", "first"), dep_a5=("dep_a5", "sum"),
                                       rate_a5=("rate_a5", "first"),
                                       n_rate=("rate_a5", "nunique")).reset_index()
    if (g["n_rate"] > 1).any():
        raise SystemExit("rate_a5 is not constant within a conglomerate-quarter; preview invalid")
    g = g.merge(m[["AnoMes", "cdi_qoq", "selic_qoq"]], on="AnoMes", how="left")

    rules = type5_rates.load_rules()
    new, diff = type5_rates.apply_rules(g, rules)
    diff = diff.merge(g[[KEY, "AnoMes", "name", "dep_a5", "selic_qoq"]], on=[KEY, "AnoMes"], how="left")
    ann = lambda r: ((1 + r) ** 4 - 1) * 100  # noqa: E731
    diff["rate_chain_ann_pct"] = ann(diff["rate_chain"])
    diff["rate_posted_ann_pct"] = ann(diff["rate_posted"])
    diff["spread_chain_ann_pp"] = ann(diff["selic_qoq"]) - diff["rate_chain_ann_pct"]
    diff["spread_posted_ann_pp"] = ann(diff["selic_qoq"]) - diff["rate_posted_ann_pct"]
    out = Path(_paths.rout_dir()) / "type5_rates_preview.csv"
    diff.to_csv(out, index=False)

    win = new[(new["AnoMes"] >= 202003) & (new["AnoMes"] <= 202412)]
    tot = win["dep_a5"].sum()
    changed = win[win["rate_a5_posted"] == 1]
    zero = win[(win["rate_a5_posted"] == 1) & (win["rate_a5"].abs() < 1e-12)]
    print(f"rules: {len(rules)} for {rules['conglomerate'].nunique()} conglomerates")
    print(f"conglomerate-quarters changed: {len(diff):,} ({diff[KEY].nunique()} conglomerates)")
    print(f"share of 2020-2024 type-5 balances whose rate changes: {changed['dep_a5'].sum() / tot:.4f}")
    print(f"share of 2020-2024 type-5 balances at a zero rate:      {zero['dep_a5'].sum() / tot:.4f}")
    stray = set(diff[KEY]) - set(rules["conglomerate"])
    print("CHECK only ruled conglomerates change:", "PASS" if not stray else f"FAIL {sorted(stray)}")
    per = diff.groupby([KEY, "name"]).agg(
        first=("AnoMes", "min"), last=("AnoMes", "max"), quarters=("AnoMes", "size"),
        share_type5=("dep_a5", lambda s: s.sum() / tot),
        spread_chain_mean_pp=("spread_chain_ann_pp", "mean"),
        spread_posted_mean_pp=("spread_posted_ann_pp", "mean")).reset_index()
    print(per.round(3).to_string(index=False))

    # Promotional and tiered rules, listed for confirmation: the type-5 balance-quarters they price.
    flagged = rules[rules["decision"].str.contains("promo|tiered", case=False)]
    if len(flagged):
        seg = type5_rates.quarter_segments(flagged, g["AnoMes"].unique())
        seg = seg.merge(g[[KEY, "AnoMes", "dep_a5"]].rename(columns={KEY: "conglomerate"}),
                        on=["conglomerate", "AnoMes"], how="left")
        seg["dep_days"] = seg["dep_a5"].fillna(0) * seg["frac"]
        lst = seg.groupby("rule_id").agg(quarters=("AnoMes", lambda s: f"{s.min()}-{s.max()}"),
                                         share_type5=("dep_days", lambda s: s.sum() / tot))
        lst = flagged.set_index("rule_id")[["conglomerate", "start", "end", "pct_cdi", "decision"]].join(lst)
        print("\npromotional or tiered rules (headline rate applied), for confirmation:")
        print(lst.assign(share_type5=lst["share_type5"].round(4)).to_string())

    # Each conglomerate's rules stay inside the quarters its rows declare (D3: C0080903 2020Q2-2024Q3,
    # C0088022 from 2024Q4).
    for c, r in rules.groupby("conglomerate"):
        q = diff.loc[diff[KEY] == c, "AnoMes"]
        if q.empty:
            continue
        lo = int(r["start"].min().year * 100 + 3 * ((r["start"].min().month - 1) // 3 + 1))
        hi = int(r["end"].max().year * 100 + 3 * ((r["end"].max().month - 1) // 3 + 1))
        ok = q.min() >= lo and q.max() <= hi
        print(f"CHECK {c} changed only in {lo}-{hi}: {'PASS' if ok else 'FAIL'} (changed {q.min()}-{q.max()})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

"""
diag_openfinance_coverage.py
============================
Assess COVERAGE and DATA QUALITY of the Open Finance Brasil open-data fee panel
(scrape_19 output). Prints a Markdown-ready report and saves a small CSV, so the
numbers in service_fees_notes.md are reproducible rather than hand-typed.

Coverage   : families, institutions (cnpj8), PF/PJ, service codes, conglomerates,
             and — the meaningful one — SHARE OF SYSTEM DEPOSITS covered (are the
             big banks in?), plus presence of the systemically-important banks.
Quality    : null/missing rates; internal consistency (min <= wavg <= max; customer
             shares sum to ~1); zero-price share; interest-rate plausibility;
             OFB-vs-BCB-listed ratio location.

Usage
-----
  python diag_openfinance_coverage.py
  python diag_openfinance_coverage.py --snapshot 2026-07-22   # a specific data_coleta
"""

from __future__ import annotations

import argparse
import logging

from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

from utils import paths
import panel_9_cosif_fees as p9
import sys

# Windows consoles default to cp1252 and raise UnicodeEncodeError on any non-ASCII
# character in a print (phi, arrows, Upsilon, x). That usually fires on a STATUS line after
# the real work is done, so the script exits non-zero and reports failure for a computation
# that succeeded -- three such false failures on 2026-07-29. Force UTF-8 (no-op if already).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

LONG_CSV = paths.TARIFAS_PROC / "openfinance_fees_panel_long.csv"
OUT_CSV  = paths.TARIFAS_PROC / "openfinance_coverage_report.csv"

# Systemically important groups (by 8-digit CNPJ root) we expect to see.
SIFI = {
    "00000000": "Banco do Brasil", "60746948": "Bradesco", "60701190": "Itaú",
    "90400888": "Santander", "00360305": "Caixa", "18236120": "Nubank",
    "00416968": "Banco Inter", "31872495": "C6", "10573521": "Mercado Pago",
    "09526594": "PagSeguro", "05840689": "BTG", "92894922": "Banco Original",
}


def _pct(n, d):
    return 100.0 * n / d if d else float("nan")


def main(snapshot: str | None = None) -> None:
    df = pd.read_csv(LONG_CSV, low_memory=False, dtype={"cnpj8": str})
    df["cnpj8"] = df["cnpj8"].astype(str).str.zfill(8)
    for c in ("product", "price_kind"):
        if c not in df.columns:
            df[c] = "accounts" if c == "product" else "per_event_brl"
    df["product"] = df["product"].fillna("accounts")
    df["price_kind"] = df["price_kind"].fillna("per_event_brl")
    df["price_weighted_avg"] = pd.to_numeric(df["price_weighted_avg"], errors="coerce")
    df["price_minimum"] = pd.to_numeric(df.get("price_minimum"), errors="coerce")
    df["price_maximum"] = pd.to_numeric(df.get("price_maximum"), errors="coerce")

    snap = snapshot or sorted(df["data_coleta"].unique())[-1]
    cur = df[df["data_coleta"] == snap].copy()

    rep_rows: list[dict] = []
    def add(section, metric, value):
        rep_rows.append({"section": section, "metric": metric, "value": value})

    print(f"\n## Open Finance fee data — coverage & quality (snapshot {snap})\n")

    # ── COVERAGE ────────────────────────────────────────────────────────────
    print("### Coverage\n")
    print("| Product | Rows | Institutions (cnpj8) | PF | PJ | Service codes |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for prod in sorted(cur["product"].unique()):
        s = cur[cur["product"] == prod]
        npf = s[s["customer_type"] == "PF"]["cnpj8"].nunique()
        npj = s[s["customer_type"] == "PJ"]["cnpj8"].nunique()
        print(f"| {prod} | {len(s):,} | {s['cnpj8'].nunique()} | {npf} | {npj} | {s['service_code'].nunique()} |")
        add("coverage", f"{prod}_rows", len(s))
        add("coverage", f"{prod}_institutions", s["cnpj8"].nunique())
    tot_inst = cur["cnpj8"].nunique()
    print(f"\n- Total distinct institutions (cnpj8) this snapshot: **{tot_inst}**")
    add("coverage", "total_institutions", tot_inst)

    # Deposit-share coverage (the meaningful metric) — accounts, PF+PJ.
    try:
        cmap = p9.build_cnpj_cong_map().rename(columns={"cnpj": "cnpj8"})
        cosif = p9.load_cosif_institution()
        cosif["cnpj8"] = cosif["cnpj"].astype(str).str.zfill(8)
        dep = (cosif.sort_values("data_base").groupby("cnpj8", as_index=False)
               .agg(dep_total=("dep_total", "last")))
        dep["dep_total"] = pd.to_numeric(dep["dep_total"], errors="coerce").clip(lower=0)
        covered = set(cur["cnpj8"])
        dep_cov = dep[dep["cnpj8"].isin(covered)]["dep_total"].sum()
        dep_all = dep["dep_total"].sum()
        print(f"- **Deposit-share coverage**: OFB-covered institutions hold "
              f"**{_pct(dep_cov, dep_all):.1f}%** of system deposits "
              f"(R$ {dep_cov/1e12:.2f}tn of {dep_all/1e12:.2f}tn).")
        add("coverage", "deposit_share_pct", round(_pct(dep_cov, dep_all), 2))
    except Exception as e:
        print(f"- deposit-share coverage unavailable: {e}")

    present = [(r, n) for r, n in SIFI.items() if r in set(cur["cnpj8"])]
    missing = [(r, n) for r, n in SIFI.items() if r not in set(cur["cnpj8"])]
    print(f"- Systemically-important banks present ({len(present)}/{len(SIFI)}): "
          f"{', '.join(n for _, n in present)}")
    if missing:
        print(f"- Missing SIFIs: {', '.join(n for _, n in missing)}")
    add("coverage", "sifi_present", len(present))

    # ── QUALITY: per-event fees ───────────────────────────────────────────────
    fees = cur[cur["price_kind"] == "per_event_brl"].copy()
    print(f"\n### Quality — per-event fees ({len(fees):,} rows)\n")
    n = len(fees)
    nn_wavg = fees["price_weighted_avg"].notna().sum()
    nn_min = fees["price_minimum"].notna().sum()
    nn_max = fees["price_maximum"].notna().sum()
    print(f"- Non-null weighted-avg price: **{_pct(nn_wavg, n):.1f}%**; "
          f"min: {_pct(nn_min, n):.1f}%; max: {_pct(nn_max, n):.1f}%")
    add("quality", "feerows", n)
    add("quality", "nonnull_wavg_pct", round(_pct(nn_wavg, n), 2))

    # Internal consistency: min <= wavg <= max
    ok = fees.dropna(subset=["price_minimum", "price_maximum", "price_weighted_avg"])
    viol = ((ok["price_weighted_avg"] < ok["price_minimum"] - 1e-6) |
            (ok["price_weighted_avg"] > ok["price_maximum"] + 1e-6)).sum()
    print(f"- min <= weighted-avg <= max holds for **{_pct(len(ok) - viol, len(ok)):.1f}%** "
          f"of the {len(ok):,} rows with all three ({viol} violations)")
    add("quality", "bound_violations", int(viol))

    # Customer shares sum to ~1
    crc = [c for c in fees.columns if c.startswith("cust_rate_")]
    if crc:
        csum = fees[crc].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
        good = csum.between(0.98, 1.02).sum()
        print(f"- Customer-share bands sum to ~1 (0.98–1.02) for **{_pct(good, csum.notna().sum()):.1f}%** "
              f"of services with band data")
        add("quality", "custshare_sumto1_pct", round(_pct(good, csum.notna().sum()), 2))

    # Zero-price share, negatives
    zero = (fees["price_weighted_avg"] == 0).sum()
    neg = (fees["price_weighted_avg"] < 0).sum()
    print(f"- Zero-priced services (free): {_pct(zero, nn_wavg):.1f}% | negative prices: {neg}")
    add("quality", "zero_price_pct", round(_pct(zero, nn_wavg), 2))
    add("quality", "negative_prices", int(neg))

    # ── QUALITY: interest rates ───────────────────────────────────────────────
    rates = cur[cur["price_kind"] == "rate_pct"].copy()
    if len(rates):
        rv = rates["price_weighted_avg"].dropna()
        print(f"\n### Quality — interest-rate distributions ({len(rates):,} rows)\n")
        print(f"- Non-null rate: {_pct(rates['price_weighted_avg'].notna().sum(), len(rates)):.1f}% | "
              f"monthly-rate range (as fraction): p50={rv.median():.4f}, p95={rv.quantile(.95):.4f}, "
              f"max={rv.max():.4f} | implausible (>1.0 = >100%/mo): {(rv > 1.0).sum()}")
        add("quality", "raterows", len(rates))
        add("quality", "rate_gt100pct", int((rv > 1.0).sum()))

    # ── Cross-source: OFB vs BCB listed ───────────────────────────────────────
    vs_path = paths.TARIFAS_PROC / "openfinance_fees_vs_listed.csv"
    if vs_path.exists():
        vs = pd.read_csv(vs_path)
        ratio_cols = [c for c in vs.columns if c.startswith("ofb_to_listed_ratio_")]
        if ratio_cols:
            allr = pd.concat([pd.to_numeric(vs[c], errors="coerce") for c in ratio_cols])
            in01 = allr.between(0, 1).mean() * 100
            print(f"\n### Cross-source coherence (OFB actual vs BCB listed ceiling)\n")
            print(f"- {len(vs)} conglomerates with both; OFB/listed ratio median **{allr.median():.3f}**, "
                  f"**{in01:.0f}%** of ratios in [0,1] (actual at/below ceiling — as expected)")
            add("coherence", "vs_listed_congs", len(vs))
            add("coherence", "ratio_median", round(float(allr.median()), 4))

    pd.DataFrame(rep_rows).to_csv(OUT_CSV, index=False)
    print(f"\n_Saved machine-readable report: {OUT_CSV.name}_")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="OFB fee coverage & quality assessment")
    ap.add_argument("--snapshot", default=None, help="data_coleta to assess (default: latest)")
    args = ap.parse_args()
    main(snapshot=args.snapshot)

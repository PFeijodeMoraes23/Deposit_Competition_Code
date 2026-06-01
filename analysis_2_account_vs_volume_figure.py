## analysis_2_account_vs_volume_figure.py
# Author: Pedro Feijó de Moraes
#
# Last edited: 2026-05-31
#
# Purpose: Produce the account-share-vs-volume-share (extensive-vs-intensive
#          deposit margin) figure for V_Main.tex sec:inst_setting:desc, from the
#          conglomerate x quarter table built by analysis_1_disclosure_join.py.
#
#   The motivating fact: digital entrants (D firms, type-5 prepaid) hold a large
#   CUSTOMER/ACCOUNT base but a small DEPOSIT-VOLUME share, while incumbents (B
#   firms) hold large volume shares. A volume-only demand model is blind to that
#   gap; this figure makes it visible.
#
#   The script is ADAPTIVE to data coverage (which depends on how far the
#   disclosure backfill reached and whether incumbent client counts are present):
#     PANEL A — for the most-covered quarter, paired bars of each firm's
#               volume_share vs. its account_share (customer count normalised over
#               the firms that disclose counts), annotated with R$/customer.
#     PANEL B — time series of deposits-per-customer (R$) by firm, digital vs
#               incumbent, showing the persistent thin-balance (extensive) gap.
#   If incumbent customer counts are absent, Panel A degrades gracefully to the
#   digital firms plus any incumbent with a count, and the caption flags it.
#
#   Output: processed/DESCRIPTIVES/account_vs_volume.{png,pdf}
#
#   CLI:
#     python analysis_2_account_vs_volume_figure.py
#     python analysis_2_account_vs_volume_figure.py --period 4Q2024
###────────────────────────────────────────────────────────────────────────────

import os
import sys
import argparse
import logging

try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None
if ensure_project_venv is not None:
    ensure_project_venv(__file__)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.disclosure_common import data_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout)
log = logging.getLogger("analysis_2")

BASE = data_root(__file__)
DESC_DIR = os.path.join(BASE, "BCB", "Egan_et_al_2025_Rep", "processed", "DESCRIPTIVES")
JOIN_CSV = os.path.join(DESC_DIR, "account_vs_volume_panel.csv")
OUT_PNG = os.path.join(DESC_DIR, "account_vs_volume.png")
OUT_PDF = os.path.join(DESC_DIR, "account_vs_volume.pdf")

DIGITAL_C = "#E64A19"   # digital / payment
INCUMB_C = "#1565C0"    # incumbent


def _seg_color(seg: str) -> str:
    return INCUMB_C if seg == "incumbent" else DIGITAL_C


def load() -> pd.DataFrame:
    if not os.path.exists(JOIN_CSV):
        log.error(f"join table not found: {JOIN_CSV}\nRun analysis_1_disclosure_join.py first.")
        sys.exit(1)
    df = pd.read_csv(JOIN_CSV)
    # unify a single customer column: prefer Brazil, else broad (consolidated/latam)
    df["customers"] = df["disclosed_customers_brazil"]
    if "disclosed_customers_broad" in df.columns:
        df["customers"] = df["customers"].fillna(df["disclosed_customers_broad"])
    return df


def pick_period(df: pd.DataFrame, forced: str | None) -> str:
    """Choose the quarter with the most firms having BOTH a volume_share and a
    customer count (and, ideally, at least one incumbent)."""
    cov = df.dropna(subset=["volume_share", "customers"])
    if forced:
        return forced
    if cov.empty:
        log.warning("no quarter has both volume_share and a customer count; "
                    "using latest quarter with any volume_share.")
        d2 = df.dropna(subset=["volume_share"])
        return d2.sort_values(["year", "quarter"]).iloc[-1]["period_label"]
    score = (cov.groupby("period_label")
                .agg(n=("firm_key", "nunique"),
                     n_inc=("segment", lambda s: (s == "incumbent").sum()),
                     yq=("year", "first"))
                .sort_values(["n_inc", "n", "yq"], ascending=False))
    best = score.index[0]
    log.info(f"selected period {best}: {score.loc[best,'n']} firms with counts "
             f"({score.loc[best,'n_inc']} incumbent)")
    return best


def panel_a(ax, df: pd.DataFrame, period: str) -> None:
    sub = df[df.period_label == period].dropna(subset=["volume_share"]).copy()
    sub = sub[sub["customers"].notna() | (sub["segment"] == "incumbent")]
    if sub.empty:
        ax.text(0.5, 0.5, "no coverage", ha="center"); return
    # account_share over firms that disclose a count (illustrative within-set)
    cov = sub.dropna(subset=["customers"])
    tot_cust = cov["customers"].sum()
    sub["account_share"] = sub["customers"] / tot_cust if tot_cust else np.nan
    sub = sub.sort_values("volume_share", ascending=False)

    y = np.arange(len(sub)); h = 0.38
    ax.barh(y + h/2, 100 * sub["volume_share"], height=h, color=[_seg_color(s) for s in sub.segment],
            alpha=0.95, label="volume share")
    ax.barh(y - h/2, 100 * sub["account_share"], height=h, color=[_seg_color(s) for s in sub.segment],
            alpha=0.45, hatch="//", label="account share")
    ax.set_yticks(y); ax.set_yticklabels(sub["firm_key"])
    ax.invert_yaxis()
    ax.set_xlabel("share of covered set (%)")
    ax.set_title(f"A. Volume share vs. account share — {period}", fontsize=10, loc="left")
    # annotate R$/customer where available
    for yi, (_, r) in zip(y, sub.iterrows()):
        if pd.notna(r.get("deposits_per_customer_brl")):
            ax.text(0.5, yi, f"  R$ {r['deposits_per_customer_brl']:,.0f}/cust",
                    va="center", fontsize=7, color="#444")
    ax.legend(loc="lower right", fontsize=7, framealpha=0.9)


def panel_b(ax, df: pd.DataFrame) -> None:
    d = df.dropna(subset=["deposits_per_customer_brl"]).copy()
    if d.empty:
        ax.text(0.5, 0.5, "no deposits-per-customer coverage", ha="center"); return
    d["t"] = d["year"] + (d["quarter"] - 1) / 4
    for fk, g in d.sort_values("t").groupby("firm_key"):
        seg = g["segment"].iloc[0]
        ax.plot(g["t"], g["deposits_per_customer_brl"], marker="o", ms=3,
                color=_seg_color(seg), alpha=0.85,
                lw=1.8 if seg == "incumbent" else 1.2,
                label=f"{fk} ({'B' if seg=='incumbent' else 'D'})")
    ax.set_yscale("log")
    ax.set_ylabel("deposits per customer (R$, log)")
    ax.set_xlabel("year")
    ax.set_title("B. Deposits per customer over time (B vs D)", fontsize=10, loc="left")
    ax.legend(fontsize=6, ncol=2, framealpha=0.9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", type=str, default=None)
    args = ap.parse_args()

    df = load()
    n_cust = df["customers"].notna().sum()
    n_dpc = df["deposits_per_customer_brl"].notna().sum() if "deposits_per_customer_brl" in df else 0
    log.info(f"join table: {len(df)} rows; {n_cust} with a customer count; "
             f"{n_dpc} with deposits-per-customer.")
    period = pick_period(df, args.period)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.6))
    panel_a(axA, df, period)
    panel_b(axB, df)
    fig.suptitle("Brazilian deposits: extensive (accounts) vs. intensive (volume) margin",
                 fontsize=12, y=1.02)
    fig.text(0.5, -0.04,
             "Digital (D, orange) hold large account/customer bases but small deposit-volume shares; "
             "incumbents (B, blue) hold large volume shares. Account share normalised over firms that "
             "disclose customer counts. Source: SEC EDGAR / CVM / MELI disclosures joined to the BCB deposit panel.",
             ha="center", fontsize=7.5, wrap=True)
    fig.tight_layout()
    os.makedirs(DESC_DIR, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    log.info(f"wrote {OUT_PNG}")
    log.info(f"wrote {OUT_PDF}")

    # Also drop a copy next to the paper (same folder as V_Main.tex) so it can be
    # \includegraphics'd directly from the draft.
    drafts = os.path.normpath(os.path.join(BASE, "Drafts", "Deposit Competition"))
    if os.path.isdir(drafts):
        fig.savefig(os.path.join(drafts, "account_vs_volume.png"), dpi=160, bbox_inches="tight")
        fig.savefig(os.path.join(drafts, "account_vs_volume.pdf"), bbox_inches="tight")
        log.info(f"copied figure -> {drafts}")
    else:
        log.warning(f"draft folder not found, skipped paper copy: {drafts}")


if __name__ == "__main__":
    main()

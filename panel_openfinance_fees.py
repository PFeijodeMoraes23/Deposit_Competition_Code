"""
panel_openfinance_fees.py
============================
Roll the Open Finance Brazil (OFB) open-data fee panel up to the model's firm —
the prudential conglomerate (CodConglomeradoPrudencial) — and build the
cross-bank comparable measure.

Input   scrape_openfinance_fees.py → BCB/Tarifas/processed/openfinance_fees_panel_long.csv
Output  BCB/Tarifas/processed/
          openfinance_fees_conglomerate_long.csv   cong × ref-period × service (per-event fees)
          openfinance_fees_conglomerate_wide.csv    cong × ref-period, one row (core basket + annual cost)
          openfinance_fees_vs_listed.csv            OFB actual vs BCB listed-max comparability check

Why a separate rollup
---------------------
The OFB scraper keys on `cnpj8` (8-digit CNPJ root). The demand/cost model's firm
is the *prudential conglomerate*, so subsidiaries must be aggregated to it, exactly
as panel_9 does for COSIF. This module reuses panel_9's conglomerate map and its
deposit-weight source, and exposes `build_conglomerate_panel()` so panel_9 can pull
OFB in as a fourth fee source later if desired.

The comparable measure
-----------------------
Primitive (per bank × service): the CUSTOMER-WEIGHTED average price
(`price_weighted_avg` from scrape_19 — each quartile band weighted by its share of
customers). Subsidiary → conglomerate aggregation of that primitive is reported
three ways so the weighting choice is auditable:
  _wmean  deposit-weighted mean  (primary; the conglomerate faces one representative
                                  price weighted by where its deposits are)
  _mean   simple mean            (robustness — deposit-weighting can annihilate tiny
                                  digital-bank subsidiaries)
  _max    maximum                (robustness / worst-case)

Coherence with the other fee sources
-------------------------------------
  BCB listed  (panel_9 listed_fee_*)  = listed CEILING     (BRL/event)
  OFB         (this module)           = ACTUAL charged     (BRL/event, customer-weighted)
  COSIF       (panel_9 cosif_fee_*)   = REALIZED revenue / deposits (dimensionless rate)
OFB and BCB share unit AND service basket, so `openfinance_fees_vs_listed.csv`
(ofb_actual / listed_ceiling, a "discount from ceiling") is a direct, assumption-free
comparability check. `ofb_acct_cost_ann_brl` is the annual BRL basket cost — a SUM
OVER EVENTS (price × assumed frequency), dimensionally distinct from COSIF's
`fee_ann_pp = ratio×4×100` (a linear per-quarter→year rate) and from the compound
`(1+q)^4-1` rate annualizer. It is the OFB analog of the annualized spread, NOT a rate.

TEMPORAL SCOPE
--------------
OFB frequency data exists only monthly since 2021 (see scrape_19). Each row is tagged
with the reference period (`ref_year`,`ref_quarter`) = the month BEFORE the scrape date
(the data references the previous month). Because the estimation window is 2016-2024
and the current scrapes are 2026, a merge onto market_panel matches zero rows today;
overlap with the estimation window is only recoverable via the Wayback reconstruction
(scrape_25). This module therefore ships a STANDALONE conglomerate dataset; it does not
patch market_panel (that stays a one-line call away via build_conglomerate_panel()).

Usage
-----
  python panel_openfinance_fees.py
  python panel_openfinance_fees.py --customer-type PF   # default PF
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

from utils import paths
# Reuse panel_9's conglomerate map, deposit-weight loader, listed-price loader.
import panel_fee_merge as p9

TARIF_DIR = paths.TARIFAS_PROC
OFB_LONG_CSV = TARIF_DIR / "openfinance_fees_panel_long.csv"

OUT_LONG = TARIF_DIR / "openfinance_fees_conglomerate_long.csv"
OUT_WIDE = TARIF_DIR / "openfinance_fees_conglomerate_wide.csv"
OUT_VS_LISTED = TARIF_DIR / "openfinance_fees_vs_listed.csv"

# ---------------------------------------------------------------------------
# Core account-fee basket.
#   ofb_code : (short column suffix, illustrative annual frequency for a
#               representative retail account holder)
# The frequencies are transparent, documented assumptions used ONLY to build the
# summary `ofb_acct_cost_ann_brl`; every per-service price is emitted separately so
# the basket can be reweighted downstream. Account opening (CADASTRO) is a one-off
# and carries freq 0 (excluded from the recurring annual flow).
# ---------------------------------------------------------------------------
CORE_BASKET: dict[str, tuple[str, float]] = {
    "CADASTRO":            ("acct_reg",         0.0),
    "SAQUE_TERMINAL":      ("atm_withdrawal",   24.0),
    "SAQUE_PESSOAL":       ("withdrawal_teller", 6.0),
    "EXTRATO_MES_E":       ("statement_elec",   12.0),
    "EXTRATO_MES_P":       ("statement_teller",  0.0),
    "TED_INTERNET":        ("ted_internet",     12.0),
    "TED_PESSOAL":         ("ted_teller",        0.0),
    "2_VIA_CARTAO_DEBITO": ("card_2nd_debit",    0.5),
    "TRANSF_RECURSO_E":    ("transfer_own_elec", 0.0),
    "SMS":                 ("sms",               0.0),
}

# Crosswalk from BCB priority-service short name (panel_9.PRIORITY_SERVICES values,
# → listed_fee_<name>_pf) to the OFB code(s) that measure the same service.
# OFB splits the "extrato" service by channel; we average across the channels.
LISTED_CROSSWALK: dict[str, list[str]] = {
    "acct_reg":           ["CADASTRO"],
    "card_2nd_debit":     ["2_VIA_CARTAO_DEBITO"],
    "statement":          ["EXTRATO_MES_P", "EXTRATO_MES_E"],
    "atm_withdrawal":     ["SAQUE_TERMINAL"],
    "cheque_stop":        ["SUSTACAO_REVOGACAO"],
    "cheque_countermand": ["SUSTACAO_REVOGACAO"],
}


# ---------------------------------------------------------------------------
# Load + normalize the OFB long panel (robust to the pre-multifamily schema)
# ---------------------------------------------------------------------------

def load_ofb_long(customer_type: str = "PF") -> pd.DataFrame:
    if not OFB_LONG_CSV.exists():
        raise FileNotFoundError(f"OFB long panel not found at {OFB_LONG_CSV} "
                                f"(run scrape_openfinance_fees.py first)")
    df = pd.read_csv(OFB_LONG_CSV, low_memory=False, dtype={"cnpj8": str})
    df["cnpj8"] = df["cnpj8"].astype(str).str.zfill(8)

    # Back-compat: rows written by the pre-multifamily scraper (accounts only,
    # per-event fees) lack these columns.
    if "product" not in df.columns:
        df["product"] = "accounts"
    else:
        df["product"] = df["product"].fillna("accounts")
    if "price_kind" not in df.columns:
        df["price_kind"] = "per_event_brl"
    else:
        df["price_kind"] = df["price_kind"].fillna("per_event_brl")

    df = df[(df["product"] == "accounts") &
            (df["price_kind"] == "per_event_brl") &
            (df["customer_type"] == customer_type)].copy()

    # Reference period = the month BEFORE the scrape (data references prior month).
    dc = pd.to_datetime(df["data_coleta"], errors="coerce")
    ref = dc - pd.DateOffset(months=1)
    df["ref_year"] = ref.dt.year
    df["ref_quarter"] = ((ref.dt.month - 1) // 3 + 1)
    df = df.dropna(subset=["ref_year"])
    df["ref_year"] = df["ref_year"].astype(int)
    df["ref_quarter"] = df["ref_quarter"].astype(int)

    df["price_weighted_avg"] = pd.to_numeric(df["price_weighted_avg"], errors="coerce")
    log.info("OFB long (%s, accounts, per-event): %d rows | %d cnpj8 | dates %s",
             customer_type, len(df), df["cnpj8"].nunique(),
             sorted(df["data_coleta"].unique()))
    return df


# ---------------------------------------------------------------------------
# Deposit weights (latest dep_total per cnpj8, from the COSIF institution panel)
# ---------------------------------------------------------------------------

def deposit_weights() -> pd.DataFrame:
    """Returns cnpj8 → dep_weight (latest available EOM total deposit stock)."""
    try:
        cosif = p9.load_cosif_institution()
    except FileNotFoundError:
        log.warning("COSIF institution panel absent — deposit weights unavailable; "
                    "wmean will fall back to simple mean.")
        return pd.DataFrame(columns=["cnpj8", "dep_weight"])
    cosif = cosif.copy()
    cosif["cnpj8"] = cosif["cnpj"].astype(str).str.zfill(8)
    cosif["data_base"] = cosif["data_base"].astype(str)
    dep_col = "dep_total" if "dep_total" in cosif.columns else None
    if dep_col is None:
        return pd.DataFrame(columns=["cnpj8", "dep_weight"])
    latest = (cosif.sort_values("data_base")
              .groupby("cnpj8", as_index=False)
              .agg(dep_weight=(dep_col, "last")))
    latest["dep_weight"] = pd.to_numeric(latest["dep_weight"], errors="coerce")
    latest.loc[latest["dep_weight"] <= 0, "dep_weight"] = np.nan
    return latest


# ---------------------------------------------------------------------------
# Aggregate cnpj8 → conglomerate
# ---------------------------------------------------------------------------

def build_service_conglomerate(customer_type: str = "PF") -> pd.DataFrame:
    """
    Long conglomerate panel: one row per
    (cod_cong_prudencial, ref_year, ref_quarter, data_coleta, service_code).

    The deposit-weighted mean is computed vectorized (group sum of price*weight
    over group sum of weight) so it works across pandas versions — no groupby.apply.
    """
    ofb = load_ofb_long(customer_type)
    cmap = p9.build_cnpj_cong_map()               # cnpj (8) -> cod_cong_prudencial
    cmap = cmap.rename(columns={"cnpj": "cnpj8"})
    ofb = ofb.merge(cmap, on="cnpj8", how="left")
    # Fallback for payment institutions, which the Prudential report does not cover:
    # 'CNPJ_<int>', the key the market panel files them under (see
    # panel_fee_merge.cnpj_fallback_key).
    ofb["cod_cong_prudencial"] = (
        ofb["cod_cong_prudencial"]
        .fillna(p9.cnpj_fallback_key(ofb["cnpj8"]))
        .astype(str).str.strip()
    )

    ofb = ofb.merge(deposit_weights(), on="cnpj8", how="left")

    keys = ["cod_cong_prudencial", "ref_year", "ref_quarter", "data_coleta",
            "customer_type", "service_code", "service_name"]

    v = pd.to_numeric(ofb["price_weighted_avg"], errors="coerce")
    w = pd.to_numeric(ofb["dep_weight"], errors="coerce")
    wmask = v.notna() & w.notna() & (w > 0)
    ofb["_v"] = v
    ofb["_wv"] = np.where(wmask, v * w, np.nan)
    ofb["_w"] = np.where(wmask, w, np.nan)
    ofb["_has_w"] = w.notna().astype(float)

    grouped = ofb.groupby(keys, as_index=False, sort=False)
    out = grouped.agg(
        _sum_wv=("_wv", "sum"),
        _sum_w=("_w", "sum"),
        ofb_price_mean=("_v", "mean"),
        ofb_price_max=("_v", "max"),
        ofb_n_subsidiaries=("cnpj8", "nunique"),
        ofb_dep_weight_covered=("_has_w", "mean"),
    )
    out["ofb_price_wmean"] = np.where(
        out["_sum_w"] > 0, out["_sum_wv"] / out["_sum_w"], out["ofb_price_mean"]
    )
    out = out.drop(columns=["_sum_wv", "_sum_w"])
    log.info("Conglomerate service panel: %d rows | %d conglomerates",
             len(out), out["cod_cong_prudencial"].nunique())
    return out


# ---------------------------------------------------------------------------
# Wide conglomerate panel (core basket + annual cost) — the public API
# ---------------------------------------------------------------------------

def build_conglomerate_panel(customer_type: str = "PF") -> pd.DataFrame:
    """
    Wide conglomerate panel: one row per (cod_cong_prudencial, ref_year,
    ref_quarter, data_coleta), with:
      ofb_fee_<name>_{wmean,mean,max}   core-basket per-service prices
      ofb_acct_cost_ann_brl             Σ price_wmean × assumed annual frequency
      ofb_n_services_covered            core-basket services present
    This is what panel_9 would import to add OFB as a fourth fee source.
    """
    svc = build_service_conglomerate(customer_type)
    idx = ["cod_cong_prudencial", "ref_year", "ref_quarter", "data_coleta", "customer_type"]

    basket = svc[svc["service_code"].isin(CORE_BASKET)].copy()
    basket["suffix"] = basket["service_code"].map(lambda c: CORE_BASKET[c][0])

    frames = []
    for stat in ("wmean", "mean", "max"):
        w = basket.pivot_table(index=idx, columns="suffix",
                               values=f"ofb_price_{stat}", aggfunc="mean")
        w.columns = [f"ofb_fee_{c}_{stat}" for c in w.columns]
        frames.append(w)
    wide = pd.concat(frames, axis=1).reset_index()

    # Annual basket cost from the deposit-weighted per-service prices.
    freq = {CORE_BASKET[c][0]: CORE_BASKET[c][1] for c in CORE_BASKET}
    cost = np.zeros(len(wide))
    n_cov = np.zeros(len(wide))
    for suffix, f in freq.items():
        col = f"ofb_fee_{suffix}_wmean"
        if col in wide.columns and f > 0:
            v = pd.to_numeric(wide[col], errors="coerce").fillna(0.0)
            cost = cost + v.to_numpy() * f
        base_col = f"ofb_fee_{suffix}_wmean"
        if base_col in wide.columns:
            n_cov = n_cov + pd.to_numeric(wide[base_col], errors="coerce").notna().to_numpy()
    wide["ofb_acct_cost_ann_brl"] = cost
    wide["ofb_n_services_covered"] = n_cov.astype(int)
    wide["ofb_fee_valid"] = (wide["ref_year"] >= 2021).astype(int)

    log.info("Wide conglomerate panel: %d rows × %d cols", len(wide), len(wide.columns))
    return wide


# ---------------------------------------------------------------------------
# Cross-source comparability: OFB actual vs BCB listed ceiling
# ---------------------------------------------------------------------------

def build_vs_listed(svc: pd.DataFrame) -> pd.DataFrame:
    """
    For each conglomerate, compare the OFB customer-weighted actual price against
    the BCB listed maximum (ceiling), on the priority-service basket. Uses the
    LISTED_CROSSWALK to line up OFB string codes with BCB numeric priority services.
    Emits ofb_actual_<name>, listed_fee_<name>_pf, ofb_to_listed_ratio_<name>.
    """
    listed = p9.load_tarifa_listed_prices()   # cong-level, listed_fee_<name>_pf
    if listed.empty:
        log.warning("No BCB listed prices — skipping vs-listed comparison.")
        return pd.DataFrame()

    # Collapse OFB service panel to one actual price per (cong, listed-name),
    # averaging the deposit-weighted price across the crosswalked OFB codes and
    # across scrape dates (the listed ceiling is ~static).
    rows = []
    for name, ofb_codes in LISTED_CROSSWALK.items():
        sub = svc[svc["service_code"].isin(ofb_codes)]
        if sub.empty:
            continue
        g = (sub.groupby("cod_cong_prudencial", as_index=False)
             .agg(ofb_actual=("ofb_price_wmean", "mean")))
        g["listed_name"] = name
        rows.append(g)
    if not rows:
        return pd.DataFrame()
    actual = pd.concat(rows, ignore_index=True)
    actual_w = actual.pivot_table(index="cod_cong_prudencial", columns="listed_name",
                                  values="ofb_actual", aggfunc="mean")
    actual_w.columns = [f"ofb_actual_{c}" for c in actual_w.columns]
    actual_w = actual_w.reset_index()

    listed = listed.rename(columns={"cod_cong_prudencial": "cod_cong_prudencial"})
    merged = actual_w.merge(listed, on="cod_cong_prudencial", how="inner")
    for name in LISTED_CROSSWALK:
        a, l = f"ofb_actual_{name}", f"listed_fee_{name}_pf"
        if a in merged.columns and l in merged.columns:
            with np.errstate(divide="ignore", invalid="ignore"):
                merged[f"ofb_to_listed_ratio_{name}"] = (
                    pd.to_numeric(merged[a], errors="coerce")
                    / pd.to_numeric(merged[l], errors="coerce").replace(0, np.nan)
                )
    log.info("vs-listed: %d conglomerates with both OFB and BCB listed prices", len(merged))
    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(customer_type: str = "PF") -> None:
    svc = build_service_conglomerate(customer_type)
    svc.to_csv(OUT_LONG, index=False)
    log.info("Saved: %s (%d rows)", OUT_LONG.name, len(svc))

    wide = build_conglomerate_panel(customer_type)
    wide.to_csv(OUT_WIDE, index=False)
    log.info("Saved: %s (%d rows × %d cols)", OUT_WIDE.name, len(wide), len(wide.columns))

    vs = build_vs_listed(svc)
    if not vs.empty:
        vs.to_csv(OUT_VS_LISTED, index=False)
        log.info("Saved: %s (%d rows)", OUT_VS_LISTED.name, len(vs))
        ratio_cols = [c for c in vs.columns if c.startswith("ofb_to_listed_ratio_")]
        if ratio_cols:
            desc = vs[ratio_cols].describe().loc[["count", "50%", "mean"]]
            log.info("OFB/listed ratio summary (should sit in ~[0,1]):\n%s", desc.to_string())

    log.info("=== Done ===")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Roll OFB open-data fees up to the prudential conglomerate")
    ap.add_argument("--customer-type", default="PF", choices=["PF", "PJ"],
                    help="Person type to roll up (default PF)")
    args = ap.parse_args()
    main(customer_type=args.customer_type)

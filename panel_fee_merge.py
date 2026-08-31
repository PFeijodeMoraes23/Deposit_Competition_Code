"""
panel_fee_merge.py
=======================
Merge all fee-price columns onto the market panel.

Three fee sources
-----------------
1. COSIF realized revenue ratios   (scrape_cosif_service_fees.py)
   - Quarterly, conglomerate-level  |  2013-Q1 → 2026-Q1
   - 5 fee/deposit ratio columns    |  cosif_fee_ratio_*
   - NOTE: 2025+ data has a COSIF account reclassification artifact
     (mandatory new plan from Jan 2025). Set cosif_fee_valid=0 for year>=2025.
   - NOTE: for payment institutions the 717xxx accounts measure card interchange,
     not deposit-account tariffs; --patch-market NaNs their ratios and marks them
     with cosif_fee_ip_masked=1.

2. BCB Tarifas listed maximum prices  (scrape_bcb_tariff_vigencia.py)
   - Static snapshot, no time variation  |  ~1,090 conglomerates
   - Key priority services (PF)          |  listed_fee_<name>_pf columns
   - Useful as cross-sectional instruments for BLP identification
   - Rolled up here from tariff_vigencia_history.csv, not from the scraper's own
     conglomerate summary, so every fee source shares one CNPJ->conglomerate map

3. DataVigencia price-stickiness instrument  (scrape_bcb_tariff_vigencia.py)
   - Time-varying, conglomerate × quarter  |  2013-Q1 → latest
   - Mean years since last tariff change across BCB priority services
   - tarifa_stickiness_yrs: grows until a price change, resets to 0 at change

Output
------
  BCB/Tarifas/processed/cosif_fee_quarterly_conglomerate.csv  (COSIF only)
  BCB/Egan_et_al_2025_Rep/processed/market_panel_with_fees.csv  (all sources)

Usage
-----
  python panel_fee_merge.py
  python panel_fee_merge.py --patch-market

Note
----
  Estimation scripts read market_panel.csv, NOT this file (changed 2026-08-04). The fee columns
  it adds are used by no estimator: they appear in no .jl, in neither X_COLS nor D_COLS, and in
  no sleepiness design — they reached the demand parquets only via IV_FEE, which is skipped when
  absent. The old "prefer with_fees if it exists" rule meant a stale copy of this file silently
  shadowed a freshly rebuilt market_panel, which is exactly what happened between 2026-07-28 and
  2026-08-03. Set USE_FEE_PANEL=1 to opt back in (e.g. to estimate a fee specification); see
  utils/paths.market_panel_csv. This script remains the way to BUILD the fee panel.
"""

from __future__ import annotations

import argparse
import logging
import os
import unicodedata
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
from utils import paths
_REPO           = Path(__file__).resolve().parents[2]
PANEL_DIR       = paths.PROCESSED
TARIF_DIR       = paths.TARIFAS_PROC

COSIF_INST_PARQUET  = TARIF_DIR / "cosif_service_fees_institution.parquet"
COSIF_INST_CSV      = TARIF_DIR / "cosif_service_fees_institution.csv"
VIGENCIA_HIST_CSV   = TARIF_DIR / "tariff_vigencia_history.csv"
MARKET_PANEL_CSV    = PANEL_DIR / "market_panel.csv"
OUTPUT_CSV          = TARIF_DIR / "cosif_fee_quarterly_conglomerate.csv"

# COSIF fee ratio columns (from scrape_12)
FEE_COLS = [
    "fee_ratio_demand",
    "fee_ratio_savings",
    "fee_ratio_time",
    "fee_ratio_all",
    "fee_ratio_total_deposits",
]

# BCB priority services (Resolucao CMN 3.919/2010 basket + ATM)
PRIORITY_SERVICES = {
    1101: "acct_reg",        # Cadastro para inicio de relacionamento
    1201: "card_2nd_debit",  # 2a via cartao debito
    1203: "cheque_stop",     # Contra-ordem de cheque
    1204: "cheque_countermand",  # Oposicao cheque
    1213: "statement",       # Extrato periodo anterior
    1501: "atm_withdrawal",  # Saque em terminal (ATM)
}

# Year of the COSIF structural break; rows with year >= this value get cosif_fee_valid=0
COSIF_BREAK_YEAR = 2025

# ---------------------------------------------------------------------------
# COSIF fee/deposit ratio plausibility band.
#
# A quarterly service-fee / deposit ratio is economically O(0.001-0.1): fees are
# at most a few percent of the deposit base per quarter.  The clean 2016-2018
# distribution confirms this (per-year MEDIAN ~0.009 for fee_ratio_all, ~0.05
# for the smallest-denominator fee_ratio_demand; p95 <= 0.13 and <= 0.57
# respectively).  The MEDIANS are stable and sane in EVERY year -- the cross-year
# "break" (means of 100-3000, maxima up to 1e10) is a pure SMALL-DENOMINATOR
# explosion, not a units mismatch or an account-definition change:
#
#   * a bank whose demand/savings/time deposits are a near-zero sliver of its
#     real balance sheet (e.g. R$142 of classified demand deposits against
#     R$21bn total) -- the dep_sub sub-sum divides fee revenue by that sliver;
#   * a payment institution with a token deposit stock (R$0.02, R$380) against
#     large fee revenue.
#
# Such values are not fee ratios and are treated as MISSING, exactly as
# panel_3_master_panel_build NaNs rate_a4 above RATE_A4_MAX_MULT x Selic and
# cosif_implicit_rate above 0.5.  Negative ratios (from svc_revenue_inc within-
# semester reversals) are likewise implausible for a fee ratio and NaN'd.
# The ceiling is env-overridable (mirrors panel_3's RATE_A4_MAX_MULT); the
# default 1.0 = "quarterly fees cannot exceed 100% of the deposit base" leaves
# the entire clean central distribution (p95 <= 0.57) untouched while removing
# every explosion.  A cosif_fee_guarded flag marks affected cong-quarters,
# mirroring panel_3's rate_a4_guarded.
COSIF_FEE_RATIO_MAX = float(os.environ.get("COSIF_FEE_RATIO_MAX", 1.0))


# ---------------------------------------------------------------------------
# 1. Load COSIF institution panel
# ---------------------------------------------------------------------------

def load_cosif_institution() -> pd.DataFrame:
    if COSIF_INST_PARQUET.exists():
        try:
            import polars as pl
            df = pl.read_parquet(str(COSIF_INST_PARQUET)).to_pandas()
            log.info("Loaded COSIF parquet: %d rows", len(df))
            return df
        except Exception:
            pass
    if COSIF_INST_CSV.exists():
        df = pd.read_csv(COSIF_INST_CSV, dtype={"cnpj": str}, low_memory=False)
        log.info("Loaded COSIF CSV: %d rows", len(df))
        return df
    raise FileNotFoundError(f"COSIF institution panel not found at {COSIF_INST_PARQUET}")


# ---------------------------------------------------------------------------
# 2. Build CNPJ -> CodConglomeradoPrudencial map from IF Data
# ---------------------------------------------------------------------------

def cnpj_fallback_key(cnpj: pd.Series) -> pd.Series:
    """
    Conglomerate key for a CNPJ the IF Data List files never place in a prudential
    conglomerate -- payment institutions above all, since the Prudential report has
    none of them.

    ``CNPJ_<int>`` with no zero padding is the market panel's own convention for
    these firms (panel_deposits.build_estban_panel, panel_market.load_digital_
    conglomerates), so the fee sources land in the same key space.  The bare
    zero-padded CNPJ this used to produce ("18236120") could never equal the panel's
    ("CNPJ_18236120"), which dropped Nubank, Mercado Pago and PicPay -- the D side of
    the B/D split -- out of every fee merge without a warning.
    """
    return "CNPJ_" + cnpj.astype(str).str.strip().astype(int).astype(str)


def load_nonbank_ip_conglomerates() -> set:
    """
    Prudential conglomerates that hold a payment-institution licence and NO banking or
    credit-union charter, as canonical CodConglomeradoPrudencial codes.

    Why not has_ip
    --------------
    `has_ip` (panel_bank_chars) is 1 when ANY member institution of the conglomerate is
    ever a payment institution, so it is 1 for Bradesco, Itau and Banco do Brasil, which
    all own one: 196 conglomerates and 71% of market-panel rows.  Masking on it would
    delete the fee ratios of precisely the institutions whose ratios are well measured.

    What the mask has to separate is licence type, and the List files carry it in
    SegmentoTb.  A conglomerate whose members include a bank ("Banco Multiplo", "Banco
    Comercial", "Caixa Economica") or a credit union takes deposits, so its 717xxx
    revenue is deposit-account tariffs over a deposit base.  A conglomerate that holds
    only payment, credit and brokerage licences does not: its 717xxx is card interchange
    and merchant processing, and its "deposits" are settlement float.  Nu Pagamentos,
    Mercado Pago and Neon fall on that side; Inter, C6, PicPay and PagSeguro do not,
    each holding a Banco Multiplo, which matches service_fees_notes.md section 3 on
    Inter and C6 being validly measured.

    Read pooled over every List snapshot, mirroring how panel_bank_chars builds has_ip.
    A firm that acquires a bank charter mid-sample (PicPay Bank, 2023) is therefore
    unmasked over its whole history, including the years it was an IP alone.
    """
    from panel_deposits import build_conglomerate_canonical_map

    def _norm(s: str) -> str:
        return (unicodedata.normalize("NFKD", str(s))
                .encode("ascii", "ignore").decode("ascii").lower())

    files = sorted(paths.IF_DATA_LIST.glob("IF_DATA_List_*.csv"))
    files = [f for f in files if f.stat().st_size > 500]
    if not files:
        log.warning("No IF Data List files — payment-institution mask will be empty.")
        return set()

    frames = []
    for f in files:
        try:
            d = pd.read_csv(f, dtype=str, encoding="latin-1", low_memory=False)
        except Exception as exc:  # noqa: BLE001 - one unreadable snapshot must not abort
            log.warning("Could not read List file %s: %s", f.name, exc)
            continue
        d.columns = [c.strip() for c in d.columns]
        if not {"CodInst", "CodConglomeradoPrudencial", "SegmentoTb"}.issubset(d.columns):
            continue
        if {"Td", "Situacao"}.issubset(d.columns):
            d = d[(d["Td"] == "I") & (d["Situacao"] == "A")]
        cong = d["CodConglomeradoPrudencial"].astype(str)
        seg = d["SegmentoTb"].map(_norm)
        frames.append(pd.DataFrame({
            "cong": np.where(cong.isin(["nan", "null", "None", ""]),
                             d["CodInst"].astype(str), cong),
            "is_ip": (seg.str.contains("institui", na=False)
                      & seg.str.contains("pagamento", na=False)),
            "is_deposit_taker": (seg.str.contains("banco", na=False)
                                 | seg.str.contains("caixa", na=False)
                                 | seg.str.contains("cooperativa", na=False)),
        }))
    if not frames:
        log.warning("No usable List snapshots — payment-institution mask will be empty.")
        return set()

    members = pd.concat(frames, ignore_index=True)
    members["cong"] = members["cong"].astype(str).str.replace(r"\.0$", "", regex=True)
    g = members.groupby("cong").agg(any_ip=("is_ip", "max"),
                                    any_dep=("is_deposit_taker", "max"))
    codes = set(g.index[g["any_ip"] & ~g["any_dep"]])

    # The panel keys firms by their canonical code, so the mask has to as well.
    canonical_code, _ = build_conglomerate_canonical_map()
    codes = {canonical_code.get(c, c) for c in codes}
    log.info("Payment-institution mask: %d conglomerates hold an IP licence and no "
             "bank/credit-union charter", len(codes))
    return codes


def build_cnpj_cong_map() -> pd.DataFrame:
    """
    Returns DataFrame: cnpj (8-digit str), cod_cong_prudencial (str).

    The map comes from ``panel_deposits.build_cnpj_conglomerate_map``, which is the
    same lookup that keys the market panel: it scans EVERY IF Data List quarter and
    resolves each code through ``build_conglomerate_canonical_map``, so a firm whose
    prudential code changed mid-sample (Banco Gerador C0081809 -> Agibank C0083694 at
    the 2016-Q3 acquisition, Indusval -> Pleno in 2025-Q3, BBM -> Bocom BBM in 2017)
    carries ONE code across its whole history.  Reading a single latest snapshot
    instead, as this function used to, labelled such a firm with its current code
    while the market panel held the canonical one, so every fee quarter before the
    rename failed to join, and firms that exited before the last snapshot joined
    nowhere at all.
    """
    from panel_deposits import build_cnpj_conglomerate_map

    mapping = build_cnpj_conglomerate_map()
    if not mapping:
        log.warning("CNPJ→conglomerate map is empty — fee sources will not join.")
        return pd.DataFrame(columns=["cnpj", "cod_cong_prudencial"])

    cmap = pd.DataFrame({
        "cnpj": [str(int(k)).zfill(8) for k in mapping],
        "cod_cong_prudencial": [str(v[0]).strip() if v[0] is not None else ""
                                for v in mapping.values()],
    })
    cmap = cmap[~cmap["cod_cong_prudencial"].isin(["", "nan", "None"])]
    cmap = cmap.drop_duplicates(subset="cnpj", keep="first").reset_index(drop=True)
    log.info("Conglomerate map: %d CNPJs, %d conglomerates (canonical codes)",
             len(cmap), cmap["cod_cong_prudencial"].nunique())
    return cmap


# ---------------------------------------------------------------------------
# 3. COSIF: aggregate to conglomerate x quarter
# ---------------------------------------------------------------------------

def _to_quarter(data_base_series: pd.Series) -> tuple[pd.Series, pd.Series]:
    dbs   = data_base_series.astype(str).str.strip().str[:6]
    year  = dbs.str[:4].astype(int)
    month = dbs.str[4:6].astype(int)
    return year, ((month - 1) // 3 + 1).astype(int)


def aggregate_cosif_to_conglomerate_quarter(
    cosif: pd.DataFrame,
    cmap:  pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate COSIF institution panel to conglomerate × quarter.

    Uses monthly revenue increments (not cumulative/6) to compute correct
    quarterly fee ratios: Q_ratio = sum(3 monthly increments) / EOM deposit stock.
    If the parquet pre-dates the increment fix, recomputes increments on the fly.
    """
    cosif = cosif.copy()
    cosif["cnpj"] = cosif["cnpj"].astype(str).str.zfill(8)
    cosif = cosif.merge(cmap, on="cnpj", how="left")
    cosif["cod_cong_prudencial"] = cosif["cod_cong_prudencial"].fillna(cnpj_fallback_key(cosif["cnpj"]))
    cosif["year"], cosif["quarter"] = _to_quarter(cosif["data_base"])
    cosif["month"] = cosif["data_base"].astype(str).str[4:6].astype(int)

    # ── Ensure svc_revenue_inc is available ────────────────────────────────
    # scrape_12 >= fix: column already present.  scrape_12 < fix: recompute.
    if "svc_revenue_inc" not in cosif.columns:
        log.info("svc_revenue_inc not in parquet — computing from cumulative svc_revenue")
        cosif["half_yr"] = np.where(cosif["month"] <= 6, 1, 2)
        cosif = cosif.sort_values(["cnpj", "year", "half_yr", "month"])
        for col, out in [("svc_revenue", "svc_revenue_inc"),
                         ("svc_revenue_pf", "svc_revenue_pf_inc"),
                         ("svc_revenue_pj", "svc_revenue_pj_inc")]:
            if col in cosif.columns:
                d = cosif.groupby(["cnpj", "year", "half_yr"])[col].diff()
                cosif[out] = d.fillna(cosif[col])
            else:
                cosif[out] = np.nan
    else:
        log.info("Using pre-computed svc_revenue_inc from parquet")
        for col, out in [("svc_revenue_pf", "svc_revenue_pf_inc"),
                         ("svc_revenue_pj", "svc_revenue_pj_inc")]:
            if out not in cosif.columns and col in cosif.columns:
                cosif["half_yr"] = cosif.get("half_yr",
                                             np.where(cosif["month"] <= 6, 1, 2))
                d = cosif.groupby(["cnpj", "year", "half_yr"])[col].diff()
                cosif[out] = d.fillna(cosif[col])

    # ── Quarterly aggregation ───────────────────────────────────────────────
    # A conglomerate-quarter group holds every member institution in each of the three
    # months, so the two sides of the ratio have to be aggregated differently.
    keys = ["cod_cong_prudencial", "year", "quarter"]
    cosif = cosif.sort_values(keys + ["cnpj", "month"])

    # Revenue is a FLOW: sum the monthly increments over every member and month.
    q_rev = (
        cosif.groupby(keys, as_index=False)
        .agg(
            svc_revenue_q    =("svc_revenue_inc",    "sum"),
            svc_revenue_pf_q =("svc_revenue_pf_inc", "sum"),
            svc_revenue_pj_q =("svc_revenue_pj_inc", "sum"),
        )
    )

    # Deposits are a STOCK: take each member's latest balance reported in the quarter
    # (groupby.last skips NaN, so a member that files only in October still counts),
    # then sum across members.
    #
    # Aggregating these with "last" over the whole group -- as this did until 2026-08-31
    # -- returned ONE arbitrary member's balance, whichever sorted last, against a
    # numerator summed over all of them.  For Itau that member was ITAUVEST DTVM and for
    # Bradesco BRAM DTVM, neither of which takes deposits: the denominator came out NaN
    # and the conglomerate lost fee_ratio_total_deposits entirely (Itau's true 2019-Q4
    # figure is R$449bn, Bradesco's R$413bn).  Where the trailing member instead held a
    # small non-zero balance the ratio exploded and COSIF_FEE_RATIO_MAX then discarded
    # it as an outlier, which is why a third to two thirds of B-type cong-quarters were
    # flagged as guarded.
    dep_cols = ["dep_demand", "dep_savings", "dep_time", "dep_total"]
    member_eom = cosif.groupby(keys + ["cnpj"], as_index=False)[dep_cols].last()
    q_dep = member_eom.groupby(keys, as_index=False)[dep_cols].sum(min_count=1)
    q_rev = q_rev.merge(q_dep, on=keys, how="left")

    # Step 2: compute quarterly fee ratios
    dep_sub = q_rev[["dep_demand", "dep_savings", "dep_time"]].sum(axis=1, min_count=1)
    dep_sub = dep_sub.where(dep_sub > 0, other=q_rev["dep_total"])

    q_rev["fee_ratio_demand"]         = q_rev["svc_revenue_q"] / q_rev["dep_demand"]
    q_rev["fee_ratio_savings"]        = q_rev["svc_revenue_q"] / q_rev["dep_savings"]
    q_rev["fee_ratio_time"]           = q_rev["svc_revenue_q"] / q_rev["dep_time"]
    q_rev["fee_ratio_all"]            = q_rev["svc_revenue_q"] / dep_sub
    q_rev["fee_ratio_total_deposits"] = q_rev["svc_revenue_q"] / q_rev["dep_total"]

    grp = q_rev[["cod_cong_prudencial", "year", "quarter"] + FEE_COLS].copy()
    for col in FEE_COLS:
        if col in grp.columns:
            grp[col] = grp[col].replace([np.inf, -np.inf], np.nan)

    # ── Plausibility guard: NaN small-denominator explosions ────────────────
    # (see COSIF_FEE_RATIO_MAX above.)  Each ratio column is guarded
    # independently so that a broken sub-classification (which blows up
    # fee_ratio_all / _demand) does not discard a bank's still-valid
    # fee_ratio_total_deposits.  cosif_fee_guarded = 1 marks any cong-quarter
    # where at least one fee ratio fell outside the sane [0, MAX] band.
    guarded = pd.Series(False, index=grp.index)
    for col in FEE_COLS:
        if col not in grp.columns:
            continue
        col_bad = ((grp[col] < 0) | (grp[col] > COSIF_FEE_RATIO_MAX)).fillna(False)
        guarded = guarded | col_bad
        grp.loc[col_bad, col] = np.nan
    grp["cosif_fee_guarded"] = guarded.astype(int)
    log.info(
        "COSIF fee plausibility guard (ratio < 0 or > %.3g): "
        "%d of %d cong-quarters flagged (%.2f%%) -> NaN",
        COSIF_FEE_RATIO_MAX, int(guarded.sum()), len(grp),
        100 * guarded.mean() if len(grp) else 0.0,
    )

    grp = grp.rename(columns={c: f"cosif_{c}" for c in FEE_COLS})
    log.info("COSIF panel: %d rows | %d conglomerates | %d year-quarters",
             len(grp), grp["cod_cong_prudencial"].nunique(),
             grp[["year", "quarter"]].drop_duplicates().__len__())
    return grp


# ---------------------------------------------------------------------------
# 4. BCB Tarifas: static listed max prices (cross-sectional instrument)
# ---------------------------------------------------------------------------

def load_tarifa_listed_prices(cmap: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Load BCB Tarifas listed maximum prices for key priority services (PF only).
    Returns a conglomerate-level DataFrame (static — no time variation).

    Columns added:
      listed_fee_<name>_pf  (median listed ceiling per conglomerate × service)

    Built from tariff_vigencia_history.csv, which is keyed on the institution's CNPJ,
    rather than from tariff_vigencia_summary.csv, which is already rolled up to a
    conglomerate by scrape_bcb_tariff_vigencia's own private CNPJ map.  That map is the
    one this module used to have -- latest List snapshot, raw codes, bare zero-padded
    CNPJ for anything unmapped -- so reading its output would reintroduce both key bugs
    on the listed-price side alone, and firms outside the prudential framework would
    compare against OFB prices filed under a different spelling of the same key.
    """
    if not VIGENCIA_HIST_CSV.exists():
        log.warning("tariff_vigencia_history.csv not found — skipping listed prices.")
        return pd.DataFrame(columns=["cod_cong_prudencial"])

    vh = pd.read_csv(VIGENCIA_HIST_CSV, low_memory=False)
    vh["cnpj"] = vh["cnpj"].astype(str).str.zfill(8)
    if cmap is None:
        cmap = build_cnpj_cong_map()
    vh = vh.merge(cmap, on="cnpj", how="left")
    vh["cod_cong_prudencial"] = (vh["cod_cong_prudencial"]
                                 .fillna(cnpj_fallback_key(vh["cnpj"]))
                                 .astype(str).str.strip())

    pf_rows = vh[
        (vh["customer_type"] == "F") &
        (vh["codigo_servico"].isin(PRIORITY_SERVICES))
    ].copy()
    pf_rows["valor_maximo"] = pd.to_numeric(pf_rows["valor_maximo"], errors="coerce")
    pf_rows["col_name"] = pf_rows["codigo_servico"].map(
        {k: f"listed_fee_{v}_pf" for k, v in PRIORITY_SERVICES.items()}
    )

    # Median across the conglomerate's member institutions and their price-change
    # history, matching what tariff_vigencia_summary reports per conglomerate.
    wide = pf_rows.pivot_table(
        index="cod_cong_prudencial",
        columns="col_name",
        values="valor_maximo",
        aggfunc="median",
    ).reset_index()
    wide.columns.name = None

    log.info(
        "Listed price columns: %s | %d conglomerates",
        [c for c in wide.columns if c.startswith("listed_")],
        len(wide),
    )
    return wide


# ---------------------------------------------------------------------------
# 5. DataVigencia stickiness: time-varying instrument
# ---------------------------------------------------------------------------

def build_stickiness_panel(cmap: pd.DataFrame) -> pd.DataFrame:
    """
    For each conglomerate × (year, quarter), compute:
      tarifa_stickiness_yrs = mean years since last tariff change,
                              averaged across BCB priority services and
                              all member institutions.

    Time-varying: stickiness grows quarter-by-quarter until a price change
    resets it. Computed from data_vigencia snapshots in tariff_vigencia_history.csv.

    Note: stickiness is NaN for quarters before a tariff's data_vigencia
    (i.e., where we have no evidence the tariff existed).
    """
    if not VIGENCIA_HIST_CSV.exists():
        log.warning("tariff_vigencia_history.csv not found — skipping stickiness.")
        return pd.DataFrame(columns=["cod_cong_prudencial", "year", "quarter"])

    vh = pd.read_csv(VIGENCIA_HIST_CSV, low_memory=False)
    vh["cnpj"] = vh["cnpj"].astype(str).str.zfill(8)

    # Filter to priority services PF only
    vh_pf = vh[
        (vh["customer_type"] == "F") &
        (vh["codigo_servico"].isin(PRIORITY_SERVICES))
    ].copy()

    vh_pf["data_vigencia"] = pd.to_datetime(vh_pf["data_vigencia"], errors="coerce")
    vh_pf = vh_pf.dropna(subset=["data_vigencia"])

    # Map CNPJ -> conglomerate
    vh_pf = vh_pf.merge(cmap, on="cnpj", how="left")
    vh_pf["cod_cong_prudencial"] = vh_pf["cod_cong_prudencial"].fillna(cnpj_fallback_key(vh_pf["cnpj"]))
    vh_pf["cod_cong_prudencial"] = vh_pf["cod_cong_prudencial"].astype(str).str.strip()

    # One data_vigencia per (conglomerate, service) — take the earliest date
    # across member institutions (most conservative: when the conglomerate first set the price)
    cong_svc = (
        vh_pf.groupby(["cod_cong_prudencial", "codigo_servico"])["data_vigencia"]
        .min()
        .reset_index()
    )
    cong_svc.columns = ["cod_cong_prudencial", "codigo_servico", "data_vigencia"]

    # Quarter end dates: 2013-Q1 through 2026-Q1
    quarters = []
    for yr in range(2013, 2027):
        for q, (m, d) in enumerate([(3, 31), (6, 30), (9, 30), (12, 31)], start=1):
            if yr == 2026 and q > 1:
                break
            quarters.append({"year": yr, "quarter": q,
                              "quarter_end": pd.Timestamp(yr, m, d)})
    qdf = pd.DataFrame(quarters)

    # Vectorised cross-join: cong_svc x quarters -> filter -> aggregate
    # cong_svc has ~6K rows, qdf has 53 rows -> cross is ~318K rows (manageable)
    cong_svc["_key"] = 1
    qdf["_key"] = 1
    cross = cong_svc.merge(qdf, on="_key").drop(columns="_key")

    cross = cross[cross["quarter_end"] >= cross["data_vigencia"]].copy()
    cross["stickiness_yrs"] = (
        (cross["quarter_end"] - cross["data_vigencia"]).dt.days / 365.25
    )

    stick = (
        cross.groupby(["cod_cong_prudencial", "year", "quarter"], as_index=False)
        .agg(
            tarifa_stickiness_yrs=("stickiness_yrs", "mean"),
            tarifa_stickiness_n=("stickiness_yrs", "count"),
        )
    )
    log.info(
        "Stickiness panel: %d rows | %d conglomerates | %d year-quarters",
        len(stick),
        stick["cod_cong_prudencial"].nunique(),
        stick[["year", "quarter"]].drop_duplicates().__len__(),
    )
    return stick


# ---------------------------------------------------------------------------
# 6. Merge all fee panels onto market_panel.csv
# ---------------------------------------------------------------------------

def patch_market_panel(
    cosif_panel:   pd.DataFrame,
    listed_prices: pd.DataFrame,
    stickiness:    pd.DataFrame,
) -> None:
    """
    Left-join all fee columns onto market_panel.csv → market_panel_with_fees.csv.

    Column groups added:
      cosif_fee_*           COSIF realized revenue ratios (time-varying)
      cosif_fee_valid       1 if year < COSIF_BREAK_YEAR (2025), else 0
      cosif_fee_ip_masked   1 if the firm is a payment institution (has_ip), whose
                            COSIF ratios are NOT deposit-account fees (see 6a-bis)
      listed_fee_*_pf       BCB Tarifas listed max prices (static)
      tarifa_stickiness_yrs DataVigencia stickiness instrument (time-varying)
      tarifa_stickiness_n   Number of priority services observed (data quality)

    The three quality flags are orthogonal and each marks a different defect:
    cosif_fee_valid the 2025 chart-of-accounts break, cosif_fee_guarded a
    small-denominator explosion in one cong-quarter, cosif_fee_ip_masked an
    institution type the 717xxx accounts do not measure the intended object for.
    """
    if not MARKET_PANEL_CSV.exists():
        log.error("market_panel.csv not found at %s", MARKET_PANEL_CSV)
        return

    market = pd.read_csv(MARKET_PANEL_CSV, dtype={"CodConglomeradoPrudencial": str},
                         low_memory=False)
    log.info("market_panel: %d rows x %d cols", len(market), len(market.columns))
    n_orig = len(market)

    def _norm(df, col):
        df[col] = df[col].astype(str).str.strip()
        return df

    market = _norm(market, "CodConglomeradoPrudencial")

    # ── 6a. COSIF time-varying ──────────────────────────────────────────────
    cpanel = cosif_panel.copy()
    cpanel = _norm(cpanel, "cod_cong_prudencial")
    cpanel = cpanel.rename(columns={"cod_cong_prudencial": "CodConglomeradoPrudencial"})

    merged = market.merge(cpanel, on=["CodConglomeradoPrudencial", "year", "quarter"], how="left")
    cosif_cols = [c for c in merged.columns if c.startswith("cosif_")]
    matched_cosif = merged[cosif_cols[0]].notna().sum() if cosif_cols else 0
    log.info("COSIF: %d cols | %d/%d rows matched", len(cosif_cols), matched_cosif, n_orig)

    # Validity flag: 0 in reclassification years (2025+)
    merged["cosif_fee_valid"] = (merged["year"] < COSIF_BREAK_YEAR).astype(int)

    # ── 6a-bis. Payment institutions: NaN the COSIF ratios ──────────────────
    # For a conglomerate with no deposit-taking charter the 717xxx accounts book card
    # interchange and merchant processing, not deposit-account tariffs, and the
    # denominator is settlement float rather than a deposit base.  The quotient is a
    # real number measuring the wrong thing: consolidating Nu Pagamentos against
    # Nubank's thin deposit stock puts Nubank at ~112 pp/yr in 2024 against 2-4 pp/yr
    # for the incumbents, i.e. the fee measure ranks the digital entrants as the most
    # expensive banks in the system, inverting their actual positioning.  The listed-
    # price and stickiness columns are left in place -- those are per-event BRL prices
    # from the BCB tariff tables and mean what they say for any institution that files
    # them.
    #
    # This is the type rule from service_fees_notes.md section 3, with the licence test
    # of load_nonbank_ip_conglomerates rather than that section's literal has_ip (see
    # its docstring for why has_ip is the wrong discriminator).  It is deliberately
    # separate from COSIF_FEE_RATIO_MAX, which stays a pooled plausibility band on the
    # standalone conglomerate panel, where sliver-denominator BANKS need catching too.
    ratio_cols = [f"cosif_{c}" for c in FEE_COLS if f"cosif_{c}" in merged.columns]
    ip_codes = load_nonbank_ip_conglomerates()
    ip_mask = merged["CodConglomeradoPrudencial"].isin(ip_codes)
    merged["cosif_fee_ip_masked"] = ip_mask.astype("int8")
    n_masked_obs = (int((ip_mask & merged[ratio_cols].notna().any(axis=1)).sum())
                    if ratio_cols else 0)
    merged.loc[ip_mask, ratio_cols] = np.nan
    log.info(
        "Payment-institution mask: %d rows flagged (%d of the panel's conglomerates); "
        "%d rows had a COSIF ratio NaN'd",
        int(ip_mask.sum()),
        merged.loc[ip_mask, "CodConglomeradoPrudencial"].nunique(),
        n_masked_obs,
    )

    # ── 6b. Listed prices (static — join on conglomerate only) ─────────────
    if not listed_prices.empty and "cod_cong_prudencial" in listed_prices.columns:
        lp = listed_prices.copy()
        lp = _norm(lp, "cod_cong_prudencial")
        lp = lp.rename(columns={"cod_cong_prudencial": "CodConglomeradoPrudencial"})
        merged = merged.merge(lp, on="CodConglomeradoPrudencial", how="left")
        listed_cols = [c for c in merged.columns if c.startswith("listed_fee_")]
        matched_listed = merged[listed_cols[0]].notna().sum() if listed_cols else 0
        log.info("Listed prices: %d cols | %d/%d rows matched", len(listed_cols),
                 matched_listed, n_orig)

    # ── 6c. Stickiness time-varying ─────────────────────────────────────────
    if not stickiness.empty and "cod_cong_prudencial" in stickiness.columns:
        st = stickiness.copy()
        st = _norm(st, "cod_cong_prudencial")
        st = st.rename(columns={"cod_cong_prudencial": "CodConglomeradoPrudencial"})
        merged = merged.merge(st, on=["CodConglomeradoPrudencial", "year", "quarter"], how="left")
        matched_stick = merged["tarifa_stickiness_yrs"].notna().sum()
        log.info("Stickiness: %d/%d rows matched", matched_stick, n_orig)

    if len(merged) != n_orig:
        log.error("Row count changed after merges! %d -> %d. Aborting.", n_orig, len(merged))
        return

    # ── 6d. Defragment then save ─────────────────────────────────────────────
    merged = merged.copy()  # collapses fragmented frame
    out_path = PANEL_DIR / "market_panel_with_fees.csv"
    merged.to_csv(out_path, index=False)
    # market_panel_with_fees.csv has its own sidecar, served whenever USE_FEE_PANEL=1.
    from utils import refresh_panel_cache
    refresh_panel_cache(out_path, merged)

    new_cols = [c for c in merged.columns if c not in market.columns or c == "cosif_fee_valid"]
    log.info("Saved: %s | %d rows x %d cols (+%d new fee cols)",
             out_path.name, len(merged), len(merged.columns), len(new_cols))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(patch_market: bool = False) -> None:
    # 1. Load COSIF institution panel
    cosif = load_cosif_institution()

    # 2. Build CNPJ -> conglomerate map (shared across all sources)
    cmap = build_cnpj_cong_map()

    # 3. COSIF: aggregate to conglomerate x quarter
    cosif_panel = aggregate_cosif_to_conglomerate_quarter(cosif, cmap)

    # 4. Save COSIF quarterly conglomerate panel (standalone output)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cosif_panel.to_csv(OUTPUT_CSV, index=False)
    log.info("Saved COSIF conglomerate panel: %s (%d rows)", OUTPUT_CSV.name, len(cosif_panel))

    # 5. BCB Tarifas listed prices (static)
    listed_prices = load_tarifa_listed_prices(cmap)

    # 6. DataVigencia stickiness (time-varying)
    stickiness = build_stickiness_panel(cmap)

    # 7. Patch market panel with all fee sources
    if patch_market:
        patch_market_panel(cosif_panel, listed_prices, stickiness)

    log.info("=== Done ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fee panel builder: COSIF + BCB Tarifas listed prices + DataVigencia stickiness"
    )
    parser.add_argument("--patch-market", action="store_true",
                        help="Merge all fee columns onto market_panel.csv -> market_panel_with_fees.csv")
    args = parser.parse_args()
    main(patch_market=args.patch_market)

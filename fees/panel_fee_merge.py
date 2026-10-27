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

2. BCB listed maximum prices, dated  (build_account_fee_history.py)
   - Conglomerate × quarter  |  2015-Q1 → 2024-Q4 from the BCB monthly census of listed
     tariffs (scrape_bcb_tarifas_monthly.py); 2025-Q1 on from the Olinda snapshot of
     2026-06-22 alone (scrape_bcb_tariff_vigencia.py); 2013-2014 NaN
   - 13 services, labelled by code (LISTED_FEE_SERVICES): Pacotes Padronizados I-IV, PJ
     account maintenance, and seven PF services
   - listed_fee_<name>: the head institution's price in force on the quarter's last day;
     when the head does not list the service, the median of the members' positive prices
     (0.00 when every listing is 0.00); listed_fee_n_inst_<name>: members with a price
   - A price is in force on a date when the date lies in [vigência, in_force_to] of one of
     its versions (account_fee_listed_history.csv for the six account-keeping services,
     build_account_fee_history.load_versions() for the others); the version with the
     latest vigência wins
   - Institutions are placed in conglomerates as the market panel places their deposits
     (utils.conglomerates.deposit_panel_codes): an ESTBAN bank under its static code in
     every quarter, any other institution by the IF Data List of that quarter
   - Listed is a ceiling, not the price charged.  An institution absent from the census
     lists nothing for that service: NaN, not zero.  The bridge across consecutive census
     files (bridged versions) carries under 1% of known months.

3. Price stickiness, from the same dated versions
   - Conglomerate × quarter  |  2015-Q1 → 2024-Q4, NaN elsewhere
   - tarifa_stickiness_yrs: years from the vigência of the price in force to the quarter
     end, averaged over the head institution's PF services of LISTED_FEE_SERVICES (over
     every member's when the head has none); grows until a price change and resets at it
   - tarifa_stickiness_n: the institution × service pairs averaged

Output
------
  BCB/Tarifas/processed/cosif_fee_quarterly_conglomerate.csv  (COSIF only)
  BCB/Egan_et_al_2025_Rep/processed/market_panel_with_fees.csv  (all sources)

Usage
-----
  python fees/panel_fee_merge.py
  python fees/panel_fee_merge.py --patch-market

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

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: utils/

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
_REPO           = Path(__file__).resolve().parents[3]
PANEL_DIR       = paths.PROCESSED
TARIF_DIR       = paths.TARIFAS_PROC

COSIF_INST_PARQUET  = TARIF_DIR / "cosif_service_fees_institution.parquet"
COSIF_INST_CSV      = TARIF_DIR / "cosif_service_fees_institution.csv"
VIGENCIA_HIST_CSV   = TARIF_DIR / "tariff_vigencia_history.csv"
CENSUS_HISTORY_CSV  = TARIF_DIR / "tariff_history_bcb_monthly.csv"
ACCOUNT_FEE_HISTORY_CSV = TARIF_DIR / "account_fee_listed_history.csv"
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

# BCB listed services carried into the panel, keyed (pessoa, codigo_servico) as in the census.
# Each code has exactly one description in the 2015-2024 census, so the label follows the code.
LISTED_FEE_SERVICES = {
    ("F", "1501"): "pkg1_pf",                # Pacote Padronizado de Serviços I
    ("F", "1502"): "pkg2_pf",                # Pacote Padronizado de Serviços II
    ("F", "1503"): "pkg3_pf",                # Pacote Padronizado de Serviços III
    ("F", "1504"): "pkg4_pf",                # Pacote Padronizado de Serviços IV
    ("J", "0402"): "maint_active_pj",        # Manutenção de conta ativa
    ("J", "0403"): "maint_inactive_pj",      # Manutenção de conta inativa
    ("F", "1101"): "acct_reg_pf",            # Confecção de cadastro para início de relacionamento
    ("F", "1201"): "card_2nd_debit_pf",      # 2a via de cartão com função débito
    ("F", "1203"): "ccf_exclusion_pf",       # Exclusão do Cadastro de Emitentes de Cheques sem Fundo
    ("F", "1204"): "cheque_countermand_pf",  # Contra-ordem e oposição ao pagamento de cheque
    ("F", "1209"): "withdrawal_teller_pf",   # Saque pessoal
    ("F", "1210"): "withdrawal_atm_pf",      # Saque em terminal
    ("F", "1213"): "statement_pf",           # Extrato de um período
}
LISTED_PRICE_COLS = [f"listed_fee_{v}" for v in LISTED_FEE_SERVICES.values()]
LISTED_COUNT_COLS = [f"listed_fee_n_inst_{v}" for v in LISTED_FEE_SERVICES.values()]
# Stickiness averages over the PF services above (1101-1213 and the four packages).
STICKINESS_SERVICES = {k for k in LISTED_FEE_SERVICES if k[0] == "F"}
# The census covers 2015-01..2024-12; from 2025 only the snapshot dates a price.
LISTED_FIRST_YEAR = 2015
CENSUS_LAST_YEAR = 2024
_VERSION_KEY = ["cnpj8", "pessoa", "codigo_servico"]

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
# panel_deposit_rates NaNs rate_a4 above RATE_A4_MAX_MULT x Selic and
# cosif_implicit_rate above 0.5.  Negative ratios (from svc_revenue_inc within-
# semester reversals) are likewise implausible for a fee ratio and NaN'd.
# The ceiling is env-overridable (mirrors panel_deposit_rates' RATE_A4_MAX_MULT); the
# default 1.0 = "quarterly fees cannot exceed 100% of the deposit base" leaves
# the entire clean central distribution (p95 <= 0.57) untouched while removing
# every explosion.  A cosif_fee_guarded flag marks affected cong-quarters,
# mirroring panel_deposit_rates' rate_a4_guarded.
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
    from utils.conglomerates import build_conglomerate_canonical_map

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
    from utils.conglomerates import build_cnpj_conglomerate_map

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
# 4. BCB listed prices: dated versions -> institution x service x quarter
# ---------------------------------------------------------------------------

def _in_listed_services(pessoa: pd.Series, codigo: pd.Series) -> np.ndarray:
    return np.array([(p, s) in LISTED_FEE_SERVICES for p, s in zip(pessoa, codigo)], dtype=bool)


def _log_non_real_rows() -> None:
    """Log the listed rows left out because they are not prices in R$.

    About 2% of the census's listed rows carry moeda "Percentual", a share rather than a
    price.  load_versions() and account_fee_listed_history.csv keep moeda == "Real" (in the
    snapshot, tipo_valor == "Real") only, and so does every listed_fee_* column."""
    c = pd.read_csv(CENSUS_HISTORY_CSV, dtype=str, usecols=["pessoa", "codigo_servico", "moeda"])
    scope = _in_listed_services(c["pessoa"], c["codigo_servico"])
    not_real = (c["moeda"] != "Real").to_numpy()
    log.info("Listed prices in R$ only. Census price spells dropped (moeda != Real): %d of %d "
             "in the %d listed services; %d of %d (%.1f%%) over every service.",
             int((scope & not_real).sum()), int(scope.sum()), len(LISTED_FEE_SERVICES),
             int(not_real.sum()), len(c), 100 * not_real.mean())
    s = pd.read_csv(VIGENCIA_HIST_CSV, dtype=str,
                    usecols=["customer_type", "codigo_servico", "tipo_valor"])
    s_scope = _in_listed_services(s["customer_type"], s["codigo_servico"].str.zfill(4))
    s_not_real = (s["tipo_valor"] != "Real").to_numpy()
    log.info("Snapshot rows dropped (tipo_valor != Real): %d of %d in the listed services.",
             int((s_scope & s_not_real).sum()), int(s_scope.sum()))


def load_listed_fee_versions() -> pd.DataFrame:
    """Dated price versions of every LISTED_FEE_SERVICES service, one row per version.

    The six account-keeping services (F 1501-1504, J 0402/0403) are read from
    account_fee_listed_history.csv, which fees/build_account_fee_history.py writes; the
    other services go through the same build_account_fee_history.load_versions() here:
    census spells and snapshot rows joined, prices in R$ only.

    A version proves its price over [vigencia, seen_to].  in_force_to extends a census
    version to the day before the next vigência when the next version first appears in the
    immediately following monthly file (bridged=True); otherwise in_force_to = seen_to.

    Columns: cnpj8, pessoa, codigo_servico, vigencia, valor_maximo, source, seen_to,
    in_force_to, bridged.
    """
    from fees.build_account_fee_history import SERVICES as ACCOUNT_SERVICES, load_versions

    cols = _VERSION_KEY + ["vigencia", "valor_maximo", "source", "seen_to", "in_force_to",
                           "bridged"]
    _log_non_real_rows()

    if not ACCOUNT_FEE_HISTORY_CSV.exists():
        raise FileNotFoundError(f"{ACCOUNT_FEE_HISTORY_CSV} not found: run "
                                f"python fees/build_account_fee_history.py first.")
    acct = pd.read_csv(ACCOUNT_FEE_HISTORY_CSV, dtype=str)
    acct["cnpj8"] = acct["cnpj8"].str.zfill(8)
    acct["codigo_servico"] = acct["codigo_servico"].str.zfill(4)
    acct = acct[_in_listed_services(acct["pessoa"], acct["codigo_servico"])].copy()
    for col in ("vigencia", "seen_to", "in_force_to"):
        acct[col] = pd.to_datetime(acct[col], errors="coerce")
    acct["valor_maximo"] = pd.to_numeric(acct["valor_maximo"], errors="coerce")

    other = {k: v for k, v in LISTED_FEE_SERVICES.items() if k not in ACCOUNT_SERVICES}
    rest = load_versions(services=other)

    v = pd.concat([acct[cols], rest[cols]], ignore_index=True)
    v["bridged"] = v["bridged"].astype(str) == "True"
    log.info("Listed-price versions: %d (%d from %s, %d built for %d other services) | "
             "%d institutions | %d bridged",
             len(v), len(acct), ACCOUNT_FEE_HISTORY_CSV.name, len(rest), len(other),
             v["cnpj8"].nunique(), int(v["bridged"].sum()))
    return v


def _quarter_index(ts: pd.Series) -> np.ndarray:
    """year * 4 + (quarter - 1) of each date."""
    return (ts.dt.year * 4 + (ts.dt.month - 1) // 3).to_numpy(dtype="int64")


def _quarter_end(qi) -> np.ndarray:
    """Last calendar day of each quarter index (datetime64)."""
    qi = np.asarray(qi, dtype="int64")
    first = pd.to_datetime(pd.DataFrame({"year": qi // 4, "month": (qi % 4) * 3 + 3, "day": 1}))
    return (first + pd.offsets.MonthEnd(0)).to_numpy(dtype="datetime64[ns]")


def listed_prices_by_quarter(versions: pd.DataFrame) -> pd.DataFrame:
    """Institution × service × quarter: the listed price in force on the quarter's last day.

    The rule of build_account_fee_history.evaluate(), at quarter ends: the price on date d
    is the version with the latest vigência (then the latest in_force_to) among those with
    vigência <= d <= in_force_to.  A version listed with a vigência after its last sighting
    (an announced price) proves nothing and is skipped.  Quarters start in 2015-Q1, the
    first census month; 2013-2014 are not filled.  After CENSUS_LAST_YEAR only versions
    the 2026-06-22 snapshot shows can be in force, and only from their vigência on: a
    snapshot price is never used for a date before its data_vigencia.

    Returns cnpj8, pessoa, codigo_servico, year, quarter, quarter_end, price, vigencia,
    source (census, snapshot, census+snapshot, or census_bridged when the quarter end falls
    after the version was last seen), conflict (versions with different prices in force on
    the same day).  Only institution-quarters with a price in force are returned: a service
    the institution does not list is absent, not zero.
    """
    v = versions[versions["vigencia"].notna() & versions["in_force_to"].notna()
                 & (versions["vigencia"] <= versions["seen_to"])].reset_index(drop=True)
    start = np.maximum(_quarter_index(v["vigencia"]), LISTED_FIRST_YEAR * 4)
    q_last = _quarter_index(v["in_force_to"])
    end = np.where(v["in_force_to"].to_numpy(dtype="datetime64[ns]") >= _quarter_end(q_last),
                   q_last, q_last - 1)
    n = np.clip(end - start + 1, 0, None)
    row = np.repeat(np.arange(len(v)), n)
    offset = np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)

    x = v.iloc[row].reset_index(drop=True)
    x["qi"] = start[row] + offset
    x = x.sort_values(_VERSION_KEY + ["qi", "vigencia", "in_force_to"], kind="mergesort")
    x["conflict"] = (x.groupby(_VERSION_KEY + ["qi"])["valor_maximo"].transform("nunique") > 1)
    x = x.drop_duplicates(_VERSION_KEY + ["qi"], keep="last").copy()

    x["year"] = x["qi"] // 4
    x["quarter"] = x["qi"] % 4 + 1
    x["quarter_end"] = _quarter_end(x["qi"].to_numpy())
    x["source"] = np.where(x["quarter_end"] > x["seen_to"], "census_bridged",
                           x["source"].astype(str))
    late = ((x["year"] > CENSUS_LAST_YEAR) & ~x["source"].str.contains("snapshot")).to_numpy()
    if late.any():
        log.warning("%d institution-quarters after %d rest on a census version alone; "
                    "dropped (after the census only the snapshot dates a price).",
                    int(late.sum()), CENSUS_LAST_YEAR)
        x = x[~late]
    x = x.rename(columns={"valor_maximo": "price"})
    return x[_VERSION_KEY + ["year", "quarter", "quarter_end", "price", "vigencia", "source",
                             "conflict"]].reset_index(drop=True)


def build_dated_cnpj_cong_map(keys: pd.DataFrame) -> pd.DataFrame:
    """Conglomerate of each institution in each quarter, keyed the way the market panel keys
    that institution's deposits (utils.conglomerates.deposit_panel_codes).

    keys holds cnpj8, year, quarter.

    * An ESTBAN bank keeps its build_cnpj_conglomerate_map code ('CNPJ_<int>' when that map
      lacks it) in every quarter, as its branch deposits do in the panel.  A bank that joins
      an acquirer's conglomerate (Kirton into Bradesco in 2016, Banco Pan into BTG from
      2021) keeps its own code, where its panel rows are.
    * Every other institution takes the CodConglomeradoPrudencial of that quarter's IF Data
      List file (build_cnpj_to_congl: individual institutions in activity), canonicalised.
      A CNPJ missing from that quarter's file takes the nearest earlier file that lists it,
      else the nearest later one.  A member that moves between conglomerates counts toward
      the one it belongs to that quarter.
    * An institution the List files place in no prudential conglomerate (a standalone,
      listed without a code), or never list, is keyed by cnpj_fallback_key(): 'CNPJ_<int>',
      the market panel's convention.

    Returns cnpj8, year, quarter, cod_cong_prudencial, cong_map_source (estban_static,
    list_quarter, list_nearest, fallback).
    """
    from utils.conglomerates import build_cnpj_to_congl, deposit_panel_codes, estban_bank_cnpjs

    lists = build_cnpj_to_congl().dropna(subset=["AnoMes_list"])
    lists = lists[lists["CNPJ"].astype(str).str.fullmatch(r"\d+")].copy()
    code = lists["CodConglomeradoPrudencial"].astype(str).str.strip()
    prudential = code.str.fullmatch(r"C\d+").to_numpy()
    # As reported; deposit_panel_codes canonicalises.
    lists["code"] = cnpj_fallback_key(lists["CNPJ"]).to_numpy(dtype=object)
    lists.loc[prudential, "code"] = code[prudential].to_numpy(dtype=object)
    lists["AnoMes_list"] = lists["AnoMes_list"].astype("int64")
    dm = (lists[["CNPJ", "AnoMes_list", "code"]].astype({"CNPJ": object, "code": object})
          .sort_values("AnoMes_list").reset_index(drop=True))

    f = keys[["cnpj8", "year", "quarter"]].drop_duplicates().astype({"cnpj8": object}).copy()
    f["AnoMes"] = (f["year"].astype("int64") * 100 + f["quarter"].astype("int64") * 3)
    f = f.sort_values("AnoMes").reset_index(drop=True)
    asof = dict(left_on="AnoMes", right_on="AnoMes_list", left_by="cnpj8", right_by="CNPJ")
    back = pd.merge_asof(f, dm, direction="backward", **asof)
    fwd = pd.merge_asof(f, dm, direction="forward", **asof)

    code_q = back["code"].where(back["code"].notna(), fwd["code"])
    reported = code_q.where(code_q.notna(), cnpj_fallback_key(f["cnpj8"])).astype(str)
    f["cod_cong_prudencial"] = (deposit_panel_codes(f["cnpj8"].astype(str), reported)
                                .astype(str).str.strip())
    estban = f["cnpj8"].isin(estban_bank_cnpjs()).to_numpy(dtype=bool)
    exact = (back["AnoMes_list"] == f["AnoMes"]).fillna(False).to_numpy(dtype=bool)
    f["cong_map_source"] = np.select([estban, exact, code_q.notna().to_numpy()],
                                     ["estban_static", "list_quarter", "list_nearest"],
                                     "fallback")
    return f[["cnpj8", "year", "quarter", "cod_cong_prudencial", "cong_map_source"]]


def conglomerate_heads(codes: pd.Series) -> pd.Series:
    """8-digit CNPJ of each conglomerate's head institution.

    For a prudential code, the canonical_lider of utils.conglomerates.
    build_conglomerate_canonical_map(): the lead institution (CnpjInstituicaoLider) on the
    member-institution rows of the code's last IF Data List snapshot.  The market panel's
    CNPJ_Lider is built from the same map.  For a 'CNPJ_<int>' key, the institution itself.
    Empty string when the code has no leader."""
    from utils.conglomerates import build_conglomerate_canonical_map

    _, lider = build_conglomerate_canonical_map()
    lut = {}
    for c in pd.unique(codes.astype(str)):
        if c.startswith("CNPJ_"):
            lut[c] = c[len("CNPJ_"):].zfill(8)
        elif c in lider and pd.notna(lider[c]):
            lut[c] = str(int(lider[c])).zfill(8)
        else:
            lut[c] = ""
    return codes.astype(str).map(lut)


def _flag_heads(d: pd.DataFrame) -> np.ndarray:
    """True on the rows of d that belong to their conglomerate's head institution."""
    heads = conglomerate_heads(d["cod_cong_prudencial"]).fillna("").astype(str)
    return (d["cnpj8"].astype(str) == heads).to_numpy(dtype=bool)


def listed_fee_institution_quarters() -> pd.DataFrame:
    """listed_prices_by_quarter() with each institution-quarter's conglomerate attached
    (build_dated_cnpj_cong_map).  The one frame both the listed prices and the stickiness
    are aggregated from."""
    iq = listed_prices_by_quarter(load_listed_fee_versions())
    iq = iq.merge(build_dated_cnpj_cong_map(iq), on=["cnpj8", "year", "quarter"], how="left")
    known = iq["price"].notna()
    log.info("Listed prices by institution x service x quarter: %d rows (%d-Q1 to %d-Q%d) | "
             "%d institutions | %d conflicting overlaps | source %s",
             len(iq), iq["year"].min(), iq["year"].max(),
             iq.loc[iq["year"] == iq["year"].max(), "quarter"].max(),
             iq["cnpj8"].nunique(), int(iq["conflict"].sum()),
             iq.loc[known, "source"].value_counts().to_dict())
    log.info("Conglomerate placement of institution-quarters: %s",
             iq.drop_duplicates(["cnpj8", "year", "quarter"])["cong_map_source"]
               .value_counts().to_dict())
    return iq


def load_tarifa_listed_prices(inst_q: pd.DataFrame | None = None) -> pd.DataFrame:
    """Conglomerate × quarter listed prices of LISTED_FEE_SERVICES.

    listed_fee_<name>         the head institution's price in force on the quarter's last
                              day (listed_prices_by_quarter; head = conglomerate_heads).
                              When the head does not list the service that quarter, the
                              median of the members' positive prices; a conglomerate whose
                              only listings are R$0.00 gets 0.00.
    listed_fee_n_inst_<name>  member institutions with a price for the service that quarter
                              (0.00 included)

    Members are placed by build_dated_cnpj_cong_map.  2015-Q1..2024-Q4 come from the monthly
    census, with a snapshot version counting from its vigência on; 2025-Q1 on, from the
    2026-06-22 snapshot alone, where its vigência is on or before the quarter end.  No row
    before 2015.

    Listed is a maximum (a ceiling), not what clients pay; many hold only the free
    essential services.  A member that does not list the service is left out: not listed is
    not a zero price.  A listed 0.00 is a price: the head's 0.00 is used as it stands, and
    only the fallback median leaves zeros out (BB Banco de Investimento lists Package I at
    0.00; the Banco do Brasil conglomerate takes BB's own price).
    """
    if inst_q is None:
        inst_q = listed_fee_institution_quarters()
    keys = ["cod_cong_prudencial", "year", "quarter"]
    cell = keys + ["name"]
    names = list(LISTED_FEE_SERVICES.values())
    d = inst_q[inst_q["price"].notna()].copy()
    d["name"] = [LISTED_FEE_SERVICES[(p, s)] for p, s in zip(d["pessoa"], d["codigo_servico"])]
    d["is_head"] = _flag_heads(d)

    n_inst = d.groupby(cell)["price"].size()
    head = d[d["is_head"]].groupby(cell)["price"].first().reindex(n_inst.index)
    positive = d[d["price"] > 0].groupby(cell)["price"].median().reindex(n_inst.index)
    value = head.fillna(positive).fillna(0.0)
    rule = np.select([head.notna().to_numpy(), positive.notna().to_numpy()],
                     ["head", "positive_median"], "all_zero")

    price = value.unstack("name").reindex(columns=names)
    count = n_inst.unstack("name").reindex(columns=names).fillna(0).astype("int64")
    price.columns = LISTED_PRICE_COLS
    count.columns = LISTED_COUNT_COLS
    wide = price.join(count).reset_index()

    log.info("Listed prices: %d conglomerate-quarters | %d conglomerates | %d-%d | "
             "conglomerate-quarter-service cells by rule %s",
             len(wide), wide["cod_cong_prudencial"].nunique(), wide["year"].min(),
             wide["year"].max(), pd.Series(rule).value_counts().to_dict())
    return wide


# ---------------------------------------------------------------------------
# 5. Price stickiness, from the same dated versions
# ---------------------------------------------------------------------------

def build_stickiness_panel(inst_q: pd.DataFrame | None = None) -> pd.DataFrame:
    """Conglomerate × quarter price stickiness, 2015-Q1..2024-Q4.

    For one institution × service: years (days / 365.25) from the vigência of the version
    in force on the quarter's last day to that day, over the PF services of
    LISTED_FEE_SERVICES (1101-1213 and the packages 1501-1504).

    tarifa_stickiness_yrs = the mean over the head institution's PF services
                            (conglomerate_heads), the same head rule as listed_fee_*; when
                            the head has none in force that quarter, the mean over every
                            member's PF services
    tarifa_stickiness_n   = the institution × service pairs in that mean

    It grows quarter by quarter until a price change and resets at the change.  vigência is
    the institution's own date for its last change.  Only pairs with a listed price in
    force count; quarters outside the census window have no row (NaN after the merge).
    """
    if inst_q is None:
        inst_q = listed_fee_institution_quarters()
    keys = ["cod_cong_prudencial", "year", "quarter"]
    pf = np.array([(p, s) in STICKINESS_SERVICES
                   for p, s in zip(inst_q["pessoa"], inst_q["codigo_servico"])], dtype=bool)
    d = inst_q[pf & inst_q["price"].notna().to_numpy()
               & inst_q["year"].between(LISTED_FIRST_YEAR, CENSUS_LAST_YEAR).to_numpy()].copy()
    d["stickiness_yrs"] = (d["quarter_end"] - d["vigencia"]).dt.days / 365.25
    d["is_head"] = _flag_heads(d)
    head_present = d.groupby(keys)["is_head"].transform("any").to_numpy(dtype=bool)
    use = d[~head_present | d["is_head"].to_numpy()]

    stick = (
        use.groupby(keys, as_index=False)
        .agg(
            tarifa_stickiness_yrs=("stickiness_yrs", "mean"),
            tarifa_stickiness_n=("stickiness_yrs", "count"),
        )
    )
    n_head = int(d[d["is_head"]].groupby(keys).ngroups)
    log.info(
        "Stickiness panel: %d rows (%d from the head's services, %d from all members) | "
        "%d conglomerates | %d year-quarters | mean %.2f yrs",
        len(stick), n_head, len(stick) - n_head,
        stick["cod_cong_prudencial"].nunique(),
        stick[["year", "quarter"]].drop_duplicates().__len__(),
        float(stick["tarifa_stickiness_yrs"].mean()) if len(stick) else float("nan"),
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
      listed_fee_<name>     BCB listed maximum price in force on the quarter's last day,
                            the head institution's (fallback: median of the members'
                            positive prices; 0.00 if all are 0.00), for the 13 services of
                            LISTED_FEE_SERVICES (pkg1_pf..pkg4_pf, maint_active_pj,
                            maint_inactive_pj, acct_reg_pf, card_2nd_debit_pf,
                            ccf_exclusion_pf, cheque_countermand_pf, withdrawal_teller_pf,
                            withdrawal_atm_pf, statement_pf); 2015-Q1 on, NaN before
      listed_fee_n_inst_<name>  member institutions with that price (0 from 2015 when none
                            lists the service, NaN before 2015)
      tarifa_stickiness_yrs years since the vigência of the price in force, mean over the
                            head's PF listed services (fallback: every member's);
                            2015-Q1..2024-Q4
      tarifa_stickiness_n   institution × service pairs in that mean (0 in 2015-2024
                            when none, NaN outside)

    Every group joins on CodConglomeradoPrudencial × year × quarter.  Listed prices are
    ceilings, not prices charged, and a service no member lists is NaN, not zero.

    The three quality flags are orthogonal and each marks a different defect:
    cosif_fee_valid the 2025 chart-of-accounts break, cosif_fee_guarded a
    small-denominator explosion in one cong-quarter, cosif_fee_ip_masked an
    institution type the 717xxx accounts do not measure the intended object for.

    Writes market_panel_with_fees.csv (and its parquet sidecar) only, and only when the
    row count is unchanged by the merges.
    """
    if not MARKET_PANEL_CSV.exists():
        log.error("market_panel.csv not found at %s", MARKET_PANEL_CSV)
        return

    market = pd.read_csv(MARKET_PANEL_CSV, dtype={"CodConglomeradoPrudencial": str},
                         low_memory=False)
    log.info("market_panel: %d rows x %d cols", len(market), len(market.columns))
    n_orig = len(market)
    # The columns this function adds are rebuilt from scratch: any copy already in the base
    # panel would come back from the merges as _x/_y pairs.
    fee_prefixes = ("cosif_fee_", "listed_fee_", "tarifa_stickiness_")
    carried = [c for c in market.columns if c.startswith(fee_prefixes)]
    if carried:
        log.warning("market_panel.csv already carries %d fee columns; dropped before the "
                    "merge: %s", len(carried), carried)
        market = market.drop(columns=carried)

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
    # price and stickiness columns are left in place -- those are listed BRL prices from
    # the BCB tariff census and mean what they say for any institution that files them.
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

    # ── 6b. Listed prices (conglomerate × quarter) ──────────────────────────
    keys = ["CodConglomeradoPrudencial", "year", "quarter"]
    if not listed_prices.empty and "cod_cong_prudencial" in listed_prices.columns:
        lp = listed_prices.copy()
        lp = _norm(lp, "cod_cong_prudencial")
        lp = lp.rename(columns={"cod_cong_prudencial": "CodConglomeradoPrudencial"})
        merged = merged.merge(lp, on=keys, how="left")
        # A conglomerate-quarter of the census or snapshot years with no member listing the
        # service has zero listing members; before 2015 the count is unknown and stays NaN.
        in_window = (merged["year"] >= LISTED_FIRST_YEAR).to_numpy()
        for c in LISTED_COUNT_COLS:
            merged.loc[in_window, c] = merged.loc[in_window, c].fillna(0)
        log.info("Listed prices: %d price cols + %d count cols | %d/%d rows with "
                 "listed_fee_pkg1_pf", len(LISTED_PRICE_COLS), len(LISTED_COUNT_COLS),
                 int(merged["listed_fee_pkg1_pf"].notna().sum()), n_orig)
        win = merged[merged["year"].between(2016, CENSUS_LAST_YEAR)]
        log.info("Share of rows with listed_fee_pkg1_pf, by year: %s",
                 {int(y): round(float(s), 3) for y, s in
                  win.groupby("year")["listed_fee_pkg1_pf"].apply(lambda x: x.notna().mean()).items()})

    # ── 6c. Stickiness (conglomerate × quarter) ─────────────────────────────
    if not stickiness.empty and "cod_cong_prudencial" in stickiness.columns:
        st = stickiness.copy()
        st = _norm(st, "cod_cong_prudencial")
        st = st.rename(columns={"cod_cong_prudencial": "CodConglomeradoPrudencial"})
        merged = merged.merge(st, on=keys, how="left")
        in_census = merged["year"].between(LISTED_FIRST_YEAR, CENSUS_LAST_YEAR).to_numpy()
        merged.loc[in_census, "tarifa_stickiness_n"] = (
            merged.loc[in_census, "tarifa_stickiness_n"].fillna(0))
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

    # 5. BCB listed prices, conglomerate x quarter (members placed quarter by quarter)
    inst_q = listed_fee_institution_quarters()
    listed_prices = load_tarifa_listed_prices(inst_q)

    # 6. Price stickiness from the same dated versions
    stickiness = build_stickiness_panel(inst_q)

    # 7. Patch market panel with all fee sources
    if patch_market:
        patch_market_panel(cosif_panel, listed_prices, stickiness)

    log.info("=== Done ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fee panel builder: COSIF + dated BCB listed prices + price stickiness"
    )
    parser.add_argument("--patch-market", action="store_true",
                        help="Merge all fee columns onto market_panel.csv -> market_panel_with_fees.csv")
    args = parser.parse_args()
    main(patch_market=args.patch_market)

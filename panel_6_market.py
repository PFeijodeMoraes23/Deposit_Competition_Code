## build_market_panel.py
# Author: Pedro Feijó de Moraes
#
# Last edited: 2026-03-06
#
# Purpose: Merge all MCA-level characteristic panels with the conglomerate
#          deposit panel to produce a single analysis-ready dataset for
#          the deposit competition / BLP demand model.
#
#   Inputs  (all built by earlier pipeline scripts):
#     A. BCB/Panel/deposits_panel.csv
#          Conglomerate x CODMUN_IBGE (7-digit) x Year x Quarter
#          Core left-hand-side data (deposit balances by type, source)
#
#     B. IBGE/muni_mca_regions_2010_2024_panel.csv
#          Municipality -> MCA crosswalk used to map CODMUN_IBGE -> mca_code
#
#     C. IBGE/mca_demographics_panel.csv
#          MCA x Year : pop_total, gdp_per_capita, fraction_65plus, fraction_young
#
#     D. BCB/PIX/pix_mca_panel.csv
#          MCA x Year x Quarter : pix_active, pix_users_pf_per1000, pix_txns_pf, ...
#
#     E. ANATEL/anatel_mca_panel.csv
#          MCA x Year x Quarter : connections_per100, frac_4g5g
#
#     F. INSS/inss_mca_panel.csv
#          MCA x Year x Quarter : retirees_per1000
#
#     G. BCB/Inclusion/bcb_inclusion_mca_panel.csv
#          MCA x Year : branches_per1000, correspondents_per1000, access_points_per1000
#
#     H. CadUnico/cadunico_mca_panel.csv
#          MCA x Year x Quarter : cadunico_families_per1000, cadunico_extreme_poverty
#
#   Output:
#     BCB/Panel/market_panel.csv
#       Unit of observation: CodConglomeradoPrudencial x mca_code x Year x Quarter
#
#       Deposit columns (from A):
#         dep_a1, dep_a2, dep_a3, dep_a4, dep_outros, dep_a5
#
#       Market-level columns (from C-H):
#         pop_total, gdp_per_capita
#         fraction_65plus, fraction_young, age_interpolated
#         pix_active, pix_users_pf_per1000, pix_txns_pf, pix_value_pf_r1000
#         connections_per100, frac_4g5g
#         retirees_per1000
#         branches_per1000, correspondents_per1000, access_points_per1000
#         cadunico_families_per1000, cadunico_extreme_poverty
#
#       Identifier / flag columns:
#         CodConglomeradoPrudencial, CNPJ_Lider, NomeInstituicao
#         CODMUN_IBGE (original 7-digit code)
#         mca_code, Year, Quarter, Source (ESTBAN | IF_DATA)
#
# Notes:
#   * Tier-1 rows (ESTBAN, CODMUN_IBGE ≠ 0) are merged to market characteristics
#     via the MCA crosswalk.  Tier-2 rows (IF Data only, CODMUN_IBGE = 0) are
#     kept as-is; their market-characteristic columns are left NaN (they appear
#     only in the aggregation-moment equation, not the local-market equations).
#   * Annual-frequency panels (C = demographics, G = BCB inclusion) are broadcast
#     to all four quarters of each year before merging.
#   * Missing panel files are handled gracefully; their columns are filled with NaN
#     and logged as warnings.  This lets the panel be built incrementally as
#     downloads complete.
###-----------------------------------------------------------------------------

import concurrent.futures
import os
import sys
import logging
try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None

if ensure_project_venv is not None:
    ensure_project_venv(__file__)

import numpy as np
import pandas as pd

try:
    from utils.toon_runtime import resolve_script_paths
except Exception:
    resolve_script_paths = None

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

# Keep the default console encoding (CP1252 / Latin-1 on Windows).
# All print strings below use plain ASCII to avoid garbled output.

## -----------------------------------------------------------------------------
## 1) PATHS
## -----------------------------------------------------------------------------
from utils import paths
BASE = str(paths.OPEN_FINANCE)

BCB_DIR    = str(paths.BCB)
IBGE_DIR   = str(paths.IBGE_DIR)
ANATEL_DIR = str(paths.ANATEL_DIR)
INSS_DIR   = os.path.join(BASE, "INSS")
CAD_DIR    = str(paths.CADUNICO_DIR)
PIX_DIR    = str(paths.PIX_DIR)

PANEL_DIR  = str(paths.PROCESSED)
os.makedirs(PANEL_DIR, exist_ok=True)

DEPOSITS_CSV   = os.path.join(PANEL_DIR,                  "deposits_panel.csv")
MCA_XWALK_CSV  = os.path.join(IBGE_DIR,                   "muni_mca_regions_2010_2024_panel.csv")
DEMO_CSV       = os.path.join(IBGE_DIR,                   "mca_demographics_panel.csv")
PIX_CSV        = os.path.join(PIX_DIR,                    "pix_mca_panel.csv")
ANATEL_CSV     = os.path.join(ANATEL_DIR,                 "anatel_mca_panel.csv")
INSS_CSV       = os.path.join(INSS_DIR,                   "inss_mca_panel.csv")
INCLUSION_CSV  = os.path.join(BCB_DIR, "Inclusion",       "bcb_inclusion_mca_panel.csv")
CADUNICO_CSV   = os.path.join(CAD_DIR,                    "cadunico_mca_panel.csv")
TARIFAS_FEE_CSV = os.path.join(BCB_DIR, "Tarifas", "processed", "tarifas_fee_summary.csv")
BANK_CHARS_CSV = os.path.join(PANEL_DIR,                  "bank_chars_panel.csv")

OUTPUT_CSV     = os.path.join(PANEL_DIR,                  "market_panel.csv")

# Hard cap: the master analysis panel ends at 2025-Q4.  Raw inputs may extend
# further, but everything after 2025-Q4 is dropped for a fixed end point.
PANEL_END_YEAR    = 2025
PANEL_END_QUARTER = 4

if resolve_script_paths is not None:
    _paths = resolve_script_paths(
        "build_market_panel",
        {
            "deposits_csv": DEPOSITS_CSV,
            "mca_xwalk_csv": MCA_XWALK_CSV,
            "demo_csv": DEMO_CSV,
            "pix_csv": PIX_CSV,
            "anatel_csv": ANATEL_CSV,
            "inss_csv": INSS_CSV,
            "inclusion_csv": INCLUSION_CSV,
            "cadunico_csv": CADUNICO_CSV,
            "tarifas_fee_csv": TARIFAS_FEE_CSV,
            "bank_chars_csv": BANK_CHARS_CSV,
            "output_csv": OUTPUT_CSV,
        },
        script_dir=os.path.dirname(os.path.abspath(__file__)),
    )
    DEPOSITS_CSV = _paths["deposits_csv"]
    MCA_XWALK_CSV = _paths["mca_xwalk_csv"]
    DEMO_CSV = _paths["demo_csv"]
    PIX_CSV = _paths["pix_csv"]
    ANATEL_CSV = _paths["anatel_csv"]
    INSS_CSV = _paths["inss_csv"]
    INCLUSION_CSV = _paths["inclusion_csv"]
    CADUNICO_CSV = _paths["cadunico_csv"]
    TARIFAS_FEE_CSV = _paths["tarifas_fee_csv"]
    BANK_CHARS_CSV = _paths["bank_chars_csv"]
    OUTPUT_CSV = _paths["output_csv"]


## -----------------------------------------------------------------------------
## 2) HELPERS
## -----------------------------------------------------------------------------

def _try_load(path: str, desc: str, **kwargs) -> pd.DataFrame | None:
    """Load a CSV if it exists; log a warning and return None otherwise."""
    if not os.path.exists(path):
        logging.warning(f"File not found (will be NaN in output): {path}  [{desc}]")
        return None
    kwargs.setdefault("encoding", "latin-1")
    df = pd.read_csv(path, **kwargs)
    logging.info(f"Loaded {desc}: {len(df):,} rows")
    return df


def _broadcast_annual_to_quarters(df: pd.DataFrame,
                                   year_col: str = "year") -> pd.DataFrame:
    """
    Given an annual panel, create four copies -- one per quarter -- and return
    a long DataFrame with an added 'quarter' column (1-4).
    """
    frames = []
    for q in (1, 2, 3, 4):
        tmp = df.copy()
        tmp["quarter"] = q
        frames.append(tmp)
    return pd.concat(frames, ignore_index=True)


## -----------------------------------------------------------------------------
## 3) LOAD & PREPARE DEPOSITS PANEL
## -----------------------------------------------------------------------------

def load_deposits() -> pd.DataFrame:
    dep = pd.read_csv(DEPOSITS_CSV, low_memory=False,
                      dtype={"CODMUN_IBGE": "Int64",
                             "CodConglomeradoPrudencial": str,
                             "CNPJ_Lider": str})
    # Normalise column names to lower-case year/quarter for consistent merging
    dep.rename(columns={"Year": "year", "Quarter": "quarter"}, inplace=True)
    dep["year"]    = dep["year"].astype(int)
    dep["quarter"] = dep["quarter"].astype(int)
    logging.info(f"Loaded deposits panel: {len(dep):,} rows  "
                 f"({dep['CodConglomeradoPrudencial'].nunique()} conglomerates  "
                 f"years {dep['year'].min()}-{dep['year'].max()})")
    return dep


## -----------------------------------------------------------------------------
## 4) ADD mca_code TO DEPOSIT PANEL (Tier-1 rows only)
## -----------------------------------------------------------------------------

def load_digital_conglomerates(digital_flags_csv: str) -> tuple[set, pd.DataFrame]:
    """
    Conglomerate-level digital flag.

    panel_5_flag_digital.py flags digital candidates at the level of the ESTBAN reporting
    BANK (CNPJ_root). The deposit panel is keyed on the prudential CONGLOMERATE
    (CodConglomeradoPrudencial), whose leader need not be that bank: PicPay Bank
    (09516419) sits in C0088022 led by the payment institution 22896431, Omni Banco in
    C0080460 led by the Omni CFI. Keying the flag on the leader CNPJ therefore misses
    them, and keying it on "any member is a candidate" would flag Bradesco (via Bradesco
    Financiamentos) or Safra (via J. Safra). The rule used here: a conglomerate is digital
    iff EVERY ESTBAN bank that panel_1 maps into it is a digital candidate, i.e. the
    conglomerate's whole ESTBAN deposit footprint is single-municipality.

    Bank -> conglomerate uses panel_1's build_cnpj_conglomerate_map, the same lookup that
    built deposits_panel.csv (latest List spell wins; unmapped banks fall back to
    'CNPJ_<cnpj>' exactly as panel_1 does), so the two sides of the join agree.

    Returns (set of digital CodConglomeradoPrudencial, member table for logging).
    """
    from panel_1_deposits import build_cnpj_conglomerate_map

    flags = pd.read_csv(digital_flags_csv)
    flags = flags[flags["Inst_Total_Dep"] > 0].copy()
    cnpj_map = build_cnpj_conglomerate_map()

    def _congl(cnpj) -> str:
        hit = cnpj_map.get(int(cnpj))
        return str(hit[0]) if hit and hit[0] is not None else f"CNPJ_{int(cnpj)}"

    flags["CodConglomeradoPrudencial"] = flags["CNPJ_root"].map(_congl)
    flags["is_digital_candidate"] = flags["is_digital_candidate"].astype(bool)
    # Physical-network evidence behind the verdict (panel_5_flag_digital.py reads it from
    # the BCB access-point data): branch / service-point counts summed over the
    # conglomerate's ESTBAN member banks, so the basis for the B/D call travels with the
    # panel instead of living only in the diagnostic CSV.
    _net_sum = [c for c in ("n_agencias", "n_postos", "n_own_points", "n_mun_own",
                            "n_correspondentes") if c in flags.columns]
    _agg = dict(n_banks=("CNPJ_root", "size"),
                n_candidates=("is_digital_candidate", "sum"),
                banks=("NOME_INSTITUICAO", lambda s: " | ".join(s)))
    _agg.update({c: (c, "sum") for c in _net_sum})
    if "has_access_point_data" in flags.columns:
        _agg["has_access_point_data"] = ("has_access_point_data", "max")
    members = flags.groupby("CodConglomeradoPrudencial").agg(**_agg).reset_index()
    members["is_digital"] = members["n_candidates"] == members["n_banks"]
    digital = set(members.loc[members["is_digital"], "CodConglomeradoPrudencial"])

    for _, r in members[members["is_digital"]].iterrows():
        logging.info(f"  digital conglomerate {r.CodConglomeradoPrudencial}: {r.banks}")
    mixed = members[(~members["is_digital"]) & (members["n_candidates"] > 0)]
    for _, r in mixed.iterrows():
        logging.info(f"  candidate bank(s) inside a branch-network conglomerate, NOT digital: "
                     f"{r.CodConglomeradoPrudencial} ({r.n_candidates}/{r.n_banks}): {r.banks}")
    return digital, members


NETWORK_EVIDENCE_COLS = ["n_agencias", "n_postos", "n_own_points", "n_mun_own",
                         "n_correspondentes", "has_access_point_data"]


def attach_network_evidence(dep: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    """Merge the conglomerate-level physical-network evidence onto the deposit panel.

    These columns are the observable basis for the is_B verdict: branch, service-point and
    correspondent counts from the BCB access-point data, summed over the conglomerate's
    ESTBAN member banks by load_digital_conglomerates. Rows for conglomerates with no
    ESTBAN footprint (IF-Data national aggregates) get NaN, which is itself the reason
    those rows are D.

    `members` is one row per conglomerate, so this cannot change the row count.
    """
    cols = [c for c in NETWORK_EVIDENCE_COLS if c in members.columns]
    if not cols:
        logging.warning("digital_banks_diagnostic.csv carries no physical-network columns; "
                        "is_B is stored without its supporting evidence.")
        return dep

    ev = members[["CodConglomeradoPrudencial", "is_digital"] + cols].copy()
    ev = ev.rename(columns={"is_digital": "congl_is_digital"})
    ev["CodConglomeradoPrudencial"] = ev["CodConglomeradoPrudencial"].astype(str)
    # Nullable dtypes: the left join leaves NA on rows whose conglomerate has no ESTBAN
    # member bank, and a plain int/bool column would be upcast to float/object there,
    # writing "1.0" or an unparseable mix into the CSV.
    ev["congl_is_digital"] = ev["congl_is_digital"].astype("boolean")
    if "has_access_point_data" in ev.columns:
        ev["has_access_point_data"] = ev["has_access_point_data"].astype("boolean")
    for c in [c for c in cols if c != "has_access_point_data"]:
        ev[c] = pd.to_numeric(ev[c], errors="coerce").astype("Int64")

    n_before = len(dep)
    idx = dep.index
    dep = dep.copy()
    dep["_congl_key"] = dep["CodConglomeradoPrudencial"].astype(str)
    dep = dep.merge(ev.rename(columns={"CodConglomeradoPrudencial": "_congl_key"}),
                    on="_congl_key", how="left")
    dep.drop(columns=["_congl_key"], inplace=True)
    assert len(dep) == n_before, f"network-evidence merge changed rows {n_before:,} -> {len(dep):,}"
    # merge() hands back a fresh RangeIndex; restore the caller's so masks built on the
    # pre-merge frame keep aligning.
    dep.index = idx

    n_matched = int(dep["congl_is_digital"].notna().sum())
    logging.info(
        f"Physical-network evidence merged onto {n_matched:,} of {len(dep):,} deposit rows "
        f"({', '.join(cols)}); the rest have no ESTBAN member bank in the diagnostic."
    )
    return dep


def count_type_tier_disagreement(df: pd.DataFrame) -> int:
    """Rows where the firm-type verdict and the market tier disagree.

    is_B says whether the firm runs a physical network; mca_code says whether its deposits
    sit in a local market. A brick-and-mortar firm whose whole ESTBAN footprint is booked
    at one head office is is_B=1 with no honest local market, and a digital firm re-keyed
    to the Tier-2 sentinel is is_B=0 and NATIONAL. Nothing forces the two to agree; this
    is the measurement, not a repair.
    """
    if "is_B" not in df.columns or "mca_code" not in df.columns:
        return 0
    local_market = (df["mca_code"] != "NATIONAL").astype("int8")
    return int((df["is_B"].fillna(-1).astype("int8") != local_market).sum())


def attach_mca_code(dep: pd.DataFrame) -> pd.DataFrame:
    """
    Map CODMUN_IBGE (7-digit) -> mca_code using the MCA crosswalk.
    Tier-1 (ESTBAN) rows have a real municipality code.
    Tier-2 (IF Data national, CODMUN_IBGE = 0) get mca_code = 'NATIONAL'.

    This is also the one site where the firm-type verdict is decided and stored, as
    `is_B` (1 = brick-and-mortar, 0 = D / digital). Every downstream step reads that
    column instead of re-testing CODMUN_IBGE.

    Firm type and market assignment are separate facts. `is_B` comes from the
    physical-network verdict in digital_banks_diagnostic.csv, plus the Tier-2 sentinel
    for rows that arrive with no ESTBAN municipality at all; `mca_code` says which market
    the row deposits sit in. A brick-and-mortar firm that books its whole ESTBAN
    footprint at one head office has no honest local market, so the two can disagree.
    They are not reconciled here -- the disagreement is counted and reported.
    """
    xwalk = pd.read_csv(MCA_XWALK_CSV,
                        usecols=["municipality_code", "mca_code", "year"],
                        dtype={"municipality_code": int, "mca_code": str, "year": int})

    dep = dep.copy()

    # Tier-2 sentinel
    dep["mca_code"] = "NATIONAL"

    # Rows arriving with no ESTBAN municipality are IF-Data national aggregates: D for a
    # reason independent of the network verdict below. Recorded before the digital
    # re-keying overwrites CODMUN_IBGE.
    tier2_sentinel = (dep["CODMUN_IBGE"].isna() | (dep["CODMUN_IBGE"] == 0)).fillna(True).astype(bool)

    # Digital conglomerates (see load_digital_conglomerates): every ESTBAN bank in the
    # conglomerate is a panel_5 digital candidate. Their ESTBAN rows carry a real
    # CODMUN_IBGE (the single municipality the bank books all deposits in) and are
    # re-keyed to the Tier-2 sentinel so they become NATIONAL like the IF-Data fintechs.
    mask_dig = pd.Series(False, index=dep.index)
    members = None
    digital_flags = os.path.join(PANEL_DIR, "PANEL_INTERMED", "digital_banks_diagnostic.csv")
    if os.path.exists(digital_flags):
        digital_congls, members = load_digital_conglomerates(digital_flags)
        mask_dig = dep["CodConglomeradoPrudencial"].astype(str).isin(digital_congls)
        n_flip = int((mask_dig & dep["CODMUN_IBGE"].notna() & (dep["CODMUN_IBGE"] != 0)).sum())
        dep.loc[mask_dig, "CODMUN_IBGE"] = 0
        logging.info(
            f"Digital conglomerates: {len(digital_congls)} flagged, "
            f"{dep.loc[mask_dig, 'CodConglomeradoPrudencial'].nunique()} present in the deposit "
            f"panel; CODMUN_IBGE=0 set on {mask_dig.sum():,} rows ({n_flip:,} were Tier-1)."
        )
        _absent = sorted(digital_congls - set(dep["CodConglomeradoPrudencial"].astype(str)))
        if _absent:
            logging.warning(f"Digital conglomerates with no deposit-panel rows: {_absent}")
    else:
        logging.warning(
            f"{digital_flags} not found: no physical-network verdict is available, so is_B "
            f"rests on the Tier-2 sentinel (CODMUN_IBGE == 0) alone. Run "
            f"panel_5_flag_digital.py."
        )

    # is_B is the pipeline authoritative firm-type column: 1 = brick-and-mortar (the firm
    # runs a physical network), 0 = D (digital, or a national aggregate with no ESTBAN
    # footprint). Decided HERE and nowhere else. Stored as int8: market_panel.csv is
    # rewritten several times in stage 4 by two different writers (pyarrow emits lowercase
    # true/false for booleans, pandas emits True/False), and 0/1 reads back identically
    # under both, whereas a boolean that is ever parsed as text would make astype(bool)
    # return True for the string "false".
    dep["is_B"] = (~(mask_dig.astype(bool) | tier2_sentinel)).astype("int8")
    logging.info(
        f"Firm type: is_B=1 on {int((dep['is_B'] == 1).sum()):,} rows (brick-and-mortar), "
        f"is_B=0 on {int((dep['is_B'] == 0).sum()):,} rows "
        f"({int((mask_dig & ~tier2_sentinel).sum()):,} by the network verdict, "
        f"{int(tier2_sentinel.sum()):,} with no ESTBAN municipality)."
    )

    # The evidence behind that verdict rides along with it.
    if members is not None:
        dep = attach_network_evidence(dep, members)

    # Merge mca_code for Tier-1 rows (CODMUN_IBGE is a real 7-digit code)
    tier1_mask = dep["CODMUN_IBGE"].notna() & (dep["CODMUN_IBGE"] != 0)

    tier1 = dep[tier1_mask].copy()
    tier1["CODMUN_IBGE_int"] = tier1["CODMUN_IBGE"].astype(int)

    tier1_merged = tier1.merge(
        xwalk.rename(columns={"municipality_code": "CODMUN_IBGE_int"}),
        on=["CODMUN_IBGE_int", "year"],
        how="left",
        suffixes=("", "_xwalk"),
    )

    n_missing = tier1_merged["mca_code_xwalk"].isna().sum()
    if n_missing > 0:
        logging.warning(
            f"{n_missing:,} Tier-1 deposit rows could not be matched to an MCA "
            f"(municipality not in crosswalk for that year)."
        )

    # Write resolved mca_code back to the main df. Unmapped municipalities get their OWN
    # pseudo-market ("UNKNOWN_<codmun>") rather than a shared "UNKNOWN" bucket: the panel is
    # aggregated to mca_code downstream, and a shared bucket would fuse unrelated
    # municipalities into one fake market.
    tier1_merged["mca_code"] = tier1_merged["mca_code_xwalk"]
    _unmapped = tier1_merged["mca_code"].isna()
    if _unmapped.any():
        tier1_merged.loc[_unmapped, "mca_code"] = (
            "UNKNOWN_" + tier1_merged.loc[_unmapped, "CODMUN_IBGE_int"].astype(str)
        )
        logging.warning(
            f"{int(_unmapped.sum()):,} unmapped Tier-1 rows across "
            f"{tier1_merged.loc[_unmapped, 'CODMUN_IBGE_int'].nunique():,} municipalities kept as "
            f"standalone UNKNOWN_<codmun> pseudo-markets (not fused)."
        )
    tier1_merged.drop(columns=["mca_code_xwalk", "CODMUN_IBGE_int"], inplace=True)

    # Recombine
    tier2 = dep[~tier1_mask].copy()
    result = pd.concat([tier1_merged, tier2], ignore_index=True)

    logging.info(
        f"MCA mapping: {result['mca_code'].nunique()} unique mca_code values  "
        f"(including NATIONAL={( result['mca_code'] == 'NATIONAL').sum():,} rows)"
    )
    logging.info(
        f"Firm type vs market tier: {count_type_tier_disagreement(result):,} rows where is_B "
        f"disagrees with (mca_code != 'NATIONAL'). Separate facts, left as they fall."
    )
    return result


## -----------------------------------------------------------------------------
## 4b) COLLAPSE MUNICIPALITIES -> MCA (the panel's documented unit of observation)
## -----------------------------------------------------------------------------

def aggregate_to_mca(dep: pd.DataFrame) -> pd.DataFrame:
    """Collapse the municipality-level deposit rows to
    CodConglomeradoPrudencial x mca_code x year x quarter.

    ESTBAN arrives as a municipality-month panel, but a market is an MCA (V_Main
    Sec. 2: "B firms compete in local markets - defined here as Minimal Comparable
    Areas"; Sec. 3 descriptives "sum MCA-level deposits within firm-quarter"). Without
    this step the panel carries ~3.84 municipality rows per market while every merged
    characteristic is already MCA-level, and downstream consumers silently
    drop_duplicates() to one arbitrary municipality -- discarding ~74% of the rows and
    understating market deposits.

    Deposit LEVELS are summed across the MCA's municipalities; every other column is
    constant within the key (market characteristics are MCA-level from the scrapers;
    spreads and bank characteristics are conglomerate-level), so first() is exact.

    NATIONAL (D-firm) rows: IF-Data rows are already one-per-key, so the sum is a no-op;
    the ESTBAN "retail digital candidate" rows zeroed in attach_mca_code() are
    municipality splits of a national bank, so summing rebuilds the national total. No
    conglomerate-quarter mixes the two sources, so this cannot double-count.
    """
    key = ["CodConglomeradoPrudencial", "mca_code", "year", "quarter"]
    dep_cols = [c for c in dep.columns if c.startswith("dep_") or c == "total_deposits"]
    other = [c for c in dep.columns if c not in key + dep_cols]

    n_before = len(dep)
    n_keys = dep.drop_duplicates(subset=key).shape[0]
    if n_before == n_keys:
        logging.info("Deposit panel already unique at conglomerate x MCA x quarter; no aggregation needed.")
        return dep

    # min_count=1 so an all-NaN group stays NaN instead of collapsing to 0.0 -- dep_a5
    # (prepaid) is legitimately NaN before 2020Q2 and must not become "zero deposits".
    agg_dep = dep.groupby(key, sort=False, dropna=False)[dep_cols].sum(min_count=1)
    agg_oth = dep.groupby(key, sort=False, dropna=False)[other].first()
    out = agg_dep.join(agg_oth).reset_index()

    logging.info(
        f"Aggregated municipalities -> MCA: {n_before:,} rows -> {len(out):,} "
        f"({n_before / max(len(out), 1):.2f} municipality rows per market). "
        f"Summed {len(dep_cols)} deposit column(s): {dep_cols}"
    )
    assert len(out) == n_keys, f"aggregation produced {len(out):,} rows, expected {n_keys:,}"
    return out


## -----------------------------------------------------------------------------
## 5) LOAD MARKET CHARACTERISTIC PANELS
## -----------------------------------------------------------------------------

def load_demographics() -> pd.DataFrame | None:
    """Annual MCA panel: pop_total, gdp_per_capita, fraction_65plus, fraction_young."""
    df = _try_load(DEMO_CSV, "IBGE demographics",
                  dtype={"mca_code": str, "year": int})
    if df is None:
        return None
    # Broadcast to quarters (annual stock variable)
    df = _broadcast_annual_to_quarters(df)
    keep = ["mca_code", "year", "quarter",
            "pop_total", "gdp_per_capita",
            "fraction_65plus", "fraction_young", "age_interpolated",
            "gdp_imputed"]
    return df[[c for c in keep if c in df.columns]]


def load_pix() -> pd.DataFrame | None:
    """Quarterly MCA panel: pix adoption."""
    df = _try_load(PIX_CSV, "PIX MCA panel",
                  dtype={"mca_code": str, "year": int, "quarter": int})
    if df is None:
        return None
    keep = ["mca_code", "year", "quarter",
            "pix_active", "pix_users_pf_per1000",
            "pix_txns_pf", "pix_value_pf_r1000", "pix_users_pj"]
    return df[[c for c in keep if c in df.columns]]


def load_anatel() -> pd.DataFrame | None:
    """Quarterly MCA panel: mobile connectivity."""
    df = _try_load(ANATEL_CSV, "ANATEL mobile panel",
                  dtype={"mca_code": str, "year": int, "quarter": int})
    if df is None:
        return None
    keep = ["mca_code", "year", "quarter",
            "connections_per100", "frac_4g5g"]
    return df[[c for c in keep if c in df.columns]]


def load_inss() -> pd.DataFrame | None:
    """Quarterly MCA panel: retirement beneficiaries."""
    df = _try_load(INSS_CSV, "INSS retirees panel",
                  dtype={"mca_code": str, "year": int, "quarter": int})
    if df is None:
        return None
    keep = ["mca_code", "year", "quarter",
            "retirees_total", "retirees_per1000"]
    return df[[c for c in keep if c in df.columns]]


def load_bcb_inclusion() -> pd.DataFrame | None:
    """Annual MCA panel: banking access-points (broadcast to quarters)."""
    df = _try_load(INCLUSION_CSV, "BCB inclusion panel",
                  dtype={"mca_code": str, "year": int})
    if df is None:
        return None
    df = _broadcast_annual_to_quarters(df)
    keep = ["mca_code", "year", "quarter",
            "branches_per1000", "correspondents_per1000", "access_points_per1000"]
    return df[[c for c in keep if c in df.columns]]


def load_cadunico() -> pd.DataFrame | None:
    """Quarterly MCA panel: CadUnico low-income families."""
    df = _try_load(CADUNICO_CSV, "CadUnico panel",
                  dtype={"mca_code": str, "year": int, "quarter": int})
    if df is None:
        return None
    keep = ["mca_code", "year", "quarter",
            "cadunico_families_per1000", "cadunico_extreme_poverty", "cadunico_poverty"]
    return df[[c for c in keep if c in df.columns]]


## -----------------------------------------------------------------------------
## 6) MERGE EVERYTHING
## -----------------------------------------------------------------------------

def merge_characteristics(dep: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join all market-characteristic panels onto the deposit panel.
    Tier-2 rows (mca_code = 'NATIONAL') receive NaN for all characteristic cols.
    """
    loaders = [
        ("demographics",  load_demographics),
        ("PIX",           load_pix),
        ("ANATEL",        load_anatel),
        ("INSS",          load_inss),
        ("BCB inclusion", load_bcb_inclusion),
        ("CadUnico",      load_cadunico),
    ]

    # Load all six characteristic panels concurrently (all are local CSV reads)
    loaded: dict[str, pd.DataFrame | None] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        future_map = {pool.submit(fn): name for name, fn in loaders}
        for future in concurrent.futures.as_completed(future_map):
            name = future_map[future]
            try:
                loaded[name] = future.result()
            except Exception as exc:
                logging.warning(f"  {name} loading error: {exc}")
                loaded[name] = None

    # Maintain original order for deterministic merge sequence
    panels = [(name, loaded.get(name)) for name, _ in loaders]

    # A missing source used to degrade silently to all-NaN columns, which is how
    # market_panel.csv was once rebuilt with no PIX columns at all.  Every source
    # below feeds the demand-side demographics, so a miss is fatal.  INSS is the
    # sole exception: its panel has never been built and nothing consumes it.
    missing = [n for n, d in panels if d is None and n != "INSS"]
    if missing:
        raise FileNotFoundError(
            "Required characteristic panel(s) not found: " + ", ".join(missing) +
            ". Rebuilding market_panel without them would silently drop demographics."
        )

    result = dep.copy()

    for name, char_df in panels:
        if char_df is None:
            logging.warning(f"  {name} panel missing -- all its columns will be NaN.")
            continue

        char_df = char_df.copy()
        char_df["mca_code"] = char_df["mca_code"].astype(str)
        char_df["year"]     = char_df["year"].astype(int)
        char_df["quarter"]  = char_df["quarter"].astype(int)

        # Drop any columns already in result (except merge keys) to avoid _x/_y
        existing = set(result.columns) - {"mca_code", "year", "quarter"}
        char_df  = char_df.drop(columns=[c for c in char_df.columns
                                          if c in existing], errors="ignore")

        result = result.merge(char_df, on=["mca_code", "year", "quarter"],
                              how="left")
        logging.info(f"  Merged {name}: result now {len(result):,} rows "
                     f"x {len(result.columns)} cols")

    return result


## ---------------------------------------------------------------------------------
## 6.5) LOAD AND MERGE TARIFF FEE SUMMARY
## ---------------------------------------------------------------------------------
#
# Tariff fees are conglomerate-level product characteristics (not MCA-level).
# They are joined on CodConglomeradoPrudencial x year x quarter using the
# most recent tariff snapshot available on or before the quarter’s end date.
# With only a single snapshot the join degenerates to a simple left-merge.
# ---------------------------------------------------------------------------------

def load_tarifas() -> pd.DataFrame | None:
    """Load the fee summary panel produced by tarifas_scrape_1.py."""
    if not os.path.exists(TARIFAS_FEE_CSV):
        logging.warning(
            f"Tarifas fee summary not found -- fee columns will be NaN: {TARIFAS_FEE_CSV}"
        )
        return None
    df = pd.read_csv(TARIFAS_FEE_CSV, dtype={"cod_cong_prudencial": str})
    df["data_coleta"] = pd.to_datetime(df["data_coleta"], errors="coerce")
    logging.info(
        f"Loaded tarifas fee summary: {len(df):,} rows  "
        f"({df['cod_cong_prudencial'].nunique()} conglomerates  "
        f"{len(df.columns) - 2} fee columns)"
    )
    return df


def merge_tarifas(panel: pd.DataFrame, tarifs: pd.DataFrame | None) -> pd.DataFrame:
    """
    Left-join tariff fee columns onto the market panel on CodConglomeradoPrudencial.

    Time matching
    -------------
    For each panel (year, quarter), the most recent scrape whose data_coleta falls
    on or before the quarter’s end date is used.  With a single snapshot this is
    trivially the one available snapshot.  When multiple scrapes exist the join
    becomes a time-varying last-observation-carried-forward.
    """
    if tarifs is None or tarifs.empty:
        return panel

    # Build a quarter-end date for each (year, quarter) in the panel
    qends = (
        panel[["year", "quarter"]]
        .drop_duplicates()
        .copy()
    )
    # Quarter Q ends on month Q*3, last day: use period arithmetic
    qends["qend"] = pd.PeriodIndex(
        qends["year"].astype(str) + "Q" + qends["quarter"].astype(str), freq="Q"
    ).to_timestamp(how="end").normalize()

    # For each (CodConglPrud, year, quarter), pick the most recent scrape date
    # that is <= qend.  Build a merge table: (CodConglPrud, year, quarter) -> fees.
    snapshots = tarifs["data_coleta"].sort_values().unique()  # sorted asc
    as_of_rows: list[dict] = []
    fee_cols = [c for c in tarifs.columns if c.startswith("fee_")]

    for _, qrow in qends.iterrows():
        valid = [s for s in snapshots if s <= qrow["qend"]]
        if not valid:
            continue
        snap = max(valid)  # most recent snapshot <= quarter end
        snap_fees = tarifs[tarifs["data_coleta"] == snap].copy()
        snap_fees = snap_fees.drop(columns=["data_coleta"])
        snap_fees["year"]    = qrow["year"]
        snap_fees["quarter"] = qrow["quarter"]
        as_of_rows.append(snap_fees)

    if not as_of_rows:
        logging.warning("merge_tarifas: no valid snapshot found for any panel quarter.")
        return panel

    as_of_df = pd.concat(as_of_rows, ignore_index=True)
    as_of_df = as_of_df.rename(columns={"cod_cong_prudencial": "CodConglomeradoPrudencial"})
    as_of_df["CodConglomeradoPrudencial"] = as_of_df["CodConglomeradoPrudencial"].astype(str)

    before = len(panel)
    panel = panel.merge(
        as_of_df,
        on=["CodConglomeradoPrudencial", "year", "quarter"],
        how="left",
    )
    assert len(panel) == before, "merge_tarifas unexpectedly duplicated rows"

    matched = panel[fee_cols[0]].notna().sum() if fee_cols else 0
    logging.info(
        f"  Tarifas merged: {len(fee_cols)} fee columns  "
        f"{matched:,}/{len(panel):,} rows matched"
    )
    return panel


## -----------------------------------------------------------------------------
## 7) FILL NATIONAL ROWS WITH POPULATION-WEIGHTED NATIONAL AVERAGES
## -----------------------------------------------------------------------------
#
# Fintechs / digital banks (Tier-2 rows, mca_code = 'NATIONAL') operate
# nationally and have no single local market.  For the BLP demand estimation
# they need well-defined market-characteristic controls and instruments.
# We assign each variable a population-weighted national average computed from
# Tier-1 (ESTBAN) rows for the same (year, quarter), using MCA pop_total as
# weights.  Two exceptions use sums instead of weighted means:
#   * pop_total      -> national population (sum of all MCAs)
#   * retirees_total -> national retiree count (sum of all MCAs)
# -----------------------------------------------------------------------------

# Columns aggregated as national total (sum)
_NATL_SUM_COLS: list[str] = ["pop_total", "retirees_total"]

# Columns aggregated as population-weighted mean
_NATL_RATE_COLS: list[str] = [
    "gdp_per_capita",
    "fraction_65plus", "fraction_young",
    "pix_users_pf_per1000", "pix_txns_pf", "pix_value_pf_r1000", "pix_users_pj",
    "connections_per100", "frac_4g5g",
    "retirees_per1000",
    "branches_per1000", "correspondents_per1000", "access_points_per1000",
    "cadunico_families_per1000", "cadunico_extreme_poverty", "cadunico_poverty",
]


def fill_national_averages(panel: pd.DataFrame) -> pd.DataFrame:
    """
    For Tier-2 rows (mca_code = 'NATIONAL'), replace NaN market-characteristic
    columns with population-weighted national averages derived from Tier-1 rows.

    Rationale
    ---------
    A fintech that operates nationally faces, in expectation, the average
    Brazilian market conditions weighted by where the population lives.  Giving
    the NATIONAL rows meaningful covariate values (a) allows the same IV/control
    strategy as for local banks and (b) prevents the BLP outside-good share from
    soaking up all macro variation.

    Aggregation rules
    -----------------
    * pop_total, retirees_total  -> sum across all MCAs  (true national total)
    * all rate / per-X columns   -> population-weighted mean across MCAs
    """
    nat_mask = panel["mca_code"] == "NATIONAL"
    if nat_mask.sum() == 0:
        return panel

    tier1 = panel[~nat_mask].copy()
    if tier1.empty or "pop_total" not in tier1.columns:
        logging.warning(
            "fill_national_averages: no Tier-1 rows or pop_total missing -- "
            "NATIONAL rows remain NaN."
        )
        return panel

    # -- deduplicate to one row per MCA x year x quarter for aggregation ----
    # (the deposit panel has many bank rows per MCA; we only need one copy of
    # the market characteristics, so drop-duplication on the key + char columns)
    char_cols = (
        [c for c in _NATL_SUM_COLS + _NATL_RATE_COLS if c in tier1.columns]
    )
    t1 = (
        tier1[["mca_code", "year", "quarter"] + char_cols]
        .drop_duplicates(subset=["mca_code", "year", "quarter"])
        .copy()
    )
    t1["_w"] = t1["pop_total"].fillna(0.0)

    # -- sum columns --------------------------------------------------------
    sum_cols_present = [c for c in _NATL_SUM_COLS if c in t1.columns]
    natl_sums = (
        t1.groupby(["year", "quarter"])[sum_cols_present]
          .sum(min_count=1)
          .reset_index()
    )

    # -- population-weighted means ------------------------------------------
    rate_cols_present = [c for c in _NATL_RATE_COLS if c in t1.columns]
    wavg_rows: list[dict] = []
    for (yr, qt), grp in t1.groupby(["year", "quarter"]):
        row: dict = {"year": yr, "quarter": qt}
        w = grp["_w"].values
        for col in rate_cols_present:
            v = grp[col].values.astype(float)
            mask = ~np.isnan(v) & (w > 0)
            row[col] = float(np.average(v[mask], weights=w[mask])) if mask.any() else np.nan
        wavg_rows.append(row)
    natl_rates = pd.DataFrame(wavg_rows) if wavg_rows else pd.DataFrame(
        columns=["year", "quarter"] + rate_cols_present
    )

    # -- combine sum + rate aggregates ------------------------------------
    natl = natl_sums.merge(natl_rates, on=["year", "quarter"], how="outer")

    # -- fill NaN cells in NATIONAL rows ----------------------------------
    panel = panel.copy()
    fill_cols = [c for c in sum_cols_present + rate_cols_present if c in panel.columns]
    panel_nat = panel[nat_mask].copy().merge(
        natl, on=["year", "quarter"], how="left", suffixes=("", "_natl")
    )
    for col in fill_cols:
        src = col + "_natl"
        if src in panel_nat.columns:
            panel_nat[col] = panel_nat[col].fillna(panel_nat[src])
            panel_nat.drop(columns=[src], inplace=True)

    panel = pd.concat([panel[~nat_mask], panel_nat], ignore_index=True)

    nat_mask2 = panel["mca_code"] == "NATIONAL"
    n_filled  = panel.loc[nat_mask2, fill_cols].notna().sum().sum()
    logging.info(
        f"fill_national_averages: filled {nat_mask.sum():,} NATIONAL rows "
        f"with pop-weighted national aggregates "
        f"across {tier1['mca_code'].nunique()} MCAs "
        f"({n_filled:,} cell values written)"
    )
    return panel


## -----------------------------------------------------------------------------
## 7b) BANK CHARS
## -----------------------------------------------------------------------------

def load_bank_chars() -> pd.DataFrame | None:
    return _try_load(BANK_CHARS_CSV, "Bank chars panel")

def merge_bank_chars(panel: pd.DataFrame, bank_chars: pd.DataFrame | None) -> pd.DataFrame:
    """Left-join bank sizes and characteristics onto the panel on CodConglomeradoPrudencial x year x quarter."""
    if bank_chars is None or bank_chars.empty:
        return panel
    
    panel = panel.merge(
        bank_chars,
        on=["CodConglomeradoPrudencial", "year", "quarter"],
        how="left"
    )

    # Co-ops and state-owned banks are strictly identified. Missing merges default to 0.0 (private/standard)
    if 'is_coop' in panel.columns:
        panel['is_coop'] = panel['is_coop'].fillna(0.0).astype(float)
    if 'is_state_owned' in panel.columns:
        panel['is_state_owned'] = panel['is_state_owned'].fillna(0.0).astype(float)

    return panel

## -----------------------------------------------------------------------------
## 8) FINAL COLUMN ORDERING & SAVE
## -----------------------------------------------------------------------------

def calculate_hausman_iv_wide(panel: pd.DataFrame) -> pd.DataFrame:
    """Calculate the leave-one-out mean spread for each deposit type (a1 through a5)."""
    for k in [1, 2, 3, 4, 5]:
        spread_col = f'spread_a{k}'
        iv_col = f'leave_one_out_mean_spread_a{k}'
        if spread_col in panel.columns:
            group_cols = ['year', 'quarter']
            g_count = panel.groupby(group_cols)[spread_col].transform(lambda x: x.notnull().sum())
            g_sum = panel.groupby(group_cols)[spread_col].transform(lambda x: x.sum())
            panel[iv_col] = (g_sum - panel[spread_col].fillna(0)) / (g_count - panel[spread_col].notnull().astype(int)).clip(lower=1)
            panel.loc[g_count <= 1, iv_col] = float('nan')
    return panel

_COL_ORDER = [
    # Identifiers
    "CodConglomeradoPrudencial", "CNPJ_Lider", "NomeInstituicao",
    "CODMUN_IBGE", "mca_code", "is_B",
    "congl_is_digital", "n_agencias", "n_postos", "n_own_points", "n_mun_own",
    "n_correspondentes", "has_access_point_data",
    "year", "quarter",
    "Source",
    # Deposit balances
    "dep_a1", "dep_a2", "dep_a3", "dep_a4", "dep_outros", "dep_a5", "total_deposits",
    "spread_a1", "spread_a2", "spread_a3", "spread_a4", "spread_a5",
    "leave_one_out_mean_spread_a1", "leave_one_out_mean_spread_a2",
    "leave_one_out_mean_spread_a3", "leave_one_out_mean_spread_a4",
    "leave_one_out_mean_spread_a5",
    # Market size
    "pop_total", "gdp_per_capita", "gdp_imputed",
    # Demographics
    "fraction_65plus", "fraction_young", "age_interpolated",
    # Digital adoption
    "pix_active", "pix_users_pf_per1000", "pix_txns_pf", "pix_value_pf_r1000",
    "pix_users_pj",
    # Mobile connectivity
    "connections_per100", "frac_4g5g",
    # Retirees (elderly income dependency)
    "retirees_total", "retirees_per1000",
    # Banking access
    "branches_per1000", "correspondents_per1000", "access_points_per1000",
    # Poverty / unbanked proxy
    "cadunico_families_per1000", "cadunico_extreme_poverty", "cadunico_poverty",
    # Bank characteristics
    "total_assets", "log_total_assets", "equity", "equity_ratio",
    "has_ip", "is_coop", "is_state_owned", "is_captive", "segment", "seg_S2", "seg_S3", "seg_S4", "seg_S5",
]



def save(panel: pd.DataFrame) -> None:
    import pyarrow as pa
    import pyarrow.csv as pa_csv

    # is_B is the authoritative firm-type column; it must reach the CSV as 0/1 and it must
    # reach it unmodified. Firm type and market tier are separate facts, so a disagreement
    # with mca_code is reported, never reconciled.
    if "is_B" not in panel.columns:
        raise AssertionError(
            "is_B is missing from the market panel -- attach_mca_code() must set it."
        )
    panel["is_B"] = panel["is_B"].fillna(0).astype("int8")
    _disagree = count_type_tier_disagreement(panel)
    if _disagree:
        _rows = panel.loc[panel["is_B"].astype(bool) != (panel["mca_code"] != "NATIONAL")]
        logging.warning(
            f"is_B disagrees with the market tier on {_disagree:,} rows across "
            f"{_rows['CodConglomeradoPrudencial'].nunique()} conglomerate(s): "
            f"{sorted(_rows['CodConglomeradoPrudencial'].astype(str).unique())[:10]}. "
            f"Firm type and market assignment are separate facts; both are written as-is."
        )
    else:
        logging.info("is_B agrees with the market tier (mca_code != 'NATIONAL') on every row.")

    # Put columns in desired order, then any remaining columns alphabetically
    ordered   = [c for c in _COL_ORDER if c in panel.columns]
    remaining = sorted([c for c in panel.columns if c not in ordered])
    panel     = panel[ordered + remaining]

    # Hard cap at 2025-Q4: drop any later quarters present in the merged inputs.
    _cap = PANEL_END_YEAR * 4 + PANEL_END_QUARTER
    _n_before = len(panel)
    panel = panel[~((panel["year"] * 4 + panel["quarter"]) > _cap)].copy()
    if (_dropped := _n_before - len(panel)):
        logging.info(f"Capped market panel at {PANEL_END_YEAR}-Q{PANEL_END_QUARTER}: dropped {_dropped:,} later-quarter rows.")

    panel.sort_values(
        ["CodConglomeradoPrudencial", "mca_code", "year", "quarter"],
        inplace=True,
    )
    panel.reset_index(drop=True, inplace=True)

    # pyarrow CSV writer is ~5x faster than pandas to_csv for large files.
    # Writes UTF-8 (downstream readers all default to UTF-8 anyway).
    table = pa.Table.from_pandas(panel, preserve_index=False)
    pa_csv.write_csv(table, OUTPUT_CSV)
    logging.info(f"Saved market panel to {OUTPUT_CSV}")
    # Keep the Parquet sidecar in step with the CSV. estimation_1_sleep reads the panel
    # through utils.load_panel_cached, which serves that sidecar, so leaving it stale
    # here would hand the sleepiness estimation the previous build.
    from utils import refresh_panel_cache
    refresh_panel_cache(OUTPUT_CSV, panel)


def get_pure_wholesale_cnpjs() -> set:
    import glob
    if_files = sorted(glob.glob(os.path.join(str(paths.IF_DATA_LIST), "IF_DATA_List*.csv")))
    
    cong_tags = {}
    for f in if_files:
        df = pd.read_csv(f, sep=None, engine='python')
        if 'CodConglomeradoPrudencial' not in df.columns: continue
        df = df.dropna(subset=['CodConglomeradoPrudencial'])

        # Vectorised: group by conglomerate, update sets in bulk (avoids iterrows)
        df['_cp'] = df['CodConglomeradoPrudencial'].astype(str).str.strip().str.replace('.0', '', regex=False)
        for cp_val, grp in df.groupby('_cp', sort=False):
            if cp_val not in cong_tags:
                cong_tags[cp_val] = {'ativ': set(), 'seg': set(), 'nomes': set()}
            if 'Atividade' in grp.columns:
                cong_tags[cp_val]['ativ'].update(
                    grp['Atividade'].dropna().astype(str).str.strip().str.lower())
            if 'SegmentoTb' in grp.columns:
                cong_tags[cp_val]['seg'].update(
                    grp['SegmentoTb'].dropna().astype(str).str.strip().str.lower())
            if 'NomeInstituicao' in grp.columns:
                cong_tags[cp_val]['nomes'].update(
                    grp['NomeInstituicao'].dropna().astype(str).str.strip().str.lower())
                
    blacklist_seg = ['banco comercial estrangeiro', 'sociedade distribuidora de tvm', 'banco de investimento', 'sociedade corretora de tvm', 'sociedade corretora de câmbio', 'agência de fomento']
    blacklist_ativ = ['tesouraria e negócios', 'crédito atacado', 'filial estrangeiro', 'câmbio', 'tesouraria e negocios', 'credito atacado', 'cambio']
    whitelist_names = ['nupagamentos', 'inter', 'c6', 'nubank', 'picpay', 'pagseguro', 'mercadopago', 'btg', 'bs2', 'genial', 'sofisa', 'rendimento']
    
    def is_pure_investment(cp):
        tags = cong_tags[cp]
        for nm in tags['nomes']:
            for w in whitelist_names:
                if w in nm: return False
                
        segs = tags['seg']
        ativs = tags['ativ']
        
        if not segs and not ativs: return False
        
        all_black_seg = all(any(bad in s for bad in blacklist_seg) for s in segs) if segs else False
        all_black_ativ = all(any(bad in a for bad in blacklist_ativ) for a in ativs) if ativs else False
        return all_black_seg or all_black_ativ

    return {k for k in cong_tags if is_pure_investment(k)}

## -----------------------------------------------------------------------------
## 9) MAIN
## -----------------------------------------------------------------------------

def main() -> None:
    # A. Load deposit panel
    dep = load_deposits()

    # B. Map CODMUN_IBGE -> mca_code
    dep = attach_mca_code(dep)

    # B2. Collapse municipalities -> MCA (the documented unit of observation). Must run
    # BEFORE the merges: fill_national_averages() pop-weights by MCA pop_total (repeated
    # municipality rows would over-weight multi-municipality MCAs by their municipality
    # count) and calculate_hausman_iv_wide() builds a leave-one-out mean over MARKETS.
    dep = aggregate_to_mca(dep)

    # C. Merge all market characteristic panels
    panel = merge_characteristics(dep)

    # C2. Access-point series carry-back. The BCB access-point panel (branches/correspondents/
    # access_points per 1000) begins in 2016 at source, but the deposit panel runs from 2013, so
    # 2013-2015 rows merge to NaN and the demand prep would otherwise fall back to a NATIONAL-median
    # fill. Branch density is a slow-moving municipal stock, so the municipality's own earliest
    # observed (2016) value is a far better estimate: back-fill within mca_code. 2016+ is 100% present,
    # so bfill only touches the leading 2013-2015 gap (no mid-series fill). See counterfactuals_plan.md
    # §0A (access-point source gap).
    _ap_cols = [c for c in ("branches_per1000", "correspondents_per1000", "access_points_per1000")
                if c in panel.columns]
    if _ap_cols:
        panel = panel.sort_values(["mca_code", "year", "quarter"])
        panel[_ap_cols] = panel.groupby("mca_code", sort=False)[_ap_cols].bfill()
        logging.info(f"Access-point carry-back (2016->2013-2015) applied to {_ap_cols}")

    # D. Fill NATIONAL (fintech) rows with pop-weighted national averages
    panel = fill_national_averages(panel)

    # D2. Merge tariff fee columns (conglomerate-level, time-matched)
    tarifs = load_tarifas()
    panel = merge_tarifas(panel, tarifs)

    # D3. Merge bank characteristics (total assets, etc.)
    bank_chars = load_bank_chars()
    panel = merge_bank_chars(panel, bank_chars)

    # Drop Captive Financiers (auto/mutuos PF) from the estimation pool
    before_drop = len(panel)
    panel = panel[panel['is_captive'] != True]
    after_drop = len(panel)
    if before_drop > after_drop:
        logging.info(f"Dropped {before_drop - after_drop} observations belonging to Captive Financiers.")

    # D4. Calculate the Leave-One-Out Mean Spread (Hausman IV) natively over wide format deposit columns
    panel = calculate_hausman_iv_wide(panel)

    # E. Enforce Global Data Exclusions
    # 1. D-type firms (digital, without physical branch network) should launch strictly > 2013
    d_type_early_mask = (panel['is_B'] == 0) & (panel['year'] <= 2013)
    before_d_drop = len(panel)
    panel = panel[~d_type_early_mask].copy()
    after_d_drop = len(panel)
    if before_d_drop > after_d_drop:
        logging.info(f"Dropped {before_d_drop - after_d_drop} early D-type firm observations (year <= 2013).")

    # 2. Exclude deposit type 3 (Interbank/CDI) completely from the dataset 
    cols_to_drop = [c for c in panel.columns if c.endswith("_a3")]
    panel.drop(columns=cols_to_drop, inplace=True, errors="ignore")
    logging.info(f"Dropped deposit type 3 columns: {cols_to_drop}")

    # 3. Exclude strictly wholesale / investment / asset management firms 
    # that never branch into retail universal banking across their entire reporting history.
    before_wholesale_drop = len(panel)
    pure_wholesale = get_pure_wholesale_cnpjs()
    panel = panel[~panel['CodConglomeradoPrudencial'].isin(pure_wholesale)].copy()
    after_wholesale_drop = len(panel)
    if before_wholesale_drop > after_wholesale_drop:
        logging.info(f"Dropped {before_wholesale_drop - after_wholesale_drop} observations belonging to strictly wholesale/investment/asset-management firms.")

    # F. Save
    save(panel)

    # -- Summary ----------------------------------------------------------------
    tier1 = panel[panel["mca_code"] != "NATIONAL"]
    tier2 = panel[panel["mca_code"] == "NATIONAL"]

    def _nn(col: str) -> str:
        # Non-null count for a column that may be absent (e.g. PIX/INSS panels
        # missing on disk) -- keep the summary from crashing the whole step.
        return f"{panel[col].notna().sum():,}" if col in panel.columns else "n/a (column absent)"

    print(
        f"\nMarket panel summary"
        f"\n  Total rows:                  {len(panel):,}"
        f"\n  -- Tier-1 (ESTBAN, local):   {len(tier1):,}  rows  "
         f"({tier1['mca_code'].nunique()} MCAs)"
        f"\n  -- Tier-2 (IF Data, national): {len(tier2):,}  rows"
        f"\n  is_B=1 (brick-and-mortar):    {int((panel['is_B'] == 1).sum()):,}  rows"
        f"\n  is_B=0 (D / digital):        {int((panel['is_B'] == 0).sum()):,}  rows"
        f"\n  is_B vs market-tier disagreements: {count_type_tier_disagreement(panel):,}  rows"
        f"\n  Conglomerates:               {panel['CodConglomeradoPrudencial'].nunique()}"
        f"\n  Years:                       {panel['year'].min()} - {panel['year'].max()}"
        f"\n  Columns:                     {len(panel.columns)}"
        f"\n  pop_total non-null:          {_nn('pop_total')}"
        f"\n  pix_users_pf_per1000 non-null: {_nn('pix_users_pf_per1000')}"
        f"\n  Output:                      {OUTPUT_CSV}"
    )
    print("\nFirst 5 rows (key columns):")
    preview_cols = ["CodConglomeradoPrudencial", "mca_code", "year", "quarter",
                    "dep_a1", "dep_a2", "pop_total", "pix_users_pf_per1000"]
    preview_cols = [c for c in preview_cols if c in panel.columns]
    print(panel[preview_cols].head(5).to_string(index=False))


if __name__ == "__main__":
    main()

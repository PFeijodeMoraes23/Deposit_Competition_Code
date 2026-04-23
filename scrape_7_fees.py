"""
tarifas_scrape_1.py
===================
Scrapes the BCB Tarifas Bancárias API for all regulated institutions and
compiles two panels:

  1. Institution panel  -- (CNPJ x customer_type x data_coleta x service)
  2. Conglomerate panel -- (CodConglomeradoPrudencial x customer_type x data_coleta x service)

API base:
  https://olinda.bcb.gov.br/olinda/servico/
  Informes_ListaTarifasPorInstituicaoFinanceira/versao/v1/odata/

Endpoints used
--------------
  GruposConsolidados
      -> segment groups (01 = Bancos privados, 02 = Bancos públicos, …)
  ListaInstituicoesDeGrupoConsolidado(CodigoGrupoConsolidado='XX')
      -> (Cnpj, Nome) for every institution in that group
  ListaTarifasPorInstituicaoFinanceira(PessoaFisicaOuJuridica='F'|'J', CNPJ='…')
      -> current tariff schedule per institution x customer type

Time dimension
--------------
The BCB API always returns the CURRENT tariff table only (with DataVigencia =
date from which the current rate is effective). Running this script repeatedly
therefore builds a forward-looking time panel. Each run tags rows with the
scrape date (data_coleta), so the long panel grows on every execution.

Output files (all under BCB/Tarifas/)
--------------------------------------
  raw/
      tarifas_raw_YYYYMMDD_pf.csv    -- raw PF snapshot for this run
      tarifas_raw_YYYYMMDD_pj.csv    -- raw PJ snapshot for this run
  processed/
      tarifas_panel_institution_long.csv   -- institution long panel (all runs)
      tarifas_panel_institution_wide.csv   -- institution wide panel (latest data)
      tarifas_panel_conglomerate_long.csv  -- conglomerate long panel (all runs)
      tarifas_panel_conglomerate_wide.csv  -- conglomerate wide panel (latest data)

Conglomerate mapping
--------------------
Uses the most recent IF-Data/List/IF_DATA_List_*.csv file, joining on the
8-digit CNPJ root (tarifas `Cnpj` ↔ IF-Data `CodInst` when CodInst is numeric).

CLI Options:
------------
usage: tarifas_scrape_1.py [-h] [--test N]

BCB Tarifas Bancárias scraper

options:
  -h, --help  show this help message and exit
  --test N    TEST MODE: scrape only the first N institutions (skips raw file
              guard).
"""

# -- stdlib ---------------------------------------------------------------------
import glob
import logging
import os
import time
from datetime import date
from pathlib import Path

# -- third-party ----------------------------------------------------------------
try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None

if ensure_project_venv is not None:
    ensure_project_venv(__file__)

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from utils.toon_runtime import resolve_script_paths
except Exception:
    resolve_script_paths = None

# -- logging --------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Repository / workspace root (this file lives in Code/Egan_et_al_2025_Rep/)
REPO_ROOT = Path(__file__).resolve().parents[2]

# IF-Data list directory for CNPJ -> conglomerate mapping
IF_LIST_DIR = REPO_ROOT / "BCB" / "IF Data" / "List"

# Output root
TARIF_ROOT = REPO_ROOT / "BCB" / "Tarifas"
RAW_DIR   = TARIF_ROOT / "raw"
PROC_DIR  = TARIF_ROOT / "processed"

if resolve_script_paths is not None:
    _paths = resolve_script_paths(
        "tarifas_scrape_1",
        {
            "if_list_dir": str(IF_LIST_DIR),
            "tarif_root": str(TARIF_ROOT),
            "raw_dir": str(RAW_DIR),
            "proc_dir": str(PROC_DIR),
        },
        script_dir=str(Path(__file__).resolve().parent),
    )
    IF_LIST_DIR = Path(_paths["if_list_dir"])
    TARIF_ROOT = Path(_paths["tarif_root"])
    RAW_DIR = Path(_paths["raw_dir"])
    PROC_DIR = Path(_paths["proc_dir"])

# API base
API_BASE = (
    "https://olinda.bcb.gov.br/olinda/servico/"
    "Informes_ListaTarifasPorInstituicaoFinanceira/versao/v1/odata"
)

# Seconds to sleep between institution-level tariff calls (be polite to BCB)
INTER_CALL_SLEEP = 0.5

# Max rows to retrieve from list endpoints
LIST_TOP = 9999

# Customer types to scrape
CUSTOMER_TYPES = ["F", "J"]   # F = Pessoa Física (retail), J = Pessoa Jurídica (corporate)

# ==============================================================================
# KEY SERVICE MAP  (used for the fee summary panel)
# ==============================================================================
# Each entry: (codigo_servico, customer_type, short_column_name)
# Selected services directly relevant to deposit competition / switching costs.
KEY_FEE_SERVICES: list[tuple[str, str, str]] = [
    # -- PF (Pessoa Física) -- retail customers ------------------------------
    ("1101", "F", "pf_account_opening"),       # Confecção de cadastro
    ("1201", "F", "pf_card_2nd_copy_debit"),   # 2ª via cartão débito
    ("1205", "F", "pf_cheque_leaves"),         # Fornecimento de talão
    ("1209", "F", "pf_withdrawal_atm"),        # Saque ATM
    ("1210", "F", "pf_withdrawal_terminal"),   # Saque terminal eletrônico
    ("1211", "F", "pf_withdrawal_corresp"),    # Saque correspondente bancário
    ("1212", "F", "pf_deposit_identified"),    # Depósito identificado
    ("1307", "F", "pf_transfer_own_bank"),     # Transferência mesma instituição
    ("1310", "F", "pf_doc_personal"),          # DOC pessoal (guichê)
    ("1311", "F", "pf_doc_electronic"),        # DOC eletrônico
    ("1312", "F", "pf_doc_internet"),          # DOC internet
    ("1313", "F", "pf_ted_personal"),          # TED pessoal (guichê)
    ("1314", "F", "pf_ted_electronic"),        # TED eletrônico
    ("1315", "F", "pf_ted_internet"),          # TED internet
    ("1401", "F", "pf_overdraft"),             # Adiantamento a depositante
    ("1501", "F", "pf_package_i"),             # Pacote padronizado I (mensal)
    ("1502", "F", "pf_package_ii"),            # Pacote padronizado II (mensal)
    ("1503", "F", "pf_package_iii"),           # Pacote padronizado III (mensal)
    # -- PJ (Pessoa Jurídica) -- corporate customers -------------------------
    ("0101", "J", "pj_ficha_cadastral"),       # Confecção de ficha cadastral
    ("0201", "J", "pj_card_annual"),           # Cartão -- anuidade
    ("0301", "J", "pj_cheque_10"),             # Talão 10 folhas
    ("0401", "J", "pj_account_opening"),       # Abertura de conta
    ("0402", "J", "pj_account_maint"),         # Manutenção de conta ativa (mensal)
    ("0403", "J", "pj_account_inactive"),      # Manutenção de conta inativa
    ("0404", "J", "pj_overdraft"),             # Adiantamento a depositante
    ("0405", "J", "pj_cheque_especial"),       # Concessão de cheque especial
    ("0406", "J", "pj_cheque_especial_renew"), # Renovação de cheque especial
    ("0501", "J", "pj_withdrawal_atm"),        # Saque caixa automática
    ("0502", "J", "pj_doc_c"),                 # Emissão de DOC C
    ("0504", "J", "pj_ordem_pagamento"),       # Ordem de pagamento
    ("0506", "J", "pj_ted"),                   # TED
]

# ==============================================================================
# HELPERS
# ==============================================================================


def build_session(retries: int = 5, backoff: float = 1.5) -> requests.Session:
    """Return a requests session with automatic retry on 429/500/503."""
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"Accept": "application/json"})
    return session


SESSION = build_session()


def odata_get(endpoint: str, params: dict | None = None) -> list[dict]:
    """
    Call an OData endpoint, follow @odata.nextLink pages, return all records.

    Parameters
    ----------
    endpoint : str
        Full URL of the OData entity set.
    params : dict, optional
        Query parameters (do NOT include $format -- it is added automatically).

    Returns
    -------
    list[dict]  All records from all pages.
    """
    if params is None:
        params = {}
    params.setdefault("$format", "json")
    params.setdefault("$top", LIST_TOP)

    records: list[dict] = []
    url: str | None = endpoint

    while url:
        try:
            resp = SESSION.get(url, params=params, timeout=30)
            resp.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            log.error("HTTP %s for %s -- skipping", exc.response.status_code, url)
            break
        except requests.exceptions.RequestException as exc:
            log.error("Request failed for %s: %s -- skipping", url, exc)
            break

        data = resp.json()
        records.extend(data.get("value", []))

        # OData pagination
        url = data.get("@odata.nextLink")
        # On subsequent pages, params are embedded in the next link, so clear them
        params = {}

    return records


# ==============================================================================
# STEP 1 -- FETCH ALL INSTITUTION CNPJs
# ==============================================================================


def fetch_groups() -> pd.DataFrame:
    """Return all GruposConsolidados as a DataFrame (Codigo, Nome)."""
    url = f"{API_BASE}/GruposConsolidados"
    records = odata_get(url)
    df = pd.DataFrame(records)
    log.info("Groups fetched: %d", len(df))
    return df


def fetch_institutions_for_group(group_code: str) -> pd.DataFrame:
    """Return institutions (Cnpj, Nome) for one GrupoConsolidado code."""
    url = f"{API_BASE}/ListaInstituicoesDeGrupoConsolidado(CodigoGrupoConsolidado='{group_code}')"
    records = odata_get(url)
    df = pd.DataFrame(records)
    df["grupo_codigo"] = group_code
    return df


def fetch_all_institutions(groups: pd.DataFrame) -> pd.DataFrame:
    """
    Iterate over all groups and return the union of all institutions.
    Deduplicates by CNPJ, keeping the first occurrence (lowest group code).
    """
    frames: list[pd.DataFrame] = []
    for _, row in groups.iterrows():
        code = row["Codigo"]
        name = row["Nome"]
        log.info("  Fetching institutions for group %s -- %s", code, name)
        df = fetch_institutions_for_group(code)
        df["grupo_nome"] = name
        frames.append(df)

    all_inst = pd.concat(frames, ignore_index=True)

    # Normalise CNPJ: strip whitespace, zero-pad to 8 digits
    all_inst["Cnpj"] = all_inst["Cnpj"].astype(str).str.strip().str.zfill(8)

    # Deduplicate -- keep first group assignment
    before = len(all_inst)
    all_inst = all_inst.drop_duplicates(subset="Cnpj", keep="first")
    log.info(
        "Institutions fetched: %d unique CNPJs (%d raw rows across all groups)",
        len(all_inst), before,
    )
    return all_inst


# ==============================================================================
# STEP 2 -- FETCH TARIFFS PER INSTITUTION
# ==============================================================================


def fetch_tariffs_for_institution(cnpj: str, customer_type: str) -> list[dict]:
    """
    Return tariff records for one (CNPJ, customer_type) combination.

    Parameters
    ----------
    cnpj : str          8-digit CNPJ root (zero-padded).
    customer_type : str 'F' or 'J'.
    """
    url = (
        f"{API_BASE}/ListaTarifasPorInstituicaoFinanceira"
        f"(PessoaFisicaOuJuridica='{customer_type}',CNPJ='{cnpj}')"
    )
    records = odata_get(url, params={"$format": "json"})
    for r in records:
        r["cnpj_api"] = cnpj
        r["customer_type"] = customer_type
    return records


def scrape_all_tariffs(
    institutions: pd.DataFrame,
    today: str,
    test_n: int | None = None,
) -> pd.DataFrame:
    """
    Scrape tariffs for every institution x customer_type.

    Parameters
    ----------
    institutions : DataFrame    Output of fetch_all_institutions().
    today        : str          ISO date string used as data_coleta label.
    test_n       : int, optional  If given, only scrape the first N institutions.

    Returns
    -------
    DataFrame   Long raw panel for this scrape run.
    """
    if test_n is not None:
        institutions = institutions.head(test_n)
        log.info("TEST MODE -- limiting to %d institutions", test_n)

    n_inst = len(institutions)
    all_records: list[dict] = []

    for i, row in enumerate(institutions.itertuples(index=False), start=1):
        cnpj      = row.Cnpj
        inst_name = row.Nome
        grupo     = row.grupo_nome

        if i % 50 == 0 or i == 1:
            log.info("  [%d / %d] %s (%s)", i, n_inst, inst_name, cnpj)

        for ct in CUSTOMER_TYPES:
            if records := fetch_tariffs_for_institution(cnpj, ct):
                for r in records:
                    r["NomeInstituicao"] = inst_name
                    r["grupo_nome"]      = grupo
                all_records.extend(records)
            time.sleep(INTER_CALL_SLEEP)

    if not all_records:
        log.warning("No tariff records returned -- returning empty DataFrame.")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df["data_coleta"] = today

    # Rename API columns to snake_case
    col_map = {
        "cnpj_api":       "cnpj",
        "customer_type":  "customer_type",
        "CodigoServico":  "codigo_servico",
        "Servico":        "servico",
        "Unidade":        "unidade",
        "DataVigencia":   "data_vigencia",
        "ValorMaximo":    "valor_maximo",
        "TipoValor":      "tipo_valor",
        "Periodicidade":  "periodicidade",
        "NomeInstituicao":"nome_instituicao",
        "grupo_nome":     "grupo_bcb",
        "data_coleta":    "data_coleta",
    }
    df = df.rename(columns=col_map)

    # Keep only mapped columns (drop any extra OData fields)
    keep = list(col_map.values())
    df = df[[c for c in keep if c in df.columns]]

    # Type coercions
    df["valor_maximo"]  = pd.to_numeric(df["valor_maximo"],  errors="coerce")
    df["data_vigencia"] = pd.to_datetime(df["data_vigencia"], errors="coerce")

    log.info(
        "Scrape complete: %d rows | %d institutions | %d unique services",
        len(df),
        df["cnpj"].nunique(),
        df["codigo_servico"].nunique(),
    )
    return df


# ==============================================================================
# STEP 3 -- CONGLOMERATE MAPPING
# ==============================================================================


def load_conglomerate_map() -> pd.DataFrame:
    """
    Build CNPJ -> CodConglomeradoPrudencial mapping from the most recent
    IF-Data List file.

    The List files contain rows of the form:
        CodInst, Data, NomeInstituicao, ..., CodConglomeradoPrudencial, CnpjInstituicaoLider
    For rows where CodInst is a pure numeric CNPJ root (no 'C' prefix), the
    join CodInst ↔ tarifas.Cnpj is direct.  For 'C'-prefixed BCB internal
    codes, we fall back to matching on CnpjInstituicaoLider.

    Returns
    -------
    DataFrame with columns: cnpj (8-digit str), cod_cong_prudencial (str),
                             nome_cong (str)
    """
    list_files = sorted(glob.glob(str(IF_LIST_DIR / "IF_DATA_List_*.csv")))
    # Filter out headers-only stubs (< 500 bytes = no data rows)
    list_files = [f for f in list_files if os.path.getsize(f) > 500]
    if not list_files:
        log.warning("No IF-Data List files found under %s -- conglomerate panel skipped.", IF_LIST_DIR)
        return pd.DataFrame(columns=["cnpj", "cod_cong_prudencial", "nome_cong"])

    latest = list_files[-1]
    log.info("Loading conglomerate map from: %s", Path(latest).name)

    df = pd.read_csv(latest, sep=",", encoding="latin-1", low_memory=False)

    # Normalise column names
    df.columns = [c.strip() for c in df.columns]

    # Build two join tables and concatenate:
    # (A) Rows where CodInst IS a pure numeric CNPJ root
    df["CodInst_str"] = df["CodInst"].astype(str).str.strip()
    numeric_mask = df["CodInst_str"].str.match(r"^\d+$")

    part_a = df.loc[numeric_mask, ["CodInst_str", "CodConglomeradoPrudencial"]].copy()
    part_a = part_a.rename(columns={"CodInst_str": "cnpj", "CodConglomeradoPrudencial": "cod_cong_prudencial"})
    part_a["cnpj"] = part_a["cnpj"].str.zfill(8)

    # (B) From CnpjInstituicaoLider: takes all rows and maps leader CNPJ -> conglomerate
    #     This catches institutions whose lead CNPJ is the join key in the tarifas API
    leader_valid = df["CnpjInstituicaoLider"].notna()
    part_b = df.loc[leader_valid, ["CnpjInstituicaoLider", "CodConglomeradoPrudencial"]].copy()
    part_b = part_b.rename(columns={
        "CnpjInstituicaoLider": "cnpj",
        "CodConglomeradoPrudencial": "cod_cong_prudencial",
    })
    part_b["cnpj"] = (
        part_b["cnpj"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
        .str.zfill(8)
    )

    cong_map = (
        pd.concat([part_a, part_b], ignore_index=True)
        .dropna(subset=["cod_cong_prudencial"])
        .drop_duplicates(subset="cnpj")
        .reset_index(drop=True)
    )
    cong_map["cod_cong_prudencial"] = cong_map["cod_cong_prudencial"].astype(str).str.strip()

    # Attach conglomerate name (NomeInstituicao where CodInst == CodConglomeradoPrudencial)
    cong_names = (
        df.loc[df["CodInst_str"] == df["CodConglomeradoPrudencial"].astype(str).str.strip(),
               ["CodConglomeradoPrudencial", "NomeInstituicao"]]
        .rename(columns={"CodConglomeradoPrudencial": "cod_cong_prudencial", "NomeInstituicao": "nome_cong"})
        .drop_duplicates(subset="cod_cong_prudencial")
    )
    cong_map = cong_map.merge(cong_names, on="cod_cong_prudencial", how="left")

    log.info(
        "Conglomerate map: %d CNPJs mapped to %d conglomerates",
        cong_map["cnpj"].nunique(),
        cong_map["cod_cong_prudencial"].nunique(),
    )
    return cong_map


# ==============================================================================
# STEP 4 -- BUILD PANELS
# ==============================================================================


def append_to_long_panel(new_df: pd.DataFrame, panel_path: Path) -> pd.DataFrame:
    """
    Append new_df to an existing long-panel CSV (or create it).
    Deduplicates on (cnpj, customer_type, codigo_servico, data_coleta).
    """
    if panel_path.exists():
        existing = pd.read_csv(panel_path, low_memory=False, dtype={"cnpj": str})
        existing["cnpj"] = existing["cnpj"].str.strip().str.zfill(8)
        existing["data_vigencia"] = pd.to_datetime(existing["data_vigencia"], errors="coerce")
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df.copy()

    dedup_cols = ["cnpj", "customer_type", "codigo_servico", "data_coleta"]
    before = len(combined)
    combined = combined.drop_duplicates(subset=dedup_cols, keep="last")
    log.info("Long panel: %d rows after dedup (was %d)", len(combined), before)
    import pyarrow as pa
    import pyarrow.csv as pa_csv
    pa_csv.write_csv(pa.Table.from_pandas(combined, preserve_index=False), str(panel_path))
    return combined


def build_wide_panel(long_df: pd.DataFrame, value_col: str = "valor_maximo") -> pd.DataFrame:
    """
    Pivot long_df to wide format.
    Rows:    (cnpj, nome_instituicao, grupo_bcb, customer_type, data_coleta)
    Columns: one column per CodigoServico (prefixed with the service code)

    For each (row, service), keeps the max value across any duplicates.
    """
    # Pivot
    wide = long_df.pivot_table(
        index=["cnpj", "nome_instituicao", "grupo_bcb", "customer_type", "data_coleta"],
        columns="codigo_servico",
        values=value_col,
        aggfunc="max",
    ).reset_index()

    # Flatten column names
    wide.columns = [
        f"svc_{c}" if c not in ["cnpj", "nome_instituicao", "grupo_bcb", "customer_type", "data_coleta"] else c
        for c in wide.columns
    ]
    return wide


def build_conglomerate_panels(
    long_df: pd.DataFrame,
    cong_map: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Join institution-level tariffs with conglomerate mapping and aggregate.

    Aggregation: for each (conglomerate, customer_type, service, scrape_date),
    take the MEDIAN and MAX across member institutions.

    Returns (long_cong, wide_cong).
    """
    if cong_map.empty:
        log.warning("Empty conglomerate map -- skipping conglomerate panels.")
        return pd.DataFrame(), pd.DataFrame()

    df = long_df.merge(
        cong_map[["cnpj", "cod_cong_prudencial", "nome_cong"]],
        on="cnpj",
        how="left",
    )

    if unmapped := df["cod_cong_prudencial"].isna().sum():
        log.info(
            "  %d tariff rows (%d unique CNPJs) not matched to a conglomerate -- kept with cod_cong=NaN",
            unmapped,
            df.loc[df["cod_cong_prudencial"].isna(), "cnpj"].nunique(),
        )

    # Aggregate at conglomerate level
    long_cong = (
        df.groupby(
            ["cod_cong_prudencial", "nome_cong", "customer_type",
             "codigo_servico", "servico", "unidade", "tipo_valor", "periodicidade",
             "data_coleta"],
            as_index=False,
        )
        .agg(
            n_institutions=("cnpj", "nunique"),
            valor_maximo_max=("valor_maximo", "max"),
            valor_maximo_median=("valor_maximo", "median"),
            data_vigencia_latest=("data_vigencia", "max"),
        )
    )

    # Wide version using max tariff
    wide_cong = long_cong.pivot_table(
        index=["cod_cong_prudencial", "nome_cong", "customer_type", "data_coleta"],
        columns="codigo_servico",
        values="valor_maximo_max",
        aggfunc="max",
    ).reset_index()
    wide_cong.columns = [
        f"svc_{c}" if c not in
        ["cod_cong_prudencial", "nome_cong", "customer_type", "data_coleta"]
        else c
        for c in wide_cong.columns
    ]

    log.info(
        "Conglomerate long panel: %d rows | %d conglomerates",
        len(long_cong),
        long_cong["cod_cong_prudencial"].nunique(),
    )
    return long_cong, wide_cong


# ==============================================================================
# STEP 5 -- KEY-FEE SUMMARY PANEL
# ==============================================================================


def build_fee_summary(long_cong: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the KEY_FEE_SERVICES from the conglomerate long panel and pivot to
    a wide (cod_cong_prudencial x data_coleta) summary suitable for merging
    into the main market panel.

    Output columns
    --------------
    cod_cong_prudencial, data_coleta, fee_pf_*, fee_pj_*
    """
    if long_cong.empty:
        return pd.DataFrame()

    # Build fast lookup: (codigo_servico_str, customer_type) -> column_name
    svc_lookup: dict[tuple[str, str], str] = {
        (code, ct): name for code, ct, name in KEY_FEE_SERVICES
    }

    # Normalise codigo_servico to 4-digit zero-padded string for matching
    df = long_cong.copy()
    df["_svc_key"] = df["codigo_servico"].astype(str).str.strip().str.zfill(4)
    df["_svc_tuple"] = list(zip(df["_svc_key"], df["customer_type"]))

    mask = df["_svc_tuple"].isin(svc_lookup)
    sub = df[mask].copy()

    if sub.empty:
        log.warning("build_fee_summary: no KEY_FEE_SERVICES found in conglomerate panel.")
        return pd.DataFrame()

    sub["fee_col"] = sub["_svc_tuple"].map(svc_lookup).apply(lambda n: f"fee_{n}")

    wide = sub.pivot_table(
        index=["cod_cong_prudencial", "data_coleta"],
        columns="fee_col",
        values="valor_maximo_max",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None

    log.info(
        "Fee summary: %d rows | %d conglomerates | %d fee columns",
        len(wide),
        wide["cod_cong_prudencial"].nunique(),
        len(wide.columns) - 2,
    )
    return wide


# ==============================================================================
# MAIN
# ==============================================================================


def _save_panel_if_not_empty(df: pd.DataFrame, filename: str, label: str) -> None:
    """Save a DataFrame to CSV if it is not empty, logging the operation."""
    if not df.empty:
        import pyarrow as pa
        import pyarrow.csv as pa_csv
        path = PROC_DIR / filename
        pa_csv.write_csv(pa.Table.from_pandas(df, preserve_index=False), str(path))
        log.info(f"{label}: {len(df)} rows x {len(df.columns)} cols -> {path}")

def main(test_n: int | None = None) -> None:
    today = date.today().isoformat()  # e.g. "2026-02-26"
    today_tag = today.replace("-", "")  # e.g. "20260226"

    # Create output dirs
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    # -- Guard: skip entirely if fee summary already exists (re-run protection) --
    fee_path = PROC_DIR / "tarifas_fee_summary.csv"
    if fee_path.exists() and fee_path.stat().st_size > 0 and test_n is None:
        log.info(
            "Fee summary already exists (%s). "
            "Delete it to force a full re-scrape.",
            fee_path,
        )
        print(f"Output already exists, skipping: {fee_path}")
        return

    # -- Guard: skip if today's raw file already exists (re-run protection) -----
    raw_pf = RAW_DIR / f"tarifas_raw_{today_tag}_pf.csv"

    if raw_pf.exists() and test_n is None:
        log.info(
            "Today's raw file already exists (%s). "
            "Delete it to force a re-scrape.",
            today_tag,
        )
        # Still rebuild panels from the existing raw file
        new_rows = pd.read_csv(raw_pf, low_memory=False, dtype={"cnpj": str})
        new_rows["cnpj"] = new_rows["cnpj"].astype(str).str.strip().str.zfill(8)
        new_rows["data_vigencia"] = pd.to_datetime(new_rows["data_vigencia"], errors="coerce")
    else:
        # -- Step 1: fetch institution list ------------------------------------
        log.info("=== Step 1: fetching institution list ===")
        groups = fetch_groups()
        institutions = fetch_all_institutions(groups)

        # -- Step 2: scrape tariffs ---------------------------------------------
        log.info("=== Step 2: scraping tariffs (%d institutions, %s) ===",
                 len(institutions) if test_n is None else min(test_n, len(institutions)),
                 today)
        new_rows = scrape_all_tariffs(institutions, today, test_n=test_n)

        if new_rows.empty:
            log.error("No data scraped -- aborting.")
            return

        # -- Save raw snapshot -------------------------------------------------
        if test_n is None:
            import pyarrow as pa
            import pyarrow.csv as pa_csv
            raw_pj = RAW_DIR / f"tarifas_raw_{today_tag}_pj.csv"
            for ct, path in [("F", raw_pf), ("J", raw_pj)]:
                subset = new_rows[new_rows["customer_type"] == ct]
                if not subset.empty:
                    pa_csv.write_csv(pa.Table.from_pandas(subset, preserve_index=False), str(path))
                    log.info("Saved raw %s: %d rows -> %s", ct, len(subset), path.name)

    # -- Step 3: load conglomerate map ------------------------------------------
    log.info("=== Step 3: loading conglomerate map ===")
    cong_map = load_conglomerate_map()

    # -- Step 4a: institution long panel ---------------------------------------
    log.info("=== Step 4a: institution long panel ===")
    inst_long_path = PROC_DIR / "tarifas_panel_institution_long.csv"
    inst_long = append_to_long_panel(new_rows, inst_long_path)
    log.info("Institution long panel: %d rows -> %s", len(inst_long), inst_long_path)

    # -- Step 4b: institution wide panel (latest scrape per institution) --------
    log.info("=== Step 4b: institution wide panel ===")
    # For the wide panel, use only the most recent scrape per (CNPJ, customer_type)
    latest_date_per_inst = (
        inst_long.groupby(["cnpj", "customer_type"])["data_coleta"]
        .max()
        .reset_index()
        .rename(columns={"data_coleta": "latest_coleta"})
    )
    inst_long_latest = inst_long.merge(
        latest_date_per_inst,
        on=["cnpj", "customer_type"],
    )
    inst_long_latest = inst_long_latest[inst_long_latest["data_coleta"] == inst_long_latest["latest_coleta"]]

    wide_inst = build_wide_panel(inst_long_latest)
    inst_wide_path = PROC_DIR / "tarifas_panel_institution_wide.csv"
    import pyarrow as pa
    import pyarrow.csv as pa_csv
    pa_csv.write_csv(pa.Table.from_pandas(wide_inst, preserve_index=False), str(inst_wide_path))
    log.info(
        "Institution wide panel: %d rows x %d cols -> %s",
        len(wide_inst), len(wide_inst.columns), inst_wide_path,
    )

    # -- Step 4c: conglomerate panels ------------------------------------------
    log.info("=== Step 4c: conglomerate panels ===")
    cong_long, cong_wide = build_conglomerate_panels(inst_long, cong_map)

    _save_panel_if_not_empty(cong_long, "tarifas_panel_conglomerate_long.csv", "Conglomerate long panel")
    _save_panel_if_not_empty(cong_wide, "tarifas_panel_conglomerate_wide.csv", "Conglomerate wide panel")

    # -- Step 5: key-fee summary panel -----------------------------------------
    log.info("=== Step 5: building fee summary panel ===")
    fee_summary = build_fee_summary(cong_long)
    _save_panel_if_not_empty(fee_summary, "tarifas_fee_summary.csv", "Fee summary")

    # -- Summary ----------------------------------------------------------------
    n_scrapes = inst_long["data_coleta"].nunique()
    first_scrape = inst_long["data_coleta"].min()
    last_scrape  = inst_long["data_coleta"].max()
    log.info(
        "=== Done ===  %d total panel rows | %d scrape dates (%s -> %s) | "
        "%d unique institutions",
        len(inst_long),
        n_scrapes,
        first_scrape,
        last_scrape,
        inst_long["cnpj"].nunique(),
    )


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BCB Tarifas Bancárias scraper")
    parser.add_argument(
        "--test",
        type=int,
        default=None,
        metavar="N",
        help="TEST MODE: scrape only the first N institutions (skips raw file guard).",
    )
    args = parser.parse_args()

    main(test_n=args.test)

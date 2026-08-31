"""
scrape_bcb_complaints.py
========================
BCB "Ranking de Instituicoes por Indice de Reclamacoes" — the complaints a citizen
files against a financial institution, which the BCB receives, analyses and closes,
published per institution and per BCB-defined conglomerate since July 2014.

Why this source
---------------
It is the only official, institution-level SERVICE-QUALITY series for Brazilian banks,
and it carries the client headcounts the deposit-competition draft wants for the D
firms in the same file: "Quantidade total de clientes - CCS e SCR" (plus CCS-only,
SCR-only, and FGC before 2023).  Both are at conglomerate x quarter, which is the
D-firm panel unit.

Source
------
  index  https://www3.bcb.gov.br/rdrweb/rest/ext/ranking
         nested JSON: anos -> periodicidades -> periodos -> tipos
  file   https://www3.bcb.gov.br/rdrweb/rest/ext/ranking/arquivo
         ?ano=YYYY&periodicidade=<KIND>&periodo=N&tipo=Bancos+e+financeiras
         semicolon-separated, latin-1, one trailing empty field per line

Two breaks the published files do not flag
------------------------------------------
PERIODICITY. It changes across eras, which is why the quarterly file is not simply the
raw download (see to_quarters):
  2014-07..2016-06  MENSAL      -> summed to quarters here
  2016-07..2016-12  BIMESTRAL   -> bimesters straddle quarter boundaries; long file only
  2014..2016        SEMESTRAL   -> a second view of periods already covered; long file only
  2017-01..         TRIMESTRAL  -> used directly
So the quarterly panel runs 2014-Q3..2016-Q2 and 2017-Q1 onward; 2016-Q3/Q4 exist only
at bimester and semester granularity and are deliberately absent rather than invented.

TAXONOMY. From 2024-Q3 the BCB stopped splitting complaints into regulated and
non-regulated and began publishing an upheld count without the "regulada" qualifier,
plus analysed and answered totals. Both regimes' columns are kept under their own
names; `complaints_upheld` carries the comparable series and `upheld_definition` marks
which definition produced each row, so pooling across the break is a visible choice
rather than an accident.

Two views, published side by side
---------------------------------
Each period is published twice: one row per "Conglomerado" and one per
"Banco/financeira".  Every large bank appears ONLY in the conglomerate view, which
carries no CNPJ -- Caixa, Bradesco, Itau, Nubank and the rest are identified by a brand
label alone.  So the firm-level file is the conglomerate view, keyed by label and
carrying `cod_cong_prudencial` wherever the label can be identified through
utils.firm_registry; the institution view is kept separately, keyed on CNPJ.  They are
not summed together: the institution rows are the same complaints disaggregated.

Outputs  (BCB/Quality/processed/)
---------------------------------
  complaints_institution_period.csv     every published row, native period, both views
  complaints_institution_quarterly.csv  Banco/financeira rows, CNPJ-keyed, quarterly
  complaints_firm_quarterly.csv         Conglomerado rows, label-keyed, quarterly

Usage
-----
  python scrape_bcb_complaints.py
  python scrape_bcb_complaints.py --refresh          # re-download cached periods
  python scrape_bcb_complaints.py --include-consorcios
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import time
import unicodedata
import urllib.request
from pathlib import Path

from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import numpy as np
import pandas as pd

from utils import paths

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

INDEX_URL = "https://www3.bcb.gov.br/rdrweb/rest/ext/ranking"
FILE_URL = "https://www3.bcb.gov.br/rdrweb/rest/ext/ranking/arquivo"
USER_AGENT = "Mozilla/5.0 (research; deposit-competition panel build)"
SLEEP_SECS = 1.0

RAW_DIR = paths.QUALITY_RAW / "complaints"
OUT_DIR = paths.QUALITY_PROC

BANK_TIPO = "Bancos e financeiras"

# Published header (accent-stripped, lowercased, collapsed spaces) -> our column.
# Every column must appear here: an unmapped one means the layout changed and the run
# stops rather than silently dropping a measure.
COLUMN_ALIASES = {
    "ano": "year",
    "mes": "_period_num", "bimestre": "_period_num",
    "trimestre": "_period_num", "semestre": "_period_num",
    "categoria": "categoria",
    "tipo": "row_kind",
    "cnpj if": "cnpj",
    "instituicao financeira": "instituicao",
    "indice": "complaint_index",
    "quantidade total de clientes - ccs e scr": "n_clients_ccs_scr",
    "quantidade de clientes - ccs": "n_clients_ccs",
    "quantidade de clientes - scr": "n_clients_scr",
    "quantidade de clientes - fgc": "n_clients_fgc",
    # -- complaint taxonomy, FIRST regime: July 2014 .. 2024-Q2 -----------------
    "quantidade de reclamacoes reguladas procedentes": "complaints_reg_procedentes",
    "quantidade de reclamacoes reguladas - outras": "complaints_reg_outras",
    "quantidade de reclamacoes nao reguladas": "complaints_nao_reguladas",
    "quantidade total de reclamacoes": "complaints_total",
    # -- complaint taxonomy, SECOND regime: 2024-Q3 onward ----------------------
    # The BCB dropped the regulated/non-regulated split and now reports an upheld
    # count without the "regulada" qualifier, an analysed total and an answered
    # total.  These are NOT the earlier columns under new names, so they keep their
    # own; `complaints_upheld` below carries the comparable series and
    # `upheld_definition` says which regime each row came from.
    "quantidade de reclamacoes procedentes": "complaints_procedentes",
    "quantidade total de reclamacoes analisadas": "complaints_total_analisadas",
    "quantidade total de reclamacoes respondidas": "complaints_total_respondidas",
    # Extrapolated upheld count: 2024-Q2 under the first regime's naming, and every
    # quarter from 2024-Q3 under the second's.
    "quantidade de reclamacoes reguladas procedentes extrapoladas":
        "complaints_procedentes_extrapolated",
    "quantidade de reclamacoes procedentes extrapoladas":
        "complaints_procedentes_extrapolated",
}

COUNT_COLS = ["complaints_reg_procedentes", "complaints_reg_outras",
              "complaints_nao_reguladas", "complaints_total",
              "complaints_procedentes", "complaints_procedentes_extrapolated",
              "complaints_total_analisadas", "complaints_total_respondidas",
              "n_clients_ccs_scr", "n_clients_ccs", "n_clients_scr", "n_clients_fgc"]

# Preference order for the comparable upheld-complaint series, best first.
UPHELD_SOURCES = [
    ("complaints_reg_procedentes", "reguladas_procedentes"),
    ("complaints_procedentes", "procedentes"),
    ("complaints_procedentes_extrapolated", "procedentes_extrapoladas"),
]

# The ranking publishes each period twice: one row per "Conglomerado" (the BCB's own
# grouping, which is where every large bank lives -- Caixa, Bradesco, Itau, Nubank --
# and which carries NO CNPJ) and one per "Banco/financeira" (individual institutions,
# with a CNPJ).  The firm-level file is built from the Conglomerado rows, because the
# institution rows are the same complaints disaggregated and summing both would double
# count.
KIND_CONGLOMERATE = "Conglomerado"
KIND_INSTITUTION = "Banco/financeira"

# BCB conglomerate labels that utils.firm_registry's own name fragments do not reach,
# either because the label is an abbreviation ("BB") or because the firm renamed
# mid-sample.  Only the label -> firm_key link lives here; every CNPJ still comes from
# firm_registry, so this table cannot attach a firm to the wrong company's identifier.
LABEL_ALIASES = {
    "bb": "bb",
    "caixa economica federal": "caixa",
    "inter": "inter",
    "xp": "xp",
    "pan": "pan",
    "nu pagamentos": "nubank",
    "banco c6": "c6",
    "pagbank-pagseguro": "pagseguro",
    "mercado pago ip": "mercadopago",
    "stone ip": "stone",
    "banco bmg": "bmg",
}

# The BCB's own definition, confirmed against published rows (Acesso 2023-Q1:
# 154 procedentes / 4,733,146 clients x 1e6 = 32.54 against a published 32,53).
INDEX_PER_CLIENTS = 1_000_000


def _norm(s: str) -> str:
    """Accent-strip, lowercase and collapse a published header for alias lookup."""
    s = str(s)
    # "clientes - CCS" carries a cp1252 en dash, which arrives as \x96 through a
    # latin-1 read.  Fold every dash-like character to a plain hyphen BEFORE the ASCII
    # step, which would otherwise drop it and leave the alias key unmatchable.
    for ch in ("\x96", "\x97", "–", "—"):
        s = s.replace(ch, "-")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return " ".join(s.lower().split())


def _get(url: str, timeout: int = 90) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


# ---------------------------------------------------------------------------
# 1. Index: which (year, periodicity, period, type) rankings exist
# ---------------------------------------------------------------------------

def fetch_index(include_consorcios: bool = False) -> list[dict]:
    raw = _get(INDEX_URL)
    tree = json.loads(raw.decode("utf-8", errors="replace"))
    out: list[dict] = []
    for yr in tree.get("anos", []):
        for per_kind in yr.get("periodicidades", []):
            for per in per_kind.get("periodos", []):
                for t in per.get("tipos", []):
                    tipo = t.get("tipo", "")
                    if not include_consorcios and "ancos" not in tipo:
                        continue
                    out.append({"ano": str(yr["ano"]),
                                "periodicidade": per_kind["periodicidade"],
                                "periodo": int(per["periodo"]),
                                "tipo": tipo})
    log.info("Index: %d rankings available (%s)", len(out),
             "banks + consorcios" if include_consorcios else "banks only")
    return out


# ---------------------------------------------------------------------------
# 2. Download (cached) and parse
# ---------------------------------------------------------------------------

def download_period(entry: dict, refresh: bool = False) -> Path | None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    slug = (f"{entry['ano']}_{entry['periodicidade']}_{entry['periodo']:02d}_"
            f"{entry['tipo'].split()[0].lower()}")
    dest = RAW_DIR / f"{slug}.csv"
    if dest.exists() and dest.stat().st_size > 200 and not refresh:
        return dest

    url = (f"{FILE_URL}?ano={entry['ano']}&periodicidade={entry['periodicidade']}"
           f"&periodo={entry['periodo']}&tipo={entry['tipo'].replace(' ', '+')}")
    try:
        blob = _get(url)
    except Exception as exc:  # noqa: BLE001 - one bad period must not stop the sweep
        log.warning("download failed %s: %s", slug, exc)
        return None
    if len(blob) < 200:
        log.warning("suspiciously small file for %s (%d bytes) — skipping", slug, len(blob))
        return None
    dest.write_bytes(blob)
    time.sleep(SLEEP_SECS)
    return dest


def parse_period(path: Path, entry: dict) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(path.read_bytes()), sep=";", encoding="latin-1",
                     dtype=str)
    # Each line ends with a separator, so pandas invents one empty trailing column.
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed:")])

    renames, unknown = {}, []
    for col in df.columns:
        key = _norm(col)
        if key in COLUMN_ALIASES:
            renames[col] = COLUMN_ALIASES[key]
        else:
            unknown.append(col)
    if unknown:
        raise ValueError(
            f"{path.name}: unmapped column(s) {unknown!r}. The BCB layout changed; add "
            f"them to COLUMN_ALIASES rather than letting a measure disappear.")
    df = df.rename(columns=renames)

    df["period_kind"] = entry["periodicidade"]
    df["period_num"] = pd.to_numeric(
        df["_period_num"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    df = df.drop(columns=["_period_num"])
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    for c in COUNT_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(r"[^\d-]", "", regex=True),
                                  errors="coerce")
        else:
            df[c] = np.nan
    df["complaint_index"] = pd.to_numeric(
        df["complaint_index"].astype(str).str.strip().str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False), errors="coerce")

    df["cnpj"] = (df["cnpj"].astype(str).str.replace(r"\D", "", regex=True)
                  .replace("", np.nan))
    df["cnpj"] = df["cnpj"].str[:8].str.zfill(8).where(df["cnpj"].notna())
    for c in ("instituicao", "categoria", "row_kind"):
        df[c] = df[c].astype(str).str.strip()
    df["tipo_arquivo"] = entry["tipo"]
    return df


# ---------------------------------------------------------------------------
# 3. Native periods -> quarters
# ---------------------------------------------------------------------------

def to_quarters(long: pd.DataFrame) -> pd.DataFrame:
    """
    Published rows on a (year, quarter) grid, one row per entity per quarter.

    TRIMESTRAL periods are quarters already.  MENSAL months are summed into the quarter
    that contains them -- complaint counts add up, client counts are a stock so the
    quarter takes the last month reported, and the index is recomputed from the two
    rather than averaged.  BIMESTRAL and SEMESTRAL periods are dropped: a bimester
    straddles a quarter boundary and a semester spans two, so neither can be assigned
    to one quarter without inventing a split.

    Both row kinds are carried through.  Dropping the Conglomerado rows would discard
    every large bank in the country: they have no CNPJ and appear only there.
    """
    rows = long.copy()
    # `entity_key` is what a firm is identified by within its row kind: the CNPJ for an
    # institution, the label for a conglomerate (whose CNPJ field is blank).
    rows["label"] = (rows["instituicao"].astype(str)
                     .str.replace(r"\s*\(conglomerado\)\s*$", "", regex=True).str.strip())
    rows["entity_key"] = np.where(rows["row_kind"] == KIND_INSTITUTION,
                                  rows["cnpj"].astype(str), rows["label"])
    rows = rows[rows["entity_key"].notna() & (rows["entity_key"] != "nan")]

    keys = ["row_kind", "entity_key", "year", "quarter"]

    tri = rows[rows["period_kind"] == "TRIMESTRAL"].copy()
    tri["quarter"] = tri["period_num"].astype("Int64")
    tri["source_granularity"] = "TRIMESTRAL"

    men = rows[rows["period_kind"] == "MENSAL"].copy()
    if not men.empty:
        men["quarter"] = ((men["period_num"] - 1) // 3 + 1).astype("Int64")
        men = men.sort_values(keys + ["period_num"])
        flow_cols = [c for c in COUNT_COLS if c.startswith("complaints_")]
        stock_cols = [c for c in COUNT_COLS if c.startswith("n_clients_")]
        counts = men.groupby(keys, as_index=False)[flow_cols].sum(min_count=1)
        stocks = men.groupby(keys, as_index=False)[stock_cols].last()
        labels = (men.groupby(keys, as_index=False)
                     [["instituicao", "label", "cnpj", "categoria", "tipo_arquivo"]].last())
        men = counts.merge(stocks, on=keys).merge(labels, on=keys)
        men["source_granularity"] = "MENSAL_SUMMED"
        men["period_kind"] = "MENSAL"
        men["period_num"] = pd.NA

    q = pd.concat([tri, men], ignore_index=True)
    # A quarter present at both granularities keeps the published quarterly row.
    q = (q.sort_values(keys + ["source_granularity"])
           .drop_duplicates(keys, keep="first"))

    q = add_upheld_series(q)
    q["complaint_index"] = (q["complaints_upheld"] * INDEX_PER_CLIENTS
                            / q["n_clients_ccs_scr"].where(q["n_clients_ccs_scr"] > 0))
    return q


def add_upheld_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add `complaints_upheld` plus the `upheld_definition` that says where it came from.

    The two taxonomy regimes never overlap in a quarter, so the coalesce below picks
    exactly one source per row.  Keeping the definition alongside the number is the
    point: a regression that pools 2024-Q2 with 2024-Q3 is pooling "reguladas
    procedentes" with "procedentes", and only this column makes that visible.
    """
    df = df.copy()
    df["complaints_upheld"] = np.nan
    df["upheld_definition"] = pd.NA
    for col, label in UPHELD_SOURCES:
        if col not in df.columns:
            continue
        take = df["complaints_upheld"].isna() & df[col].notna()
        df.loc[take, "complaints_upheld"] = df.loc[take, col]
        df.loc[take, "upheld_definition"] = label
    counts = df["upheld_definition"].value_counts(dropna=False).to_dict()
    log.info("upheld-complaint definitions in use: %s", counts)
    return df


def build_label_crosswalk() -> dict:
    """
    {normalised BCB conglomerate label -> CodConglomeradoPrudencial}.

    The BCB labels its conglomerates with a brand ("BRADESCO", "NU PAGAMENTOS"), never
    a CNPJ, so the link to the panel's key has to be made by name.  It is made ONLY
    through utils.firm_registry: a label matches a firm when it equals one of that
    firm's registered name fragments, or when LABEL_ALIASES says so, and the CNPJ then
    comes from the registry entry and the code from the shared canonical map.  No CNPJ
    is written down here, so a wrong label can leave a firm unresolved but cannot
    attach it to another company's identifier.

    Labels the registry does not cover stay unresolved on purpose -- roughly 300 small
    institutions whose identity would otherwise rest on a guess.
    """
    from panel_deposits import build_cnpj_conglomerate_map
    from utils.firm_registry import FIRMS

    cmap = build_cnpj_conglomerate_map()
    by_key = {f["firm_key"]: f for f in FIRMS}

    label_to_firm: dict[str, str] = {}
    for f in FIRMS:
        for frag in f.get("names", []):
            label_to_firm[_norm(frag)] = f["firm_key"]
        label_to_firm[_norm(f["firm_key"])] = f["firm_key"]
    label_to_firm.update(LABEL_ALIASES)

    out: dict[str, str] = {}
    for label, firm_key in label_to_firm.items():
        firm = by_key.get(firm_key)
        if firm is None:
            continue
        hit = cmap.get(int(firm["cnpj_root"]))
        code = (str(hit[0]) if hit and hit[0] is not None
                else firm.get("cong_prud") or None)
        if code:
            out[label] = code
    return out


def to_firms(q: pd.DataFrame) -> pd.DataFrame:
    """
    Firm-level quarters, built from the published Conglomerado rows.

    Those rows already aggregate a group's member institutions, so the Banco/financeira
    rows are the same complaints disaggregated: adding both would double count.  The
    firm-level file therefore uses the conglomerate view alone, and the institution view
    stays available in complaints_institution_quarterly.csv.

    `cod_cong_prudencial` is filled where build_label_crosswalk can identify the firm
    and left empty otherwise, with `label` always present so an unresolved firm is still
    usable and still visible.
    """
    cong = q[q["row_kind"] == KIND_CONGLOMERATE].copy()
    xwalk = build_label_crosswalk()
    cong["label_key"] = cong["label"].map(_norm)
    cong["cod_cong_prudencial"] = cong["label_key"].map(xwalk)

    resolved = cong["cod_cong_prudencial"].notna()
    cli = cong["n_clients_ccs_scr"].fillna(0)
    log.info("Label crosswalk: %d/%d firm-quarters resolved to a conglomerate code "
             "(%.1f%% of client-quarters); %d distinct labels unresolved",
             int(resolved.sum()), len(cong),
             100 * cli[resolved].sum() / cli.sum() if cli.sum() else 0.0,
             cong.loc[~resolved, "label"].nunique())
    return cong


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(refresh: bool = False, include_consorcios: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    entries = fetch_index(include_consorcios)

    frames, missing = [], 0
    for e in entries:
        path = download_period(e, refresh)
        if path is None:
            missing += 1
            continue
        frames.append(parse_period(path, e))
    if not frames:
        log.error("No ranking files parsed — aborting.")
        return
    long = pd.concat(frames, ignore_index=True)
    log.info("Parsed %d periods (%d unavailable) -> %d rows, %d institutions",
             len(frames), missing, len(long), long["cnpj"].nunique())

    long_path = OUT_DIR / "complaints_institution_period.csv"
    long.to_csv(long_path, index=False)
    log.info("Saved %s (%d rows)", long_path.name, len(long))

    q = to_quarters(long)
    span = (f"{int(q['year'].min())}-Q{int(q.loc[q['year'].idxmin(), 'quarter'])} to "
            f"{int(q['year'].max())}-Q{int(q.loc[q['year'].idxmax(), 'quarter'])}")

    inst = q[q["row_kind"] == KIND_INSTITUTION]
    i_path = OUT_DIR / "complaints_institution_quarterly.csv"
    inst.to_csv(i_path, index=False)
    log.info("Saved %s (%d rows | %d institutions | %s)",
             i_path.name, len(inst), inst["cnpj"].nunique(), span)

    firms = to_firms(q)
    f_path = OUT_DIR / "complaints_firm_quarterly.csv"
    firms.to_csv(f_path, index=False)
    log.info("Saved %s (%d rows | %d labels | %d with a conglomerate code | %s)",
             f_path.name, len(firms), firms["label"].nunique(),
             firms["cod_cong_prudencial"].nunique(), span)
    log.info("=== Done ===")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true",
                    help="re-download periods already cached under Quality/raw/complaints")
    ap.add_argument("--include-consorcios", action="store_true",
                    help="also pull the consortium-administrator ranking")
    a = ap.parse_args()
    main(a.refresh, a.include_consorcios)

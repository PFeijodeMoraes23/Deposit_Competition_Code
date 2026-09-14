"""
scrape_cvm_advertising.py
=========================
Advertising-type expense lines from the CVM structured filings (DFP annual, ITR quarterly),
2013 onward, for every listed company, with quarterly flows.

Where the line lives
--------------------
Advertising is not a fixed account. Banks add it themselves, mostly as a sub-line of the
DVA (Demonstracao do Valor Adicionado) under 7.03.04 "Outros" in "Insumos adquiridos de
terceiros", and a few as a sub-line of the DRE under 3.04. Sub-line codes drift across
years and filings (ST_CONTA_FIXA = N), so lines are identified by their DESCRIPTION, never by
CD_CONTA. Each line's description is reduced to the components it names (propaganda,
publicidade, promocoes, publicacoes, relacoes_publicas, marketing), kept as line_components.
Labels drift for the same account, so a company-scope-statement that never files two
advertising lines in one period is ONE series, line_id "single"; a company that files several
side by side keeps the component-based line_id. The label says what the filer called the
line; which COSIF accounts it contains differs by bank and is established downstream by
reconciling against COSIF 2025, not here.

Nothing in this script picks banks or versions. It keeps every advertising-type line of every
company and produces both a first-reported and a restated series, so choices are made where
the panel is assembled.

Units, periods and versions
---------------------------
  - VL_CONTA is scaled by ESCALA_MOEDA (MIL = thousand, UNIDADE = one). Expense lines are
    stored negative; value_brl is the absolute amount and sign_raw keeps the filed sign.
  - Within a company and filing (CD_CVM, document, DT_REFER), only the highest VERSAO is kept.
  - ORDEM_EXERC "ULTIMO" rows are the period being reported (first-reported). "PENULTIMO"
    rows are the prior-year comparatives in the following year's filing (restated).
  - A period is year-to-date when DT_INI_EXERC is 1 January of DT_FIM_EXERC's year. ITR DRE
    files also carry quarter-only columns; those are kept as period_type "quarter" and used
    only to cross-check the year-to-date differencing.
  - Quarterly flows difference year-to-date values within the calendar year:
        Q1 = YTD(Mar), Q2 = YTD(Jun) - YTD(Mar), Q3 = YTD(Sep) - YTD(Jun), Q4 = YTD(Dec) - YTD(Sep)
    with YTD(Dec) the DFP annual figure. A flow is missing when a value it needs is missing.
  - The filings for fiscal year 2025 carry their 2024 comparatives as zeros (the 2025
    accounting-rule change). A zero comparative against a non-zero first-reported value is
    therefore treated as missing, not as a restatement to zero, and counted in the log.

Outputs (paths.AWARENESS_PROC)
------------------------------
  cvm_advertising_lines.{parquet,csv}      one row per filed value after version selection
  cvm_advertising_quarterly.{parquet,csv}  company x scope x statement x line_id x quarter:
                                           flow_first, flow_restated and flags
    ytd_falls        a year-to-date value below the previous quarter's (negative flow)
    scale_suspect    first-reported and restated values differ by a factor near 1000
    sign_flip        a filed sign that differs from the company-line's usual sign
    qo_mismatch      the ITR DRE quarter-only figure disagrees with the year-to-date difference
    summed_siblings  (lines) same-level accounts with the same words and different values,
                     summed; see resolve_collisions for the other collision cases
  bcb_supervised marks filers whose CNPJ8 appears as an institution or conglomerate leader in
  any IF.data list (paths.IF_DATA_LIST), 2013 onward.

Validation (any failure aborts before writing)
----------------------------------------------
  1. Every DFP and ITR year in range is available (cache or download) with the DVA and DRE
     members and the expected columns.
  2. Banco do Brasil (CD_CVM 001023) has its individual DVA propaganda line with a
     first-reported annual value in every year 2013-2024.
  3. Among BCB-supervised filers, where the ITR DRE quarter-only value and the year-to-date
     difference both exist, they agree within 0.5% in at least 95% of cases. Other listed
     companies are reported in the log but not gated: several have non-calendar or
     internally inconsistent filings that the bank panel never uses.
  4. If the COSIF advertising panel from scrape_cosif_advertising.py is on disk: Banco do
     Brasil's 2025 individual DVA annual equals the COSIF institution adv total for 2025
     within 0.5%.

Usage
-----
  python scrape_cvm_advertising.py
  python scrape_cvm_advertising.py --since 2013 --until 2026
  python scrape_cvm_advertising.py --refresh           # redownload the yearly zips
  python scrape_cvm_advertising.py --out-dir <dir>
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import logging
import os
import re
import unicodedata
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from utils import paths
from utils.disclosure_common import http_get

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

RAW_DIR = paths.FIRM_DISCLOSURES / "CVM" / "raw"       # shared with scrape_cvm_fee_income.py
IF_DATA_LIST = paths.IF_DATA_LIST
OUT_DIR = paths.AWARENESS_PROC
CVM_URL = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC/{doc}/DADOS/{doc_l}_cia_aberta_{year}.zip"
DOCS = ("DFP", "ITR")
MEMBERS = ("DVA_ind", "DVA_con", "DRE_ind", "DRE_con")
SCALE = {"MIL": 1e3, "UNIDADE": 1.0}
COLUMNS = ["CNPJ_CIA", "DT_REFER", "VERSAO", "DENOM_CIA", "CD_CVM", "ESCALA_MOEDA",
           "ORDEM_EXERC", "DT_INI_EXERC", "DT_FIM_EXERC", "CD_CONTA", "DS_CONTA", "VL_CONTA",
           "ST_CONTA_FIXA"]

COMPONENTS = {
    "propaganda": re.compile(r"propaganda"),
    "publicidade": re.compile(r"publicidade"),
    "promocoes": re.compile(r"\bprom(oc|o\b|\.|\s|$)"),
    "publicacoes": re.compile(r"publicac"),
    "relacoes_publicas": re.compile(r"relacoes publicas"),
    "marketing": re.compile(r"marketing"),
}
EXCLUDE = re.compile(r"receita")

BB_CVM = "001023"
QUARTER_OF_MONTH = {3: 1, 6: 2, 9: 3, 12: 4}


# ---------------------------------------------------------------------------
# Download and read
# ---------------------------------------------------------------------------
def yearly_zip(doc: str, year: int, refresh: bool) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{doc}_{year}.zip"
    if path.exists() and path.stat().st_size > 1000 and not refresh:
        return path
    url = CVM_URL.format(doc=doc, doc_l=doc.lower(), year=year)
    code, raw = http_get(url, binary=True, timeout=300)
    if not raw:
        if path.exists():
            log.warning("%s %d: download failed (HTTP %s); using the cached copy from %s",
                        doc, year, code, dt.date.fromtimestamp(path.stat().st_mtime))
            return path
        raise SystemExit(f"{doc} {year}: not cached and download failed (HTTP {code}) at {url}")
    path.write_bytes(raw)
    return path


def normalise(label: str) -> str:
    s = unicodedata.normalize("NFD", str(label).lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^a-z0-9.]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def components_of(label_norm: str) -> str:
    if EXCLUDE.search(label_norm):
        return ""
    return "+".join(k for k, rx in COMPONENTS.items() if rx.search(label_norm))


def read_members(path: Path, doc: str, year: int) -> pd.DataFrame:
    frames = []
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        for member in MEMBERS:
            name = f"{doc.lower()}_cia_aberta_{member}_{year}.csv"
            if name not in names:
                raise AssertionError(f"{path.name}: member {name} missing")       # check 1
            df = pd.read_csv(io.BytesIO(zf.read(name)), sep=";", encoding="latin1", dtype=str)
            missing = set(COLUMNS) - set(df.columns)
            if missing:
                raise AssertionError(f"{path.name}/{name}: missing columns {sorted(missing)}")
            df = df[COLUMNS]
            norm = df["DS_CONTA"].map(normalise)
            comp = norm.map(components_of)
            keep = comp != ""
            if not keep.any():
                continue
            sub = df[keep].copy()
            sub["label_norm"] = norm[keep]
            sub["line_id"] = comp[keep]
            sub["statement"], sub["scope"] = member.split("_")
            sub["doc"] = doc
            frames.append(sub)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Lines
# ---------------------------------------------------------------------------
def build_lines(since: int, until: int, refresh: bool) -> pd.DataFrame:
    frames = []
    for year in range(since, until + 1):
        for doc in DOCS:
            path = yearly_zip(doc, year, refresh)
            df = read_members(path, doc, year)
            log.info("  %s %d (%s, cached %s): %d advertising-type rows", doc, year, path.name,
                     dt.date.fromtimestamp(path.stat().st_mtime), len(df))
            frames.append(df)
    lines = pd.concat(frames, ignore_index=True)

    lines["VERSAO"] = pd.to_numeric(lines["VERSAO"])
    newest = lines.groupby(["CD_CVM", "doc", "DT_REFER"])["VERSAO"].transform("max")
    lines = lines[lines["VERSAO"] == newest].copy()

    scale = lines["ESCALA_MOEDA"].str.upper().map(SCALE)
    if scale.isna().any():
        raise AssertionError(f"unknown ESCALA_MOEDA values: "
                             f"{sorted(lines.loc[scale.isna(), 'ESCALA_MOEDA'].unique())}")
    raw_value = pd.to_numeric(lines["VL_CONTA"]) * scale
    lines["value_brl"] = raw_value.abs()
    lines["sign_raw"] = np.sign(raw_value).astype(int)

    fim = pd.to_datetime(lines["DT_FIM_EXERC"])
    ini = pd.to_datetime(lines["DT_INI_EXERC"])
    lines["period_end"] = fim.dt.strftime("%Y-%m-%d")
    lines["year"] = fim.dt.year
    lines["quarter"] = fim.dt.month.map(QUARTER_OF_MONTH)
    is_ytd = (ini.dt.month == 1) & (ini.dt.day == 1) & (ini.dt.year == fim.dt.year)
    lines["period_type"] = np.where(is_ytd, "ytd", "quarter")
    lines["vintage"] = np.where(lines["ORDEM_EXERC"].map(normalise).str.startswith("ultimo"),
                                "first", "restated")
    lines = lines[lines["quarter"].notna()].copy()
    lines["quarter"] = lines["quarter"].astype(int)
    lines["cnpj8"] = lines["CNPJ_CIA"].str.replace(r"\D", "", regex=True).str[:8]
    lines["CD_CVM"] = lines["CD_CVM"].str.zfill(6)

    return unify_single_lines(resolve_collisions(lines))


def unify_single_lines(lines: pd.DataFrame) -> pd.DataFrame:
    """Give a company-statement one series when it never files two advertising lines at once.

    Labels drift for the same account (Itau's consolidated DVA line reads "Propaganda,
    Promocoes e Publicidade" in 2013Q3 and "... e Publicacoes" in every other quarter), so a
    label-derived line_id would split one account into fragments. Where no single filing
    period of a company-scope-statement carries more than one line, every line is the same
    series: line_id becomes "single" and the label-derived id is kept as line_components.
    Companies that file several lines side by side (Banese: propaganda, promocoes and
    publicacoes separately) keep label-derived ids.
    """
    df = lines.copy()
    df["line_components"] = df["line_id"]
    series = ["CD_CVM", "scope", "statement"]
    per_filing = (df.groupby(series + ["DT_REFER", "doc", "period_end", "period_type", "vintage"])
                    ["line_id"].nunique())
    widest = per_filing.groupby(level=series).max()
    single = widest[widest == 1].index
    is_single = df.set_index(series).index.isin(single)
    df.loc[is_single, "line_id"] = "single"
    log.info("company-scope-statements with one advertising line per filing: %d of %d",
             len(single), len(widest))
    return df


def resolve_collisions(lines: pd.DataFrame) -> pd.DataFrame:
    """One value per company-line-period-vintage key.

    A key repeats in three ways, each resolved by accounting structure and counted:
      parent/child   a line and its own sub-line carry the same words -> keep the parent
                     (fewest dots in CD_CONTA); the child is part of it.
      zero sibling   same-level accounts where one is zero and another is not -> drop the
                     zero (a filer leaves an empty account next to the one it uses; for
                     example Cielo's two "Vendas e Marketing" accounts 3.04.02.03/.04).
      siblings       same-level accounts that remain: equal values are one entry filed twice
                     -> keep one; different values are distinct accounts -> sum, and mark
                     summed_siblings.
    If the same key also appears in two documents (DFP and ITR), the DFP rows are used.
    """
    key = ["CD_CVM", "scope", "statement", "line_id", "period_end", "period_type", "vintage"]
    df = lines.copy()
    df["depth"] = df["CD_CONTA"].str.count(r"\.")

    counts = {"parent_child": 0, "zero_sibling": 0, "duplicate_entry": 0, "summed_siblings": 0,
              "two_documents": 0}
    first_doc = df.groupby(key)["doc"].transform("min")
    counts["two_documents"] = int((df["doc"] != first_doc).sum())
    df = df[df["doc"] == first_doc]

    top = df.groupby(key)["depth"].transform("min")
    counts["parent_child"] = int((df["depth"] != top).sum())
    df = df[df["depth"] == top].copy()

    has_nonzero = df.groupby(key)["value_brl"].transform(lambda s: (s != 0).any())
    zero_sib = (df["value_brl"] == 0) & has_nonzero
    counts["zero_sibling"] = int(zero_sib.sum())
    df = df[~zero_sib]

    n = df.groupby(key)["value_brl"].transform("size")
    distinct = df.groupby(key)["value_brl"].transform("nunique")
    dup_entry = (n > 1) & (distinct == 1)
    counts["duplicate_entry"] = int((dup_entry & df.duplicated(key)).sum())
    df = df[~(dup_entry & df.duplicated(key))].copy()

    multi = df.groupby(key)["value_brl"].transform("size") > 1
    counts["summed_siblings"] = int(multi.sum())
    df["summed_siblings"] = multi
    summed = (df[multi].groupby(key, as_index=False)
                .agg(value_brl=("value_brl", "sum"),
                     **{c: (c, "first") for c in df.columns
                        if c not in key + ["value_brl"]}))
    summed["summed_siblings"] = True
    df = pd.concat([df[~multi], summed], ignore_index=True).drop(columns="depth")
    log.info("key collisions resolved: %s", counts)
    return df


# ---------------------------------------------------------------------------
# Quarterly flows
# ---------------------------------------------------------------------------
def quarterly(lines: pd.DataFrame) -> pd.DataFrame:
    ids = ["CD_CVM", "scope", "statement", "line_id"]
    ytd = lines[lines["period_type"] == "ytd"]
    wide = (ytd.pivot_table(index=ids + ["year", "quarter"], columns="vintage",
                            values="value_brl", aggfunc="first")
                .reset_index())
    wide.columns.name = None
    for v in ("first", "restated"):
        if v not in wide.columns:
            wide[v] = np.nan

    zero_comp = (wide["restated"] == 0) & (wide["first"].fillna(0) != 0)
    log.info("zero comparatives against a non-zero first-reported value, treated as missing: %d",
             int(zero_comp.sum()))
    wide.loc[zero_comp, "restated"] = np.nan

    # Complete calendar per company-line so a missing quarter stays visible.
    spans = wide.groupby(ids)["year"].agg(["min", "max"]).reset_index()
    grid = [(*r[:4], y, q) for r in spans.itertuples(index=False, name=None)
            for y in range(r[4], r[5] + 1) for q in (1, 2, 3, 4)]
    grid = pd.DataFrame(grid, columns=ids + ["year", "quarter"])
    wide = grid.merge(wide, on=ids + ["year", "quarter"], how="left")
    wide = wide.sort_values(ids + ["year", "quarter"]).reset_index(drop=True)

    same_year = ((wide[ids + ["year"]] == wide[ids + ["year"]].shift()).all(axis=1)).to_numpy()
    q1 = (wide["quarter"] == 1).to_numpy()
    for v in ("first", "restated"):
        cur = wide[v].to_numpy()
        prev = wide[v].shift().to_numpy()
        wide[f"flow_{v}"] = np.where(q1, cur, np.where(same_year, cur - prev, np.nan))
    wide = wide.rename(columns={"first": "ytd_first", "restated": "ytd_restated"})

    wide["ytd_falls"] = (wide["flow_first"] < 0) | (wide["flow_restated"] < 0)
    ratio = wide["ytd_first"] / wide["ytd_restated"]
    wide["scale_suspect"] = ratio.between(0.00095, 0.00105) | ratio.between(950, 1050)

    # Zeros carry no sign; the usual sign is the mode of the non-zero filed values.
    nonzero = lines[(lines["period_type"] == "ytd") & (lines["sign_raw"] != 0)]
    signs = nonzero.groupby(ids + ["year", "quarter"])["sign_raw"].first()
    usual = signs.groupby(level=ids).agg(lambda s: s.mode().iloc[0])
    sign_flip = (signs != usual.reindex(signs.index.droplevel(["year", "quarter"])).to_numpy())
    wide = wide.merge(sign_flip.rename("sign_flip").reset_index(), on=ids + ["year", "quarter"],
                      how="left")
    wide["sign_flip"] = wide["sign_flip"].eq(True)

    # The ITR DRE quarter-only figure, where filed, against the year-to-date difference. A
    # mismatch means the filer's own figures disagree (for example a Q1 year-to-date filed as
    # zero while the Q2 filing implies a positive Q1); it is flagged, not corrected.
    qo = (lines[(lines["period_type"] == "quarter") & (lines["vintage"] == "first")]
          .groupby(ids + ["year", "quarter"])["value_brl"].first()
          .rename("quarter_only_first").reset_index())
    wide = wide.merge(qo, on=ids + ["year", "quarter"], how="left")
    both = wide["quarter_only_first"].notna() & wide["flow_first"].notna()
    wide["qo_mismatch"] = both & ((wide["flow_first"] - wide["quarter_only_first"]).abs()
                                  > 0.005 * wide["quarter_only_first"].abs())

    names = lines.groupby(ids, as_index=False).agg(DENOM_CIA=("DENOM_CIA", "last"),
                                                   cnpj8=("cnpj8", "last"),
                                                   bcb_supervised=("bcb_supervised", "last"),
                                                   DS_CONTA=("DS_CONTA", "last"),
                                                   n_label_variants=("label_norm", "nunique"))
    return names.merge(wide, on=ids, how="right")


def supervised_cnpj8() -> set[str]:
    """CNPJ8 of every institution and conglomerate leader in the IF.data lists, 2013 onward."""
    files = sorted(IF_DATA_LIST.glob("IF_DATA_List_*.csv"))
    if not files:
        raise SystemExit(f"No IF_DATA_List_*.csv under {IF_DATA_LIST}; needed to mark "
                         f"BCB-supervised filers")
    found: set[str] = set()
    for f in files:
        d = pd.read_csv(f, sep=None, engine="python", dtype=str, encoding="latin1",
                        usecols=["CodInst", "CnpjInstituicaoLider"])
        for col in ("CodInst", "CnpjInstituicaoLider"):
            v = d[col].dropna().str.strip()
            found |= set(v[v.str.fullmatch(r"\d{1,8}")].str.zfill(8))
    log.info("BCB-supervised CNPJ8s from %d IF.data lists: %d", len(files), len(found))
    return found


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate(lines: pd.DataFrame, q: pd.DataFrame, since: int, until: int) -> None:
    bb = q[(q["CD_CVM"] == BB_CVM) & (q["scope"] == "ind") & (q["statement"] == "DVA")
           & (q["quarter"] == 4)]
    if bb["line_id"].nunique() != 1:
        raise AssertionError(f"Banco do Brasil individual DVA should be one advertising series, "
                             f"found {sorted(bb['line_id'].unique())}")
    years = set(bb.loc[bb["ytd_first"].notna(), "year"])
    need = set(range(max(since, 2013), min(until, 2024) + 1))
    if need - years:                                                          # check 2
        raise AssertionError(f"Banco do Brasil individual DVA propaganda annual missing for "
                             f"{sorted(need - years)}")

    both = q["quarter_only_first"].notna() & q["flow_first"].notna() \
        & (q["quarter_only_first"] > 0)                                       # check 3
    for label, mask in (("BCB-supervised", both & q["bcb_supervised"]),
                        ("other listed companies", both & ~q["bcb_supervised"])):
        if mask.any():
            log.info("quarter-only DRE values vs year-to-date differences, %s: %.1f%% agree "
                     "(n=%d)", label, 100 * (1 - q.loc[mask, "qo_mismatch"].mean()),
                     int(mask.sum()))
    sup = both & q["bcb_supervised"]
    if sup.any():
        agree = 1 - q.loc[sup, "qo_mismatch"].mean()
        if agree < 0.95:
            raise AssertionError(f"BCB-supervised filers: only {agree:.1%} of quarter-only "
                                 f"values match the year-to-date differences")

    cosif = OUT_DIR / "cosif_advertising_quarterly.parquet"                  # check 4
    if not cosif.exists():
        log.warning("COSIF advertising panel not found at %s; skipping the 2025 anchor", cosif)
        return
    c = pd.read_parquet(cosif)
    c25 = c[(c["level"] == "institution") & (c["entity_key"] == "00000000") & (c["year"] == 2025)]
    bb25 = bb[bb["year"] == 2025]["ytd_first"]
    if len(c25) == 4 and len(bb25) == 1 and c25["n_months_filed"].eq(3).all():
        ratio = float(bb25.iloc[0]) / float(c25["adv"].sum())
        log.info("Banco do Brasil 2025: CVM individual DVA / COSIF institution adv = %.4f", ratio)
        if abs(ratio - 1) > 0.005:
            raise AssertionError(f"Banco do Brasil 2025 anchor failed: ratio {ratio:.4f}")
    else:
        log.warning("Banco do Brasil 2025 anchor not testable (COSIF quarters %d, CVM annual %d)",
                    len(c25), len(bb25))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="CVM advertising-type expense lines")
    ap.add_argument("--since", type=int, default=2013)
    ap.add_argument("--until", type=int, default=dt.date.today().year)
    ap.add_argument("--refresh", action="store_true", help="redownload the yearly zips")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    a = ap.parse_args()

    lines = build_lines(a.since, a.until, a.refresh)
    lines["bcb_supervised"] = lines["cnpj8"].isin(supervised_cnpj8())
    q = quarterly(lines)
    validate(lines, q, a.since, a.until)

    log.info("flags: ytd_falls %d, scale_suspect %d, sign_flip %d (company-line-quarters)",
             int(q["ytd_falls"].sum()), int(q["scale_suspect"].sum()), int(q["sign_flip"].sum()))
    suspects = q[q["scale_suspect"]]
    if len(suspects):
        log.warning("scale_suspect rows:\n%s", suspects[["DENOM_CIA", "scope", "statement",
                    "line_id", "year", "quarter", "ytd_first", "ytd_restated"]].to_string(index=False))

    a.out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (("cvm_advertising_lines", lines), ("cvm_advertising_quarterly", q)):
        frame.to_parquet(a.out_dir / f"{name}.parquet", index=False)
        frame.to_csv(a.out_dir / f"{name}.csv", index=False)
        log.info("%s: %d rows -> %s", name, len(frame), a.out_dir / f"{name}.parquet")
    log.info("companies with an advertising-type line: %d", q["CD_CVM"].nunique())


if __name__ == "__main__":
    main()

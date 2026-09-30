"""
build_advertising_panel.py
==========================
Assembles the collected advertising sources into panels keyed to the market panel's prudential
conglomerate codes:

  advertising_panel_quarterly   one row per conglomerate x quarter x SOURCE x measure
  advertising_panel_annual      the SEC 20-F series, annual, kept apart from the quarterly panel
  advertising_ancine_quarterly  ANCINE advertising-film registrations (CRTs) per conglomerate x
                                quarter. They count films, not reais, so they are kept out of the
                                amount panels, where the deflator and the deposit intensities
                                would treat them as money (user decision 2026-09-29).

Sources are stored SEPARATELY and never spliced (user decision 2026-09-20): a source change
inside a bank's history is a level jump that bank fixed effects cannot absorb, and the measured
2025 ratios differ by source (Caixa's outlays are 1.36x the accrued account, Banestes 0.56-0.80,
BRB only matches once sponsorship is included). The estimation picks a source, or adds source
dummies; nothing is rescaled here.

Source rows
-----------
  cosif_conglomerate  accrued expense per prudential conglomerate, 2025-01 onward
  cosif_institution   accrued expense per institution, 2025-01 onward
  cvm                 securities-filing lines, quarterly 2013-2025, INDIVIDUAL statements summed
                      over the filers mapped to one conglomerate. EXCEPTION: Itau Unibanco
                      Holding uses its CONSOLIDATED statement (its operating bank does not file
                      with the CVM and the holding's parent-only line is a different, tiny
                      object), and the individual filings of entities inside that consolidation
                      are then excluded so nothing is counted twice.
  caixa_own           Caixa's published monthly outlays, 2013 onward
  statebank_own       Banestes, BASA, Banrisul, BRB published outlays
  sponsorship         BB, BNB and Caixa sponsorship, a separate robustness series

Measures (one row each, so a consumer picks what it wants)
---------------------------------------------------------
  adv                 advertising only
  adv_production      advertising plus campaign production - the main own-file measure, which is
                      what matches the COSIF advertising account
  promo, publ         promotions and publications, where the source reports them apart
  all3                advertising plus promotions plus publications
  legal_notice        statutory publications from the own files
  sponsorship         sponsorship, by amount kind (paid, contracted, disbursed)

Vintages and values
-------------------
  - Filing values use the LATEST filed figure, except where first and latest differ by a factor
    near 1000, which is a keying slip and not a revision; there the non-slipped value is used and
    `flags` records it (Santander FY2022 latest is the slip; FY2024 first is).
  - amount_brl is nominal. amount_brl_real is deflated by the IPCA index to the base quarter named
    in `deflator_base`. Intensity columns divide the nominal amount by the conglomerate's deposits
    in that quarter (from market_panel.parquet, read-only) and, where 2025 COSIF data exists, by
    its administrative expenses.
  - `validated` is False for rows from a source whose own published totals did not reconcile
    against a 90% floor, and for a Caixa quarter holding a month whose lines do not close against
    Caixa's printed totals (flag month_totals_do_not_close; the reviewed months are Caixa's own
    totals disagreeing with its cells, and their amounts still count). Every state bank clears it; BRB, the lowest, matches 96.6% (393 of 407
    published totals, measured 2026-09-24 once duplicate gazette copies were counted once).
  - statebank_own sums only the entities inside the bank's prudential conglomerate: BRB's gazette
    pages also carry Cartao BRB, which the IF.data registry never places in C0080288, so its lines
    stay in the scraper's files and out of the panel.
  - `value_status` is 'imputed_from_row_total' on a cell that includes a line the scraper imputed
    from its table's own arithmetic, and `imputed_brl` holds the imputed part (one cell group
    today: BRB 2018Q1, where a printed January figure contradicts both the row's own total and the
    published January total). Every other cell's status is unchanged.

Inputs (all under paths.AWARENESS_PROC unless noted)
----------------------------------------------------
  cosif_advertising_quarterly, cvm_advertising_quarterly, sec_advertising_annual,
  caixa_advertising_monthly, caixa_scan_advertising_monthly, statebank_advertising_periods,
  sponsorship_periods, ancine_ad_films_monthly,
  advertising_entity_crosswalk{,_periods}; market_panel.parquet for deposits; the IPCA series from
  the central bank's SGS API, cached under paths.AWARENESS_RAW/deflator.

Validation (any failure aborts before writing)
----------------------------------------------
  1. Every input on disk; each source contributes at least one row.
  2. Every row carries a panel_code present in the market panel, or is dropped with a counted
     reason (never silently).
  3. No conglomerate-quarter-source-measure key appears twice.
  4. The deflator covers every quarter in the panel.
  5. Double-count guard: no conglomerate-quarter mixes a consolidated filing with an individual
     filing of an entity inside it.
  6. Reconciliation printed, not gated: for 2025, each source against the COSIF conglomerate
     accounts.

Usage
-----
  python build_advertising_panel.py
  python build_advertising_panel.py --refresh-deflator --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from utils import paths
from utils.disclosure_common import http_get

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

PROC = paths.AWARENESS_PROC
RAW = paths.AWARENESS_RAW / "deflator"
MARKET_PANEL = Path(str(paths.market_panel_csv()).replace(".csv", ".parquet"))
IPCA_URL = ("https://api.bcb.gov.br/dados/serie/bcdata.sgs.433/dados"
            "?formato=json&dataInicial=01/01/2012&dataFinal=31/12/2026")
ITAU_CONSOLIDATED_CD_CVM = "019348"

# One element per SEC filer, in preference order: (element, kind, note). Chosen from the
# reconciliation in ADVERTISING_DATA.md section 1.3, not from a name pattern.
SEC_ELEMENTS: dict[str, list[tuple[str, str, str]]] = {
    "1160330": [("bbd:OtherAdministrativeExpensesAdvertisingPromotionsAndPublicRelations",
                 "advertising_specific", "advertising, promotions and public relations")],
    "1132597": [("ifrs-full:SalesAndMarketingExpense", "advertising_specific",
                 "generic element name, but equals Itau's CVM DVA advertising line; the fact is "
                 "dimensioned on AttributionOfExpensesByNatureToTheirFunctionAxis")],
    "1471055": [("ifrs-full:SalesAndMarketingExpense", "broad_sales_marketing",
                 "broader than advertising; Santander tags no advertising-specific element")],
    "1691493": [("nu:BrandingAndAdvertising", "advertising_specific", "USD, group-wide"),
                ("ifrs-full:SalesAndMarketingExpense", "broad_sales_marketing",
                 "USD, group-wide; used for years before the branding element exists")],
    "1864163": [("ifrs-full:AdvertisingExpense", "advertising_specific", "")],
    "1787425": [("ifrs-full:AdvertisingExpense", "advertising_specific", "")],
    "1712807": [("ifrs-full:AdvertisingExpense", "advertising_specific", "")],
    "1745431": [("ifrs-full:SalesAndMarketingExpense", "broad_sales_marketing",
                 "includes selling costs; StoneCo tags no advertising-specific element")],
}
DEP_COLS = ["dep_a1", "dep_a2", "dep_a4", "dep_a5"]
KEY = ["panel_code", "year", "quarter", "source", "measure", "amount_kind", "scope"]


# ---------------------------------------------------------------------------
# Deflator
# ---------------------------------------------------------------------------
def ipca_index(refresh: bool) -> pd.DataFrame:
    """Monthly IPCA variation from the central bank's SGS series 433, as a chained index."""
    RAW.mkdir(parents=True, exist_ok=True)
    cache = RAW / "sgs_433_ipca.json"
    if not cache.exists() or refresh:
        code, body = http_get(IPCA_URL, timeout=120)
        if not body:
            raise SystemExit(f"IPCA download failed (HTTP {code}); cannot deflate")
        cache.write_text(body, encoding="utf-8")
    rows = json.loads(cache.read_text(encoding="utf-8"))
    d = pd.DataFrame(rows)
    d["date"] = pd.to_datetime(d["data"], format="%d/%m/%Y")
    d["pct"] = pd.to_numeric(d["valor"].str.replace(",", "."), errors="raise")
    d = d.sort_values("date")
    d["index"] = (1 + d["pct"] / 100).cumprod()
    d["year"] = d["date"].dt.year
    d["quarter"] = d["date"].dt.quarter
    q = (d.groupby(["year", "quarter"], as_index=False)
          .agg(**{"index": ("index", "mean"), "n_months": ("pct", "size")}))
    # Base pinned to the estimation window's last quarter, so the real series does not shift
    # every time a new month of inflation is published.
    complete = q[q["n_months"] == 3]
    pinned = complete[(complete["year"] == 2024) & (complete["quarter"] == 4)]
    base = pinned.iloc[0] if len(pinned) else complete.iloc[-1]
    q["deflator"] = base["index"] / q["index"]          # multiply nominal to reach base prices
    log.info("IPCA deflator: %d quarters, base %d Q%d", len(q), int(base["year"]),
             int(base["quarter"]))
    return q, f"{int(base['year'])}Q{int(base['quarter'])}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def crosswalk() -> tuple[pd.DataFrame, pd.DataFrame]:
    xw = pd.read_parquet(PROC / "advertising_entity_crosswalk.parquet")
    per = pd.read_parquet(PROC / "advertising_entity_crosswalk_periods.parquet")
    return xw, per


def code_for(xw: pd.DataFrame, per: pd.DataFrame, source: str, source_id: str,
             year: int, quarter: int, panel_names: dict[str, str] | None = None,
             entity_name: str | None = None) -> str | None:
    """The conglomerate code for a source entity, as the MARKET PANEL identifies it.

    A filer with one panel code takes it. Where the registry gives several, the quarter's
    registry code is NOT automatically right: the registry moved Banco Pan into BTG's prudential
    conglomerate in 2021Q2, while the market panel keeps BANCO PAN SA as its own entity with
    R$10-22bn of deposits through 2025Q4. Attaching Pan's advertising to BTG's code would
    contaminate BTG and empty Pan. So among the candidates present in the panel, the one whose
    panel name shares a distinctive token with the filer's own name wins; the quarter's registry
    code is used only when no name evidence exists, and the choice is logged either way.
    """
    row = xw[(xw["source"] == source) & (xw["source_id"] == source_id)]
    if len(row) and pd.notna(row["panel_code"].iloc[0]):
        return row["panel_code"].iloc[0]
    ym = f"{year}{quarter * 3:02d}"
    cand = per[(per["source"] == source) & (per["source_id"] == source_id) & per["in_panel"]]
    if cand.empty:
        return None
    codes = list(dict.fromkeys(cand["panel_code_candidate"]))
    if panel_names and entity_name and len(codes) > 1:
        # Three characters is the floor, not four: "PAN" is the only distinctive token in
        # "BCO PAN S.A.", and dropping it sent Pan's advertising to BTG's conglomerate.
        tokens = {t for t in _norm_tokens(entity_name) if len(t) >= 3}
        scored = [(len(tokens & set(_norm_tokens(panel_names.get(c, "")))), c) for c in codes]
        best, code = max(scored)
        if best:
            _CODE_CHOICE.setdefault((source, source_id), f"name match on {code} among {codes}")
            return code
    live = cand[(cand["first_quarter"] <= ym) & (cand["last_quarter"] >= ym)]
    code = live["panel_code_candidate"].iloc[0] if len(live) else codes[0]
    _CODE_CHOICE.setdefault((source, source_id), f"registry span -> {code} among {codes}")
    return code


_CODE_CHOICE: dict[tuple[str, str], str] = {}


def _norm_tokens(name: str) -> list[str]:
    import unicodedata
    s = unicodedata.normalize("NFD", str(name).upper())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    drop = {"BANCO", "BCO", "S.A.", "SA", "S/A", "LTDA", "HOLDING", "PRUDENCIAL", "DO", "DA",
            "DE", "E", "S.A", "FINANCEIRA", "LEASING", "ARRENDAMENTO"}
    return [t for t in re.split(r"[^A-Z0-9]+", s) if t and t not in drop]


# The crosswalk keys its periods by its own source names; the panel's sources are finer.
REGISTRY_SOURCE = {"cosif_institution": "cosif_institution", "cvm": "cvm", "sec": "sec",
                   "statebank_own": "statebank", "caixa_own": "statebank",
                   "sponsorship": "statebank"}


def attach_registry_code(panel: pd.DataFrame, xw: pd.DataFrame,
                         per: pd.DataFrame) -> pd.DataFrame:
    """Store the code the IF.data registry gives the filer IN THAT QUARTER beside the panel code.

    The two disagree where the registry moves a bank between conglomerates while the market panel
    keeps it as its own entity (Banco Pan into BTG's prudential conglomerate from 2021Q2). code_for
    deliberately follows the panel in that case, and `code_conflict` keeps that decision visible in
    the output instead of leaving it in a log line. Conglomerate-file rows are keyed by the
    registry's own conglomerate code, so for those the source id IS the registry verdict.
    """
    src = panel["source"].map(REGISTRY_SOURCE)
    rows = pd.DataFrame({"source": src, "ids": panel["source_entity"].astype(str),
                         "ym": panel["year"].astype("Int64").astype(str)
                               + (panel["quarter"] * 3).map(lambda q: f"{int(q):02d}")})
    rows = rows.reset_index().assign(source_id=lambda d: d["ids"].str.split(";")).explode("source_id")
    m = rows.merge(per[["source", "source_id", "panel_code_candidate", "first_quarter",
                        "last_quarter"]], on=["source", "source_id"], how="inner")
    m = m[(m["first_quarter"] <= m["ym"]) & (m["last_quarter"] >= m["ym"])]
    got = (m.groupby("index")["panel_code_candidate"]
             .agg(lambda x: "|".join(sorted(set(x.dropna())))).replace("", np.nan))
    # An entity the registry maps to ONE code the whole window has no dated row, only the
    # undated crosswalk entry; without this fill its registry_code would be NaN and a reader
    # could not tell "the registry agreed" from "the registry says nothing".
    flat = (xw.dropna(subset=["panel_code"]).set_index(["source", "source_id"])["panel_code"])
    idx = pd.MultiIndex.from_arrays([rows["source"], rows["source_id"]])
    rows = rows.assign(flat_code=flat.reindex(idx).to_numpy())
    fill = (rows.dropna(subset=["flat_code"]).groupby("index")["flat_code"]
                .agg(lambda x: "|".join(sorted(set(x)))))
    reg = got.reindex(panel.index).fillna(fill.reindex(panel.index))
    congl = panel["source"] == "cosif_conglomerate"
    panel["registry_code"] = reg.where(~congl, panel["source_entity"])
    panel["code_conflict"] = (panel["registry_code"].notna()
                              & (panel["registry_code"] != panel["panel_code"]))
    n = int(panel["code_conflict"].sum())
    if n:
        pairs = (panel[panel["code_conflict"]]
                 .groupby(["source", "source_entity", "entity_name", "registry_code",
                           "panel_code"], dropna=False).size())
        log.info("registry code differs from the panel code on %d rows (the panel wins by "
                 "design):\n%s", n, pairs.to_string())
    log.info("registry code stored on %d of %d rows", int(panel["registry_code"].notna().sum()),
             len(panel))
    return panel


def rows_from(frame: pd.DataFrame, **fixed) -> pd.DataFrame:
    out = frame.copy()
    for k, v in fixed.items():
        out[k] = v
    return out


# ---------------------------------------------------------------------------
# Source builders
# ---------------------------------------------------------------------------
def from_cosif(xw: pd.DataFrame, per: pd.DataFrame, panel_names: dict[str, str]) -> pd.DataFrame:
    q = pd.read_parquet(PROC / "cosif_advertising_quarterly.parquet")
    q = q[q["n_months_filed"] == 3].copy()
    q["measure_map"] = None
    # Institution rows are keyed by CNPJ8, so their conglomerate comes from the crosswalk (a
    # bank that changed conglomerate is mapped per quarter). Conglomerate rows already carry the
    # panel code.
    inst = q[q["level"] == "institution"].drop_duplicates("entity_key")[["entity_key",
                                                                        "entity_name"]]
    inst_code = {r.entity_key: code_for(xw, per, "cosif_institution", r.entity_key, 2025, 4,
                                        panel_names, r.entity_name)
                 for r in inst.itertuples()}
    out = []
    for level, source in (("conglomerate", "cosif_conglomerate"),
                          ("institution", "cosif_institution")):
        sub = q[q["level"] == level]
        for measure, col in (("adv", "adv"), ("promo", "promo"), ("publ", "publ"),
                             ("all3", "all3")):
            out.append(pd.DataFrame({
                "panel_code": (sub["panel_key"] if level == "conglomerate"
                               else sub["entity_key"].map(inst_code)),
                "source_entity": sub["entity_key"], "entity_name": sub["entity_name"],
                "year": sub["year"], "quarter": sub["quarter"], "source": source,
                "measure": measure, "amount_kind": "accrued", "scope": level,
                "amount_brl": sub[col], "n_months_observed": 3,
                "validated": True,
                "flags": np.where(sub["any_adv_spike"] & (measure in ("adv", "all3")),
                                  "adv_spike", "") ,
                "detail": sub["taxonomia"],
            }))
    df = pd.concat(out, ignore_index=True)
    admin = q.set_index(["level", "entity_key", "year", "quarter"])["admin"]
    idx = pd.MultiIndex.from_arrays([df["scope"], df["source_entity"], df["year"], df["quarter"]])
    df["admin_brl"] = admin.reindex(idx).to_numpy()

    # Several institutions belong to one conglomerate (Banco do Brasil's conglomerate holds the
    # bank, its finance company and more), so the institution rows are summed to the conglomerate,
    # the same aggregation the filings get. The result is deliberately NOT identical to the
    # conglomerate document, which consolidates non-bank members too; both are kept as separate
    # sources so the difference stays visible.
    inst = df[df["scope"] == "institution"]
    keys = ["panel_code", "year", "quarter", "source", "measure", "amount_kind", "scope"]
    inst_agg = (inst.groupby(keys, as_index=False, dropna=False)
                    .agg(amount_brl=("amount_brl", "sum"), admin_brl=("admin_brl", "sum"),
                         n_entities=("source_entity", "nunique"),
                         source_entity=("source_entity", lambda s: ";".join(sorted(set(s))[:4])),
                         entity_name=("entity_name", "first"),
                         flags=("flags", lambda s: ";".join(sorted({x for x in s if x}))),
                         detail=("detail", "first"),
                         n_months_observed=("n_months_observed", "max"),
                         validated=("validated", "all")))
    congl = df[df["scope"] == "conglomerate"].assign(n_entities=1)
    return pd.concat([congl, inst_agg], ignore_index=True)


def panel_name_map() -> dict[str, str]:
    mp = pd.read_parquet(MARKET_PANEL, columns=["CodConglomeradoPrudencial", "NomeInstituicao"])
    return (mp.dropna(subset=["NomeInstituicao"]).drop_duplicates("CodConglomeradoPrudencial")
              .set_index("CodConglomeradoPrudencial")["NomeInstituicao"].to_dict())


def from_cvm(xw: pd.DataFrame, per: pd.DataFrame, panel_names: dict[str, str]) -> pd.DataFrame:
    q = pd.read_parquet(PROC / "cvm_advertising_quarterly.parquet")
    q = q[q["bcb_supervised"]].copy()
    # Latest filed wins; a factor-1000 gap is a keying slip, so the non-slipped value is used.
    latest = q["flow_restated"].where(q["flow_restated"].notna(), q["flow_first"])
    slip = q["scale_suspect"].fillna(False)
    # In a near-1000 ratio pair the SMALL member is the keying slip, so the LARGE one is kept.
    # The test therefore fires when the restated (default) value is the small one, i.e. when
    # first/restated is large. Reading it the other way round kept the slip and produced a
    # negative half-billion Itau quarter in 2013Q3.
    ratio = (q["ytd_first"].abs() / q["ytd_restated"].abs()).replace([np.inf, -np.inf], np.nan)
    use_first = slip & (ratio > 10)
    value = latest.where(~use_first, q["flow_first"])
    q = q.assign(value=value,
                 flags=np.where(slip, "scale_suspect_resolved", "")
                       + np.where(q["qo_mismatch"].fillna(False), ";qo_mismatch", "")
                       + np.where(q["ytd_falls"].fillna(False), ";ytd_falls", ""))
    itau_con = q[(q["CD_CVM"] == ITAU_CONSOLIDATED_CD_CVM) & (q["scope"] == "con")]
    others = q[(q["scope"] == "ind") & (q["CD_CVM"] != ITAU_CONSOLIDATED_CD_CVM)]
    # Entities inside Itau's consolidation must not be added to it (Dibens Leasing files too).
    itau_code = xw.loc[(xw["source"] == "cvm")
                       & (xw["source_id"] == ITAU_CONSOLIDATED_CD_CVM), "panel_code"]
    itau_code = itau_code.iloc[0] if len(itau_code) else None
    keep = []
    for r in pd.concat([itau_con, others]).itertuples():
        code = code_for(xw, per, "cvm", r.CD_CVM, r.year, r.quarter, panel_names, r.DENOM_CIA)
        if code is None or pd.isna(r.value):
            continue
        if code == itau_code and r.CD_CVM != ITAU_CONSOLIDATED_CD_CVM:
            continue                      # inside the consolidated figure already
        keep.append({"panel_code": code, "source_entity": r.CD_CVM, "entity_name": r.DENOM_CIA,
                     "year": r.year, "quarter": r.quarter, "source": "cvm",
                     "measure": "as_filed", "amount_kind": "accrued",
                     "scope": "consolidated" if r.scope == "con" else "individual",
                     "amount_brl": float(r.value), "n_months_observed": 3, "validated": True,
                     "flags": r.flags.strip(";"), "detail": f"{r.statement} {r.line_id}"})
    df = pd.DataFrame(keep)
    agg = (df.groupby(["panel_code", "year", "quarter", "source", "measure", "amount_kind",
                       "scope"], as_index=False)
             .agg(amount_brl=("amount_brl", "sum"), n_filers=("source_entity", "nunique"),
                  entity_name=("entity_name", "first"), source_entity=("source_entity",
                                                                       lambda s: ";".join(sorted(set(s)))),
                  flags=("flags", lambda s: ";".join(sorted({x for x in s if x}))),
                  detail=("detail", lambda s: ";".join(sorted(set(s))[:3]))))
    agg["n_months_observed"] = 3
    agg["validated"] = True
    agg["admin_brl"] = np.nan
    return agg


def from_caixa(xw: pd.DataFrame, per: pd.DataFrame) -> pd.DataFrame:
    """Caixa's own monthly files, from two builders that read disjoint months: the text-read months
    (scrape_caixa_advertising.py) and the scanned ones it cannot read, 2013-01..2014-06 and
    2016-04..05 (scrape_caixa_scan_advertising.py, OCR). A quarter is validated only when every
    month in it closes against Caixa's printed totals, exactly or within 5 centavos. A month that
    does not close is still counted - its lines are what Caixa printed; the reviewed cases are
    Caixa's own totals disagreeing with its cells - and the quarter's flags say so."""
    cols = ["period", "amount_brl", "validated", "validation_status", "tolerance_flagged"]
    text = pd.read_parquet(PROC / "caixa_advertising_monthly.parquet")
    text = text.loc[text["status"].eq("parsed"), cols].assign(read="text")
    scan = pd.read_parquet(PROC / "caixa_scan_advertising_monthly.parquet")[cols].assign(read="scan")
    both = set(text["period"].astype(str)) & set(scan["period"].astype(str))
    if both:
        raise SystemExit(f"Caixa months read by both builders: {sorted(both)}")
    m = pd.concat([text, scan], ignore_index=True)
    m["validated"] = m["validated"].astype(bool)
    m["tolerance_flagged"] = m["tolerance_flagged"].fillna(False).astype(bool)
    m["year"] = m["period"].astype(str).str[:4].astype(int)
    m["quarter"] = ((m["period"].astype(str).str[4:6].astype(int) - 1) // 3 + 1)
    g = (m.groupby(["year", "quarter"], as_index=False)
          .agg(amount_brl=("amount_brl", "sum"), n_months_observed=("period", "nunique"),
               validated=("validated", "all"),
               n_unvalidated=("validated", lambda s: int((~s).sum())),
               n_tolerance=("tolerance_flagged", "sum"),
               n_scanned=("read", lambda s: int((s == "scan").sum()))))
    code = code_for(xw, per, "statebank", "caixa", 2020, 1)
    g["panel_code"], g["source_entity"], g["entity_name"] = code, "caixa", "CAIXA ECONOMICA FEDERAL"
    g["source"], g["measure"], g["amount_kind"] = "caixa_own", "adv_production", "custos"
    g["scope"], g["admin_brl"] = "institution", np.nan
    g["flags"] = (np.where(g["n_months_observed"] < 3, "incomplete_quarter;", "")
                  + np.where(g["n_unvalidated"] > 0, "month_totals_do_not_close;", "")
                  + np.where(g["n_tolerance"] > 0, "month_within_5_centavos;", "")
                  + np.where(g["n_scanned"] > 0, "month_read_by_ocr;", ""))
    g["flags"] = g["flags"].str.strip(";")
    g["detail"] = "media, production and services by agency"
    return g.drop(columns=["n_unvalidated", "n_tolerance", "n_scanned"])


def brb_with_sponsorship(statebank: pd.DataFrame) -> pd.DataFrame:
    """BRB books sponsorship inside its advertising account: its own advertising plus sponsorship
    equals 0.94-1.06 of that account in 2025, while advertising alone is 0.21-0.32. So BRB gets an
    extra measure that includes sponsorship, flagged; the standard measures are untouched."""
    brb = statebank[statebank["source_entity"] == "brb"]
    if brb.empty:
        return pd.DataFrame()
    parts = brb[brb["measure"].isin(["adv_production", "sponsorship"])]
    g = (parts.groupby(["panel_code", "year", "quarter", "source", "scope", "amount_kind"],
                       as_index=False)
              .agg(amount_brl=("amount_brl", "sum"),
                   imputed_brl=("imputed_brl", lambda s: s.sum(min_count=1)),
                   n_months_observed=("n_months_observed", "max"),
                   validated=("validated", "all"), entity_name=("entity_name", "first"),
                   source_entity=("source_entity", "first"), admin_brl=("admin_brl", "first")))
    g["measure"] = "adv_production_plus_sponsorship"
    g["flags"] = "brb_books_sponsorship_in_advertising_account"
    g["detail"] = "advertising plus production plus sponsorship, to match BRB's own account"
    return g


def from_statebanks(xw: pd.DataFrame, per: pd.DataFrame) -> pd.DataFrame:
    """Own-file outlays summed per bank, quarter and measure.

    Only entities inside the bank's prudential conglomerate are summed. BRB's gazette pages also
    carry Cartao BRB's notice, which the IF.data registry never places in BRB's conglomerate, and
    a table whose heading could not be read has no entity to place; both stay in the scraper's
    files, marked `in_conglomerate` = False, and out of these totals. `imputed_brl` is the part of
    a cell that the scraper imputed from the table's own arithmetic (`imputed_<group>` in the
    periods file); it is empty wherever every line was taken as printed."""
    p = pd.read_parquet(PROC / "statebank_advertising_periods.parquet")
    p = p[p["period"].astype(str).str.len() == 6].copy()          # monthly rows
    out_of = p[~p["in_conglomerate"].astype(bool)]
    if len(out_of):
        log.info("statebank rows outside the conglomerate, left out of the panel:\n%s",
                 out_of.groupby(["bank_key", "entity"]).agg(
                     months=("period", "size"), first=("period", "min"), last=("period", "max"),
                     n_lines=("n_lines", "sum")).to_string())
    p = p[p["in_conglomerate"].astype(bool)].copy()
    p["year"] = p["period"].astype(str).str[:4].astype(int)
    p["quarter"] = ((p["period"].astype(str).str[4:6].astype(int) - 1) // 3 + 1)
    groups = {"adv": ["advertising"], "adv_production": ["advertising", "production"],
              "legal_notice": ["legal_notice"], "sponsorship": ["sponsorship"]}
    out = []
    for measure, cols in groups.items():
        have = [c for c in cols if c in p.columns]
        if not have:
            continue
        imp = [f"imputed_{c}" for c in have if f"imputed_{c}" in p.columns]
        g = (p.assign(v=p[have].sum(axis=1, min_count=1),
                      imp=p[imp].sum(axis=1) if imp else 0.0)
              .groupby(["bank_key", "year", "quarter"], as_index=False)
              .agg(amount_brl=("v", "sum"), imputed_brl=("imp", "sum"),
                   n_months_observed=("period", "nunique"),
                   validated=("validated", "all"),
                   # `p["basis"]` already joins a period's distinct bases ("accrued/undocumented",
                   # from period_table() in the scraper); taking "first" across the quarter's
                   # months and entities threw that away, so a quarter with one accrued month and
                   # two undocumented months was labelled by whichever row pandas kept first.
                   basis=("basis", lambda s: "/".join(sorted(
                       {b for v in s.dropna() for b in str(v).split("/") if b}))),
                   n_entities_observed=("entity", "nunique")))
        g["measure"] = measure
        out.append(g)
    df = pd.concat(out, ignore_index=True)
    df = df[df["amount_brl"].notna()]
    df["imputed_brl"] = df["imputed_brl"].where(df["imputed_brl"].abs() > 0.005)
    df["panel_code"] = [code_for(xw, per, "statebank", r.bank_key, r.year, r.quarter)
                        for r in df.itertuples()]
    df["source"], df["scope"] = "statebank_own", "institution"
    df["source_entity"], df["entity_name"] = df["bank_key"], df["bank_key"].str.upper()
    df["amount_kind"] = df["basis"].fillna("undocumented")
    # BRB's conglomerate publishes three entities (bank, finance company, brokerage); a quarter
    # where fewer than that were observed (2019Q3: the finance company's listing link serves the
    # wrong gazette page, see parse_brb) is flagged here rather than left silently under-counted
    # inside a conglomerate-level sum. Other banks publish for a single entity, so this never fires
    # for them.
    brb_incomplete = (df["bank_key"] == "brb") & (df["n_entities_observed"] < 3)
    df["flags"] = (np.where(df["n_months_observed"] < 3, "incomplete_quarter;", "")
                   + np.where(brb_incomplete, "entity_missing", ""))
    df["flags"] = df["flags"].str.strip(";")
    df["detail"] = "published categories summed per measure"
    df["admin_brl"] = np.nan
    return df.drop(columns=["bank_key", "basis"])


def sponsorship_annual(xw: pd.DataFrame, per: pd.DataFrame) -> pd.DataFrame:
    """Sponsorship filed by YEAR rather than month: BNB's contract lists and Caixa's contracted
    amounts. Kept annual, never spread over quarters, and never pooled with the paid series."""
    out = []
    s = pd.read_parquet(PROC / "sponsorship_periods.parquet")
    ann = s[s["period"].astype(str).str.len() == 4].copy()
    if len(ann):
        ann["year"] = ann["period"].astype(int)
        g = (ann.groupby(["bank_key", "amount_kind", "year", "period_part"], dropna=False,
                         as_index=False).agg(amount_brl=("amount_brl", "sum")))
        g["source"] = "sponsorship"
        out.append(g)
    cx = PROC / "caixa_sponsorship_contracts.parquet"
    if cx.exists():
        c = pd.read_parquet(cx)
        ycol = "year_file" if "year_file" in c.columns else "year"
        g = (c.groupby([ycol, "program"], as_index=False)
              .agg(amount_brl=("amount_brl", "sum")))
        g = g.rename(columns={ycol: "year", "program": "period_part"})
        g["bank_key"], g["amount_kind"], g["source"] = "caixa", "contracted", "sponsorship"
        out.append(g)
    if not out:
        return pd.DataFrame()
    df = pd.concat(out, ignore_index=True)
    df["panel_code"] = [code_for(xw, per, "statebank", r.bank_key, int(r.year), 4)
                        for r in df.itertuples()]
    df["entity_name"] = df["bank_key"].str.upper()
    df["element"], df["element_kind"] = "sponsorship", "sponsorship"
    df["element_note"] = "filed by year, not by month; contracted amounts recur across years"
    df["unit"], df["geographic_scope"], df["dimensions"] = "BRL", "brazil", ""
    df["cik"] = None
    return df.rename(columns={"amount_brl": "value_brl_or_usd", "year": "fiscal_year"})


def from_sponsorship(xw: pd.DataFrame, per: pd.DataFrame) -> pd.DataFrame:
    s = pd.read_parquet(PROC / "sponsorship_periods.parquet")
    s = s[s["period"].astype(str).str.len() == 6].copy()
    s["year"] = s["period"].astype(str).str[:4].astype(int)
    s["quarter"] = ((s["period"].astype(str).str[4:6].astype(int) - 1) // 3 + 1)
    g = (s.groupby(["bank_key", "amount_kind", "year", "quarter"], as_index=False)
          .agg(amount_brl=("amount_brl", "sum"), n_months_observed=("period", "nunique")))
    g["panel_code"] = [code_for(xw, per, "statebank", r.bank_key, r.year, r.quarter)
                       for r in g.itertuples()]
    g["source"], g["measure"], g["scope"] = "sponsorship", "sponsorship", "institution"
    g["source_entity"], g["entity_name"] = g["bank_key"], g["bank_key"].str.upper()
    g["validated"], g["admin_brl"] = True, np.nan
    g["flags"] = np.where(g["n_months_observed"] < 3, "incomplete_quarter", "")
    g["detail"] = "published sponsorship"
    return g.drop(columns=["bank_key"])


def annual_panel(xw: pd.DataFrame) -> pd.DataFrame:
    a = pd.read_parquet(PROC / "sec_advertising_annual.parquet")
    a = a[a["is_annual"]].copy()

    # One named element per filer, in preference order, rather than a name pattern: the pattern
    # let through Santander's MarketingOfNonbankingFinancialProducts* (revenue) and Bradesco's
    # card-programme marketing, and it could not express that Itau's generic element IS its
    # advertising line (it equals the CVM DVA figure, reconciled at 0.91 of the three COSIF
    # accounts for 2025).
    wanted = []
    for cik, prefs in SEC_ELEMENTS.items():
        for rank, (element, kind, note) in enumerate(prefs):
            wanted.append({"cik": cik, "element": element, "element_kind": kind,
                           "element_note": note, "pref_rank": rank})
    want = pd.DataFrame(wanted)
    a["cik"] = a["cik"].astype(str)
    chosen = a.merge(want, on=["cik", "element"], how="inner")
    unknown = set(a["cik"]) - set(want["cik"])
    if unknown:
        raise AssertionError(f"SEC filers with no element table entry: {sorted(unknown)}")

    # One row per filer-year: the preferred element, and within it the undimensioned fact where
    # one exists (Itau publishes only a dimensioned fact, so it keeps that).
    chosen["dim_rank"] = (chosen["dimensions"] != "").astype(int)
    chosen = (chosen.sort_values(["cik", "fiscal_year", "pref_rank", "dim_rank"])
                    .drop_duplicates(["cik", "fiscal_year"], keep="first").copy())
    slip = chosen["scale_suspect"].fillna(False)
    ratio = (chosen["value_first_abs"] / chosen["value_latest_abs"]).replace([np.inf, -np.inf], np.nan)
    value = chosen["value_latest_abs"].where(~(slip & (ratio > 10)), chosen["value_first_abs"])
    chosen = chosen.assign(
        value_brl_or_usd=value,
        panel_code=chosen["cik"].astype(str).map(
            xw[xw["source"] == "sec"].set_index("source_id")["panel_code"].to_dict()),
        flags=np.where(slip, "scale_suspect_resolved", "")
              + np.where(chosen["restated"].fillna(False), ";restated", "")
              + np.where(chosen["sign_flip"].fillna(False), ";sign_flip", "")
              + np.where(chosen["dimensions"].ne(""), ";dimensioned", ""))
    keep = ["panel_code", "cik", "entity_name", "fiscal_year", "element", "element_kind",
            "element_note", "unit", "value_brl_or_usd", "value_first", "value_latest",
            "n_filings", "flags", "period_start", "period_end", "accession_latest", "dimensions"]
    out = chosen[keep].copy()
    out["flags"] = out["flags"].str.strip(";")
    # Nu reports group-wide (Brazil, Mexico, Colombia) in USD and the split cannot be recovered
    # from the filings: kept as filed, with the scope on the row (user decision 2026-09-20).
    out["geographic_scope"] = np.where(out["cik"] == "1691493", "group_br_mx_co", "brazil")
    out["source"] = "sec"
    return out.sort_values(["entity_name", "fiscal_year"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def overlap_ratios(panel: pd.DataFrame) -> pd.DataFrame:
    """Every non-COSIF source against the COSIF accounts, under EACH candidate account bundle.

    These ratios are the only evidence that will ever exist for what each pre-2025 source
    measures, because they can only be computed where the two series overlap, which is 2025
    onward. Stored per bank, quarter, source, COSIF level and bundle, and never applied: the
    user's decision is that sources are not spliced or rescaled.
    """
    cos = panel[panel["source"].isin(["cosif_conglomerate", "cosif_institution"])]
    wide = (cos.pivot_table(index=["panel_code", "year", "quarter", "source"], columns="measure",
                            values="amount_brl", aggfunc="first").reset_index())
    wide["adv_promo"] = wide.get("adv", np.nan) + wide.get("promo", np.nan)
    others = panel[~panel["source"].isin(["cosif_conglomerate", "cosif_institution"])
                   & panel["amount_brl"].notna() & (panel["amount_brl"] > 0)]
    rows = []
    for bundle in ("adv", "adv_promo", "all3"):
        if bundle not in wide.columns:
            continue
        ref = wide[["panel_code", "year", "quarter", "source", bundle]].rename(
            columns={"source": "cosif_level", bundle: "cosif_value"})
        j = others.merge(ref, on=["panel_code", "year", "quarter"], how="inner")
        j = j[j["cosif_value"].notna() & (j["cosif_value"] > 0)]
        j["bundle"], j["ratio"] = bundle, j["amount_brl"] / j["cosif_value"]
        rows.append(j[["panel_code", "entity_name", "year", "quarter", "source", "measure",
                       "amount_kind", "cosif_level", "bundle", "amount_brl", "cosif_value",
                       "ratio"]])
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not out.empty:
        summary = (out.groupby(["source", "measure", "cosif_level", "bundle"])["ratio"]
                      .agg(["size", "median"]).round(2))
        log.info("overlap ratios by source and bundle (median, never applied):\n%s",
                 summary.to_string())
    return out


ANCINE_COUNTS = ["n_crt", "n_crt_own", "n_crt_affiliate", "n_crt_holding", "n_crt_media_sponsored",
                 "n_crt_naming_rights_suspect", "n_crt_coop_system"]


def ancine_quarterly() -> pd.DataFrame:
    """ANCINE film registrations per conglomerate and quarter, from scrape_ancine_ad_films.py's
    monthly table. n_crt is the bank's total (own + affiliate + holding + media_sponsored), with
    the components beside it; n_crt_naming_rights_suspect is the part of n_crt whose film only
    names the bank in an event, venue or institute name, and n_crt_coop_system (Sicoob and Sicredi
    co-operatives' films, set against their co-operative banks) is not in n_crt. in_panel is False
    in a quarter with no market-panel rows for the conglomerate, so its zeros there are not
    observations; partial_quarter marks a quarter with fewer than three months of data or with the
    file's last, incomplete month."""
    m = pd.read_parquet(PROC / "ancine_ad_films_monthly.parquet")
    m["quarter"] = (m["month"] - 1) // 3 + 1
    g = (m.groupby(["panel_code", "year", "quarter"], as_index=False)
          .agg(panel_name=("panel_name", "first"), **{c: (c, "sum") for c in ANCINE_COUNTS},
               n_months=("month", "nunique"), in_panel=("in_panel", "all"),
               partial_quarter=("partial_month", "any")))
    if not g[ANCINE_COUNTS].sum().equals(m[ANCINE_COUNTS].sum()):
        raise SystemExit("ANCINE quarterly counts do not add up to the monthly table")
    g["partial_quarter"] = g["partial_quarter"] | (g["n_months"] < 3)
    log.info("ANCINE CRTs: %d conglomerates, %d quarters, n_crt %d (co-operative systems %d apart)",
             g["panel_code"].nunique(), len(g), int(g["n_crt"].sum()),
             int(g["n_crt_coop_system"].sum()))
    return g


def deposits_by_quarter() -> pd.Series:
    mp = pd.read_parquet(MARKET_PANEL, columns=["CodConglomeradoPrudencial", "year", "quarter"]
                         + DEP_COLS)
    mp["dep"] = mp[DEP_COLS].sum(axis=1, min_count=1)
    return mp.groupby(["CodConglomeradoPrudencial", "year", "quarter"])["dep"].sum()


def main() -> None:
    ap = argparse.ArgumentParser(description="Assemble the advertising panels")
    ap.add_argument("--refresh-deflator", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=PROC)
    a = ap.parse_args()

    needed = ["cosif_advertising_quarterly", "cvm_advertising_quarterly", "sec_advertising_annual",
              "caixa_advertising_monthly", "caixa_scan_advertising_monthly",
              "statebank_advertising_periods", "sponsorship_periods",
              "advertising_entity_crosswalk", "advertising_entity_crosswalk_periods",
              "ancine_ad_films_monthly"]
    missing = [n for n in needed if not (PROC / f"{n}.parquet").exists()]
    if missing:                                                              # check 1
        raise SystemExit(f"missing inputs: {missing}")

    xw, per = crosswalk()
    panel_names = panel_name_map()
    builders = {"cosif": lambda: from_cosif(xw, per, panel_names),
                "cvm": lambda: from_cvm(xw, per, panel_names),
                "caixa": lambda: from_caixa(xw, per), "statebank": lambda: from_statebanks(xw, per),
                "sponsorship": lambda: from_sponsorship(xw, per)}
    parts = []
    for name, fn in builders.items():
        df = fn()
        if df.empty:
            raise AssertionError(f"source builder {name} produced no rows")
        log.info("  %-12s %6d rows", name, len(df))
        parts.append(df)
    panel = pd.concat(parts, ignore_index=True)
    extra = brb_with_sponsorship(panel[panel["source"] == "statebank_own"])
    if not extra.empty:
        panel = pd.concat([panel, extra], ignore_index=True)
        log.info("  %-12s %6d rows (BRB sponsorship-inclusive measure)", "brb_extra", len(extra))

    dropped = panel["panel_code"].isna().sum()                               # check 2
    if dropped:
        log.warning("%d rows without a market-panel conglomerate code, dropped:\n%s", dropped,
                    panel[panel["panel_code"].isna()]
                    .groupby(["source", "source_entity"]).size().to_string())
    panel = panel[panel["panel_code"].notna()].copy()

    dup = panel.duplicated(KEY, keep=False)                                  # check 3
    if dup.any():
        raise AssertionError(f"{int(dup.sum())} duplicate rows on {KEY}:\n"
                             f"{panel[dup].head(8)[KEY + ['amount_brl']].to_string(index=False)}")

    defl, base = ipca_index(a.refresh_deflator)
    panel = panel.merge(defl[["year", "quarter", "deflator"]], on=["year", "quarter"], how="left")
    if panel["deflator"].isna().any():                                       # check 4
        gaps = panel.loc[panel["deflator"].isna(), ["year", "quarter"]].drop_duplicates()
        raise AssertionError(f"deflator missing for {gaps.to_dict('records')}")
    panel["amount_brl_real"] = panel["amount_brl"] * panel["deflator"]
    panel["deflator_base"] = base

    both = (panel[panel["source"] == "cvm"].groupby(["panel_code", "year", "quarter"])["scope"]
            .nunique())
    if (both > 1).any():                                                     # check 5
        raise AssertionError("a conglomerate-quarter mixes consolidated and individual filings: "
                             f"{both[both > 1].head().to_dict()}")

    if _CODE_CHOICE:
        log.info("conglomerate code chosen among several candidates:\n%s",
                 "\n".join(f"  {k[0]} {k[1]}: {v}" for k, v in sorted(_CODE_CHOICE.items())))

    # Explicit status per cell: one NaN cannot carry four meanings. Gaps inside a series' own
    # span become their own rows, so a reader can tell a published zero from an unobserved
    # quarter from a source that exists but could not be read.
    panel["value_status"] = np.where(panel["amount_brl"] > 0, "observed_positive",
                                    np.where(panel["amount_brl"] == 0, "observed_zero",
                                             "observed_negative"))
    # A cell that includes a line the scraper imputed from its table's own arithmetic says so,
    # with the imputed part in imputed_brl; nothing else changes status.
    if "imputed_brl" not in panel.columns:
        panel["imputed_brl"] = np.nan
    imputed = panel["imputed_brl"].notna() & (panel["imputed_brl"] != 0)
    panel.loc[imputed, "value_status"] = "imputed_from_row_total"
    log.info("cells including an imputed line: %d (%s)", int(imputed.sum()),
             panel.loc[imputed, ["source_entity", "year", "quarter", "measure", "imputed_brl"]]
                  .to_dict("records"))
    series_keys = ["panel_code", "source", "scope", "measure", "amount_kind"]
    spans = panel.groupby(series_keys, dropna=False)["year"].agg(["min", "max"]).reset_index()
    grid = []
    for r in spans.itertuples(index=False):
        for y in range(int(r[len(series_keys)]), int(r[len(series_keys) + 1]) + 1):
            for qtr in (1, 2, 3, 4):
                grid.append(tuple(r[:len(series_keys)]) + (y, qtr))
    full = pd.DataFrame(grid, columns=series_keys + ["year", "quarter"])
    panel = full.merge(panel, on=series_keys + ["year", "quarter"], how="left")
    panel["value_status"] = panel["value_status"].fillna("not_observed_in_span")
    log.info("value_status: %s", panel["value_status"].value_counts().to_dict())
    for col in ("entity_name", "source_entity", "validated"):
        panel[col] = panel.groupby(series_keys, dropna=False)[col].transform(
            lambda s: s.ffill().bfill())

    # A segment is a stretch with one source, scope, bundle and key; it increments whenever any of
    # them changes, so a later reader can see every join without recomputing it.
    panel = panel.sort_values(["panel_code", "measure", "year", "quarter", "source"]).reset_index(
        drop=True)
    seg_change = (panel[["panel_code", "measure", "source", "scope", "amount_kind"]]
                  != panel[["panel_code", "measure", "source", "scope", "amount_kind"]].shift())
    panel["segment_id"] = seg_change.any(axis=1).cumsum()

    panel = attach_registry_code(panel, xw, per)

    dep = deposits_by_quarter()
    idx = pd.MultiIndex.from_arrays([panel["panel_code"], panel["year"], panel["quarter"]])
    panel["deposits_brl"] = dep.reindex(idx).to_numpy()

    # The panel's deposit aggregate jumps implausibly in some conglomerate-quarters (Mercantil
    # C0080123 falls to R$15m in 2023Q3 from R$12.3bn; Itau C0080099 doubles in 2025Q3; several
    # halve in 2016Q1). Those cells are flagged so a ratio built on them is never used unaware.
    d = dep.rename("dep").reset_index()
    d = d.sort_values(["CodConglomeradoPrudencial", "year", "quarter"])
    prev = d.groupby("CodConglomeradoPrudencial")["dep"].shift()
    ratio = d["dep"] / prev
    d["deposits_suspect"] = ratio.gt(1.5) | ratio.lt(0.5)
    suspect = d.set_index(["CodConglomeradoPrudencial", "year", "quarter"])["deposits_suspect"]
    panel["deposits_suspect"] = suspect.reindex(idx).fillna(False).to_numpy()
    log.info("deposit denominator flagged in %d of %d rows (market-panel jumps > 50%%)",
             int(panel["deposits_suspect"].sum()), len(panel))
    panel["adv_over_deposits"] = panel["amount_brl"] / panel["deposits_brl"].where(
        panel["deposits_brl"] > 0)
    # Total assets is the headline intensity denominator: it is constant within a
    # conglomerate-quarter and free of the jumps that affect the deposit aggregate.
    mp = pd.read_parquet(MARKET_PANEL, columns=["CodConglomeradoPrudencial", "year", "quarter",
                                               "total_assets"])
    assets = (mp.dropna(subset=["total_assets"])
                .groupby(["CodConglomeradoPrudencial", "year", "quarter"])["total_assets"].first())
    panel["total_assets_brl"] = assets.reindex(idx).to_numpy()
    panel["adv_over_assets"] = panel["amount_brl"] / panel["total_assets_brl"].where(
        panel["total_assets_brl"] > 0)
    panel["adv_over_admin"] = panel["amount_brl"] / panel["admin_brl"].where(
        panel["admin_brl"].notna() & (panel["admin_brl"] > 0))

    annual = pd.concat([annual_panel(xw), sponsorship_annual(xw, per)], ignore_index=True)
    annual = annual[annual["panel_code"].notna()].copy()

    log.info("2025 reconciliation against the COSIF conglomerate accounts (printed, not gated):")
    ref = (panel[(panel["source"] == "cosif_conglomerate") & (panel["measure"] == "adv")
                 & (panel["year"] == 2025)]
           .set_index(["panel_code", "quarter"])["amount_brl"])
    for src in ("cvm", "caixa_own", "statebank_own"):
        s = panel[(panel["source"] == src) & (panel["year"] == 2025)
                  & panel["measure"].isin(["as_filed", "adv_production"])]
        if s.empty:
            continue
        r = s.set_index(["panel_code", "quarter"])["amount_brl"] / ref
        log.info("  %-14s median ratio %.2f over %d conglomerate-quarters", src,
                 float(r.dropna().median()) if r.notna().any() else float("nan"),
                 int(r.notna().sum()))

    a.out_dir.mkdir(parents=True, exist_ok=True)
    overlaps = overlap_ratios(panel)
    panel = panel.rename(columns={"flags": "flag_notes"})   # 'flags' collides with DataFrame.flags
    annual = annual.rename(columns={"flags": "flag_notes"})
    for name, frame in (("advertising_panel_quarterly", panel),
                        ("advertising_panel_annual", annual),
                        ("advertising_overlap_ratios", overlaps),
                        ("advertising_ancine_quarterly", ancine_quarterly())):
        frame.to_parquet(a.out_dir / f"{name}.parquet", index=False)
        frame.to_csv(a.out_dir / f"{name}.csv", index=False)
        log.info("%s: %d rows -> %s", name, len(frame), a.out_dir / f"{name}.parquet")

    log.info("quarterly panel rows by source and measure:\n%s",
             panel.pivot_table(index="source", columns="measure", values="amount_brl",
                               aggfunc="size", fill_value=0).to_string())
    log.info("conglomerates and span by source:\n%s",
             panel.groupby("source").agg(conglomerates=("panel_code", "nunique"),
                                         first=("year", "min"), last=("year", "max"),
                                         unvalidated=("validated", lambda s: int((s == False).sum())))
                  .to_string())


if __name__ == "__main__":
    main()

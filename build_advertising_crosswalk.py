"""
build_advertising_crosswalk.py
==============================
Crosswalk from every advertising and sponsorship source's institution identifier to the
market panel's prudential conglomerate code. Read-only with respect to the data pipeline: it
reads the IF.data lists and `market_panel.parquet` and imports no pipeline module.

Sources covered
---------------
  cosif_conglomerate  entity_key is already a prudential code; only its presence in the
                      market panel is checked
  cosif_institution   CNPJ8 from the BANCOS balancetes
  cvm                 CNPJ8 of BCB-supervised CVM filers with an advertising-type line
  sec                 SEC filer CIK, through the CNPJ8 of its Brazilian operating institution
  statebank           state-owned banks' own disclosures and sponsorship files
The SEC and state-bank CNPJ8s are entered by hand below; each must carry an expected token in
its own IF.data name, or the run aborts.

Mapping rule
------------
For a CNPJ8, the candidate codes are every CodConglomeradoPrudencial on an IF.data row whose
CodInst is that CNPJ8 or whose CnpjInstituicaoLider is that CNPJ8, across all quarters. A bank
can carry more than one code over time; the market panel uses one per bank. So:
  mapped           exactly one candidate code appears in the market panel -> panel_code
  time_varying     several panel codes in non-overlapping periods (an ownership change, for
                   example HSBC's institutions moving into Bradesco in 2016): map each quarter
                   to the code valid in that quarter, from the periods table
  ambiguous        several panel codes whose periods overlap; none is chosen
  not_in_panel     no candidate code appears in the market panel
  not_in_registry  the CNPJ8 has no IF.data row with a conglomerate code
Every candidate is listed with its first and last quarter, so nothing is hidden.

Outputs (paths.AWARENESS_PROC)
------------------------------
  advertising_entity_crosswalk.{parquet,csv}
    source, source_id, source_name, cnpj8, registry_name, candidates, panel_code,
    n_panel_codes, status, panel_name
  advertising_entity_crosswalk_periods.{parquet,csv}
    source, source_id, cnpj8, panel_code_candidate, first_quarter, last_quarter, in_panel

Validation (any failure aborts before writing)
----------------------------------------------
  1. Every hand-entered CNPJ8 has an IF.data name containing its expected token.
  2. Anchors: Banco do Brasil 00000000 -> C0080329, Caixa 00360305 -> C0080738,
     Itau Unibanco Holding 60872504 -> C0080099, Nu Pagamentos 18236120 -> C0084693.
  3. Every entity present in the advertising outputs on disk appears in the crosswalk.

Usage
-----
  python build_advertising_crosswalk.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from utils import paths

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

PROC = paths.AWARENESS_PROC
MARKET_PANEL = Path(str(paths.market_panel_csv()).replace(".csv", ".parquet"))

# SEC filer CIK -> (CNPJ8 of the Brazilian operating institution, token expected in its IF.data name)
SEC_ENTITIES = {
    "1132597": ("60872504", "ITA"),          # Itau Unibanco Holding S.A.
    "1160330": ("60746948", "BRADESCO"),     # Banco Bradesco S.A.
    "1471055": ("90400888", "SANTANDER"),    # Banco Santander (Brasil) S.A.
    "1691493": ("18236120", "NU PAGAMENTOS"),
    "1712807": ("08561701", "PAGSEGURO"),
    "1745431": ("16501555", "STONE"),
    "1787425": ("02332886", "XP INVESTIMENTOS"),
    "1864163": ("00416968", "INTER"),
}
# state-owned bank key -> (CNPJ8, token expected in its IF.data name)
STATE_BANKS = {
    "bb": ("00000000", "BRASIL"),
    "caixa": ("00360305", "CAIXA"),
    "bnb": ("07237373", "NORDESTE"),
    "basa": ("04902979", "AMAZ"),
    "brb": ("00000208", "BRB"),
    "banestes": ("28127603", "BANESTES"),
    "banrisul": ("92702067", "RIO GRANDE DO SUL"),
    "banpara": ("04913711", "PAR"),
    "banese": ("13009717", "SERGIPE"),
}
ANCHORS = {"00000000": "C0080329", "00360305": "C0080738", "60872504": "C0080099",
           "18236120": "C0084693"}


def load_registry() -> pd.DataFrame:
    files = sorted(Path(paths.IF_DATA_LIST).glob("IF_DATA_List_*.csv"))
    if not files:
        raise SystemExit(f"No IF_DATA_List_*.csv under {paths.IF_DATA_LIST}")
    cols = ["Data", "CodInst", "NomeInstituicao", "CodConglomeradoPrudencial",
            "CnpjInstituicaoLider"]
    frames = [pd.read_csv(f, sep=None, engine="python", dtype=str, encoding="utf-8",
                          encoding_errors="replace", usecols=cols) for f in files]
    reg = pd.concat(frames, ignore_index=True)
    for c in cols:
        reg[c] = reg[c].str.strip()
    log.info("IF.data lists: %d files, %d rows, %s-%s", len(files), len(reg),
             reg["Data"].min(), reg["Data"].max())
    return reg


def registry_name(reg: pd.DataFrame, cnpj8: str) -> str:
    numeric = reg["CodInst"].str.fullmatch(r"\d{1,8}", na=False)
    own = reg[numeric & (reg["CodInst"].str.zfill(8) == cnpj8)].sort_values("Data")
    return own["NomeInstituicao"].iloc[-1] if len(own) else ""


def candidates(reg: pd.DataFrame, cnpj8: str) -> pd.DataFrame:
    # Only fields that actually hold digits are compared: Banco do Brasil's CNPJ8 is 00000000,
    # so zero-filling a blank field would match it to unrelated conglomerates.
    numeric = reg["CodInst"].str.fullmatch(r"\d{1,8}", na=False)
    as_member = reg[numeric & (reg["CodInst"].str.zfill(8) == cnpj8)]
    has_leader = reg["CnpjInstituicaoLider"].str.fullmatch(r"\d{1,8}", na=False)
    as_leader = reg[has_leader & (reg["CnpjInstituicaoLider"].str.zfill(8) == cnpj8)]
    rows = pd.concat([as_member, as_leader]).dropna(subset=["CodConglomeradoPrudencial"])
    return (rows.groupby("CodConglomeradoPrudencial")["Data"].agg(["min", "max"])
                .reset_index().sort_values("min"))


PERIODS: list[dict] = []


def map_entity(reg, panel_codes, panel_names, source, source_id, source_name, cnpj8):
    cand = candidates(reg, cnpj8)
    cand["in_panel"] = cand["CodConglomeradoPrudencial"].isin(panel_codes)
    for r in cand.itertuples():
        PERIODS.append({"source": source, "source_id": source_id, "cnpj8": cnpj8,
                        "panel_code_candidate": r.CodConglomeradoPrudencial,
                        "first_quarter": r.min, "last_quarter": r.max,
                        "in_panel": bool(r.in_panel)})
    live = cand[cand["in_panel"]].sort_values("min")
    in_panel = list(live["CodConglomeradoPrudencial"])
    overlapping = bool(len(live) > 1 and (live["min"].iloc[1:].to_numpy()
                                          <= live["max"].iloc[:-1].to_numpy()).any())
    if cand.empty:
        status = "not_in_registry"
    elif len(in_panel) == 1:
        status = "mapped"
    elif not in_panel:
        status = "not_in_panel"
    elif overlapping:
        status = "ambiguous"
    else:
        status = "time_varying"
    code = in_panel[0] if status == "mapped" else None
    return {"source": source, "source_id": source_id, "source_name": source_name,
            "cnpj8": cnpj8, "registry_name": registry_name(reg, cnpj8),
            "candidates": "; ".join(f"{r.CodConglomeradoPrudencial} ({r.min}-{r.max})"
                                    for r in cand.itertuples()),
            "panel_code": code, "n_panel_codes": len(in_panel), "status": status,
            "panel_name": panel_names.get(code, "") if code else ""}


def main() -> None:
    reg = load_registry()
    panel = pd.read_parquet(MARKET_PANEL, columns=["CodConglomeradoPrudencial", "NomeInstituicao"])
    panel_codes = set(panel["CodConglomeradoPrudencial"].astype(str))
    panel_names = (panel.drop_duplicates("CodConglomeradoPrudencial")
                        .set_index("CodConglomeradoPrudencial")["NomeInstituicao"].to_dict())
    log.info("market panel: %d conglomerate codes", len(panel_codes))

    for label, table in (("SEC", SEC_ENTITIES), ("state bank", STATE_BANKS)):  # check 1
        for key, (cnpj8, token) in table.items():
            name = registry_name(reg, cnpj8)
            if token not in name.upper():
                raise AssertionError(f"{label} {key}: CNPJ8 {cnpj8} has IF.data name {name!r}, "
                                     f"expected to contain {token!r}")

    rows = []
    cq = pd.read_parquet(PROC / "cosif_advertising_quarterly.parquet")
    for r in cq[cq["level"] == "conglomerate"].drop_duplicates("entity_key").itertuples():
        rows.append({"source": "cosif_conglomerate", "source_id": r.entity_key,
                     "source_name": r.entity_name, "cnpj8": str(r.cnpj_leader).zfill(8),
                     "registry_name": "", "candidates": r.entity_key,
                     "panel_code": r.entity_key if r.entity_key in panel_codes else None,
                     "n_panel_codes": int(r.entity_key in panel_codes),
                     "status": "mapped" if r.entity_key in panel_codes else "not_in_panel",
                     "panel_name": panel_names.get(r.entity_key, "")})
    for r in cq[cq["level"] == "institution"].drop_duplicates("entity_key").itertuples():
        rows.append(map_entity(reg, panel_codes, panel_names, "cosif_institution",
                               r.entity_key, r.entity_name, r.entity_key))

    cvm = pd.read_parquet(PROC / "cvm_advertising_quarterly.parquet")
    for r in cvm[cvm["bcb_supervised"]].drop_duplicates("CD_CVM").itertuples():
        rows.append(map_entity(reg, panel_codes, panel_names, "cvm", r.CD_CVM, r.DENOM_CIA,
                               str(r.cnpj8).zfill(8)))

    sec = pd.read_parquet(PROC / "sec_advertising_annual.parquet")
    for cik, name in sec.drop_duplicates("cik")[["cik", "entity_name"]].itertuples(index=False):
        cnpj8 = SEC_ENTITIES[str(cik)][0]
        rows.append(map_entity(reg, panel_codes, panel_names, "sec", str(cik), name, cnpj8))

    for key, (cnpj8, _) in STATE_BANKS.items():
        rows.append(map_entity(reg, panel_codes, panel_names, "statebank", key,
                               registry_name(reg, cnpj8), cnpj8))

    xw = pd.DataFrame(rows)

    for cnpj8, code in ANCHORS.items():                                             # check 2
        got = set(xw.loc[(xw["cnpj8"] == cnpj8) & (xw["source"] != "cosif_conglomerate"),
                         "panel_code"].dropna())
        if got != {code}:
            raise AssertionError(f"anchor {cnpj8}: expected {code}, got {sorted(got)}")

    expected = {("cosif_conglomerate", k) for k in cq.loc[cq["level"] == "conglomerate", "entity_key"]}
    expected |= {("cosif_institution", k) for k in cq.loc[cq["level"] == "institution", "entity_key"]}
    expected |= {("cvm", k) for k in cvm.loc[cvm["bcb_supervised"], "CD_CVM"]}
    expected |= {("sec", str(k)) for k in sec["cik"]}
    present = set(zip(xw["source"], xw["source_id"]))
    if expected - present:                                                          # check 3
        raise AssertionError(f"entities missing from the crosswalk: {sorted(expected - present)[:10]}")

    PROC.mkdir(parents=True, exist_ok=True)
    xw.to_parquet(PROC / "advertising_entity_crosswalk.parquet", index=False)
    xw.to_csv(PROC / "advertising_entity_crosswalk.csv", index=False)
    periods = pd.DataFrame(PERIODS).drop_duplicates()
    periods.to_parquet(PROC / "advertising_entity_crosswalk_periods.parquet", index=False)
    periods.to_csv(PROC / "advertising_entity_crosswalk_periods.csv", index=False)
    log.info("crosswalk: %d entities, %d dated code periods -> %s", len(xw), len(periods), PROC)
    log.info("status by source:\n%s", xw.groupby(["source", "status"]).size().unstack(fill_value=0)
             .to_string())
    show = xw[xw["source"].isin(["cvm", "sec", "statebank"]) | (xw["status"] == "ambiguous")]
    log.info("named sources and ambiguous rows:\n%s",
             show[["source", "source_id", "source_name", "cnpj8", "panel_code", "status",
                   "candidates"]].to_string(index=False))


if __name__ == "__main__":
    main()

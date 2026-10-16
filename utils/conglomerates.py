"""utils/conglomerates.py — CNPJ / conglomerate identity, shared by the panel scripts.

Canonical identity (one code per firm over its whole life; see build_conglomerate_canonical_map):
  build_conglomerate_canonical_map()  -> (canonical_code, canonical_lider)
  build_cnpj_conglomerate_map()       -> {cnpj_int: (canonical code, leader CNPJ, normalised name)}
As-reported, time-varying identity (the code a CNPJ held at each List snapshot):
  build_cnpj_to_congl()               -> CNPJ x snapshot -> CodConglomeradoPrudencial
  map_foundation_to_congl(found, map) -> as-of merge of a CNPJ x quarter frame onto that map
Helpers: normalize_str, coerce_cnpj.

Moved verbatim from panel_deposits.py and panel_cosif_calibrate.py (2026-10-15) so that no
script imports a pipeline step to borrow a helper. Readers: panel_deposits, panel_market,
panel_cosif_calibrate, panel_demand_chars, panel_estban_instrument, panel_local_cdb_spread,
fees/panel_fee_merge, scrape_bcb_complaints, utils/app_registry, diagnostics.
"""
from __future__ import annotations

import glob
import logging
import os
import string
import unicodedata

import pandas as pd

from utils import paths

IF_LIST_DIR = str(paths.IF_DATA_LIST)


# String normalization
def normalize_str(s):
    """Remove accents, punctuation, and non-ASCII characters from a string."""
    if pd.isna(s):
        return s
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = "".join(c for c in s if ord(c) < 128)
    return s.strip().upper()


def coerce_cnpj(series):
    """Convert a series to nullable Int64 CNPJ root (8-digit integer)."""
    return pd.to_numeric(series, errors="coerce").astype("Int64")

# CNPJ -> conglomerate maps (canonical identity)

_IF_LIST_SNAPSHOTS: "pd.DataFrame | None" = None


def _load_if_list_snapshots() -> pd.DataFrame:
    """Concatenate every IF Data List snapshot into one frame (cached for the run).

    Columns: CodInst, Data, NomeInstituicao, CodConglomeradoPrudencial,
    CnpjInstituicaoLider.  ``Data`` is the snapshot stamp (YYYYMM); it is what lets a
    caller order a conglomerate's spells.  ``build_cnpj_conglomerate_map`` drops it
    again and de-duplicates on the other four columns.
    """
    global _IF_LIST_SNAPSHOTS
    if _IF_LIST_SNAPSHOTS is not None:
        return _IF_LIST_SNAPSHOTS

    list_files = sorted(glob.glob(os.path.join(IF_LIST_DIR, "IF_DATA_List_*.csv")))
    # Skip headers-only stubs (< 500 bytes = no data rows)
    list_files = [f for f in list_files if os.path.getsize(f) > 500]
    if not list_files:
        raise FileNotFoundError(f"No IF Data List files found in {IF_LIST_DIR}")

    frames = []
    for path in list_files:
        try:
            df = pd.read_csv(path, encoding="latin1", low_memory=False,
                             usecols=["CodInst", "Data", "NomeInstituicao",
                                      "CodConglomeradoPrudencial", "CnpjInstituicaoLider"])
            frames.append(df)
        except Exception as e:
            logging.warning(f"Could not read List file {os.path.basename(path)}: {e}")

    _IF_LIST_SNAPSHOTS = pd.concat(frames, ignore_index=True)
    return _IF_LIST_SNAPSHOTS


# A prudential code only starts carrying reported balances -- IF Data report 3, and
# bank_chars_panel.csv, which panel_bank_chars.py builds from the same source -- in 2016.  Spell
# length is measured from this snapshot on, so a firm's identity is the code it holds
# over the data the rest of the pipeline can actually price, not over List history no
# downstream table covers.  Banco PSA Finance/Stellantis is the case that turns on it:
# C0080594 spans more List snapshots outright (10 vs 9) purely on its 2014-15 rows,
# while every reported balance the firm has sits under C0087298.
IDENTITY_SPELL_START = 201603

_CANONICAL_CONGLOMERATE_MAP: "tuple | None" = None


def build_conglomerate_canonical_map() -> tuple:
    """
    Return ``(canonical_code, canonical_lider)``.

    ``canonical_code``  maps every CodConglomeradoPrudencial appearing in the IF Data
                        List files to the ONE code that stands for that firm over its
                        whole life.  A code needing no collapse maps to itself.
    ``canonical_lider`` maps a canonical code to the CNPJ of its lead institution.

    Why this exists
    ---------------
    BCB issues a new prudential code whenever a conglomerate's composition or lead
    institution changes, so one bank carries several codes over the sample: Banco
    Gerador C0081809 becomes Agibank C0083694 at the 2016-Q3 acquisition, Banco
    Indusval C0080415 becomes Banco Pleno C0088668 in 2025-Q3, Banco BBM C0080161
    becomes Bocom BBM C0084624 in 2017.  The two tiers of ``deposits_panel`` pick the
    code up at different moments -- ESTBAN once, through the static CNPJ map built
    below; IF Data quarter by quarter, as reported -- so one bank can end up filed
    under two codes at the same time.  The ESTBAN/IF Data de-duplication in ``panel_deposits.main()``
    compares CODE to CODE, so it cannot see that collision and the bank's deposits are
    counted twice: Agibank's 2025 ``dep_a4`` summed to R$162.6bn against a true
    R$81.3bn, and Industval was simultaneously a national firm and a local bank in 12
    markets.  Mapping both tiers onto one code per firm BEFORE that comparison is what
    lets the de-duplication do its job.  It also holds a firm's identity together
    across codes that merely succeed one another with no overlap at all, which is what
    the entry-dynamics and BBL steps downstream read as entry and exit.

    How a firm is identified
    ------------------------
    By the CNPJ of its lead institution (``CnpjInstituicaoLider``), read ONLY off rows
    whose ``CodInst`` is a numeric CNPJ, i.e. individual member institutions.  A List
    file also carries rows whose ``CodInst`` is a *financial* conglomerate code
    (C005xxxx); on those the leader field names the financial group's leader, a
    different firm, and using them would chain unrelated prudential codes together --
    it is why CNPJ 61723847 looks like the leader of both Magliano C0082839 and Neon
    C0085702.  Each code then takes the leader of its LAST snapshot, which makes
    code -> firm a partition rather than a graph: Banco Original C0080903 is not
    swallowed by PicPay C0088022 merely because PicPay's payment institution led it
    during 2023-24.

    Which of a firm's codes wins
    ----------------------------
    The code the firm occupies in the most List snapshots from ``IDENTITY_SPELL_START``
    on -- its dominant spell over the reported period -- ties broken by the later
    spell, then alphabetically, so the pick is deterministic.  Dominant spell rather
    than newest code because the code is also the merge key against the sources that
    describe the firm: bank_chars_panel.csv (equity, total_assets, segment, is_coop,
    is_state_owned, and through them the LOO instruments) and the COSIF rates panel_3
    appends.  Picking a code those files barely cover would blank a firm's
    characteristics for most of its life without leaving a hole anyone would notice.
    Measured over the nine firms this collapses, the rule selects the better-covered
    code in bank_chars_panel.csv in all nine.

    What this does NOT decide
    -------------------------
    Whether a firm ends up national (Tier 2) or local (Tier 1).  That is the tier rule
    in ``panel_deposits.main()``, and it returns the same answer whichever code is canonical, because
    after canonicalisation both tiers carry that one code either way.
    """
    global _CANONICAL_CONGLOMERATE_MAP
    if _CANONICAL_CONGLOMERATE_MAP is not None:
        return _CANONICAL_CONGLOMERATE_MAP

    lf = _load_if_list_snapshots()
    lf = lf.dropna(subset=["CodConglomeradoPrudencial"]).copy()
    lf["Data"] = pd.to_numeric(lf["Data"], errors="coerce")
    lf = lf.dropna(subset=["Data"])
    lf["lider_int"]   = coerce_cnpj(lf["CnpjInstituicaoLider"])
    lf["codinst_int"] = coerce_cnpj(lf["CodInst"])

    # Leader of each code, from member-institution rows in the code's last snapshot.
    # Modal leader, then highest CNPJ, so the pick never depends on row order.
    inst = lf[lf["codinst_int"].notna() & lf["lider_int"].notna()]
    tail = inst[inst["Data"] == inst.groupby("CodConglomeradoPrudencial")["Data"].transform("max")]
    lider_pick = (
        tail.groupby(["CodConglomeradoPrudencial", "lider_int"]).size().rename("n").reset_index()
            .sort_values(["CodConglomeradoPrudencial", "n", "lider_int"],
                         ascending=[True, False, False])
            .drop_duplicates(subset=["CodConglomeradoPrudencial"])
    )
    code_lider = {row.CodConglomeradoPrudencial: int(row.lider_int)
                  for row in lider_pick.itertuples(index=False)}

    spells = (lf.groupby("CodConglomeradoPrudencial")["Data"]
                .agg(n_all="nunique", last_snapshot="max"))
    n_reported = (lf[lf["Data"] >= IDENTITY_SPELL_START]
                    .groupby("CodConglomeradoPrudencial")["Data"].nunique()
                    .rename("n_reported"))
    spells = spells.join(n_reported, how="left").reset_index()
    # A code seen only before IDENTITY_SPELL_START keeps its full-history count, so a
    # firm whose whole life predates the reported period still resolves to something.
    spells["n_spell"] = spells["n_reported"].fillna(0)
    spells.loc[spells["n_spell"] == 0, "n_spell"] = spells["n_all"]
    spells["lider"] = spells["CodConglomeradoPrudencial"].map(code_lider)

    canonical_code  = {c: c for c in spells["CodConglomeradoPrudencial"]}
    canonical_lider = dict(code_lider)

    n_collapsed = 0
    for lider, grp in spells.dropna(subset=["lider"]).groupby("lider"):
        if len(grp) < 2:
            continue
        grp = grp.sort_values(["n_spell", "last_snapshot", "CodConglomeradoPrudencial"],
                              ascending=[False, False, True])
        keep   = grp["CodConglomeradoPrudencial"].iloc[0]
        merged = list(grp["CodConglomeradoPrudencial"].iloc[1:])
        for code in merged:
            canonical_code[code] = keep
            canonical_lider.pop(code, None)
        n_collapsed += len(merged)
        spans = ", ".join(
            f"{row.CodConglomeradoPrudencial} ({int(row.n_spell)} snapshots from "
            f"{IDENTITY_SPELL_START}, last {int(row.last_snapshot)})"
            for row in grp.itertuples(index=False)
        )
        logging.info(f"Conglomerate identity: leader {int(lider)} -> {keep}   [{spans}]")

    logging.info(f"Conglomerate canonical map: {len(canonical_code)} codes, "
                 f"{n_collapsed} collapsed onto another code of the same firm")

    _CANONICAL_CONGLOMERATE_MAP = (canonical_code, canonical_lider)
    return _CANONICAL_CONGLOMERATE_MAP


def canonicalize_codes(codes: pd.Series) -> pd.Series:
    """Map as-reported CodConglomeradoPrudencial codes onto each firm's canonical code
    (``build_conglomerate_canonical_map``), the key deposits_panel.csv and market_panel.csv
    carry.  A code the map does not know (a standalone's own CodInst) is kept as is.  Apply
    it to a CNPJ -> code map BEFORE aggregating member institutions, so a firm's quarters
    filed under a superseded code still merge onto the panel."""
    canonical_code, _ = build_conglomerate_canonical_map()
    return codes.map(canonical_code).fillna(codes)


_ESTBAN_CNPJS: "set | None" = None


def estban_bank_cnpjs() -> set:
    """8-digit CNPJ roots (zero-padded strings) of every bank in ESTBAN.csv (cached)."""
    global _ESTBAN_CNPJS
    if _ESTBAN_CNPJS is None:
        c = pd.read_csv(paths.ESTBAN_CSV, usecols=["CNPJ"], dtype=str, encoding="latin-1")["CNPJ"]
        _ESTBAN_CNPJS = set(c.dropna().str.strip().str.zfill(8).unique())
    return _ESTBAN_CNPJS


def deposit_panel_codes(cnpj: pd.Series, reported_code: pd.Series) -> pd.Series:
    """The code panel_deposits files each CNPJ's deposits under, for keying COSIF rows.

    An ESTBAN bank's deposits (Tier 1) carry its ``build_cnpj_conglomerate_map`` code in
    EVERY quarter, or ``'CNPJ_<cnpj>'`` when the map lacks it, so its COSIF rows must carry
    the same code.  The quarter-by-quarter reported code would follow the bank into an
    acquirer's conglomerate (Banco Pan from 2021, C6 Consignado from 2020) and leave the
    bank's own deposits without a rate.  Every other institution reaches the panel through
    IF Data as reported (Tier 2), so it keeps its reported code, canonicalised.
    ``cnpj`` holds 8-digit CNPJ roots as strings."""
    static = build_cnpj_conglomerate_map()
    estban = estban_bank_cnpjs()
    lookup = {}
    for c in cnpj.dropna().unique():
        if c in estban:
            k = int(c)
            lookup[c] = static[k][0] if k in static and static[k][0] is not None else f"CNPJ_{k}"
    st = cnpj.map(lookup)
    return st.fillna(canonicalize_codes(reported_code))


def build_cnpj_conglomerate_map() -> dict:
    """
    Return {cnpj_int: (CodConglomeradoPrudencial, CNPJ_Lider_int, NomeInstituicao)}
    by scanning IF Data List files (all quarters).

    Two matching strategies:
      1. CnpjInstituicaoLider  →  conglomerate  (always available; maps the leader's CNPJ)
      2. numeric CodInst       →  conglomerate  (when an individual institution's CodInst
                                                 equals its own CNPJ, i.e. it is the leader)
    Strategy 1 is applied after strategy 2 so leaders take priority.

    The code returned is the CANONICAL one from ``build_conglomerate_canonical_map``
    and the leader is that code's leader, so a bank keeps one identity for its whole
    history and carries the same code IF Data reports for it.  Which of a firm's codes
    the two strategies happen to land on is therefore immaterial -- they all resolve to
    the same canonical code.
    """
    _snap = _load_if_list_snapshots()
    _name_cols = ["CodInst", "NomeInstituicao", "CodConglomeradoPrudencial",
                  "CnpjInstituicaoLider"] + (["Data"] if "Data" in _snap.columns else [])
    lf = _snap[_name_cols].drop_duplicates()

    # An institution's own display name is the one on ITS OWN List row -- the row whose
    # CodInst is that CNPJ. Reading a name off any other row of the same conglomerate names
    # a different company: the Original conglomerate contained PicPay until 2024, Bradesco
    # contains Agora Corretora, Banco PAN contains an entity called "SS".
    _own = lf.copy()
    _own["codinst_int"] = coerce_cnpj(_own["CodInst"])
    _own = _own.dropna(subset=["codinst_int"])
    if "Data" in _own.columns:
        _own = _own.sort_values("Data")          # latest name a firm reported for itself
    own_name = (_own.drop_duplicates(subset=["codinst_int"], keep="last")
                    .set_index(_own.drop_duplicates(subset=["codinst_int"], keep="last")
                               ["codinst_int"].astype(int))["NomeInstituicao"])

    # Keep only rows that have a prudential conglomerate code
    lf = lf.dropna(subset=["CodConglomeradoPrudencial"])
    lf["lider_int"] = coerce_cnpj(lf["CnpjInstituicaoLider"])

    mapping: dict = {}

    # Strategy 2: numeric CodInst entries (individual institution CNPJs)
    lf["codinst_int"] = coerce_cnpj(lf["CodInst"])
    strat2 = lf.dropna(subset=["codinst_int"]).copy()
    strat2["cnpj_k"] = strat2["codinst_int"].astype(int)
    # strat2 is keyed on CodInst, so its NomeInstituicao is already the institution's own.
    strat2["norm_name"]   = strat2["NomeInstituicao"].map(normalize_str)
    strat2["lider_clean"]  = strat2["lider_int"].where(strat2["lider_int"].notna()).map(
        lambda v: int(v) if pd.notna(v) else None
    )
    for row in strat2[["cnpj_k", "CodConglomeradoPrudencial", "lider_clean", "norm_name"]].itertuples(index=False):
        mapping[row.cnpj_k] = (row.CodConglomeradoPrudencial, row.lider_clean, row.norm_name)

    # Strategy 1: CnpjInstituicaoLider (overrides – leaders take precedence)
    strat1 = lf.dropna(subset=["lider_int"]).copy()
    # One row per leader. The code it carries is replaced by the firm's canonical code
    # below, so the pick has no bearing on identity -- it only has to be reproducible.
    name_map = (strat1.sort_values("NomeInstituicao")
                      .drop_duplicates(subset=["lider_int"], keep="last")).copy()
    name_map["cnpj_k"] = name_map["lider_int"].astype(int)
    # Display name: the leader's OWN List row, falling back to this row only when the
    # leader never appears as an institution in its own right.
    name_map["display"] = (name_map["cnpj_k"].map(own_name)
                                             .fillna(name_map["NomeInstituicao"]))
    name_map["norm_name"] = name_map["display"].map(normalize_str)
    for row in name_map[["cnpj_k", "CodConglomeradoPrudencial", "norm_name"]].itertuples(index=False):
        mapping[row.cnpj_k] = (row.CodConglomeradoPrudencial, row.cnpj_k, row.norm_name)

    # Resolve every bank onto its firm's canonical code and that code's leader, so the
    # ESTBAN side of the panel names a firm exactly the way the IF Data side does.
    canonical_code, canonical_lider = build_conglomerate_canonical_map()
    mapping = {
        cnpj: (canonical_code.get(code, code),
               canonical_lider.get(canonical_code.get(code, code), lider),
               name)
        for cnpj, (code, lider, name) in mapping.items()
    }

    logging.info(f"CNPJ→conglomerate map: {len(mapping)} entries")
    return mapping


# As-reported (time-varying) identity

def build_cnpj_to_congl():
    """CNPJ(=CodInst for Td=='I') -> CodConglomeradoPrudencial, time-varying."""
    files = sorted(glob.glob(os.path.join(IF_LIST_DIR, "IF_DATA_List_*.csv")))
    if not files:
        raise FileNotFoundError(f"No IF-Data List files in {IF_LIST_DIR}")
    frames = []
    for fp in files:
        try:
            df = pd.read_csv(fp, dtype=str, encoding="latin-1")
        except Exception:
            df = pd.read_csv(fp, dtype=str, encoding="utf-8")
        df.columns = [c.strip() for c in df.columns]
        mask = (df["Td"] == "I") & (df["Situacao"] == "A")
        sub = df.loc[mask, ["CodInst", "Data", "CodConglomeradoPrudencial"]].copy()
        standalone = (sub["CodConglomeradoPrudencial"].isna()
                      | (sub["CodConglomeradoPrudencial"] == "null"))
        sub.loc[standalone, "CodConglomeradoPrudencial"] = sub.loc[standalone, "CodInst"]
        sub.rename(columns={"CodInst": "CNPJ", "Data": "AnoMes_list"}, inplace=True)
        frames.append(sub)
    m = pd.concat(frames, ignore_index=True)
    m["AnoMes_list"] = pd.to_numeric(m["AnoMes_list"], errors="coerce").astype("Int64")
    m["CNPJ"] = m["CNPJ"].str.strip().str.zfill(8)
    return m.drop_duplicates(subset=["CNPJ", "AnoMes_list"])


def map_foundation_to_congl(found, df_map):
    """As-of merge: each (CNPJ, quarter) -> nearest List snapshot (backward,
    then earliest-forward for pre-2016 quarters)."""
    f = found.sort_values(["AnoMes", "CNPJ"]).copy()
    dm = df_map.dropna(subset=["AnoMes_list"]).copy()
    dm["AnoMes_list"] = dm["AnoMes_list"].astype(int)
    dm = dm.sort_values(["AnoMes_list", "CNPJ"])
    back = pd.merge_asof(f, dm, left_on="AnoMes", right_on="AnoMes_list",
                         by="CNPJ", direction="backward")
    fwd = pd.merge_asof(f, dm, left_on="AnoMes", right_on="AnoMes_list",
                        by="CNPJ", direction="forward")
    merged = back.copy()
    fill = merged["CodConglomeradoPrudencial"].isna()
    merged.loc[fill, "CodConglomeradoPrudencial"] = fwd.loc[fill, "CodConglomeradoPrudencial"]
    return merged.dropna(subset=["CodConglomeradoPrudencial"])

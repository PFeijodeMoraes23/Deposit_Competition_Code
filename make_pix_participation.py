"""
make_pix_participation.py
=========================
Pix participation by panel conglomerate and quarter, 2020Q4-2024Q4, with the launch-mandate flag
(task R1, for the bank-pair Pix design of the switching cost). A side file: nothing is estimated,
no paper-number key reads it, and it is not merged into market_panel.

Inputs (read only)
  shared/PIX/pix_roster_history.csv           roster rows by roster date (scrape_pix_roster.py)
  <PIX_RAW>/roster_cache/*.pdf                the cached roster PDFs, re-read for the operating phase
                                              of each row, which the history leaves blank from 2021-06
  shared/PIX/pix_participant_keys_panel.csv   month-end Pix-key stock by ISPB (scrape_pix_participants.py)
  shared/PIX/pix_participant_entry.csv        key-based entry month by ISPB
  shared/PIX/pix_mandatory_institutions.csv   institutions ever flagged mandatory, with the first date
  DATA_RAW/BCB/pix_participants/chaves/chavespix_from*.csv   the newer key file, a cross-check only
  DATA_RAW/BCB/pix_participants/compiled/ispb_conglomerate_map.csv   ISPB -> conglomerate (P1)
  processed/market_panel.parquet              conglomerates, is_B, type-2 (savings) balances

Rules
  1. Roster leg. At a quarter's end the roster read is the latest one dated on or before it. An
     institution is on that roster if its row there is in an operating phase: CONCLUIDO (the 2020
     lists, adhesion completed), PLENA or RESTRITA (full or restricted operation). Rows in the
     joining section (HOMOLOGATORIA, AGUARDANDO inicio) are not on Pix. A CNPJ listed again in the
     initiation or cash-withdrawal sub-lists takes the phase of its main-list row.
  2. Key leg. An ISPB holds Pix keys at a quarter's end if its key stock is positive in the
     quarter's last month.
  3. Institution -> conglomerate. The ISPB crosswalk of the Pix-keys task (P1); a roster
     institution without an ISPB in it maps by its CNPJ root through
     utils.conglomerates.build_cnpj_conglomerate_map(), the lookup behind that crosswalk. PicPay
     (ISPB 22896431) maps to C0088022 in every quarter, where the panel holds all of PicPay's
     balances (utils/firm_remaps.py); the script stops if the crosswalk says otherwise.
  4. on_pix = 1 if any member institution is on the roster or holds keys at the quarter's end;
     source says which leg (roster, keys, both). first_on_pix is the first quarter with on_pix = 1.
  5. mandated = 1 if a member institution is one of the 34 in pix_mandatory_institutions.csv whose
     first mandatory roster is dated by the launch (2020-11-16), as in make_pix_exposure.py.
  6. Rows: every conglomerate-quarter present in the panel, 2020Q4-2024Q4.

Outputs (Rout)
  pix_participation_conglomerate_quarter.csv
      conglomerate, year, quarter, name, is_B, on_pix, first_on_pix, mandated, source,
      on_roster, on_keys, roster_date (the roster read), n_pix_members (member institutions ever in
      either leg), n_members_on, members_off (ISPBs of members ever on Pix but off this quarter),
      dep_a{1,2,4,5}_brl_mn and dep_total_brl_mn (balances by deposit type summed over markets,
      R$ million; type 2 is savings)
  pix_participation_r1.json   the report tables (roster coverage, agreement of the legs,
                              control group, mandated and voluntary joiners, member disagreements,
                              conglomerates on Pix only through a roster dated before the quarter)

Usage:  python make_pix_participation.py
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import datetime
import json
import re

import numpy as np
import pandas as pd
import pdfplumber

from utils import paths as _paths
from utils import raw_paths
from utils import firm_remaps
from utils.conglomerates import build_cnpj_conglomerate_map, build_cnpj_to_congl

PIX = _paths.PIX_DIR
ROSTER = PIX / "pix_roster_history.csv"
CACHE = _paths.PIX_RAW / "roster_cache"
KEYS = PIX / "pix_participant_keys_panel.csv"
ENTRY = PIX / "pix_participant_entry.csv"
MANDATORY = PIX / "pix_mandatory_institutions.csv"
KEYS_RAW_DIR = raw_paths.BCB_PIX_KEYS
XWALK = raw_paths.BCB_PIX_PARTICIPANTS / "compiled" / "ispb_conglomerate_map.csv"
PANEL = _paths.PROCESSED / "market_panel.parquet"
OUT = _paths.rout_dir() / "pix_participation_conglomerate_quarter.csv"
SIDE = _paths.rout_dir() / "pix_participation_r1.json"

LAUNCH = pd.Timestamp("2020-11-16")
FIRST_Q, LAST_Q = (2020, 4), (2024, 4)
FIRST_ROLLING = pd.Timestamp("2020-10-22")     # the first list of participants, not of applicants
PICPAY_ISPB = "22896431"
DEP_TYPES = (1, 2, 4, 5)                       # demand, savings, time, prepaid
ON_PHASES = {"concluido", "plena", "restrita"}
CNPJ_RE = re.compile(r"\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}")
COVER_RE = re.compile(r"(?:Atualizado em|Publicado em:?)\s*(\d{2})/(\d{2})/(\d{4})")


def quarters() -> pd.DataFrame:
    q = pd.period_range(pd.Period(f"{FIRST_Q[0]}Q{FIRST_Q[1]}", "Q"),
                        pd.Period(f"{LAST_Q[0]}Q{LAST_Q[1]}", "Q"), freq="Q")
    return pd.DataFrame({"period": q, "year": q.year, "quarter": q.quarter,
                         "qend": q.end_time.normalize(), "anomes": q.year * 100 + q.quarter * 3})


def qlabel(ts) -> str | None:
    return None if pd.isna(ts) else str(pd.Timestamp(ts).to_period("Q"))


# ── roster leg ───────────────────────────────────────────────────────────────
def _phase(text: str) -> str | None:
    u = text.upper()
    for token, ph in (("HOMOLOG", "homologacao"), ("AGUARD", "aguardando"), ("PLENA", "plena"),
                      ("RESTRIT", "restrita"), ("CONCLU", "concluido"), ("ANDAMENTO", "andamento")):
        if token in u:
            return ph
    return None


def roster_phases(dates: set) -> tuple[pd.DataFrame, dict]:
    """(roster_date, cnpj, phase) from the cached PDFs whose cover date is in `dates`; and, per
    date, the CNPJs found in the page text and in the tables (the parse-gap check)."""
    recs, check = [], {}
    for f in sorted(CACHE.glob("*.pdf")):
        with pdfplumber.open(f) as pdf:
            m = COVER_RE.search(pdf.pages[0].extract_text() or "")
            if not m:
                continue
            d = pd.Timestamp(f"{m.group(3)}-{m.group(2)}-{m.group(1)}")
            if d not in dates or d in check:
                continue
            text_cnpj, rows = set(), []
            for page in pdf.pages:
                text_cnpj |= set(CNPJ_RE.findall(page.extract_text() or ""))
                for tbl in page.extract_tables() or []:
                    for cells in tbl:
                        joined = " ".join(str(c) for c in cells if c)
                        c = CNPJ_RE.search(joined)
                        if c:
                            rows.append((c.group(0), _phase(joined)))
        df = pd.DataFrame(rows, columns=["cnpj", "phase"])
        # A CNPJ listed again in a sub-list (initiators, cash-withdrawal facilitators) carries no
        # phase there: keep the first row that has one, in document order (the main list first).
        df["has"] = df["phase"].notna()
        df = df.sort_values("has", ascending=False, kind="stable").drop_duplicates("cnpj")
        recs.append(df.drop(columns="has").assign(roster_date=d))
        check[d] = {"file": f.name, "cnpj_text": len(text_cnpj), "cnpj_tables": int(df["cnpj"].nunique()),
                    "phases": {str(k): int(v) for k, v in df["phase"].value_counts(dropna=False).items()}}
    return pd.concat(recs, ignore_index=True), check


def load_roster(cmap: dict, xw: pd.Series) -> tuple[pd.DataFrame, dict]:
    r = pd.read_csv(ROSTER, dtype=str)
    r["roster_date"] = pd.to_datetime(r["roster_date"])
    last_end = quarters()["qend"].max()
    window = sorted(d for d in r["roster_date"].unique() if FIRST_ROLLING <= d <= last_end)
    r = r[r["roster_date"].isin(window)].copy()
    ph, check = roster_phases(set(window))
    missing = set(window) - set(check)
    if missing:
        raise SystemExit(f"no cached PDF for roster dates {sorted(str(d.date()) for d in missing)}")
    for d in window:
        hist = set(r.loc[r["roster_date"] == d, "cnpj"].dropna())
        pdfc = set(ph.loc[ph["roster_date"] == d, "cnpj"])
        check[d].update(history_rows=len(hist), history_not_in_pdf=len(hist - pdfc),
                        pdf_not_in_history=len(pdfc - hist))
    r = r.merge(ph, on=["roster_date", "cnpj"], how="left")
    r["active"] = r["phase"].isin(ON_PHASES)
    r["ispb"] = r["ispb_filled"].fillna(r["cnpj8"])
    r["conglomerate"] = r["ispb"].map(xw)
    miss = r["conglomerate"].isna()
    r.loc[miss, "conglomerate"] = r.loc[miss, "cnpj8"].map(cmap)
    return r, check


# ── key leg ──────────────────────────────────────────────────────────────────
def load_keys(xw: pd.Series, cmap: dict) -> tuple[pd.DataFrame, dict]:
    k = pd.read_csv(KEYS, dtype={"ISPB": str})
    k["ISPB"] = k["ISPB"].str.zfill(8)
    k = k[k["qtd_chaves"] > 0]
    k["conglomerate"] = k["ISPB"].map(xw)
    miss = k["conglomerate"].isna()
    k.loc[miss, "conglomerate"] = k.loc[miss, "ISPB"].map(cmap)
    # Cross-check against the newer raw download at the quarter-end months both files cover.
    files = sorted(KEYS_RAW_DIR.glob("chavespix_from*.csv"))
    raw = pd.read_csv(files[-1], dtype={"ISPB": str}, usecols=["Data", "ISPB", "qtdChaves"])
    raw["anomes"] = pd.to_datetime(raw["Data"]).dt.strftime("%Y%m").astype(int)
    raw = raw.groupby(["anomes", "ISPB"])["qtdChaves"].sum()
    raw = raw[raw > 0]
    chk = {}
    for a in quarters()["anomes"]:
        s_old = set(k.loc[k["AnoMes"] == a, "ISPB"])
        if a not in raw.index.get_level_values(0):
            continue
        s_new = set(raw.loc[a].index)
        chk[int(a)] = {"panel_ispb": len(s_old), "raw_ispb": len(s_new),
                       "only_panel": len(s_old - s_new), "only_raw": len(s_new - s_old)}
    return k, {"raw_file": files[-1].name, "by_quarter_end": chk}


# ── assembly ─────────────────────────────────────────────────────────────────
def panel_conglomerates() -> pd.DataFrame:
    p = pd.read_parquet(PANEL, columns=["CodConglomeradoPrudencial", "year", "quarter", "is_B",
                                        "NomeInstituicao"] + [f"dep_a{t}" for t in DEP_TYPES])
    q = quarters()
    p = p.merge(q[["year", "quarter"]], on=["year", "quarter"])
    g = p.groupby(["CodConglomeradoPrudencial", "year", "quarter"], as_index=False).agg(
        is_B=("is_B", "first"), name=("NomeInstituicao", "first"),
        **{f"dep_a{t}": (f"dep_a{t}", "sum") for t in DEP_TYPES})
    if (p.groupby("CodConglomeradoPrudencial")["is_B"].nunique() > 1).any():
        raise SystemExit("is_B varies within a conglomerate")
    for t in DEP_TYPES:
        g[f"dep_a{t}_brl_mn"] = g.pop(f"dep_a{t}") / 1e6
    g["dep_total_brl_mn"] = g[[f"dep_a{t}_brl_mn" for t in DEP_TYPES]].sum(axis=1)
    return g.rename(columns={"CodConglomeradoPrudencial": "conglomerate"})


def mandated_sets(xw: pd.Series, cmap: dict) -> tuple[set, pd.DataFrame, set]:
    m = pd.read_csv(MANDATORY, dtype=str)
    m["since"] = pd.to_datetime(m["mandatory_since"])
    m["cnpj8"] = m["cnpj"].str.replace(r"\D", "", regex=True).str.zfill(14).str[:8]
    m["conglomerate"] = m["ispb"].str.zfill(8).map(xw)
    miss = m["conglomerate"].isna()
    m.loc[miss, "conglomerate"] = m.loc[miss, "cnpj8"].map(cmap)
    launch = m[m["since"] <= LAUNCH]
    if len(launch) != 34:
        raise SystemExit(f"{len(launch)} institutions mandated at launch, expected 34")
    # make_pix_exposure.py's mapping (2019 IF.data lists), for comparison only.
    l19 = build_cnpj_to_congl()
    l19 = l19[l19["AnoMes_list"].astype(float).between(201901, 201912)]
    exp_set = set(l19.loc[l19["CNPJ"].isin(set(launch["cnpj8"])), "CodConglomeradoPrudencial"])
    return set(launch["conglomerate"].dropna()), m, exp_set


def build(roster: pd.DataFrame, keys: pd.DataFrame, panel: pd.DataFrame, mand: set) -> pd.DataFrame:
    q = quarters()
    rdates = np.array(sorted(roster["roster_date"].unique()))
    q["roster_date"] = [rdates[rdates <= e].max() for e in q["qend"]]

    on_r = (roster[roster["active"] & roster["conglomerate"].notna()]
            .merge(q[["roster_date", "year", "quarter"]], on="roster_date")
            [["conglomerate", "ispb", "year", "quarter"]].assign(on_roster=True))
    on_k = (keys[keys["conglomerate"].notna()]
            .merge(q[["anomes", "year", "quarter"]], left_on="AnoMes", right_on="anomes")
            .rename(columns={"ISPB": "ispb"})[["conglomerate", "ispb", "year", "quarter"]]
            .assign(on_keys=True))
    inst = on_r.merge(on_k, on=["conglomerate", "ispb", "year", "quarter"], how="outer")
    inst[["on_roster", "on_keys"]] = inst[["on_roster", "on_keys"]].fillna(False).astype(bool)

    idx = ["conglomerate", "year", "quarter"]
    agg = inst.groupby(idx).agg(on_roster=("on_roster", "any"), on_keys=("on_keys", "any"),
                                n_members_on=("ispb", "nunique")).reset_index()
    out = panel.merge(agg, on=idx, how="left").merge(q[["year", "quarter", "roster_date"]],
                                                     on=["year", "quarter"])
    out[["on_roster", "on_keys"]] = out[["on_roster", "on_keys"]].fillna(False).astype(bool)
    out["n_members_on"] = out["n_members_on"].fillna(0).astype(int)
    out["on_pix"] = (out["on_roster"] | out["on_keys"]).astype(int)
    out["source"] = np.select([out["on_roster"] & out["on_keys"], out["on_roster"], out["on_keys"]],
                              ["both", "roster", "keys"], default="")
    per = out["year"] * 4 + out["quarter"]
    first = out[out["on_pix"] == 1].assign(per=per).groupby("conglomerate")["per"].min()
    fp = out["conglomerate"].map(first)
    out["first_on_pix"] = [f"{int((v - 1) // 4)}Q{int((v - 1) % 4 + 1)}" if pd.notna(v) else ""
                           for v in fp]
    out["mandated"] = out["conglomerate"].isin(mand).astype(int)

    # Members ever in either leg, and those off in a quarter while the conglomerate is on.
    members = inst.groupby("conglomerate")["ispb"].apply(lambda s: sorted(set(s)))
    out["n_pix_members"] = out["conglomerate"].map(members.str.len()).fillna(0).astype(int)
    on_sets = inst.groupby(idx)["ispb"].apply(set)
    out["members_off"] = [
        " ".join(sorted(set(members.get(c, [])) - on_sets.get((c, y, qq), set())))
        if on else "" for c, y, qq, on in zip(out["conglomerate"], out["year"], out["quarter"], out["on_pix"])]
    out["roster_date"] = out["roster_date"].dt.date
    cols = ["conglomerate", "year", "quarter", "name", "is_B", "on_pix", "first_on_pix", "mandated",
            "source", "on_roster", "on_keys", "roster_date", "n_pix_members", "n_members_on",
            "members_off"] + [f"dep_a{t}_brl_mn" for t in DEP_TYPES] + ["dep_total_brl_mn"]
    return out[cols].sort_values(["conglomerate", "year", "quarter"]).reset_index(drop=True), inst


# ── report ───────────────────────────────────────────────────────────────────
def rep_coverage(roster: pd.DataFrame, check: dict) -> dict:
    """Item 1: roster dates from 2020-10 to 2021-06, counts, gaps; key entries in the gap months."""
    d = roster.groupby("roster_date").agg(n_listed=("cnpj", "nunique"), n_active=("active", "sum"))
    d["gap_days"] = d.index.to_series().diff().dt.days
    first6 = d[(d.index >= "2020-10-01") & (d.index <= "2021-06-30")]
    entry = pd.read_csv(ENTRY, dtype={"ISPB": str})
    em = entry["entry_anomes"].value_counts().sort_index()
    return {
        "first_six_months": [{"roster_date": str(i.date()), "n_listed": int(v.n_listed),
                              "n_active": int(v.n_active),
                              "gap_days": None if pd.isna(v.gap_days) else int(v.gap_days)}
                             for i, v in first6.iterrows()],
        "all_in_window": [{"roster_date": str(i.date()), "n_listed": int(v.n_listed),
                           "n_active": int(v.n_active),
                           "gap_days": None if pd.isna(v.gap_days) else int(v.gap_days)}
                          for i, v in d.iterrows()],
        "parse_check": {str(k.date()): v for k, v in sorted(check.items())},
        "key_entries_by_month_2020_10_to_2021_06": {int(k): int(v) for k, v in em.items()
                                                    if 202010 <= k <= 202106},
    }


def rep_agreement(roster: pd.DataFrame) -> dict:
    """Item 2: quarter of the first operating roster vs quarter of the key-based entry, both
    dated no earlier than the launch."""
    entry = pd.read_csv(ENTRY, dtype={"ISPB": str})
    entry["ISPB"] = entry["ISPB"].str.zfill(8)
    entry["entry_date"] = pd.to_datetime(entry["entry_date"])
    act = roster[roster["active"]]
    fr = act.groupby("ispb").agg(first_roster=("roster_date", "min"), cnpj8=("cnpj8", "first"),
                                 name=("razao_social", "last"), conglomerate=("conglomerate", "first"),
                                 modalidade=("modalidade", "last"))
    rdates = np.array(sorted(roster["roster_date"].unique()))
    fr["prev_roster"] = [rdates[rdates < d].max() if (rdates < d).any() else pd.NaT
                         for d in fr["first_roster"]]
    both = fr.join(entry.set_index("ISPB")[["entry_date", "Nome"]], how="inner")
    clip = lambda s: s.where(s >= LAUNCH, LAUNCH)
    both["q_roster"] = clip(both["first_roster"]).dt.to_period("Q")
    both["q_keys"] = clip(both["entry_date"]).dt.to_period("Q")
    both["diff_q"] = [(a - b).n for a, b in zip(both["q_roster"], both["q_keys"])]
    both["same"] = both["diff_q"] == 0
    # A roster lag is explained by the archive when the keys appear after the previous roster
    # (on which the institution was not operating) and by the first roster that lists it.
    both["gap_explained"] = (both["diff_q"] > 0) & (both["entry_date"] > both["prev_roster"]) & \
                            (both["entry_date"] <= both["first_roster"])
    post = both[(both["q_roster"] > pd.Period("2020Q4")) | (both["q_keys"] > pd.Period("2020Q4"))]
    worst = both[~both["same"] & ~both["gap_explained"]].assign(a=lambda x: x["diff_q"].abs()) \
        .sort_values("a", ascending=False).head(15)
    only_roster = fr.index.difference(entry["ISPB"])
    only_keys = pd.Index(entry["ISPB"]).difference(fr.index)
    return {
        "n_both": int(len(both)), "share_same_quarter": round(float(both["same"].mean()), 4),
        "n_after_launch_quarter": int(len(post)),
        "share_same_quarter_after_launch_quarter": round(float(post["same"].mean()), 4),
        "roster_later": int((both["diff_q"] > 0).sum()), "roster_earlier": int((both["diff_q"] < 0).sum()),
        "roster_later_explained_by_roster_gap": int(both["gap_explained"].sum()),
        "diff_q_distribution": {int(k): int(v) for k, v in both["diff_q"].value_counts().sort_index().items()},
        "largest_unexplained": [
            {"ispb": i, "name": (r["Nome"] if isinstance(r["Nome"], str) else r["name"]),
             "conglomerate": r["conglomerate"], "first_roster": str(r["first_roster"].date()),
             "key_entry": str(r["entry_date"].date()), "diff_q": int(r["diff_q"]),
             "modalidade": r["modalidade"]} for i, r in worst.iterrows()],
        "n_roster_only": int(len(only_roster)), "n_keys_only": int(len(only_keys)),
        "roster_only_modalidade": {str(k): int(v) for k, v in
                                   fr.loc[only_roster, "modalidade"].value_counts(dropna=False).head(6).items()},
    }


def rep_control(out: pd.DataFrame) -> dict:
    """Item 3: B conglomerates not on Pix and their share of B savings (type 2), by quarter, under
    on_pix and under each leg alone; and, for B and D, the share of each deposit type held by
    conglomerates off Pix."""
    b = out[out["is_B"] == 1]
    rows = []
    for (y, q), s in b.groupby(["year", "quarter"]):
        tot = s["dep_a2_brl_mn"].sum()
        row = {"quarter": f"{y}Q{q}", "n_B": int(len(s)), "B_savings_brl_bn": round(tot / 1e3, 1)}
        for lab, off in (("on_pix", s["on_pix"] == 0), ("roster_only", ~s["on_roster"]),
                         ("keys_only", ~s["on_keys"])):
            row[f"n_off_{lab}"] = int(off.sum())
            row[f"sav_share_off_{lab}"] = round(float(s.loc[off, "dep_a2_brl_mn"].sum() / tot), 5)
        rows.append(row)
    off = b[b["on_pix"] == 0]
    biggest = (off.assign(q=off["year"].astype(str) + "Q" + off["quarter"].astype(str))
               .groupby("conglomerate").agg(name=("name", "last"), quarters_off=("q", "size"),
                                            first_off=("q", "min"), last_off=("q", "max"),
                                            max_sav_brl_mn=("dep_a2_brl_mn", "max"))
               .sort_values("max_sav_brl_mn", ascending=False).head(12).reset_index())
    by_type = []
    for (y, q, isb), s in out.groupby(["year", "quarter", "is_B"]):
        off = s["on_pix"] == 0
        row = {"quarter": f"{y}Q{q}", "group": "B" if isb == 1 else "D", "n": int(len(s)),
               "n_off": int(off.sum())}
        for c in [f"dep_a{t}_brl_mn" for t in DEP_TYPES] + ["dep_total_brl_mn"]:
            tot = s[c].sum()
            row[f"share_off_{c[:-7]}"] = round(float(s.loc[off, c].sum() / tot), 5) if tot > 0 else None
        by_type.append(row)
    stale = out[(out["on_pix"] == 1) & ~out["on_keys"]
                & (pd.to_datetime(out["roster_date"]) < pd.PeriodIndex(
                    out["year"].astype(str) + "Q" + out["quarter"].astype(str), freq="Q").start_time)]
    return {"by_quarter": rows, "largest_off_pix_B": biggest.round(1).to_dict("records"),
            "off_share_by_type": by_type,
            "on_by_stale_roster_only": stale.groupby(["conglomerate", "name", "is_B"]).size()
            .rename("quarters").reset_index().to_dict("records")}


def rep_mandate(out: pd.DataFrame, mrows: pd.DataFrame, exp_set: set) -> dict:
    """Item 4: mandated vs voluntary among panel conglomerates; voluntary first quarter."""
    c = out.sort_values(["year", "quarter"]).groupby("conglomerate").agg(
        is_B=("is_B", "first"), mandated=("mandated", "first"), ever_on=("on_pix", "max"),
        first_on=("first_on_pix", "first"), name=("name", "last"))
    c["group"] = np.where(c["mandated"] == 1, "mandated",
                          np.where(c["ever_on"] == 1, "voluntary", "never_on"))
    res = {"counts": {}, "voluntary_first_quarter": {}}
    for lab, s in (("all", c), ("B", c[c["is_B"] == 1]), ("D", c[c["is_B"] == 0])):
        res["counts"][lab] = {k: int(v) for k, v in s["group"].value_counts().items()}
        v = s[s["group"] == "voluntary"]["first_on"].value_counts().sort_index()
        res["voluntary_first_quarter"][lab] = {k: int(n) for k, n in v.items()}
    in_panel = set(out["conglomerate"])
    launch = mrows[mrows["since"] <= LAUNCH]
    res["mandated_institutions"] = [
        {"ispb": r["ispb"], "nome": r["nome"], "conglomerate": r["conglomerate"],
         "in_panel": r["conglomerate"] in in_panel} for _, r in launch.iterrows()]
    mand_panel = set(c.index[c["mandated"] == 1])
    res["mandated_conglomerates_in_panel"] = sorted(mand_panel)
    res["vs_make_pix_exposure_2019_lists"] = {
        "only_here": sorted(mand_panel - exp_set), "only_exposure": sorted((exp_set & in_panel) - mand_panel)}
    res["mandated_without_date"] = mrows[mrows["since"].isna()][["ispb", "nome", "conglomerate"]] \
        .to_dict("records")
    res["mandated_never_on"] = sorted(c.index[(c["mandated"] == 1) & (c["ever_on"] == 0)])
    exits = out.sort_values(["conglomerate", "year", "quarter"]).groupby("conglomerate")["on_pix"] \
        .apply(lambda s: int(((s.shift(1) == 1) & (s == 0)).sum()))
    res["conglomerates_with_an_exit"] = {k: c.loc[k, "name"] for k in exits[exits > 0].index}
    return res


def rep_members(out: pd.DataFrame, inst: pd.DataFrame, roster: pd.DataFrame, keys: pd.DataFrame) -> dict:
    """Item 5: conglomerate-quarters where a member ever on Pix is off while another is on."""
    names = pd.concat([keys.drop_duplicates("ISPB", keep="last").set_index("ISPB")["Nome"],
                       roster.drop_duplicates("ispb", keep="last").set_index("ispb")["razao_social"]])
    names = names[~names.index.duplicated()]
    d = out[(out["on_pix"] == 1) & (out["members_off"] != "")]
    d = d.assign(q=d["year"].astype(str) + "Q" + d["quarter"].astype(str))
    by_c = d.groupby("conglomerate").agg(
        name=("name", "last"), is_B=("is_B", "first"),
        quarters=("q", "size"), first=("q", "min"), last=("q", "max"),
        members_off=("members_off", lambda s: " ".join(sorted(set(" ".join(s).split()))))).reset_index()
    by_c["members_off_names"] = by_c["members_off"].map(
        lambda s: "; ".join(f"{i} {names.get(i, '?')}" for i in s.split()))
    return {"n_conglomerate_quarters": int(len(d)), "n_conglomerates": int(by_c["conglomerate"].nunique()),
            "rule": "any member on -> conglomerate on",
            "list": by_c.sort_values("quarters", ascending=False).to_dict("records")}


def main():
    xwdf = pd.read_csv(XWALK, dtype=str)
    if xwdf["ispb"].duplicated().any():
        raise SystemExit("the crosswalk lists an ISPB twice")
    xw = xwdf.set_index("ispb")["conglomerate"].dropna()
    if xw.get(PICPAY_ISPB) != firm_remaps.PICPAY_TO:
        raise SystemExit(f"the crosswalk no longer maps PicPay to {firm_remaps.PICPAY_TO}")
    cmap = {str(int(k)).zfill(8): str(v[0]).strip() for k, v in build_cnpj_conglomerate_map().items()
            if v and v[0] is not None and str(v[0]).strip() not in ("", "nan", "None")}

    roster, check = load_roster(cmap, xw)
    keys, kchk = load_keys(xw, cmap)
    panel = panel_conglomerates()
    mand, mrows, exp_set = mandated_sets(xw, cmap)
    out, inst = build(roster, keys, panel, mand)
    out.to_csv(OUT, index=False, encoding="utf-8")

    rep = {"coverage": rep_coverage(roster, check), "keys_crosscheck": kchk,
           "agreement": rep_agreement(roster), "control": rep_control(out),
           "mandate": rep_mandate(out, mrows, exp_set), "members": rep_members(out, inst, roster, keys)}
    rep["_meta"] = {"written": datetime.datetime.now().isoformat(timespec="seconds"),
                    "rows": int(len(out)), "conglomerates": int(out["conglomerate"].nunique()),
                    "B_conglomerates": int(out.loc[out["is_B"] == 1, "conglomerate"].nunique()),
                    "roster_dates_read": [str(pd.Timestamp(d).date()) for d in sorted(roster["roster_date"].unique())]}
    SIDE.write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    for r in rep["control"]["by_quarter"]:
        print(f"  {r['quarter']}: B {r['n_B']:>3}, off Pix {r['n_off_on_pix']:>3} "
              f"(savings share {r['sav_share_off_on_pix']:.4f}); roster leg alone {r['n_off_roster_only']:>3}, "
              f"keys leg alone {r['n_off_keys_only']:>3}")
    print(f"  wrote {OUT} ({len(out):,} rows)\n  wrote {SIDE}")


if __name__ == "__main__":
    main()

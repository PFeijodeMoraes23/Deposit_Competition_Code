"""
build_cosif_lai_advertising.py
==============================
Advertising expense per institution and per prudential conglomerate, monthly from 2013-01 to
2024-12, from the file the central bank released on appeal under the access-to-information law
(Fala.BR; received 2026-10-05). The file carries the three level-4 accounts that the public
balancetes omit before 2025, with each institution's CNPJ and name, so its series run into those
of `scrape_cosif_advertising.py`, which start in 2025-01.

Input
-----
  <AWARENESS_RAW>/lai_bcb_cosif/Contas 81742-45e48 demanda recursal.zip
  One CSV, latin-1, `;`-separated, columns DATA;CNPJ;INSTITUICAO;DOC;CONTA;SALDO.

Accounts (the 8-digit plan in force until 2024; the 10-digit codes of 2025 are the same accounts)
------------------------------------------------------------------------------------------------
  81745009  Despesas de Propaganda e Publicidade        -> adv
  81742002  Despesas de Promocoes e Relacoes Publicas   -> promo
  81748006  Despesas de Publicacoes                     -> publ
The level-3 total of administrative expenses is not in the file, so `admin` and the shares of it
are empty in these tables.

Documents
---------
  4010   individual balancete, keyed by the institution's CNPJ8          -> level institution
  40605  block 5 (consolidated) of the prudential conglomerate's balancete, from 2014-07, keyed
         by the CNPJ8 of the conglomerate's leader                        -> level conglomerate
The other documents (4020, 4030, 4040, 4060, 40601-40604, 40606, 40607, 4090) are kept in the
long table only. 40604 lists the individual entities consolidated into a conglomerate and
repeats keys; it is never added to 40605.

For conglomerate rows `entity_key` is the leader's CNPJ8, not a prudential code: the file has
no code. `cod_congl_2025` is the code that leader carries in the 2025 public files, where it
leads exactly one conglomerate there. Mapping to the market panel's code quarter by quarter is
the crosswalk's job.

Units, accumulation, and what an absent row means
-------------------------------------------------
  - SALDO is in reais and stored negative. Balances are converted once, to positive reais.
  - Balances accumulate within each half-year, as in the public files, and flows come from the
    same differencing (`scrape_cosif_advertising.monthly_flows` and `quarterly_flows`).
  - The file holds non-zero balances only, and institutions file at different frequencies:
    banks every month, many small institutions only at quarter ends or half-year ends. An
    absent month is therefore either a zero balance or a month the institution does not file,
    and the file alone cannot always tell which. The rule:
      * An entity-year is MONTHLY when it has a balance in a month other than March, June,
        September or December.
      * In a monthly entity-year, a month with no row is a true zero when it precedes the
        first balance of its half-year, or lies in a half-year with no balance at all, and
        falls between the entity's first and last month in the file. It is written with zero
        balances and zero_inferred = True.
      * Every other absent month stays missing, and a flow that needs it is missing.
    Zeros are inferred for institutions only. A conglomerate's document is filed by its leader,
    and the leader can change: Banco Original led C0080903 until 2023Q2, PicPay from 2023Q3 to
    2024Q3, Banco Original again from 2024Q4. A leader's month without a row may be a month
    another institution filed, so at the conglomerate level it always stays missing.
    Quarter flows use quarter-end balances only, so an institution that files at quarter ends
    still gets quarterly flows; one that files at half-year ends gets the half-year table.

Outputs (paths.AWARENESS_PROC)
------------------------------
  cosif_lai_advertising_monthly.{parquet,csv}    entity x month: balances and monthly flows
  cosif_lai_advertising_quarterly.{parquet,csv}  entity x quarter: flows; flow_known says
                                                 whether the quarter's flow could be computed,
                                                 flow_uses_inferred_zero whether it rests on
                                                 an inferred zero, and n_months_inferred how
                                                 many of its months are inferred zeros
  cosif_lai_advertising_halfyear.{parquet,csv}   entity x half-year: the June and December
                                                 balances, which are the flows of the half
  cosif_lai_advertising_long.parquet             every row of the file, all documents
The first two have the columns of the public-file tables, so the two can be stacked; rows are
kept from an entity's first to its last month in the file. `panel_key` is CNPJ_<cnpj8> for
institution rows and LEADER_<cnpj8> for conglomerate rows.

Validation (any failure aborts before writing)
----------------------------------------------
  1. The header is the six expected columns.
  2. Months run from 201301 to 202412 with none missing.
  3. Every CNPJ is 8 digits and every account is one of the three.
  4. Every SALDO is present and not positive.
  5. No (month, CNPJ, account) repeats within document 4010 or within 40605.
  6. Identity anchors by CNPJ: Banco do Brasil, Caixa, Itau Unibanco, Santander among the
     institutions; Itau Unibanco Holding and Banco do Brasil among the leaders.
  7. Reset: among entities with an adv balance in both June and July, the median ratio of
     July to June is below one half, every year.
  8. Within-half falls of the adv, promo and publ running totals stay under 5% of consecutive
     month pairs.
  9. Where all three months of a quarter are known, the monthly flows sum to the quarter flow.
 10. In monthly entity-years, months left missing after a balance in the same half-year stay
     under 2% of the months that have a row, an inferred zero or such a gap.
 11. Where the statement-note table is on disk: Santander's individual-statement note for a
     full year is within 3% of its document-4010 sum of the three accounts, every year both
     have. Other banks' ratios are logged, not asserted.

Usage
-----
  python build_cosif_lai_advertising.py
  python build_cosif_lai_advertising.py --out-dir <dir>      # write somewhere else (smoke runs)
"""

from __future__ import annotations

import argparse
import io
import logging
import unicodedata
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import scrape_cosif_advertising as cosif
from utils import paths

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

LAI_FILE = (paths.AWARENESS_RAW / "lai_bcb_cosif"
            / "Contas 81742-45e48 demanda recursal.zip")
OUT_DIR = paths.AWARENESS_PROC

COLUMNS = ["DATA", "CNPJ", "INSTITUICAO", "DOC", "CONTA", "SALDO"]
ACCOUNTS = {"81745009": "adv", "81742002": "promo", "81748006": "publ"}
AD_FIELDS = cosif.AD_FIELDS
FIRST_MONTH, LAST_MONTH = "201301", "202412"
LEVEL_OF_DOC = {"4010": "institution", "40605": "conglomerate"}
QUARTER_END_MONTHS = (3, 6, 9, 12)
KEYS = cosif.KEYS

ANCHORS = {"00000000": "BANCO DO BRASIL", "00360305": "CAIXA ECONOMICA", "60701190": "ITAU",
           "90400888": "SANTANDER"}
LEADER_ANCHORS = {"60872504": "ITAU", "00000000": "BANCO DO BRASIL"}
MAX_HOLE_SHARE = 0.02
MAX_JULY_OVER_JUNE = 0.5
NOTE_ANCHOR = ("santander", "90400888")
NOTE_TOLERANCE = 0.03
# Statement-note banks whose yearly ratio to the file is logged: bank_key -> (CNPJ8, entity name
# in the note table, or None for any).
NOTE_LOGGED = {"santander": ("90400888", None), "btg": ("30306294", None),
               "daycoval": ("62232889", None),
               "volkswagen": ("59109165", "Banco Volkswagen S.A.")}
NOTE_GROUPS = ("adv_promo_publ_bundled", "advertising", "adv_promo_bundled")


def fold(text: str) -> str:
    return unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().upper()


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def read_file(path: Path) -> pd.DataFrame:
    """The released file as it is, one row per (month, CNPJ, document, account)."""
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            members = [m for m in z.namelist() if m.lower().endswith(".csv")]
            if len(members) != 1:
                raise SystemExit(f"{path.name}: expected one CSV inside, found {members}")
            raw = z.read(members[0])
    else:
        raw = path.read_bytes()
    df = pd.read_csv(io.BytesIO(raw), sep=";", encoding="latin-1",
                     dtype={"DATA": str, "CNPJ": str, "DOC": str, "CONTA": str})
    if list(df.columns) != COLUMNS:                                          # check 1
        raise AssertionError(f"{path.name}: columns {list(df.columns)}, expected {COLUMNS}")
    return df


def check_structure(df: pd.DataFrame) -> None:
    expected = [m.strftime("%Y%m") for m in pd.period_range(
        pd.Period(FIRST_MONTH, "M"), pd.Period(LAST_MONTH, "M"), freq="M")]
    if sorted(df["DATA"].unique()) != expected:                              # check 2
        raise AssertionError(f"months are not exactly {FIRST_MONTH}-{LAST_MONTH}: "
                             f"{df['DATA'].min()} to {df['DATA'].max()}, "
                             f"{df['DATA'].nunique()} distinct")
    if not df["CNPJ"].str.fullmatch(r"\d{8}").all():                         # check 3
        raise AssertionError("a CNPJ is not 8 digits")
    unknown = set(df["CONTA"]) - set(ACCOUNTS)
    if unknown:
        raise AssertionError(f"accounts outside the three requested: {sorted(unknown)}")
    if df["SALDO"].isna().any() or (df["SALDO"] > 0).any():                  # check 4
        raise AssertionError("a SALDO is missing or positive in an expense account")
    for doc in LEVEL_OF_DOC:                                                 # check 5
        dup = df[df["DOC"] == doc].duplicated(["DATA", "CNPJ", "CONTA"])
        if dup.any():
            raise AssertionError(f"document {doc}: {int(dup.sum())} repeated "
                                 f"(month, CNPJ, account) rows")


# ---------------------------------------------------------------------------
# Balances
# ---------------------------------------------------------------------------
def observed_balances(df: pd.DataFrame, doc: str) -> pd.DataFrame:
    """Entity x month balances (positive reais) for the months the file has a row for.

    An account with no row in a month that has another account's row is a true zero: the file
    leaves out zero balances, and the entity filed that month.
    """
    x = df[df["DOC"] == doc]
    wide = (x.assign(field=x["CONTA"].map(ACCOUNTS), bal=-x["SALDO"])
             .pivot(index=["CNPJ", "DATA"], columns="field", values="bal")
             .reindex(columns=list(AD_FIELDS)).fillna(0.0))
    wide.columns = [f"{f}_bal" for f in AD_FIELDS]
    names = x.groupby(["CNPJ", "DATA"])["INSTITUICAO"].first().rename("entity_name")
    out = wide.join(names).reset_index().rename(columns={"CNPJ": "entity_key",
                                                         "DATA": "data_base"})
    out["zero_inferred"] = False
    return out


def with_inferred_zeros(obs: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Add the months that are true zeros under the module's rule.

    Returns the panel, the number of months added, and the number of months of monthly
    entity-years left missing after a balance in the same half-year.
    """
    cal = cosif._calendar(obs[["entity_key", "data_base"]].copy())
    span = cal.groupby("entity_key")["data_base"].agg(first="min", last="max")
    monthly_years = (cal.loc[~cal["month"].isin(QUARTER_END_MONTHS), ["entity_key", "year"]]
                        .drop_duplicates())
    grid = monthly_years.merge(pd.DataFrame({"month": range(1, 13)}), how="cross")
    grid["data_base"] = (grid["year"] * 100 + grid["month"]).astype(str)
    grid["half"] = np.where(grid["month"] <= 6, 1, 2)
    grid = grid.merge(span, left_on="entity_key", right_index=True)
    grid = grid[(grid["data_base"] >= grid["first"]) & (grid["data_base"] <= grid["last"])]
    first_in_half = (cal.groupby(["entity_key", "year", "half"])["month"].min()
                        .rename("first_in_half").reset_index())
    grid = grid.merge(first_in_half, on=["entity_key", "year", "half"], how="left")
    grid = grid.merge(cal[["entity_key", "data_base"]].assign(observed=True),
                      on=["entity_key", "data_base"], how="left")
    absent = grid[grid["observed"].isna()]
    is_zero = absent["first_in_half"].isna() | (absent["month"] < absent["first_in_half"])
    zeros = absent.loc[is_zero, ["entity_key", "data_base"]].copy()
    for f in AD_FIELDS:
        zeros[f"{f}_bal"] = 0.0
    zeros["zero_inferred"] = True
    panel = pd.concat([obs, zeros], ignore_index=True)
    return panel, len(zeros), int((~is_zero).sum())


def leader_codes(out_dir: Path) -> dict[str, str]:
    """Leader CNPJ8 -> prudential code, from the 2025 public-file table, where unambiguous."""
    path = out_dir / "cosif_advertising_quarterly.parquet"
    if not path.exists():
        path = OUT_DIR / "cosif_advertising_quarterly.parquet"
    if not path.exists():
        log.warning("no cosif_advertising_quarterly.parquet; cod_congl_2025 left empty")
        return {}
    q = pd.read_parquet(path, columns=["level", "entity_key", "cnpj_leader"])
    c = (q[q["level"] == "conglomerate"].dropna(subset=["cnpj_leader"])
         .assign(cnpj_leader=lambda d: d["cnpj_leader"].astype(str))
         .drop_duplicates(["entity_key", "cnpj_leader"]))
    n_codes = c.groupby("cnpj_leader")["entity_key"].nunique()
    one = c[c["cnpj_leader"].isin(n_codes[n_codes == 1].index)]
    return dict(zip(one["cnpj_leader"], one["entity_key"]))


def build_tables(df: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame,
                                                             pd.DataFrame, dict]:
    """Monthly, quarterly and half-year tables for both levels, and the counts to report."""
    panels, counts = [], {}
    for doc, level in LEVEL_OF_DOC.items():
        obs = observed_balances(df, doc)
        panel, n_zero, n_hole = with_inferred_zeros(obs)
        if level == "conglomerate":
            panel, n_zero = obs, 0          # a leader's absent month is never taken as a zero
        panel["level"] = level
        panel["cnpj_leader"] = panel["entity_key"] if level == "conglomerate" else np.nan
        panel["taxonomia"] = np.nan
        panel["admin_bal"] = np.nan
        panels.append(panel)
        counts[level] = {"entities": obs["entity_key"].nunique(), "observed": len(obs),
                         "zero_inferred": n_zero, "holes": n_hole}
    monthly = cosif.monthly_flows(pd.concat(panels, ignore_index=True))
    monthly["zero_inferred"] = monthly["zero_inferred"].eq(True)

    # Rows from an entity's first to its last month in the file; outside it nothing is known.
    filed = monthly[monthly["filed"]]
    span = filed.groupby(KEYS)["data_base"].agg(first="min", last="max").reset_index()
    monthly = monthly.merge(span, on=KEYS)
    monthly = monthly[(monthly["data_base"] >= monthly["first"])
                      & (monthly["data_base"] <= monthly["last"])]
    monthly = monthly.drop(columns=["first", "last"]).reset_index(drop=True)

    quarterly = cosif.quarterly_flows(monthly)
    quarterly["flow_known"] = quarterly["all3"].notna()
    # As in the public-file table, n_months_filed counts months with a row; inferred zeros are
    # counted apart. A quarter's flow rests on an inferred zero when its last month, or the last
    # month of the quarter before it in the same half-year, is one.
    qkeys = KEYS + ["year", "quarter"]
    inferred = monthly.groupby(qkeys)["zero_inferred"].sum().rename("n_months_inferred")
    quarterly = quarterly.merge(inferred.reset_index(), on=qkeys, how="left")
    quarterly["n_months_filed"] -= quarterly["n_months_inferred"]
    end_zero = (monthly[monthly["month"].isin(QUARTER_END_MONTHS)]
                .set_index(qkeys)["zero_inferred"])
    at_end = (end_zero.reindex(pd.MultiIndex.from_frame(quarterly[qkeys])).fillna(False)
              .to_numpy(bool))
    before = quarterly[qkeys].assign(quarter=quarterly["quarter"] - 1)
    at_start = (end_zero.reindex(pd.MultiIndex.from_frame(before)).fillna(False).to_numpy(bool)
                & quarterly["quarter"].isin([2, 4]).to_numpy())
    quarterly["flow_uses_inferred_zero"] = quarterly["flow_known"] & (at_end | at_start)

    ends = monthly[monthly["month"].isin([6, 12])]
    halfyear = ends[KEYS + ["year", "half", "entity_name", "cnpj_leader", "filed",
                            "zero_inferred"]].copy()
    for f in AD_FIELDS:
        halfyear[f] = ends[f"{f}_bal"]
    halfyear["all3"] = halfyear[list(AD_FIELDS)].sum(axis=1, min_count=len(AD_FIELDS))
    halfyear = (halfyear.rename(columns={"filed": "half_end_known",
                                         "zero_inferred": "half_end_inferred"})
                .reset_index(drop=True))

    codes = leader_codes(out_dir)
    for frame in (monthly, quarterly, halfyear):
        is_congl = frame["level"] == "conglomerate"
        frame["cod_congl_2025"] = frame["entity_key"].map(codes).where(is_congl)
        frame["panel_key"] = np.where(is_congl, "LEADER_" + frame["entity_key"],
                                      "CNPJ_" + frame["entity_key"])
    return monthly, quarterly, halfyear, counts


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate(monthly: pd.DataFrame, quarterly: pd.DataFrame, counts: dict) -> None:
    filed = monthly[monthly["filed"]]

    names = (filed[filed["level"] == "institution"].groupby("entity_key")["entity_name"]
             .agg(lambda s: " | ".join(sorted({fold(v) for v in s.dropna()}))))
    for cnpj8, token in ANCHORS.items():                                     # check 6
        if cnpj8 not in names.index or token not in names[cnpj8]:
            raise AssertionError(f"identity anchor failed: {cnpj8} should contain {token}, "
                                 f"got {names.get(cnpj8)!r}")

    # The same entities are compared in both months: June also holds the quarter-end filers,
    # which July does not, so the two months' medians are over different populations.
    for level in LEVEL_OF_DOC.values():                                      # check 7
        x = filed[(filed["level"] == level) & (filed["adv_bal"] > 0)
                  & filed["month"].isin([6, 7])]
        pair = x.pivot(index=["entity_key", "year"], columns="month", values="adv_bal").dropna()
        ratio = (pair[7] / pair[6]).groupby(level="year").median()
        if ratio.empty or (ratio >= MAX_JULY_OVER_JUNE).any():
            raise AssertionError(f"{level}: the median July/June adv balance of an entity is "
                                 f"not below {MAX_JULY_OVER_JUNE} in "
                                 f"{ratio[ratio >= MAX_JULY_OVER_JUNE].round(3).to_dict()}; "
                                 f"the half-year reset changed")
        log.info("  %-12s median July/June adv balance by year: %.3f to %.3f", level,
                 ratio.min(), ratio.max())

    pairs = filed[~filed["month"].isin([1, 7])]                              # check 8
    for f in AD_FIELDS:
        x = pairs[f"{f}_flow"].dropna()
        share = float((x < -1.0).mean()) if len(x) else 0.0
        log.info("  within-half falls of the %s running total: %.2f%% of %d month pairs",
                 f, 100 * share, len(x))
        if share > cosif.MAX_REVERSAL_SHARE:
            raise AssertionError(f"{f}: {share:.1%} of month pairs fall, above "
                                 f"{cosif.MAX_REVERSAL_SHARE:.0%}")

    leaders = (filed[filed["level"] == "conglomerate"].groupby("entity_key")["entity_name"]
               .agg(lambda s: " | ".join(sorted({fold(v) for v in s.dropna()}))))
    for cnpj8, token in LEADER_ANCHORS.items():
        if cnpj8 not in leaders.index or token not in leaders[cnpj8]:
            raise AssertionError(f"leader anchor failed: {cnpj8} should contain {token}, "
                                 f"got {leaders.get(cnpj8)!r}")

    flows = [f"{f}_flow" for f in AD_FIELDS]                                 # check 9
    sums = monthly.groupby(KEYS + ["year", "quarter"])[flows].sum(min_count=3).reset_index()
    three_known = quarterly["n_months_filed"] + quarterly["n_months_inferred"] == 3
    j = quarterly[three_known].merge(sums, on=KEYS + ["year", "quarter"])
    for f in AD_FIELDS:
        both = j[j[f].notna() & j[f"{f}_flow"].notna()]
        diff = (both[f] - both[f"{f}_flow"]).abs()
        tol = np.maximum(0.01, 1e-9 * both[f].abs())
        if (diff > tol).any():
            raise AssertionError(f"{f}: monthly flows do not sum to the quarter flow for "
                                 f"{int((diff > tol).sum())} entity-quarters")
    log.info("  %d entity-quarters with three known months reconcile to their monthly flows",
             len(j))

    for level, c in counts.items():                                          # check 10
        share = c["holes"] / max(1, c["observed"] + c["zero_inferred"] + c["holes"])
        log.info("  %-12s %5d entities, %6d months with a row, %6d inferred zeros, "
                 "%5d left missing after a balance (%.2f%%)", level, c["entities"],
                 c["observed"], c["zero_inferred"], c["holes"], 100 * share)
        if share > MAX_HOLE_SHARE:
            raise AssertionError(f"{level}: {share:.1%} of the months of monthly entity-years "
                                 f"are missing after a balance, above {MAX_HOLE_SHARE:.0%}")


def check_against_notes(halfyear: pd.DataFrame, out_dir: Path) -> None:
    """Check 11: full-year statement-note figures against the file's yearly sum."""
    path = out_dir / "bank_notes_advertising_periods.parquet"
    if not path.exists():
        path = OUT_DIR / "bank_notes_advertising_periods.parquet"
    if not path.exists():
        log.warning("no statement-note table on disk; the note comparison is skipped")
        return
    notes = pd.read_parquet(path)
    start, end = pd.to_datetime(notes["window_start"]), pd.to_datetime(notes["window_end"])
    full_year = ((notes["frequency"] == "year") & (start.dt.month == 1) & (start.dt.day == 1)
                 & (end.dt.month == 12) & (start.dt.year == end.dt.year))
    notes = notes[full_year & (notes["scope"] == "individual")
                  & notes["category_group"].isin(NOTE_GROUPS)].assign(year=end.dt.year)

    inst = halfyear[(halfyear["level"] == "institution") & halfyear["half_end_known"]]
    yearly = (inst.groupby(["entity_key", "year"])["all3"].agg(["sum", "size"]).reset_index())
    yearly = yearly[yearly["size"] == 2].rename(columns={"sum": "file_all3"})

    rows = []
    for bank, (cnpj8, entity) in NOTE_LOGGED.items():
        n = notes[notes["bank_key"] == bank]
        if entity is not None:
            n = n[n["entity"] == entity]
        n = n.groupby("year")["amount_brl"].agg(lambda s: sorted(set(s.round(0))))
        f = yearly[yearly["entity_key"] == cnpj8].set_index("year")["file_all3"]
        for year in sorted(set(n.index) & set(f.index)):
            for value in n[year]:
                rows.append({"bank": bank, "year": year, "note": value, "file_all3": f[year],
                             "ratio": value / f[year] if f[year] else np.nan})
    table = pd.DataFrame(rows)
    if table.empty:
        log.warning("no statement-note year overlaps the file; nothing compared")
        return
    log.info("statement note (individual, full year) over the file's three accounts:\n%s",
             table.assign(note=(table["note"] / 1e6).round(1),
                          file_all3=(table["file_all3"] / 1e6).round(1),
                          ratio=table["ratio"].round(4)).to_string(index=False))
    anchor = table[table["bank"] == NOTE_ANCHOR[0]]
    off = anchor[(anchor["ratio"] - 1).abs() > NOTE_TOLERANCE]
    if anchor.empty:
        log.warning("%s has no note year to anchor on", NOTE_ANCHOR[0])
    elif len(off):
        raise AssertionError(f"{NOTE_ANCHOR[0]}: note and file differ by more than "
                             f"{NOTE_TOLERANCE:.0%} in {off['year'].tolist()}")
    else:
        log.info("  %s: %d years within %.0f%% of the file (ratios %.4f to %.4f)",
                 NOTE_ANCHOR[0], len(anchor), 100 * NOTE_TOLERANCE, anchor["ratio"].min(),
                 anchor["ratio"].max())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="COSIF advertising accounts 2013-2024, from the "
                                             "file released by the central bank")
    ap.add_argument("--file", type=Path, default=LAI_FILE)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    a = ap.parse_args()
    if not a.file.exists():
        raise SystemExit(f"missing input: {a.file}")

    log.info("reading %s", a.file)
    df = read_file(a.file)
    check_structure(df)
    log.info("%d rows, %d CNPJs, %s to %s; documents: %s", len(df), df["CNPJ"].nunique(),
             df["DATA"].min(), df["DATA"].max(), df["DOC"].value_counts().to_dict())

    monthly, quarterly, halfyear, counts = build_tables(df, a.out_dir)
    validate(monthly, quarterly, counts)
    check_against_notes(halfyear, a.out_dir)

    long = df.assign(field=df["CONTA"].map(ACCOUNTS), balance_brl=-df["SALDO"])
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (("cosif_lai_advertising_monthly", monthly),
                        ("cosif_lai_advertising_quarterly", quarterly),
                        ("cosif_lai_advertising_halfyear", halfyear)):
        frame.to_parquet(a.out_dir / f"{name}.parquet", index=False)
        frame.to_csv(a.out_dir / f"{name}.csv", index=False)
        log.info("%s: %d rows -> %s", name, len(frame), a.out_dir / f"{name}.parquet")
    long.to_parquet(a.out_dir / "cosif_lai_advertising_long.parquet", index=False)
    log.info("cosif_lai_advertising_long: %d rows", len(long))

    known = quarterly[quarterly["flow_known"]]
    summary = (known.groupby(["level", "year"])
                    .agg(entities=("entity_key", "nunique"), quarters=("all3", "size"),
                         adv_brl_m=("adv", "sum"), all3_brl_m=("all3", "sum")))
    summary[["adv_brl_m", "all3_brl_m"]] = (summary[["adv_brl_m", "all3_brl_m"]] / 1e6).round(1)
    log.info("yearly totals over entity-quarters with a known flow (R$ million):\n%s",
             summary.to_string())


if __name__ == "__main__":
    main()

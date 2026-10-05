"""
scrape_cosif_advertising.py
===========================
Advertising expense per institution and per prudential conglomerate from the COSIF
balancetes, monthly from 2025-01, with quarterly flows.

Accounts (10-digit plan, level 4 of group 8.1.7 Despesas Administrativas)
-------------------------------------------------------------------------
  8174500005  (-) Despesas de Propaganda e Publicidade        -> adv
  8174200006  (-) Despesas de Promocoes e Relacoes Publicas   -> promo
  8174800004  (-) Despesas de Publicacoes                     -> publ
  8170000004  (-) Despesas Administrativas (level-3 total)    -> admin

Coverage. The BCB publishes balancetes at level 4 only from data-base 202501
(Comunicado 44.132/2025). Institutions always REPORTED these accounts; before 2025 the
public files stop at the level-3 total 81700006, so there is nothing to extract before 2025.

Files
-----
  {YYYYMM}BANCOS.*, SOCIEDADES.*, CONSORCIOS.*, COOPERATIVAS.*, LIQUIDACAO.*
                           document 4010, one row set per institution, keyed by CNPJ
                           (COD_CONGL is blank on these rows). The five files are one
                           universe split by kind of institution: banks; finance, payment,
                           brokerage and leasing companies; consortium administrators;
                           cooperatives; institutions in liquidation. A conglomerate's members
                           are spread across them (Nu Pagamentos is in SOCIEDADES), so all are
                           read. CONSORCIOS also carries document 4110, the consortium groups,
                           which is left out.
  {YYYYMM}BLOPRUDENCIAL.*  document 4060, one row set per prudential conglomerate, keyed by
                           COD_CONGL, the same code space as the panel's
                           CodConglomeradoPrudencial
Some months exist twice on disk (a .csv.zip and a plain .CSV with different generation
dates). The file with the latest "Data de geracao dos dados" wins and the choice is logged.

Units and accumulation
----------------------
  - SALDO is in reais, Brazilian decimal format, stored NEGATIVE (expense accounts).
    Balances are converted once, to positive reais.
  - Result accounts ACCUMULATE within each half-year: the June and December balancetes hold
    the whole half, July and January start again. Flows come from differencing within the
    half-year:
        monthly    flow(m) = bal(m) - bal(m-1), and bal(m) in January and July
        quarterly  Q1 = bal(Mar), Q2 = bal(Jun) - bal(Mar), Q3 = bal(Sep), Q4 = bal(Dec) - bal(Sep)
    A flow is missing when a balance it needs is missing; a gap is never absorbed.
  - Zero balances are not published. For an entity that filed the document in a month, an
    absent account is a true zero: the level-4 children of 8.1.7 sum exactly to the level-3
    total for every filer, which is asserted on every file. For an entity that did not file
    that month, every balance is missing.
  - A running total can fall within a half-year when an expense is reclassified. Such
    months get a negative flow and is_reversal = True; they are kept, not dropped.
  - A month whose adv flow is more than 5x the entity's median positive monthly flow, and
    more than R$10m above it, gets adv_flow_spike = True (for example NU PAGAMENTOS in
    202603: R$1.55bn in one month against about R$60m a month before). Kept unaltered.

Outputs (paths.AWARENESS_PROC)
------------------------------
  cosif_advertising_monthly.{parquet,csv}    entity x month: balances and monthly flows
  cosif_advertising_quarterly.{parquet,csv}  entity x quarter: flows and shares of admin
Rows carry `level` (institution | conglomerate), `entity_key` (CNPJ8 or COD_CONGL) and the
raw names. `panel_key` is COD_CONGL for conglomerate rows and CNPJ_<cnpj8> for institution
rows, the form the market panel uses for institutions outside any conglomerate.

Validation (any failure aborts before writing)
----------------------------------------------
  1. No level-4 advertising account appears in the 2024 files on disk.
  2. Every raw balance of the four accounts is negative.
  3. The level-4 children of 8.1.7 sum to the level-3 total for every filer and month.
  4. Reset: across entities, the median adv balance in July is below June's, every year.
  5. Advertising reporters per month within the bounds set for each file type
     (REPORTER_BOUNDS).
  6. Identity anchors: C0080099 is ITAU, C0080738 is CAIXA, C0080329 is BB.
  7. No duplicate (entity, account, month) rows after vintage selection, and no entity in two
     file types in the same month.
  8. Within-half falls of the adv, promo and publ running totals stay under 5% of
     consecutive month pairs.
  9. Where all three months of a quarter were filed, the monthly flows sum to the quarter
     flow computed from quarter-end balances.

Usage
-----
  python scrape_cosif_advertising.py
  python scrape_cosif_advertising.py --since 202501 --until 202603
  python scrape_cosif_advertising.py --out-dir <dir>      # write somewhere else (smoke runs)
"""

from __future__ import annotations

import argparse
import io
import logging
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from utils import paths

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

COSIF_RAW = paths.COSIF_RAW
OUT_DIR = paths.AWARENESS_PROC

ACCOUNTS = {"8174500005": "adv", "8174200006": "promo", "8174800004": "publ",
            "8170000004": "admin"}
FIELDS = tuple(ACCOUNTS.values())
AD_FIELDS = ("adv", "promo", "publ")
ADMIN_TOTAL = "8170000004"
FIRST_LEVEL4_MONTH = "202501"

FILE_TYPES = {"BANCOS": ("4010", "institution"), "SOCIEDADES": ("4010", "institution"),
              "CONSORCIOS": ("4010", "institution"), "COOPERATIVAS": ("4010", "institution"),
              "LIQUIDACAO": ("4010", "institution"), "BLOPRUDENCIAL": ("4060", "conglomerate")}
FILE_RE = re.compile(r"^(\d{6})(" + "|".join(FILE_TYPES) + r")\.(csv\.zip|zip|csv)$",
                     re.IGNORECASE)

# Advertising reporters per month, by file type. Measured over 2025-01 to 2026-03: BANCOS
# 96-116, SOCIEDADES 270-351, CONSORCIOS 94-106, COOPERATIVAS 523-583, LIQUIDACAO 0.
REPORTER_BOUNDS = {"BANCOS": (80, 200), "SOCIEDADES": (200, 500), "CONSORCIOS": (60, 160),
                   "COOPERATIVAS": (400, 800), "LIQUIDACAO": (0, 20),
                   "BLOPRUDENCIAL": (100, 250)}
ANCHORS = {"C0080099": "ITAU", "C0080738": "CAIXA", "C0080329": "BB"}
MAX_REVERSAL_SHARE = 0.05
SPIKE_MULTIPLE = 5.0
SPIKE_MIN_BRL = 10e6
KEYS = ["level", "entity_key"]
QUARTER_ENDS = {1: (3, None), 2: (6, 3), 3: (9, None), 4: (12, 9)}


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def _raw_bytes(path: Path) -> bytes:
    if path.name.lower().endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            return zf.open(zf.namelist()[0]).read()
    return path.read_bytes()


def _generation_date(path: Path) -> str:
    """'Data de geracao dos dados: YYYY-MM-DD' from the preamble, or '' if absent."""
    head = _raw_bytes(path)[:2000].decode("latin1")
    m = re.search(r"geracao dos dados:\s*([0-9-]+)", head, re.IGNORECASE)
    return m.group(1) if m else ""


def read_balancete(path: Path) -> pd.DataFrame:
    text = _raw_bytes(path).decode("latin1")
    lines = text.splitlines()
    header = next((i for i, line in enumerate(lines[:20]) if line.lstrip().startswith("#")), None)
    if header is None:
        raise ValueError(f"{path.name}: no '#'-prefixed column line in the first 20 lines")
    df = pd.read_csv(io.StringIO(text), sep=";", dtype=str, skiprows=header)
    df.columns = [c.replace("#", "").strip() for c in df.columns]
    required = {"DATA_BASE", "DOCUMENTO", "CNPJ", "NOME_INSTITUICAO", "COD_CONGL",
                "NOME_CONGL", "TAXONOMIA", "CONTA", "SALDO"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    return df


def parse_saldo(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
                         errors="raise")


def select_vintages(root: Path, since: str, until: str) -> dict[tuple[str, str], Path]:
    """One file per (yyyymm, file type): the latest generation date among duplicates."""
    candidates: dict[tuple[str, str], list[Path]] = {}
    for p in root.iterdir():
        m = FILE_RE.match(p.name)
        if m and since <= m.group(1) <= until:
            candidates.setdefault((m.group(1), m.group(2).upper()), []).append(p)
    chosen = {}
    for key, files in sorted(candidates.items()):
        if len(files) == 1:
            chosen[key] = files[0]
            continue
        dated = sorted(((_generation_date(f), f.name, f) for f in files), reverse=True)
        chosen[key] = dated[0][2]
        log.info("  %s %s: %d copies on disk, using %s (generated %s)", key[0], key[1],
                 len(files), dated[0][1], dated[0][0] or "unknown")
    return chosen


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def extract_month(path: Path, file_type: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(account rows, filers) for one balancete file."""
    document, level = FILE_TYPES[file_type]
    df = read_balancete(path)
    df = df[df["DOCUMENTO"].str.strip() == document].copy()
    if level == "conglomerate":
        df["entity_key"] = df["COD_CONGL"].fillna("").str.strip()
        df["entity_name"] = df["NOME_CONGL"].fillna("").str.strip()
    else:
        df["entity_key"] = df["CNPJ"].fillna("").str.strip().str.zfill(8)
        df["entity_name"] = df["NOME_INSTITUICAO"].fillna("").str.strip()
    if (df["entity_key"].str.strip("0") == "").any() and level == "conglomerate":
        raise ValueError(f"{path.name}: blank COD_CONGL on document {document} rows")

    group = df[df["CONTA"].str.match(r"^817\d{7}$", na=False)].copy()
    group["v"] = parse_saldo(group["SALDO"])

    # Check 3: the level-4 children of 8.1.7 sum to the level-3 total for every filer.
    total = group[group["CONTA"] == ADMIN_TOTAL].groupby("entity_key")["v"].sum()
    level4 = group[(group["CONTA"] != ADMIN_TOTAL) & (group["CONTA"].str[5:9] == "0000")]
    children = level4.groupby("entity_key")["v"].sum().reindex(total.index, fill_value=0.0)
    gap = (children - total).abs() > np.maximum(1.0, 1e-9 * total.abs())
    if gap.any():
        raise AssertionError(f"{path.name}: level-4 children of 8.1.7 do not sum to the total "
                             f"for {int(gap.sum())} entities, e.g. {list(total[gap].index[:5])}")
    orphan = set(level4["entity_key"]) - set(total.index)
    if orphan:
        raise AssertionError(f"{path.name}: level-4 8.1.7 rows without a level-3 total for "
                             f"{sorted(orphan)[:5]}")

    rows = group[group["CONTA"].isin(ACCOUNTS)].copy()
    rows["field"] = rows["CONTA"].map(ACCOUNTS)
    filers = (df.groupby("entity_key", as_index=False)
                .agg(entity_name=("entity_name", "first"),
                     cnpj_leader=("CNPJ", "first"),
                     taxonomia=("TAXONOMIA", "first")))
    return rows[["entity_key", "field", "v"]], filers


def build_monthly(since: str, until: str) -> pd.DataFrame:
    """Entity x month balances (positive reais) for every entity that filed that month."""
    files = select_vintages(COSIF_RAW, since, until)
    if not files:
        raise SystemExit(f"No BANCOS/BLOPRUDENCIAL files between {since} and {until} "
                         f"under {COSIF_RAW}")
    frames, filer_frames = [], []
    for (ym, file_type), path in sorted(files.items()):
        rows, filers = extract_month(path, file_type)
        level = FILE_TYPES[file_type][1]
        dup = rows.duplicated(["entity_key", "field"])
        if dup.any():                                                        # check 7
            raise AssertionError(f"{path.name}: {int(dup.sum())} duplicate entity-account rows")
        n_adv = rows.loc[rows["field"] == "adv", "entity_key"].nunique()
        lo, hi = REPORTER_BOUNDS[file_type]                                  # check 5
        if not lo <= n_adv <= hi:
            raise AssertionError(f"{ym} {file_type}: {n_adv} advertising reporters outside "
                                 f"[{lo}, {hi}]")
        for frame in (rows, filers):
            frame["level"] = level
            frame["data_base"] = ym
        frames.append(rows)
        filer_frames.append(filers)
        log.info("  %s %-13s %-28s %4d filers, %3d advertising reporters",
                 ym, file_type, path.name, len(filers), n_adv)

    long = pd.concat(frames, ignore_index=True)
    filers = pd.concat(filer_frames, ignore_index=True)

    twice = filers.duplicated(KEYS + ["data_base"], keep=False)             # check 7
    if twice.any():
        raise AssertionError("an entity files in two file types in the same month:\n"
                             + filers[twice].head(8)[KEYS + ["data_base", "entity_name"]]
                             .to_string(index=False))

    if (long["v"] >= 0).any():                                               # check 2
        bad = long[long["v"] >= 0].head(5).to_dict("records")
        raise AssertionError(f"non-negative raw balances in expense accounts: {bad}")

    wide = (long.set_index(KEYS + ["data_base", "field"])["v"]
                .unstack("field").reset_index())
    wide.columns.name = None
    panel = filers.merge(wide, on=KEYS + ["data_base"], how="left")
    for f in FIELDS:
        if f not in panel.columns:
            panel[f] = np.nan
        panel[f] = -panel[f].fillna(0.0)          # filer without the account -> true zero
    return panel.rename(columns={f: f"{f}_bal" for f in FIELDS})


def assert_no_level4_in_2024() -> None:
    """Check 1: the level-4 advertising accounts must not appear in the 2024 files on disk."""
    files = select_vintages(COSIF_RAW, "202401", "202412")
    if not files:
        log.warning("no 2024 COSIF files on disk; skipping the pre-2025 absence check")
        return
    codes = [k for k, f in ACCOUNTS.items() if f in AD_FIELDS]
    for (ym, file_type), path in sorted(files.items()):
        hits = int(read_balancete(path)["CONTA"].isin(codes).sum())
        if hits:
            raise AssertionError(f"{path.name}: {hits} level-4 advertising rows before "
                                 f"{FIRST_LEVEL4_MONTH}; the coverage premise is wrong")
    log.info("pre-2025 absence check passed on %d files from 2024", len(files))


# ---------------------------------------------------------------------------
# Flows
# ---------------------------------------------------------------------------
def _calendar(df: pd.DataFrame) -> pd.DataFrame:
    df["year"] = df["data_base"].str[:4].astype(int)
    df["month"] = df["data_base"].str[4:].astype(int)
    df["half"] = np.where(df["month"] <= 6, 1, 2)
    df["quarter"] = (df["month"] - 1) // 3 + 1
    return df


def monthly_flows(panel: pd.DataFrame) -> pd.DataFrame:
    """Complete entity x month grid with balances, filed flag and monthly flows."""
    months = [m.strftime("%Y%m") for m in pd.period_range(
        pd.Period(panel["data_base"].min(), "M"), pd.Period(panel["data_base"].max(), "M"),
        freq="M")]
    entities = panel[KEYS].drop_duplicates()
    grid = entities.merge(pd.DataFrame({"data_base": months}), how="cross")
    p = grid.merge(panel.assign(filed=True), on=KEYS + ["data_base"], how="left")
    p["filed"] = p["filed"].eq(True)
    p = _calendar(p).sort_values(KEYS + ["year", "month"]).reset_index(drop=True)
    for col in ("entity_name", "cnpj_leader", "taxonomia"):
        p[col] = p.groupby(KEYS)[col].transform(lambda s: s.ffill().bfill())

    starts_half = p["month"].isin([1, 7]).to_numpy()
    same_entity = ((p["entity_key"] == p["entity_key"].shift())
                   & (p["level"] == p["level"].shift())).to_numpy()
    for f in FIELDS:
        bal = p[f"{f}_bal"].to_numpy()
        prev = p[f"{f}_bal"].shift().to_numpy()
        p[f"{f}_flow"] = np.where(starts_half, bal, np.where(same_entity, bal - prev, np.nan))
    p["is_reversal"] = (p[[f"{f}_flow" for f in AD_FIELDS]] < -1.0).any(axis=1)

    # Spikes are flagged, never altered: a month whose adv flow exceeds SPIKE_MULTIPLE times
    # the entity's median positive monthly adv flow, and exceeds it by at least SPIKE_MIN_BRL.
    median_flow = (p["adv_flow"].where(p["adv_flow"] > 0)
                   .groupby([p["level"], p["entity_key"]]).transform("median"))
    p["adv_flow_spike"] = ((p["adv_flow"] > SPIKE_MULTIPLE * median_flow)
                           & (p["adv_flow"] - median_flow > SPIKE_MIN_BRL))
    return p


def quarterly_flows(monthly: pd.DataFrame) -> pd.DataFrame:
    """Entity x quarter flows from quarter-end balances."""
    ends = (monthly[monthly["month"].isin([3, 6, 9, 12])]
            .set_index(KEYS + ["year", "month"])[[f"{f}_bal" for f in FIELDS]]
            .unstack("month"))
    parts = []
    for quarter, (end, start) in QUARTER_ENDS.items():
        q = pd.DataFrame(index=ends.index)
        for f in FIELDS:
            col = f"{f}_bal"
            e = ends[(col, end)] if (col, end) in ends.columns else np.nan
            s = 0.0 if start is None else (ends[(col, start)] if (col, start) in ends.columns
                                          else np.nan)
            q[f] = e - s
        q["quarter"] = quarter
        parts.append(q.reset_index())
    qf = pd.concat(parts, ignore_index=True)

    info = (monthly.groupby(KEYS + ["year", "quarter"])
                   .agg(n_months_filed=("filed", "sum"),
                        entity_name=("entity_name", "first"),
                        cnpj_leader=("cnpj_leader", "first"),
                        taxonomia=("taxonomia", "first"),
                        any_reversal=("is_reversal", "any"),
                        any_adv_spike=("adv_flow_spike", "any"))
                   .reset_index())
    qf = info.merge(qf, on=KEYS + ["year", "quarter"], how="left")
    last = monthly["data_base"].max()
    qf = qf[(qf["year"] * 100 + qf["quarter"] * 3) <= int(last)].copy()
    qf["all3"] = qf[list(AD_FIELDS)].sum(axis=1, min_count=len(AD_FIELDS))
    admin = qf["admin"].where(qf["admin"] > 0)
    qf["adv_share_admin"] = qf["adv"] / admin
    qf["all3_share_admin"] = qf["all3"] / admin
    return qf.sort_values(KEYS + ["year", "quarter"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Remaining validation
# ---------------------------------------------------------------------------
def validate(monthly: pd.DataFrame, quarterly: pd.DataFrame) -> None:
    filed = monthly[monthly["filed"]]

    for level in ("institution", "conglomerate"):                            # check 4
        x = filed[(filed["level"] == level) & (filed["adv_bal"] > 0)]
        for year in sorted(x["year"].unique()):
            jun = x[(x["year"] == year) & (x["month"] == 6)]["adv_bal"]
            jul = x[(x["year"] == year) & (x["month"] == 7)]["adv_bal"]
            if len(jun) and len(jul) and not jul.median() < jun.median():
                raise AssertionError(f"{level} {year}: median July adv balance "
                                     f"{jul.median():,.0f} is not below June "
                                     f"{jun.median():,.0f}; the half-year reset changed")

    names = (filed[filed["level"] == "conglomerate"]
             .drop_duplicates("entity_key").set_index("entity_key")["entity_name"])
    for code, token in ANCHORS.items():                                      # check 6
        if code not in names.index or token not in str(names[code]).upper():
            raise AssertionError(f"identity anchor failed: {code} should contain {token}, "
                                 f"got {names.get(code)!r}")

    pairs = filed[~filed["month"].isin([1, 7])]                              # check 8
    for f in AD_FIELDS:
        x = pairs[f"{f}_flow"].dropna()
        share = float((x < -1.0).mean()) if len(x) else 0.0
        log.info("  within-half falls of the %s running total: %.2f%% of %d month pairs",
                 f, 100 * share, len(x))
        if share > MAX_REVERSAL_SHARE:
            raise AssertionError(f"{f}: {share:.1%} of month pairs fall, above "
                                 f"{MAX_REVERSAL_SHARE:.0%}")

    sums = (monthly.groupby(KEYS + ["year", "quarter"])[[f"{f}_flow" for f in FIELDS]]
                   .sum(min_count=3).reset_index())                          # check 9
    j = quarterly[quarterly["n_months_filed"] == 3].merge(sums, on=KEYS + ["year", "quarter"])
    for f in FIELDS:
        diff = (j[f] - j[f"{f}_flow"]).abs()
        tol = np.maximum(0.01, 1e-9 * j[f].abs())
        if (diff > tol).any():
            raise AssertionError(f"{f}: monthly flows do not sum to the quarter flow for "
                                 f"{int((diff > tol).sum())} complete entity-quarters")
    log.info("validation passed: %d complete entity-quarters reconciled", len(j))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="COSIF advertising accounts, 2025 onward")
    ap.add_argument("--since", default=FIRST_LEVEL4_MONTH, help="first data-base, YYYYMM")
    ap.add_argument("--until", default="209912", help="last data-base, YYYYMM")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    a = ap.parse_args()
    if a.since < FIRST_LEVEL4_MONTH:
        raise SystemExit(f"--since {a.since}: the level-4 accounts are published only from "
                         f"{FIRST_LEVEL4_MONTH}")

    assert_no_level4_in_2024()
    log.info("reading balancetes %s-%s from %s", a.since, a.until, COSIF_RAW)
    monthly = monthly_flows(build_monthly(a.since, a.until))
    quarterly = quarterly_flows(monthly)
    for frame in (monthly, quarterly):
        frame["panel_key"] = np.where(frame["level"] == "conglomerate", frame["entity_key"],
                                      "CNPJ_" + frame["entity_key"])
    validate(monthly, quarterly)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (("cosif_advertising_monthly", monthly),
                        ("cosif_advertising_quarterly", quarterly)):
        frame.to_parquet(a.out_dir / f"{name}.parquet", index=False)
        frame.to_csv(a.out_dir / f"{name}.csv", index=False)
        log.info("%s: %d rows -> %s", name, len(frame), a.out_dir / f"{name}.parquet")

    spikes = monthly[monthly["adv_flow_spike"]]
    if len(spikes):
        log.warning("%d entity-months flagged adv_flow_spike (kept unaltered):\n%s", len(spikes),
                    spikes.assign(adv_flow_brl_m=(spikes["adv_flow"] / 1e6).round(1),
                                  admin_flow_brl_m=(spikes["admin_flow"] / 1e6).round(1))
                          [["level", "entity_key", "entity_name", "data_base",
                            "adv_flow_brl_m", "admin_flow_brl_m"]].to_string(index=False))

    complete = quarterly[quarterly["n_months_filed"] == 3]
    summary =(complete.groupby(["level", "year", "quarter"])
                       .agg(entities=("entity_key", "nunique"),
                            adv_brl_m=("adv", "sum"), all3_brl_m=("all3", "sum")))
    summary[["adv_brl_m", "all3_brl_m"]] = (summary[["adv_brl_m", "all3_brl_m"]] / 1e6).round(1)
    log.info("quarterly totals over complete entity-quarters (R$ million):\n%s",
             summary.to_string())


if __name__ == "__main__":
    main()

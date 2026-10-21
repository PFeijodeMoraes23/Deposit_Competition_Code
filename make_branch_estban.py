"""
make_branch_estban.py
=====================
ESTBAN by branch (agencia) for B firms: feasibility (brief K2) and descriptive keys (K3).

Source: DATA_RAW/estban_agencia/YYYYMM_ESTBAN_AG.{ZIP,csv.zip,csv} (BCB, "ESTBAN (Documento
4500) por agencia"), read directly and never modified. One row per branch and month, the
municipal files' columns plus AGENCIA, the branch's 14-digit CNPJ (institution root, 4-digit
sequence, check digits), and CODMUN_IBGE. Deposit lines are named with the pipeline's own map
(utils.estban): checking dep_a1 = V400_401 (verbetes 401-419), savings dep_a2 = V420, time
dep_a4 = V432.

  --stage feasibility  2016-01..2024-12: identifier stability, MCA mapping, and the branch sums
                       against the municipal ESTBAN.csv by institution x municipality x month
                       -> Rout/branch_estban_feasibility.json
  --stage keys         B firms, quarter-end months 2016Q1..2024Q4 -> Rout/branch_estban.json
                       (desc.branch.* in make_paper_numbers.py)

Usage:  python make_branch_estban.py --stage feasibility | keys
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import argparse
import datetime
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from utils import paths as _paths
from utils.estban import _PRE2022_RENAME, _POST2022_RENAME, _rename_verbete_cols, _FORMAT_CHANGE_DATE
from make_xi_diagnostics import _stamp

RAW = _paths.OPEN_FINANCE / "DATA_RAW" / "estban_agencia"
MUNI = _paths.ESTBAN_CSV
XWALK = _paths.IBGE_DIR / "muni_mca_regions_2010_2024_panel.csv"
CACHE = _paths.rout_dir() / "branch_estban_cache"
FEAS = _paths.rout_dir() / "branch_estban_feasibility.json"
KEYS = _paths.rout_dir() / "branch_estban.json"
DEP = {"V400_401": "dep_a1", "V420": "dep_a2", "V432": "dep_a4"}
Y0, Y1 = 2016, 2024


def _files() -> dict:
    out = {}
    for p in sorted(RAW.iterdir()):
        m = re.match(r"^(\d{6})_ESTBAN_AG", p.name, re.I)
        if m and p.is_file():
            out.setdefault(int(m.group(1)), p)
    return out


def read_month(p: Path) -> pd.DataFrame:
    """One branch file: AGENCIA (string), CNPJ root, CODMUN_IBGE, the three deposit lines."""
    def _read(fh):
        return pd.read_csv(fh, sep=";", decimal=",", encoding="latin1", skiprows=2,
                           dtype={"AGENCIA": str, "CNPJ": str}, low_memory=False)
    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p) as z:
            with z.open(z.namelist()[0]) as fh:
                d = _read(fh)
    else:
        d = _read(p)
    d.columns = [c.strip().upper().replace(" ", "_").lstrip("#") for c in d.columns]
    ym = int(str(d["DATA_BASE"].dropna().iloc[0]).strip()[:6])
    date = pd.Timestamp(year=ym // 100, month=ym % 100, day=1)
    d = _rename_verbete_cols(d, _PRE2022_RENAME if date < _FORMAT_CHANGE_DATE else _POST2022_RENAME)
    out = pd.DataFrame({
        "ym": ym,
        "agencia": d["AGENCIA"].astype(str).str.replace("'", "", regex=False).str.strip().str.zfill(14),
        "cnpj": d["CNPJ"].astype(str).str.replace("'", "", regex=False).str.strip().str.zfill(8),
        "codmun_ibge": pd.to_numeric(d.get("CODMUN_IBGE"), errors="coerce").astype("Int64"),
    })
    for v, k in DEP.items():
        out[k] = pd.to_numeric(d[v], errors="coerce").fillna(0.0) if v in d.columns else np.nan
    return out


def load(refresh: bool = False) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"branches_{Y0}_{Y1}.parquet"
    files = {ym: p for ym, p in _files().items() if Y0 * 100 < ym < (Y1 + 1) * 100}
    key = {str(ym): _stamp(p)["size"] for ym, p in files.items()}
    meta = CACHE / f"branches_{Y0}_{Y1}.json"
    if cache.is_file() and meta.is_file() and not refresh and json.loads(meta.read_text()) == key:
        return pd.read_parquet(cache)
    d = pd.concat([read_month(p) for p in files.values()], ignore_index=True)
    d.to_parquet(cache)
    meta.write_text(json.dumps(key))
    return d


def feasibility(refresh: bool) -> dict:
    d = load(refresh)
    months = sorted(d["ym"].unique())
    out = dict(months=dict(n=len(months), first=int(months[0]), last=int(months[-1]),
                           missing=[int(m) for m in pd.period_range(f"{Y0}-01", f"{Y1}-12", freq="M")
                                    .strftime("%Y%m").astype(int) if m not in set(months)]),
               rows=int(len(d)), rows_per_month=dict(min=int(d.groupby("ym").size().min()),
                                                     max=int(d.groupby("ym").size().max())))
    # Identifier: AGENCIA is the branch CNPJ; its first eight digits should be the CNPJ root.
    out["id"] = dict(n_branches=int(d["agencia"].nunique()),
                     root_matches_cnpj=float((d["agencia"].str[:8] == d["cnpj"]).mean()),
                     dup_branch_month=int(d.duplicated(["ym", "agencia"]).sum()))
    # Breaks: a branch that leaves and comes back, or that changes municipality.
    idx = {m: i for i, m in enumerate(months)}
    g = d.assign(i=d["ym"].map(idx)).groupby("agencia")["i"].agg(["min", "max", "nunique"])
    out["id"]["gappy_share"] = float(((g["max"] - g["min"] + 1) > g["nunique"]).mean())
    out["id"]["moves_municipality"] = int((d.groupby("agencia")["codmun_ibge"].nunique() > 1).sum())
    # MCA mapping.
    xw = pd.read_csv(XWALK, usecols=["municipality_code", "year", "mca_code"])
    xw = xw.sort_values("year").drop_duplicates("municipality_code", keep="last")
    mapped = d["codmun_ibge"].isin(set(xw["municipality_code"]))
    out["mca"] = dict(share_rows_mapped=float(mapped.mean()),
                      share_dep_mapped=float(d.loc[mapped, ["dep_a1", "dep_a2", "dep_a4"]].sum().sum()
                                             / d[["dep_a1", "dep_a2", "dep_a4"]].sum().sum()),
                      missing_code_rows=int(d["codmun_ibge"].isna().sum()))
    # Reconciliation against the municipal ESTBAN (institution x municipality x month).
    mu = pd.read_csv(MUNI, usecols=["CNPJ", "CODMUN_IBGE", "YEAR", "MONTH", *DEP],
                     dtype={"CNPJ": str}, low_memory=False, encoding="latin-1")
    mu = mu[mu["YEAR"].between(Y0, Y1)]
    mu["ym"] = mu["YEAR"] * 100 + mu["MONTH"]
    mu["cnpj"] = mu["CNPJ"].str.strip().str.zfill(8)
    mu["codmun_ibge"] = pd.to_numeric(mu["CODMUN_IBGE"], errors="coerce").astype("Int64")
    mu = mu.rename(columns=DEP).groupby(["ym", "cnpj", "codmun_ibge"], as_index=False)[list(DEP.values())].sum()
    br = d.groupby(["ym", "cnpj", "codmun_ibge"], as_index=False)[list(DEP.values())].sum()
    m = mu.merge(br, on=["ym", "cnpj", "codmun_ibge"], how="outer", suffixes=("_mun", "_br"),
                 indicator=True)
    rec = dict(cells=int(len(m)), only_municipal=int((m["_merge"] == "left_only").sum()),
               only_branch=int((m["_merge"] == "right_only").sum()))
    both = m[m["_merge"] == "both"]
    for k in DEP.values():
        a, b = both[f"{k}_mun"].astype(float), both[f"{k}_br"].astype(float)
        gap = (b - a).abs()
        rel = gap / a.abs().where(a.abs() > 0)
        rec[k] = dict(exact_share=float((gap < 0.5).mean()),
                      rel_gap_q=dict(zip(["p50", "p90", "p99", "max"],
                                         [float(x) for x in rel.quantile([0.5, 0.9, 0.99, 1.0]).fillna(0)])),
                      total_gap_share=float(gap.sum() / a.abs().sum()),
                      only_municipal_balance_share=float(
                          m.loc[m["_merge"] == "left_only", f"{k}_mun"].abs().sum()
                          / m[f"{k}_mun"].abs().sum()))
    out["reconciliation"] = rec
    out["_meta"] = dict(written=datetime.datetime.now().isoformat(timespec="seconds"),
                        raw=str(RAW), municipal=_stamp(MUNI), years=[Y0, Y1])
    return out


SANTANDER = "C0080185"           # its sweep is a default account feature from 2022-05
SWEEP_T = 2022 * 4 + 1            # 2022Q2 on t = 4 * year + quarter - 1


def keys(refresh: bool) -> dict:
    """B firms' branches at quarter-end months, 2016Q1-2024Q4 (2015Q4 read only to classify the
    first quarter). Branch = AGENCIA. Openings: a branch's first quarter-end in the window
    (branches present in 2015Q4 never open). Closings: the quarter after its last quarter-end,
    when that is in the window. A branch that skips quarters counts once. Time-deposit change of
    a bank (conglomerate) in an MCA from one quarter-end to the next, split into existing
    branches (present at both), new branches (level at t) and closed branches (minus the level
    at t-1); share_existing = sum |existing| / sum (|existing| + |new| + |closed|), over all bank
    x MCA x quarter cells. corr_dcheck_dtime: correlation of the quarterly change in checking and
    in time deposits within branch (both demeaned by branch), over branches present at both
    quarter-ends."""
    from utils import conglomerates as cc
    d = load(refresh)
    d = d[(d["ym"] % 100) % 3 == 0]
    files = _files()
    d = pd.concat([read_month(files[201512]), d], ignore_index=True)
    d["t"] = (d["ym"] // 100) * 4 + (d["ym"] % 100) // 3 - 1
    # Institution root -> conglomerate (IF-Data List as-of merge), B conglomerates of the panel.
    uniq = d[["cnpj", "ym"]].drop_duplicates().rename(columns={"cnpj": "CNPJ", "ym": "AnoMes"})
    cmap = cc.map_foundation_to_congl(uniq, cc.build_cnpj_to_congl())[["CNPJ", "AnoMes",
                                                                       "CodConglomeradoPrudencial"]]
    d = d.merge(cmap.rename(columns={"CNPJ": "cnpj", "AnoMes": "ym", "CodConglomeradoPrudencial": "bank"}),
                on=["cnpj", "ym"], how="left")
    panel = pd.read_parquet(_paths.PROCESSED / "market_panel.parquet",
                            columns=["CodConglomeradoPrudencial", "is_B"])
    b_banks = set(panel.loc[panel["is_B"] == 1, "CodConglomeradoPrudencial"].astype(str))
    n_rows = len(d)
    t0, t1 = Y0 * 4, Y1 * 4 + 3
    # An acquisition renumbers the target's branches under the acquirer's CNPJ: a quarter in
    # which one institution closes and another opens 100 or more branches is a transfer
    # (Bradesco's purchase of HSBC Brasil, 2016Q4). Found on every institution, before the B
    # filter, since the target need not be a B firm of the panel. Transferred branches are
    # neither openings nor closings, and the two banks' cells of that quarter leave the
    # time-deposit decomposition.
    full = d.sort_values("t").groupby("agencia").agg(first=("t", "min"), last=("t", "max"),
                                                     cnpj_first=("cnpj", "first"),
                                                     cnpj_last=("cnpj", "last"))
    oq = full[full["first"].between(t0, t1)].groupby(["first", "cnpj_first"]).size()
    cq = full[(full["last"] + 1).between(t0, t1)].assign(q=lambda f: f["last"] + 1) \
        .groupby(["q", "cnpj_last"]).size()
    root_bank = d.dropna(subset=["bank"]).groupby("cnpj")["bank"].agg(lambda s: s.mode().iloc[0])
    transfers = []
    for (q, tgt), nc in cq[cq >= 100].items():
        for (q2, acq), no in oq[oq >= 100].items():
            if q2 == q and acq != tgt:
                transfers.append(dict(t=int(q), quarter=f"{int(q) // 4}Q{int(q) % 4 + 1}",
                                      target_cnpj=str(tgt), acquirer_cnpj=str(acq),
                                      target_bank=str(root_bank.get(tgt, "")),
                                      acquirer_bank=str(root_bank.get(acq, "")),
                                      closed=int(nc), opened=int(no)))
    d = d[d["bank"].astype(str).isin(b_banks)].copy()
    xw = pd.read_csv(XWALK, usecols=["municipality_code", "year", "mca_code"])
    xw = xw.sort_values("year").drop_duplicates("municipality_code", keep="last")
    d = d.merge(xw[["municipality_code", "mca_code"]], left_on="codmun_ibge",
                right_on="municipality_code", how="left")
    out = dict(rows_B_share=float(len(d) / n_rows), n_B_banks=int(d["bank"].nunique()))
    # Counts by year.
    life = d.sort_values("t").groupby("agencia").agg(first=("t", "min"), last=("t", "max"),
                                                     cnpj_first=("cnpj", "first"),
                                                     cnpj_last=("cnpj", "last"))
    opened = life[life["first"].between(t0, t1)]
    closed = life[(life["last"] + 1).between(t0, t1)].assign(q=lambda f: f["last"] + 1)
    tr_open = {(x["t"], x["acquirer_cnpj"]) for x in transfers}
    tr_close = {(x["t"], x["target_cnpj"]) for x in transfers}
    o_keep = ~pd.Series(list(zip(opened["first"], opened["cnpj_first"])), index=opened.index).isin(tr_open)
    c_keep = ~pd.Series(list(zip(closed["q"], closed["cnpj_last"])), index=closed.index).isin(tr_close)
    dec = d[(d["t"] % 4 == 3) & d["t"].between(t0, t1)]
    out["n_branches"] = {str(y): int(dec.loc[dec["t"] == y * 4 + 3, "agencia"].nunique())
                         for y in range(Y0, Y1 + 1)}
    out["n_open"] = {str(y): int(((opened["first"] // 4 == y) & o_keep).sum()) for y in range(Y0, Y1 + 1)}
    out["n_close"] = {str(y): int(((closed["q"] // 4 == y) & c_keep).sum()) for y in range(Y0, Y1 + 1)}
    out["transfers"] = transfers
    out["n_open_raw"] = {str(y): int((opened["first"] // 4 == y).sum()) for y in range(Y0, Y1 + 1)}
    out["n_close_raw"] = {str(y): int((closed["q"] // 4 == y).sum()) for y in range(Y0, Y1 + 1)}
    # Decomposition of the time-deposit change by bank x MCA x quarter.
    d = d.dropna(subset=["mca_code"])
    cur = d[["agencia", "bank", "mca_code", "t", "dep_a1", "dep_a4"]]
    prev = cur.assign(t=cur["t"] + 1)
    m = cur.merge(prev, on=["agencia", "bank", "mca_code", "t"], how="outer", suffixes=("", "_l"))
    m = m[m["t"].between(t0, t1)]
    ex = m["dep_a4"].notna() & m["dep_a4_l"].notna()
    new = m["dep_a4"].notna() & m["dep_a4_l"].isna()
    gone = m["dep_a4"].isna() & m["dep_a4_l"].notna()
    m["c_ex"] = np.where(ex, m["dep_a4"] - m["dep_a4_l"], 0.0)
    m["c_new"] = np.where(new, m["dep_a4"], 0.0)
    m["c_gone"] = np.where(gone, -m["dep_a4_l"], 0.0)
    cell = m.groupby(["bank", "mca_code", "t"])[["c_ex", "c_new", "c_gone"]].sum()
    drop = ({(x["acquirer_bank"], x["t"]) for x in transfers}
            | {(x["target_bank"], x["t"]) for x in transfers})
    bt = list(zip(cell.index.get_level_values("bank"), cell.index.get_level_values("t")))
    n_cells_all = len(cell)
    cell = cell[[k not in drop for k in bt]]
    gross = cell.abs().sum()

    def share(c):
        g = c.abs().sum()
        return float(g["c_ex"] / g.sum())
    out["share_existing_dtime"] = share(cell)
    out["share_existing_dtime_detail"] = dict(
        gross_existing=float(gross["c_ex"]), gross_new=float(gross["c_new"]),
        gross_closed=float(gross["c_gone"]), n_cells=int(len(cell)),
        n_cells_dropped_transfers=int(n_cells_all - len(cell)),
        cells_with_new_or_closed=float(((cell["c_new"] != 0) | (cell["c_gone"] != 0)).mean()),
        no_santander=share(cell[cell.index.get_level_values("bank") != SANTANDER]))
    # Within-branch correlation of the checking and time-deposit changes.
    e = m[ex].assign(dc=lambda f: f["dep_a1"] - f["dep_a1_l"], dt=lambda f: f["dep_a4"] - f["dep_a4_l"])

    def wcorr(f):
        x = f["dc"] - f.groupby("agencia")["dc"].transform("mean")
        y = f["dt"] - f.groupby("agencia")["dt"].transform("mean")
        return float(np.corrcoef(x, y)[0, 1]), int(len(f))
    c_all, n_all = wcorr(e)
    out["corr_dcheck_dtime"] = c_all
    out["corr_detail"] = dict(
        n=n_all, n_branches=int(e["agencia"].nunique()),
        no_santander=wcorr(e[e["bank"] != SANTANDER])[0],
        santander_before_sweep=wcorr(e[(e["bank"] == SANTANDER) & (e["t"] < SWEEP_T)])[0],
        santander_from_sweep=wcorr(e[(e["bank"] == SANTANDER) & (e["t"] >= SWEEP_T)])[0],
        all_before_2022q2=wcorr(e[e["t"] < SWEEP_T])[0],
        all_from_2022q2=wcorr(e[e["t"] >= SWEEP_T])[0])
    out["_meta"] = dict(written=datetime.datetime.now().isoformat(timespec="seconds"), raw=str(RAW),
                        window=f"{Y0}Q1-{Y1}Q4 quarter-end months (2015Q4 to classify 2016Q1)",
                        sweep_flag="Santander's sweep is a default account feature from 2022-05 "
                                   "(sweep_products.csv SANT-PF-4); every result here spans it")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", choices=["feasibility", "keys"], required=True)
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    if a.stage == "feasibility":
        r = feasibility(a.refresh)
        FEAS.write_text(json.dumps(r, indent=2, default=float), encoding="utf-8")
        print(json.dumps({k: v for k, v in r.items() if k != "_meta"}, indent=1, default=float)[:4000])
        print(f"  wrote {FEAS}")
    else:
        r = keys(a.refresh)
        KEYS.write_text(json.dumps(r, indent=2, default=float), encoding="utf-8")
        print(json.dumps({k: v for k, v in r.items() if k != "_meta"}, indent=1, default=float))
        print(f"  wrote {KEYS}")


if __name__ == "__main__":
    main()

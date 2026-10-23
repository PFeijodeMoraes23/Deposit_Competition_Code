"""
make_tesouro_mca.py
===================
Tesouro Direto investor registrations by MCA and quarter, 2016Q1-2024Q4 (scraping task S8a).

Source: DATA_RAW/TesouroNacional/investidores_td/live_manual/
investidorestesourodireto{ate2017,2018..2026}.zip (Tesouro Nacional open data, downloaded by hand),
read directly and never modified. One row per investor and institution: Codigo do Investidor,
Data de Adesao, UF do Investidor, Cidade do Investidor (free text), Pais do Investidor, and the
account status at extraction (not a history, so not used).

  1. Each investor once: the earliest registration across institutions and files, with that
     record's state and city; investors abroad are left out.
  2. City to IBGE municipality: state plus the city name, upper case, accents and punctuation
     removed, against the official municipality names of the BCB Pix statistics by municipality
     (DATA_RAW/BCB/pix_municipal); the state of a municipality code is
     its first two digits. The match rate is reported, unmatched names listed.
  3. Municipality to MCA with the panel's IBGE crosswalk (each municipality's latest MCA).
  4. Per MCA and quarter: new investors (first registration in the quarter) and the stock of
     registered investors (first registration at or before the quarter's end, from 2002), and
     both per 1,000 inhabitants with the panel's population.

Outputs: Rout/tesouro_investors_mca_quarter.parquet, Rout/tesouro_investors_mca.json (coverage).
Usage:  python make_tesouro_mca.py
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import datetime
import json
import re
import unicodedata
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from utils import paths as _paths
from utils import raw_paths
from make_xi_diagnostics import _stamp

RAW = raw_paths.TESOURO_INVESTIDORES_LIVE
PIX = raw_paths.BCB_PIX_MUNICIPAL_RAW
XWALK = _paths.IBGE_DIR / "muni_mca_regions_2010_2024_panel.csv"
PANEL = _paths.PROCESSED / "market_panel.parquet"
OUT = _paths.rout_dir() / "tesouro_investors_mca_quarter.parquet"
SIDE = _paths.rout_dir() / "tesouro_investors_mca.json"
Y0, Y1 = 2016, 2024
COLS = {"Codigo do Investidor": "code", "Data de Adesao": "date", "UF do Investidor": "uf",
        "Cidade do Investidor": "city", "Pais do Investidor": "country"}


def norm(s: pd.Series) -> pd.Series:
    """Upper case, accents removed, punctuation to spaces, single spaces."""
    def one(x: str) -> str:
        x = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode()
        x = re.sub(r"[^A-Z0-9 ]", " ", x.upper())
        return re.sub(r"\s+", " ", x).strip()
    u = pd.Series(s.unique())
    return s.map(dict(zip(u, u.map(one))))


def investors() -> pd.DataFrame:
    """Each investor once, at the earliest registration across institutions and files. Cached on
    the zips' sizes (Rout/tesouro_cache/investors.parquet)."""
    cache = _paths.rout_dir() / "tesouro_cache" / "investors.parquet"
    key = {p.name: p.stat().st_size for p in sorted(RAW.glob("investidorestesourodireto*.zip"))}
    meta = cache.with_suffix(".json")
    if cache.is_file() and meta.is_file() and json.loads(meta.read_text()) == key:
        return pd.read_parquet(cache)
    parts = []
    for p in sorted(RAW.glob("investidorestesourodireto*.zip")):
        with zipfile.ZipFile(p) as z:
            with z.open(z.namelist()[0]) as fh:
                d = pd.read_csv(fh, sep=";", usecols=list(COLS), dtype=str, encoding="utf-8",
                                encoding_errors="replace").rename(columns=COLS)
        d["date"] = pd.to_datetime(d["date"], format="%d/%m/%Y", errors="coerce")
        d = d.dropna(subset=["code", "date"])
        d = d.sort_values("date").drop_duplicates("code", keep="first")
        parts.append(d)
        print(f"  {p.name}: {len(d):,} investors")
    d = pd.concat(parts, ignore_index=True).sort_values("date").drop_duplicates("code", keep="first")
    d = d.reset_index(drop=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(cache)
    cache.with_suffix(".json").write_text(json.dumps(key))
    return d


def municipality_names() -> pd.DataFrame:
    """(state, normalized name) -> IBGE code, from the Pix files' official names (all months,
    so codes created during the period are included)."""
    frames = []
    for p in sorted(PIX.glob("pix_mun_*.csv")):
        d = pd.read_csv(p, usecols=["Municipio_Ibge", "Municipio"], dtype=str)
        frames.append(d)
    m = pd.concat(frames).dropna().drop_duplicates()
    m["ibge"] = pd.to_numeric(m["Municipio_Ibge"], errors="coerce").astype("Int64")
    m = m.dropna(subset=["ibge"])
    xw = pd.read_csv(XWALK, usecols=["state_code", "state_uf"]).drop_duplicates()
    st = dict(zip(xw["state_code"], xw["state_uf"]))
    m["uf"] = (m["ibge"] // 100000).astype(int).map(st)
    m["name"] = norm(m["Municipio"])
    m = m.drop_duplicates(["uf", "name", "ibge"])
    dup = m.duplicated(["uf", "name"], keep=False)
    return m[~dup][["uf", "name", "ibge"]]


def main():
    inv = investors()
    n_all = len(inv)
    abroad = inv["country"].fillna("").str.upper().str.strip().ne("BRASIL")
    inv = inv[~abroad].copy()
    inv["uf"] = inv["uf"].str.strip().str.upper()
    inv["name"] = norm(inv["city"].fillna(""))
    names = municipality_names()
    inv = inv.merge(names, on=["uf", "name"], how="left")
    hit = inv["ibge"].notna()
    win = inv["date"].dt.year.between(Y0, Y1)
    match_all, match_win = float(hit.mean()), float(hit[win].mean())
    unmatched = (inv.loc[~hit].groupby(["uf", "name"]).size().sort_values(ascending=False)
                 .head(40).reset_index(name="n"))
    xw = pd.read_csv(XWALK, usecols=["municipality_code", "year", "mca_code"])
    xw = xw.sort_values("year").drop_duplicates("municipality_code", keep="last")
    xw["mca_code"] = xw["mca_code"].astype(str)
    inv = inv.merge(xw[["municipality_code", "mca_code"]], left_on="ibge",
                    right_on="municipality_code", how="left")
    inv = inv.dropna(subset=["mca_code"])
    inv["t"] = inv["date"].dt.year * 4 + (inv["date"].dt.month - 1) // 3
    # New investors by MCA x quarter, and the stock from the first registration on.
    new = inv.groupby(["mca_code", "t"]).size().rename("new_investors")
    t0, t1 = Y0 * 4, Y1 * 4 + 3
    grid = pd.MultiIndex.from_product([sorted(inv["mca_code"].unique()), range(int(inv["t"].min()), t1 + 1)],
                                      names=["mca_code", "t"])
    q = new.reindex(grid, fill_value=0).reset_index()
    q["stock_investors"] = q.groupby("mca_code")["new_investors"].cumsum()
    q = q[q["t"].between(t0, t1)].copy()
    q["year"], q["quarter"] = q["t"] // 4, q["t"] % 4 + 1
    pop = pd.read_parquet(PANEL, columns=["mca_code", "is_B", "year", "quarter", "pop_total"])
    pop = pop[pop["is_B"] == 1].astype({"mca_code": str})
    pop = pop.groupby(["mca_code", "year", "quarter"], as_index=False)["pop_total"].first()
    q = q.merge(pop, on=["mca_code", "year", "quarter"], how="left")
    q["new_per1000"] = 1000 * q["new_investors"] / q["pop_total"]
    q["stock_per1000"] = 1000 * q["stock_investors"] / q["pop_total"]
    q = q.drop(columns="t")
    q.to_parquet(OUT)
    in_win = inv["date"].dt.year.between(Y0, Y1)
    side = dict(
        investors=n_all, abroad=int(abroad.sum()), domestic=int((~abroad).sum()),
        matched_share=match_all, matched_share_2016_2024=match_win,
        mapped_to_mca=int(len(inv)), new_in_window=int(in_win.sum()),
        n_mca=int(q["mca_code"].nunique()), n_quarters=int(q[["year", "quarter"]].drop_duplicates().shape[0]),
        pop_missing_cells=int(q["pop_total"].isna().sum()),
        top_unmatched=unmatched.to_dict(orient="records"),
        _meta=dict(written=datetime.datetime.now().isoformat(timespec="seconds"), raw=str(RAW),
                   files={p.name: _stamp(p)["size"] for p in sorted(RAW.glob("*.zip"))}))
    SIDE.write_text(json.dumps(side, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in side.items() if k not in ("_meta", "top_unmatched")}, indent=1))
    print("  top unmatched:", unmatched.head(12).to_dict(orient="records"))
    print(f"  wrote {OUT}\n  wrote {SIDE}")


if __name__ == "__main__":
    main()

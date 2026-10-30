"""
diag_tesouro_vs_b3.py
=====================
Does the Tesouro Direto (TD) investor registry rank Brazilian states the way B3 does? (task TD1)

Inputs (read only):
  - Rout/tesouro_cache/investors.parquet, built by make_tesouro_mca.py: each investor once, earliest
    registration date, state (uf), city. Investors abroad are excluded. The state is the investor-level
    uf, because one municipality-to-MCA cell crosses state lines in the IBGE crosswalk; the national
    sum of Rout/tesouro_investors_mca_quarter.parquet is reported next to it as a cross-check.
  - DATA_RAW/B3/mub3_acervo/compiled/mub3_state_tables.csv: thousand CPFs with a position, 27 states,
    2019-12 and 2020-03.
  - DATA_RAW/B3/perfil_pf/wayback/GetOpenState_20230904222452.json: custody accounts, 27 states, 2023-09.
  - DATA_RAW/B3/ri_banco_de_dados/BD_Operational Figures_dl20261027.xlsx: national monthly series
    (Tesouro Direto investors, unique CPFs).
  - IBGE/muni_demographics_panel.csv: total population by municipality and year (no adult-only
    population is held), summed to states by the first two digits of the municipality code.
Outputs: Rout/tesouro_vs_b3_state.csv (state x date), Rout/tesouro_vs_b3_national.csv (month),
  and a summary printed to the screen.
Units differ across dates (CPFs in 2019-12 and 2020-03, accounts in 2023-09): never difference across them.
Usage:  python diagnostics/diag_tesouro_vs_b3.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import re

import numpy as np
import openpyxl
import pandas as pd

from utils import paths
from utils import raw_paths  # noqa: F401  (keeps the raw-path registry importable)

OF = paths.IBGE_DIR.parent
B3 = OF / "DATA_RAW" / "B3"
ROUT = paths.rout_dir()
DATES = {"2019-12": "2019-12-31", "2020-03": "2020-03-31", "2023-09": "2023-09-02"}
POP_YEAR = {"2019-12": 2019, "2020-03": 2020, "2023-09": 2023}
XLSX = B3 / "ri_banco_de_dados" / "BD_Operational Figures_dl20261027.xlsx"
PT_MONTH = {"jan": 1, "fev": 2, "mar": 3, "abr": 4, "mai": 5, "jun": 6, "jul": 7, "ago": 8,
            "set": 9, "out": 10, "nov": 11, "dez": 12}


def load_td() -> pd.DataFrame:
    """Domestic investors, one row each, with the first registration date and state."""
    inv = pd.read_parquet(ROUT / "tesouro_cache" / "investors.parquet",
                          columns=["code", "date", "uf", "country"])
    inv = inv[inv["country"].fillna("").str.upper().str.strip().eq("BRASIL")].copy()
    inv["uf"] = inv["uf"].str.strip().str.upper()
    return inv


def load_b3_states() -> pd.DataFrame:
    """B3 state values by date, with the unit."""
    m = pd.read_csv(B3 / "mub3_acervo" / "compiled" / "mub3_state_tables.csv")
    m = m[m["date"].isin(["2019-12", "2020-03"])]
    out = [pd.DataFrame({"uf": m["uf"], "date": m["date"], "b3_value": m["value"] * 1000.0,
                         "b3_unit": "CPFs with a position (thousands rounded, x1000)"})]
    j = json.loads((B3 / "perfil_pf" / "wayback" / "GetOpenState_20230904222452.json")
                   .read_text(encoding="utf-8"))
    a = pd.DataFrame(j["result"])
    out.append(pd.DataFrame({"uf": a["State"], "date": "2023-09",
                             "b3_value": a["TotalAccount"].str.replace(".", "", regex=False).astype(float),
                             "b3_unit": "custody accounts"}))
    return pd.concat(out, ignore_index=True)


def load_population() -> pd.DataFrame:
    """Total population by state and year (municipality population summed to states)."""
    xw = pd.read_csv(paths.IBGE_DIR / "muni_mca_regions_2010_2024_panel.csv",
                     usecols=["state_code", "state_uf"]).drop_duplicates()
    st = dict(zip(xw["state_code"], xw["state_uf"]))
    d = pd.read_csv(paths.IBGE_DIR / "muni_demographics_panel.csv",
                    usecols=["municipality_code", "year", "population"])
    d["uf"] = (d["municipality_code"] // 100000).map(st)
    return d.groupby(["uf", "year"], as_index=False)["population"].sum()


def ri_series() -> pd.DataFrame:
    """Monthly national series from the RI spreadsheet: TD investors and unique CPFs."""
    wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)
    out = {}
    dep = [s for s in wb.sheetnames if s.startswith("Dep")][0]
    rf = [s for s in wb.sheetnames if s.startswith("Renda Fixa")][0]

    def row(sheet, pred, hdr_pred):
        rows = list(wb[sheet].iter_rows(values_only=True))
        hdr = next(r for r in rows if hdr_pred(r))
        r = next(r for r in rows if r[0] and pred(str(r[0])))
        return hdr, r

    h, r = row(dep, lambda s: "nicos" in s and "Individuals" in s,
               lambda r: r[1] is not None and hasattr(r[1], "year"))
    out["b3_unique_cpfs"] = {(c.year, c.month): v for c, v in zip(h[1:], r[1:])
                             if hasattr(c, "year") and v is not None}
    h, r = row(rf, lambda s: s.startswith("# de investidores"),
               lambda r: isinstance(r[1], str) and re.match(r"[a-z]{3}/\d\d", r[1]) is not None)
    td = {}
    for c, v in zip(h[1:], r[1:]):
        if v is None:
            continue
        if isinstance(c, str) and re.match(r"[a-z]{3}/\d\d", c):
            td[(2000 + int(c[4:6]), PT_MONTH[c[:3]])] = v
        elif hasattr(c, "year"):
            td[(c.year, c.month)] = v
    out["ri_td_investors"] = td
    s = pd.DataFrame(out).apply(pd.to_numeric, errors="coerce")
    s.index = pd.MultiIndex.from_tuples(s.index, names=["year", "month"])
    return s.sort_index().reset_index()


def corr(x, y):
    """Pearson and Spearman correlations."""
    return float(np.corrcoef(x, y)[0, 1]), float(pd.Series(x).corr(pd.Series(y), method="spearman"))


def main():
    inv = load_td()
    b3 = load_b3_states()
    pop = load_population()
    mca = pd.read_parquet(ROUT / "tesouro_investors_mca_quarter.parquet")
    rows = []
    for d, dt in DATES.items():
        stock = (inv[inv["date"] <= pd.Timestamp(dt)].groupby("uf").size().rename("td_stock"))
        b = b3[b3["date"] == d].set_index("uf")
        x = pd.concat([stock, b[["b3_value", "b3_unit"]]], axis=1, join="inner").reset_index()
        x = x.rename(columns={"index": "uf"})
        x.insert(1, "date", d)
        x = x.merge(pop[pop["year"] == POP_YEAR[d]][["uf", "population"]], on="uf", how="left")
        x["td_per_1000"] = 1000 * x["td_stock"] / x["population"]
        x["b3_per_1000"] = 1000 * x["b3_value"] / x["population"]
        x["ratio_td_b3"] = x["td_stock"] / x["b3_value"]
        rows.append(x)
    st = pd.concat(rows, ignore_index=True)
    st.to_csv(ROUT / "tesouro_vs_b3_state.csv", index=False)

    print("STATE CHECK (total population, not adult)")
    for d, g in st.groupby("date"):
        pl, sl = corr(g["td_stock"], g["b3_value"])
        pp, sp = corr(g["td_per_1000"], g["b3_per_1000"])
        pm, _ = corr(np.log(g["td_stock"]), np.log(g["b3_value"]))
        print(f"{d}: n={len(g)} levels r={pl:.3f}, log-levels r={pm:.3f}, per-1000 r={pp:.3f}, "
              f"Spearman levels={sl:.3f}, Spearman per-1000={sp:.3f}, "
              f"national ratio={g['td_stock'].sum() / g['b3_value'].sum():.3f}")
        med = g["ratio_td_b3"].median()
        g = g.assign(rel=g["ratio_td_b3"] / med)
        g = g.reindex(np.abs(np.log(g["rel"])).sort_values(ascending=False).index).head(5)
        print("  median ratio", f"{med:.3f}", "; outliers (ratio, relative to median):",
              "; ".join(f"{r.uf} {r.ratio_td_b3:.3f} ({r.rel:.3f})" for r in g.itertuples()))
    # Cross-check against the MCA output (national sums at the same quarter end).
    for d, dt in DATES.items():
        y, q = int(d[:4]), (int(d[5:]) - 1) // 3 + 1
        m = mca[(mca["year"] == y) & (mca["quarter"] == q)]["stock_investors"].sum()
        u = int((inv["date"] <= pd.Timestamp(dt)).sum())
        print(f"cross-check {d}: MCA output sum {int(m):,}; investor-level domestic {u:,}")

    # National, monthly 2016-01..2024-12.
    ri = ri_series()
    mon = pd.period_range("2016-01", "2024-12", freq="M")
    ends = mon.to_timestamp(how="end").normalize()
    reg = np.sort(inv["date"].values)
    stock = np.searchsorted(reg, ends.values, side="right")
    nat = pd.DataFrame({"year": mon.year, "month": mon.month, "td_registry_stock": stock})
    nat = nat.merge(ri, on=["year", "month"], how="left")
    nat["ratio_registry_to_ri_td"] = nat["td_registry_stock"] / nat["ri_td_investors"]
    nat["ratio_registry_to_b3_cpfs"] = nat["td_registry_stock"] / nat["b3_unique_cpfs"]
    nat["g12_registry"] = nat["td_registry_stock"].pct_change(12)
    nat["g12_ri_td"] = nat["ri_td_investors"].pct_change(12)
    nat["g12_b3_cpfs"] = nat["b3_unique_cpfs"].pct_change(12)
    nat.to_csv(ROUT / "tesouro_vs_b3_national.csv", index=False)
    print("NATIONAL (2016-01..2024-12)")
    for a, b_ in [("ri_td_investors", "RI TD investors"), ("b3_unique_cpfs", "B3 unique CPFs")]:
        ok = nat[["td_registry_stock", a]].dropna()
        g = a.replace("ri_td_investors", "g12_ri_td").replace("b3_unique_cpfs", "g12_b3_cpfs")
        gg = nat[["g12_registry", g]].dropna()
        r = nat["td_registry_stock"] / nat[a]
        print(f"vs {b_}: n={len(ok)}, levels r={ok.corr().iloc[0, 1]:.3f}, "
              f"12m growth r={gg.corr().iloc[0, 1]:.3f}, ratio first={r.iloc[0]:.3f}, "
              f"2019-12={r[(nat.year == 2019) & (nat.month == 12)].iloc[0]:.3f}, "
              f"2024-12={r.iloc[-1]:.3f}, min={r.min():.3f}, max={r.max():.3f}")
    print(nat[nat["month"].isin([12])][["year", "td_registry_stock", "ri_td_investors", "b3_unique_cpfs",
                                         "ratio_registry_to_ri_td", "ratio_registry_to_b3_cpfs"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()

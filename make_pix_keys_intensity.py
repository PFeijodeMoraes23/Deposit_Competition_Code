"""
make_pix_keys_intensity.py
==========================
Pix-key intensity by prudential conglomerate and quarter, 2020Q4-2024Q4 (task P1, for the
convenience channel of V_Friction Section 7.1.2). A side file: it is not merged into
market_panel and no estimator reads it.

Inputs (read only)
  DATA_RAW/BCB/pix_participants/chaves/chavespix_from*.csv
      month-end stock of Pix keys by participant (ISPB) x PF/PJ x key type (scrape_pix_keys.py)
  DATA_RAW/BCB/pix_participants/compiled/ispb_conglomerate_map.csv
      ISPB -> prudential conglomerate (make_pix_keys_conglomerate.py), time-invariant
  processed/market_panel.parquet
      type-1 and type-5 balances (dep_a1, dep_a5, in R$) and is_B, by conglomerate x market x quarter

Rules
  1. Quarter-end months only (March, June, September, December), 2020Q4-2024Q4.
  2. Every ISPB follows the crosswalk. PicPay (ISPB 22896431) maps to C0088022 in every month, which
     is where the panel holds all of PicPay's balances (utils/firm_remaps.py).
  3. A key whose ISPB has no conglomerate is "unmapped". It counts in the totals of the report and
     in no conglomerate's row.

Output  Rout/pix_keys_conglomerate_quarter.csv, one row per conglomerate x quarter with any key:
  conglomerate, year, quarter, name (panel name, else the largest participant's Pix name),
  in_panel (the conglomerate has panel rows that quarter), is_B (panel; empty when not in panel),
  n_ispb, keys_pf, keys_pj, keys_total, keys_pf_{cpf,phone,email,random},
  share_pf_national (keys_pf over all mapped PF keys that quarter),
  bal_t1t5_brl_mn (type-1 plus type-5 balances summed over markets, R$ million),
  keys_pf_per_bal (keys_pf per R$ million of bal_t1t5_brl_mn; empty when the balance is not positive)
Report  Rout/pix_keys_conglomerate_quarter.json: mapped shares by quarter, B and D shares of PF keys,
  the largest unmapped ISPBs, the top conglomerates in 2024Q4, PicPay's PF keys.

Usage:  python make_pix_keys_intensity.py
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import datetime
import json

import numpy as np
import pandas as pd

from utils import paths as _paths
from utils import raw_paths
from utils import firm_remaps

KEYS_DIR = raw_paths.BCB_PIX_KEYS
XWALK = raw_paths.BCB_PIX_PARTICIPANTS / "compiled" / "ispb_conglomerate_map.csv"
PANEL = _paths.PROCESSED / "market_panel.parquet"
OUT = _paths.rout_dir() / "pix_keys_conglomerate_quarter.csv"
SIDE = _paths.rout_dir() / "pix_keys_conglomerate_quarter.json"

FIRST, LAST = "2020-12-31", "2024-12-31"
PICPAY_ISPB = "22896431"
# ChavesPix TipoChave -> column suffix. PF users hold no CNPJ keys (checked on the 2026-10-27 file).
PF_TYPES = {"CPF": "cpf", "Celular": "phone", "e-mail": "email", "Aleatória": "random"}


def load_keys() -> tuple[pd.DataFrame, str]:
    """Quarter-end rows in the window, with the conglomerate each ISPB maps to; and the file name."""
    files = sorted(KEYS_DIR.glob("chavespix_from*.csv"))
    if not files:
        raise SystemExit(f"no ChavesPix file in {KEYS_DIR}: run scrape_pix_keys.py first")
    k = pd.read_csv(files[-1], dtype={"ISPB": str}, encoding="utf-8")
    k["Data"] = pd.to_datetime(k["Data"])
    k = k[k["Data"].between(FIRST, LAST) & k["Data"].dt.month.isin([3, 6, 9, 12])].copy()
    unknown = set(k.loc[k["NaturezaUsuario"] == "PF", "TipoChave"]) - set(PF_TYPES)
    if unknown:
        raise SystemExit(f"PF key types not in PF_TYPES: {sorted(unknown)}")

    xw = pd.read_csv(XWALK, dtype=str)
    if xw["ispb"].duplicated().any():
        raise SystemExit("the crosswalk lists an ISPB twice")
    k["conglomerate"] = k["ISPB"].map(xw.set_index("ispb")["conglomerate"])
    if xw.loc[xw["ispb"] == PICPAY_ISPB, "conglomerate"].tolist() != [firm_remaps.PICPAY_TO]:
        raise SystemExit(f"the crosswalk no longer maps PicPay to {firm_remaps.PICPAY_TO}, "
                         f"where the panel holds PicPay's balances")
    k["year"] = k["Data"].dt.year
    k["quarter"] = (k["Data"].dt.month - 1) // 3 + 1
    return k, files[-1].name


def panel_balances() -> pd.DataFrame:
    """Type-1 plus type-5 balances (R$ million), is_B and name by conglomerate x quarter."""
    p = pd.read_parquet(PANEL, columns=["CodConglomeradoPrudencial", "year", "quarter", "is_B",
                                        "NomeInstituicao", "dep_a1", "dep_a5"])
    p = p[p["year"].between(2020, 2024)]
    p["bal"] = p["dep_a1"].fillna(0) + p["dep_a5"].fillna(0)
    g = p.groupby(["CodConglomeradoPrudencial", "year", "quarter"], as_index=False).agg(
        bal=("bal", "sum"), is_B=("is_B", "first"), name=("NomeInstituicao", "first"))
    g["bal_t1t5_brl_mn"] = g["bal"] / 1e6
    return g.rename(columns={"CodConglomeradoPrudencial": "conglomerate"}).drop(columns="bal")


def build(k: pd.DataFrame, bal: pd.DataFrame) -> pd.DataFrame:
    mk = k[k["conglomerate"].notna()]
    idx = ["conglomerate", "year", "quarter"]
    pf = mk[mk["NaturezaUsuario"] == "PF"]
    out = mk.groupby(idx).agg(n_ispb=("ISPB", "nunique")).join(
        pf.groupby(idx)["qtdChaves"].sum().rename("keys_pf")).join(
        mk[mk["NaturezaUsuario"] == "PJ"].groupby(idx)["qtdChaves"].sum().rename("keys_pj"))
    by_type = pf.pivot_table(index=idx, columns="TipoChave", values="qtdChaves", aggfunc="sum")
    out = out.join(by_type.rename(columns={t: f"keys_pf_{s}" for t, s in PF_TYPES.items()}))
    out = out.fillna(0).astype("int64").reset_index()
    out["keys_total"] = out["keys_pf"] + out["keys_pj"]
    out["share_pf_national"] = out["keys_pf"] / out.groupby(["year", "quarter"])["keys_pf"].transform("sum")

    out = out.merge(bal, on=idx, how="left")
    out["in_panel"] = out["bal_t1t5_brl_mn"].notna()
    pos = out["bal_t1t5_brl_mn"] > 0
    out["keys_pf_per_bal"] = np.where(pos, out["keys_pf"] / out["bal_t1t5_brl_mn"].where(pos), np.nan)
    # Name: the panel's, else the Pix name of the conglomerate's largest participant.
    big = (k[k["conglomerate"].notna()].groupby(["conglomerate", "Nome"])["qtdChaves"].sum()
           .reset_index().sort_values("qtdChaves").drop_duplicates("conglomerate", keep="last")
           .set_index("conglomerate")["Nome"])
    out["name"] = out["name"].fillna(out["conglomerate"].map(big))
    cols = ["conglomerate", "year", "quarter", "name", "in_panel", "is_B", "n_ispb",
            "keys_pf", "keys_pj", "keys_total"] + [f"keys_pf_{s}" for s in PF_TYPES.values()] + [
            "share_pf_national", "bal_t1t5_brl_mn", "keys_pf_per_bal"]
    return out[cols].sort_values(["year", "quarter", "keys_pf"], ascending=[True, True, False])


def report(k: pd.DataFrame, out: pd.DataFrame) -> dict:
    rep = {"by_quarter": {}}
    for (y, q), s in k.groupby(["year", "quarter"]):
        o = out[(out["year"] == y) & (out["quarter"] == q)]
        row = {}
        for lab, sub in (("all", s), ("PF", s[s["NaturezaUsuario"] == "PF"])):
            tot = sub["qtdChaves"].sum()
            mapped = sub["conglomerate"].notna()
            row[lab] = {"keys": int(tot),
                        "mapped_share": round(float(sub.loc[mapped, "qtdChaves"].sum() / tot), 4)}
        pf_tot = s.loc[s["NaturezaUsuario"] == "PF", "qtdChaves"].sum()
        inp = o[o["in_panel"]]
        row["PF"]["in_panel_share"] = round(float(inp["keys_pf"].sum() / pf_tot), 4)
        # B and D shares: of all PF keys, and among PF keys at panel conglomerates.
        b, d = inp.loc[inp["is_B"] == 1, "keys_pf"].sum(), inp.loc[inp["is_B"] == 0, "keys_pf"].sum()
        row["PF"]["share_B_of_all"] = round(float(b / pf_tot), 4)
        row["PF"]["share_D_of_all"] = round(float(d / pf_tot), 4)
        row["PF"]["share_D_of_panel"] = round(float(d / (b + d)), 4)
        rep["by_quarter"][f"{y}Q{q}"] = row

    last = k[(k["year"] == 2024) & (k["quarter"] == 4)]
    un = (last[last["conglomerate"].isna()].groupby(["ISPB", "Nome", "Segmento"], dropna=False)["qtdChaves"]
          .sum().sort_values(ascending=False).head(10).reset_index())
    rep["largest_unmapped_2024Q4"] = un.rename(columns={"qtdChaves": "keys"}).to_dict("records")
    o = out[(out["year"] == 2024) & (out["quarter"] == 4)].nlargest(10, "keys_pf")
    rep["top10_2024Q4_by_pf_keys"] = o[["conglomerate", "name", "is_B", "keys_pf", "share_pf_national",
                                         "bal_t1t5_brl_mn", "keys_pf_per_bal"]].round(4).to_dict("records")
    pp = k[(k["ISPB"] == PICPAY_ISPB) & (k["NaturezaUsuario"] == "PF")]
    rep["picpay_pf_keys"] = {f"{y}Q{q}": {"keys_pf": int(s["qtdChaves"].sum()), "to": s["conglomerate"].iloc[0]}
                             for (y, q), s in pp.groupby(["year", "quarter"])}
    return rep


def main():
    k, src = load_keys()
    out = build(k, panel_balances())
    out.to_csv(OUT, index=False, encoding="utf-8")
    rep = report(k, out)
    rep["_meta"] = {"written": datetime.datetime.now().isoformat(timespec="seconds"), "source": src,
                    "rows": int(len(out)), "quarters": int(out[["year", "quarter"]].drop_duplicates().shape[0]),
                    "conglomerates": int(out["conglomerate"].nunique())}
    SIDE.write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    for q, r in rep["by_quarter"].items():
        print(f"  {q}: PF keys {r['PF']['keys']:>12,}  mapped {r['PF']['mapped_share']:.3f}  "
              f"in panel {r['PF']['in_panel_share']:.3f}  B {r['PF']['share_B_of_all']:.3f}  "
              f"D {r['PF']['share_D_of_all']:.3f}")
    print(f"  wrote {OUT} ({len(out):,} rows)\n  wrote {SIDE}")


if __name__ == "__main__":
    main()

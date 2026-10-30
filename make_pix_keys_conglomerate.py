###─────────────────────────────────────────────────────────────────────────────
# make_pix_keys_conglomerate.py
#
# Author: Pedro Feijó de Moraes
#
# PURPOSE
#   Link the Pix participants of the ChavesPix key stock (scrape_pix_keys.py) to the market
#   panel's prudential conglomerates, and report how much of the key stock the panel covers
#   (scraping task S13, item 4). A compiled crosswalk, not raw data; the raw file is not modified.
#
# LINKS
#   1. ISPB -> CNPJ root (8 digits): the Pix participant rosters (shared/PIX/pix_roster_history.csv,
#      from scrape_pix_roster.py), whose modern rosters carry both ISPB and CNPJ. An ISPB absent
#      from the rosters takes the ISPB itself as the CNPJ root (the ISPB of an institution is its
#      CNPJ root); `cnpj8_source` says which.
#   2. CNPJ root -> conglomerate: utils.conglomerates.build_cnpj_conglomerate_map(), the lookup that
#      keys the market panel (every IF.data quarter, canonical codes).
#   3. in_panel: the conglomerate appears in market_panel.parquet in 2020-2024.
#
# OUTPUT  (raw_paths.BCB_PIX_PARTICIPANTS / "compiled")
#   ispb_conglomerate_map.csv   ispb, nome (latest), cnpj8, cnpj8_source, conglomerate, in_panel,
#                               keys_2024_12 (all keys at 2024-12-31)
#   coverage.json               share of keys linked to a conglomerate and to a panel conglomerate,
#                               all keys and PF keys, at each December 2020-2024; largest unlinked
#
# USAGE
#   python make_pix_keys_conglomerate.py
###─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import glob
import json
import os

import pandas as pd

from utils import paths, raw_paths
from utils.conglomerates import build_cnpj_conglomerate_map

OUT = raw_paths.BCB_PIX_PARTICIPANTS / "compiled"
ROSTER = os.path.join(str(paths.PIX_DIR), "pix_roster_history.csv")
PANEL = paths.OPEN_FINANCE / "BCB" / "Egan_et_al_2025_Rep" / "processed" / "market_panel.parquet"


def main() -> None:
    files = sorted(glob.glob(str(raw_paths.BCB_PIX_KEYS / "chavespix_from*.csv")))
    if not files:
        raise SystemExit("no ChavesPix file: run scrape_pix_keys.py first")
    keys = pd.read_csv(files[-1], dtype={"ISPB": str}, encoding="utf-8")
    keys["Data"] = pd.to_datetime(keys["Data"])
    keys = keys[keys["Data"] <= "2024-12-31"]

    roster = pd.read_csv(ROSTER, dtype=str)
    xw = (roster.dropna(subset=["ispb_filled", "cnpj8"])
          .sort_values("roster_date")
          .drop_duplicates("ispb_filled", keep="last")
          .set_index("ispb_filled")["cnpj8"])

    cmap = {str(int(k)).zfill(8): str(v[0]).strip() for k, v in build_cnpj_conglomerate_map().items()
            if v and v[0] is not None and str(v[0]).strip() not in ("", "nan", "None")}
    panel = pd.read_parquet(PANEL, columns=["CodConglomeradoPrudencial", "year"])
    in_panel = set(panel.loc[panel["year"].between(2020, 2024), "CodConglomeradoPrudencial"].astype(str))

    names = keys.sort_values("Data").drop_duplicates("ISPB", keep="last").set_index("ISPB")["Nome"]
    last = keys[keys["Data"] == keys["Data"].max()].groupby("ISPB")["qtdChaves"].sum()
    m = pd.DataFrame({"ispb": sorted(keys["ISPB"].unique())})
    m["nome"] = m["ispb"].map(names)
    m["cnpj8"] = m["ispb"].map(xw)
    m["cnpj8_source"] = m["cnpj8"].notna().map({True: "pix_roster", False: "ispb_as_root"})
    m["cnpj8"] = m["cnpj8"].fillna(m["ispb"])
    m["conglomerate"] = m["cnpj8"].map(cmap)
    m["in_panel"] = m["conglomerate"].isin(in_panel)
    m["keys_2024_12"] = m["ispb"].map(last).fillna(0).astype("int64")

    os.makedirs(OUT, exist_ok=True)
    m.to_csv(OUT / "ispb_conglomerate_map.csv", index=False, encoding="utf-8")

    k = keys.merge(m[["ispb", "conglomerate", "in_panel"]], left_on="ISPB", right_on="ispb", how="left")
    cov = {}
    for d in sorted(k["Data"].unique()):
        d = pd.Timestamp(d)
        if d.month != 12 and d != k["Data"].max():
            continue
        s = k[k["Data"] == d]
        row = {}
        for label, sub in (("all", s), ("PF", s[s["NaturezaUsuario"] == "PF"])):
            tot = sub["qtdChaves"].sum()
            row[label] = {
                "keys": int(tot),
                "share_linked_to_conglomerate": round(sub.loc[sub["conglomerate"].notna(), "qtdChaves"].sum() / tot, 4),
                "share_in_panel": round(sub.loc[sub["in_panel"], "qtdChaves"].sum() / tot, 4),
            }
        row["participants"] = int(s["ISPB"].nunique())
        cov[str(d.date())] = row
    unl = m[~m["in_panel"]].sort_values("keys_2024_12", ascending=False).head(15)
    cov["largest_not_in_panel_2024_12"] = unl[["ispb", "nome", "conglomerate", "keys_2024_12"]].to_dict("records")
    cov["source_file"] = os.path.basename(files[-1])
    with open(OUT / "coverage.json", "w", encoding="utf-8") as f:
        json.dump(cov, f, ensure_ascii=False, indent=1, default=str)

    for d, v in cov.items():
        if isinstance(v, dict) and "all" in v:
            print(f"{d}: participants {v['participants']}, linked {v['all']['share_linked_to_conglomerate']:.1%}, "
                  f"in panel {v['all']['share_in_panel']:.1%} (PF {v['PF']['share_in_panel']:.1%})")
    print("written:", OUT)


if __name__ == "__main__":
    main()

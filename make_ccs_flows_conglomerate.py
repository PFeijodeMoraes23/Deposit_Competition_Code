"""make_ccs_flows_conglomerate.py -- the CCS relationship flows (make_ccs_flows.py) by prudential
conglomerate and quarter, with the institutional events flagged.

Not a pipeline step: nothing here enters market_panel or an estimator, and no paper number reads it.

Mapping of an institution's CNPJ root to a conglomerate, in this order:
  1. utils.conglomerates.build_cnpj_conglomerate_map(), called as make_pix_keys_conglomerate.py does;
  2. for roots still unmapped, the S13 ispb_conglomerate_map.csv (an institution's ISPB is its CNPJ root);
  3. the E1 firm remap: PicPay is utils.firm_remaps.PICPAY_TO in every quarter, for PicPay IP
     (22896431) and PicPay Bank (09516419).
A root left unmapped keeps its own rows, with conglomerate = "root_<cnpj8>" and mapped = 0 (BCO BEM,
which has no CNPJ in the BCB file, is "root_missing").
is_B is the panel's own column (one value per firm, the 2024-12 rule); in_panel = 1 when the
conglomerate is in market_panel.parquet in 2016-2024.

Quarters sum the three months. net = opened_gross - closed. retention = opened_active / opened_gross,
a property of the opening cohort ("active" at the BCB's unstated extraction date, after 2024-12).
n_institutions = institutions of the conglomerate with any flow in the quarter.

Institutional events are flagged, never dropped: the 25 institution-months with the largest PF
closures, each with closures as a share of the institution's cumulative gross openings since 2016-01
(above 1 when it closes relationships opened before 2016). The event is written only where
config/ccs_flow_events.csv documents it with a source, else "unknown". event_closed = the PF closures
of the conglomerate-quarter that fall in those flagged institution-months.

Honesty flags: a relationship is one CPF or CNPJ at one institution, not a person and not an account,
so a conglomerate's sum counts a client once per institution; dormant accounts stay active; closures
mix client choices with corporate events; 2020 openings are partly policy-driven; no geography, no
product.

Outputs (utils.paths.rout_dir()):
  ccs_flows_conglomerate_quarter.csv   conglomerate x quarter x type
  ccs_flows_institution_month_mapped.csv  institution x type x month with its conglomerate, in_panel,
                                       is_B and event_closed (the monthly view the report charts)
  ccs_flow_events_pf_top25.csv         the flagged institution-months
  ccs_flows_mapping_report.json        share of gross openings mapped, by year; largest unmapped roots
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import paths, raw_paths  # noqa: E402
from utils.conglomerates import build_cnpj_conglomerate_map  # noqa: E402
from utils.firm_remaps import PICPAY_TO  # noqa: E402

FLOWS = raw_paths.BCB_CCS_LAI.parent / "compiled" / "ccs_flows_institution_month.csv"
ISPB_MAP = raw_paths.BCB_PIX_PARTICIPANTS / "compiled" / "ispb_conglomerate_map.csv"
PANEL = paths.OPEN_FINANCE / "BCB" / "Egan_et_al_2025_Rep" / "processed" / "market_panel.parquet"
EVENTS_CFG = Path(__file__).resolve().parent / "config" / "ccs_flow_events.csv"
ROUT = paths.rout_dir()
OUT = ROUT / "ccs_flows_conglomerate_quarter.csv"
MONTHLY = ROUT / "ccs_flows_institution_month_mapped.csv"
EVENTS_OUT = ROUT / "ccs_flow_events_pf_top25.csv"
REPORT = ROUT / "ccs_flows_mapping_report.json"
PICPAY_ROOTS = ("22896431", "09516419")
N_EVENTS = 25
MEASURES = ["opened_gross", "opened_active", "opened_closed_later", "closed"]


def root_map(roots) -> pd.DataFrame:
    """cnpj8 -> (conglomerate, map_source) for every root of the flows file."""
    cmap = {str(int(k)).zfill(8): str(v[0]).strip() for k, v in build_cnpj_conglomerate_map().items()
            if v and v[0] is not None and str(v[0]).strip() not in ("", "nan", "None")}
    ispb = pd.read_csv(ISPB_MAP, dtype=str).dropna(subset=["conglomerate"])
    ispb_map = dict(zip(ispb["ispb"].str.zfill(8), ispb["conglomerate"].str.strip()))
    out = []
    for r in roots:
        if r in PICPAY_ROOTS:
            out.append((r, PICPAY_TO, "picpay_remap"))
        elif r in cmap:
            out.append((r, cmap[r], "if_data_list"))
        elif r in ispb_map:
            out.append((r, ispb_map[r], "pix_ispb_map"))
        else:
            out.append((r, None, "unmapped"))
    return pd.DataFrame(out, columns=["cnpj8", "conglomerate", "map_source"])


def main():
    f = pd.read_csv(FLOWS, dtype={"cnpj8": str})
    f["cnpj8"] = f["cnpj8"].where(f["cnpj_missing"] == 0, None)
    f["inst"] = f["cnpj8"].fillna("missing")
    roots = sorted(f["cnpj8"].dropna().unique())
    rm = root_map(roots)
    f = f.merge(rm, on="cnpj8", how="left")
    f["map_source"] = f["map_source"].fillna("unmapped")
    f["mapped"] = f["conglomerate"].notna().astype(int)
    f["conglomerate"] = f["conglomerate"].fillna("root_" + f["inst"])

    panel = pd.read_parquet(PANEL, columns=["CodConglomeradoPrudencial", "year", "quarter", "is_B"])
    panel = panel[panel["year"].between(2016, 2024)]
    panel["CodConglomeradoPrudencial"] = panel["CodConglomeradoPrudencial"].astype(str)
    in_panel = set(panel["CodConglomeradoPrudencial"])
    nb = panel.groupby("CodConglomeradoPrudencial")["is_B"].nunique()
    is_b = (panel.sort_values(["year", "quarter"])
            .groupby("CodConglomeradoPrudencial")["is_B"].last())
    f["in_panel"] = f["conglomerate"].isin(in_panel).astype(int)
    f["is_B"] = f["conglomerate"].map(is_b)

    # ── Institutional events: the 25 largest institution-months by PF closures ────────────────────
    pf = f[f["type"] == "PF"].sort_values(["inst", "name", "ym"]).copy()
    pf["cum_gross"] = pf.groupby(["inst", "name"])["opened_gross"].cumsum()
    top = pf.nlargest(N_EVENTS, "closed").copy()
    top["closed_share_of_cum_gross"] = top["closed"] / top["cum_gross"].replace(0, np.nan)
    if EVENTS_CFG.is_file():
        cfg = pd.read_csv(EVENTS_CFG, dtype=str).fillna("")
        top = top.merge(cfg[["cnpj8", "ym", "event", "source"]], on=["cnpj8", "ym"], how="left")
    else:
        top["event"], top["source"] = "", ""
    top["event"] = top["event"].fillna("").replace("", "unknown")
    top["source"] = top["source"].fillna("")
    ev = top[["cnpj8", "name", "segment", "ym", "closed", "opened_gross", "cum_gross",
              "closed_share_of_cum_gross", "conglomerate", "map_source", "in_panel", "is_B",
              "event", "source"]].sort_values("closed", ascending=False)
    flagged = set(zip(ev["cnpj8"].fillna("missing"), ev["ym"]))
    in_event = np.array([(i, m) in flagged for i, m in zip(f["inst"], f["ym"])])
    f["event_closed"] = np.where((f["type"] == "PF").to_numpy() & in_event, f["closed"], 0)

    # ── Conglomerate x quarter x type ─────────────────────────────────────────────────────────────
    f["quarter"] = f["ym"].str[:4] + "Q" + ((f["ym"].str[5:7].astype(int) - 1) // 3 + 1).astype(str)
    f["active"] = (f[MEASURES].sum(axis=1) > 0)
    keys = ["conglomerate", "quarter", "type"]
    q = f.groupby(keys)[MEASURES + ["event_closed"]].sum()
    q["n_institutions"] = f[f["active"]].groupby(keys)["inst"].nunique()
    q = q.reset_index()
    q["n_institutions"] = q["n_institutions"].fillna(0).astype(int)
    q["net"] = q["opened_gross"] - q["closed"]
    q["retention"] = q["opened_active"] / q["opened_gross"].replace(0, np.nan)
    attr = f.drop_duplicates("conglomerate").set_index("conglomerate")[["mapped", "in_panel", "is_B"]]
    q = q.join(attr, on="conglomerate")
    q = q[["conglomerate", "quarter", "type", "opened_gross", "opened_active", "opened_closed_later", "closed",
           "net", "retention", "n_institutions", "is_B", "in_panel", "mapped", "event_closed"]]
    q = q.sort_values(keys).reset_index(drop=True)

    # totals survive the aggregation
    for m in MEASURES:
        assert int(q[m].sum()) == int(f[m].sum()), m

    # ── Mapping report ────────────────────────────────────────────────────────────────────────────
    f["year"] = f["ym"].str[:4].astype(int)
    by_year = {}
    for (y, t), g in f.groupby(["year", "type"]):
        tot = g["opened_gross"].sum()
        by_year.setdefault(str(y), {})[t] = {
            "opened_gross_m": round(tot / 1e6, 3),
            "share_mapped": round(g.loc[g["mapped"] == 1, "opened_gross"].sum() / tot, 4),
            "share_in_panel": round(g.loc[g["in_panel"] == 1, "opened_gross"].sum() / tot, 4),
        }
    unm = (f[f["mapped"] == 0].groupby(["inst", "name"])["opened_gross"].sum()
           .sort_values(ascending=False).head(15).reset_index())
    src = f.drop_duplicates("inst")["map_source"].value_counts().to_dict()
    gross_src = (f.groupby("map_source")["opened_gross"].sum() / f["opened_gross"].sum()).round(4).to_dict()
    report = {
        "flows_file": FLOWS.name,
        "institutions_by_map_source": src,
        "share_of_gross_openings_by_map_source": gross_src,
        "by_year": by_year,
        "largest_unmapped_roots": [{"cnpj8": r.inst, "name": r.name, "opened_gross_m": round(r.opened_gross / 1e6, 3)}
                                   for r in unm.itertuples()],
        "panel_conglomerates_with_varying_is_B": [k for k, v in nb.items() if v > 1],
        "flags": ["a relationship is one CPF or CNPJ at one institution, not a person and not an account",
                  "still active is measured at an unstated extraction date after 2024-12",
                  "dormant accounts are active relationships",
                  "closures mix client choices with corporate events",
                  "2020 openings are partly policy-driven (Caixa, emergency aid)",
                  "no geography and no product"],
    }
    ROUT.mkdir(parents=True, exist_ok=True)
    q.to_csv(OUT, index=False)
    f[["inst", "name", "type", "ym", "conglomerate", "map_source", "mapped", "in_panel", "is_B"]
      + MEASURES + ["event_closed"]].rename(columns={"inst": "cnpj8"}).to_csv(MONTHLY, index=False)
    ev.to_csv(EVENTS_OUT, index=False)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT.name} ({len(q):,} rows), {EVENTS_OUT.name}, {REPORT.name}")
    print("institutions by map source:", src)
    print("share of gross openings by map source:", gross_src)
    for y, d in by_year.items():
        print(f"  {y}: " + "  ".join(f"{t} mapped {v['share_mapped']:.3f} in_panel {v['share_in_panel']:.3f}"
                                     for t, v in d.items()))
    print("largest unmapped:", [(u["cnpj8"], u["name"], u["opened_gross_m"]) for u in report["largest_unmapped_roots"][:8]])
    print("panel conglomerates with varying is_B:", report["panel_conglomerates_with_varying_is_B"])
    print(ev[["cnpj8", "name", "ym", "closed", "closed_share_of_cum_gross", "conglomerate", "event"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()

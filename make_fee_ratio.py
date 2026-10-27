"""
make_fee_ratio.py
=================
Fee revenue over total deposits, incumbents against D firms: the COSIF fee ratio
(cosif_fee_ratio_total_deposits, quarterly increments) of the December quarter x 4 x 100 (pp a
year), as service_fees_notes.md Section 5, from the fee panel (market_panel_with_fees.parquet,
stage 7). The median of the five incumbents (BB, Caixa, Bradesco, Itau, Santander) and of the D
firms, by year, 2016-2024 (2025 is the COSIF reclassification and is dropped).

Output: Rout/fee_ratio.json, read by make_paper_numbers.py (desc.fees) and make_slide_figures.py
--only followups (fig_slide_fee_ratio_ref).

Usage:  python make_fee_ratio.py
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import datetime
import json

import pandas as pd

from utils import paths as _paths
from make_xi_diagnostics import _stamp

FEES = _paths.PROCESSED / "market_panel_with_fees.parquet"
SIDECAR = _paths.rout_dir() / "fee_ratio.json"
INCUMBENTS = {"C0080329": "BB", "C0080738": "Caixa", "C0080075": "Bradesco", "C0080099": "Itau",
              "C0080185": "Santander"}


def fee_ratio() -> dict:
    d = pd.read_parquet(FEES, columns=["CodConglomeradoPrudencial", "is_B", "year", "quarter",
                                       "cosif_fee_ratio_total_deposits", "cosif_fee_valid"])
    d = d.drop_duplicates(["CodConglomeradoPrudencial", "year", "quarter"])
    d = d[(d["quarter"] == 4) & d["year"].between(2016, 2024) & (d["cosif_fee_valid"] == 1)]
    d = d.dropna(subset=["cosif_fee_ratio_total_deposits"])
    d["fee_pp"] = 400.0 * d["cosif_fee_ratio_total_deposits"]
    inc = d[d["CodConglomeradoPrudencial"].isin(INCUMBENTS)].groupby("year")["fee_pp"]
    dig = d[d["is_B"] == 0].groupby("year")["fee_pp"]
    out = dict(incumbent={str(y): float(v) for y, v in inc.median().items()},
               digital={str(y): float(v) for y, v in dig.median().items()},
               n_incumbent={str(y): int(v) for y, v in inc.size().items()},
               n_digital={str(y): int(v) for y, v in dig.size().items()})
    out["incumbent_change_2019_2024"] = out["incumbent"]["2024"] - out["incumbent"]["2019"]
    out["digital_change_2019_2024"] = out["digital"]["2024"] - out["digital"]["2019"]
    return out


def main():
    res = fee_ratio()
    print("incumbent", {k: round(v, 2) for k, v in res["incumbent"].items()})
    print("digital  ", {k: round(v, 2) for k, v in res["digital"].items()})
    meta = dict(fees=_stamp(FEES), written=datetime.datetime.now().isoformat(timespec="seconds"))
    SIDECAR.write_text(json.dumps(dict(_meta=meta, **res), indent=2, default=float),
                       encoding="utf-8")
    print(f"  wrote {SIDECAR}")


if __name__ == "__main__":
    main()

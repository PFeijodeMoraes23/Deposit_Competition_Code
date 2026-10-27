"""
make_dshare_decomp.py
=====================
The change in the D firms' national deposit share, 2016 to 2024, split into a deposit-type mix
part (between) and a within-type part, two-fold, with 2016 and with 2024 type weights. Balances
are annual means of the quarterly national balances by firm type and deposit type (types 1, 2,
4, 5), from the market panel.

Output: Rout/dshare_decomp.json, read by make_paper_numbers.py (desc.dshare_decomp).

Usage:  python make_dshare_decomp.py
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import datetime
import json

import pandas as pd

from utils import paths as _paths
from make_xi_diagnostics import _stamp

PANEL = _paths.PROCESSED / "market_panel.parquet"
SIDECAR = _paths.rout_dir() / "dshare_decomp.json"


def dshare_decomp() -> dict:
    cols = {"dep_a1": "k1", "dep_a2": "k2", "dep_a4": "k4", "dep_a5": "k5"}
    p = pd.read_parquet(PANEL, columns=["is_B", "year", "quarter", *cols])
    p = p[p["year"].isin([2016, 2024])]
    # Annual mean over quarters of the national balance by firm type and deposit type.
    q = p.groupby(["year", "quarter", "is_B"])[list(cols)].sum().groupby(["year", "is_B"]).mean()
    out = {}
    for y in (2016, 2024):
        D, B = q.loc[(y, 0)], q.loc[(y, 1)]
        tot = D + B
        out[y] = dict(w=(tot / tot.sum()).rename(cols).to_dict(),
                      s=(D / tot.where(tot > 0)).fillna(0.0).rename(cols).to_dict(),
                      S=float(D.sum() / tot.sum()))
    a, z = out[2016], out[2024]
    ks = list(cols.values())
    dS = z["S"] - a["S"]
    within16 = sum(a["w"][k] * (z["s"][k] - a["s"][k]) for k in ks)
    between16 = sum((z["w"][k] - a["w"][k]) * z["s"][k] for k in ks)
    within24 = sum(z["w"][k] * (z["s"][k] - a["s"][k]) for k in ks)
    between24 = sum((z["w"][k] - a["w"][k]) * a["s"][k] for k in ks)
    for x, y_ in ((within16 + between16, dS), (within24 + between24, dS)):
        assert abs(x - y_) < 1e-12
    return dict(total=dS, share2016=a["S"], share2024=z["S"],
                between=dict(w2016=between16, w2024=between24),
                within=dict(w2016=within16, w2024=within24), weights=dict(y2016=a["w"], y2024=z["w"]),
                within_type_share=dict(y2016=a["s"], y2024=z["s"]))


def main():
    res = dshare_decomp()
    print({k: res[k] for k in ("total", "between", "within")})
    meta = dict(panel=_stamp(PANEL), written=datetime.datetime.now().isoformat(timespec="seconds"))
    SIDECAR.write_text(json.dumps(dict(_meta=meta, **res), indent=2, default=float),
                       encoding="utf-8")
    print(f"  wrote {SIDECAR}")


if __name__ == "__main__":
    main()

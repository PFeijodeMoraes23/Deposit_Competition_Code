"""backup_precenter.py -- snapshot the pre-centering artefacts before the estimators are re-run.

Author: Pedro Feijó de Moraes

Written to a SIBLING of DEMAND_PREP, never a subdirectory of it: blp_1_logit.jl and
blp_2_rc.jl auto-discover `demand_*_spec_*.parquet` by scanning DEMAND_PREP, and their
"newest mtime per estimator id" rule would happily promote a backup copy to the input of
record.  A sibling cannot be seen by that scan at all.

market_panel_phis.csv is ~700 MB per estimator and is 95% columns the diff gate never
reads, so it is stored REDUCED (identifier keys + every phi_mt_* column) as parquet.
Everything else is copied verbatim.
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from utils import paths as _paths_mod

EST_OUT = _paths_mod.PROCESSED / "ESTIMATION_OUTPUT"
DEMAND_PREP = EST_OUT / "DEMAND_PREP"
KEYS = ["CodConglomeradoPrudencial", "mca_code", "deposit_type", "year", "quarter",
        "year_quarter", "entity_id", "time_id"]


def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = EST_OUT / f"_PRECENTER_{stamp}"
    dest.mkdir(parents=True, exist_ok=False)
    print(f"backup -> {dest}\n")
    total = 0

    for est in range(1, 10):
        src = DEMAND_PREP / f"est{est}"
        if not src.exists():
            continue
        d = dest / f"est{est}"
        d.mkdir(parents=True, exist_ok=True)
        for fn in ("estimation_results.pkl", "national_phi_t.csv"):
            if (src / fn).exists():
                shutil.copy2(src / fn, d / fn)
                total += (src / fn).stat().st_size
                print(f"  est{est}/{fn}")
        phis = src / "market_panel_phis.csv"
        if phis.exists():
            head = pd.read_csv(phis, nrows=0)
            cols = [c for c in head.columns if c.startswith("phi_mt_")]
            use = [k for k in KEYS if k in head.columns] + cols
            df = pd.read_csv(phis, usecols=use, dtype={"mca_code": str}, low_memory=False)
            out = d / "market_panel_phis.parquet"
            df.to_parquet(out, engine="pyarrow", index=False)
            total += out.stat().st_size
            print(f"  est{est}/market_panel_phis.parquet  "
                  f"{len(df):,} rows x {len(cols)} phi cols  "
                  f"({phis.stat().st_size/1e6:.0f} MB -> {out.stat().st_size/1e6:.0f} MB)")

    for pat, sub in ((("demand_*_spec_*.parquet", "demand_prep_summary_*.json"), DEMAND_PREP),
                     (("upsilon_pix_E*.json", "phi_nopix_E*.parquet"), EST_OUT / "CF_FOUNDATION"),
                     (("*.tex", "ts_link_band_est*.pkl"), EST_OUT / "Rout")):
        if not sub.exists():
            continue
        d = dest / sub.name
        for g in pat:
            for f in sorted(sub.glob(g)):
                d.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, d / f.name)
                total += f.stat().st_size
        n = len(list(d.glob("*"))) if d.exists() else 0
        print(f"  {sub.name}/  {n} files")

    blp = EST_OUT / "BLP_RESULTS"
    if blp.exists():
        d = dest / "BLP_RESULTS"
        for g in ("logit/*.json", "logit/*.tex", "logit_delta_*.bin", "*.json"):
            for f in sorted(blp.glob(g)):
                (d / f.parent.relative_to(blp)).mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, d / f.relative_to(blp))
                total += f.stat().st_size
        print(f"  BLP_RESULTS/  {len(list(d.rglob('*')))} entries")

    for f in (DEMAND_PREP / "state_centering_means.json",):
        if f.exists():
            shutil.copy2(f, dest / f.name)

    print(f"\ntotal {total/1e9:.2f} GB")
    print(f"\ngate with:\n  .venv/Scripts/python check_state_centering.py --gate \"{dest}\"")
    return dest


if __name__ == "__main__":
    main()

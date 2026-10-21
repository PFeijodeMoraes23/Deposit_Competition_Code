"""
make_demand_type5_variants.py
=============================
Sample variants of the specification-12 demand parquets of E3 and E4 for the prepaid (type 5)
robustness (brief C8, 2026-10-21). No institution-level prepaid rate is observed, so every type-5
cell carries the CDI fallback and a spread near zero, and B firms' type-5 balances are imputed
from their checking footprint. The variants change the sample, not the model:

  no5     every type-5 row dropped
  d5only  type-5 rows kept for D firms only (B firms' type-5 rows dropped)

Each variant is the production parquet with rows removed, nothing recomputed: the market sizes
and active shares of the remaining rows are the production ones, so the dropped products move
to the outside good. Written to ESTIMATION_OUTPUT/DEMAND_PREP_VARIANTS/<variant>/ under the
production file names (a separate folder, so the demand stage never sees a second vintage of a
routine), with counts in DEMAND_PREP_VARIANTS/type5_variants.json.

Usage:  python make_demand_type5_variants.py                  write the variants
        python make_demand_type5_variants.py --logit-compare  after the variant logit fits
        python make_demand_type5_variants.py --stage-upload   the cluster upload (zip + sha256)
"""
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)

import datetime
from pathlib import Path
import json

import pandas as pd

from utils import paths as _paths
from utils import routines as _routines
from make_xi_diagnostics import _stamp

SPEC = 12
ROOT = _paths.demand_parquet_dir().parent / "DEMAND_PREP_VARIANTS"
VARIANTS = {"no5": lambda d: d["deposit_type"] != 5,
            "d5only": lambda d: ~((d["deposit_type"] == 5) & d["is_B"].astype(bool))}


def logit_compare() -> dict:
    """The logit benchmark (full_dtype sub-model) on each variant against production: the price
    coefficient, its standard error and the mean own-price elasticity alpha x mean(rho (1 - s))
    over the variant's sample (make_blp_rc_table.mean_rho_one_minus_s's formula). The variant
    fits are blp_logit.jl --est <k> --demand-dir <ROOT>/<variant> --out-dir
    <ROOT>/<variant>/logit_run; production is BLP_RESULTS/logit/logit_summary_spec_12.json."""
    import numpy as np

    def entry(path, e):
        s = json.loads(Path(path).read_text(encoding="utf-8"))
        p = s[f"E{e}_full_dtype"]
        nm = p.get("param_names") or []
        i = nm.index("alpha") if "alpha" in nm else 0
        se = p.get("theta1_se") or p["se"]
        return float(p["theta1"][i]), float(se[i]), int(p["n_obs"]), path

    def rho1s(pq):
        df = pd.read_parquet(pq, columns=["spread_ann", "share_D", "share_B_cond", "is_B"])
        isb = df["is_B"].fillna(False).astype(bool).to_numpy()
        s = np.where(isb, df["share_B_cond"].to_numpy(float), df["share_D"].to_numpy(float))
        r = df["spread_ann"].to_numpy(float) / 100.0
        m = np.isfinite(r) & np.isfinite(s) & (s >= 0.0) & (s < 1.0)
        return float(np.mean(r[m] * (1.0 - s[m])))

    out = {}
    prod = _paths.blp_results_dir() / "logit" / "logit_summary_spec_12.json"
    for e in _routines.LINK_ESTS:
        name = f"{_routines.demand_prefix(e)}_spec_{SPEC}.parquet"
        runs = {"prod": (prod, _paths.demand_parquet_dir() / name)}
        for v in VARIANTS:
            runs[v] = (ROOT / v / "logit_run" / "logit" / "logit_summary_spec_12.json", ROOT / v / name)
        for tag, (summ, pq) in runs.items():
            if not Path(summ).is_file():
                print(f"  E{e} {tag}: no logit fit at {summ}")
                continue
            a, se, n, _ = entry(summ, e)
            r = rho1s(pq)
            out.setdefault(f"E{e}", {})[tag] = dict(alpha=a, se=se, elast=a * r, rho1s=r, n_obs=n,
                                                    source=_stamp(Path(summ)))
            print(f"  E{e} {tag:7s} alpha {a:+.4f} (se {se:.4f}) elasticity {a * r:+.4f} n {n:,}")
    return out


def stage_upload() -> Path:
    """The cluster upload for blp_type5_variants.sh: one zip whose top folder is
    upload_type5_20261021/ with <variant>/ (the two parquets and logit/, the variant's logit step
    folder), and scripts/ (the launcher and its job script), plus a sha256 list of every member."""
    import hashlib
    import zipfile
    here = Path(__file__).resolve().parent
    out = _paths.demand_parquet_dir().parent / "CLUSTER_OUT" / "upload_type5_20261021.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    members = []
    for v in VARIANTS:
        for e in _routines.LINK_ESTS:
            name = f"{_routines.demand_prefix(e)}_spec_{SPEC}.parquet"
            members.append((ROOT / v / name, f"upload_type5_20261021/{v}/{name}"))
        lg = ROOT / v / "logit_run" / "logit"
        for f in sorted(lg.iterdir()):
            members.append((f, f"upload_type5_20261021/{v}/logit/{f.name}"))
    for s in ("blp_type5_variants.sh", "blp_type5_variant_job.sh"):
        members.append((here / s, f"scripts/{s}"))
    sums = []
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for src, arc in members:
            z.write(src, arc)
            sums.append(f"{hashlib.sha256(src.read_bytes()).hexdigest()}  {arc}")
    out.with_suffix(".sha256").write_text("\n".join(sums) + "\n", encoding="utf-8", newline="\n")
    print(f"  wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(members)} files) and its .sha256")
    return out


def main():
    import sys
    if "--stage-upload" in sys.argv:
        stage_upload()
        return
    if "--logit-compare" in sys.argv:
        out = logit_compare()
        (ROOT / "type5_logit_compare.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"  wrote {ROOT / 'type5_logit_compare.json'}")
        return
    out = {"_meta": dict(written=datetime.datetime.now().isoformat(timespec="seconds"))}
    for e in _routines.LINK_ESTS:
        name = f"{_routines.demand_prefix(e)}_spec_{SPEC}.parquet"
        src = _paths.demand_parquet_dir() / name
        d = pd.read_parquet(src)
        out[f"E{e}"] = dict(source=_stamp(src), rows=int(len(d)),
                            by_type={f"k{int(k)}": int(v) for k, v in d["deposit_type"].value_counts().sort_index().items()},
                            type5_B=int(((d["deposit_type"] == 5) & d["is_B"].astype(bool)).sum()),
                            type5_D=int(((d["deposit_type"] == 5) & ~d["is_B"].astype(bool)).sum()))
        for v, keep in VARIANTS.items():
            dv = d[keep(d)]
            (ROOT / v).mkdir(parents=True, exist_ok=True)
            dv.to_parquet(ROOT / v / name)
            out[f"E{e}"][v] = dict(rows=int(len(dv)), dropped=int(len(d) - len(dv)),
                                   dropped_balance_share=float(
                                       1 - dv["deposit_balance"].sum() / d["deposit_balance"].sum()))
            print(f"E{e} {v}: {len(dv):,} of {len(d):,} rows ({len(d) - len(dv):,} dropped, "
                  f"{out[f'E{e}'][v]['dropped_balance_share']:.3%} of balances)")
    (ROOT / "type5_variants.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"  wrote {ROOT}")


if __name__ == "__main__":
    main()

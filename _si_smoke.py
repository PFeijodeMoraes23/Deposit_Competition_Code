"""Smoke test for the joint single-index SLS engine (Est7 sieve) on a subsample."""
import os
os.environ["OMP_NUM_THREADS"] = "2"; os.environ["MKL_NUM_THREADS"] = "2"
os.environ["MPLBACKEND"] = "Agg"
from utils.venv_guard import ensure_project_venv
ensure_project_venv(__file__)
import numpy as np, pandas as pd, time
from estimation_2_sleep import build_pooled_data, define_specifications, run_pooled_first_stage
from utils.sleep_links import fit_joint_single_index, phi_from_native

t0 = time.time()
df = build_pooled_data()
_, iv_specs, state_blocks = define_specifications()
s_cols = state_blocks["Tech"]; iv = iv_specs["IV_HausmanFull"]
df, res_fs = run_pooled_first_stage(df, iv, [c for c in s_cols if c != "constant"])
print(f"frame+first-stage built in {time.time()-t0:.0f}s; rows={len(df)}")

# subsample for a fast machinery check (keep entities intact-ish)
sub = df.sample(60000, random_state=1).copy()
t1 = time.time()
res = fit_joint_single_index(sub, s_cols, has_cf=True, link="sieve", loss="ls",
                             n_interior=5, n_starts=2, boot_B=199, seed=1, label="smoke")
print(f"SLS fit in {time.time()-t1:.0f}s")
print("theta_native:", {k: round(v, 4) for k, v in res.params_native.items()})
print("||theta_native|| (raw):", round(float(np.linalg.norm(res.params_native.values)), 4))
print("AMEs:", {k: round(v, 5) for k, v in res.params.items()})
print("bse :", {k: round(v, 5) for k, v in res.bse.items()})
print("pval:", {k: round(v, 4) for k, v in res.pvalues.items()})
print("link grid monotone:", bool(np.all(np.diff(res.si_ggrid) >= -1e-9)),
      "| grid range:", round(float(res.si_ggrid.min()), 3), "-", round(float(res.si_ggrid.max()), 3))
sub["year_quarter"] = sub["time_id"]
phi = phi_from_native(sub, res, "sieve")
print(f"phi: min={phi.min():.3f} max={phi.max():.3f} mean={phi.mean():.3f} in[0,1]={bool((phi>=0).all() and (phi<=1).all())}")
print("DONE")

"""
sleep_joint_julia.py — production bridge to the Julia joint-sieve theta-search.

`julia_theta(...)` builds the SAME design that utils.sleep_links.fit_joint_single_index
builds internally (so the returned theta is in that function's standardised Sn frame),
estimates the index direction theta on a SUBSAMPLE of entities via sleep_joint_sieve.jl
(allocation-free, Nelder-Mead with a capped iteration count), and returns theta. The
caller then passes it to fit_joint_single_index(..., theta_fixed=theta) which refits the
monotone sieve link, AMEs, phi grid and the wild bootstrap on the FULL sample.

Why this split: the theta-search is ~90% of the cost and the index direction is a low-dim
object identified on a fraction of the data (validated: subsample-theta gives national
phi_t with corr 1.0000, max|diff| 0.004 vs the full-sample fit). The link/phi/inference
stay exact on full N in Python.
"""
import os, subprocess, tempfile, shutil
import numpy as np
from utils.sleep_links import _fast_demean, _twoway_demean

_HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.join(_HERE, "sleep_joint_sieve.jl")
SYSIMAGE = os.path.join(_HERE, "sleep_sysimage.so")          # used if present (opt 2)


def _write_bin(path, arr, dtype):
    with open(path, "wb") as f:
        f.write(np.ascontiguousarray(arr, dtype=dtype).tobytes())


def julia_theta(df, state_cols, has_cf, loss="robust", fe_time_col="time_id",
                subsample_frac=0.2, maxiter_mult=40, init_theta=None, seed=0,
                n_starts=2, n_interior=5, degree=3, threads=1,
                nbins=int(os.environ.get("SLEEP_RAMP_NBINS", "1000"))):
    """Estimate the joint-sieve index direction theta on a subsample via Julia.
    Returns theta in the standardised (Sn) frame used by fit_joint_single_index,
    or None on failure (caller should fall back to the Python search)."""
    CF_cols = ["v_hat_x_lagged_dep"] if has_cf else []
    cols = state_cols + ["deposit_balance", "nr_lagged_dep", "entity_id"]
    df_ss = df.dropna(subset=cols + CF_cols).copy()
    if len(df_ss) == 0:
        return None
    idx_cols = [c for c in state_cols if c != "constant"]
    S = df_ss[idx_cols].values.astype(float)
    S_mu = S.mean(0); S_sd = S.std(0); S_sd[S_sd <= 0] = 1.0
    Sn = (S - S_mu) / S_sd
    Z = df_ss["nr_lagged_dep"].values.astype(float)
    y = df_ss["deposit_balance"].values.astype(float)
    cf = df_ss["v_hat_x_lagged_dep"].values.astype(float) if has_cf else None
    _, einv = np.unique(df_ss["entity_id"].values, return_inverse=True)
    d = Sn.shape[1]
    twoway = fe_time_col is not None
    tinv = np.unique(df_ss[fe_time_col].values, return_inverse=True)[1] if twoway else None

    rng = np.random.RandomState(seed)
    if subsample_frac is None or subsample_frac >= 1.0:
        ridx = np.arange(len(df_ss))
    else:
        ents = np.unique(einv)
        k = max(2, int(round(subsample_frac * len(ents))))
        keep = rng.choice(ents, size=min(k, len(ents)), replace=False)
        ridx = np.where(np.isin(einv, keep))[0]

    Sn_s, Z_s = Sn[ridx], Z[ridx]
    _, eis = np.unique(einv[ridx], return_inverse=True); ec_s = np.bincount(eis).astype(float)
    if twoway:
        _, tis = np.unique(tinv[ridx], return_inverse=True); tc_s = np.bincount(tis).astype(float)
        y_dm = _twoway_demean(y[ridx], eis, ec_s, tis, tc_s)
        cf_dm = _twoway_demean(cf[ridx], eis, ec_s, tis, tc_s) if has_cf else None
    else:
        tis = tc_s = None
        y_dm = _fast_demean(y[ridx], eis, ec_s)
        cf_dm = _fast_demean(cf[ridx], eis, ec_s) if has_cf else None

    wd = tempfile.mkdtemp(prefix="jsieve_")
    try:
        _write_bin(os.path.join(wd, "Sn.bin"), np.asfortranarray(Sn_s), np.float64)  # column-major
        _write_bin(os.path.join(wd, "Z.bin"), Z_s, np.float64)
        _write_bin(os.path.join(wd, "y_dm.bin"), y_dm, np.float64)
        if has_cf:
            _write_bin(os.path.join(wd, "cf_dm.bin"), cf_dm, np.float64)
        _write_bin(os.path.join(wd, "ecode.bin"), eis, np.int32)
        if twoway:
            _write_bin(os.path.join(wd, "tcode.bin"), tis, np.int32)
        if init_theta is not None:
            _write_bin(os.path.join(wd, "init.bin"), np.asarray(init_theta, float), np.float64)
        nE = len(ec_s); nT = len(tc_s) if twoway else 0
        lines = [f"N={len(ridx)}", f"d={d}", f"nE={nE}", f"twoway={1 if twoway else 0}"]
        if twoway:
            lines.append(f"nT={nT}")
        lines += [f"has_cf={1 if has_cf else 0}", f"n_interior={n_interior}", f"degree={degree}",
                  f"loss={loss}", f"n_starts={n_starts}", f"maxiter={maxiter_mult * d}",
                  f"nbins={int(nbins)}"]
        with open(os.path.join(wd, "manifest.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")

        cmd = ["julia"]
        if os.path.exists(SYSIMAGE):
            cmd += ["--sysimage", SYSIMAGE]
        cmd += ["-t", str(threads), ENGINE, os.path.join(wd, "manifest.txt")]
        # STREAM the engine's output instead of capture_output=True. The search runs for
        # tens of minutes on the full sample, and buffering everything until exit made the
        # log look dead -- on 2026-07-29 that cost real time diagnosing a "hung" job that
        # was simply blocked on this subprocess. Echo each line as it arrives, and keep a
        # tail for the failure message.
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        tail = []
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                print(f"    [julia] {line}", flush=True)
                tail.append(line)
                del tail[:-20]
        rc = proc.wait()
        tpath = os.path.join(wd, "theta_out.bin")
        if rc != 0 or not os.path.exists(tpath):
            print(f"  [julia_theta] FAILED (rc={rc}); falling back to Python.\n"
                  "    " + "\n    ".join(tail[-8:]))
            return None
        theta = np.frombuffer(open(tpath, "rb").read(), np.float64).copy()
        return theta / (np.linalg.norm(theta) + 1e-12)
    finally:
        shutil.rmtree(wd, ignore_errors=True)

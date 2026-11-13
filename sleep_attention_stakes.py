"""
sleep_attention_stakes.py
================================================================================
W2: the stakes in the attention term of the sleepiness step, single-index routines E3 and E4.

    phi_mt = G( Upsilon'S_mt - tau0 Selic_{t-1} - tau1 stakes_mt )          (channel, A2)

The stakes are P3's S_mt (diagnostics/diag_p_shifters.py p3): the dispersion of the type-4 spreads of
a market's B banks, weighted by their type-4 balances, times the market's mean balance per
relationship. They enter the specification-12 index of the production routine as one more state
column, with their market and quarter means removed (tau1 is identified within a market over time),
beside a control function for their endogeneity (the first-stage residual) and the first stage's
controls (log HHI of type-4 balances and log number of type-4 banks at t-1, log mean balance at
t-1), treated the same way. The excluded instrument is the RECENTRED cost-shifter dispersion
(Borusyak and Hull 2023): z - mu, with z = S~ (the dispersion of the cost-fitted spreads with lagged
weights, times the lagged mean balance) and mu its expectation over the 99 within-quarter
permutations of the cost-fitted spreads across conglomerates that P3 draws (seed 20261103). The raw
instrument is reported beside it. The form of the stakes (levels in R$100 a year, or logs) is fixed
by a rule BEFORE any second stage is fitted: levels if the first stage's state-cluster effective F
of the recentred instrument is at least 23.1 (Montiel Olea and Pflueger, tau = 10%), else logs; the
other form is fitted beside it. Sample: market-quarters with at least two B banks holding time
deposits at t and at t-1 (S and S~ both defined and positive), plus every D row, whose W2 columns
are zero; the baseline is refitted on the same sample (the matched baseline).

The production kernels are called unchanged (sleep_est_e2.build_pooled_data /
define_specifications / run_pooled_first_stage; utils.sleep_links.fit_nlls_link /
fit_single_index / phi_from_native / twostage_ame_boot). The base index is the production Tech
block, read from define_specifications, so the driver follows whatever state vector production
carries. No production file is edited and no production output is written. The market panel is
read through pyarrow column subsets: sleep_est_e2.load_panel_cached is replaced, in this process
only, by a loader of the same parquet, and the panel md5 is checked by every mode.

MODES (one per call):
  --prep                    LOCAL. Rebuilds P3's cost fit, its dispersions and its 99 permutations
                            from the panel, checks them against P3's outputs (gate G-W2-1), checks
                            P3's level first stage (gate G-W2-2), and writes the market-quarter
                            inputs every other mode reads: w2_inputs.parquet, w2_draws.parquet,
                            w2_prep.json. Needs P3's CSVs and the IBGE crosswalk; nothing else does.
  --fs --est N              The first stage at the market-quarter level, both forms, raw and
                            recentred instruments: pi, F by state and by MCA (G, G*), the wild
                            restricted p by state, the permutation F and slope p over the 99 draws;
                            the form rule; the design metrics of the extended index (Panel B's
                            definition). Writes w2_fs_E{N}.json. No second stage.
  --check-base --est N      Gate G-W2-0: refits production specification 12 on the full frame and
                            compares it with the stored fit (mean phi 1e-4, Selic AME 1e-3
                            relative, n equal). Caches the stored fit to w2_prod_ref_E{N}.pkl.
  --fit --est N             The matched baseline, the stakes fit and the reduced form (S~ in place
                            of the stakes and the CF), in the chosen form and the other one; point
                            estimates (set SLEEP_BOOT_B=19 for speed, it changes no point
                            estimate). Writes w2_fit_E{N}_{model}.pkl and the attention phi
                            (est{N}/attention_phi_mt.csv, est{N}/attention_national_phi_t.csv).
  --twostage --est N --model M [--B 999 --workers W --scheme congl|mca|state]
                            The two-stage (direction + link) AME bootstrap of a stored W2 fit.
                            Writes w2_twostage_E{N}_{M}[_{scheme}].pkl.
  --placebo --est N --draws a-b
                            The joint permutation placebo: (rho, rho~) permuted together within the
                            quarter (P3's 99 draws), stakes and instrument rebuilt, first stage and
                            index refitted; writes w2_placebo_E{N}_d{a}-{b}.csv (appends per draw).
  --placebo-merge --est N   Folds the chunks into w2_placebo_E{N}.json.
  --numbers                 Writes w2_numbers.json: every number the sleep.attn.* keys will read,
                            with the md5s of the panel, the inputs and this driver.

PATHS: outputs go to $W2_OUT_DIR, else <rout_dir>/attention_stakes (rout_dir follows SLEEP_OUT_ROOT).
Inputs written by --prep are read from $W2_INPUTS_DIR, else the output directory. P3's files are read
from $W2_P3_DIR, else rout_dir. The stored production fits are read from utils.paths.est_dir(N).

    python sleep_attention_stakes.py --prep
    python sleep_attention_stakes.py --fs --est 3
    python sleep_attention_stakes.py --check-base --est 3
    python sleep_attention_stakes.py --fit --est 3
    python sleep_attention_stakes.py --twostage --est 3 --model stakes_log --B 999 --workers 64
    python sleep_attention_stakes.py --placebo --est 3 --draws 1-9
"""
import os

# BLAS pools pinned before numpy is imported (as sleep_ame_twostage.py): with --workers the children
# inherit this environment, and a fixed thread count fixes the reduction order between the fit run
# and the bootstrap run, which re-checks the fit's fingerprints.
_THR = os.environ.get("W2_BLAS_THREADS", "1")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _THR
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import argparse
import gc
import hashlib
import json
import pickle
import re
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from utils.venv_guard import ensure_project_venv
except Exception:
    ensure_project_venv = None
if ensure_project_venv is not None:
    ensure_project_venv(__file__)

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from utils import paths as _paths
from utils import routines as R

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ── Constants ─────────────────────────────────────────────────────────────────────────────────
SEED = 20261103            # P3's permutation seed (diag_p_shifters.SEED)
N_DRAWS = 99               # P3's permutations (diag_p_shifters.N_SHUFFLES)
T0, T1 = (2016, 1), (2024, 4)
MOP_CRIT = 23.1            # Montiel Olea-Pflueger critical value, one instrument, tau = 10%
FE_TIME_COL = R.FE_TIME_COL
SPEC12 = R.SPEC12
SPEC12_TAG = R.SPEC12_TAG
LINK = "index_sieve"
NLLS_N_STARTS = int(os.environ.get("SLEEP_NLLS_N_STARTS", "4"))
PANEL = _paths.PROCESSED / "market_panel.parquet"
XWALK = _paths.IBGE_DIR / "muni_mca_regions_2010_2024_panel.csv"

# The W2 columns of the index. None of them starts with "v_hat" (phi_from_native and
# fit_single_index drop parameters whose names do), none names a registered state, none says "pix".
C_STAKES, C_CF, C_LHHI, C_LNB, C_LBAL, C_Z = ("attn_stakes", "attn_cf", "attn_lhhi", "attn_lnb",
                                              "attn_lbal", "attn_z")
W2_CTRL = (C_LHHI, C_LNB, C_LBAL)
W2_ALL = (C_STAKES, C_CF, C_LHHI, C_LNB, C_LBAL, C_Z)
FORMS = ("level", "log")
T0_ = time.time()


# ── Small helpers ─────────────────────────────────────────────────────────────────────────────
def stamp(msg):
    print(f"[{datetime.now():%H:%M:%S} +{time.time() - T0_:7.0f}s] {msg}", flush=True)


def rss_gb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e9
    except Exception:
        return float("nan")


def md5_file(p) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def md5_lf(p) -> str:
    """md5 of a text file with CR removed, comparable across the CRLF local and LF cluster copies."""
    return hashlib.md5(Path(p).read_bytes().replace(b"\r", b"")).hexdigest()


def qi(year, quarter):
    return np.asarray(year, dtype=int) * 4 + np.asarray(quarter, dtype=int) - 1


def time_id_of(q):
    q = np.asarray(q, dtype=int)
    return pd.Series([f"{a}Q{b}" for a, b in zip(q // 4, q % 4 + 1)])


def codes(s) -> np.ndarray:
    return pd.factorize(pd.Series(s).to_numpy())[0]


def out_dir() -> Path:
    d = Path(os.environ.get("W2_OUT_DIR", "").strip() or (_paths.rout_dir() / "attention_stakes"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def inputs_dir() -> Path:
    return Path(os.environ.get("W2_INPUTS_DIR", "").strip() or out_dir())


def p3_dir() -> Path:
    return Path(os.environ.get("W2_P3_DIR", "").strip() or _paths.rout_dir())


def wsd_by(df, keys, x, w):
    """P3's weighted mean and standard deviation of x within groups (population formula), and the
    number of rows that enter (diag_p_shifters.wsd_by, copied so the cluster needs no diagnostics/)."""
    d = df[keys].copy()
    d["sw"], d["swx"], d["swx2"] = df[w], df[w] * df[x], df[w] * df[x] ** 2
    d["n"] = 1
    g = d.groupby(keys)[["sw", "swx", "swx2", "n"]].sum()
    mu = g["swx"] / g["sw"]
    var = (g["swx2"] / g["sw"] - mu ** 2).clip(lower=0.0)
    return pd.DataFrame({"mean": mu, "sd": np.sqrt(var), "n": g["n"]})


def resid_fe(Y, groups, tol=1e-10, maxit=10000):
    """Unweighted residuals of the columns of Y on the fixed effects `groups` by alternating
    projections: make_xi_diagnostics._resid with unit weights (the residualizer P3 uses), copied."""
    Y = np.array(Y, dtype=float, copy=True)
    if Y.ndim == 1:
        Y = Y[:, None]
    wsum = [np.bincount(g).astype(float) for g in groups]
    scale = np.sqrt((Y ** 2).mean(0)) + 1e-300
    step = np.inf
    for _ in range(maxit):
        step = 0.0
        for g, ws in zip(groups, wsum):
            for j in range(Y.shape[1]):
                m = np.bincount(g, weights=Y[:, j]) / ws
                Y[:, j] -= m[g]
                step = max(step, float(np.abs(m).max() / scale[j]))
        if len(groups) == 1 or step < tol:
            return Y
    raise RuntimeError(f"fixed-effect residuals did not converge in {maxit} sweeps (step {step:.2e})")


def gstar(labels) -> dict:
    from utils.cluster_stats import effective_cluster_stats
    s = effective_cluster_stats(pd.Series(labels).value_counts().values)
    return dict(G=int(s["G_nominal"]), Gstar=float(s["G_star"]))


def jdump(obj, path):
    Path(path).write_text(json.dumps(obj, indent=1, default=_jdefault), encoding="utf-8")


def _jdefault(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (pd.Series,)):
        return {str(k): _jdefault(v) if not isinstance(v, (int, float, str)) else v for k, v in o.items()}
    if isinstance(o, Path):
        return str(o)
    return str(o)


# ── The panel, read through pyarrow ───────────────────────────────────────────────────────────
def panel_md5(expect=None) -> str:
    m = md5_file(PANEL)
    if expect and not m.startswith(expect):
        raise SystemExit(f"market_panel.parquet md5 {m[:8]} is not the expected {expect[:8]}")
    return m


def frame_columns() -> list:
    """Every panel column build_pooled_data and demand_chars.apply read (the S5 Panel B row is
    reproduced exactly from this subset, scratchpad w2_1113 probe_panelB of 11-13)."""
    from sleep_est_e2 import define_specifications
    from utils import demand_chars
    names = pq.read_schema(PANEL).names
    stub = re.compile(r"^(dep_a|spread_a|spread_ann_a|rival_mean_spread_a)\d+$")
    _, iv, blocks = define_specifications(time_block=True)
    states = [s for s in blocks["Tech"] if s != "constant"]
    want = (["CodConglomeradoPrudencial", "mca_code", "year", "quarter", "CODMUN_IBGE", "is_B",
             "pop_total", "risk_free_qoq", "gdp_per_capita", "fraction_young"]
            + [n for n in names if stub.match(n)] + states
            + [c for c in iv["IV_RivalSpread"] if c != "rival_mean_spread"]
            + list(demand_chars.CHAR_COLS) + list(demand_chars.SEG_COLS) + ["segment"]
            + list(demand_chars.LOO_COLS) + ["estban_rival_branches_lag"])
    return [c for c in dict.fromkeys(want) if c in names]


def patch_panel_loader():
    import sleep_est_e2 as E2
    cols = frame_columns()

    def _pyarrow_loader(panel_csv, **kw):
        df = pq.read_table(PANEL, columns=cols).to_pandas()
        df["mca_code"] = df["mca_code"].astype(str)
        return df

    E2.load_panel_cached = _pyarrow_loader


def tech_block(est) -> list:
    from sleep_est_e2 import define_specifications
    return list(define_specifications(time_block=(est == 4))[2]["Tech"])


def build_frame(est) -> pd.DataFrame:
    """The production pooled frame of routine `est` (build_pooled_data), slimmed to the columns the
    kernels read."""
    from sleep_est_e2 import build_pooled_data, define_specifications
    patch_panel_loader()
    df = build_pooled_data(time_block=(est == 4))
    s_cols = tech_block(est)
    iv = define_specifications(time_block=(est == 4))[1]["IV_RivalSpread"]
    keep = (["entity_id", "time_id", "CodConglomeradoPrudencial", "deposit_type", "deposit_balance",
             "nr_lagged_dep", "lagged_deposits", "spread_qoq", "CODMUN_IBGE", "pop_total", "mca_code",
             "year", "quarter", "constant"] + [s for s in s_cols if s != "constant"] + list(iv))
    df = df[[c for c in dict.fromkeys(keep) if c in df.columns]].copy()
    df["year_quarter"] = df["time_id"]
    df["is_D_row"] = (df["CODMUN_IBGE"].astype(str) == "0").to_numpy()
    df["qi"] = qi(df["year"], df["quarter"])
    stamp(f"E{est} frame: {len(df):,} rows; rss {rss_gb():.2f} GB")
    return df


def spread_first_stage(df, s_cols):
    """sleep_est_single._exec_spec's control function for IV_RivalSpread, unchanged: v_hat and
    v_hat_x_lagged_dep on the frame given (production v_hat when the frame is the full frame)."""
    from sleep_est_e2 import define_specifications, run_pooled_first_stage
    iv_cols = define_specifications()[1]["IV_RivalSpread"]
    iv_act = [c for c in iv_cols if c in df.columns and df[c].notnull().sum() > 0]
    exog_act = [c for c in s_cols if c in df.columns and df[c].notnull().sum() > 0]
    return run_pooled_first_stage(df, iv_act, exog_act)


def market_states(df, s_cols):
    """The index's market-level states by (mca_code, time_id) from the frame's B rows: every state of
    the block but the constant and the national ones (absorbed by the quarter effects at the market
    level). Fails if a state varies within a market-quarter."""
    from utils.se_national import NATIONAL_VARS
    ms = [s for s in s_cols if s != "constant" and s not in NATIONAL_VARS]
    b = df.loc[~df["is_D_row"], ["mca_code", "time_id"] + ms]
    g = b.groupby(["mca_code", "time_id"])
    rng = float((g[ms].max() - g[ms].min()).abs().max().max()) if ms else 0.0
    if rng > 1e-9:
        raise RuntimeError(f"a market state varies within a market-quarter (max range {rng:.3e})")
    return g[ms].first().reset_index(), ms


# ── --prep: P3 rebuilt, gated, and its permutations ───────────────────────────────────────────
def _p3_panel(got):
    """P3's frame (diag_p_shifters.load_panel for the p3 columns): the panel from 2015 with the 2015
    balances of the demand-chars side file when it was built from the same panel."""
    from utils import demand_chars
    cost = list(demand_chars.COST_IV)
    cols = ["CodConglomeradoPrudencial", "mca_code", "is_B", "year", "quarter", "dep_a1", "dep_a2",
            "dep_a4", "spread_ann_a4", "spread_ann_a4_national", "rate_a4_national", "risk_free_qoq",
            *cost]
    df = pq.read_table(PANEL, columns=cols, filters=[("year", ">=", 2015)]).to_pandas()
    df = df.rename(columns={"CodConglomeradoPrudencial": "j", "mca_code": "m"})
    df["m"], df["j"] = df["m"].astype(str), df["j"].astype(str)
    if df.duplicated(["j", "m", "year", "quarter"]).any():
        raise SystemExit("panel not unique on conglomerate x market x quarter")
    side = (pq.read_schema(demand_chars.chars_path()).metadata or {}).get(
        demand_chars.PANEL_MD5_KEY, b"").decode()
    if side != got:
        raise SystemExit(f"demand_chars side file built from {side[:8]}, not {got[:8]}: P3 cannot be rebuilt")
    r = pq.read_table(demand_chars.chars_path(),
                      columns=["CodConglomeradoPrudencial", "mca_code", "year", "quarter",
                               "r__dep_a1", "r__dep_a2", "r__dep_a4"],
                      filters=[("year", "==", 2015)]).to_pandas()
    r = r.rename(columns={"CodConglomeradoPrudencial": "j", "mca_code": "m"})
    r["j"], r["m"] = r["j"].astype(str), r["m"].astype(str)
    df = df.merge(r, on=["j", "m", "year", "quarter"], how="left", validate="one_to_one")
    pre = (df["year"] < 2016) & df[["r__dep_a1", "r__dep_a2", "r__dep_a4"]].notna().any(axis=1)
    for k in (1, 2, 4):
        df[f"dep_a{k}"] = np.where(pre, df[f"r__dep_a{k}"], df[f"dep_a{k}"])
    df = df.drop(columns=["r__dep_a1", "r__dep_a2", "r__dep_a4"])
    df["qi"] = qi(df["year"], df["quarter"])
    return df, cost


def mca_state():
    """The state of each MCA: the mode of state_code over its municipalities, each at its latest MCA
    (diag_p_shifters.mca_state)."""
    xw = pd.read_csv(XWALK, usecols=["municipality_code", "year", "mca_code", "state_code"])
    xw = xw.sort_values("year").drop_duplicates("municipality_code", keep="last")
    xw["mca_code"] = xw["mca_code"].astype(str)
    return xw.groupby("mca_code")["state_code"].agg(lambda s: int(s.mode().iloc[0]))


def run_prep():
    import statsmodels.api as sm
    got = panel_md5()
    stamp(f"--prep | panel md5 {got[:8]}")
    p3csv = p3_dir() / "p3_stakes_mca_quarter.csv"
    p3fit = p3_dir() / "p3_cost_shifter_fit.csv"
    p3json = p3_dir() / "p_shifters.json"
    csv = pd.read_csv(p3csv, dtype={"mca_code": str})
    gates = {}
    csv_md5s = sorted(csv["panel_md5"].astype(str).unique())
    from utils import demand_chars
    dc = (pq.read_schema(demand_chars.chars_path()).metadata or {}).get(demand_chars.PANEL_MD5_KEY, b"").decode()
    gates["G-W2-1_md5"] = dict(panel=got[:8], p3_csv=csv_md5s, demand_chars=dc[:8],
                               ok=bool(csv_md5s == [got[:8]] and dc == got))
    if not gates["G-W2-1_md5"]["ok"]:
        raise SystemExit(f"G-W2-1: md5 mismatch {gates['G-W2-1_md5']}")

    df, cost = _p3_panel(got)
    b = df[df["is_B"] == 1].copy()
    b["dep4"] = b["dep_a4"].fillna(0.0)
    agg = dict(spread=("spread_ann_a4_national", "first"), dep4=("dep4", "sum"),
               **{c: (c, "first") for c in cost})
    cqv = b.groupby(["j", "qi"]).agg(**agg).reset_index()
    fit = cqv[(cqv["qi"] >= qi(*T0)) & (cqv["qi"] <= qi(*T1)) & (cqv["dep4"] > 0)].dropna(
        subset=["spread", *cost])
    X = pd.concat([fit[cost], pd.get_dummies(fit["qi"], prefix="q", dtype=float)], axis=1)
    res = sm.WLS(fit["spread"], X, weights=fit["dep4"]).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(fit["j"])[0]})
    beta = res.params[cost].to_numpy()
    ref = pd.read_csv(p3fit).set_index("shifter").loc[cost, "coef"].to_numpy()
    d_beta = float(np.max(np.abs(beta - ref)))
    gates["G-W2-1_cost_fit"] = dict(max_abs_diff=d_beta, ok=bool(d_beta <= 1e-10))
    cqv["rho_fit"] = cqv[cost].to_numpy() @ beta
    cqv["cq_row"] = np.arange(len(cqv))

    # Real dispersions: sd at t (cell spreads, weights Dep4 at t), sd~ (rho~, weights Dep4 at t-1),
    # and the national-spread sd at t (the placebo's object at the identity permutation).
    c4 = b[(b["dep4"] > 0) & b["spread_ann_a4"].notna()]
    sd = wsd_by(c4, ["m", "qi"], "spread_ann_a4", "dep4")["sd"]
    bt = b[b["dep4"] > 0][["j", "m", "qi", "dep4"]].merge(cqv[["j", "qi", "cq_row", "spread"]],
                                                          on=["j", "qi"], how="left")
    sd_nat = wsd_by(bt.dropna(subset=["spread"]), ["m", "qi"], "spread", "dep4")["sd"]
    lag4 = b[["j", "m", "qi", "dep4"]].assign(qi=b["qi"] + 1).rename(columns={"dep4": "dep4_lag"})
    t = b[["j", "m", "qi"]].merge(lag4, on=["j", "m", "qi"], how="left").merge(
        cqv[["j", "qi", "rho_fit", "cq_row"]], on=["j", "qi"], how="left")
    t = t[t["dep4_lag"] > 0]
    sdt = wsd_by(t.dropna(subset=["rho_fit"]), ["m", "qi"], "rho_fit", "dep4_lag")["sd"]

    csv["qi"] = qi(csv["year"], csv["quarter"])
    mq = pd.MultiIndex.from_arrays([csv["mca_code"].to_numpy(), csv["qi"].to_numpy()])
    csv["sd_spread"] = sd.reindex(mq).to_numpy()
    csv["sd_spread_nat"] = sd_nat.reindex(mq).to_numpy()
    csv["sd_rho_fit"] = sdt.reindex(mq).to_numpy()
    d_sd = float(np.nanmax(np.abs(100 * csv["sd_spread"] - csv["sd_spread_pp"])))
    d_sdt = float(np.nanmax(np.abs(100 * csv["sd_rho_fit"] - csv["sd_rho_fit_pp"])))
    nan_sd = int((csv["sd_spread"].isna() != csv["sd_spread_pp"].isna()).sum())
    nan_sdt = int((csv["sd_rho_fit"].isna() != csv["sd_rho_fit_pp"].isna()).sum())
    gates["G-W2-1_dispersions_pp"] = dict(sd_max_abs_diff=d_sd, sd_tilde_max_abs_diff=d_sdt,
                                          nan_pattern_mismatch_sd=nan_sd, nan_pattern_mismatch_sd_tilde=nan_sdt,
                                          ok=bool(d_sd <= 1e-12 and d_sdt <= 1e-12 and nan_sd == 0 and nan_sdt == 0))
    stamp(f"G-W2-1: cost fit {d_beta:.2e}; sd {d_sd:.2e} pp; sd~ {d_sdt:.2e} pp")

    # The 99 permutations, P3's rule: within each quarter (sorted), the non-missing rho~ of the
    # conglomerate-quarters are permuted by one generator in sequence. The same index permutation is
    # applied to the national spread, so (rho, rho~) move together (the joint placebo); for rho~ it is
    # P3's draw exactly (a 1-d shuffle permutes positions independently of the values).
    rng = np.random.default_rng(SEED)
    rho = cqv["rho_fit"].to_numpy()
    spr = cqv["spread"].to_numpy()
    grp = [np.flatnonzero((cqv["qi"].to_numpy() == q) & ~np.isnan(rho)) for q in np.sort(cqv["qi"].unique())]
    has_t = t["cq_row"].notna().to_numpy()
    row_t = t["cq_row"].fillna(0).to_numpy(dtype=int)
    row_b = bt["cq_row"].to_numpy(dtype=int)
    keep_mq = (csv["n_banks4"] >= 2) & (csv["n_banks4_lag"] >= 2)
    sub = csv[keep_mq].reset_index(drop=True)
    mqs = pd.MultiIndex.from_arrays([sub["mca_code"].to_numpy(), sub["qi"].to_numpy()])
    SDT = np.full((N_DRAWS, len(sub)), np.nan)
    SDP = np.full((N_DRAWS, len(sub)), np.nan)
    n_nan_spr = []
    for k in range(N_DRAWS):
        pr, ps = rho.copy(), spr.copy()
        for rows in grp:
            p = rng.permutation(len(rows))
            pr[rows] = rho[rows][p]
            ps[rows] = spr[rows][p]
        tp = t[["m", "qi", "dep4_lag"]].assign(rho_perm=np.where(has_t, pr[row_t], np.nan))
        SDT[k] = wsd_by(tp.dropna(subset=["rho_perm"]), ["m", "qi"], "rho_perm", "dep4_lag")["sd"].reindex(mqs).to_numpy()
        bp = bt[["m", "qi", "dep4"]].assign(spr_perm=ps[row_b])
        n_nan_spr.append(float(bp.loc[bp["spr_perm"].isna(), "dep4"].sum() / bp["dep4"].sum()))
        SDP[k] = wsd_by(bp.dropna(subset=["spr_perm"]), ["m", "qi"], "spr_perm", "dep4")["sd"].reindex(mqs).to_numpy()
        if (k + 1) % 20 == 0:
            stamp(f"  permutation {k + 1}/{N_DRAWS}")
    with np.errstate(divide="ignore", invalid="ignore"):
        LOGT = np.where(SDT > 0, np.log(SDT), np.nan)
    sub["mu_sd"] = SDT.mean(axis=0)
    sub["n_draws_pos"] = (SDT > 0).sum(axis=0)
    sub["mu_logsd"] = np.where(sub["n_draws_pos"] == N_DRAWS, np.nanmean(LOGT, axis=0), np.nan)

    # Reproduction of P3's recentred first stage (log sd on log sd~ - mu, MCA and quarter effects):
    # the same 99 draws give the same slope.
    p3 = json.loads(p3json.read_text(encoding="utf-8"))["p3"]
    real = ((sub["sd_spread"] > 0) & (sub["sd_rho_fit"] > 0) & (sub["S"] > 0) & (sub["S_tilde"] > 0)
            & (sub["n_draws_pos"] == N_DRAWS)).to_numpy()
    d = sub[real]
    Rr = resid_fe(np.column_stack([np.log(d["sd_spread"]), np.log(d["sd_rho_fit"]) - d["mu_logsd"]]),
                  [codes(d["mca_code"]), codes(d["qi"])])
    slope = float(sm.OLS(Rr[:, 0], Rr[:, 1:]).fit().params[0])
    ref_slope = float(p3["recentred"]["alone"]["slope"])
    gates["G-W2-1_recentred_slope"] = dict(slope=slope, p3=ref_slope, abs_diff=abs(slope - ref_slope),
                                           ok=bool(abs(slope - ref_slope) <= 1e-10))
    # G-W2-2: P3's level first stage (S on S~, MCA and quarter effects, MCA clusters), on P3's sample.
    dl = sub[(sub["S"] > 0) & (sub["S_tilde"] > 0)]
    Rl = resid_fe(np.column_stack([dl["S"], dl["S_tilde"]]), [codes(dl["mca_code"]), codes(dl["qi"])])
    rl = sm.OLS(Rl[:, 0], Rl[:, 1:]).fit(cov_type="cluster", cov_kwds={"groups": codes(dl["mca_code"])})
    f_level, ref_f = float(rl.tvalues[0] ** 2), float(p3["S_on_S_tilde"]["level"]["f"])
    gates["G-W2-2_level_F"] = dict(f=f_level, p3=ref_f, rel_diff=abs(f_level - ref_f) / ref_f,
                                   ok=bool(abs(f_level - ref_f) / ref_f <= 1e-6))
    stamp(f"G-W2-1 recentred slope {slope:.12f} vs P3 {ref_slope:.12f}; G-W2-2 level F {f_level:.3f} vs {ref_f:.3f}")

    # The identity-permutation gap between cell and national spreads (the placebo uses national ones).
    both = sub.dropna(subset=["sd_spread", "sd_spread_nat"])
    within = resid_fe(np.column_stack([both["sd_spread"], both["sd_spread"] - both["sd_spread_nat"]]),
                      [codes(both["mca_code"]), codes(both["qi"])])
    gates["identity_gap"] = dict(n=int(len(both)),
                                 max_abs_pp=float(100 * (both["sd_spread"] - both["sd_spread_nat"]).abs().max()),
                                 share_of_within_variance=float(within[:, 1].var() / within[:, 0].var()),
                                 placebo_weight_lost_median=float(np.median(n_nan_spr)),
                                 placebo_weight_lost_max=float(np.max(n_nan_spr)))
    st = mca_state()
    stc = sub["mca_code"].map(st)
    sub["state_cl"] = np.where(stc.notna(), "uf" + stc.fillna(0).astype(int).astype(str), "mca" + sub["mca_code"])
    sub["time_id"] = time_id_of(sub["qi"]).to_numpy()
    keep_cols = ["mca_code", "year", "quarter", "qi", "time_id", "state_cl", "n_banks4", "n_banks4_lag",
                 "hhi4_lag", "bal", "bal_lag", "S", "S_tilde", "sd_spread", "sd_spread_nat", "sd_rho_fit",
                 "mu_sd", "mu_logsd", "n_draws_pos"]
    inp = sub[keep_cols].copy()
    draws = pd.DataFrame({"mca_code": sub["mca_code"], "qi": sub["qi"]})
    draws = pd.concat([draws,
                       pd.DataFrame(SDT.T, columns=[f"sdt_{k + 1:03d}" for k in range(N_DRAWS)]),
                       pd.DataFrame(SDP.T, columns=[f"sdp_{k + 1:03d}" for k in range(N_DRAWS)])], axis=1)
    od = out_dir()
    inp.to_parquet(od / "w2_inputs.parquet", index=False)
    draws.to_parquet(od / "w2_draws.parquet", index=False)
    ok = all(v.get("ok", True) for v in gates.values())
    meta = dict(written=datetime.now().isoformat(timespec="seconds"), panel_md5=got, seed=SEED,
                n_draws=N_DRAWS, n_mq=int(len(inp)), gates=gates, ok=ok,
                p3_csv_md5=md5_lf(p3csv), p3_fit_md5=md5_lf(p3fit), driver_md5=md5_lf(__file__),
                inputs_md5=md5_file(od / "w2_inputs.parquet"), draws_md5=md5_file(od / "w2_draws.parquet"))
    jdump(meta, od / "w2_prep.json")
    after = panel_md5()
    stamp(f"--prep done: {len(inp):,} market-quarters; gates {'PASS' if ok else 'FAIL'}; panel md5 after {after[:8]}")
    if not ok or after != got:
        raise SystemExit("--prep: a gate failed or the panel changed")


# ── Shared: the inputs, the sample, the first stage, the W2 columns ──────────────────────────
def load_inputs(need_draws=False):
    d = inputs_dir()
    meta = json.loads((d / "w2_prep.json").read_text(encoding="utf-8"))
    got = panel_md5()
    if meta["panel_md5"] != got:
        raise SystemExit(f"w2 inputs were built from panel {meta['panel_md5'][:8]}, not {got[:8]}")
    if not meta.get("ok"):
        raise SystemExit("w2_prep.json records a failed gate")
    inp = pd.read_parquet(d / "w2_inputs.parquet")
    inp["mca_code"] = inp["mca_code"].astype(str)
    draws = None
    if need_draws:
        draws = pd.read_parquet(d / "w2_draws.parquet")
        draws["mca_code"] = draws["mca_code"].astype(str)
    return inp, draws, meta


def w2_sample(inp, mqs, ms):
    """The W2 market-quarters: S, S~, both dispersions and every permuted dispersion positive, two or
    more banks at t and t-1, structure and balance controls defined, and a B row in the frame."""
    d = inp.merge(mqs, on=["mca_code", "time_id"], how="inner")
    ok = ((d["S"] > 0) & (d["S_tilde"] > 0) & (d["sd_spread"] > 0) & (d["sd_rho_fit"] > 0)
          & (d["n_draws_pos"] == N_DRAWS) & (d["n_banks4"] >= 2) & (d["n_banks4_lag"] >= 2)
          & (d["hhi4_lag"] > 0) & (d["bal_lag"] > 0) & d[ms].notna().all(axis=1))
    return d[ok].reset_index(drop=True)


def form_vars(d, form, sd_t=None, sdt_r=None):
    """(y, z_raw, mu, z_recentred) in the form's units. Levels: R$100 a year. Logs: log points.
    sd_t replaces the stakes dispersion (the placebo); sdt_r gives a permuted instrument."""
    sdt = d["sd_spread"].to_numpy() if sd_t is None else sd_t
    sdz = d["sd_rho_fit"].to_numpy() if sdt_r is None else sdt_r
    bal, bal_lag = d["bal"].to_numpy(), d["bal_lag"].to_numpy()
    if form == "level":
        y = sdt * bal / 100.0
        z = sdz * bal_lag / 100.0
        mu = d["mu_sd"].to_numpy() * bal_lag / 100.0
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.log(sdt * bal)
            z = np.log(sdz * bal_lag)
        mu = d["mu_logsd"].to_numpy() + np.log(bal_lag)
    return y, z, mu, z - mu


def controls(d, ms):
    W = np.column_stack([np.log(d["hhi4_lag"]), np.log(d["n_banks4_lag"]), np.log(d["bal_lag"])])
    return np.column_stack([W] + [d[s].to_numpy(float)[:, None] for s in ms]) if ms else W


def first_stage(d, y, z, C, clusters=("state_cl", "mca_code"), wcr=False):
    """y on z and the controls C with MCA and quarter effects (residualized), OLS, clustered by state
    and by MCA. Returns (stats, residualized [y, z, C], first-stage residual v)."""
    import statsmodels.api as sm
    Rr = resid_fe(np.column_stack([y, z, C]), [codes(d["mca_code"]), codes(d["qi"])])
    yr = pd.Series(Rr[:, 0], name="y")
    Xr = pd.DataFrame(Rr[:, 1:], columns=["z"] + [f"c{i}" for i in range(Rr.shape[1] - 2)])
    st = {}
    v = None
    for cl in clusters:
        g = codes(d[cl])
        r = sm.OLS(yr, Xr).fit(cov_type="cluster", cov_kwds={"groups": g})
        st[cl] = dict(pi=float(r.params["z"]), se=float(r.bse["z"]), f=float(r.tvalues["z"] ** 2),
                      **gstar(d[cl].to_numpy()))
        if v is None:
            v = (yr - Xr @ r.params).to_numpy()
        if wcr and cl == "state_cl":
            from utils.sleep_links import linear_wild_cluster_bootstrap
            _, _, pv = linear_wild_cluster_bootstrap(r, B=999, scheme="webb", seed=0, mode="wcr")
            st[cl]["p_wcr"] = float(pv["z"])
    st["n"] = int(len(d))
    st["corr_raw"] = float(np.corrcoef(y, z)[0, 1])
    return st, Rr, v


def w2_columns(d, form, ms, sd_t=None, sdt_r=None, rf=False):
    """The market-quarter W2 columns of one form: the stakes, the CF and the controls with their
    market and quarter means removed (one residualization), and, for the reduced form, the recentred
    instrument the same way."""
    y, z, mu, zc = form_vars(d, form, sd_t, sdt_r)
    C = controls(d, ms)
    st, Rr, v = first_stage(d, y, zc, C, clusters=("state_cl",))
    cols = pd.DataFrame({"mca_code": d["mca_code"].to_numpy(), "time_id": d["time_id"].to_numpy(),
                         C_STAKES: Rr[:, 0], C_CF: v, C_LHHI: Rr[:, 2], C_LNB: Rr[:, 3], C_LBAL: Rr[:, 4]})
    if rf:
        cols[C_Z] = Rr[:, 1]
    return cols, st


def attach(df, cols, model_cols):
    """The frame restricted to the W2 sample: every D row (W2 columns 0) and the B rows of the W2
    market-quarters (their values). Returns a new frame."""
    keep = df["is_D_row"] | pd.MultiIndex.from_arrays([df["mca_code"], df["time_id"]]).isin(
        pd.MultiIndex.from_arrays([cols["mca_code"], cols["time_id"]]))
    out = df.loc[keep].copy()
    m = out[["mca_code", "time_id"]].merge(cols, on=["mca_code", "time_id"], how="left")
    for c in model_cols:
        v = m[c].to_numpy()
        out[c] = np.where(out["is_D_row"].to_numpy(), 0.0, v)
    if out.loc[~out["is_D_row"], list(model_cols)].isna().any().any():
        raise RuntimeError("a kept B row has no W2 value")
    return out


MODEL_COLS = {"matched": (), "stakes": (C_STAKES, C_CF, C_LHHI, C_LNB, C_LBAL),
              "rf": (C_Z, C_LHHI, C_LNB, C_LBAL)}


def model_s_cols(est, kind):
    return tech_block(est) + list(MODEL_COLS[kind])


def prepare(est, form, need_draws=False):
    """Frame, production spread CF, market states, inputs and the W2 market-quarter sample."""
    inp, draws, meta = load_inputs(need_draws)
    df = build_frame(est)
    s_cols = tech_block(est)
    df, _ = spread_first_stage(df, s_cols)
    mqs, ms = market_states(df, s_cols)
    d = w2_sample(inp, mqs, ms)
    stamp(f"W2 sample: {len(d):,} market-quarters, {d['mca_code'].nunique():,} markets; market states {ms}")
    return df, d, ms, draws, meta


# ── --fs ──────────────────────────────────────────────────────────────────────────────────────
def design_metrics(df_rows, s_cols):
    """Panel B's metrics (diag_state_vector_selection.design_metrics): the two-way-demeaned design
    [Dep_lag x NR-factor, states x it, v_hat x Dep_lag], its condition number, effective dimension
    and VIF by column."""
    from sleep_est_e2 import demean_variables_2way
    names = ["nr_lagged_dep"] + [f"interaction_{s}" for s in s_cols if s != "constant"] + ["v_hat_x_lagged_dep"]
    work = df_rows[["entity_id", "time_id", "deposit_balance", "nr_lagged_dep", "v_hat_x_lagged_dep"]
                   + [s for s in s_cols if s != "constant"]].copy()
    for s in s_cols:
        if s != "constant":
            work[f"interaction_{s}"] = work[s] * work["nr_lagged_dep"]
    work = work.dropna(subset=names + ["deposit_balance"])
    X = demean_variables_2way(work, names, "entity_id", "time_id").to_numpy(float)
    Z = (X - X.mean(0)) / X.std(0)
    C = np.corrcoef(Z, rowvar=False)
    lam = np.clip(np.linalg.eigvalsh(C)[::-1], 1e-300, None)
    vif = np.diag(np.linalg.inv(C))
    return dict(n=int(len(work)), cond=float(np.sqrt(lam[0] / lam[-1])),
                effdim=float(lam.sum() ** 2 / (lam ** 2).sum()), max_vif=float(vif.max()),
                max_vif_col=names[int(vif.argmax())], vif={n: float(v) for n, v in zip(names, vif)})


def run_fs(est):
    df, d, ms, draws, meta = prepare(est, None, need_draws=True)
    dr = d[["mca_code", "qi"]].merge(draws, on=["mca_code", "qi"], how="left")
    SDT = dr[[f"sdt_{k + 1:03d}" for k in range(N_DRAWS)]].to_numpy().T
    C = controls(d, ms)
    res = dict(est=est, n_mq=int(len(d)), n_markets=int(d["mca_code"].nunique()), market_states=ms,
               panel_md5=meta["panel_md5"], inputs_md5=meta["inputs_md5"], driver_md5=md5_lf(__file__))
    for form in FORMS:
        y, z, mu, zc = form_vars(d, form)
        out = {}
        for lab, zz in (("raw", z), ("recentred", zc)):
            st, Rr, _ = first_stage(d, y, zz, C, wcr=True)
            out[lab] = st
        out["corr_z_mu"] = float(np.corrcoef(z, mu)[0, 1])
        # The permutation of the instrument (P3's 99 draws), recentred with the same mu, real y.
        Rc = resid_fe(np.column_stack([y, C]), [codes(d["mca_code"]), codes(d["qi"])])
        sl, fs_ = [], []
        import statsmodels.api as sm
        g_state = codes(d["state_cl"])
        for k in range(N_DRAWS):
            _, zk, _, zck = form_vars(d, form, sdt_r=SDT[k])
            zr = resid_fe(zck, [codes(d["mca_code"]), codes(d["qi"])])[:, 0]
            r = sm.OLS(Rc[:, 0], np.column_stack([zr, Rc[:, 1:]])).fit(
                cov_type="cluster", cov_kwds={"groups": g_state})
            sl.append(float(r.params[0]))
            fs_.append(float(r.tvalues[0] ** 2))
        sl, fs_ = np.array(sl), np.array(fs_)
        b0, f0 = out["recentred"]["mca_code"]["pi"], out["recentred"]["state_cl"]["f"]
        out["permutation"] = dict(n=N_DRAWS, seed=SEED, slope_median=float(np.median(sl)),
                                  slope_q05=float(np.quantile(sl, 0.05)), slope_q95=float(np.quantile(sl, 0.95)),
                                  se_ri=float(sl.std(ddof=1)), t_ri=float(b0 / sl.std(ddof=1)),
                                  n_abs_ge=int((np.abs(sl) >= abs(b0)).sum()),
                                  p_ri=float((1 + (np.abs(sl) >= abs(b0)).sum()) / (1 + N_DRAWS)),
                                  f_state_median=float(np.median(fs_)),
                                  share_f_state_ge=float((fs_ >= f0).mean()))
        res[form] = out
        stamp(f"E{est} {form}: recentred pi {out['recentred']['state_cl']['pi']:.4f}, F state "
              f"{out['recentred']['state_cl']['f']:.2f} (WCR p {out['recentred']['state_cl'].get('p_wcr', float('nan')):.3f}), "
              f"F MCA {out['recentred']['mca_code']['f']:.2f}; raw F state {out['raw']['state_cl']['f']:.2f}; "
              f"permutation p {out['permutation']['p_ri']:.3f}, t_ri {out['permutation']['t_ri']:.2f}")
    f_level = res["level"]["recentred"]["state_cl"]["f"]
    form = "level" if f_level >= MOP_CRIT else "log"
    res["form_rule"] = dict(criterion=MOP_CRIT, f_state_level_recentred=f_level, form_chosen=form,
                            form_other=[f for f in FORMS if f != form][0])
    # Design metrics of the extended index (chosen form) against the baseline on the same sample.
    s_cols = tech_block(est)
    cols, _ = w2_columns(d, form, ms)
    dfw = attach(df, cols, MODEL_COLS["stakes"])
    res["design"] = dict(baseline_w2_sample=design_metrics(dfw, s_cols),
                         stakes=design_metrics(dfw, s_cols + list(MODEL_COLS["stakes"])))
    bk = ~dfw["is_D_row"] & dfw["deposit_type"].isin([4, 5]) & dfw["v_hat"].notna()
    res["corr_spread_cf_stakes_cf"] = float(np.corrcoef(dfw.loc[bk, "v_hat"], dfw.loc[bk, C_CF])[0, 1])
    res["sample_rows"] = sample_stats(df, dfw)
    jdump(res, out_dir() / f"w2_fs_E{est}.json")
    stamp(f"E{est}: form rule -> {form} (level recentred F state {f_level:.2f} vs {MOP_CRIT}); design "
          f"cond {res['design']['stakes']['cond']:.2f} vs baseline {res['design']['baseline_w2_sample']['cond']:.2f}")


def sample_stats(df, dfw):
    b_all = ~df["is_D_row"]
    b_w = ~dfw["is_D_row"]
    return dict(rows_frame=int(len(df)), rows_w2=int(len(dfw)), b_rows_frame=int(b_all.sum()),
                b_rows_w2=int(b_w.sum()),
                b_dep_share_kept=float(dfw.loc[b_w, "deposit_balance"].sum() / df.loc[b_all, "deposit_balance"].sum()),
                congl=gstar(dfw["CodConglomeradoPrudencial"].astype(str).to_numpy()))


# ── --check-base: gate G-W2-0 ─────────────────────────────────────────────────────────────────
def load_prod_ref(est):
    """The stored production specification-12 fit, cached slim (the estimator pickle is ~1 GB)."""
    pkl = _paths.est_dir(est) / "estimation_results.pkl"
    cache = out_dir() / f"w2_prod_ref_E{est}.pkl"
    src = dict(path=str(pkl), size=pkl.stat().st_size, mtime_ns=pkl.stat().st_mtime_ns)
    if cache.exists():
        with open(cache, "rb") as fh:
            ref = pickle.load(fh)
        if ref.get("source", {}).get("size") == src["size"] and ref["source"].get("mtime_ns") == src["mtime_ns"]:
            return ref
    with open(pkl, "rb") as fh:
        d = pickle.load(fh)
    e = d[SPEC12]
    fs = e.get("first_stage")
    ref = dict(si=e["second_stage"], fs_params=(fs.params.copy() if fs is not None else None),
               fs_nobs=(int(fs.nobs) if fs is not None else None), source=src)
    del d, e, fs
    gc.collect()
    with open(cache, "wb") as fh:
        pickle.dump(ref, fh)
    return ref


def national_mean_phi(df, phi, b_only=False):
    """Population-weighted national phi_t over (quarter, market) cells, as sleep_est_single.
    _calculate_phis (all markets, the NATIONAL cell included) or over the B markets only."""
    t = pd.DataFrame({"yq": df["year_quarter"].to_numpy(), "m": df["mca_code"].to_numpy(), "phi": phi,
                      "M": df["pop_total"].fillna(0).to_numpy(), "D": df["is_D_row"].to_numpy()})
    if b_only:
        t = t[~t["D"]]
    a = t.groupby(["yq", "m"], observed=True).agg(phi=("phi", "mean"), M=("M", "mean")).reset_index()
    num = (a["phi"] * a["M"]).groupby(a["yq"]).sum()
    den = a["M"].groupby(a["yq"]).sum().replace(0, np.nan)
    return (num / den).fillna(0)


def fit_si(df_w, s_cols, n_starts=None, init=None):
    from utils.sleep_links import fit_nlls_link, fit_single_index
    lg = fit_nlls_link(df_w, s_cols, has_cf=True, link="logit", loss="cauchy", fe_time_col=FE_TIME_COL,
                       bootstrap=False, init=init, n_starts=(n_starts or NLLS_N_STARTS))
    if lg is None:
        raise RuntimeError("the NLLS direction failed")
    return fit_single_index(df_w, s_cols, has_cf=True, logit_res=lg, degree=3, fe_time_col=FE_TIME_COL,
                            phi_band=False)


def run_check_base(est):
    from utils.sleep_links import phi_from_native
    got = panel_md5()
    ref = load_prod_ref(est)
    stamp(f"E{est}: stored fit loaded (n {ref['si'].nobs:,}); rss {rss_gb():.2f} GB")
    df = build_frame(est)
    s_cols = tech_block(est)
    df, res_fs = spread_first_stage(df, s_cols)
    d_fs = (float(np.max(np.abs(res_fs.params.reindex(ref["fs_params"].index) - ref["fs_params"])
                         / np.maximum(np.abs(ref["fs_params"]), 1e-12))) if ref["fs_params"] is not None else None)
    si = fit_si(df, s_cols)
    phi = phi_from_native(df, si, LINK)
    mp = float(national_mean_phi(df, phi).mean())
    nat = pd.read_csv(_paths.est_dir(est) / "national_phi_t.csv")
    mp_ref = float(nat[f"phi_t_{SPEC12_TAG}"].mean())
    st = ref["si"]
    k = "interaction_risk_free_qoq_lag"
    sel_rel = (abs(float(si.params[k]) - float(st.params[k])) / abs(float(st.params[k]))
               if k in si.params.index else None)
    d_th = float(np.max(np.abs(si.params_native - st.params_native.reindex(si.params_native.index))))
    ok = (abs(mp - mp_ref) <= 1e-4 and (sel_rel is None or sel_rel <= 1e-3) and int(si.nobs) == int(st.nobs))
    out = dict(est=est, gate="G-W2-0", ok=bool(ok), mean_phi=mp, mean_phi_stored=mp_ref,
               abs_diff_mean_phi=abs(mp - mp_ref), selic_ame=float(si.params.get(k, np.nan)),
               selic_ame_stored=float(st.params.get(k, np.nan)), selic_rel_diff=sel_rel,
               n=int(si.nobs), n_stored=int(st.nobs), max_abs_theta_diff=d_th,
               first_stage_max_rel_diff=d_fs, s_cols=s_cols, panel_md5=got,
               sleep_pix_state=os.environ.get("SLEEP_PIX_STATE", ""), rss_gb=rss_gb(),
               seconds=round(time.time() - T0_, 1), driver_md5=md5_lf(__file__))
    jdump(out, out_dir() / f"w2_check_base_E{est}.json")
    stamp(f"G-W2-0 E{est}: mean phi {mp:.6f} vs stored {mp_ref:.6f}; Selic AME rel diff "
          f"{sel_rel if sel_rel is None else f'{sel_rel:.2e}'}; n {si.nobs:,} vs {st.nobs:,}; max |d theta| "
          f"{d_th:.2e}; first stage {d_fs if d_fs is None else f'{d_fs:.2e}'} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(3)


# ── --fit ─────────────────────────────────────────────────────────────────────────────────────
def fs_json(est):
    p = out_dir() / f"w2_fs_E{est}.json"
    if not p.exists():
        raise SystemExit(f"{p.name} is missing: run --fs first (the form is fixed before any second stage)")
    return json.loads(p.read_text(encoding="utf-8"))


def fit_path(est, model):
    return out_dir() / f"w2_fit_E{est}_{model}.pkl"


def run_fit(est, models):
    from utils.sleep_links import phi_from_native
    fsj = fs_json(est)
    form, other = fsj["form_rule"]["form_chosen"], fsj["form_rule"]["form_other"]
    df, d, ms, _, meta = prepare(est, form)
    s_base = tech_block(est)
    todo = []
    for m in models:
        if m == "matched":
            todo.append(("matched", "matched", form))
        else:
            kind, fm = m.split("_")
            todo.append((m, kind, {"chosen": form, "other": other}.get(fm, fm)))
    fits = {}
    for name, kind, fm in todo:
        cols, st = w2_columns(d, fm, ms, rf=(kind == "rf"))
        dfw = attach(df, cols, MODEL_COLS[kind] if kind != "matched" else MODEL_COLS["stakes"])
        s_cols = model_s_cols(est, kind)
        stamp(f"E{est} {name} ({fm}): {len(dfw):,} rows; index {len(s_cols)} columns")
        si = fit_si(dfw, s_cols)
        label = name if kind == "matched" else f"{kind}_{fm}"
        payload = dict(schema=1, est=est, model=label, kind=kind, form=fm, s_cols=s_cols, si=si,
                       fs_stats=st, n=int(si.nobs), sample=sample_stats(df, dfw),
                       panel_md5=meta["panel_md5"], inputs_md5=meta["inputs_md5"],
                       driver_md5=md5_lf(__file__), sleep_pix_state=os.environ.get("SLEEP_PIX_STATE", ""),
                       sleep_boot_b=os.environ.get("SLEEP_BOOT_B", ""),
                       written=datetime.now().isoformat(timespec="seconds"))
        with open(fit_path(est, label), "wb") as fh:
            pickle.dump(payload, fh)
        fits[label] = payload
        a = si.params
        msg = ", ".join(f"{k.replace('interaction_', '')} {float(a[k]):+.4g}" for k in a.index
                        if k.replace("interaction_", "") in ("risk_free_qoq_lag",) + W2_ALL)
        stamp(f"E{est} {label}: n {si.nobs:,}; AMEs {msg}; saturated {getattr(si, 'si_qp', {}).get('saturated_pct', float('nan')):.2f}%")
        # The attention phi, from the stakes fit in the chosen form.
        if kind == "stakes" and fm == form:
            write_attention_phi(est, df, cols, si, fits.get("matched", {}).get("si"), form)
        del dfw
        gc.collect()


def write_attention_phi(est, df, cols, si, si_matched, form):
    """Per market-quarter phi of the attention fit, in the form the demand prep would merge: B
    market-quarters of the W2 sample carry their W2 values, every other row is evaluated with the W2
    columns at zero (no deviation from the market's own average stakes). The national series is the
    population-weighted mean over the B markets, as the fixed prep forms phi_t."""
    from utils.sleep_links import phi_from_native
    e = df.copy()
    m = e[["mca_code", "time_id"]].merge(cols, on=["mca_code", "time_id"], how="left")
    ins = (~e["is_D_row"]).to_numpy() & m[C_STAKES].notna().to_numpy()
    for c in MODEL_COLS["stakes"]:
        e[c] = np.where(ins, m[c].to_numpy(), 0.0)
    e["phi_attn"] = phi_from_native(e, si, LINK)
    if si_matched is not None:
        e["phi_matched"] = phi_from_native(e, si_matched, LINK)
    try:
        ref = load_prod_ref(est)
        e["phi_baseline"] = phi_from_native(e, ref["si"], LINK)
    except Exception as exc:
        stamp(f"  baseline phi not written ({type(exc).__name__}: {exc})")
    e["in_sample"] = ins.astype(int)
    pcols = [c for c in ("phi_attn", "phi_matched", "phi_baseline") if c in e.columns]
    g = e.groupby(["mca_code", "time_id"], observed=True)
    mq = g.agg(year=("year", "first"), quarter=("quarter", "first"), is_D=("is_D_row", "first"),
               pop_total=("pop_total", "mean"), in_sample=("in_sample", "max"),
               **{c: (c, "mean") for c in pcols}).reset_index()
    rng = float(max((g[c].max() - g[c].min()).max() for c in pcols))
    if rng > 1e-12:
        raise RuntimeError(f"phi varies within a market-quarter ({rng:.3e})")
    od = out_dir() / f"est{est}"
    od.mkdir(parents=True, exist_ok=True)
    mq["is_B"] = ~mq["is_D"]
    mq.drop(columns=["is_D"]).to_csv(od / "attention_phi_mt.csv", index=False)
    b = mq[mq["is_B"]]
    nat = pd.DataFrame({"year_quarter": sorted(b["time_id"].unique())})
    for c in pcols:
        w = b["pop_total"].fillna(0)
        s = (b[c] * w).groupby(b["time_id"]).sum() / w.groupby(b["time_id"]).sum().replace(0, np.nan)
        nat[f"phi_t_{c.replace('phi_', '')}"] = nat["year_quarter"].map(s).to_numpy()
    nat.to_csv(od / "attention_national_phi_t.csv", index=False)
    jdump(dict(est=est, form=form, n_mq=int(len(mq)), n_mq_in_sample=int(mq["in_sample"].sum()),
               b_pop_share_in_sample=float(b.loc[b["in_sample"] == 1, "pop_total"].sum() / b["pop_total"].sum()),
               max_within_mq_range=rng, written=datetime.now().isoformat(timespec="seconds"),
               files=["attention_phi_mt.csv", "attention_national_phi_t.csv"]),
          od / "attention_phi_meta.json")
    stamp(f"E{est}: attention phi written ({len(mq):,} market-quarters; mean national phi_t "
          + ", ".join(f"{c} {nat[f'phi_t_' + c.replace('phi_', '')].mean():.4f}" for c in pcols) + ")")


# ── --twostage ────────────────────────────────────────────────────────────────────────────────
def load_fit(est, model):
    p = fit_path(est, model)
    if not p.exists():
        raise SystemExit(f"{p.name} is missing: run --fit first")
    with open(p, "rb") as fh:
        return pickle.load(fh)


def run_twostage(est, model, B, workers, scheme):
    from utils.sleep_links import twostage_ame_boot
    fit = load_fit(est, model)
    df, d, ms, _, meta = prepare(est, fit["form"])
    if meta["inputs_md5"] != fit["inputs_md5"]:
        raise SystemExit("the fit was made on other W2 inputs")
    kind = fit["kind"]
    cols, _ = w2_columns(d, fit["form"], ms, rf=(kind == "rf"))
    dfw = attach(df, cols, MODEL_COLS[kind] if kind != "matched" else MODEL_COLS["stakes"])
    if scheme == "mca":
        dfw["CodConglomeradoPrudencial"] = dfw["mca_code"].astype(str)
    elif scheme == "state":
        st = d.drop_duplicates("mca_code").set_index("mca_code")["state_cl"]
        dfw["CodConglomeradoPrudencial"] = dfw["mca_code"].map(st).fillna("NATIONAL").astype(str)
    out = twostage_ame_boot(dfw, fit["s_cols"], True, fit["si"], "cauchy", degree=3, fe_time_col=FE_TIME_COL,
                            B=B, seed=0, workers=workers, keep_draws=True)
    from sleep_ame_twostage import counters_ok
    bad = counters_ok(out, B)
    out["meta"].update(est=est, model=model, form=fit["form"], scheme=scheme, counters_ok=not bad,
                       counters_bad=bad, driver_md5=md5_lf(__file__), panel_md5=meta["panel_md5"],
                       created=datetime.now().isoformat(timespec="seconds"))
    tag = "" if scheme == "congl" else f"_{scheme}"
    with open(out_dir() / f"w2_twostage_E{est}_{model}{tag}.pkl", "wb") as fh:
        pickle.dump(out, fh)
    for s in ("congl", "quarter"):
        for _, r in out[s]["band"].iterrows():
            if str(r["name"]).replace("interaction_", "") in ("risk_free_qoq_lag",) + W2_ALL:
                stamp(f"  [{s}] {r['name']:32s} ame {r['ame']:+.4e} BC [{r['lo_bc']:+.4e}, {r['hi_bc']:+.4e}] "
                      f"p_bc {r['p_bc']:.4f}{r['stars']}")
    stamp(f"E{est} {model} two-stage B={B} ({scheme}): "
          + ("counters ok" if not bad else "DEGRADED: " + "; ".join(bad)))


# ── --placebo ─────────────────────────────────────────────────────────────────────────────────
def run_placebo(est, a, b, starts):
    fsj = fs_json(est)
    form = fsj["form_rule"]["form_chosen"]
    real = load_fit(est, f"stakes_{form}")
    df, d0, ms, draws, meta = prepare(est, form, need_draws=True)
    dr = d0[["mca_code", "qi"]].merge(draws, on=["mca_code", "qi"], how="left")
    s_cols = model_s_cols(est, "stakes")
    path = out_dir() / f"w2_placebo_E{est}_d{a:03d}-{b:03d}.csv"
    done = set(pd.read_csv(path)["draw"]) if path.exists() else set()
    tau1_real = -float(real["si"].params[f"interaction_{C_STAKES}"])
    for k in range(a, b + 1):
        if k in done:
            continue
        t0 = time.time()
        sdp = dr[f"sdp_{k:03d}"].to_numpy()
        sdt = dr[f"sdt_{k:03d}"].to_numpy()
        ok = np.isfinite(sdp) & (sdp > 0)
        d = d0[ok].reset_index(drop=True)
        cols, st = w2_columns(d, form, ms, sd_t=sdp[ok], sdt_r=sdt[ok])
        dfw = attach(df, cols, MODEL_COLS["stakes"])
        si = fit_si(dfw, s_cols, n_starts=starts)
        row = dict(draw=k, n_mq=int(len(d)), n=int(si.nobs), pi=st["state_cl"]["pi"], f_state=st["state_cl"]["f"],
                   tau1=-float(si.params[f"interaction_{C_STAKES}"]),
                   tau0=(-float(si.params["interaction_risk_free_qoq_lag"])
                         if "interaction_risk_free_qoq_lag" in si.params.index else np.nan),
                   theta_cf=float(si.params[f"interaction_{C_CF}"]), tau1_real=tau1_real,
                   seconds=round(time.time() - t0, 1))
        pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)
        stamp(f"placebo E{est} draw {k}: tau1 {row['tau1']:+.4e} (real {tau1_real:+.4e}); F state "
              f"{row['f_state']:.2f}; n {row['n']:,} [{row['seconds']:.0f}s, rss {rss_gb():.2f} GB]")
        del dfw
        gc.collect()


def run_placebo_merge(est):
    parts = sorted(out_dir().glob(f"w2_placebo_E{est}_d*.csv"))
    p = pd.concat([pd.read_csv(x) for x in parts], ignore_index=True).drop_duplicates("draw")
    t1 = float(p["tau1_real"].iloc[0])
    n_ge = int((p["tau1"].abs() >= abs(t1)).sum())
    out = dict(est=est, R=int(len(p)), seed=SEED, files=[x.name for x in parts], tau1_real=t1,
               tau1_median=float(p["tau1"].median()), tau1_q05=float(p["tau1"].quantile(0.05)),
               tau1_q95=float(p["tau1"].quantile(0.95)), n_abs_ge=n_ge,
               share_abs_ge=float(n_ge / len(p)), p_perm=float((1 + n_ge) / (1 + len(p))),
               f_state_median=float(p["f_state"].median()))
    jdump(out, out_dir() / f"w2_placebo_E{est}.json")
    stamp(f"placebo E{est}: R {out['R']}; median tau1 {out['tau1_median']:+.4e}; p {out['p_perm']:.3f}")


# ── --numbers ─────────────────────────────────────────────────────────────────────────────────
def run_numbers():
    od = out_dir()
    num = dict(written=datetime.now().isoformat(timespec="seconds"), driver_md5=md5_lf(__file__),
               panel_md5=panel_md5(), prep=json.loads((inputs_dir() / "w2_prep.json").read_text(encoding="utf-8")),
               routines={})
    for est in (3, 4):
        e = {}
        for nm in (f"w2_fs_E{est}.json", f"w2_check_base_E{est}.json", f"w2_placebo_E{est}.json"):
            if (od / nm).exists():
                e[nm.split(".")[0]] = json.loads((od / nm).read_text(encoding="utf-8"))
        fits = {}
        for p in sorted(od.glob(f"w2_fit_E{est}_*.pkl")):
            with open(p, "rb") as fh:
                f = pickle.load(fh)
            a = f["si"].params
            fits[f["model"]] = dict(form=f["form"], n=f["n"], ame={k: float(v) for k, v in a.items()},
                                    sample=f["sample"], fs_stats=f["fs_stats"])
        e["fits"] = fits
        ts = {}
        for p in sorted(od.glob(f"w2_twostage_E{est}_*.pkl")):
            with open(p, "rb") as fh:
                o = pickle.load(fh)
            ts[p.stem] = {s: o[s]["band"].to_dict(orient="records") for s in ("congl", "quarter")}
            ts[p.stem]["meta"] = {k: o["meta"].get(k) for k in ("B", "scheme", "counters_ok", "counters_bad", "n", "n_cl", "n_t")}
            ts[p.stem]["n_cos_idx_neg"] = {s: o[s].get("n_cos_idx_neg") for s in ("congl", "quarter")}
        e["twostage"] = ts
        num["routines"][f"E{est}"] = e
    jdump(num, od / "w2_numbers.json")
    stamp(f"wrote {od / 'w2_numbers.json'}")


# ── CLI ───────────────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="W2: the stakes in the attention term of E3/E4 (see the header).")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--prep", action="store_true")
    g.add_argument("--fs", action="store_true")
    g.add_argument("--check-base", action="store_true")
    g.add_argument("--fit", action="store_true")
    g.add_argument("--twostage", action="store_true")
    g.add_argument("--placebo", action="store_true")
    g.add_argument("--placebo-merge", action="store_true")
    g.add_argument("--numbers", action="store_true")
    p.add_argument("--est", type=int, choices=(3, 4))
    p.add_argument("--models", default="matched,stakes_chosen,rf_chosen,stakes_other,rf_other",
                   help="--fit: comma list of matched, stakes_{chosen|other|level|log}, rf_{...}")
    p.add_argument("--model", help="--twostage: a stored fit, e.g. stakes_log, rf_log, matched")
    p.add_argument("--B", type=int, default=None, help="--twostage: draws (default SLEEP_BOOT_B, else 999)")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--scheme", choices=("congl", "mca", "state"), default="congl",
                   help="--twostage: the cluster of the second stage (mca/state: a frame copy whose "
                        "conglomerate column holds the market or its state)")
    p.add_argument("--draws", default="1-2", help="--placebo: a-b, 1-based, inclusive")
    p.add_argument("--placebo-starts", type=int, default=2,
                   help="--placebo: NLLS starts per draw (zeros, then the linear warm start)")
    a = p.parse_args()
    if not a.prep and not a.numbers and a.est is None:
        p.error("--est is required for this mode")
    pd.options.mode.chained_assignment = None
    stamp(f"sleep_attention_stakes | out {out_dir()} | inputs {inputs_dir()} | BLAS threads {_THR} | "
          f"SLEEP_PIX_STATE={os.environ.get('SLEEP_PIX_STATE', '') or '(unset)'}")
    if a.prep:
        run_prep()
    elif a.fs:
        run_fs(a.est)
    elif a.check_base:
        run_check_base(a.est)
    elif a.fit:
        run_fit(a.est, [m.strip() for m in a.models.split(",") if m.strip()])
    elif a.twostage:
        if not a.model:
            p.error("--twostage needs --model")
        from utils.sleep_links import boot_cfg
        run_twostage(a.est, a.model, a.B if a.B is not None else boot_cfg()[0], a.workers, a.scheme)
    elif a.placebo:
        lo, hi = (int(x) for x in a.draws.split("-"))
        if not (1 <= lo <= hi <= N_DRAWS):
            p.error(f"--draws must lie in 1-{N_DRAWS}")
        run_placebo(a.est, lo, hi, a.placebo_starts)
    elif a.placebo_merge:
        run_placebo_merge(a.est)
    elif a.numbers:
        run_numbers()
    stamp(f"done; rss {rss_gb():.2f} GB")


if __name__ == "__main__":
    main()

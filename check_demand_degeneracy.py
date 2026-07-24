"""
check_demand_degeneracy.py
===========================
Local diagnostic for the BLP SingularException(3) / θ₂→0 issue.

Reproduces how blp_1_estimation.jl builds the design matrix X_hat = [spread_hat | x_mat]
from a DEMAND_PREP parquet, then:
  * reports which X_COLS / D_COLS / IV columns are missing, all-zero, or zero-variance
  * runs the SAME unpivoted QR that blp_2's IFT gradient uses and prints diag(R),
    so the column whose |R[i,i]|≈0 is exactly the one that triggers SingularException(i).

Usage:
  python check_demand_degeneracy.py [path/to/demand_*_final_spec_*.parquet]
Default targets E5 spec 12.
"""
import sys
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ── Column lists copied verbatim from blp_1_estimation.jl ────────────────────
X_COLS = ["fgc_covered", "has_ip", "seg_S2", "seg_S3", "seg_S4", "seg_S5",
          "log_total_assets_lag", "is_state_owned"]
D_COLS = ["gdp_per_capita", "fraction_65plus", "fraction_young",
          "pix_users_pf_per1000", "connections_per100", "frac_4g5g",
          "branches_per1000", "cadunico_families_per1000"]
IV_BLP_LOO = ["loo_log_assets", "mean_loo_log_assets",
              "loo_equity_ratio", "mean_loo_equity_ratio",
              "loo_basileia", "mean_loo_basileia",
              "loo_credit_assets", "mean_loo_credit_assets",
              "loo_npl_provision", "mean_loo_npl_provision", "n_rivals"]
IV_COST = ["personnel_cost_ratio_lag", "admin_cost_ratio_lag", "tax_cost_ratio_lag"]
IV_CAPITAL = ["indice_basileia_lag"]
IV_ESTBAN = ["estban_rival_branches_lag"]   # ESTBAN branch-competition IV (panel_10)
KEY_COLS = ["spread_ann", "deposit_type", "is_B", "mca_code", "time_id",
            "CodConglomeradoPrudencial"]

DEFAULT = ("../../BCB/Egan_et_al_2025_Rep/processed/ESTIMATION_OUTPUT/"
           "DEMAND_PREP/demand_5_logistic_final_spec_12.parquet")


def summarize(df, cols, label):
    print(f"\n{'='*78}\n  {label}\n{'='*78}")
    print(f"  {'column':<28}{'present':<9}{'n_nonnull':<11}{'nunique':<9}"
          f"{'std':<12}{'frac_zero':<10}flag")
    missing, degenerate = [], []
    n = len(df)
    for c in cols:
        if c not in df.columns:
            print(f"  {c:<28}{'NO':<9}{'-':<11}{'-':<9}{'-':<12}{'-':<10}*** MISSING ***")
            missing.append(c); degenerate.append(c)
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        nn = int(s.notna().sum())
        nu = int(s.nunique(dropna=True))
        sd = float(s.std(skipna=True)) if nn else float("nan")
        # how the Julia code sees it: coalesce(missing -> 0.0)
        filled = s.fillna(0.0).to_numpy(dtype=float)
        frac0 = float(np.mean(filled == 0.0))
        flag = ""
        if nu <= 1 or (np.nanstd(filled) == 0.0):
            flag = "*** CONSTANT/ALL-ZERO ***"; degenerate.append(c)
        elif np.allclose(filled, 0.0):
            flag = "*** ALL-ZERO ***"; degenerate.append(c)
        print(f"  {c:<28}{'yes':<9}{nn:<11}{nu:<9}{sd:<12.5g}{frac0:<10.3f}{flag}")
    return missing, degenerate


def qr_rank_report(df):
    print(f"\n{'='*78}\n  X_hat RANK / SINGULARITY (reproduces blp_2 unpivoted QR)\n{'='*78}")
    n = len(df)
    # Build x_mat exactly as build_regressor_matrices: zeros, fill if col present.
    x_mat = np.zeros((n, len(X_COLS)))
    for i, c in enumerate(X_COLS):
        if c in df.columns:
            x_mat[:, i] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy(float)
    # spread (proxy for spread_hat — the projection only rewrites col 1, so the
    # degeneracy among cols 2..K is identical to the real X_hat).
    if "spread_ann" in df.columns:
        spread = pd.to_numeric(df["spread_ann"], errors="coerce").fillna(0.0).to_numpy(float) / 100.0
    else:
        spread = np.zeros(n)
    X_hat = np.column_stack([spread, x_mat])
    labels = ["spread_hat(~spread)"] + X_COLS

    rank = int(np.linalg.matrix_rank(X_hat))
    print(f"  X_hat shape: {X_hat.shape}   numpy rank: {rank}   "
          f"({'FULL' if rank == X_hat.shape[1] else 'RANK-DEFICIENT'})")

    # Unpivoted QR — identical to Julia `qr(X_hat_v)`; SingularException(i) fires
    # at the first i with R[i,i] == 0 (1-based).
    _, R = np.linalg.qr(X_hat)
    diag = np.abs(np.diag(R))
    print(f"\n  {'col (1-based)':<15}{'variable':<24}{'|R[i,i]|':<14}note")
    tol = 1e-9 * (diag.max() if diag.size else 1.0)
    for i, (lab, d) in enumerate(zip(labels, diag), start=1):
        note = ""
        if d <= tol:
            note = "<-- SINGULAR (would throw SingularException here)"
        print(f"  {i:<15}{lab:<24}{d:<14.5g}{note}")
    return rank, X_hat.shape[1]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    print(f"Loading: {path}")
    df = pd.read_parquet(path)
    print(f"rows={len(df):,}  cols={df.shape[1]}")

    summarize(df, KEY_COLS, "KEY COLUMNS")
    _, xdeg = summarize(df, X_COLS, "X_COLS  (build x_mat -> X_hat cols 2..K)")
    summarize(df, D_COLS, "D_COLS  (demographics -> pi interactions / theta2)")
    summarize(df, IV_BLP_LOO + IV_ESTBAN + IV_COST + IV_CAPITAL, "INSTRUMENTS (Z)")

    rank, ncol = qr_rank_report(df)

    print(f"\n{'='*78}\n  VERDICT\n{'='*78}")
    if "has_ip" not in df.columns:
        print("  * has_ip is MISSING from the parquet -> X_hat column 3 is all-zero.")
    elif "has_ip" in xdeg:
        print("  * has_ip is present but CONSTANT/ALL-ZERO -> X_hat column 3 degenerate.")
    else:
        print("  * has_ip is present and varies.")
    if rank < ncol:
        print(f"  * X_hat is RANK-DEFICIENT ({rank} < {ncol}) -> blp_2 SingularException, "
              f"blp_1 silent rank-deficient solve.")
    else:
        print(f"  * X_hat is FULL RANK ({rank}={ncol}).")
    if xdeg:
        print(f"  * Degenerate X_COLS: {xdeg}")


if __name__ == "__main__":
    main()

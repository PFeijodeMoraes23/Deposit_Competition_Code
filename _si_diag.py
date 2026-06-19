"""Temporary diagnostic for the monotone single-index sleepiness curve.
Checks (1) estimation: index variation over time, binned link g_b, PAVA;
(2) graph construction: phi_mt spread, phi_t over time; (3) the band.
Also confirms what the linear (E2) phi_t actually does on the current panel.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np, pandas as pd
import estimation_timeseries_test as T

e1_df, pooled, s_tech, iv = T.build_frames()
s_time = s_tech + T.TIME_VARS

def run_block(state_cols, tag):
    print(f"\n================ {tag} ================")
    # E2 linear
    df2, r2, _ = T._run_linear(pooled, iv, state_cols, T.e2_first_stage, T.e2_second_stage, tag)
    phi2, _, _ = T.implied_phi_t(df2, r2, "linear")
    print(f"[E2 linear ] phi_t mean={phi2.mean():.3f} min={phi2.min():.3f} max={phi2.max():.3f}  (>1 cells share: ", end="")
    # cross-sectional phi_mt for E2
    pp = [p for p in r2.params.index if not p.startswith('v_hat')]
    X2 = T._build_phi_X(df2, pp); phi2mt = X2 @ r2.params[pp].values
    print(f"{(phi2mt>1).mean():.3f})")

    # E3 logit (direction)
    df3, r3, _ = T._run_logistic(pooled, iv, state_cols)
    phi3, _, _ = T.implied_phi_t(df3, r3, "logit")
    print(f"[E3 logit  ] phi_t mean={phi3.mean():.3f} min={phi3.min():.3f} max={phi3.max():.3f} std={phi3.std():.4f}")

    # ---- single index internals (replicate run_single_index with prints) ----
    phi_params = [p for p in r3.params.index if not str(p).startswith("v_hat")]
    theta = r3.params_native[phi_params].values.astype(float)
    print("[SI index  ] theta (logit native):")
    for nm, th in zip(phi_params, theta):
        print(f"             {nm:34} {th: .4f}")
    dff = pooled.copy()
    dff, _ = T.e3_first_stage(dff, iv, [c for c in state_cols if c != "constant"])
    df_ss = dff.dropna(subset=state_cols + ["deposit_balance","nr_lagged_dep","entity_id","v_hat_x_lagged_dep"]).copy()
    X = T._build_phi_X(df_ss, phi_params); v = X @ theta
    print(f"[SI index  ] v: cross-sectional std={v.std():.4f}  range=[{v.min():.3f},{v.max():.3f}]")
    # per-quarter mean index -> how much does the index move over time?
    vq = pd.Series(v, index=df_ss['year_quarter'].values).groupby(level=0).mean()
    print(f"[SI index  ] per-quarter MEAN index: range over time = {vq.max()-vq.min():.4f} (vs x-sec std {v.std():.4f})")

    n_bins = 20
    edges = np.quantile(v, np.linspace(0,1,n_bins+1)); edges[0]-=1e-9; edges[-1]+=1e-9
    binid = np.clip(np.digitize(v, edges[1:-1]), 0, n_bins-1)
    present = np.array(sorted(np.unique(binid))); B=len(present)
    remap={b:i for i,b in enumerate(present)}; binpos=np.array([remap[b] for b in binid])
    import statsmodels.api as sm
    Z=df_ss['nr_lagged_dep'].values.astype(float)
    D=np.zeros((len(df_ss),B)); D[np.arange(len(df_ss)),binpos]=Z
    cols=[f"_g{b}" for b in range(B)]
    work=pd.DataFrame(D,columns=cols,index=df_ss.index); work['entity_id']=df_ss['entity_id'].values
    work['_y']=df_ss['deposit_balance'].values; work['_cf']=df_ss['v_hat_x_lagged_dep'].values
    cols2=cols+['_cf']
    ydm=work['_y']-work.groupby('entity_id')['_y'].transform('mean')
    Xdm=work[cols2]-work.groupby('entity_id')[cols2].transform('mean')
    res=sm.OLS(ydm.values,Xdm.values).fit(cov_type='cluster',cov_kwds={'groups':df_ss['CodConglomeradoPrudencial'].astype(str)})
    g_raw=res.params[:B]; cov_g=res.cov_params()[:B,:B]
    counts=np.bincount(binpos,minlength=B).astype(float)
    centers=np.array([v[binpos==b].mean() for b in range(B)])
    g_mono=np.clip(T._pava_increasing(g_raw,w=counts),0,1)
    print(f"[SI link   ] g_raw   range=[{g_raw.min():.3f},{g_raw.max():.3f}]  monotone-violations={(np.diff(g_raw)<0).sum()}/{B-1}")
    print(f"[SI link   ] g_mono  range=[{g_mono.min():.3f},{g_mono.max():.3f}]  n_unique={len(np.unique(g_mono.round(4)))} (PAVA pooled {B-len(np.unique(g_mono.round(4)))} bins)")
    print(f"[SI link   ] per-bin SE sqrt(diag cov_g): median={np.median(np.sqrt(np.abs(np.diag(cov_g)))):.4f}")
    phimt=g_mono[binpos]
    print(f"[SI phi_mt ] cross-sectional: mean={phimt.mean():.3f} std={phimt.std():.4f} range=[{phimt.min():.3f},{phimt.max():.3f}]")
    # phi_t via run_single_index for the official series
    rsi, phisi, lo, hi = T.run_single_index(dff, state_cols, has_cf=True, logit_res=r3)
    print(f"[SI phi_t  ] mean={phisi.mean():.3f} min={phisi.min():.3f} max={phisi.max():.3f} std={phisi.std():.4f}")
    print(f"[SI band   ] halfwidth mean={((hi-lo)/2).mean():.4f}  (vs phi_t time-range {phisi.max()-phisi.min():.4f})")
    # how many DISTINCT bins does the national pop actually occupy, and does it move over time?
    fr=pd.DataFrame({'yq':df_ss['year_quarter'].values,'bin':binpos,'w':df_ss['pop_total'].fillna(0).values})
    binshare=fr.groupby(['yq','bin'])['w'].sum().groupby('yq').apply(lambda s:(s/s.sum()).to_dict())
    # dominant bin share over time
    dom=fr.groupby('yq').apply(lambda d: d.groupby('bin')['w'].sum().idxmax())
    print(f"[SI occ.   ] dominant pop bin over time: {sorted(dom.unique())} (if 1 value -> pop stuck in one bin)")

run_block(s_tech, "BASE (Tech)")
run_block(s_time, "+TIME (Tech+time)")
print("\nDONE")

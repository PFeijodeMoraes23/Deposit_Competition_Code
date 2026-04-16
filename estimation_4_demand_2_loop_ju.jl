"""
estimation_4_demand_2_loop_ju.jl
=================================
Julia port of estimation_4_demand_2_loop.py.

Identical to estimation_1_demand_2_loop_ju.jl except:
  - SPEC_NUM = 4  →  input: demand_4_final_spec_*.parquet
  - output:  blp_results_spec_4_*_<stage>.jls

See estimation_1_demand_2_loop_ju.jl for full commentary.
"""

using Parquet2, DataFrames, SparseArrays, LinearAlgebra, Statistics
using Random, Optim, QuasiMonteCarlo, Distributions
using JSON3, Serialization, ArgParse, Printf, Dates

const SPEC_NUM   = 4
const X_COLS     = ["fgc_covered", "has_ip", "seg_S2", "seg_S3", "seg_S4", "seg_S5",
                    "log_total_assets_lag", "equity_ratio_lag"]
const L_PROD     = length(X_COLS)
const K_TYPES    = 4
const K_LIST     = [1, 2, 4, 5]
const D_COLS     = ["gdp_per_capita", "fraction_65plus", "fraction_young",
                    "pix_users_pf_per1000", "connections_per100", "frac_4g5g",
                    "branches_per1000", "cadunico_families_per1000"]
const D_DIM      = length(D_COLS)
const IV_BLP_LOO = ["loo_log_assets", "mean_loo_log_assets",
                    "loo_equity_ratio", "mean_loo_equity_ratio",
                    "loo_basileia", "mean_loo_basileia",
                    "loo_credit_assets", "mean_loo_credit_assets",
                    "loo_npl_provision", "mean_loo_npl_provision", "n_rivals"]
const IV_COST    = ["personnel_cost_ratio_lag", "admin_cost_ratio_lag", "tax_cost_ratio_lag"]
const IV_CAPITAL = ["indice_basileia_lag"]

const _log_buf  = String[]
const _log_lock = ReentrantLock()

function get_paths(is_hpc::Bool; local_dir::Union{String,Nothing}=nothing)
    if is_hpc
        return "/home/pf382/dep_comp/data/input", "/home/pf382/dep_comp/data/output"
    else
        data_dir = local_dir !== nothing ? local_dir :
                   joinpath(dirname(dirname(dirname(abspath(@__FILE__)))), "BCB", "Egan_et_al_2025_Rep", "processed")
        return joinpath(data_dir, "ESTIMATION_OUTPUT", "DEMAND_PREP"),
               joinpath(data_dir, "ESTIMATION_OUTPUT", "BLP_RESULTS")
    end
end

function log_status(msg::String)
    stamped = "[$(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))] $msg"
    lock(_log_lock) do; push!(_log_buf, stamped); end
    println(stamped); flush(stdout)
end

function load_merged_spec_data(spec_id::Int; is_hpc::Bool=false, local_dir=nothing)
    input_dir, _ = get_paths(is_hpc; local_dir=local_dir)
    path = joinpath(input_dir, "demand_$(SPEC_NUM)_final_spec_$(spec_id).parquet")
    isfile(path) || error("Missing: $path")
    return DataFrame(Parquet2.Dataset(path); copycols=true)
end

function generate_halton_draws(R::Int, dim::Int, seed::Int)::Matrix{Float64}
    skip = seed * R
    n_total = R + skip
    pts_full = QuasiMonteCarlo.sample(n_total, zeros(dim), ones(dim), HaltonSample())  # (dim, n_total)
    pts = pts_full[:, (skip+1):end]  # (dim, R)
    pts = clamp.(pts, 1e-6, 1.0 - 1e-6)
    return quantile.(Normal(), pts)'
end

function generate_demographic_draws(df::DataFrame, R::Int, seed::Int)
    rng    = MersenneTwister(seed)
    d_cols = [c for c in D_COLS if c in names(df)]
    D      = length(d_cols)
    mca_time  = unique(df[:, ["mca_code", "time_id", d_cols...]])
    nat_std   = [std(skipmissing(mca_time[!, c])) for c in d_cols]
    nat_std   = [s == 0.0 ? 1.0 : s for s in nat_std]
    draws     = Dict{Tuple{String,String}, Matrix{Float64}}()
    for row in eachrow(mca_time)
        key    = (string(row.mca_code), string(row.time_id))
        mu     = [coalesce(row[c], 0.0) for c in d_cols]
        draws[key] = (mu .+ (nat_std .* 0.1) .* randn(rng, D, R)) |> transpose |> Matrix
    end
    return draws
end

function build_theta2_structure(stage::String)
    stage == "logit"  && return Int[], Tuple{Int,Int}[], 0
    stage == "sigma"  && return [1], Tuple{Int,Int}[], 1
    if stage == "full"
        sigma_idx = [1, 1 + findfirst(==("log_total_assets_lag"), X_COLS)]
        pi_inter  = [(1, findfirst(==("gdp_per_capita"),     D_COLS)),
                     (1, findfirst(==("fraction_65plus"),    D_COLS)),
                     (1, findfirst(==("connections_per100"), D_COLS))]
        return sigma_idx, pi_inter, length(sigma_idx) + length(pi_inter)
    end
    if stage == "extended"
        sigma_idx = [1, 1 + findfirst(==("log_total_assets_lag"), X_COLS)]
        pi_inter  = [(1, findfirst(==("gdp_per_capita"), D_COLS)),
                     (1, findfirst(==("fraction_65plus"), D_COLS)),
                     (1, findfirst(==("connections_per100"), D_COLS)),
                     (1 + findfirst(==("log_total_assets_lag"), X_COLS), findfirst(==("gdp_per_capita"), D_COLS)),
                     (1 + findfirst(==("fgc_covered"), X_COLS),          findfirst(==("fraction_65plus"), D_COLS)),
                     (1 + findfirst(==("equity_ratio_lag"), X_COLS),     findfirst(==("cadunico_families_per1000"), D_COLS))]
        return sigma_idx, pi_inter, length(sigma_idx) + length(pi_inter)
    end
    error("Unknown stage: $stage")
end

function unpack_theta2(theta2, sigma_indices, pi_interactions)
    n_s = length(sigma_indices); return theta2[1:n_s], theta2[n_s+1:end]
end

function compute_mu(prod_vec, nu_draws, stacked_draws, obs_key_idx,
                    sigma_vals, sigma_indices, pi_vals, pi_interactions,
                    R::Int, coef_dim::Int)::Matrix{Float64}
    sigma_diag = zeros(coef_dim)
    for (p, c) in enumerate(sigma_indices); c <= coef_dim && (sigma_diag[c] = sigma_vals[p]); end
    mu = prod_vec * (nu_draws[:, 1:coef_dim] .* sigma_diag')'
    D_dim = size(stacked_draws, 3)
    for (pi_idx, (cidx, didx)) in enumerate(pi_interactions)
        (cidx <= coef_dim && didx <= D_dim) || continue
        mu .+= pi_vals[pi_idx] .* prod_vec[:, cidx] .* stacked_draws[obs_key_idx, :, didx]
    end
    return mu
end

struct Precomp
    b_mkt_idx::Vector{Int}; d_time_enc::Vector{Int}
    unique_pairs::Vector{Tuple{String,String}}; unique_times::Vector{String}
    pair_time_enc::Vector{Int}; pop_weights::Vector{Float64}
    B_agg::SparseMatrixCSC{Float64,Int}; D_agg::SparseMatrixCSC{Float64,Int}; PT_agg::SparseMatrixCSC{Float64,Int}
    sort_b::Vector{Int}; b_grp_start::Vector{Int}
    sort_d::Vector{Int}; d_uval::Vector{Int}; d_grp_start::Vector{Int}
    sort_pt::Vector{Int}; pt_uval::Vector{Int}; pt_grp_start::Vector{Int}
    b_mask::BitVector; d_mask::BitVector
    ln_s_data_D::Vector{Float64}; ln_s_data_B_cond::Vector{Float64}
    Z_moments::Matrix{Float64}; X_full::Matrix{Float64}; X_hat::Matrix{Float64}
    theta1_valid::BitVector; clusters::Vector{String}
end

function _unique_with_starts(sv::Vector{Int})
    isempty(sv) && return Int[], Int[]
    uval=Int[sv[1]]; gs=Int[1]
    for i in 2:length(sv); sv[i]!=sv[i-1] && (push!(uval,sv[i]);push!(gs,i)); end
    return uval, gs
end

function build_precomp(df, Z, Xf, Xh, valid, cl)
    bm=BitVector(Bool.(coalesce.(df.is_B,false))); dm=.!bm; NB=sum(bm); ND=sum(dm)
    mca=string.(df.mca_code); time=string.(df.time_id); pt=coalesce.(df.pop_total,0.0)
    bp=collect(zip(mca[bm],time[bm])); up=sort(unique(bp)); p2i=Dict(p=>i for (i,p) in enumerate(up))
    bmi=[p2i[(m,t)] for (m,t) in bp]
    ut=sort(unique(time)); t2i=Dict(t=>i for (i,t) in enumerate(ut)); dte=[t2i[t] for t in time[dm]]
    np=length(up); nt=length(ut); pte=[t2i[p[2]] for p in up]
    pp=zeros(np); for (i,w) in zip(bmi,pt[bm]); pp[i]+=w; end
    tp=zeros(nt); for (i,w) in zip(pte,pp); tp[i]+=w; end; pw=pp./max.(tp[pte],1e-30)
    Ba=sparse(bmi,1:NB,ones(NB),np,NB); Da=sparse(dte,1:ND,ones(ND),nt,ND); Pa=sparse(pte,1:np,pw,nt,np)
    sb=sortperm(bmi); _,bg=_unique_with_starts(bmi[sb])
    sd=sortperm(dte); du,dg=_unique_with_starts(dte[sd])
    sp=sortperm(pte); ptu,ptg=_unique_with_starts(pte[sp])
    ln_sD=log.(clamp.(Float64.(coalesce.(df.share_D,0.0)),1e-15,Inf)); ln_sB=log.(clamp.(Float64.(coalesce.(df.share_B_cond,0.0)),1e-15,Inf))
    return Precomp(bmi,dte,up,ut,pte,pw,Ba,Da,Pa,sb,bg,sd,du,dg,sp,ptu,ptg,bm,dm,ln_sD,ln_sB,Z,Xf,Xh,valid,cl)
end

function logsumexp_groups(V, sort_idx, grp_start, n_groups, uval, R; weights=nothing)
    n_uniq=length(grp_start); res=fill(-Inf,n_groups,R)
    for g in 1:n_uniq
        gi=grp_start[g]; ge=g<n_uniq ? grp_start[g+1]-1 : length(sort_idx)
        rows=sort_idx[gi:ge]; Vg=V[rows,:]
        mx=maximum(Vg,dims=1)
        lse=if weights!==nothing; mx.+log.(max.(sum(weights[rows].*exp.(Vg.-mx),dims=1),1e-300))
        else; mx.+log.(max.(sum(exp.(Vg.-mx),dims=1),1e-300)); end
        res[uval[g],:].=vec(lse)
    end; return res
end

function compute_model_shares(delta, mu, pc::Precomp, R::Int)
    np2=length(pc.unique_pairs); nt=length(pc.unique_times)
    VB=clamp.(delta[pc.b_mask].+mu[pc.b_mask,:], -500.0,500.0)
    VD=clamp.(delta[pc.d_mask].+mu[pc.d_mask,:], -500.0,500.0)
    lsD=logsumexp_groups(VD,pc.sort_d,pc.d_grp_start,nt,pc.d_uval,R)
    lsB=logsumexp_groups(VB,pc.sort_b,pc.b_grp_start,np2,1:np2,R)
    ldp=lsD[pc.pair_time_enc,:]
    jm=max.(max.(lsB,ldp),log(1.0)); logd=jm.+log.(max.(exp.(log(1.0).-jm).+exp.(lsB.-jm).+exp.(ldp.-jm),1e-300))
    sB=vec(mean(exp.(VB.-logd[pc.b_mkt_idx,:]),dims=2))
    nld=-logd; mxn=fill(-Inf,nt,R)
    for g in 1:length(pc.pt_grp_start)
        gi=pc.pt_grp_start[g]; ge=g<length(pc.pt_grp_start) ? pc.pt_grp_start[g+1]-1 : length(pc.sort_pt)
        rows=pc.sort_pt[gi:ge]; Vg=nld[rows,:]; mx=maximum(Vg,dims=1); mxn[pc.pt_uval[g],:].=vec(mx)
    end
    sw=pc.PT_agg*exp.(nld.-mxn[pc.pair_time_enc,:]); li=mxn.+log.(max.(sw,1e-300))
    lsD_r=VD.+li[pc.d_time_enc,:]; rm=maximum(lsD_r,dims=2)
    sD=vec(mean(exp.(lsD_r.-rm),dims=2)).*vec(exp.(rm))
    return sB, sD
end

function blp_contraction(di, mu, pc::Precomp, R; tol=1e-12, max_iter=1500)
    N=length(pc.b_mask)
    delta=if di!==nothing&&length(di)==N; copy(di)
    else; d=zeros(N); d[pc.d_mask].=pc.ln_s_data_D[pc.d_mask]; d[pc.b_mask].=pc.ln_s_data_B_cond[pc.b_mask]; d; end
    m=5; Fh=zeros(N,m); Xh=zeros(N,m); ptr=1; hl=0; nh=Float64[]
    for h in 1:max_iter
        sB,sD=compute_model_shares(delta,mu,pc,R); dn=copy(delta)
        dn[pc.d_mask].=delta[pc.d_mask].+pc.ln_s_data_D[pc.d_mask].-log.(clamp.(sD,1e-15,Inf))
        dn[pc.b_mask].=delta[pc.b_mask].+pc.ln_s_data_B_cond[pc.b_mask].-log.(clamp.(sB,1e-15,Inf))
        clamp!(dn,-500.0,500.0); !all(isfinite,dn) && return delta,false,h,nh
        f=dn.-delta; nm=maximum(abs.(f)); push!(nh,nm)
        (h%50==0||nm<tol) && println("    [iter=$h] norm=$(round(nm,sigdigits=4))")
        nm<tol && return dn,true,h,nh
        Fh[:,ptr]=f; Xh[:,ptr]=delta; ptr=mod1(ptr+1,m); hl=min(hl+1,m)
        if hl>1; Fk=Fh[:,1:hl]; dF=Fk[:,2:end].-Fk[:,1:1]
            try; cb=(dF'*dF+1e-10*I(hl-1))\(-(dF'*Fk[:,1])); c=vcat(1.0-sum(cb),cb); da=(Xh[:,1:hl].+Fh[:,1:hl])*c; delta=all(isfinite,da) ? clamp.(da,-500.0,500.0) : dn; catch; delta=dn; end
        else; delta=dn; end
    end; return delta,false,max_iter,nh
end

function blp_contraction_draws(di,R,pv,nu,sd,oki,sv,si,pv2,pi2,cd,pc;tol=1e-12,max_iter=1500,chunk_size=nothing)
    if chunk_size===nothing||chunk_size>=R; mu=compute_mu(pv,nu,sd,oki,sv,si,pv2,pi2,R,cd); return blp_contraction(di,mu,pc,R;tol=tol,max_iter=max_iter); end
    N=length(pc.b_mask); NB=sum(pc.b_mask); ND=sum(pc.d_mask)
    delta=if di!==nothing&&length(di)==N; copy(di) else; d=zeros(N); d[pc.d_mask].=pc.ln_s_data_D[pc.d_mask]; d[pc.b_mask].=pc.ln_s_data_B_cond[pc.b_mask]; d; end
    m=5; Fh=zeros(N,m); Xh=zeros(N,m); ptr=1; hl=0; nh=Float64[]
    for h in 1:max_iter
        sB_a=zeros(NB); sD_a=zeros(ND); r0=1
        while r0<=R; r1=min(r0+chunk_size-1,R); rc=r1-r0+1
            mc=compute_mu(pv,nu[r0:r1,:],sd[:,r0:r1,:],oki,sv,si,pv2,pi2,rc,cd)
            sB,sD=compute_model_shares(delta,mc,pc,rc); sB_a.+=sB.*rc; sD_a.+=sD.*rc; r0=r1+1; end
        dn=copy(delta); dn[pc.d_mask].=delta[pc.d_mask].+pc.ln_s_data_D[pc.d_mask].-log.(clamp.(sD_a./R,1e-15,Inf))
        dn[pc.b_mask].=delta[pc.b_mask].+pc.ln_s_data_B_cond[pc.b_mask].-log.(clamp.(sB_a./R,1e-15,Inf))
        clamp!(dn,-500.0,500.0); !all(isfinite,dn) && return delta,false,h,nh
        f=dn.-delta; nm=maximum(abs.(f)); push!(nh,nm)
        (h%50==0||nm<tol) && println("    [iter=$h] norm=$(round(nm,sigdigits=4))")
        nm<tol && return dn,true,h,nh
        Fh[:,ptr]=f; Xh[:,ptr]=delta; ptr=mod1(ptr+1,m); hl=min(hl+1,m)
        if hl>1; Fk=Fh[:,1:hl]; dF=Fk[:,2:end].-Fk[:,1:1]
            try; cb=(dF'*dF+1e-10*I(hl-1))\(-(dF'*Fk[:,1])); c=vcat(1.0-sum(cb),cb); da=(Xh[:,1:hl].+Fh[:,1:hl])*c; delta=all(isfinite,da) ? clamp.(da,-500.0,500.0) : dn; catch; delta=dn; end
        else; delta=dn; end
    end; return delta,false,max_iter,nh
end

function build_regressor_matrices(df)
    N=nrow(df); sc=coalesce.(df.spread_qoq,0.0)
    xm=zeros(N,L_PROD); for (i,c) in enumerate(X_COLS); c in names(df) && (xm[:,i].=coalesce.(df[!,c],0.0)); end
    ivc=[c for c in vcat(IV_BLP_LOO,IV_COST,IV_CAPITAL) if c in names(df)]
    Zm=zeros(N,length(ivc)); for (i,c) in enumerate(ivc); v=coalesce.(df[!,c],0.0);replace!(v,Inf=>0.0,-Inf=>0.0);Zm[:,i].=v; end
    return sc,xm,Zm,ivc
end

function project_endogenous_spreads(sc, H, dt)
    sh=reshape(copy(sc),:,1)
    for k in [4,5]; mask=dt.==k; any(mask)||continue; Hk=H[mask,:]; sk=sc[mask]; valid=all.(isfinite,eachrow(Hk)).&isfinite.(sk); sum(valid)>size(Hk,2)||continue
        try; b=Hk[valid,:]\sk[valid]; kh=copy(sk); kh[valid].=Hk[valid,:]*b; sh[mask,1].=kh; catch; end; end
    return sh
end

function estimate_theta1(delta, pc::Precomp)
    valid=pc.theta1_valid.&isfinite.(delta); theta1=try pc.X_hat[valid,:]\delta[valid] catch; zeros(size(pc.X_hat,2)) end
    return theta1, delta.-pc.X_full*theta1
end

compute_gmm_moments(xi, pc::Precomp) = vec(mean(xi .* pc.Z_moments, dims=1))

function gmm_objective(theta2, df, pv, nu, sd, oki, si, pi, R, cd, W, ti, mi, dc::Ref, pc; chunk_size=nothing)::Float64
    sv,piv=unpack_theta2(theta2,si,pi); delta,_,_,_=blp_contraction_draws(dc[],R,pv,nu,sd,oki,sv,si,piv,pi,cd,pc;tol=ti,max_iter=mi,chunk_size=chunk_size)
    dc[]=copy(delta); _,xi=estimate_theta1(delta,pc); G=compute_gmm_moments(xi,pc); return dot(G,W*G)
end

function run_blp_for_spec(spec_id::Int, args)
    println("\n"*"="^60); println("  Specification $spec_id"); println("="^60)
    df=try load_merged_spec_data(spec_id;is_hpc=args["hpc"],local_dir=get(args,"local_dir",nothing)) catch e; println("  [!] $e"); return nothing; end
    nrow(df)==0 && (println("  [!] Empty."); return nothing); println("  Loaded $(nrow(df)) observations")
    si,pi_i,np=build_theta2_structure(args["stage"]); R=args["R"]; seed=args["seed"]; cd=1+L_PROD
    nu=generate_halton_draws(R,cd,seed); res=Dict{String,Any}("spec_id"=>spec_id,"stage"=>args["stage"])

    if args["stage"]=="logit"
        isB=Bool.(coalesce.(df.is_B,false)); sDc=Float64.(coalesce.(df.share_D,0.0)); sBc=Float64.(coalesce.(df.share_B_cond,0.0)); delta=zeros(nrow(df)); delta[.!isB].=log.(clamp.(sDc[.!isB],1e-15,Inf)); delta[isB].=log.(clamp.(sBc[isB],1e-15,Inf))
        sc,xm,Zm,ivc=build_regressor_matrices(df); H=hcat(xm,Zm); dt=Int.(coalesce.(df.deposit_type,0)); sh=project_endogenous_spreads(sc,H,dt)
        Xf=hcat(sc,xm); Xh=hcat(sh,xm); valid=BitVector(all.(isfinite,eachrow(Xh))); cl=string.(df.CodConglomeradoPrudencial).*"_".*first.(split.(string.(df.time_id),"Q"))
        iv_ok=[c for c in ivc if std(replace(coalesce.(df[!,c],0.0),Inf=>0.0,-Inf=>0.0))>1e-10]
        Zc=zeros(nrow(df),length(iv_ok)); for (i,c) in enumerate(iv_ok); v=coalesce.(df[!,c],0.0);replace!(v,Inf=>0.0,-Inf=>0.0);Zc[:,i].=v; end
        pc=build_precomp(df,Zc,Xf,Xh,valid,cl); t1,xi=estimate_theta1(delta,pc); G=compute_gmm_moments(xi,pc); W=Matrix(I(length(iv_ok))*1.0); Q=dot(G,W*G)
        merge!(res,Dict("theta1"=>t1,"theta2"=>Float64[],"delta"=>delta,"xi"=>xi,"Q_value"=>Q,"converged"=>true,"param_names_theta1"=>vcat(["alpha"],X_COLS)))
        println("  alpha=$(round(t1[1],sigdigits=6))  Q=$(round(Q,sigdigits=6))"); return res
    end

    println("  Stage: $(uppercase(args["stage"])) ($np params)")
    dd=generate_demographic_draws(df,R,seed); d_cols=[c for c in D_COLS if c in names(df)]; Dd=length(d_cols); N=nrow(df)
    mca=string.(df.mca_code); time=string.(df.time_id); uk=sort(collect(keys(dd))); k2i=Dict(k=>i for (i,k) in enumerate(uk))
    nk=length(uk); sd=zeros(Float64,nk,R,Dd)
    for (i,k) in enumerate(uk); sd[i,:,:].=dd[k]; end
    sdp=vcat(sd,zeros(1,R,Dd)); padi=nk+1
    oki=[get(k2i,(mca[i],time[i]),padi) for i in 1:N]
    pv_mat=zeros(N,cd); sp=coalesce.(df.spread_qoq,0.0); dt=Int.(coalesce.(df.deposit_type,0)); pv_mat[:,1].=sp
    for (i,col) in enumerate(X_COLS); col in names(df) && (pv_mat[:,1+i].=coalesce.(df[!,col],0.0)); end
    sc,xm,Zm,ivc=build_regressor_matrices(df)
    iv_ok=[c for c in ivc if std(replace(coalesce.(df[!,c],0.0),Inf=>0.0,-Inf=>0.0))>1e-10]
    Z=zeros(N,length(iv_ok)); for (i,c) in enumerate(iv_ok); v=coalesce.(df[!,c],0.0);replace!(v,Inf=>0.0,-Inf=>0.0);Z[:,i].=v; end
    W=try inv(Z'*Z./N) catch; Matrix(I(length(iv_ok))*1.0) end
    H=hcat(xm,Zm); sh=project_endogenous_spreads(sc,H,dt); Xf=hcat(sc,xm); Xh=hcat(sh,xm)
    valid=BitVector(all.(isfinite,eachrow(Xh))); cl=string.(df.CodConglomeradoPrudencial).*"_".*first.(split.(string.(df.time_id),"Q"))
    pc=build_precomp(df,Z,Xf,Xh,valid,cl)
    cs=let c=get(args,"chunk_size",0); (c===nothing||c==0) ? nothing : c; end
    _,od=get_paths(args["hpc"]); prev=Dict("full"=>"sigma","extended"=>"full"); t2_0=nothing
    if args["stage"] in keys(prev)
        pp=joinpath(od,"blp_checkpoint_spec_$(spec_id)_$(prev[args["stage"]]).jld2_compat.jls")
        if isfile(pp); try t2p=deserialize(pp)["theta2_star"]; t2_0=zeros(np); t2_0[1:length(t2p)].=t2p; catch e; end; end
    end
    t2_0===nothing && (t2_0=randn(MersenneTwister(seed),np).*0.01)
    get(args,"dry_run",false) && begin
        sv,piv=unpack_theta2(t2_0,si,pi_i); t0=time(); blp_contraction_draws(nothing,R,pv_mat,nu,sdp,oki,sv,si,piv,pi_i,cd,pc;tol=args["tol_inner"],max_iter=10,chunk_size=cs)
        println("  [DRY RUN] $(round(time()-t0,digits=2))s/10. Exiting."); return Dict("dry_run"=>true); end
    oc=Ref(0); dc=Ref{Union{Vector{Float64},Nothing}}(nothing)
    obj=t2->(val=gmm_objective(t2,df,pv_mat,nu,sdp,oki,si,pi_i,R,cd,W,args["tol_inner"],args["max_inner"],dc,pc;chunk_size=cs);oc[]+=1;oc[]%10==0&&log_status("  [OUTER=$(oc[])] theta2=$(round.(t2,digits=4))");val)
    r=optimize(obj,fill(-15.0,np),fill(15.0,np),t2_0,Fminbox(LBFGS()),Optim.Options(iterations=500,f_tol=args["tol_outer"],show_trace=false))
    t2s=Optim.minimizer(r); println("  Converged: $(Optim.converged(r))  Q=$(round(Optim.minimum(r),sigdigits=6))")
    sv,piv=unpack_theta2(t2s,si,pi_i)
    ds,_,_,_=blp_contraction_draws(dc[],R,pv_mat,nu,sdp,oki,sv,si,piv,pi_i,cd,pc;tol=args["tol_inner"],max_iter=args["max_inner"],chunk_size=cs)
    t1s,xis=estimate_theta1(ds,pc)
    merge!(res,Dict("theta1"=>t1s,"theta2"=>t2s,"delta"=>ds,"xi"=>xis,"Q_value"=>Optim.minimum(r),"converged"=>Optim.converged(r),"param_names_theta1"=>vcat(["alpha"],X_COLS)))
    println("  alpha=$(round(t1s[1],sigdigits=6))")
    mkpath(od); chk=joinpath(od,"blp_checkpoint_spec_$(spec_id)_$(args["stage"]).jld2_compat.jls")
    try serialize(chk,Dict("theta2_star"=>t2s,"delta_star"=>ds)); println("  [CHECKPOINT] $(basename(chk))") catch e; end
    return res
end

function parse_args_blp()
    s=ArgParseSettings(description="BLP Demand Estimation Loop – Spec 4")
    @add_arg_table! s begin
        "--spec";arg_type=String;default="12"; "--stage";arg_type=String;default="logit"
        "--R";arg_type=Int;default=100; "--seed";arg_type=Int;default=42
        "--tol-inner";arg_type=Float64;default=1e-12;dest_name="tol_inner"
        "--max-inner";arg_type=Int;default=2000;dest_name="max_inner"
        "--tol-outer";arg_type=Float64;default=1e-6;dest_name="tol_outer"
        "--method";arg_type=String;default="l-bfgs-b"; "--workers";arg_type=Int;default=Threads.nthreads()
        "--hpc";action=:store_true; "--local-dir";arg_type=String;default=nothing;dest_name="local_dir"
        "--dry-run";action=:store_true;dest_name="dry_run"; "--chunk-size";arg_type=Int;default=0;dest_name="chunk_size"
    end; return parse_args(s)
end

function main()
    args=parse_args_blp(); spec_ids=args["spec"]=="all" ? collect(1:12) : [parse(Int,args["spec"])]
    log_status("BLP Demand Estimation – Spec$(SPEC_NUM) – START")
    log_status("  Stage: $(args["stage"]) | Specs: $spec_ids | R=$(args["R"])")
    _,od=get_paths(args["hpc"];local_dir=get(args,"local_dir",nothing)); mkpath(od)
    stages=args["stage"]=="sequence" ? ["logit","sigma","full","extended"] : [args["stage"]]
    for cs in stages
        args["stage"]=cs; all_res=Dict{Int,Any}(); lk=ReentrantLock()
        Threads.@threads for sp in spec_ids
            res=try run_blp_for_spec(sp,args) catch e; println("  [!] Spec $sp: $e"); nothing; end
            res===nothing&&continue; get(res,"dry_run",false)&&continue
            pkl=joinpath(od,"blp_results_spec_4_$(sp)_$(cs).jls"); serialize(pkl,res); println("  Saved: $(basename(pkl))")
            lock(lk) do; all_res[sp]=Dict("Q_value"=>get(res,"Q_value",0.0),"converged"=>get(res,"converged",true),"theta1_alpha"=>isempty(get(res,"theta1",Float64[])) ? [] : [res["theta1"][1]],"theta2"=>get(res,"theta2",Float64[]),"stage"=>cs); end
        end
        sp_path=joinpath(od,"blp_summary_$(SPEC_NUM)_$(cs).json"); open(sp_path,"w") do f; JSON3.write(f,all_res); end
        log_status("[DONE] Stage=$cs  Summary: $(basename(sp_path))")
    end
end

main()

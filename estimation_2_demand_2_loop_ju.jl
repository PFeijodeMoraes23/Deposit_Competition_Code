"""
estimation_2_demand_2_loop_ju.jl
=================================
Julia port of estimation_2_demand_2_loop.py.

Identical to estimation_1_demand_2_loop_ju.jl except:
  - SPEC_NUM = 2  →  input: demand_2_final_spec_*.parquet
  - output:  blp_results_spec_2_*_<stage>.jls

See estimation_1_demand_2_loop_ju.jl for full commentary.
"""

using Parquet2, DataFrames, SparseArrays, LinearAlgebra, Statistics
using Random, Optim, QuasiMonteCarlo, Distributions
using JSON3, Serialization, ArgParse, Printf, Dates

const SPEC_NUM   = 2
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
        key      = (string(row.mca_code), string(row.time_id))
        mu       = [coalesce(row[c], 0.0) for c in d_cols]
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
    n_s = length(sigma_indices)
    return theta2[1:n_s], theta2[n_s+1:end]
end

function compute_mu(prod_vec, nu_draws, stacked_draws, obs_key_idx,
                    sigma_vals, sigma_indices, pi_vals, pi_interactions,
                    R::Int, coef_dim::Int)::Matrix{Float64}
    sigma_diag = zeros(coef_dim)
    for (p, c) in enumerate(sigma_indices); c <= coef_dim && (sigma_diag[c] = sigma_vals[p]); end
    sigma_nu = nu_draws[:, 1:coef_dim] .* sigma_diag'
    mu       = prod_vec * sigma_nu'
    D_dim = size(stacked_draws, 3)
    for (pi_idx, (cidx, didx)) in enumerate(pi_interactions)
        (cidx <= coef_dim && didx <= D_dim) || continue
        mu .+= pi_vals[pi_idx] .* prod_vec[:, cidx] .* stacked_draws[obs_key_idx, :, didx]
    end
    return mu
end

struct Precomp
    b_mkt_idx    ::Vector{Int}; d_time_enc::Vector{Int}
    unique_pairs ::Vector{Tuple{String,String}}; unique_times::Vector{String}
    pair_time_enc::Vector{Int}; pop_weights::Vector{Float64}
    B_agg::SparseMatrixCSC{Float64,Int}; D_agg::SparseMatrixCSC{Float64,Int}
    PT_agg::SparseMatrixCSC{Float64,Int}
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
    uval = Int[sv[1]]; gstart = Int[1]
    for i in 2:length(sv)
        sv[i] != sv[i-1] && (push!(uval, sv[i]); push!(gstart, i))
    end
    return uval, gstart
end

function build_precomp(df::DataFrame, Z, X_full, X_hat, valid, clusters)
    N = nrow(df)
    b_mask = BitVector(Bool.(coalesce.(df.is_B, false))); d_mask = .!b_mask
    N_B = sum(b_mask); N_D = sum(d_mask)
    mca = string.(df.mca_code); time = string.(df.time_id)
    ptot = coalesce.(df.pop_total, 0.0)

    b_pairs     = collect(zip(mca[b_mask], time[b_mask]))
    unique_pairs = sort(unique(b_pairs))
    p2i          = Dict(p => i for (i,p) in enumerate(unique_pairs))
    b_mkt_idx    = [p2i[(m,t)] for (m,t) in b_pairs]

    unique_times = sort(unique(time))
    t2i          = Dict(t => i for (i,t) in enumerate(unique_times))
    d_time_enc   = [t2i[t] for t in time[d_mask]]

    n_pairs  = length(unique_pairs); n_times = length(unique_times)
    pte      = [t2i[p[2]] for p in unique_pairs]
    pp       = zeros(n_pairs)
    for (i,w) in zip(b_mkt_idx, ptot[b_mask]); pp[i] += w; end
    tp = zeros(n_times); for (i,w) in zip(pte, pp); tp[i] += w; end
    pw = pp ./ max.(tp[pte], 1e-30)

    B_agg  = sparse(b_mkt_idx,  1:N_B, ones(N_B),  n_pairs, N_B)
    D_agg  = sparse(d_time_enc, 1:N_D, ones(N_D),  n_times, N_D)
    PT_agg = sparse(pte,        1:n_pairs, pw,      n_times, n_pairs)

    sort_b        = sortperm(b_mkt_idx); _, bg     = _unique_with_starts(b_mkt_idx[sort_b])
    sort_d        = sortperm(d_time_enc); du, dg    = _unique_with_starts(d_time_enc[sort_d])
    sort_pt       = sortperm(pte);        ptu, ptg  = _unique_with_starts(pte[sort_pt])

    ln_sD = log.(clamp.(Float64.(coalesce.(df.share_D,      0.0)), 1e-15, Inf))
    ln_sB = log.(clamp.(Float64.(coalesce.(df.share_B_cond, 0.0)), 1e-15, Inf))
    return Precomp(b_mkt_idx, d_time_enc, unique_pairs, unique_times, pte, pw,
                   B_agg, D_agg, PT_agg, sort_b, bg, sort_d, du, dg, sort_pt, ptu, ptg,
                   b_mask, d_mask, ln_sD, ln_sB, Z, X_full, X_hat, valid, clusters)
end

function logsumexp_groups(V, sort_idx, grp_start, n_groups, uval, R; weights=nothing)
    n_uniq = length(grp_start); result = fill(-Inf, n_groups, R)
    for g in 1:n_uniq
        gi   = grp_start[g]; ge = g < n_uniq ? grp_start[g+1]-1 : length(sort_idx)
        rows = sort_idx[gi:ge]; V_g = V[rows, :]
        mx   = maximum(V_g, dims=1)
        lse  = if weights !== nothing
            mx .+ log.(max.(sum(weights[rows] .* exp.(V_g .- mx), dims=1), 1e-300))
        else
            mx .+ log.(max.(sum(exp.(V_g .- mx), dims=1), 1e-300))
        end
        result[uval[g], :] .= vec(lse)
    end
    return result
end

function compute_model_shares(delta, mu, pc::Precomp, R::Int)
    N_B = sum(pc.b_mask); N_D = sum(pc.d_mask)
    n_pairs = length(pc.unique_pairs); n_times = length(pc.unique_times)
    V_B = clamp.(delta[pc.b_mask] .+ mu[pc.b_mask, :], -500.0, 500.0)
    V_D = clamp.(delta[pc.d_mask] .+ mu[pc.d_mask, :], -500.0, 500.0)

    lsD = logsumexp_groups(V_D, pc.sort_d, pc.d_grp_start, n_times, pc.d_uval, R)
    lsB = logsumexp_groups(V_B, pc.sort_b, pc.b_grp_start, n_pairs, 1:n_pairs, R)
    ldp = lsD[pc.pair_time_enc, :]

    jm    = max.(max.(lsB, ldp), log(1.0))
    logd  = jm .+ log.(max.(exp.(log(1.0) .- jm) .+ exp.(lsB .- jm) .+ exp.(ldp .- jm), 1e-300))
    logdB = logd[pc.b_mkt_idx, :]
    s_B   = vec(mean(exp.(V_B .- logdB), dims=2))

    neg_ld  = -logd
    mx_nld  = fill(-Inf, n_times, R)
    for g in 1:length(pc.pt_grp_start)
        gi = pc.pt_grp_start[g]; ge = g < length(pc.pt_grp_start) ? pc.pt_grp_start[g+1]-1 : length(pc.sort_pt)
        rows = pc.sort_pt[gi:ge]; V_g = neg_ld[rows, :]
        mx   = maximum(V_g, dims=1); mx_nld[pc.pt_uval[g], :] .= vec(mx)
    end
    sh_inv = exp.(neg_ld .- mx_nld[pc.pair_time_enc, :])
    sw     = pc.PT_agg * sh_inv
    li_wtd = mx_nld .+ log.(max.(sw, 1e-300))
    lsD_r  = V_D .+ li_wtd[pc.d_time_enc, :]
    rm     = maximum(lsD_r, dims=2)
    s_D    = vec(mean(exp.(lsD_r .- rm), dims=2)) .* vec(exp.(rm))
    return s_B, s_D
end

function blp_contraction(delta_init, mu, pc::Precomp, R::Int; tol=1e-12, max_iter=1500)
    N = length(pc.b_mask)
    delta = if delta_init !== nothing && length(delta_init) == N
        copy(delta_init)
    else
        d = zeros(N); d[pc.d_mask] .= pc.ln_s_data_D[pc.d_mask]
        d[pc.b_mask] .= pc.ln_s_data_B_cond[pc.b_mask]; d
    end
    m_aa=5; Fh=zeros(N,m_aa); Xh=zeros(N,m_aa); ptr=1; hl=0; nh=Float64[]
    for h in 1:max_iter
        sB, sD = compute_model_shares(delta, mu, pc, R)
        dn = copy(delta)
        dn[pc.d_mask] .= delta[pc.d_mask] .+ pc.ln_s_data_D[pc.d_mask] .- log.(clamp.(sD,1e-15,Inf))
        dn[pc.b_mask] .= delta[pc.b_mask] .+ pc.ln_s_data_B_cond[pc.b_mask] .- log.(clamp.(sB,1e-15,Inf))
        clamp!(dn,-500.0,500.0)
        !all(isfinite, dn) && return delta,false,h,nh
        f=dn.-delta; norm=maximum(abs.(f)); push!(nh,norm)
        (h%50==0 || norm<tol) && println("    [iter=$h] norm=$(round(norm,sigdigits=4))")
        norm<tol && return dn,true,h,nh
        Fh[:,ptr]=f; Xh[:,ptr]=delta; ptr=mod1(ptr+1,m_aa); hl=min(hl+1,m_aa)
        if hl>1
            Fk=Fh[:,1:hl]; dF=Fk[:,2:end].-Fk[:,1:1]
            try
                cb=(dF'*dF+1e-10*I(hl-1))\(-(dF'*Fk[:,1]))
                c=vcat(1.0-sum(cb),cb); da=(Xh[:,1:hl].+Fh[:,1:hl])*c
                delta=all(isfinite,da) ? clamp.(da,-500.0,500.0) : dn
            catch; delta=dn; end
        else; delta=dn; end
    end
    return delta,false,max_iter,nh
end

function blp_contraction_draws(delta_init, R, prod_vec, nu_draws, stacked_draws,
                                 obs_key_idx, sigma_vals, sigma_indices,
                                 pi_vals, pi_interactions, coef_dim, pc;
                                 tol=1e-12, max_iter=1500, chunk_size=nothing)
    if chunk_size === nothing || chunk_size >= R
        mu = compute_mu(prod_vec, nu_draws, stacked_draws, obs_key_idx,
                        sigma_vals, sigma_indices, pi_vals, pi_interactions, R, coef_dim)
        return blp_contraction(delta_init, mu, pc, R; tol=tol, max_iter=max_iter)
    end
    N=length(pc.b_mask); NB=sum(pc.b_mask); ND=sum(pc.d_mask)
    delta = if delta_init !== nothing && length(delta_init)==N
        copy(delta_init)
    else
        d=zeros(N); d[pc.d_mask].=pc.ln_s_data_D[pc.d_mask]; d[pc.b_mask].=pc.ln_s_data_B_cond[pc.b_mask]; d
    end
    m_aa=5; Fh=zeros(N,m_aa); Xh=zeros(N,m_aa); ptr=1; hl=0; nh=Float64[]
    for h in 1:max_iter
        sB_acc=zeros(NB); sD_acc=zeros(ND); r0=1
        while r0<=R
            r1=min(r0+chunk_size-1,R); rc=r1-r0+1
            mc=compute_mu(prod_vec,nu_draws[r0:r1,:],stacked_draws[:,r0:r1,:],
                          obs_key_idx,sigma_vals,sigma_indices,pi_vals,pi_interactions,rc,coef_dim)
            sB,sD=compute_model_shares(delta,mc,pc,rc); sB_acc.+=sB.*rc; sD_acc.+=sD.*rc; r0=r1+1
        end
        sB=sB_acc./R; sD=sD_acc./R
        dn=copy(delta)
        dn[pc.d_mask].=delta[pc.d_mask].+pc.ln_s_data_D[pc.d_mask].-log.(clamp.(sD,1e-15,Inf))
        dn[pc.b_mask].=delta[pc.b_mask].+pc.ln_s_data_B_cond[pc.b_mask].-log.(clamp.(sB,1e-15,Inf))
        clamp!(dn,-500.0,500.0)
        !all(isfinite,dn) && return delta,false,h,nh
        f=dn.-delta; norm=maximum(abs.(f)); push!(nh,norm)
        (h%50==0||norm<tol) && println("    [iter=$h] norm=$(round(norm,sigdigits=4))")
        norm<tol && return dn,true,h,nh
        Fh[:,ptr]=f; Xh[:,ptr]=delta; ptr=mod1(ptr+1,m_aa); hl=min(hl+1,m_aa)
        if hl>1
            Fk=Fh[:,1:hl]; dF=Fk[:,2:end].-Fk[:,1:1]
            try
                cb=(dF'*dF+1e-10*I(hl-1))\(-(dF'*Fk[:,1]))
                c=vcat(1.0-sum(cb),cb); da=(Xh[:,1:hl].+Fh[:,1:hl])*c
                delta=all(isfinite,da) ? clamp.(da,-500.0,500.0) : dn
            catch; delta=dn; end
        else; delta=dn; end
    end
    return delta,false,max_iter,nh
end

function build_regressor_matrices(df::DataFrame)
    N = nrow(df)
    spread_cols = coalesce.(df.spread_qoq, 0.0)
    x_mat = zeros(N, L_PROD)
    for (i,col) in enumerate(X_COLS)
        col in names(df) && (x_mat[:,i] .= coalesce.(df[!,col],0.0))
    end
    iv_cols = [c for c in vcat(IV_BLP_LOO,IV_COST,IV_CAPITAL) if c in names(df)]
    Z_mat = zeros(N, length(iv_cols))
    for (i,col) in enumerate(iv_cols)
        v=coalesce.(df[!,col],0.0); replace!(v,Inf=>0.0,-Inf=>0.0); Z_mat[:,i].=v
    end
    return spread_cols, x_mat, Z_mat, iv_cols
end

function project_endogenous_spreads(spread_cols, H, dep_types)
    spread_hat = reshape(copy(spread_cols),:,1)
    for k_val in [4,5]
        mask=dep_types.==k_val; any(mask)||continue
        Hk=H[mask,:]; sk=spread_cols[mask]
        valid=all.(isfinite, eachrow(Hk)).&isfinite.(sk)
        sum(valid)>size(Hk,2)||continue
        try
            b=Hk[valid,:]\sk[valid]; khat=copy(sk); khat[valid].=Hk[valid,:]*b
            spread_hat[mask,1].=khat
        catch; end
    end
    return spread_hat
end

function estimate_theta1(delta, pc::Precomp)
    valid = pc.theta1_valid .& isfinite.(delta)
    theta1 = try pc.X_hat[valid,:]\delta[valid] catch; zeros(size(pc.X_hat,2)) end
    xi = delta .- pc.X_full * theta1
    return theta1, xi
end

function compute_gmm_moments(xi, pc::Precomp)
    return vec(mean(xi .* pc.Z_moments, dims=1))
end

function gmm_objective(theta2, df, prod_vec, nu_draws, stacked_draws, obs_key_idx,
                        sigma_indices, pi_interactions, R, coef_dim, W,
                        tol_inner, max_inner, delta_cache::Ref, pc;
                        chunk_size=nothing)::Float64
    sv,pv = unpack_theta2(theta2, sigma_indices, pi_interactions)
    delta,_,_,_ = blp_contraction_draws(delta_cache[], R, prod_vec, nu_draws, stacked_draws,
                                          obs_key_idx, sv, sigma_indices, pv, pi_interactions,
                                          coef_dim, pc; tol=tol_inner, max_iter=max_inner, chunk_size=chunk_size)
    delta_cache[] = copy(delta)
    _,xi = estimate_theta1(delta, pc)
    G = compute_gmm_moments(xi, pc)
    return dot(G, W*G)
end

function run_blp_for_spec(spec_id::Int, args)
    println("\n" * "="^60); println("  Specification $spec_id"); println("="^60)
    df = try load_merged_spec_data(spec_id; is_hpc=args["hpc"], local_dir=get(args,"local_dir",nothing))
    catch e; println("  [!] No data for spec $spec_id: $e. Skipping."); return nothing; end
    nrow(df)==0 && (println("  [!] Empty. Skipping."); return nothing)
    println("  Loaded $(nrow(df)) observations")

    sigma_indices, pi_interactions, n_params = build_theta2_structure(args["stage"])
    R=args["R"]; seed=args["seed"]; coef_dim=1+L_PROD
    nu_draws = generate_halton_draws(R, coef_dim, seed)
    results  = Dict{String,Any}("spec_id"=>spec_id,"stage"=>args["stage"])

    if args["stage"]=="logit"
        is_B = Bool.(coalesce.(df.is_B, false))
        share_D_clean      = Float64.(coalesce.(df.share_D,      0.0))
        share_B_cond_clean = Float64.(coalesce.(df.share_B_cond, 0.0))
        delta = zeros(nrow(df))
        delta[.!is_B].=log.(clamp.(share_D_clean[.!is_B],1e-15,Inf))
        delta[is_B].=log.(clamp.(share_B_cond_clean[is_B],1e-15,Inf))
        spread_cols,x_mat,Z_mat,iv_cols = build_regressor_matrices(df)
        H = hcat(x_mat,Z_mat); dep_types=Int.(coalesce.(df.deposit_type, 0))
        sh = project_endogenous_spreads(spread_cols,H,dep_types)
        Xf=hcat(spread_cols,x_mat); Xh=hcat(sh,x_mat)
        valid=BitVector(all.(isfinite,eachrow(Xh)))
        cl=string.(df.CodConglomeradoPrudencial).*"_".*first.(split.(string.(df.time_id),"Q"))
        iv_ok=[c for c in iv_cols if std(replace(coalesce.(df[!,c],0.0),Inf=>0.0,-Inf=>0.0))>1e-10]
        Zc=zeros(nrow(df),length(iv_ok))
        for (i,c) in enumerate(iv_ok); v=coalesce.(df[!,c],0.0);replace!(v,Inf=>0.0,-Inf=>0.0);Zc[:,i].=v;end
        pc=build_precomp(df,Zc,Xf,Xh,valid,cl)
        theta1,xi=estimate_theta1(delta,pc); G=compute_gmm_moments(xi,pc)
        W=Matrix(I(length(iv_ok))*1.0); Q=dot(G,W*G)
        results["theta1"]=theta1; results["theta2"]=Float64[]
        results["delta"]=delta; results["xi"]=xi; results["Q_value"]=Q
        results["converged"]=true; results["param_names_theta1"]=vcat(["alpha"],X_COLS)
        println("  theta1 (alpha): $(round(theta1[1],sigdigits=6))"); println("  Q(0)=$(round(Q,sigdigits=6))")
        return results
    end

    println("  Stage: $(uppercase(args["stage"])) ($n_params params)")
    demo_draws = generate_demographic_draws(df,R,seed)
    d_cols=[c for c in D_COLS if c in names(df)]; D_dim=length(d_cols); N_obs=nrow(df)
    mca=string.(df.mca_code); time=string.(df.time_id)
    ukeys=sort(collect(keys(demo_draws))); k2i=Dict(k=>i for (i,k) in enumerate(ukeys))
    sd=permutedims(cat([reshape(demo_draws[k],R,1,D_dim) for k in ukeys]...,dims=2),(2,1,3))
    sdp=vcat(sd,zeros(1,R,D_dim)); padi=size(sd,1)+1
    obs_key_idx=[get(k2i,(mca[i],time[i]),padi) for i in 1:N_obs]

    prod_vec=zeros(N_obs,coef_dim); spreads=coalesce.(df.spread_qoq,0.0); dep_types=Int.(coalesce.(df.deposit_type,0))
    prod_vec[:,1].=spreads
    for (i,col) in enumerate(X_COLS); col in names(df) && (prod_vec[:,1+i].=coalesce.(df[!,col],0.0)); end

    spread_cols,x_mat,Z_mat,iv_cols=build_regressor_matrices(df)
    iv_ok=[c for c in iv_cols if std(replace(coalesce.(df[!,c],0.0),Inf=>0.0,-Inf=>0.0))>1e-10]
    Z=zeros(N_obs,length(iv_ok))
    for (i,c) in enumerate(iv_ok); v=coalesce.(df[!,c],0.0);replace!(v,Inf=>0.0,-Inf=>0.0);Z[:,i].=v; end
    W=try inv(Z'*Z./N_obs) catch; Matrix(I(length(iv_ok))*1.0) end
    H=hcat(x_mat,Z_mat); sh=project_endogenous_spreads(spread_cols,H,dep_types)
    Xf=hcat(spread_cols,x_mat); Xh=hcat(sh,x_mat)
    valid=BitVector(all.(isfinite,eachrow(Xh)))
    cl=string.(df.CodConglomeradoPrudencial).*"_".*first.(split.(string.(df.time_id),"Q"))
    pc=build_precomp(df,Z,Xf,Xh,valid,cl)

    chunk_size=let cs=get(args,"chunk_size",0); (cs===nothing||cs==0) ? nothing : cs; end

    _, out_dir = get_paths(args["hpc"])
    prev_s=Dict("full"=>"sigma","extended"=>"full")
    theta2_0=nothing
    if args["stage"] in keys(prev_s)
        pp=joinpath(out_dir,"blp_checkpoint_spec_$(spec_id)_$(prev_s[args["stage"]]).jld2_compat.jls")
        if isfile(pp)
            try t2p=deserialize(pp)["theta2_star"]; theta2_0=zeros(n_params); theta2_0[1:length(t2p)].=t2p; catch e; end
        end
    end
    theta2_0===nothing && (theta2_0=randn(MersenneTwister(seed),n_params).*0.01)

    get(args,"dry_run",false) && begin
        sv,pv=unpack_theta2(theta2_0,sigma_indices,pi_interactions); t0=time()
        blp_contraction_draws(nothing,R,prod_vec,nu_draws,sdp,obs_key_idx,sv,sigma_indices,pv,pi_interactions,coef_dim,pc;tol=args["tol_inner"],max_iter=10,chunk_size=chunk_size)
        println("  [DRY RUN] $(round(time()-t0,digits=2))s for 10 iters."); println("  Exiting."); return Dict("dry_run"=>true)
    end

    outer_iter=Ref(0); dc=Ref{Union{Vector{Float64},Nothing}}(nothing)
    function obj_fn(t2)
        val=gmm_objective(t2,df,prod_vec,nu_draws,sdp,obs_key_idx,sigma_indices,pi_interactions,R,coef_dim,W,args["tol_inner"],args["max_inner"],dc,pc;chunk_size=chunk_size)
        outer_iter[]+=1; outer_iter[]%10==0 && log_status("  [OUTER=$(outer_iter[])] theta2=$(round.(t2,digits=4))")
        return val
    end

    res=optimize(obj_fn,fill(-15.0,n_params),fill(15.0,n_params),theta2_0,Fminbox(LBFGS()),Optim.Options(iterations=500,f_tol=args["tol_outer"],show_trace=false))
    t2s=Optim.minimizer(res)
    println("  Converged: $(Optim.converged(res))  Q=$(round(Optim.minimum(res),sigdigits=6))")
    sv,pv=unpack_theta2(t2s,sigma_indices,pi_interactions)
    ds,_,_,_=blp_contraction_draws(dc[],R,prod_vec,nu_draws,sdp,obs_key_idx,sv,sigma_indices,pv,pi_interactions,coef_dim,pc;tol=args["tol_inner"],max_iter=args["max_inner"],chunk_size=chunk_size)
    t1s,xis=estimate_theta1(ds,pc)

    results["theta1"]=t1s; results["theta2"]=t2s; results["delta"]=ds; results["xi"]=xis
    results["Q_value"]=Optim.minimum(res); results["converged"]=Optim.converged(res)
    results["param_names_theta1"]=vcat(["alpha"],X_COLS)
    println("  theta1 (alpha): $(round(t1s[1],sigdigits=6))")

    mkpath(out_dir)
    chk=joinpath(out_dir,"blp_checkpoint_spec_$(spec_id)_$(args["stage"]).jld2_compat.jls")
    try serialize(chk,Dict("theta2_star"=>t2s,"delta_star"=>ds)); println("  [CHECKPOINT] $(basename(chk))") catch e; end
    return results
end

function parse_args_blp()
    s=ArgParseSettings(description="BLP Demand Estimation Loop – Spec 2")
    @add_arg_table! s begin
        "--spec";    arg_type=String;  default="12"
        "--stage";   arg_type=String;  default="logit"
        "--R";       arg_type=Int;     default=100
        "--seed";    arg_type=Int;     default=42
        "--tol-inner"; arg_type=Float64; default=1e-12; dest_name="tol_inner"
        "--max-inner"; arg_type=Int;   default=2000;    dest_name="max_inner"
        "--tol-outer"; arg_type=Float64; default=1e-6;  dest_name="tol_outer"
        "--method";  arg_type=String;  default="l-bfgs-b"
        "--workers"; arg_type=Int;     default=Threads.nthreads()
        "--hpc";     action=:store_true
        "--local-dir"; arg_type=String; default=nothing; dest_name="local_dir"
        "--dry-run"; action=:store_true; dest_name="dry_run"
        "--chunk-size"; arg_type=Int;  default=0;       dest_name="chunk_size"
    end
    return parse_args(s)
end

function main()
    args=parse_args_blp()
    spec_ids=args["spec"]=="all" ? collect(1:12) : [parse(Int,args["spec"])]
    log_status("BLP Demand Estimation – Spec$(SPEC_NUM) – START")
    log_status("  Stage: $(args["stage"]) | Specs: $spec_ids | R=$(args["R"])")
    _,out_dir=get_paths(args["hpc"];local_dir=get(args,"local_dir",nothing)); mkpath(out_dir)
    stages=args["stage"]=="sequence" ? ["logit","sigma","full","extended"] : [args["stage"]]
    for cs in stages
        args["stage"]=cs; all_res=Dict{Int,Any}(); lk=ReentrantLock()
        Threads.@threads for sp in spec_ids
            res=try run_blp_for_spec(sp,args) catch e; println("  [!] Spec $sp failed: $e"); nothing; end
            res===nothing&&continue; get(res,"dry_run",false)&&continue
            pkl=joinpath(out_dir,"blp_results_spec_2_$(sp)_$(cs).jls")
            serialize(pkl,res); println("  Saved: $(basename(pkl))")
            lock(lk) do
                all_res[sp]=Dict("Q_value"=>get(res,"Q_value",0.0),"converged"=>get(res,"converged",true),
                                 "theta1_alpha"=>isempty(get(res,"theta1",Float64[])) ? [] : [res["theta1"][1]],
                                 "theta2"=>get(res,"theta2",Float64[]),"stage"=>cs)
            end
        end
        sp_path=joinpath(out_dir,"blp_summary_$(SPEC_NUM)_$(cs).json")
        open(sp_path,"w") do f; JSON3.write(f,all_res); end
        log_status("[DONE] Stage=$cs  Summary: $(basename(sp_path))")
    end
end

main()

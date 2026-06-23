"""
blp_estimation_gpu.jl
=====================
GPU-accelerated BLP demand estimation.

Reuses all CPU utilities from blp_estimation.jl via include().
Only the hot-path inner loop (logsumexp grouped reductions + share
computation) runs on GPU; SQUAREM delta updates and outer GMM
optimisation remain on CPU.

Design principles
-----------------
* Float32 on device — halves bandwidth vs Float64; on Tensor-Core GPUs
  (A100/V100) the logsumexp kernels run ~16× faster than Float64 BLAS.
* Static index arrays (sort_b, grp_start, …) are uploaded once in
  allocate_gpu_buffers and never touched again.
* PT_agg is tiny (n_times × n_pairs ≈ 60×500) — stored dense on device
  so CUBLAS GEMM handles sum_wtd without a sparse-matrix call.
* mu is computed by MKL on CPU (fast, small k-dim), then bulk-copied to
  mu_gpu once per outer optimiser iteration before each contraction.

Requirements
------------
* NVIDIA GPU with CUDA >= 11 / CUDA.jl >= 5
* Yale Bouchet: `module load CUDA/12.x` before julia

Usage (cluster GPU node)
------------------------
  julia --project=\${PROJECT_DIR} --threads=4 \\
      blp_estimation_gpu.jl --estim 5 --spec 12 --stage sigma \\
      --R 2000 --seed 42 --hpc
"""

# ── Load CPU baseline ────────────────────────────────────────────────────────
# Check if baseline has already been loaded (e.g., via blp_2_estimation.jl)
# to avoid constant redefinition warnings when both blp_1 and blp_2 GPU scripts run.
if !isdefined(Main, :X_COLS)
    include(joinpath(@__DIR__, "blp_1_estimation.jl"))
end

using CUDA
CUDA.allowscalar(false)   # hard-fail on accidental scalar GPU indexing

# ── Device floating-point precision ─────────────────────────────────────────
# Change to Float64 if the GPU has fast FP64 (e.g. A100 SXM) and you need
# tighter numerical tolerances on the inner contraction.
const GPU_T = Float64

# ==========================================================================
# GPU Buffers
# ==========================================================================
"""
    GpuBuffers

Holds `CuArray` mirrors of the hot-path matrices used inside the BLP inner
loop.  Fields are organised into four groups:

1. **Per-observation tensors** (`V_B`, `V_D`, `q_B`, `mu_gpu`, …) —
   overwritten every `T_inplace!` call.
2. **Grouped reduction results** (`log_sum_D_time`, `log_denom`, …) —
   intermediate outputs of the CUDA kernels.
3. **Output vectors** (`s_B_gpu`, `s_D_gpu`) — copied back to CPU after
   each call so SQUAREM can update `delta_work`.
4. **Static index arrays** (`sort_b_gpu`, `b_grp_start_gpu`, …) —
   uploaded once in `allocate_gpu_buffers`, never modified.

CPU fields that require arbitrary-precision arithmetic (delta, mu, theta1
solve, SQUAREM step) live in the ordinary `HotBuffers`.
"""
mutable struct GpuBuffers
    # ── Per-observation tensors (rewritten every contraction step) ────────
    V_B              ::CuMatrix{GPU_T}   # N_B × R
    V_D              ::CuMatrix{GPU_T}   # N_D × R
    q_B              ::CuMatrix{GPU_T}   # N_B × R
    delta_B          ::CuVector{GPU_T}   # N_B
    delta_D          ::CuVector{GPU_T}   # N_D
    mu_gpu           ::CuMatrix{GPU_T}   # N   × R  (bulk-copy of buf.mu each outer step)

    # ── Grouped reduction intermediate results ────────────────────────────
    log_sum_D_time   ::CuMatrix{GPU_T}   # n_times × R
    log_sum_B_mkt    ::CuMatrix{GPU_T}   # n_pairs × R
    log_D_sum_pair   ::CuMatrix{GPU_T}   # n_pairs × R
    log_denom        ::CuMatrix{GPU_T}   # n_pairs × R
    neg_log_denom    ::CuMatrix{GPU_T}   # n_pairs × R
    max_neg_ld       ::CuMatrix{GPU_T}   # n_times × R
    shifted_inv      ::CuMatrix{GPU_T}   # n_pairs × R
    sum_wtd          ::CuMatrix{GPU_T}   # n_times × R
    log_inv_wtd      ::CuMatrix{GPU_T}   # n_times × R
    log_s_D_r        ::CuMatrix{GPU_T}   # N_D  × R
    row_max_D        ::CuVector{GPU_T}   # N_D  (per-obs row max for s_D mean)

    # ── Output (copied back to CPU after each T_inplace! call) ────────────
    s_B_gpu          ::CuVector{GPU_T}   # N_B
    s_D_gpu          ::CuVector{GPU_T}   # N_D

    # ── Static index arrays (uploaded once at allocation time) ────────────
    # Observation-to-group mappings
    b_mkt_idx_gpu    ::CuVector{Int32}   # N_B  → pair index  (1-based)
    d_time_enc_gpu   ::CuVector{Int32}   # N_D  → time index  (1-based)
    pair_time_enc_gpu::CuVector{Int32}   # n_pairs → time index (1-based)

    # Original position of B / D observations in the full N-vector
    b_mask_idx_gpu   ::CuVector{Int32}   # N_B  (position in 1:N)
    d_mask_idx_gpu   ::CuVector{Int32}   # N_D  (position in 1:N)

    # Sorted-order arrays for grouped reductions (B groups = mkt-pairs)
    sort_b_gpu       ::CuVector{Int32}   # N_B
    b_uval_gpu       ::CuVector{Int32}   # n_pairs  (== 1:n_pairs; kept for uniform kernel)
    b_grp_start_gpu  ::CuVector{Int32}   # n_pairs+1  (sentinel-ended)

    # Sorted-order arrays for D groups (= time periods)
    sort_d_gpu       ::CuVector{Int32}   # N_D
    d_uval_gpu       ::CuVector{Int32}   # n_d_groups (unique time indices in d_time_enc)
    d_grp_start_gpu  ::CuVector{Int32}   # n_d_groups+1 (sentinel-ended)

    # Sorted-order arrays for pair-time groups (used for max_neg_ld)
    sort_pt_gpu      ::CuVector{Int32}   # n_pairs
    pt_uval_gpu      ::CuVector{Int32}   # n_pt_groups
    pt_grp_start_gpu ::CuVector{Int32}   # n_pt_groups+1 (sentinel-ended)

    # PT_agg dense mirror for CUBLAS GEMM (n_times × n_pairs)
    PT_agg_gpu       ::CuMatrix{GPU_T}

    # ── Dimensions (stored for kernel-launch arithmetic) ──────────────────
    N                ::Int
    N_B              ::Int
    N_D              ::Int
    R                ::Int
    n_pairs          ::Int
    n_times          ::Int
    n_b_groups       ::Int   # == n_pairs
    n_d_groups       ::Int   # == length(pc.d_uval)
    n_pt_groups      ::Int   # == length(pc.pt_uval)
end

# ==========================================================================
# Helpers
# ==========================================================================

"""
    _sentinel_grp_start(grp_start, n_obs) → Vector{Int32}

Appends a sentinel entry `n_obs + 1` so kernels can compute group bounds
as `grp_start[g] : grp_start[g+1] - 1` without a bounds check.
"""
function _sentinel_grp_start(grp_start::Vector{Int}, n_obs::Int)::Vector{Int32}
    return Int32.(vcat(grp_start, n_obs + 1))
end

# ==========================================================================
# Allocator
# ==========================================================================

"""
    allocate_gpu_buffers(buf, pc, N, N_B, N_D, R, n_pairs, n_times) → GpuBuffers

Allocates all device arrays and uploads static index/weight arrays once.
`buf` and `pc` are the existing CPU `HotBuffers` / `Precomp` from the
normal CPU estimation path.

Call once per spec; reuse across all outer optimiser iterations.
"""
function allocate_gpu_buffers(buf::HotBuffers, pc::Precomp,
                               N::Int, N_B::Int, N_D::Int,
                               R::Int, n_pairs::Int, n_times::Int)::GpuBuffers

    n_d_groups  = length(pc.d_uval)
    n_pt_groups = length(pc.pt_uval)
    n_b_groups  = n_pairs   # one group per market-pair

    # Footprint estimate (GPU_T bytes per element)
    bytes_per = sizeof(GPU_T)
    gb = (
        N_B * R * 3 +   # V_B, q_B, (reused for shifted_inv borrowing)
        N_D * R * 2 +   # V_D, log_s_D_r
        N   * R     +   # mu_gpu
        N_B         +   # delta_B, s_B_gpu  (×2)
        N_D         +   # delta_D, s_D_gpu, row_max_D  (×3)
        n_times * R * 4 +   # log_sum_D_time, max_neg_ld, sum_wtd, log_inv_wtd
        n_pairs * R * 5     # log_sum_B_mkt, log_D_sum_pair, log_denom,
                            # neg_log_denom, shifted_inv
    ) * bytes_per / 1e9
    println("  [GPU] Allocating buffers — estimated footprint: $(round(gb, digits=2)) GB")

    # Static index arrays
    b_mask_idx_cpu  = Int32.(findall(pc.b_mask))
    d_mask_idx_cpu  = Int32.(findall(pc.d_mask))
    b_uval_cpu      = Int32.(1:n_pairs)

    b_grp_sent  = _sentinel_grp_start(pc.b_grp_start,  N_B)
    d_grp_sent  = _sentinel_grp_start(pc.d_grp_start,  N_D)
    pt_grp_sent = _sentinel_grp_start(pc.pt_grp_start, n_pairs)

    # PT_agg: sparse → dense for CUBLAS (n_times × n_pairs, tiny)
    PT_agg_dense = Matrix{GPU_T}(pc.PT_agg)

    gbuf = GpuBuffers(
        # Per-observation tensors
        CUDA.zeros(GPU_T, N_B, R),           # V_B
        CUDA.zeros(GPU_T, N_D, R),           # V_D
        CUDA.zeros(GPU_T, N_B, R),           # q_B
        CUDA.zeros(GPU_T, N_B),              # delta_B
        CUDA.zeros(GPU_T, N_D),              # delta_D
        CUDA.zeros(GPU_T, N,  R),            # mu_gpu

        # Grouped reduction intermediates
        CUDA.fill(GPU_T(-Inf32), n_times, R), # log_sum_D_time
        CUDA.fill(GPU_T(-Inf32), n_pairs, R), # log_sum_B_mkt
        CUDA.zeros(GPU_T, n_pairs, R),        # log_D_sum_pair
        CUDA.zeros(GPU_T, n_pairs, R),        # log_denom
        CUDA.zeros(GPU_T, n_pairs, R),        # neg_log_denom
        CUDA.fill(GPU_T(-Inf32), n_times, R), # max_neg_ld
        CUDA.zeros(GPU_T, n_pairs, R),        # shifted_inv
        CUDA.zeros(GPU_T, n_times, R),        # sum_wtd
        CUDA.zeros(GPU_T, n_times, R),        # log_inv_wtd
        CUDA.zeros(GPU_T, N_D, R),            # log_s_D_r
        CUDA.fill(GPU_T(-Inf32), N_D),        # row_max_D

        # Output
        CUDA.zeros(GPU_T, N_B),              # s_B_gpu
        CUDA.zeros(GPU_T, N_D),              # s_D_gpu

        # Static index arrays — uploaded once
        CuVector{Int32}(Int32.(pc.b_mkt_idx)),
        CuVector{Int32}(Int32.(pc.d_time_enc)),
        CuVector{Int32}(Int32.(pc.pair_time_enc)),
        CuVector{Int32}(b_mask_idx_cpu),
        CuVector{Int32}(d_mask_idx_cpu),
        CuVector{Int32}(Int32.(pc.sort_b)),
        CuVector{Int32}(b_uval_cpu),
        CuVector{Int32}(b_grp_sent),
        CuVector{Int32}(Int32.(pc.sort_d)),
        CuVector{Int32}(Int32.(pc.d_uval)),
        CuVector{Int32}(d_grp_sent),
        CuVector{Int32}(Int32.(pc.sort_pt)),
        CuVector{Int32}(Int32.(pc.pt_uval)),
        CuVector{Int32}(pt_grp_sent),
        CuMatrix{GPU_T}(PT_agg_dense),

        # Dimensions
        N, N_B, N_D, R,
        n_pairs, n_times,
        n_b_groups, n_d_groups, n_pt_groups,
    )

    free_b, total_b = CUDA.memory_info()
    @printf("  [GPU] VRAM after alloc: %.2f / %.2f GB free\n", free_b / 2^30, total_b / 2^30)
    return gbuf
end

# ==========================================================================
# CUDA Kernels
# ==========================================================================

# ── Kernel tuning constants ──────────────────────────────────────────────────
# RBLOCK: number of draws handled per CUDA thread block column.
# 32 = one warp; keeps shared-memory pressure low while hiding memory latency.
const RBLOCK = Int32(32)

"""
    logsumexp_groups_kernel!(result, V, sort_idx, uval, grp_start,
                              n_groups, N_obs, R)

For each group `g` (1-based) and draw `r`:

  result[uval[g], r] = max_i V[sort_idx[i], r]
                     + log Σ_i exp(V[sort_idx[i], r] − max_i)

where the sum ranges over `i ∈ grp_start[g] : grp_start[g+1]-1`.

**Grid / block layout**

  gridDim  = (n_groups, cld(R, RBLOCK))
  blockDim = (RBLOCK, 1, 1)

Each thread handles exactly one `(g, r)` pair: sequential scan over the
group for the max pass, then a second scan for the exp-sum.  No shared
memory or atomics are needed because every thread writes to a distinct
`result` cell.

**Arguments** (all device arrays)

- `result`    — output matrix, row-major layout `[n_result_rows, R]`
- `V`         — value matrix `[N_obs, R]`
- `sort_idx`  — observation sort order `[N_obs]`, 1-based
- `uval`      — group → result-row mapping `[n_groups]`, 1-based
- `grp_start` — sentinel-ended group starts `[n_groups+1]`, 1-based
- `n_groups`  — number of groups (Int32)
- `N_obs`     — total observations (Int32, for bounds guard)
- `R`         — number of draws (Int32)
"""
function logsumexp_groups_kernel!(
        result   ::CuDeviceMatrix{GPU_T},
        V        ::CuDeviceMatrix{GPU_T},
        sort_idx ::CuDeviceVector{Int32},
        uval     ::CuDeviceVector{Int32},
        grp_start::CuDeviceVector{Int32},
        n_groups ::Int32,
        N_obs    ::Int32,
        R        ::Int32)

    g = Int32(blockIdx().x)
    r = Int32((blockIdx().y - 1) * blockDim().x + threadIdx().x)

    (g > n_groups || r > R) && return nothing

    gs  = grp_start[g]
    ge  = grp_start[g + Int32(1)] - Int32(1)
    row = uval[g]

    # Single-pass online log-sum-exp: maintain running max `mx` and running sum
    # `s = Σ exp(v_j − mx)`; when a larger value appears, rescale `s` to the new
    # max before adding. Reads each V element ONCE (vs twice for the max-then-sum
    # two-pass), halving the gather traffic of this bandwidth-bound kernel.
    # Numerically identical to the two-pass form (verify_gpu_shares guards it).
    #   First element: mx=-Inf, s=0 → s = 0·exp(-Inf) + 1 = 1, mx = v  (exact).
    mx = GPU_T(-Inf32)
    s  = GPU_T(0)
    @inbounds for ii in gs:ge
        v = V[sort_idx[ii], r]
        if v > mx
            s  = s * CUDA.exp(mx - v) + GPU_T(1)
            mx = v
        else
            s += CUDA.exp(v - mx)
        end
    end

    @inbounds result[row, r] = mx + CUDA.log(max(s, GPU_T(1e-30)))
    return nothing
end

"""
    groupmax_kernel!(result, V, sort_idx, uval, grp_start,
                     n_groups, N_obs, R)

Fills `result[uval[g], r]` with the **maximum** of `V[sort_idx[i], r]`
over all `i` in group `g`.  Same grid/block layout as
`logsumexp_groups_kernel!`.

Used to compute `max_neg_ld[t, r]` — the per-time-period maximum of
`neg_log_denom` over all market-pairs in period `t`.  This maximum is
needed to numerically stabilise the `shifted_inv` computation:

  shifted_inv[p, r] = exp(neg_log_denom[p, r] − max_neg_ld[time(p), r])
"""
function groupmax_kernel!(
        result   ::CuDeviceMatrix{GPU_T},
        V        ::CuDeviceMatrix{GPU_T},
        sort_idx ::CuDeviceVector{Int32},
        uval     ::CuDeviceVector{Int32},
        grp_start::CuDeviceVector{Int32},
        n_groups ::Int32,
        N_obs    ::Int32,
        R        ::Int32)

    g = Int32(blockIdx().x)
    r = Int32((blockIdx().y - 1) * blockDim().x + threadIdx().x)

    (g > n_groups || r > R) && return nothing

    gs  = grp_start[g]
    ge  = grp_start[g + Int32(1)] - Int32(1)
    row = uval[g]

    mx = GPU_T(-Inf32)
    @inbounds for ii in gs:ge
        v = V[sort_idx[ii], r]
        v > mx && (mx = v)
    end

    @inbounds result[row, r] = mx
    return nothing
end

# ── Convenience launch wrappers ──────────────────────────────────────────────

"""
    launch_logsumexp!(result, V, sort_idx, uval, grp_start, n_groups, N_obs, R)

Compute grouped log-sum-exp on the GPU.

`result` is filled **in-place** (`fill!(-Inf)` is called first so rows
belonging to absent groups remain `-Inf`).
"""
function launch_logsumexp!(result::CuMatrix{GPU_T},
                            V        ::CuMatrix{GPU_T},
                            sort_idx ::CuVector{Int32},
                            uval     ::CuVector{Int32},
                            grp_start::CuVector{Int32},
                            n_groups ::Int, N_obs::Int, R::Int)
    fill!(result, GPU_T(-Inf32))
    n_groups == 0 && return
    grid  = (n_groups, cld(R, Int(RBLOCK)))
    block = (Int(RBLOCK),)
    CUDA.@cuda threads=block blocks=grid logsumexp_groups_kernel!(
        result, V, sort_idx, uval, grp_start,
        Int32(n_groups), Int32(N_obs), Int32(R))
end

"""
    launch_groupmax!(result, V, sort_idx, uval, grp_start, n_groups, N_obs, R)

Compute grouped maximum on the GPU (`result` filled in-place).
"""
function launch_groupmax!(result::CuMatrix{GPU_T},
                           V        ::CuMatrix{GPU_T},
                           sort_idx ::CuVector{Int32},
                           uval     ::CuVector{Int32},
                           grp_start::CuVector{Int32},
                           n_groups ::Int, N_obs::Int, R::Int)
    fill!(result, GPU_T(-Inf32))
    n_groups == 0 && return
    grid  = (n_groups, cld(R, Int(RBLOCK)))
    block = (Int(RBLOCK),)
    CUDA.@cuda threads=block blocks=grid groupmax_kernel!(
        result, V, sort_idx, uval, grp_start,
        Int32(n_groups), Int32(N_obs), Int32(R))
end

# ==========================================================================
# GPU Share Computation
# ==========================================================================

"""
    compute_model_shares_gpu!(buf, gbuf, delta, pc, R)

GPU-accelerated drop-in replacement for `compute_model_shares!`.

Steps and GPU ops:
  1. Copy buf.mu (CPU Float64) → gbuf.mu_gpu (GPU Float32)
  2. Scatter δ → delta_B / delta_D via index gather
  3. V_B = clamp(δ_B + μ_B, -500, 500); same for V_D  [broadcast]
  4. log_sum_D_time via logsumexp_groups_kernel!
  5. Broadcast log_sum_D_time → log_D_sum_pair (pair-level)
  6. log_sum_B_mkt via logsumexp_groups_kernel!
  7. log_denom = log(outside + Σ_B + Σ_D)  [broadcast]
  8. q_B = exp(V_B − log_denom[mkt])        [broadcast]
  9. s_B = mean_r q_B                        [sum + scale]
 10. max_neg_ld via groupmax_kernel!
 11. shifted_inv = exp(neg_log_denom − max_neg_ld)  [broadcast]
 12. sum_wtd = PT_agg × shifted_inv          [CUBLAS GEMM]
 13. log_inv_wtd = max_neg_ld + log(sum_wtd) [broadcast]
 14. log_s_D_r = V_D + log_inv_wtd           [broadcast]
 15. s_D via row-max numerically stable mean  [broadcast + sum]
 16. Copy s_B, s_D back to CPU Float64

GPU_T precision (Float32) is used throughout; s_B and s_D are widened
to Float64 when written back to buf.
"""
function compute_model_shares_gpu!(buf::HotBuffers, gbuf::GpuBuffers,
                                    delta::Vector{Float64},
                                    pc::Precomp, R::Int)

    N_B     = gbuf.N_B
    N_D     = gbuf.N_D
    n_pairs = gbuf.n_pairs
    n_times = gbuf.n_times

    # ── 1. Copy mu (CPU Float64 → GPU Float32) ──────────────────────────
    copyto!(gbuf.mu_gpu, GPU_T.(buf.mu))

    # ── 2. Scatter δ → delta_B / delta_D ────────────────────────────────
    delta_gpu_full = CuVector{GPU_T}(GPU_T.(delta))
    gbuf.delta_B .= @view delta_gpu_full[gbuf.b_mask_idx_gpu]
    gbuf.delta_D .= @view delta_gpu_full[gbuf.d_mask_idx_gpu]

    # ── 3. V_B / V_D ────────────────────────────────────────────────────
    gbuf.V_B .= clamp.(gbuf.delta_B .+ @view(gbuf.mu_gpu[gbuf.b_mask_idx_gpu, :]),
                       GPU_T(-500f0), GPU_T(500f0))
    gbuf.V_D .= clamp.(gbuf.delta_D .+ @view(gbuf.mu_gpu[gbuf.d_mask_idx_gpu, :]),
                       GPU_T(-500f0), GPU_T(500f0))

    # ── 4. log_sum_D_time ────────────────────────────────────────────────
    launch_logsumexp!(gbuf.log_sum_D_time,
                      gbuf.V_D, gbuf.sort_d_gpu, gbuf.d_uval_gpu,
                      gbuf.d_grp_start_gpu,
                      gbuf.n_d_groups, N_D, R)

    # ── 5. Broadcast to pairs ────────────────────────────────────────────
    gbuf.log_D_sum_pair .= @view gbuf.log_sum_D_time[gbuf.pair_time_enc_gpu, :]

    # ── 6. log_sum_B_mkt ────────────────────────────────────────────────
    launch_logsumexp!(gbuf.log_sum_B_mkt,
                      gbuf.V_B, gbuf.sort_b_gpu, gbuf.b_uval_gpu,
                      gbuf.b_grp_start_gpu,
                      gbuf.n_b_groups, N_B, R)

    # ── 7. log_denom ─────────────────────────────────────────────────────
    # log_outside = 0 (one outside good with utility 0)
    let lsB = gbuf.log_sum_B_mkt, lsD = gbuf.log_D_sum_pair
        jm = max.(GPU_T(0f0), lsB, lsD)
        gbuf.log_denom .= jm .+ log.(max.(
            exp.(GPU_T(0f0) .- jm) .+ exp.(lsB .- jm) .+ exp.(lsD .- jm),
            GPU_T(1e-30)))
    end

    # ── 8–9. q_B → s_B ──────────────────────────────────────────────────
    gbuf.q_B     .= exp.(gbuf.V_B .- @view(gbuf.log_denom[gbuf.b_mkt_idx_gpu, :]))
    gbuf.s_B_gpu .= vec(sum(gbuf.q_B; dims=2)) .* GPU_T(1.0 / R)

    # ── 10. max_neg_ld ───────────────────────────────────────────────────
    gbuf.neg_log_denom .= .-gbuf.log_denom
    launch_groupmax!(gbuf.max_neg_ld,
                     gbuf.neg_log_denom, gbuf.sort_pt_gpu, gbuf.pt_uval_gpu,
                     gbuf.pt_grp_start_gpu,
                     gbuf.n_pt_groups, n_pairs, R)

    # ── 11. shifted_inv ──────────────────────────────────────────────────
    gbuf.shifted_inv .= exp.(gbuf.neg_log_denom .-
                             @view(gbuf.max_neg_ld[gbuf.pair_time_enc_gpu, :]))

    # ── 12. sum_wtd = PT_agg × shifted_inv  (CUBLAS GEMM) ───────────────
    mul!(gbuf.sum_wtd, gbuf.PT_agg_gpu, gbuf.shifted_inv)

    # ── 13. log_inv_wtd ──────────────────────────────────────────────────
    gbuf.log_inv_wtd .= gbuf.max_neg_ld .+
                        log.(max.(gbuf.sum_wtd, GPU_T(1e-30)))

    # ── 14. log_s_D_r ────────────────────────────────────────────────────
    gbuf.log_s_D_r .= gbuf.V_D .+ @view(gbuf.log_inv_wtd[gbuf.d_time_enc_gpu, :])

    # ── 15. s_D via row-max trick ────────────────────────────────────────
    gbuf.row_max_D .= vec(maximum(gbuf.log_s_D_r; dims=2))
    gbuf.s_D_gpu   .= exp.(gbuf.row_max_D) .*
                      (vec(sum(exp.(gbuf.log_s_D_r .- gbuf.row_max_D); dims=2)) .*
                       GPU_T(1.0 / R))

    # ── 16. Copy results back to CPU Float64 ─────────────────────────────
    CUDA.synchronize()
    buf.s_B .= Float64.(Array(gbuf.s_B_gpu))
    buf.s_D .= Float64.(Array(gbuf.s_D_gpu))
    return nothing
end

# ==========================================================================
# GPU↔CPU Consistency Guard
# ==========================================================================

"""
    verify_gpu_shares(buf, gbuf, delta, pc, R; rtol=1e-9) → Float64

The callers pass `rtol = tol_inner`: the SQUAREM residual floor equals the
share relative error, so shares must agree to `tol_inner` for the inner loop to
reach it. (The original `1f0/R` bug gave a ~3.6e-9 share error — above a
`tol_inner` of 1e-10, hence non-convergence — which this guard now flags.)

Compute model shares at `delta` on BOTH the CPU reference
(`compute_model_shares!`) and the GPU (`compute_model_shares_gpu!`) and compare.
Returns the max relative difference; **errors** if it exceeds `rtol`.

This is a cheap startup guard against silent GPU precision regressions — e.g. a
Float32 `1/R` scaling factor — that don't crash but bias every share by a constant
factor, pinning the SQUAREM residual above `tol_inner` so the inner loop spins to
`max_iter` every time. Such a bug shows up here as a clear failure instead of a
day-long job that quietly never converges.

`buf.mu` must already be populated (call `compute_mu!` at the same θ₂ first).
Both share routines read `buf.mu` and write `buf.s_B`/`buf.s_D`, so the GPU call
leaves `buf` holding the GPU shares on return.
"""
function verify_gpu_shares(buf::HotBuffers, gbuf::GpuBuffers,
                            delta::Vector{Float64}, pc::Precomp, R::Int;
                            rtol::Float64=1e-9)
    # CPU reference shares
    compute_model_shares!(buf, delta, pc, R)
    cpu_sB = copy(buf.s_B); cpu_sD = copy(buf.s_D)

    # GPU shares (overwrites buf.s_B / buf.s_D)
    compute_model_shares_gpu!(buf, gbuf, delta, pc, R)

    maxrel(c::Vector{Float64}, g::Vector{Float64}) = begin
        m = 0.0
        @inbounds for i in eachindex(c)
            d = abs(g[i] - c[i]) / max(abs(c[i]), 1e-12)
            d > m && (m = d)
        end
        m
    end
    eB = maxrel(cpu_sB, buf.s_B)
    eD = maxrel(cpu_sD, buf.s_D)
    emax = max(eB, eD)

    # Floored log-share diff — the *actual* SQUAREM residual ingredient
    # (T(δ) = δ + ln_s_data − log(max(s, 1e-15))).  Raw-share agreement can hide a
    # mismatch that only appears after the log-floor: a tiny-share obs floored at a
    # slightly-off constant (e.g. Float32 `1f-15` = 1.0000000363e-15) leaves the raw
    # shares equal yet gaps log(s) by ~3.6e-9, pinning the contraction above tol_inner.
    # Floor at the SAME exact 1e-15 the CPU/GPU contraction uses.
    maxabs_logfloor(c::Vector{Float64}, g::Vector{Float64}) = begin
        m = 0.0
        @inbounds for i in eachindex(c)
            d = abs(log(max(g[i], 1e-15)) - log(max(c[i], 1e-15)))
            d > m && (m = d)
        end
        m
    end
    lB = maxabs_logfloor(cpu_sB, buf.s_B)
    lD = maxabs_logfloor(cpu_sD, buf.s_D)
    lmax = max(lB, lD)

    log_status("  [GPU CHECK] max rel share diff CPU vs GPU: " *
               "B=$(round(eB, sigdigits=3)) | D=$(round(eD, sigdigits=3))")
    log_status("  [GPU CHECK] max |Δ log(max(s,1e-15))| (residual ingredient): " *
               "B=$(round(lB, sigdigits=3)) | D=$(round(lD, sigdigits=3))")
    if emax > rtol || lmax > rtol
        error("[GPU CHECK FAILED] GPU vs CPU mismatch: raw-share rel=$(round(emax, sigdigits=4)), " *
              "log-floor abs=$(round(lmax, sigdigits=4)) (rtol=$rtol). Likely a GPU precision " *
              "regression — a Float32 scaling factor (`1f0/R`) or log-floor literal (`1f-15`). " *
              "The inner loop will not converge to tol_inner — aborting before wasting the job.")
    end
    log_status("  [GPU CHECK] ✓ GPU matches CPU within rtol=$rtol (shares and log-floor)")
    return max(emax, lmax)
end

# ==========================================================================
# GPU Device-Resident Share Computation (for on-device SQUAREM)
# ==========================================================================

"""
    model_shares_dev!(gbuf, delta_gpu, mu_B, mu_D, s_model_full, R)

Device-resident sibling of `compute_model_shares_gpu!`.  Differences:

* `delta_gpu` is a **CuVector already on device** — no host→device copy per call.
* `gbuf.mu_gpu` is assumed already populated (uploaded **once** by the caller
  before the inner loop) — no per-step mu copy.
* Model shares are scattered into the device vector `s_model_full` (length N)
  — **no device→host copy**.

This keeps the entire SQUAREM contraction on the GPU so the device stays busy
across all inner iterations (raises GPU utilisation; eliminates per-step PCIe
transfers).  `compute_model_shares_gpu!` is retained unchanged for the IFT
forward passes, which need `buf.s_B`/`buf.s_D` back on the CPU.
"""
function model_shares_dev!(gbuf::GpuBuffers, delta_gpu::CuVector{GPU_T},
                            mu_B::CuMatrix{GPU_T}, mu_D::CuMatrix{GPU_T},
                            s_model_full::CuVector{GPU_T}, R::Int)
    N_B = gbuf.N_B; N_D = gbuf.N_D
    n_pairs = gbuf.n_pairs

    # Gather δ → delta_B / delta_D (delta already on device).
    # @view fuses the indexed gather into the assignment — no materialised temp.
    gbuf.delta_B .= @view delta_gpu[gbuf.b_mask_idx_gpu]
    gbuf.delta_D .= @view delta_gpu[gbuf.d_mask_idx_gpu]

    # V_B / V_D (mu_B / mu_D pre-gathered once per outer iter — μ is constant
    # during the inner loop, so we avoid re-gathering the N×R μ matrix per step)
    gbuf.V_B .= clamp.(gbuf.delta_B .+ mu_B, GPU_T(-500f0), GPU_T(500f0))
    gbuf.V_D .= clamp.(gbuf.delta_D .+ mu_D, GPU_T(-500f0), GPU_T(500f0))

    launch_logsumexp!(gbuf.log_sum_D_time, gbuf.V_D, gbuf.sort_d_gpu,
                      gbuf.d_uval_gpu, gbuf.d_grp_start_gpu, gbuf.n_d_groups, N_D, R)
    gbuf.log_D_sum_pair .= @view gbuf.log_sum_D_time[gbuf.pair_time_enc_gpu, :]
    launch_logsumexp!(gbuf.log_sum_B_mkt, gbuf.V_B, gbuf.sort_b_gpu,
                      gbuf.b_uval_gpu, gbuf.b_grp_start_gpu, gbuf.n_b_groups, N_B, R)
    let lsB = gbuf.log_sum_B_mkt, lsD = gbuf.log_D_sum_pair
        jm = max.(GPU_T(0f0), lsB, lsD)
        gbuf.log_denom .= jm .+ log.(max.(
            exp.(GPU_T(0f0) .- jm) .+ exp.(lsB .- jm) .+ exp.(lsD .- jm),
            GPU_T(1e-30)))
    end
    gbuf.q_B     .= exp.(gbuf.V_B .- @view(gbuf.log_denom[gbuf.b_mkt_idx_gpu, :]))
    gbuf.s_B_gpu .= vec(sum(gbuf.q_B; dims=2)) .* GPU_T(1.0 / R)

    gbuf.neg_log_denom .= .-gbuf.log_denom
    launch_groupmax!(gbuf.max_neg_ld, gbuf.neg_log_denom, gbuf.sort_pt_gpu,
                     gbuf.pt_uval_gpu, gbuf.pt_grp_start_gpu, gbuf.n_pt_groups, n_pairs, R)
    gbuf.shifted_inv .= exp.(gbuf.neg_log_denom .-
                             @view(gbuf.max_neg_ld[gbuf.pair_time_enc_gpu, :]))
    mul!(gbuf.sum_wtd, gbuf.PT_agg_gpu, gbuf.shifted_inv)
    gbuf.log_inv_wtd .= gbuf.max_neg_ld .+ log.(max.(gbuf.sum_wtd, GPU_T(1e-30)))
    gbuf.log_s_D_r .= gbuf.V_D .+ @view(gbuf.log_inv_wtd[gbuf.d_time_enc_gpu, :])
    gbuf.row_max_D .= vec(maximum(gbuf.log_s_D_r; dims=2))
    gbuf.s_D_gpu   .= exp.(gbuf.row_max_D) .*
                      (vec(sum(exp.(gbuf.log_s_D_r .- gbuf.row_max_D); dims=2)) .*
                       GPU_T(1.0 / R))

    # Scatter compact shares → full-N model-share vector (stays on device)
    s_model_full[gbuf.b_mask_idx_gpu] .= gbuf.s_B_gpu
    s_model_full[gbuf.d_mask_idx_gpu] .= gbuf.s_D_gpu
    return nothing
end

# ==========================================================================
# GPU BLP Contraction (SQUAREM) — fully on-device
# ==========================================================================

"""
    blp_contraction_gpu!(buf, gbuf, delta, pc, R; tol, max_iter)

SQUAREM BLP fixed-point contraction running **entirely on the GPU**.

`delta` (CPU `Vector{Float64}`) is uploaded **once** at entry and downloaded
**once** at convergence.  `buf.mu` is uploaded to `gbuf.mu_gpu` once.  Every
SQUAREM step — the contraction map `T`, the sup-norm/`‖·‖²` reductions, and the
acceleration step — executes as device broadcasts/reductions, so the H200 stays
busy throughout (no per-step PCIe transfers; only a handful of scalar reductions
sync to the host per step).

Algebraically identical to the CPU `blp_contraction!`:
  T(δ) = clamp(δ + ln s_data − ln s_model(δ), −500, 500)
"""
function blp_contraction_gpu!(buf::HotBuffers, gbuf::GpuBuffers,
                               delta::Vector{Float64},
                               pc::Precomp, R::Int;
                               tol::Float64=1e-12, max_iter::Int=5000)

    N            = length(delta)
    norm_history = Float64[]

    # ── Upload loop-constant data ONCE ────────────────────────────────────
    # mu is fixed during the inner loop (depends only on θ₂); upload once.
    copyto!(gbuf.mu_gpu, GPU_T.(buf.mu))

    # Pre-gather μ for B / D observations once (constant in the loop).
    mu_B = gbuf.mu_gpu[gbuf.b_mask_idx_gpu, :]
    mu_D = gbuf.mu_gpu[gbuf.d_mask_idx_gpu, :]

    # Full-N data log-share vector: ln_s_data_full[i] = ln_s_data_B_cond[i] for
    # B obs, ln_s_data_D[i] for D obs.  Constant across the loop.
    ln_s_data_cpu = Vector{GPU_T}(undef, N)
    @inbounds for i in 1:N
        ln_s_data_cpu[i] = pc.b_mask[i] ? GPU_T(pc.ln_s_data_B_cond[i]) :
                                          GPU_T(pc.ln_s_data_D[i])
    end
    ln_s_data = CuVector{GPU_T}(ln_s_data_cpu)

    # ── Device-resident SQUAREM work vectors (~7 × N × 8 B ≈ 17 MB) ───────
    d_cur  = CuVector{GPU_T}(GPU_T.(delta))
    x1     = CUDA.zeros(GPU_T, N)
    x2     = CUDA.zeros(GPU_T, N)
    r_vec  = CUDA.zeros(GPU_T, N)
    v_vec  = CUDA.zeros(GPU_T, N)
    x_prop = CUDA.zeros(GPU_T, N)
    s_full = CUDA.zeros(GPU_T, N)

    lo = GPU_T(-500f0); hi = GPU_T(500f0)
    floor_s = GPU_T(1e-15)   # Float64 literal — MUST match the CPU `clamp(s, 1e-15, Inf)`
                             # exactly. A Float32 `1f-15` here is 1.0000000363e-15, which
                             # gaps log(s) by ~3.627e-9 at floored shares and pins the
                             # SQUAREM residual above tol_inner (silent non-convergence).

    # Contraction map T(src) → dest, both device vectors
    Tstep!(dest::CuVector{GPU_T}, src::CuVector{GPU_T}) = begin
        model_shares_dev!(gbuf, src, mu_B, mu_D, s_full, R)
        dest .= clamp.(src .+ ln_s_data .- log.(max.(s_full, floor_s)), lo, hi)
        return nothing
    end

    # Finiteness guards are device reductions that sync a scalar to host. The
    # convergence norms (nm, nm2) and SQUAREM norms (norm_r_sq, norm_v_sq,
    # norm_prop) are needed EVERY step, but the isfinite guards on x1/x2 are not:
    # a NaN/Inf makes the norm non-finite (so `< tol` is false → no false
    # convergence), and the periodic check below aborts within FINITE_CHECK_EVERY
    # steps. The x_prop safeguard is folded into the norm_prop test (a non-finite
    # x_prop ⇒ non-finite norm_prop ⇒ fall back to the Picard step x2), removing a
    # separate sync with identical behaviour.
    FINITE_CHECK_EVERY = 10
    fevals = 0
    while fevals + 2 <= max_iter
        Tstep!(x1, d_cur); fevals += 1
        if fevals % FINITE_CHECK_EVERY == 0 && !all(isfinite, x1)
            println("    [SQUAREM-GPU ABORT] non-finite at eval=$fevals")
            copyto!(delta, Float64.(Array(d_cur)))
            return false, fevals, norm_history
        end
        r_vec .= x1 .- d_cur
        nm = Float64(maximum(abs, r_vec))
        push!(norm_history, nm)
        (fevals % 50 == 0 || nm < tol) &&
            println("    [SQUAREM-GPU eval=$fevals/$max_iter] norm=$(round(nm, sigdigits=4))")
        if nm < tol
            copyto!(delta, Float64.(Array(x1))); return true, fevals, norm_history
        end

        Tstep!(x2, x1); fevals += 1
        if fevals % FINITE_CHECK_EVERY == 0 && !all(isfinite, x2)
            copyto!(delta, Float64.(Array(x1))); return false, fevals, norm_history
        end
        nm2 = Float64(maximum(abs, x2 .- x1))
        push!(norm_history, nm2)
        (fevals % 50 == 0 || nm2 < tol) &&
            println("    [SQUAREM-GPU eval=$fevals/$max_iter] norm=$(round(nm2, sigdigits=4))")
        if nm2 < tol
            copyto!(delta, Float64.(Array(x2))); return true, fevals, norm_history
        end

        v_vec .= (x2 .- x1) .- r_vec
        norm_r_sq = Float64(sum(abs2, r_vec))
        norm_v_sq = Float64(sum(abs2, v_vec))
        if norm_v_sq < 1e-28
            d_cur .= x2; continue
        end
        α = GPU_T(-sqrt(norm_r_sq / norm_v_sq))
        x_prop .= clamp.(d_cur .- GPU_T(2f0) * α .* r_vec .+ α^2 .* v_vec, lo, hi)
        # Non-finite x_prop ⇒ norm_prop non-finite ⇒ fall back to Picard step x2
        # (folds the old `all(isfinite, x_prop)` sync into this existing reduction).
        norm_prop = Float64(maximum(abs, x_prop .- x2))
        if !isfinite(norm_prop) || norm_prop > 100.0 * nm2 + 1.0
            d_cur .= x2
        else
            d_cur .= x_prop
        end
    end
    copyto!(delta, Float64.(Array(d_cur)))
    return false, fevals, norm_history
end

# ==========================================================================
# GPU GMM Objective
# ==========================================================================

"""
    gmm_objective_gpu!(buf, gbuf, theta2, prod_vec, nu_draws,
                       sigma_indices, pi_interactions, R, coef_dim,
                       W, tol_inner, max_inner, delta, pc) → Float64

Drop-in GPU replacement for `gmm_objective!`.  `compute_mu!` runs on CPU
(MKL GEMM, ~0 ms for small k-dim), then `blp_contraction_gpu!` handles
the inner fixed-point entirely on device.
"""
function gmm_objective_gpu!(buf::HotBuffers, gbuf::GpuBuffers,
                             theta2::Vector{Float64},
                             prod_vec, nu_draws,
                             sigma_indices, pi_interactions,
                             R::Int, coef_dim::Int,
                             W::Matrix{Float64},
                             tol_inner::Float64, max_inner::Int,
                             delta::Vector{Float64},
                             pc::Precomp)::Float64
    sigma_vals, pi_vals = unpack_theta2(theta2, sigma_indices, pi_interactions)
    compute_mu!(buf, prod_vec, nu_draws, sigma_vals, sigma_indices, pi_vals, R, coef_dim)
    converged, n_iter, _ = blp_contraction_gpu!(buf, gbuf, delta, pc, R;
                                                 tol=tol_inner, max_iter=max_inner)
    converged || println("  [!] Inner loop (GPU) did not converge in $n_iter iterations")
    theta1, xi = estimate_theta1(delta, pc)
    G          = compute_gmm_moments(xi, pc)
    return dot(G, W * G)
end

# ==========================================================================
# GPU Estimation Runner
# ==========================================================================

"""
    run_blp_estimation_gpu(estim, spec_id, args, nu_draws, draws_3d, key_index)

GPU-accelerated version of `run_blp_estimation`.

For the **logit** stage there is no inner contraction loop, so this
function delegates directly to the CPU `run_blp_estimation`.

For **sigma / full / extended** stages, all inner-loop share computations
run on the GPU via `compute_model_shares_gpu!`; MKL handles `compute_mu!`
on CPU (small k-dim matrix multiply); SQUAREM delta updates and outer
L-BFGS-B optimisation run on CPU as before.
"""
function run_blp_estimation_gpu(estim::Int, spec_id::Int, args,
                                 nu_draws::Matrix{Float64},
                                 draws_3d::Array{Float64,3},
                                 key_index::Dict{Tuple{String,String},Int})

    println("\n" * "=" ^ 60)
    println("  Estimation $estim — Specification $spec_id  [GPU]")
    println("=" ^ 60)

    # Logit: no inner contraction → CPU path is fine
    if args["stage"] == "logit"
        return run_blp_estimation(estim, spec_id, args, nu_draws, draws_3d, key_index)
    end

    input_dir, _, _ = get_paths(args["hpc"]; local_dir=args["local_dir"])
    fname = input_filename(estim, spec_id)
    path  = joinpath(input_dir, fname)
    isfile(path) || (println("  [!] Missing: $path"); return nothing)

    df = DataFrame(Parquet2.Dataset(path); copycols=true)
    nrow(df) == 0 && (println("  [!] Empty dataframe."); return nothing)
    println("  Loaded: $(nrow(df)) observations")

    sigma_indices, pi_interactions, n_params = build_theta2_structure(args["stage"])
    R        = args["R"]
    seed     = args["seed"]
    coef_dim = 1 + L_PROD

    results = Dict{String,Any}("spec_id" => spec_id, "estim" => estim,
                                "stage" => args["stage"])
    println("  Stage: $(uppercase(args["stage"])) ($n_params parameters)")

    # ── Map observations to pre-computed draw indices ─────────────────────
    N_obs     = nrow(df)
    mca_codes = string.(df.mca_code)
    time_ids  = string.(df.time_id)
    pad_idx   = size(draws_3d, 1)

    obs_key_idx = [get(key_index, (mca_codes[i], time_ids[i]), pad_idx)
                   for i in 1:N_obs]
    n_mapped = count(i -> i != pad_idx, obs_key_idx)
    log_status("  Obs mapped to draws: $n_mapped / $N_obs ($(round(100*n_mapped/N_obs, digits=1))%)")

    # ── Build product characteristics and IV matrices ─────────────────────
    prod_vec  = zeros(N_obs, coef_dim)
    spreads   = coalesce.(df.spread_ann, 0.0) ./ 100.0  # bps → percentage points (÷100)
    dep_types = Int.(coalesce.(df.deposit_type, 0))
    prod_vec[:, 1] .= spreads
    for (i, col) in enumerate(X_COLS)
        col in names(df) && (prod_vec[:, 1+i] .= coalesce.(df[!, col], 0.0))
    end

    spread_cols, x_mat, Z_mat, iv_cols = build_regressor_matrices(df)
    iv_avail = [c for c in iv_cols
                if std(replace(coalesce.(df[!, c], 0.0), Inf=>0.0, -Inf=>0.0)) > 1e-10]
    Z = zeros(N_obs, length(iv_avail))
    for (i, c) in enumerate(iv_avail)
        v = coalesce.(df[!, c], 0.0); replace!(v, Inf=>0.0, -Inf=>0.0)
        Z[:, i] .= v
    end
    W = try inv(Z' * Z ./ N_obs) catch; Matrix(I(length(iv_avail)) * 1.0) end

    H          = hcat(x_mat, Z_mat)
    spread_hat = project_endogenous_spreads(spread_cols, H, dep_types)
    X_full     = hcat(spread_cols, x_mat)
    X_hat      = hcat(spread_hat,  x_mat)
    valid      = BitVector(all.(isfinite, eachrow(X_hat)))
    clusters   = string.(df.CodConglomeradoPrudencial)
    pc         = build_precomp(df, Z, X_full, X_hat, valid, clusters)

    N_B     = sum(pc.b_mask)
    N_D     = sum(pc.d_mask)
    n_pairs = length(pc.unique_pairs)
    n_times = length(pc.unique_times)

    # ── Design-rank guard (aborts on degenerate/collinear X_hat column) ───
    check_design_rank(pc)

    # ── CPU hot buffers + GPU mirror ──────────────────────────────────────
    buf  = allocate_hot_buffers(N_obs, N_B, N_D, R, n_pairs, n_times, coef_dim,
                                 length(pi_interactions))
    precompute_pi_products!(buf, prod_vec, draws_3d, obs_key_idx,
                             pi_interactions, coef_dim)
    log_status("  [GPU] Device: $(CUDA.name(CUDA.device()))")
    log_status("  [GPU] Allocating GPU buffers (~$(round(21.68, digits=1)) GB)...")
    flush(stdout); flush(stderr)
    gbuf = allocate_gpu_buffers(buf, pc, N_obs, N_B, N_D, R, n_pairs, n_times)
    log_status("  [GPU] ✓ GPU buffers allocated")

    # ── θ₂ warm-start ─────────────────────────────────────────────────────
    _, _, out_dir = get_paths(args["hpc"])
    prev_stages  = Dict("rc2" => "sigma", "rc3" => "rc2", "rc4" => "rc3",
                        "full" => "rc4", "ext1" => "full", "ext2" => "ext1",
                        "extended" => "ext2")
    theta2_0     = nothing
    if args["stage"] in keys(prev_stages)
        prev_path = joinpath(out_dir,
            "blp_checkpoint_E$(estim)_spec_$(spec_id)_$(prev_stages[args["stage"]])$(output_suffix()).jls")
        if isfile(prev_path)
            try
                prev = deserialize(prev_path)
                t2p  = get(prev, "theta2_star", nothing)
                if t2p !== nothing && length(t2p) <= n_params
                    theta2_0 = zeros(n_params)
                    theta2_0[1:length(t2p)] .= t2p
                    println("  [WARM-START] theta2_0 from $(basename(prev_path))")
                end
            catch e
                println("  [WARM-START] Could not load checkpoint: $e")
            end
        end
    end
    theta2_0 === nothing &&
        (theta2_0 = randn(MersenneTwister(seed), n_params) .* 0.01)

    lo = fill(-5.0, n_params)
    hi = fill( 5.0, n_params)
    # σ parameters are standard deviations → bound ≥ 0 (same fix as the IFT engine):
    # removes the ±σ sign degeneracy that makes the outer optimiser oscillate without
    # converging. σ are the first length(sigma_indices) entries of θ₂.
    n_sigma = length(sigma_indices)
    lo[1:n_sigma] .= 0.0
    theta2_0 .= clamp.(theta2_0, lo, hi)   # a warm-start θ₂ may carry a negative σ

    println("  Outer minimisation: $(args["method"])")
    println("  Inner tolerance: $(args["tol_inner"])")

    # ── δ warm-start ──────────────────────────────────────────────────────
    delta_work = zeros(N_obs)
    let _loaded = false
        # BLP_DELTA_SUFFIX lets a coherence run warm-start from the suffixed delta
        # (logit_delta_E{k}_spec_{s}_coherence.*); default "" keeps the legacy name.
        _suffix = get(ENV, "BLP_DELTA_SUFFIX", "")
        for _bin_cand in unique([
                joinpath(out_dir, "logit_delta_E$(estim)_spec_$(spec_id)$(_suffix).bin"),
                joinpath(out_dir, "logit_delta_E$(estim)_spec_$(spec_id).bin"),
            ])
            isfile(_bin_cand) || continue
            _d = load_delta_bin(_bin_cand)
            if _d !== nothing && length(_d) == N_obs
                copyto!(delta_work, _d)
                log_status("  [δ WARM-START] Loaded $(basename(_bin_cand)) [binary]")
                _loaded = true
                break
            end
        end
        if !_loaded
            for _cand in unique([
                joinpath(out_dir, "logit_delta_E$(estim)_spec_$(spec_id)$(_suffix).jls"),
                joinpath(out_dir, "logit_delta_E$(estim)_spec_$(spec_id).jls"),
                joinpath(out_dir, "blp_checkpoint_E$(estim)_spec_$(spec_id)_logit.jls"),
                joinpath(out_dir, "blp_results_E$(estim)_spec_$(spec_id)_logit.jls"),
            ])
                isfile(_cand) || continue
                try
                    _ck = deserialize(_cand)
                    _d  = get(_ck, "delta", nothing)
                    if _d !== nothing && length(_d) == N_obs
                        copyto!(delta_work, _d)
                        log_status("  [δ WARM-START] Loaded $(basename(_cand))")
                        _loaded = true; break
                    end
                catch _e
                    log_status("  [δ WARM-START] Skipped $(basename(_cand)): $(_e)")
                end
            end
        end
        if !_loaded
            delta_work[pc.d_mask] .= pc.ln_s_data_D[pc.d_mask]
            delta_work[pc.b_mask] .= pc.ln_s_data_B_cond[pc.b_mask]
            log_status("  [δ WARM-START] No logit checkpoint — log-share init")
        end
    end

    # ── GPU↔CPU share consistency guard (catches precision regressions) ───
    let (sv, pv) = unpack_theta2(theta2_0, sigma_indices, pi_interactions)
        compute_mu!(buf, prod_vec, nu_draws, sv, sigma_indices, pv, R, coef_dim)
        verify_gpu_shares(buf, gbuf, delta_work, pc, R; rtol=args["tol_inner"])
    end

    # ── Dry-run timing (GPU) ──────────────────────────────────────────────
    if get(args, "dry_run", false)
        println("  [DRY RUN GPU] 10 contraction iterations for timing...")
        sv, pv = unpack_theta2(theta2_0, sigma_indices, pi_interactions)
        compute_mu!(buf, prod_vec, nu_draws, sv, sigma_indices, pv, R, coef_dim)
        CUDA.synchronize()
        t0 = time()
        blp_contraction_gpu!(buf, gbuf, copy(delta_work), pc, R;
                              tol=args["tol_inner"], max_iter=10)
        CUDA.synchronize()
        elapsed = time() - t0
        println("  [DRY RUN GPU] 10 iters: $(round(elapsed, digits=2))s " *
                "($(round(elapsed/10, digits=3)) s/iter)")
        return Dict("dry_run" => true)
    end

    # ── Outer optimisation (true L-BFGS-B + cached forward-difference gradient) ──
    # The numerical engine has no analytical gradient, so L-BFGS-B is fed a forward-
    # difference ∇Q: n_params extra share solves per gradient — the same cost as the
    # finite-difference gradient Optim used before. A one-point cache stops the base
    # objective from being re-solved between the f and g! calls at the same θ₂.
    outer_iter = Ref(0)
    _raw_obj(t2) = gmm_objective_gpu!(buf, gbuf, t2, prod_vec, nu_draws,
                                       sigma_indices, pi_interactions, R, coef_dim,
                                       W, args["tol_inner"], args["max_inner"],
                                       delta_work, pc)
    fd_t2 = fill(NaN, n_params); fd_Q = Ref(0.0)
    function obj_fn(t2)
        if !isequal(t2, fd_t2)
            fd_Q[] = _raw_obj(t2); copyto!(fd_t2, t2)
            outer_iter[] += 1
            outer_iter[] % 10 == 0 &&
                log_status("  [OUTER iter=$(outer_iter[])] Q=$(round(fd_Q[], sigdigits=6)) " *
                           "theta2=$(round.(t2, digits=4))")
        end
        return fd_Q[]
    end
    function grad_fn!(G, t2)
        f0 = obj_fn(t2)                          # base value (cached → 1 solve)
        @inbounds for k in 1:n_params
            h    = 1e-6 * max(1.0, abs(t2[k]))   # relative forward-difference step
            tp   = copy(t2); tp[k] += h
            G[k] = (_raw_obj(tp) - f0) / h
        end
        return G
    end

    theta2_star, Q_min, n_outer, conv_outer = solve_box_lbfgsb(
        obj_fn, grad_fn!, theta2_0, lo, hi;
        tol_outer = args["tol_outer"], maxiter = 500)

    println("  Optimiser converged: $(conv_outer)")
    println("  Q(theta2*) = $(round(Q_min, sigdigits=6))")
    println("  theta2* = $theta2_star")

    # ── Final contraction at optimum (GPU) ───────────────────────────────
    sv, pv = unpack_theta2(theta2_star, sigma_indices, pi_interactions)
    compute_mu!(buf, prod_vec, nu_draws, sv, sigma_indices, pv, R, coef_dim)
    delta_final = copy(delta_work)
    conv, n_it, _ = blp_contraction_gpu!(buf, gbuf, delta_final, pc, R;
                                          tol=args["tol_inner"],
                                          max_iter=args["max_inner"])

    theta1_star, xi_star      = estimate_theta1(delta_final, pc)
    theta1_se, n_cl, G_s      = compute_cluster_se(theta1_star, delta_final, pc)

    results["theta1"]             = theta1_star
    results["theta1_se"]          = theta1_se
    results["theta2"]             = theta2_star
    results["theta2_se"]          = fill(NaN, length(theta2_star))
    results["delta"]              = delta_final
    results["xi"]                 = xi_star
    results["Q_value"]            = Q_min
    results["converged"]          = conv_outer
    results["n_outer_iter"]       = n_outer
    results["param_names_theta1"] = vcat(["alpha"], X_COLS)
    results["sigma_indices"]      = sigma_indices
    results["pi_interactions"]    = pi_interactions
    results["n_clusters"]         = n_cl
    results["G_star"]             = G_s
    println("  theta1 (alpha): $(round(theta1_star[1], sigdigits=6))")

    # ── Save checkpoint ───────────────────────────────────────────────────
    mkpath(out_dir)
    chk_path = joinpath(out_dir,
        "blp_checkpoint_E$(estim)_spec_$(spec_id)_$(args["stage"])$(output_suffix()).jls")
    try
        serialize(chk_path, Dict("theta2_star" => theta2_star,
                                  "delta_star"  => delta_final))
        println("  [CHECKPOINT] Saved $(basename(chk_path))")
    catch e
        println("  [CHECKPOINT-WARN] $e")
    end
    chk_bin_path = replace(chk_path, ".jls" => ".bin")
    save_delta_bin(chk_bin_path, delta_final)
    println("  [CHECKPOINT BIN] Saved $(basename(chk_bin_path))")

    return results
end

# ==========================================================================
# CLI + Main (GPU)
# ==========================================================================

"""
    main_gpu()

Entry point for the GPU estimation script.  Parses the same CLI flags as
`main()` in blp_estimation.jl; processes specs **sequentially** (one GPU,
one spec at a time) so device memory is not over-subscribed.

Identical JSON / JLS output layout to the CPU path — downstream export
scripts work unchanged.
"""
function main_gpu()
    # ── GPU availability check ───────────────────────────────────────────
    log_status("[GPU] Checking CUDA functionality...")
    flush(stdout); flush(stderr)
    if !CUDA.functional()
        log_status("[GPU] ERROR: CUDA.functional() = false")
        flush(stdout); flush(stderr)
        error("[GPU] CUDA is not functional on this node.  " *
              "Use blp_estimation.jl for CPU-only estimation.")
    end
    dev = CUDA.device()
    log_status("[GPU] ✓ CUDA functional")
    log_status("[GPU] Using device: $(CUDA.name(dev)) " *
               "($(round(CUDA.totalmem(dev)/2^30, digits=1)) GB VRAM)")
    flush(stdout); flush(stderr)

    args     = parse_args_est()    # reuses CPU parser from blp_estimation.jl
    estim    = args["estim"]
    spec_ids = args["spec"] == "all" ? collect(1:12) : [parse(Int, args["spec"])]

    log_status("BLP Estimation GPU — START")
    log_status("  E$estim | Stage: $(args["stage"]) | Specs: $spec_ids")
    log_status("  R=$(args["R"]) | seed=$(args["seed"]) | HPC=$(args["hpc"])")
    log_status("  tol_inner=$(args["tol_inner"]) | tol_outer=$(args["tol_outer"]) " *
               "| threads=$(Threads.nthreads())")

    _, draws_dir, out_dir = get_paths(args["hpc"]; local_dir=args["local_dir"])
    mkpath(out_dir)

    nu_draws, draws_3d, key_index = load_precomputed_draws(
        draws_dir, args["R"], args["seed"])

    # "sequence" → all 8 stages; a comma list (e.g. "sigma,rc2,rc3,rc4,full,ext1")
    # → that SUBSET in one process (option-6 grouping: amortise Julia/CUDA/parquet
    # startup over several stages, each warm-starting the next from its on-disk
    # checkpoint); a single name → just that stage.
    stages_to_run = if args["stage"] == "sequence"
        ["sigma", "rc2", "rc3", "rc4", "full", "ext1", "ext2", "extended"]
    elseif occursin(',', args["stage"])
        String.(strip.(split(args["stage"], ',')))
    else
        [args["stage"]]
    end

    for current_stage in stages_to_run
        args["stage"] = current_stage

        all_done = all(isfile(joinpath(out_dir,
            "blp_results_E$(estim)_spec_$(sp)_$(current_stage)$(output_suffix()).jls"))
            for sp in spec_ids)
        if all_done
            log_status("[SKIP] Stage '$current_stage' already complete.")
            continue
        end

        println("\n  Starting stage '$current_stage' — $(length(spec_ids)) spec(s) " *
                "(sequential GPU execution)...")

        all_results = Dict{Int,Any}()

        # GPU: sequential spec loop — avoids multi-context VRAM contention.
        for sp in spec_ids
            out_path = joinpath(out_dir,
                "blp_results_E$(estim)_spec_$(sp)_$(current_stage)$(output_suffix()).jls")
            if isfile(out_path)
                log_status("  [SKIP] Spec $sp already done for '$current_stage'")
                continue
            end

            res = try
                run_blp_estimation_gpu(estim, sp, args, nu_draws, draws_3d, key_index)
            catch e
                println("  [!] Spec $sp failed: $e"); nothing
            end
            res === nothing && continue
            get(res, "dry_run", false) && continue

            serialize(out_path, res)
            println("  Saved: $(basename(out_path))")

            json_path = replace(out_path, ".jls" => ".json")
            try
                json_out = Dict{String,Any}(
                    "estim"              => get(res, "estim",   estim),
                    "spec_id"            => get(res, "spec_id", sp),
                    "stage"              => get(res, "stage",   current_stage),
                    "theta1"             => round.(get(res, "theta1",    Float64[]), sigdigits=8),
                    "theta1_se"          => round.(get(res, "theta1_se", Float64[]), sigdigits=8),
                    "theta2"             => round.(get(res, "theta2",    Float64[]), sigdigits=8),
                    "theta2_se"          => round.(get(res, "theta2_se", Float64[]), sigdigits=8),
                    "Q_value"            => get(res, "Q_value",   0.0),
                    "converged"          => get(res, "converged", false),
                    "param_names_theta1" => get(res, "param_names_theta1", String[]),
                    "sigma_indices"      => get(res, "sigma_indices",     Int[]),
                    "pi_interactions"    => get(res, "pi_interactions",   []),
                    "n_clusters"         => get(res, "n_clusters",        0),
                    "G_star"             => get(res, "G_star",            0.0),
                )
                open(json_path, "w") do f; JSON3.write(f, json_out); end
            catch e
                println("  [JSON-WARN] $e")
            end

            all_results[sp] = Dict(
                "Q_value"      => get(res, "Q_value", 0.0),
                "converged"    => get(res, "converged", true),
                "theta1_alpha" => isempty(get(res, "theta1", Float64[])) ?
                                  [] : [res["theta1"][1]],
                "theta2"       => get(res, "theta2", Float64[]),
                "stage"        => current_stage)
        end

        summary_path = joinpath(out_dir, "blp_summary_E$(estim)_$(current_stage)_gpu$(output_suffix()).json")
        open(summary_path, "w") do f; JSON3.write(f, all_results); end
        log_status("Summary saved to: $summary_path")
        log_status("[DONE] BLP GPU E$estim ($current_stage) complete for specs $spec_ids.")
    end
end

# Guard: run main_gpu only when this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main_gpu()
end


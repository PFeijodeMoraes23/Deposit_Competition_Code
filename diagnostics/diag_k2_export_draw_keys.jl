# diag_k2_export_draw_keys.jl
# Step 7 of diag_k2_nls_simulation.py reads the paper's demographic draws (BLP_DRAWS/
# demo_draws_R2000_seed42.jls, a 14 GB Array{Float64,3}) by memory map from Python, which needs the
# (mca_code, time_id) -> row map and the Halton nu draws, both Julia-serialized. This writes them as
# CSV (read only; nothing in BLP_DRAWS changes).
#   Rout/k2_rc_draw_keys.csv   mca_code,time_id,idx   (idx 1-based; row 1 of the array is padding)
#   Rout/k2_rc_halton_nu.csv   2000 rows x 9 columns  (the nu draws, columns = spread + X_COLS)
# Usage: julia --startup-file=no diagnostics/diag_k2_export_draw_keys.jl <BLP_DRAWS dir> <Rout dir>
using Serialization
dir, out = ARGS[1], ARGS[2]
ki = deserialize(joinpath(dir, "demo_key_index_R2000_seed42.jls"))::Dict{Tuple{String,String},Int}
open(joinpath(out, "k2_rc_draw_keys.csv"), "w") do io
    println(io, "mca_code,time_id,idx")
    for ((m, t), i) in sort(collect(ki); by = last)
        println(io, m, ",", t, ",", i)
    end
end
nu = deserialize(joinpath(dir, "halton_nu_R2000_seed42.jls"))::Matrix{Float64}
open(joinpath(out, "k2_rc_halton_nu.csv"), "w") do io
    for r in 1:size(nu, 1)
        println(io, join(string.(nu[r, :]), ","))
    end
end
println("keys ", length(ki), "; nu ", size(nu))

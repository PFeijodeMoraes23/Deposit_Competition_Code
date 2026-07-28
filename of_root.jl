# of_root.jl — validated Open-Finance root resolution for the Julia entry points.
#
# Mirrors the anchor logic of utils/paths.py (OPEN_FINANCE_ROOT override, else
# three levels up from this file).  The validation exists because a June 2026 run
# resolved the root one level short — at .../Open-Finance/Code — and mkpath'd a
# full BCB/Egan_et_al_2025_Rep/processed/... tree there, quietly accumulating
# 8.9 GB of draws nobody read.  A wrong root must fail loudly, not create dirs.

"""
    resolve_of_root() -> String

The Open-Finance data root, validated by basename and by sentinel directories.
Throws rather than returning a root that would be silently wrong.
"""
function resolve_of_root()::String
    root = get(ENV, "OPEN_FINANCE_ROOT",
               dirname(dirname(dirname(abspath(@__FILE__)))))

    basename(root) == "Open-Finance" || error(
        "Refusing to run: resolved root '$root' has basename " *
        "'$(basename(root))', expected 'Open-Finance'. " *
        "Set OPEN_FINANCE_ROOT to override.")

    (isdir(joinpath(root, "shared")) &&
     isdir(joinpath(root, "BCB", "Egan_et_al_2025_Rep", "processed"))) || error(
        "Refusing to run: '$root' lacks the sentinel dirs shared/ and " *
        "BCB/Egan_et_al_2025_Rep/processed/. Not creating them — a missing " *
        "sentinel means the root is wrong, not that the tree needs building.")

    return root
end

#!/bin/bash
# julia_sysimage_env.sh — sourced by the BLP submit scripts to set up the Julia run.
#
# If blp_sysimage.so exists (built once by submit_build_sysimage.sh) it is used via
# --sysimage and the expensive Pkg.precompile is SKIPPED, so the job starts in seconds.
# Otherwise it falls back to resolve/instantiate/precompile. Verifies CUDA loads; if the
# sysimage can't load CUDA it falls back to no-sysimage automatically.
#
# Requires PROJECT_DIR to be set. Exposes the array JULIA_SYS to the caller — splice it
# into every julia invocation as:  julia --project="${PROJECT_DIR}" ${JULIA_SYS[@]+"${JULIA_SYS[@]}"} ...
SYSIMAGE="${PROJECT_DIR}/blp_sysimage.so"
if [ -f "${SYSIMAGE}" ]; then
    JULIA_SYS=(--sysimage "${SYSIMAGE}")
    echo "Using sysimage ${SYSIMAGE} (skipping precompile): $(date)"
    if ! julia --project="${PROJECT_DIR}" "${JULIA_SYS[@]}" -e 'using CUDA'; then
        echo "[!] Sysimage failed to load CUDA — falling back to no sysimage."
        JULIA_SYS=()
        julia --project="${PROJECT_DIR}" -e 'using Pkg; Pkg.instantiate(); using CUDA'
    fi
else
    JULIA_SYS=()
    echo "No sysimage — resolving/precompiling Julia packages: $(date)"
    julia --project="${PROJECT_DIR}" -e '
        using Pkg
        Pkg.resolve()
        Pkg.instantiate()
        Pkg.precompile()
        using CUDA
    '
fi

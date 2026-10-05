#!/bin/bash
# ==============================================================================
# dx_bbl_run.sh -- the BBL launcher of the `dx` demand variant, WITHOUT a change to bbl_run.sh.
#
# bbl_run.sh builds the forward simulation's command line and has no flag for the output suffix
# of the demand fit, so on its own it simulates on blp_results_E{k}_spec_12_{stage}.jls, the
# main specification's. The variant needs `--suffix _dx` on that command line and nothing else.
#
# This script does not copy bbl_run.sh. At run time it DERIVES the launcher from it:
#   1. bbl_run.sh must be the file this was tested against (md5 pinned below); otherwise refuse.
#   2. Three lines are transformed, each matched as a WHOLE line that must occur exactly once:
#        - after the line that reads PSI_TAG from the environment, one line is added:
#              DEMAND_SUFFIX="${DEMAND_SUFFIX:-}"
#        - in the two lines that give the simulation its `--psi-tag`, the tag becomes
#              ${DEMAND_SUFFIX:+--suffix ${DEMAND_SUFFIX} }--psi-tag ${PSI_TAG#"${DEMAND_SUFFIX}"}
#      so with --psi-tag _dx_ms982 the simulation receives `--suffix _dx --psi-tag _ms982`: it
#      reads blp_results_..._extended_dx.jls and writes psi_*_extended_dx_ms982*, the names the
#      sweep, the solve (--psi-tag _dx_ms982) and the slim archive look for.
#   3. The result must have the md5 pinned below (the file that was dry-run and stub-tested
#      locally). It is written to a temporary folder (mktemp under TMPDIR), never into this
#      folder, together with a one-line cluster_lib.sh that sources the real one: the derived
#      launcher sources `cluster_lib.sh` beside itself, and CL_ROOT is exported as this folder so
#      every path it builds (bbl_job.sh, logs, the sysimage) is the scripts folder's.
#   4. It is executed with DEMAND_SUFFIX=_dx and this script's arguments.
#   5. When this script exits, the two files it wrote and their folder are removed (nothing it
#      submitted refers to them: the jobs run bbl_job.sh of the scripts folder). DX_BBL_KEEP=1
#      keeps the folder, to read the derived launcher.
# Every other line of the launcher, and everything it calls (cluster_lib.sh, bbl_job.sh, the sweep,
# the solve), is the main run's code. A --repair of the variant's tag needs neither this script
# nor the suffix: bbl_run.sh --repair takes the simulation's flags from the run's context.
#
# It refuses:
#   - a launch without an explicit --routines (bbl_run.sh's default set is not the variant's);
#   - a --psi-tag that does not start with _dx_ (the tag is what separates the variant's psi and
#     cost parameters from the main run's);
#   - a live launch for which blp_results_E{k}_spec_12_{stage}_dx.jls is missing for ANY routine
#     it was given: the simulation does not stop on a missing fit, it warns and runs on
#     placeholder parameters (cf_demand_eval.jl).
#
# WHO RUNS IT. dx_suite_20261001.sh, from its hand-off, and nobody else for a live launch: the
# suite first records the sha256 of the demand fit the launch is simulated on, and afterwards
# refuses any file of a BBL run that has no such record. A live launch made by hand with this
# script writes no record, so the suite would refuse everything it leaves (psi files, launch
# context, cost parameters). By hand it is for a DRY RUN only (it submits nothing), with the flags
# of bbl_run.sh:
#   bash dx_bbl_run.sh --routines "3 4" --multi-start --n-paths 1 --psi-tag _dx_ms982 --no-polfunc --no-warmup --no-tables --no-zip --dry-run
# ==============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HERE}" || exit 2

DX="_dx"
SRC="${HERE}/bbl_run.sh"
SRC_MD5="33c97ed1e56cdfd5368fae0187e9fb4f"     # bbl_run.sh as shipped 2026-09-30
GEN_MD5="e372f2e8be439baf7e440a699021e207"     # the derived launcher, as tested

die () { printf 'dx_bbl_run.sh REFUSED: %s\n' "$*" >&2; exit 2; }
md5 () { md5sum "$1" | cut -d' ' -f1; }

case " $* " in *" -h "*|*" --help "*) sed -n '2,/^# =\{20,\}$/p' "${BASH_SOURCE[0]}"; exit 0 ;; esac

[[ -f "${SRC}" ]] || die "${SRC} is missing"
got="$(md5 "${SRC}")"
[[ "${got}" == "${SRC_MD5}" ]] || die "bbl_run.sh has md5 ${got}, not the ${SRC_MD5} this launcher was derived from and tested with. Nothing was submitted."

# The psi tag and the routines, read from the arguments to refuse a launch that is not the variant's.
tag=""; routines=""; stage="${CF_STAGE:-extended}"; prev=""
for a in "$@"; do
    [[ "${prev}" == "--psi-tag" ]] && tag="${a}"
    [[ "${prev}" == "--routines" ]] && routines="${a}"
    prev="${a}"
done
[[ -n "${routines}" ]] || die "give --routines explicitly (for example --routines \"3 4\"): without it bbl_run.sh takes its own default set, and the variant's fit exists only for the routines its ladder ran."
[[ "${tag}" == "${DX}"_* ]] || die "--psi-tag '${tag}' must start with '${DX}_' (for example ${DX}_ms982): the tag is what separates the variant's psi and cost parameters from the main run's."
if [[ "${CL_DRYRUN:-0}" != "1" && " $* " != *" --dry-run "* && " $* " != *" --repair "* ]]; then
    blp="${HERE}/../data/output/blp"; [[ -n "${DATA_ROOT:-}" ]] && blp="${DATA_ROOT}/output/blp"
    for k in ${routines}; do
        f="${blp}/blp_results_E${k}_spec_12_${stage}${DX}.jls"
        [[ -f "${f}" ]] || die "the variant's demand fit ${f} is missing (run the variant's RC ladder for E${k} first). The simulation would not stop on it: it would run on placeholder parameters."
    done
fi

# The three whole lines of bbl_run.sh, and what they become.
export DXL1='MULTI_START="${MULTI_START:-0}"; N_PATHS="${N_PATHS:-8}"; PSI_TAG="${PSI_TAG:-}"'
export DXA1='DEMAND_SUFFIX="${DEMAND_SUFFIX:-}"   # output suffix of the demand fit the sim reads; the psi tag starts with it'
export DXL2='    ms_flags="${ms_flags} --n-paths ${N_PATHS} --psi-tag ${PSI_TAG}"'
export DXR2='    ms_flags="${ms_flags} --n-paths ${N_PATHS} ${DEMAND_SUFFIX:+--suffix ${DEMAND_SUFFIX} }--psi-tag ${PSI_TAG#"${DEMAND_SUFFIX}"}"'
export DXL3='    ms_flags="--psi-tag ${PSI_TAG}"'
export DXR3='    ms_flags="${DEMAND_SUFFIX:+--suffix ${DEMAND_SUFFIX} }--psi-tag ${PSI_TAG#"${DEMAND_SUFFIX}"}"'
gen="$(awk '
    $0 == ENVIRON["DXL1"] { print; print ENVIRON["DXA1"]; n1++; next }
    $0 == ENVIRON["DXL2"] { print ENVIRON["DXR2"]; n2++; next }
    $0 == ENVIRON["DXL3"] { print ENVIRON["DXR3"]; n3++; next }
    { print }
    END { if (n1 != 1 || n2 != 1 || n3 != 1) { printf "matched %d/%d/%d times, expected 1/1/1\n", n1, n2, n3 > "/dev/stderr"; exit 3 } }
' "${SRC}")" || die "bbl_run.sh does not contain the three lines exactly once each (above)."
unset DXL1 DXA1 DXL2 DXR2 DXL3 DXR3
got="$(printf '%s\n' "${gen}" | md5sum | cut -d' ' -f1)"
[[ "${got}" == "${GEN_MD5}" ]] || die "the derived launcher has md5 ${got}, not the tested ${GEN_MD5}."

# Written only once verified, and outside the scripts folder. On exit the two files written here
# are removed, then the folder if it is empty: nothing else is touched.
tmpd="$(mktemp -d "${TMPDIR:-/tmp}/dx_bbl_run.XXXXXX")" || die "cannot create a temporary folder under ${TMPDIR:-/tmp}"
GEN="${tmpd}/bbl_run_dx.sh"
cleanup () {
    [[ "${DX_BBL_KEEP:-0}" == "1" ]] && return 0
    rm -f -- "${GEN}" "${tmpd}/cluster_lib.sh"
    rmdir -- "${tmpd}" 2>/dev/null
    return 0
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
printf '%s\n' "${gen}" > "${GEN}" || die "cannot write ${GEN}"
[[ "$(md5 "${GEN}")" == "${GEN_MD5}" ]] || die "${GEN} was not written whole (md5 $(md5 "${GEN}"))."
printf '. "%s/cluster_lib.sh"\n' "${HERE}" > "${tmpd}/cluster_lib.sh" || die "cannot write ${tmpd}/cluster_lib.sh"
echo "dx_bbl_run.sh: derived launcher ${GEN} (md5 ${GEN_MD5}); scripts folder ${HERE}"

export CL_ROOT="${HERE}"
DEMAND_SUFFIX="${DX}" bash "${GEN}" "$@"
exit $?

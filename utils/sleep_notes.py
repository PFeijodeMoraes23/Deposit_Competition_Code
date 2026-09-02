"""utils/sleep_notes.py

The note text shared by the sleepiness SECOND-STAGE tables: the four-routine spec-12
comparison and the per-routine appendix tables written by sleep_export_e1 /
sleep_export_e2 / sleep_export_link.

Defined once here so the wording cannot drift between the comparison table and the
appendix tables that report the same estimates. Each caller keeps its own size/wrapper
command and appends its own significance-level line; this module returns the body only.
"""

SECOND_STAGE_NOTE = (
    r"Columns \ref{estimation:local} and \ref{estimation:pooled} exhibit standard errors "
    r"(WCB at the conglomerate level, except on the rows marked "
    r"$\dagger$ below; \textcite{cameron2008bootstrap}, \textcite{mackinnon2017wild}) in "
    r"parentheses. Columns \ref{estimation:single_idx} and "
    r"\ref{estimation:single_idx_time} display \textcite{efron1987better}'s "
    r"bias-corrected percentile interval instead. Columns index the estimation strategies "
    r"enumerated in Section~\ref{sec:empirical:sleep}; the linear strategies report "
    r"coefficients and the single-index strategies report average marginal effects (AME), "
    r"in percentage points of the sleepy share per the unit given in the row label, with "
    r"shares and rates in percentage points and Pix Available a discrete $0\to1$ "
    r"difference. $t$-statistics and stars are invariant to these units. State variables "
    r"are grand-mean centered. Rows marked $\dagger$ are national regressors: Pix "
    r"Available and the lagged Selic rate take a common value across all conglomerates "
    r"within a quarter, so clustering on the conglomerate treats each firm's copy of a "
    r"national value as independent evidence and understates their sampling uncertainty. "
)


def second_stage_note() -> str:
    """-> the shared second-stage note body, without a leading `Notes:` label."""
    return SECOND_STAGE_NOTE

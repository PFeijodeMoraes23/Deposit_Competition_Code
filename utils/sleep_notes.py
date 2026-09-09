"""utils/sleep_notes.py

The note text shared by the sleepiness SECOND-STAGE tables: the four-routine spec-12
comparison and the per-routine appendix tables written by sleep_export_e1 /
sleep_export_e2 / sleep_export_link.

Defined once here so the wording cannot drift between the comparison table and the
appendix tables that report the same estimates. Each caller keeps its own size/wrapper
command and appends its own significance-level line; this module returns the body only.

The opening sentence is chosen by what the table ACTUALLY printed, not by what it was meant
to print. A band is a cluster artifact (sleep_ame_twostage.py for the single-index routines,
sleep_wcb_band.py for the linear ones); when one has not landed, the cell falls back to a
standard error, and a note that still promised an interval would be describing a calculation
that did not run. `second_stage_note(bands=...)` is how the caller says which happened.
"""

_OPEN_INTERVAL = (
    r"All columns report \textcite{efron1987better}'s bias-corrected percentile interval in "
    r"brackets, not a standard error: with roughly six effective clusters the $t$ reference "
    r"behind a Wald interval is itself an approximation, so the bootstrap distribution is read "
    r"directly. Columns \ref{estimation:local} and \ref{estimation:pooled} bootstrap the "
    r"coefficients (WCB at the conglomerate level, except on the rows marked $\dagger$ below; "
    r"\textcite{cameron2008bootstrap}, \textcite{mackinnon2017wild}); columns "
    r"\ref{estimation:single_idx} and \ref{estimation:single_idx_time} additionally re-solve the "
    r"index direction and re-profile the link at every draw, so their intervals carry that "
    r"uncertainty too. Stars are read from the printed interval in every column. "
)

_OPEN_SE = (
    r"All columns report standard errors in parentheses (WCB at the conglomerate level, except "
    r"on the rows marked $\dagger$ below; \textcite{cameron2008bootstrap}, "
    r"\textcite{mackinnon2017wild}). "
)

_OPEN_MIXED = (
    r"Standard errors appear in parentheses and bias-corrected percentile intervals "
    r"(\textcite{efron1987better}) in brackets; the two are not comparable in width, an "
    r"interval at this many effective clusters being several times a standard error. Both come "
    r"from a WCB at the conglomerate level, except on the rows marked $\dagger$ below "
    r"(\textcite{cameron2008bootstrap}, \textcite{mackinnon2017wild}). Where an interval is "
    r"printed, the stars are read from it. "
)

_BODY = (
    r"Columns index the estimation strategies "
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

# The note is built BEFORE the cells are rendered, so which opening applies is not yet known.
# Emit a placeholder and let the caller substitute it once the table is assembled -- the same
# mechanism the AME_CI / national-SE notes already use, and for the same reason: the note must
# describe what ran, not what was planned.
OPEN_TOKEN = "%%SECOND_STAGE_OPEN%%"

# Kept for callers that want the all-intervals wording verbatim.
SECOND_STAGE_NOTE = _OPEN_INTERVAL + _BODY


def second_stage_note() -> str:
    """-> the shared second-stage note body with the opening left as OPEN_TOKEN.

    Substitute it with `note_open(bands)` after rendering. A caller that forgets will emit a
    visible `%%SECOND_STAGE_OPEN%%` in the .tex rather than a plausible-but-wrong sentence.
    """
    return OPEN_TOKEN + _BODY


def note_open(bands=True) -> str:
    """The opening sentence, chosen by what the table actually printed.

      True    -- every second line is an interval (the intended state)
      False   -- no band was available; every second line is a standard error
      'mixed' -- some columns had a band and some did not, so the note describes both and
                 warns against comparing their widths
    """
    if bands is True:
        return _OPEN_INTERVAL
    if not bands:
        return _OPEN_SE
    return _OPEN_MIXED

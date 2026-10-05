"""utils/sleep_notes.py

The note text shared by the sleepiness tables: the spec-12 comparison (first and second stage)
and the per-routine appendix tables written by sleep_export_e1 / sleep_export_e2 /
sleep_export_link.

Each convention is spelled ONCE here -- the unit of observation, the effect units, what the
brackets or parentheses hold, the star rule, the reversed-draw count -- so every table states it
in the same words. Notes carry only what a reader needs to read the numbers; the method and its
justification live in V_Main's Section sec:empirical:sleep, which the inference clause cites.
Callers keep their own size/wrapper command; these functions return note text only.

The inference clause is chosen by what the table ACTUALLY printed, not by what it was meant to
print. A band is a cluster artifact (sleep_ame_twostage.py for the single-index routines,
sleep_wcb_band.py for the linear ones); when one has not landed, the cell falls back to a
standard error, and a note that still promised an interval would be describing a calculation
that did not run. `note_open(bands)` is how the caller says which happened.
"""

SECTION_REF = r"Section~\ref{sec:empirical:sleep}"
UNIT_OBS = r"conglomerate $\times$ deposit type $\times$ MCA $\times$ quarter"
PER_UNIT = r"in pp of the sleepy share per row-label unit"
_NATIONAL = rf"$\dagger$: quarter-clustered, national regressors ({SECTION_REF})"
_STARS_INTERVAL = r"*/**/***: the 90/95/99\% interval excludes zero. "
_STARS_P = r"*** $p<0.01$, ** $p<0.05$, * $p<0.1$. "


def _unit_obs(pooled: bool) -> str:
    """The unit-of-observation phrase, shared by `sample_clause` and `first_stage_note`.
    D-firm rows are national cells rather than MCAs, so whenever the sample pools D firms in
    with B firms the clause carries that qualifier; a B-only sample has no D-firm row to
    qualify, so it keeps the plain phrase."""
    return (r"conglomerate $\times$ deposit type $\times$ MCA (national for D firms) "
            r"$\times$ quarter" if pooled else UNIT_OBS)


def sample_clause(pooled: bool = True, extra: str = "") -> str:
    """Unit of observation and which firms enter. `extra` appends a caveat to the same clause
    (e.g. that one column of a comparison drops the D firms)."""
    who = (r"$\mathrm{B}$ and $\mathrm{D}$ firms" if pooled
           else r"$\mathrm{B}$ firms only ($\mathrm{D}$ firms excluded)")
    return rf"Unit: {_unit_obs(pooled)}; {who}{extra}. "


def effects_clause(kind: str, linear_cols: str = "", link_cols: str = "") -> str:
    """What the numbers are: 'coef' (linear routine), 'ame' (single-index routine) or 'both'
    (a comparison; `linear_cols` / `link_cols` name the columns of each family)."""
    if kind == "coef":
        return rf"Coefficients {PER_UNIT}. "
    if kind == "ame":
        return rf"Average marginal effects {PER_UNIT}. "
    return (rf"Coefficients {linear_cols} and average marginal effects {link_cols} "
            rf"{PER_UNIT}. ")


def linear_link_caveat(obj: str, where: str = "") -> str:
    """The one caveat every linear-link table needs: nothing bounds the fitted share. `where`
    names the columns when only some of the table is linear, e.g. " of (I)--(II)"."""
    return rf"The linear link{where} is unconstrained, so {obj} can exceed 100. "


# The inference clause is only known once the cells are rendered, so the note carries a
# placeholder that the caller substitutes -- the same mechanism the AME_CI / national-SE notes
# use, and for the same reason: the note must describe what ran, not what was planned.
OPEN_TOKEN = "%%SECOND_STAGE_OPEN%%"


def second_stage_note(sample: str, effects: str, caveats: str = "") -> str:
    """-> sample + effects + caveats, then OPEN_TOKEN for the inference clause.

    Substitute the token with `note_open(bands, two_stage)` (plus `reversed_draws_note`) after
    rendering. A caller that forgets will emit a visible `%%SECOND_STAGE_OPEN%%` in the .tex
    rather than a plausible-but-wrong sentence.
    """
    return sample + effects + caveats + OPEN_TOKEN


def note_open(bands=True, two_stage: str = "") -> str:
    """The inference clause and star rule, chosen by what the table actually printed.

      True    -- every second line is an interval (the intended state)
      False   -- no band was available; every second line is a standard error
      'mixed' -- some cells carry an interval and some an SE

    `two_stage` qualifies the bootstrap where the index direction is re-solved at every draw,
    e.g. ", two-stage" for a single-index routine.
    """
    if bands is True:
        return (rf"Brackets: bias-corrected 95\% WCB intervals, conglomerate-clustered{two_stage}; "
                rf"{_NATIONAL}. {_STARS_INTERVAL}")
    if not bands:
        return (rf"Parentheses: WCB standard errors, conglomerate-clustered; {_NATIONAL}. "
                rf"{_STARS_P}")
    return (rf"Brackets: bias-corrected 95\% WCB intervals{two_stage}, starred when the "
            r"90/95/99\% interval excludes zero; parentheses: WCB standard errors, starred at "
            r"$p<0.1/0.05/0.01$; both conglomerate-clustered; " + _NATIONAL + ". ")


# The two F rows of every first-stage table, and the sentence that says what each one is. The
# labels live beside the sentence because the sentence points at them ("Effective $F$", "the
# row above it"): ROW_F_ALL is the fit's own F, ROW_F_EFF the value sleep_first_stage_strength.py
# stores for the column.
ROW_F_ALL = r"$F$, all slopes"
ROW_F_EFF = r"Effective $F$ (excluded instruments)"
_FIRST_STAGE_F = (r"Effective $F$ is the weak-instrument statistic for the excluded instruments "
                  r"\parencite{oleapflueger2013}, conglomerate-clustered; the row above it is the "
                  r"joint $F$ of all slopes, instruments and state controls together. ")


def first_stage_note(pooled: bool = True, extra: str = "") -> str:
    """The first-stage (deposit-spread) note: dependent variable and units, sample, SEs, what
    the two $F$ rows hold, stars."""
    who = (r"$\mathrm{B}$ and $\mathrm{D}$ firms" if pooled
           else r"$\mathrm{B}$ firms only ($\mathrm{D}$ firms excluded)")
    return (r"Dependent variable: quarterly deposit spread, in pp per row-label unit. "
            rf"Unit: {_unit_obs(pooled)}, endogenously priced types $k=4,5$; {who}{extra}. "
            r"Parentheses: WCB standard errors, conglomerate-clustered. " + _FIRST_STAGE_F
            + _STARS_P)


def reversed_draws_note(fits) -> str:
    """The clause reporting reversed-index draws RETAINED in the two-stage intervals, or "".

    `fits` is any iterable of fitted results; those carrying an attached `ame_boot` contribute.
    The count is the largest over the columns, per clustering arm, because the note speaks for
    the table rather than for one column. Nothing is said when every count is zero.
    """
    worst = {"conglomerate": 0, "quarter": 0}
    B = None
    for r in fits:
        boot = getattr(r, "ame_boot", None)
        if not isinstance(boot, dict):
            continue
        for arm, lab in (("congl", "conglomerate"), ("quarter", "quarter")):
            blk = boot.get(arm) or {}
            worst[lab] = max(worst[lab], int(blk.get("n_cos_idx_neg", 0) or 0))
            b = blk.get("B_used") or (getattr(r, "ame_2s_meta", None) or {}).get("B")
            if b:
                B = int(b)
    if not any(worst.values()):
        return ""
    of_b = rf" of $B={B}$" if B else ""
    return (rf"Bootstrap draws whose re-solved index reverses sign are kept: at most "
            rf"{worst['conglomerate']} (conglomerate) and {worst['quarter']} (quarter){of_b} "
            rf"per column. ")

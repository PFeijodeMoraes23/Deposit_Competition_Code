"""
utils/window.py
===============
THE estimation window — one definition, imported by every stage that touches it.

    from utils.window import MIN_YEAR, MAX_YEAR, apply_window

Why this module exists
----------------------
The window was previously re-declared in each script. That is exactly how the panel drifted out of
sync before: the estimation ran on one sample while the descriptives described another, and nothing
in the code made the inconsistency visible. A single constant makes the sample definition auditable
and keeps the tables, figures and estimates describing the same data.

THE WINDOW: 2016-2024 (inclusive). Both bounds are DATA CONSTRAINTS, not modelling preferences.

Lower bound 2016 — the model's firm does not exist before it
------------------------------------------------------------
The model's firm j is a PRUDENTIAL CONGLOMERATE (CodConglomeradoPrudencial): deposits are aggregated
there and every bank covariate (assets, equity, capital, cost ratios) must be measured on that same
entity. BCB's prudential-conglomerate framework (Res. 4.280/2013) was phased in over 2015-2016.
Before 2016:
  * the type-1 "Prudential Conglomerate" IF-Data report contains ONLY credit cooperatives — the banks
    are simply absent (2015: the largest type-1 institution is a cooperative at R$10bn; Banco do
    Brasil, Itau and Bradesco do not appear; 0 of 4,449 rows carry a C00 conglomerate code);
  * the banks report individually (type-3) or as financial conglomerates (type-2) — a DIFFERENT
    consolidation basis that does not splice: summing type-3 individual entities up to a conglomerate
    over-counts intra-group exposures by ~20% against the netted prudential figure (e.g. C0080075:
    1.19e12 individual-sum vs 0.875e12 prudential).
So a pre-2016 bank characteristic ON THE MODEL'S ENTITY DEFINITION does not exist, and cannot be
constructed without inventing a ~20% level break exactly at the boundary — a break that would land in
the cost-shifter and LOO instruments used to identify the price coefficient.
Independently, BCB's access-point series (branches/correspondents) also begins in 2016, so
`branches_per1000` — a demand demographic — has no pre-2016 source either. Two unrelated constraints
give the same floor.

Upper bound 2024 — 2025 has no clean asset return
--------------------------------------------------
2025 concentrates several BCB measurement breaks. The report renumbering (78182 -> 140220, ...) is
handled by panel_4's period-aware alias map, and the COSIF fee recode by the fee guard. But the DRE
was RESTRUCTURED, not merely renumbered: the old gross "Receitas de Intermediacao Financeira" (78208)
has NO 2025 counterpart — TVM income is now reported net of fair-value/hedge adjustments and
derivatives became a net result. The closest reconstruction recovers only ~0.89 of the old level, so
`asset_return_qoq` (r^j) is biased down in 2025. r^j enters psi1 of the cost stage: the deposit
franchise is priced off (r^j - r^f). Rather than carry a known-biased final year into the structural
estimates, the window ends at 2024.

Why ONE window for estimation AND descriptives
----------------------------------------------
The stages are chained: the sleep's spec-12 coefficients reconstruct phi-hat; phi-hat sets the demand's
active-market denominator; the demand's (alpha-hat, delta-hat) enter the cost stage's psi. Estimating
them on different samples would identify phi off years the demand never sees, and price a franchise
using a delta-hat from another period. The descriptives motivate the model, so they must describe the
same data the model is fit on — a table over 2013-2025 next to an estimate over 2016-2024 invites the
reader to attribute the estimate to years it never saw.

What the window costs
---------------------
36 quarters (2016Q1-2024Q4). Pix (Nov 2020) sits INSIDE it with 19 pre- and 17 post-quarters, so the
paper's central event remains identified. See counterfactuals_plan.md section 0A.
"""
import os

MIN_YEAR = int(os.environ.get("DEMAND_MIN_YEAR", 2016))
MAX_YEAR = int(os.environ.get("DEMAND_MAX_YEAR", 2024))

WINDOW_LABEL = f"{MIN_YEAR}--{MAX_YEAR}"          # for LaTeX captions/notes
WINDOW_LABEL_PLAIN = f"{MIN_YEAR}-{MAX_YEAR}"     # for console/plot text


def apply_window(df, year_col="year", label=None, verbose=True):
    """Restrict `df` to [MIN_YEAR, MAX_YEAR] on `year_col`. Returns the filtered copy.

    No-op (with a warning) if `year_col` is absent, so callers that reshape first stay safe.
    """
    if year_col not in df.columns:
        if verbose:
            print(f"  [window] WARNING: no '{year_col}' column — window NOT applied"
                  f"{f' ({label})' if label else ''}")
        return df
    n0 = len(df)
    out = df[(df[year_col] >= MIN_YEAR) & (df[year_col] <= MAX_YEAR)].copy()
    if verbose:
        print(f"  [window] {WINDOW_LABEL_PLAIN}{f' ({label})' if label else ''}: "
              f"kept {len(out):,} of {n0:,} rows (dropped {n0 - len(out):,})")
    return out

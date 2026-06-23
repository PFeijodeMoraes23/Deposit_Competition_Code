"""export_8_sleep_results.py — Est8 (Joint Single-Index, kernel local-linear) TeX tables.
Spec 12 only; non-spec-12 columns render as blank by construction."""
from export_sleep_link_common import export_link_results

if __name__ == "__main__":
    export_link_results(
        8,
        title=r"Est 8 (Pooled B+D Joint Single-Index, kernel local-linear)",
        ss_caption=r"Pooled B+D --- Second Stage Estimation (Est.\ 8, Joint Single-Index, kernel; spec 12)",
        est_note=(r"Reported estimates are average-derivative Average Marginal Effects (AME) from a "
                  r"joint single-index (Ichimura 1993 SLS) with a kernel local-linear link profiled by "
                  r"backfitting (for the entity FE and the multiplicative carry-forward term) and "
                  r"monotonised ex post by rearrangement (Chernozhukov-Fernandez-Val-Galichon 2009). "
                  r"The index direction $\theta$ ($\lVert\theta\rVert=1$) and link are estimated "
                  r"\emph{together}. \textbf{Spec 12 only} (kernel cost); the robust (Cauchy) fit is "
                  r"shown, the plain-LS fit stored alongside. "
                  r"CF: control function residual $\hat{v}$ interacted with lagged deposits."),
    )

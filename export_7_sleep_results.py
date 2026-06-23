"""export_7_sleep_results.py — Est7 (Joint Single-Index, monotone sieve) TeX tables."""
from export_sleep_link_common import export_link_results

if __name__ == "__main__":
    export_link_results(
        7,
        title=r"Est 7 (Pooled B+D Joint Single-Index, monotone I-spline sieve)",
        ss_caption=r"Pooled B+D --- Second Stage Estimation (Est.\ 7, Joint Single-Index, sieve)",
        est_note=(r"Reported estimates are average-derivative Average Marginal Effects (AME) from a "
                  r"joint single-index (Ichimura 1993 SLS): the index direction $\theta$ "
                  r"($\lVert\theta\rVert=1$) and the monotone I-spline link $G$ are estimated "
                  r"\emph{together}, so inference accounts for estimating $\theta$. The robust (Cauchy) "
                  r"fit is shown; the plain-LS fit is stored alongside. "
                  r"CF: control function residual $\hat{v}$ interacted with lagged deposits."),
    )

"""export_4_sleep_results.py — Est4 (Constrained Linear, uniform eta) TeX tables."""
from export_sleep_link_common import export_link_results

if __name__ == "__main__":
    export_link_results(
        4,
        title=r"Est 4 (Pooled B+D Constrained Linear, uniform $\eta$)",
        ss_caption=r"Pooled B+D --- Second Stage Estimation (Est.\ 4, Constrained Linear)",
        est_note=(r"Reported estimates are Average Marginal Effects (AME) from a constrained-linear "
                  r"(uniform-$\eta$) NLLS with the $[0,1]$ bound imposed in estimation; the clip is "
                  r"non-differentiable, so standard errors are approximate. "
                  r"CF: control function residual $\hat{v}$ interacted with lagged deposits."),
    )

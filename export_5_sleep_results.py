"""export_5_sleep_results.py — Est5 (Probit, normal eta) TeX tables."""
from export_sleep_link_common import export_link_results

if __name__ == "__main__":
    export_link_results(
        5,
        title=r"Est 5 (Pooled B+D Probit, normal $\eta$)",
        ss_caption=r"Pooled B+D --- Second Stage Estimation (Est.\ 5, Probit AME)",
        est_note=(r"Reported estimates are Average Marginal Effects (AME) from an NLLS probit "
                  r"(normal-$\eta$) specification. "
                  r"CF: control function residual $\hat{v}$ interacted with lagged deposits."),
    )

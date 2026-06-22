"""export_6_sleep_results.py — Est6 (Single-Index, nonparametric eta) TeX tables."""
from export_sleep_link_common import export_link_results

if __name__ == "__main__":
    export_link_results(
        6,
        title=r"Est 6 (Pooled B+D Single-Index, nonparametric monotone $\eta$)",
        ss_caption=r"Pooled B+D --- Second Stage Estimation (Est.\ 6, Single-Index)",
        est_note=(r"Reported estimates are average-derivative Average Marginal Effects (AME) from a "
                  r"monotone single-index (cubic series/sieve link, logit-direction index); inference "
                  r"is conditional on the estimated index direction. "
                  r"CF: control function residual $\hat{v}$ interacted with lagged deposits."),
    )

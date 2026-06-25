"""export_9_sleep_results.py — E9 (OPTIONAL: Joint Single-Index, kernel) TeX tables; spec 12."""
from export_sleep_link_common import export_link_results

if __name__ == "__main__":
    export_link_results(
        9,
        title=r"E9 (optional): Pooled Joint Single-Index (kernel local-linear)",
        ss_caption=r"Pooled B+D --- Second Stage Estimation (Est.\ 9, Joint Single-Index kernel; spec 12)",
        est_note=(r"Joint single-index with a kernel local-linear link, monotonised ex post by "
                  r"rearrangement; \textbf{spec 12 only}. The robust kernel link can degenerate to the "
                  r"$[0,1]$ ceiling, so the plain-LS fit is stored alongside; use LS for $\phi$. "
                  r"CF: $\hat{v} \times$ lagged deposits."),
    )

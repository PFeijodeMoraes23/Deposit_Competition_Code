"""Shared cluster-imbalance arithmetic.

Single source of the effective-number-of-clusters formula used across the
sleepiness estimators (inference-time G*) and the descriptive cluster-imbalance
reporting (`desc_3.py`). Effective clusters follow Carter, Schnepel & Steigerwald
(2017): G* = G / (1 + CV^2), where CV is the coefficient of variation of the
per-cluster observation counts. Under wildly unequal cluster sizes G* collapses
far below the nominal G, which is what motivates the wild cluster bootstrap over
CRVE / Delta-method standard errors (MacKinnon & Webb 2017).
"""
import numpy as np


def effective_cluster_stats(sizes) -> dict:
    """Summarise cluster-size imbalance from per-cluster observation counts.

    Parameters
    ----------
    sizes : iterable of per-cluster observation counts (e.g. the values of
            ``cluster_series.value_counts()`` or ``df.groupby(cluster).size()``).

    Returns
    -------
    dict with keys: G_nominal (int), G_star (float, Carter et al. 2017),
    cv (float, ddof=0), mean_size (float), std_size (float), total_obs (float).
    """
    arr = np.asarray(list(sizes), dtype=float)
    G = int(arr.size)
    if G == 0:
        return {"G_nominal": 0, "G_star": 1.0, "cv": 0.0,
                "mean_size": 0.0, "std_size": 0.0, "total_obs": 0.0}
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    cv = std / mean if mean > 0 else 0.0
    G_star = max(1.0, G / (1.0 + cv ** 2))
    return {"G_nominal": G, "G_star": G_star, "cv": cv,
            "mean_size": mean, "std_size": std, "total_obs": float(arr.sum())}

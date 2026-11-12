"""Household price indexes and summary statistics."""

from __future__ import annotations

import numpy as np


def laspeyres_inflation(W: np.ndarray, rel: np.ndarray) -> np.ndarray:
    """Each household's Laspeyres inflation: sum_i w_hi * r_i - 1 (W is H x I, rel is I)."""
    return W @ rel - 1


def weighted_quantile(x: np.ndarray, weights: np.ndarray, p: float) -> float:
    """Value below which a share p of the total weight lies."""
    order = np.argsort(x)
    cum = np.cumsum(weights[order]) / weights.sum()
    return float(x[order][np.searchsorted(cum, p)])

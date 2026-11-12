"""Household price indexes and summary statistics.

Two views of the same formulas:

* From spending shares and price relatives, for every household at once:
  ``laspeyres_inflation(W, rel) = W @ rel - 1``.
* Bilateral indexes from prices and quantities ``(p0, p1, q0, q1)``. Passing
  quantities in place of prices, ``fisher(q0, q1, p0, p1)``, gives the matching
  quantity index; the factor-reversal tests use this.
"""

from __future__ import annotations

import numpy as np


def laspeyres_inflation(W: np.ndarray, rel: np.ndarray) -> np.ndarray:
    """Each household's Laspeyres inflation, sum_i w_hi * r_i - 1.

    W is H x I; rel is a vector of I relatives or an I x T matrix.
    """
    return W @ rel - 1


def weighted_quantile(x: np.ndarray, weights: np.ndarray, p: float) -> float:
    """Value below which a share p of the total weight lies."""
    order = np.argsort(x)
    cum = np.cumsum(weights[order]) / weights.sum()
    return float(x[order][np.searchsorted(cum, p)])


def laspeyres(p0: np.ndarray, p1: np.ndarray, q0: np.ndarray, q1: np.ndarray) -> float:
    """Cost of the base-period basket at new prices relative to base prices."""
    return float(np.dot(p1, q0) / np.dot(p0, q0))


def paasche(p0: np.ndarray, p1: np.ndarray, q0: np.ndarray, q1: np.ndarray) -> float:
    """Cost of the current-period basket at new prices relative to base prices."""
    return float(np.dot(p1, q1) / np.dot(p0, q1))


def fisher(p0: np.ndarray, p1: np.ndarray, q0: np.ndarray, q1: np.ndarray) -> float:
    """Geometric mean of Laspeyres and Paasche."""
    return float(np.sqrt(laspeyres(p0, p1, q0, q1) * paasche(p0, p1, q0, q1)))


def tornqvist(p0: np.ndarray, p1: np.ndarray, q0: np.ndarray, q1: np.ndarray) -> float:
    """Geometric mean of price relatives weighted by the average of both periods' shares."""
    s0 = p0 * q0 / np.dot(p0, q0)
    s1 = p1 * q1 / np.dot(p1, q1)
    return float(np.exp(np.dot((s0 + s1) / 2, np.log(p1 / p0))))

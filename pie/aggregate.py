"""Group-level aggregation and the basket-effect decomposition."""

from __future__ import annotations

import numpy as np


def averaging_matrix(labels: np.ndarray, survey_weight: np.ndarray) -> tuple[list, np.ndarray]:
    """G x H matrix whose row g holds the normalized survey weights of group g's households.

    Note: this is dense, so multiplying by it costs O(G*H*...). See the feedback
    on algorithm A4; a sparse or groupby version is the planned replacement.
    """
    levels = sorted(set(labels))
    A = np.array([(labels == level) * survey_weight for level in levels], dtype=float)
    return levels, A / A.sum(axis=1, keepdims=True)


def basket_effect(w_a: np.ndarray, w_b: np.ndarray, rel: np.ndarray) -> np.ndarray:
    """Per-category contribution to the inflation gap caused by different baskets.

    (w_a - w_b) * (r - 1); the contributions sum to the total basket effect.
    """
    return (w_a - w_b) * (rel - 1)

"""Group-level aggregation and the basket-effect decomposition.

Group results are weighted means over households, written as a G x H matrix A
whose row g holds the normalized survey weights of group g's households. Each
household belongs to one group, so A has exactly H nonzeros. Stored sparse,
building it and multiplying it by an H x I matrix both cost O(H * I); stored
dense, the same product costs O(G * H * I).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

# Keeps a replicate's weighted mean defined when every household in a group has
# replicate weight 0 in that half-sample; matches pie.variance.brr_mean.
REPLICATE_EPS = 1e-12


def group_index(labels: np.ndarray) -> tuple[list, np.ndarray]:
    """Sorted distinct labels and each household's position in that list."""
    levels, codes = np.unique(np.asarray(labels), return_inverse=True)
    return list(levels), codes


def averaging_matrix(labels: np.ndarray, survey_weight: np.ndarray) -> tuple[list, sp.csr_array]:
    """Sparse G x H weighted-mean matrix. Built in O(H)."""
    levels, codes = group_index(labels)
    n = len(codes)
    A = sp.csr_array(
        (np.asarray(survey_weight, dtype=float), (codes, np.arange(n))), shape=(len(levels), n)
    )
    totals = np.asarray(A.sum(axis=1)).ravel()
    return levels, sp.csr_array(sp.diags_array(1 / totals) @ A)


def dense_averaging_matrix(
    labels: np.ndarray, survey_weight: np.ndarray
) -> tuple[list, np.ndarray]:
    """The same matrix stored dense (G x H). Kept for the benchmark comparison."""
    levels, codes = group_index(labels)
    A = np.zeros((len(levels), len(codes)))
    A[codes, np.arange(len(codes))] = survey_weight
    return levels, A / A.sum(axis=1, keepdims=True)


def replicate_averaging_matrix(
    labels: np.ndarray, weights: np.ndarray, replicate_eps: float = REPLICATE_EPS
) -> tuple[list, sp.csr_array]:
    """Stack one averaging matrix per weight column into a sparse (K*G) x H matrix.

    weights is H x K: column 0 the full-sample weight, columns 1..K-1 the BRR
    replicate weights. Row k*G + g averages group g with weight column k.
    Replicate columns are clipped at 0 and shifted by replicate_eps, exactly as
    pie.variance.brr_mean does, so the two give identical estimates.
    """
    levels, codes = group_index(labels)
    h, k = weights.shape
    g = len(levels)
    w = np.asarray(weights, dtype=float).copy()
    w[:, 1:] = np.clip(w[:, 1:], 0, None) + replicate_eps
    rows = (np.arange(k)[None, :] * g + codes[:, None]).ravel()  # H*K row indices
    cols = np.repeat(np.arange(h), k)
    B = sp.csr_array((w.ravel(), (rows, cols)), shape=(k * g, h))
    totals = np.asarray(B.sum(axis=1)).ravel()
    return levels, sp.csr_array(sp.diags_array(1 / totals) @ B)


def household_first(W: np.ndarray, R: np.ndarray, B: sp.csr_array | np.ndarray) -> np.ndarray:
    """B @ (W @ R): every household's index first, then group means.

    Cost O(H*I*S + nnz(B)*S) for S columns of R.
    """
    return np.asarray(B @ (W @ R))


def aggregate_first(W: np.ndarray, R: np.ndarray, B: sp.csr_array | np.ndarray) -> np.ndarray:
    """(B @ W) @ R: group mean shares first, then the index.

    Cost O(nnz(B)*I + rows(B)*I*S). Exact only for indexes linear in the
    shares (see docs/theory.md).
    """
    return np.asarray(B @ W) @ R


class GroupedAggregator:
    """Aggregate first with one dense BLAS product per group instead of a sparse product.

    For group g with households H_g and weight columns K, the group's mean
    shares under every weight column are (w_g^T @ W_g) / column sums, a dense
    K x H_g by H_g x I product. Summed over groups this is the same O(K*H*I)
    work as B @ W, but it runs in multithreaded BLAS rather than a
    single-threaded sparse kernel. Rows come out in the same order as
    replicate_averaging_matrix (row k*G + g).
    """

    def __init__(
        self, labels: np.ndarray, weights: np.ndarray, replicate_eps: float = REPLICATE_EPS
    ) -> None:
        self.levels, codes = group_index(labels)
        w = np.asarray(weights, dtype=float)
        w = (w[:, None] if w.ndim == 1 else w).copy()
        w[:, 1:] = np.clip(w[:, 1:], 0, None) + replicate_eps
        self.k = w.shape[1]
        self.index = [np.flatnonzero(codes == g) for g in range(len(self.levels))]
        self.norm_weights = [w[idx] / w[idx].sum(axis=0) for idx in self.index]

    def shares(self, W: np.ndarray) -> np.ndarray:
        """(K*G) x I matrix of group mean shares."""
        g = len(self.levels)
        out = np.empty((self.k, g, W.shape[1]))
        for j, (idx, w) in enumerate(zip(self.index, self.norm_weights, strict=True)):
            out[:, j, :] = w.T @ W[idx]
        return out.reshape(self.k * g, W.shape[1])

    def aggregate_first(self, W: np.ndarray, R: np.ndarray) -> np.ndarray:
        return self.shares(W) @ R


def basket_effect(w_a: np.ndarray, w_b: np.ndarray, rel: np.ndarray) -> np.ndarray:
    """Per-category contribution to the inflation gap caused by different baskets.

    (w_a - w_b) * (r - 1); the contributions sum to the total basket effect.
    """
    return (w_a - w_b) * (rel - 1)

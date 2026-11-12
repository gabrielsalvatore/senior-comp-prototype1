"""Sparse aggregation, stacked replicate weights and the chain-order identity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pie.aggregate import (
    aggregate_first,
    averaging_matrix,
    dense_averaging_matrix,
    household_first,
    replicate_averaging_matrix,
)
from pie.config import N_REP, REP_COLS
from pie.variance import brr_from_stacked, brr_mean


def test_averaging_matrix_by_hand() -> None:
    levels, A = averaging_matrix(np.array(["b", "a", "b"]), np.array([1.0, 2.0, 3.0]))
    assert levels == ["a", "b"]
    np.testing.assert_allclose(A.toarray(), [[0, 1, 0], [0.25, 0, 0.75]])


def test_sparse_matrix_has_one_entry_per_household(rng: np.random.Generator) -> None:
    labels = rng.integers(0, 7, 500)
    _, A = averaging_matrix(labels, rng.uniform(1, 2, 500))
    assert A.nnz == 500
    np.testing.assert_allclose(A.sum(axis=1), 1.0)


def test_sparse_equals_dense(rng: np.random.Generator) -> None:
    labels, w = rng.integers(0, 4, 60), rng.uniform(1, 9, 60)
    np.testing.assert_allclose(
        averaging_matrix(labels, w)[1].toarray(), dense_averaging_matrix(labels, w)[1]
    )


@settings(max_examples=50, deadline=None)
@given(
    h=st.integers(5, 60),
    i=st.integers(1, 12),
    s=st.integers(1, 8),
    g=st.integers(1, 5),
    seed=st.integers(0, 2**32 - 1),
)
def test_chain_order_gives_same_answer(h: int, i: int, s: int, g: int, seed: int) -> None:
    """(A W) R = A (W R) for sparse and dense A: aggregate first is exact for Laspeyres."""
    rng = np.random.default_rng(seed)
    W = rng.dirichlet(np.ones(i), size=h)
    R = rng.uniform(0.8, 1.5, size=(i, s))
    labels = rng.integers(0, g, h)
    w = rng.uniform(1, 100, h)
    for A in (averaging_matrix(labels, w)[1], dense_averaging_matrix(labels, w)[1]):
        np.testing.assert_allclose(aggregate_first(W, R, A), household_first(W, R, A))


def test_stacked_replicates_match_brr_mean(rng: np.random.Generator) -> None:
    """One sparse product gives every group's estimate and BRR SE, same as the loop."""
    h, i = 300, 5
    W = rng.dirichlet(np.ones(i), size=h)
    rel = rng.uniform(0.9, 1.4, size=i)
    labels = rng.choice(["Q1", "Q2", "Q3"], size=h)
    hh = pd.DataFrame({"FINLWT21": rng.uniform(500, 5000, h)})
    for col in REP_COLS:  # half-sample style: some replicate weights are 0
        hh[col] = hh["FINLWT21"] * rng.choice([0.0, 2.0], size=h)
    levels, B = replicate_averaging_matrix(labels, hh[["FINLWT21", *REP_COLS]].to_numpy())
    assert B.shape == ((N_REP + 1) * 3, h)
    est, se = brr_from_stacked(aggregate_first(W, rel[:, None], B) - 1, len(levels))
    pi = W @ rel - 1
    for g, level in enumerate(levels):
        e, s = brr_mean(pi, hh, labels == level)
        assert est[g, 0] == pytest.approx(e)
        assert se[g, 0] == pytest.approx(s)


def test_brr_from_stacked_checks_shape() -> None:
    with pytest.raises(ValueError, match="rows"):
        brr_from_stacked(np.zeros((10, 2)), 3)

"""The aggregate-first theorem and the Jensen-gap bounds (docs/theory.md)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pie.indexes import laspeyres, paasche, tornqvist
from pie.theory import MEANS, group_second_moments, household_index, jensen_gap

cases = st.tuples(
    st.integers(2, 40),  # households
    st.integers(1, 10),  # items
    st.integers(0, 2**32 - 1),  # seed
)


def draw(h: int, i: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    W = rng.dirichlet(np.ones(i), size=h)
    rel = rng.uniform(0.7, 1.8, size=i)
    weights = rng.uniform(1, 100, size=h)
    return W, rel, weights


def test_family_reproduces_named_indexes() -> None:
    """phi(sum s g(r)) with the right shares gives Laspeyres, Paasche and Törnqvist."""
    p0, p1 = np.array([1.0, 2.0, 4.0]), np.array([1.5, 2.0, 3.0])
    q0, q1 = np.array([10.0, 5.0, 1.0]), np.array([6.0, 6.0, 2.0])
    s0 = p0 * q0 / np.dot(p0, q0)
    s1 = p1 * q1 / np.dot(p1, q1)
    rel = p1 / p0
    assert household_index(s0[None], rel, "arithmetic")[0] == pytest.approx(
        laspeyres(p0, p1, q0, q1)
    )
    assert household_index(s1[None], rel, "harmonic")[0] == pytest.approx(paasche(p0, p1, q0, q1))
    avg = ((s0 + s1) / 2)[None]
    assert household_index(avg, rel, "geometric")[0] == pytest.approx(tornqvist(p0, p1, q0, q1))


@settings(max_examples=100, deadline=None)
@given(cases)
def test_aggregate_share_index_is_aggregate_first(case: tuple[int, int, int]) -> None:
    """Proposition 1: phi(S_g . g(r)) uses only the group mean shares S_g = omega @ W."""
    W, rel, weights = draw(*case)
    omega = weights / weights.sum()
    for kind in MEANS:
        row = jensen_gap(W, rel, weights, kind)
        S = omega @ W
        assert row.aggregate_share_index == pytest.approx(household_index(S[None], rel, kind)[0])


@settings(max_examples=100, deadline=None)
@given(cases)
def test_democratic_mean_commutes_for_laspeyres(case: tuple[int, int, int]) -> None:
    """Proposition 2, affine case: the gap is exactly zero."""
    row = jensen_gap(*draw(*case), "arithmetic")
    assert row.gap == pytest.approx(0.0, abs=1e-12)
    assert row.lower_bound == row.upper_bound == 0.0


@settings(max_examples=200, deadline=None)
@given(cases, st.sampled_from(["harmonic", "geometric"]))
def test_gap_is_nonnegative_and_within_bounds(case: tuple[int, int, int], kind: str) -> None:
    """Propositions 2-3: phi convex, so 0 <= (1/2) min phi'' Var <= gap <= (1/2) max phi'' Var."""
    row = jensen_gap(*draw(*case), kind)
    tol = 1e-12
    assert row.gap >= -tol
    assert row.lower_bound - tol <= row.gap <= row.upper_bound + tol
    assert row.lower_bound - tol <= row.second_order <= row.upper_bound + tol


def test_gap_is_strictly_positive_when_households_differ() -> None:
    """Proposition 2, converse: a non-affine phi breaks commutation for two households."""
    W = np.array([[1.0, 0.0], [0.0, 1.0]])
    rel = np.array([1.0, 2.0])
    for kind in ("harmonic", "geometric"):
        assert jensen_gap(W, rel, np.ones(2), kind).gap > 1e-3


def test_identical_households_have_no_gap() -> None:
    W = np.tile([0.2, 0.3, 0.5], (4, 1))
    for kind in MEANS:
        row = jensen_gap(W, np.array([1.1, 0.9, 1.4]), np.arange(1.0, 5.0), kind)
        assert row.gap == pytest.approx(0.0, abs=1e-14)


@settings(max_examples=100, deadline=None)
@given(cases)
def test_variance_from_second_moments(case: tuple[int, int, int]) -> None:
    """Var(x) can be computed aggregate first from S and M, without household x_h."""
    W, rel, weights = draw(*case)
    for kind in MEANS:
        row = jensen_gap(W, rel, weights, kind)
        assert row.second_order_from_moments == pytest.approx(row.second_order, abs=1e-10)


def test_second_moments_by_hand() -> None:
    W = np.array([[1.0, 0.0], [0.0, 1.0]])
    S, M = group_second_moments(W, np.array([1.0, 3.0]))
    np.testing.assert_allclose(S, [0.25, 0.75])
    np.testing.assert_allclose(M, [[0.25, 0.0], [0.0, 0.75]])


def test_geometric_gap_by_hand() -> None:
    """Two equal-weight households with indexes 1 and 2: AM - GM = 1.5 - sqrt(2)."""
    W = np.array([[1.0, 0.0], [0.0, 1.0]])
    row = jensen_gap(W, np.array([1.0, 2.0]), np.ones(2), "geometric")
    assert row.democratic_mean == pytest.approx(1.5)
    assert row.aggregate_share_index == pytest.approx(np.sqrt(2))
    assert row.gap == pytest.approx(1.5 - np.sqrt(2))


def test_gap_table_has_one_row_per_group() -> None:
    from pie.theory import gap_table

    rng = np.random.default_rng(0)
    W = rng.dirichlet(np.ones(4), size=30)
    labels = pd.Series(["a"] * 10 + ["b"] * 15 + [None] * 5)
    table = gap_table(W, rng.uniform(0.9, 1.5, 4), labels, np.ones(30), "geometric")
    assert list(table.index) == ["a", "b"]
    assert list(table["n"]) == [10, 15]

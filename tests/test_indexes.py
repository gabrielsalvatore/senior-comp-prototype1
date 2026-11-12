"""Index formulas: hand-computed cases and the index-number axioms."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from pie.indexes import fisher, laspeyres, laspeyres_inflation, paasche, tornqvist

FORMULAS = {"laspeyres": laspeyres, "paasche": paasche, "fisher": fisher, "tornqvist": tornqvist}

# Two goods: bread doubles in price, people buy half as much; milk is unchanged.
P0, P1 = np.array([1.0, 2.0]), np.array([2.0, 2.0])
Q0, Q1 = np.array([10.0, 5.0]), np.array([5.0, 5.0])


# --- hand-computed cases ----------------------------------------------------


def test_laspeyres_by_hand() -> None:
    # (2*10 + 2*5) / (1*10 + 2*5) = 30 / 20
    assert laspeyres(P0, P1, Q0, Q1) == pytest.approx(1.5)


def test_paasche_by_hand() -> None:
    # (2*5 + 2*5) / (1*5 + 2*5) = 20 / 15
    assert paasche(P0, P1, Q0, Q1) == pytest.approx(4 / 3)


def test_fisher_by_hand() -> None:
    assert fisher(P0, P1, Q0, Q1) == pytest.approx(np.sqrt(1.5 * 4 / 3))


def test_tornqvist_by_hand() -> None:
    # s0 = (10, 10)/20 = (.5, .5); s1 = (10, 10)/20 = (.5, .5); relatives (2, 1)
    assert tornqvist(P0, P1, Q0, Q1) == pytest.approx(np.exp(0.5 * np.log(2)))


def test_laspeyres_inflation_by_hand() -> None:
    shares = np.array([[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]])
    relatives = np.array([1.10, 1.02])
    np.testing.assert_allclose(laspeyres_inflation(shares, relatives), [0.06, 0.10, 0.02])


def test_laspeyres_inflation_equals_laspeyres() -> None:
    """Share form sum_i s_i r_i equals the quantity form when s_i = p0_i q0_i / sum p0 q0."""
    shares = (P0 * Q0) / np.dot(P0, Q0)
    assert laspeyres_inflation(shares[None, :], P1 / P0)[0] + 1 == pytest.approx(
        laspeyres(P0, P1, Q0, Q1)
    )


# --- property-based tests of the axioms --------------------------------------

N_ITEMS = 6
positive = st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False)
vectors = arrays(np.float64, N_ITEMS, elements=positive)
scalars = st.floats(min_value=0.1, max_value=10.0)


@pytest.mark.parametrize("formula", FORMULAS.values(), ids=FORMULAS.keys())
@given(p0=vectors, q0=vectors, q1=vectors)
def test_identity(formula, p0, q0, q1) -> None:
    """Unchanged prices give an index of 1, whatever the quantities."""
    assert formula(p0, p0, q0, q1) == pytest.approx(1.0)


@pytest.mark.parametrize("formula", FORMULAS.values(), ids=FORMULAS.keys())
@given(p0=vectors, q0=vectors, q1=vectors, lam=scalars)
def test_proportionality(formula, p0, q0, q1, lam) -> None:
    """If every price is multiplied by lambda, the index is lambda."""
    assert formula(p0, lam * p0, q0, q1) == pytest.approx(lam)


@pytest.mark.parametrize("formula", FORMULAS.values(), ids=FORMULAS.keys())
@given(p0=vectors, p1=vectors, q0=vectors, q1=vectors)
def test_mean_value(formula, p0, p1, q0, q1) -> None:
    """The index lies between the smallest and largest price relative."""
    rel = p1 / p0
    value = formula(p0, p1, q0, q1)
    assert rel.min() * (1 - 1e-9) <= value <= rel.max() * (1 + 1e-9)


@pytest.mark.parametrize("formula", FORMULAS.values(), ids=FORMULAS.keys())
@given(p0=vectors, p1=vectors, q0=vectors, q1=vectors, units=vectors)
def test_commensurability(formula, p0, p1, q0, q1, units) -> None:
    """Changing units (price per kg -> per gram) leaves the index unchanged."""
    original = formula(p0, p1, q0, q1)
    rescaled = formula(p0 / units, p1 / units, q0 * units, q1 * units)
    assert rescaled == pytest.approx(original)


@pytest.mark.parametrize("formula", [fisher, tornqvist], ids=["fisher", "tornqvist"])
@given(p0=vectors, p1=vectors, q0=vectors, q1=vectors)
def test_time_reversal_passes(formula, p0, p1, q0, q1) -> None:
    """P(0 -> 1) * P(1 -> 0) = 1."""
    assert formula(p0, p1, q0, q1) * formula(p1, p0, q1, q0) == pytest.approx(1.0)


@pytest.mark.parametrize("formula", [laspeyres, paasche], ids=["laspeyres", "paasche"])
def test_time_reversal_fails(formula) -> None:
    assert formula(P0, P1, Q0, Q1) * formula(P1, P0, Q1, Q0) != pytest.approx(1.0)


@given(p0=vectors, p1=vectors, q0=vectors, q1=vectors)
def test_laspeyres_reversed_is_reciprocal_paasche(p0, p1, q0, q1) -> None:
    """Why Laspeyres fails time reversal: running it backward gives 1 / Paasche.

    So L(0->1) * L(1->0) = L / P, which is 1 only when the two baskets agree.
    """
    assert laspeyres(p1, p0, q1, q0) == pytest.approx(1 / paasche(p0, p1, q0, q1))


@given(p0=vectors, p1=vectors, q0=vectors, q1=vectors)
def test_fisher_factor_reversal(p0, p1, q0, q1) -> None:
    """Fisher price index * Fisher quantity index = change in total spending."""
    value_ratio = np.dot(p1, q1) / np.dot(p0, q0)
    assert fisher(p0, p1, q0, q1) * fisher(q0, q1, p0, p1) == pytest.approx(value_ratio)


def test_laspeyres_factor_reversal_fails() -> None:
    value_ratio = np.dot(P1, Q1) / np.dot(P0, Q0)
    assert laspeyres(P0, P1, Q0, Q1) * laspeyres(Q0, Q1, P0, P1) != pytest.approx(value_ratio)


@settings(max_examples=200)
@given(p0=vectors, p1=vectors, q0=vectors, q1=vectors)
def test_fisher_between_laspeyres_and_paasche(p0, p1, q0, q1) -> None:
    lo, hi = sorted([laspeyres(p0, p1, q0, q1), paasche(p0, p1, q0, q1)])
    assert lo * (1 - 1e-12) <= fisher(p0, p1, q0, q1) <= hi * (1 + 1e-12)


def test_weighted_quantile_by_hand() -> None:
    from pie.indexes import weighted_quantile

    x = np.array([3.0, 1.0, 2.0, 4.0])
    w = np.ones(4)
    assert weighted_quantile(x, w, 0.5) == 2.0
    assert weighted_quantile(x, w, 0.9) == 4.0
    assert weighted_quantile(x, np.array([0.0, 0.0, 0.0, 1.0]), 0.1) == 4.0

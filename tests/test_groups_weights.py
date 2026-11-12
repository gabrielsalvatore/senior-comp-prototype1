from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pie.aggregate import basket_effect
from pie.groups import income_quintile, label_groups
from pie.indexes import laspeyres_inflation
from pie.weights import group_mean_shares, plutocratic_weights


def households(n: int, **cols: object) -> pd.DataFrame:
    base: dict[str, object] = {
        "FINCBTXM": np.arange(1, n + 1) * 10_000.0,
        "FAM_SIZE": 1,
        "FINLWT21": 1.0,
        "BLS_URBN": 1,
        "FAM_TYPE": 1,
        "SEX_REF": 1,
        "AGE_REF": 40,
        "CUTENURE": 1,
    }
    base.update(cols)
    return pd.DataFrame(base, index=range(100, 100 + n))


def test_income_quintiles_split_evenly() -> None:
    """Regression test: a household exactly on a boundary used to move up a quintile."""
    q = income_quintile(households(10))
    assert q.value_counts().to_dict() == {1: 2, 2: 2, 3: 2, 4: 2, 5: 2}
    assert q.loc[100] == 1 and q.loc[109] == 5


def test_quintiles_use_equivalized_income() -> None:
    hh = households(2, FINCBTXM=[40_000.0, 40_000.0], FAM_SIZE=[4, 1])
    q = income_quintile(hh)
    assert q.loc[100] < q.loc[101]


def test_household_type_labels() -> None:
    hh = households(5, FAM_TYPE=[1, 7, 6, 8, 8], SEX_REF=[1, 2, 1, 2, 1])
    assert list(label_groups(hh)["Household type"]) == [
        "Married couple",
        "Single mother",
        "Single father",
        "Single woman",
        "Single man",
    ]


def test_area_age_and_tenure_labels() -> None:
    hh = households(3, BLS_URBN=[1, 2, 1], AGE_REF=[25, 50, 80], CUTENURE=[1, 2, 4])
    g = label_groups(hh)
    assert list(g["Area"]) == ["Urban", "Rural", "Urban"]
    assert list(g["Age of reference person"]) == ["Under 35", "35-64", "65+"]
    assert list(g["Housing tenure"]) == ["Owner, mortgage", "Owner, no mortgage", "Renter"]


def test_plutocratic_weights_favour_big_spenders() -> None:
    W = np.array([[1.0, 0.0], [0.0, 1.0]])
    got = plutocratic_weights(W, np.ones(2), np.array([100.0, 300.0]))
    np.testing.assert_allclose(got, [0.25, 0.75])


def test_basket_effect_by_hand() -> None:
    contrib = basket_effect(np.array([0.8, 0.2]), np.array([0.2, 0.8]), np.array([1.5, 1.0]))
    np.testing.assert_allclose(contrib, [0.3, 0.0])


def test_basket_effect_sums_to_gap(rng: np.random.Generator) -> None:
    W = rng.dirichlet(np.ones(5), size=40)
    rel = rng.uniform(0.95, 1.6, 5)
    w = rng.uniform(1, 10, 40)
    a = np.arange(40) < 15
    contrib = basket_effect(group_mean_shares(W, a, w), group_mean_shares(W, ~a, w), rel)
    pi = laspeyres_inflation(W, rel)
    gap = np.average(pi[a], weights=w[a]) - np.average(pi[~a], weights=w[~a])
    assert contrib.sum() == pytest.approx(gap)

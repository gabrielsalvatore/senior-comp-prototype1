"""Assign households to comparison groups."""

from __future__ import annotations

import numpy as np
import pandas as pd


def income_quintile(hh: pd.DataFrame) -> pd.Series:
    """Weighted quintile (1-5) of income adjusted for household size (income / sqrt(size))."""
    eq_income = hh["FINCBTXM"] / np.sqrt(hh["FAM_SIZE"].clip(lower=1))
    order = eq_income.sort_values().index
    cum = hh.loc[order, "FINLWT21"].cumsum() / hh["FINLWT21"].sum()
    # Quintile k holds households whose cumulative weight (including their own)
    # is within the first k/5 of the population. A household exactly on a
    # boundary (cum = 0.2) stays in the lower quintile; the tolerance absorbs
    # floating-point round-off in the cumulative sum.
    q = np.clip(np.ceil(cum * 5 - 1e-9), 1, 5).astype(int)
    return q.reindex(hh.index)


def quintile_label(q: int) -> str:
    return f"Q{q}" + (" (lowest)" if q == 1 else " (highest)" if q == 5 else "")


def label_groups(hh: pd.DataFrame) -> pd.DataFrame:
    """One column per grouping dimension; households outside every level get NaN/None."""
    g = pd.DataFrame(index=hh.index)
    g["Area"] = hh["BLS_URBN"].map({1: "Urban", 2: "Rural"})
    g["Income quintile"] = income_quintile(hh).map(quintile_label)
    ft, sex = hh["FAM_TYPE"], hh["SEX_REF"]
    g["Household type"] = np.select(
        [ft.eq(7), ft.eq(6), ft.eq(8) & sex.eq(2), ft.eq(8) & sex.eq(1), ft.isin([1, 2, 3, 4, 5])],
        ["Single mother", "Single father", "Single woman", "Single man", "Married couple"],
        default=None,
    )
    g["Age of reference person"] = pd.cut(
        hh["AGE_REF"], [0, 34, 64, 200], labels=["Under 35", "35-64", "65+"]
    ).astype(str)
    g["Housing tenure"] = hh["CUTENURE"].map(
        {1: "Owner, mortgage", 2: "Owner, no mortgage", 4: "Renter"}
    )
    return g

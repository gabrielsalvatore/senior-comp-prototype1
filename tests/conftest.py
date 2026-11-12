"""Shared fixtures: tiny synthetic CE files containing every required variable."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pie.config import REP_COLS
from pie.load import required_columns


def make_fmli(n: int, **overrides: object) -> pd.DataFrame:
    """n interview rows; every variable 0 except sensible defaults.

    Each row is its own household (NEWID = 10 * (k + 1) + 1) unless NEWID is
    overridden. Keyword arguments set whole columns (a scalar or a list).
    """
    df = pd.DataFrame(0.0, index=range(n), columns=required_columns())
    df["NEWID"] = [10 * (k + 1) + 1 for k in range(n)]
    df["FINLWT21"] = 1000.0
    df[REP_COLS] = 1000.0
    df["FAM_SIZE"] = 1
    df["AGE_REF"] = 40
    df["SEX_REF"] = 1
    df["FAM_TYPE"] = 8
    df["BLS_URBN"] = 1
    df["CUTENURE"] = 4
    df["REGION"] = 1
    df["FINCBTXM"] = 50_000.0
    df["FDHOMECQ"] = 100.0  # every household has some spending
    for name, value in overrides.items():
        df[name] = value
    return df


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)

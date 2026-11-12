from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pie.config import N_REP, REP_COLS
from pie.variance import brr_mean


def frame(main: list[float], reps: list[list[float]]) -> pd.DataFrame:
    df = pd.DataFrame({"FINLWT21": main})
    for col, w in zip(REP_COLS, reps, strict=True):
        df[col] = w
    return df


def test_replicate_columns() -> None:
    cols = REP_COLS
    assert len(cols) == N_REP == 44
    assert cols[0] == "WTREP01" and cols[-1] == "WTREP44"


def test_identical_replicates_give_zero_se() -> None:
    hh = frame([1.0, 3.0], [[1.0, 3.0]] * N_REP)
    est, se = brr_mean(np.array([0.10, 0.20]), hh, np.array([True, True]))
    assert est == pytest.approx(0.175)
    assert se == pytest.approx(0.0, abs=1e-9)


def test_half_samples_by_hand() -> None:
    """Values 0 and 1 with equal weight. Half the replicates keep only the first
    household (t_r = 0), half only the second (t_r = 1). Every (t_r - t)^2 is
    0.25, so var = 44 * 0.25 / 44 and se = 0.5."""
    reps = [[2.0, 0.0], [0.0, 2.0]] * (N_REP // 2)
    est, se = brr_mean(np.array([0.0, 1.0]), frame([1.0, 1.0], reps), np.array([True, True]))
    assert est == pytest.approx(0.5)
    assert se == pytest.approx(0.5)


def test_mask_restricts_to_group() -> None:
    hh = frame([1.0, 1.0, 1.0], [[1.0, 1.0, 1.0]] * N_REP)
    est, _ = brr_mean(np.array([0.1, 0.2, 0.9]), hh, np.array([True, True, False]))
    assert est == pytest.approx(0.15)

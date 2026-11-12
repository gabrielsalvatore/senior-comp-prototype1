"""Spending shares (the H x I weight matrix) and CPI-style aggregate weights."""

from __future__ import annotations

import numpy as np
import pandas as pd


def spending_shares(hh: pd.DataFrame, hh_spend: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Each household's share of spending by category.

    Households with zero total spending are dropped. Returns the matching
    household table (with TOTAL_SPEND added) and the shares, which sum to 1 per row.
    """
    total = hh_spend.sum(axis=1)
    ok = total > 0
    shares = hh_spend[ok].div(total[ok], axis=0)
    hh = hh.loc[shares.index].copy()
    hh["TOTAL_SPEND"] = total[ok]
    return hh, shares


def plutocratic_weights(
    W: np.ndarray, survey_weight: np.ndarray, total_spend: np.ndarray
) -> np.ndarray:
    """Aggregate expenditure shares, CPI style: households weighted by survey weight x spending."""
    agg = (W * (survey_weight * total_spend)[:, None]).sum(axis=0)
    return agg / agg.sum()


def group_mean_shares(W: np.ndarray, mask: np.ndarray, survey_weight: np.ndarray) -> np.ndarray:
    """Survey-weighted average spending shares for the households in mask."""
    return np.average(W[mask], axis=0, weights=survey_weight[mask])

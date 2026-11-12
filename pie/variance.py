"""Standard errors from the BLS balanced repeated replication (BRR) weights."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pie.config import N_REP, REP_COLS


def brr_mean(values: np.ndarray, hh: pd.DataFrame, mask: np.ndarray) -> tuple[float, float]:
    """Weighted mean and BRR standard error.

    BLS formula: var = sum_r (t_r - t)^2 / 44, where t_r is the estimate
    recomputed with replicate weight r.
    """
    v = values[mask]
    est = float(np.average(v, weights=hh.loc[mask, "FINLWT21"]))
    reps = np.array(
        [np.average(v, weights=hh.loc[mask, col].clip(lower=0) + 1e-12) for col in REP_COLS]
    )
    se = float(np.sqrt(((reps - est) ** 2).sum() / N_REP))
    return est, se


def brr_from_stacked(stacked: np.ndarray, n_groups: int) -> tuple[np.ndarray, np.ndarray]:
    """Estimates and BRR standard errors from stacked replicate estimates.

    stacked is (K*G) x S, as returned by aggregate_first / household_first with
    a replicate_averaging_matrix: rows 0..G-1 use the full-sample weight, rows
    k*G..(k+1)*G-1 use replicate weight k. Returns two G x S arrays.
    """
    k = stacked.shape[0] // n_groups
    if k * n_groups != stacked.shape[0] or k != N_REP + 1:
        raise ValueError(f"expected {(N_REP + 1) * n_groups} rows, got {stacked.shape[0]}")
    blocks = stacked.reshape(k, n_groups, -1)
    est = blocks[0]
    se = np.sqrt(((blocks[1:] - est) ** 2).sum(axis=0) / N_REP)
    return est, se

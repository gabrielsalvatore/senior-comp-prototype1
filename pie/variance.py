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

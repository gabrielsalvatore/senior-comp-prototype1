"""Generalized-mean indexes and the aggregate-first theorem (see docs/theory.md).

Every index here has the form P = phi(sum_i s_i g(r_i)) with phi = g^-1:

==========  =========  =========  ================================================
kind        g(r)       phi(x)     with base shares s0 / current shares s1
==========  =========  =========  ================================================
arithmetic  r          x          Laspeyres (s0)
harmonic    1/r        1/x        Paasche (s1); harmonic Laspeyres with s0
geometric   log r      exp(x)     geometric Laspeyres (s0); Törnqvist ((s0+s1)/2)
==========  =========  =========  ================================================

For a group with normalized weights omega_h and x_h = sum_i s_hi g(r_i):

* aggregate-share index  phi(sum_h omega_h x_h) = phi(S_g . g(r)), S_g = sum_h omega_h s_h
* democratic mean        sum_h omega_h phi(x_h)

The first is linear in the shares inside phi, so it can be computed aggregate
first. The second equals the first for every group only when phi is affine;
otherwise the difference is a Jensen gap, bounded below.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

Fn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Mean:
    """A quasi-arithmetic mean: g, its inverse phi, and phi''."""

    name: str
    g: Fn
    phi: Fn
    phi2: Fn  # second derivative of phi
    phi2_increasing: bool  # whether phi'' is increasing on the domain (for bounds)


MEANS: dict[str, Mean] = {
    "arithmetic": Mean("arithmetic", lambda r: r, lambda x: x, np.zeros_like, True),
    "harmonic": Mean("harmonic", lambda r: 1 / r, lambda x: 1 / x, lambda x: 2 / x**3, False),
    "geometric": Mean("geometric", np.log, np.exp, np.exp, True),
}


def household_index(W: np.ndarray, rel: np.ndarray, kind: str) -> np.ndarray:
    """phi(W @ g(rel)) for every household (index levels, not inflation rates)."""
    m = MEANS[kind]
    return m.phi(W @ m.g(rel))


def group_second_moments(W: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean shares S = sum omega_h s_h (I) and second moments M = sum omega_h s_h s_h^T (I x I).

    Var_omega(x_h) = g^T M g - (S . g)^2 for any vector g, so the Jensen gap's
    second-order term needs only these I + I^2 numbers per group, not the
    household-level x_h. Cost O(H * I^2) once per group.
    """
    omega = weights / weights.sum()
    S = omega @ W
    M = (W * omega[:, None]).T @ W
    return S, M


@dataclass
class GapRow:
    group: str
    n: int
    democratic_mean: float
    aggregate_share_index: float
    gap: float
    lower_bound: float
    upper_bound: float
    second_order: float
    second_order_from_moments: float


def jensen_gap(W: np.ndarray, rel: np.ndarray, weights: np.ndarray, kind: str) -> GapRow:
    """Democratic mean vs aggregate-share index for one group, with bounds.

    Taylor's theorem with Lagrange remainder gives
        (1/2) min phi'' Var(x) <= gap <= (1/2) max phi'' Var(x),
    min and max taken over [min x_h, max x_h]; second_order = (1/2) phi''(xbar) Var(x).
    """
    m = MEANS[kind]
    omega = weights / weights.sum()
    gr = m.g(rel)
    x = W @ gr
    xbar = float(omega @ x)
    var = float(omega @ (x - xbar) ** 2)
    democratic = float(omega @ m.phi(x))
    aggregate = float(m.phi(np.array(xbar)))
    lo_x, hi_x = float(x.min()), float(x.max())
    d2_lo, d2_hi = (float(m.phi2(np.array(v))) for v in (lo_x, hi_x))
    d2_min, d2_max = (d2_lo, d2_hi) if m.phi2_increasing else (d2_hi, d2_lo)
    S, M = group_second_moments(W, weights)
    var_moments = float(gr @ M @ gr - (S @ gr) ** 2)
    d2_bar = float(m.phi2(np.array(xbar)))
    return GapRow(
        group="",
        n=len(x),
        democratic_mean=democratic,
        aggregate_share_index=aggregate,
        gap=democratic - aggregate,
        lower_bound=0.5 * d2_min * var,
        upper_bound=0.5 * d2_max * var,
        second_order=0.5 * d2_bar * var,
        second_order_from_moments=0.5 * d2_bar * var_moments,
    )


def gap_table(
    W: np.ndarray, rel: np.ndarray, labels: pd.Series, weights: np.ndarray, kind: str
) -> pd.DataFrame:
    """jensen_gap for every group in labels (missing labels skipped)."""
    rows = []
    for level in sorted(labels.dropna().unique(), key=str):
        mask = (labels == level).to_numpy()
        row = jensen_gap(W[mask], rel, weights[mask], kind)
        row.group = str(level)
        rows.append(row.__dict__)
    return pd.DataFrame(rows).set_index("group")

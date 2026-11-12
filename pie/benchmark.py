"""Timing comparison of equivalent ways to compute household inflation."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

import numpy as np

from pie.aggregate import averaging_matrix

T = TypeVar("T")


def best_of(fn: Callable[[], T], n: int = 7) -> tuple[T, float]:
    """Run fn once to warm up, then n more times; return the result and the fastest time."""
    fn()
    best, res = float("inf"), None
    for _ in range(n):
        t = time.perf_counter()
        res = fn()
        best = min(best, time.perf_counter() - t)
    return res, best  # type: ignore[return-value]


def synthetic_relatives(n_items: int, n_months: int, seed: int = 0) -> np.ndarray:
    """I x T cumulative monthly price relatives (placeholder until real CPI series are used)."""
    rng = np.random.default_rng(seed)
    return 1 + rng.normal(0.005, 0.01, size=(n_items, n_months)).cumsum(axis=1)


def naive_loops(W: np.ndarray, R: np.ndarray) -> list[list[float]]:
    """A1: pure Python triple loop."""
    Wl, Rl = W.tolist(), R.tolist()
    n_items, n_months = len(Rl), len(Rl[0])
    return [
        [sum(Wl[h][i] * Rl[i][m] for i in range(n_items)) - 1 for m in range(n_months)]
        for h in range(len(Wl))
    ]


def run(W: np.ndarray, labels: np.ndarray, survey_weight: np.ndarray, n_months: int = 12) -> dict:
    """Time A1 (loops), A2 (matrix product) and A4 (aggregate first) on the same inputs."""
    R = synthetic_relatives(W.shape[1], n_months)

    t = time.perf_counter()
    out1 = naive_loops(W, R)
    a1 = time.perf_counter() - t

    out2, a2 = best_of(lambda: W @ R - 1)

    _, A = averaging_matrix(labels, survey_weight)
    g4, a4 = best_of(lambda: (A @ W) @ R - 1)
    g2 = A @ out2

    return {
        "a1": a1,
        "a2": a2,
        "a4": a4,
        "diff_a1_a2": float(np.abs(np.array(out1) - out2).max()),
        "diff_a4_a2": float(np.abs(g4 - g2).max()),
    }

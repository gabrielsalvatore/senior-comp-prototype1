"""Timing experiments for equivalent ways to compute household and group inflation.

Algorithms (H households, I items, T months, G groups, K weight columns):

=======  ===================================  ==================================
A1       pure Python triple loop              O(H*I*T), interpreted
A2       dense product W @ R                  O(H*I*T), BLAS
A3       household first, A @ (W @ R)         O(H*I*T + H*T)
A4d      aggregate first, dense A             O(G*H*I + G*I*T)
A4s      aggregate first, sparse A            O(H*I + G*I*T)
A4g      aggregate first, one BLAS GEMV/group O(H*I + G*I*T)
=======  ===================================  ==================================

A1 and A2 do the same arithmetic; their gap is a constant factor (interpreter
vs compiled BLAS), not a different algorithm. A3 vs A4 is a real algorithmic
choice: the matrix-chain order. With K = 45 replicate weights and S columns of
a sensitivity grid, the stacked averaging matrix B is (K*G) x H with K*H
nonzeros, and

* household first, B @ (W @ R):  O(H*I*S + K*H*S)
* aggregate first, (B @ W) @ R:  O(K*H*I + K*G*I*S)

so aggregate first wins once S exceeds roughly K*I / (I + K) (ignoring the
small K*G*I*S term). That count assumes every flop costs the same; a sparse
kernel runs far slower per flop than BLAS, which moves the measured crossover.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike

from pie.aggregate import (
    GroupedAggregator,
    aggregate_first,
    averaging_matrix,
    dense_averaging_matrix,
    household_first,
    replicate_averaging_matrix,
)

Inputs = dict[str, np.ndarray]


@dataclass
class Timing:
    algorithm: str
    H: int
    I: int  # noqa: E741  (I is the item count in the notation)
    T: int
    G: int
    K: int
    mean: float
    sd: float
    repeats: int

    def row(self) -> dict:
        return asdict(self)


def time_repeats(fn: Callable[[], object], repeats: int) -> np.ndarray:
    """Run fn once to warm up, then time `repeats` runs."""
    fn()
    out = np.empty(repeats)
    for r in range(repeats):
        t = time.perf_counter()
        fn()
        out[r] = time.perf_counter() - t
    return out


def naive_loops(W: np.ndarray, R: np.ndarray) -> list[list[float]]:
    """A1: pure Python triple loop."""
    Wl, Rl = W.tolist(), R.tolist()
    n_items, n_months = len(Rl), len(Rl[0])
    return [
        [sum(row[i] * Rl[i][m] for i in range(n_items)) - 1 for m in range(n_months)] for row in Wl
    ]


def synthetic_relatives(n_items: int, n_months: int, seed: int = 0) -> np.ndarray:
    """I x T random-walk price relatives (used when no real CPI series are given)."""
    rng = np.random.default_rng(seed)
    return 1 + rng.normal(0.003, 0.01, size=(n_items, n_months)).cumsum(axis=1)


def synthetic_shares(n_households: int, n_items: int, seed: int = 0) -> np.ndarray:
    """H x I spending shares drawn from a Dirichlet distribution."""
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.full(n_items, 0.5), size=n_households)


def synthetic_inputs(H: int, I: int, T: int, G: int, K: int = 1, seed: int = 0) -> Inputs:  # noqa: E741
    rng = np.random.default_rng(seed)
    return {
        "W": synthetic_shares(H, I, seed),
        "R": synthetic_relatives(I, T, seed),
        "labels": rng.integers(0, G, size=H),
        "weights": rng.uniform(500, 30_000, size=(H, K)),
    }


def single_weight_algorithms(inp: Inputs, include_loops: bool) -> dict[str, Callable[[], object]]:
    W, R, labels, w = inp["W"], inp["R"], inp["labels"], inp["weights"][:, 0]
    _, A_sparse = averaging_matrix(labels, w)
    _, A_dense = dense_averaging_matrix(labels, w)
    grouped = GroupedAggregator(labels, w)
    algs: dict[str, Callable[[], object]] = {
        "A2 dense W@R": lambda: W @ R - 1,
        "A3 A@(W@R) sparse": lambda: household_first(W, R, A_sparse) - 1,
        "A4d (A@W)@R dense A": lambda: aggregate_first(W, R, A_dense) - 1,
        "A4s (A@W)@R sparse A": lambda: aggregate_first(W, R, A_sparse) - 1,
        "A4g grouped BLAS": lambda: grouped.aggregate_first(W, R) - 1,
    }
    if include_loops:
        algs = {"A1 Python loops": lambda: naive_loops(W, R), **algs}
    return algs


def replicate_algorithms(inp: Inputs) -> dict[str, Callable[[], object]]:
    W, R = inp["W"], inp["R"]
    _, B = replicate_averaging_matrix(inp["labels"], inp["weights"])
    grouped = GroupedAggregator(inp["labels"], inp["weights"])
    return {
        "household first B@(W@R)": lambda: household_first(W, R, B),
        "aggregate first (B@W)@R sparse": lambda: aggregate_first(W, R, B),
        "aggregate first grouped BLAS": lambda: grouped.aggregate_first(W, R),
    }


def measure(
    algs: dict[str, Callable[[], object]], shape: dict[str, int], repeats: int
) -> list[Timing]:
    out = []
    for name, fn in algs.items():
        t = time_repeats(fn, repeats)
        sd = float(t.std(ddof=1)) if repeats > 1 else 0.0
        out.append(Timing(name, mean=float(t.mean()), sd=sd, repeats=repeats, **shape))
    return out


def sweep(
    vary: str,
    sizes: Sequence[int],
    base: dict[str, int],
    workload: str = "single",
    repeats: int = 5,
    loops_max_work: int = 2_000_000,
    relatives: np.ndarray | None = None,
) -> list[Timing]:
    """Time every algorithm while one dimension (H, I, T or K) varies.

    If real `relatives` (I x T) are given, I and T are cut from them instead of
    simulated; varying I or T then requires sizes within the real matrix.
    """
    rows: list[Timing] = []
    for size in sizes:
        shape = {**base, vary: int(size)}
        inp = synthetic_inputs(shape["H"], shape["I"], shape["T"], shape["G"], shape["K"])
        if relatives is not None:
            inp["R"] = relatives[: shape["I"], : shape["T"]]
        if workload == "single":
            work = shape["H"] * shape["I"] * shape["T"]
            algs = single_weight_algorithms(inp, include_loops=work <= loops_max_work)
        else:
            algs = replicate_algorithms(inp)
        rows.extend(measure(algs, shape, repeats))
    return rows


def loglog_slope(sizes: ArrayLike, times: ArrayLike) -> float:
    """Least-squares slope of log(time) on log(size): the empirical growth exponent."""
    slope, _ = np.polyfit(
        np.log(np.asarray(sizes, dtype=float)), np.log(np.asarray(times, dtype=float)), 1
    )
    return float(slope)


def crossover_columns(K: int, I: int) -> float:  # noqa: E741
    """Predicted S above which aggregate first beats household first (leading terms)."""
    return K * I / (I + K)


def run(W: np.ndarray, labels: np.ndarray, survey_weight: np.ndarray, n_months: int = 12) -> dict:
    """Quick comparison on the real CE weights, for the CLI report."""
    R = synthetic_relatives(W.shape[1], n_months)
    inp = {"W": W, "R": R, "labels": labels, "weights": survey_weight[:, None]}
    algs = single_weight_algorithms(inp, include_loops=True)
    timings = {
        t.algorithm: t
        for t in measure(
            algs,
            {"H": W.shape[0], "I": W.shape[1], "T": n_months, "G": len(set(labels)), "K": 1},
            repeats=7,
        )
    }
    out_loops = np.array(naive_loops(W, R))
    out_dense = W @ R - 1
    _, A = averaging_matrix(labels, survey_weight)
    return {
        "timings": timings,
        "diff_a1_a2": float(np.abs(out_loops - out_dense).max()),
        "diff_a4_a3": float(np.abs(aggregate_first(W, R, A) - household_first(W, R, A)).max()),
    }

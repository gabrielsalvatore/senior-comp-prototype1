"""Scaling experiments for the household-inflation algorithms.

Usage:
    python benchmarks/run_benchmarks.py                      # synthetic prices
    python benchmarks/run_benchmarks.py --cpi-series cu.data.0.Current
    python benchmarks/run_benchmarks.py --quick              # small grid, for a smoke test

Writes one CSV per experiment and a Markdown summary (mean ± sd, log-log slopes
against the predicted exponents) to benchmarks/results/.
"""

from __future__ import annotations

import argparse
import platform
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from pie import benchmark
from pie.cpi_series import complete_block, load_bls_series, relatives_matrix

RESULTS = Path(__file__).resolve().parent / "results"

# (name, varied dimension, sizes, fixed dimensions, workload, predicted exponents)
Experiment = tuple[str, str, list[int], dict[str, int], str, dict[str, int]]

SINGLE = [
    "A2 dense W@R",
    "A3 A@(W@R) sparse",
    "A4d (A@W)@R dense A",
    "A4s (A@W)@R sparse A",
    "A4g grouped BLAS",
]
REPLICATE = [
    "household first B@(W@R)",
    "aggregate first (B@W)@R sparse",
    "aggregate first grouped BLAS",
]


def exponents(names: list[str], *values: int) -> dict[str, int]:
    return dict(zip(names, values, strict=True))


# Predicted exponents follow the complexity table in pie.benchmark. For T and G
# the aggregate-first cost O(H*I + G*I*T) is dominated by H*I at these sizes,
# so its predicted exponent is 0.
FULL: list[Experiment] = [
    (
        "H_small",
        "H",
        [1_000, 2_000, 5_000, 10_000, 20_000],
        {"I": 16, "T": 12, "G": 5, "K": 1},
        "single",
        exponents(["A1 Python loops", *SINGLE], 1, 1, 1, 1, 1, 1),
    ),
    (
        "H",
        "H",
        [2_000, 5_000, 10_000, 20_000, 50_000],
        {"I": 250, "T": 120, "G": 5, "K": 1},
        "single",
        exponents(SINGLE, 1, 1, 1, 1, 1),
    ),
    (
        "I",
        "I",
        [16, 32, 64, 128, 256, 512],
        {"H": 10_000, "T": 120, "G": 5, "K": 1},
        "single",
        exponents(SINGLE, 1, 1, 1, 1, 1),
    ),
    (
        "T",
        "T",
        [12, 24, 48, 96, 192, 384],
        {"H": 10_000, "I": 250, "G": 5, "K": 1},
        "single",
        exponents(SINGLE, 1, 1, 0, 0, 0),
    ),
    (
        "G",
        "G",
        [2, 5, 20, 100, 500, 2_000],
        {"H": 10_000, "I": 250, "T": 120, "K": 1},
        "single",
        exponents(SINGLE, 0, 0, 1, 0, 0),
    ),
    (
        "replicates_I16",
        "T",
        [1, 2, 4, 8, 16, 32, 64, 128, 256],
        {"H": 10_000, "I": 16, "G": 5, "K": 45},
        "replicates",
        exponents(REPLICATE, 1, 0, 0),
    ),
    (
        "replicates_I250",
        "T",
        [1, 2, 4, 8, 16, 32, 64, 128, 256],
        {"H": 10_000, "I": 250, "G": 5, "K": 45},
        "replicates",
        exponents(REPLICATE, 1, 0, 0),
    ),
]


def quick_grid() -> list[Experiment]:
    out = []
    for name, vary, sizes, base, workload, pred in FULL:
        small = {**base, "H": min(base.get("H", 2_000), 2_000)}
        out.append((name, vary, sizes[:3], small, workload, pred))
    return out


def summarize(df: pd.DataFrame, vary: str, predicted: dict[str, int]) -> str:
    """Markdown: a mean ± sd table (ms) and the log-log slope of each algorithm."""
    table = df.assign(
        cell=lambda d: (
            (d["mean"] * 1e3).map("{:.3f}".format) + " ± " + (d["sd"] * 1e3).map("{:.3f}".format)
        )
    ).pivot(index=vary, columns="algorithm", values="cell")
    lines = [
        "| " + vary + " | " + " | ".join(table.columns) + " |",
        "|" + "---|" * (len(table.columns) + 1),
    ]
    for size, row in table.iterrows():
        lines.append(
            f"| {size:,} | " + " | ".join(str(v) if isinstance(v, str) else "—" for v in row) + " |"
        )
    lines += [
        "",
        "| Algorithm | Slope, all sizes | Slope, largest 3 | Predicted |",
        "|---|---|---|---|",
    ]
    for alg, grp in df.groupby("algorithm"):
        if len(grp) < 3:
            continue
        grp = grp.sort_values(vary)
        slope = benchmark.loglog_slope(grp[vary], grp["mean"])
        top = benchmark.loglog_slope(grp[vary].iloc[-3:], grp["mean"].iloc[-3:])
        lines.append(f"| {alg} | {slope:.2f} | {top:.2f} | {predicted.get(str(alg), '—')} |")
    return "\n".join(lines)


def crossover(df: pd.DataFrame) -> str:
    """First S at which each aggregate-first variant beats household first."""
    wide = df.pivot(index="T", columns="algorithm", values="mean")
    hf = wide["household first B@(W@R)"]
    parts = []
    for alg in [c for c in wide.columns if c.startswith("aggregate first")]:
        winners = wide.index[wide[alg] < hf]
        when = f"S = {int(winners.min())}" if len(winners) else "never on this grid"
        parts.append(f"{alg}: {when}")
    return "; ".join(parts)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--cpi-series", help="BLS CPI flat file (e.g. cu.data.0.Current)")
    p.add_argument("--start", default="2015-01", help="first month of real series")
    p.add_argument("--end", default="2024-12", help="last month of real series")
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", type=Path, default=RESULTS)
    args = p.parse_args(argv)

    relatives = None
    source = "synthetic random-walk price relatives"
    if args.cpi_series:
        levels = complete_block(load_bls_series(args.cpi_series), args.start, args.end)
        relatives = relatives_matrix(levels)
        source = (
            f"{relatives.shape[0]} real CPI series from {Path(args.cpi_series).name}, "
            f"{args.start} to {args.end} ({relatives.shape[1]} months)"
        )
        print(f"Using {source}")

    args.out.mkdir(parents=True, exist_ok=True)
    report = [
        "# Benchmark results",
        "",
        f"- Prices: {source}",
        "- Spending shares: synthetic (Dirichlet), so H can exceed the CE sample",
        f"- Repeats per point: {args.repeats} (after one warm-up run); times in ms, mean ± sd",
        f"- Machine: {platform.platform()}, Python {platform.python_version()}, "
        f"NumPy {np.__version__}",
        "",
    ]
    for name, vary, sizes, base, workload, predicted in quick_grid() if args.quick else FULL:
        if relatives is not None and vary in ("I", "T"):
            limit = relatives.shape[0] if vary == "I" else relatives.shape[1]
            sizes = [s for s in sizes if s <= limit]
        base = {k: v for k, v in base.items()}
        base.setdefault("H", sizes[0])
        base.setdefault("I", 16)
        base.setdefault("T", 12)
        if relatives is not None:
            base["I"] = min(base["I"], relatives.shape[0])
            base["T"] = min(base["T"], relatives.shape[1])
        print(f"{name}: varying {vary} over {sizes} with {base}", flush=True)
        rows = benchmark.sweep(vary, sizes, base, workload, args.repeats, relatives=relatives)
        df = pd.DataFrame([r.row() for r in rows])
        df.to_csv(args.out / f"{name}.csv", index=False)
        fixed = ", ".join(f"{k}={v:,}" for k, v in base.items() if k != vary)
        report += [f"## {name}: vary {vary} ({fixed})", "", summarize(df, vary, predicted), ""]
        if workload == "replicates":
            pred = benchmark.crossover_columns(base["K"], base["I"])
            report += [
                f"First S where aggregate first wins — {crossover(df)}. "
                f"Flop-count prediction: S ≈ K·I/(I+K) = {pred:.1f}.",
                "",
            ]

    (args.out / "summary.md").write_text("\n".join(report))
    print(f"Wrote {args.out / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

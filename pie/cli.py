"""Command-line report: python -m pie path/to/intrvw23 [--quick] [--cpi FILE]."""

from __future__ import annotations

import argparse
import sys
import time
import warnings

import numpy as np

from pie import benchmark
from pie.aggregate import basket_effect
from pie.config import CATEGORIES, DEFAULT_CPI_FILE, N_REP
from pie.cpi import headline_inflation, load_cpi, relatives
from pie.groups import label_groups
from pie.indexes import laspeyres_inflation, weighted_quantile
from pie.load import MissingCEVariableError, load_households
from pie.variance import brr_mean
from pie.weights import group_mean_shares, plutocratic_weights, spending_shares

# Apple's Accelerate BLAS raises spurious matmul warnings on M-series Macs.
warnings.filterwarnings("ignore", message=".*encountered in matmul")

PEAK = ("2021-06", "2022-06")  # peak inflation year
CUMULATIVE = ("2019-12", "2023-12")  # whole surge
MIN_GROUP_SIZE = 30


def hr(title: str = "") -> None:
    print("\n" + "=" * 72)
    if title:
        print(title)
        print("=" * 72)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Household inflation from CE Interview microdata.")
    p.add_argument("ce_dir", help="folder with the CE Interview files (e.g. intrvw23)")
    p.add_argument("--quick", action="store_true", help="skip the timing section")
    p.add_argument("--cpi", default=DEFAULT_CPI_FILE, help="CPI price-level CSV")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cpi = load_cpi(args.cpi)

    hr("PIE: household inflation from Consumer Expenditure Survey microdata")
    t0 = time.perf_counter()
    try:
        df, hh, hh_spend = load_households(args.ce_dir)
    except (FileNotFoundError, MissingCEVariableError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    hh, shares = spending_shares(hh, hh_spend)
    W = shares.to_numpy()  # H x I weight matrix
    print(f"  Loaded {len(df):,} household-quarter interviews -> {len(hh):,} households")
    print(
        f"  Spending weights: {W.shape[0]:,} households x {W.shape[1]} CPI categories "
        f"({time.perf_counter() - t0:.2f}s)"
    )

    rel_peak = relatives(cpi, CATEGORIES, *PEAK)
    rel_cum = relatives(cpi, CATEGORIES, *CUMULATIVE)
    cpi_peak = headline_inflation(cpi, *PEAK)
    cpi_cum = headline_inflation(cpi, *CUMULATIVE)
    pi_peak = laspeyres_inflation(W, rel_peak)
    pi_cum = laspeyres_inflation(W, rel_cum)
    w = hh["FINLWT21"].to_numpy()

    hr("1. Every household gets its own inflation rate (peak year, Jun 2021 -> Jun 2022)")
    print(f"  Official CPI-U:                        {cpi_peak:6.1%}")
    print(f"  Households, 10th percentile:           {weighted_quantile(pi_peak, w, 0.10):6.1%}")
    print(f"  Households, median:                    {weighted_quantile(pi_peak, w, 0.50):6.1%}")
    print(f"  Households, 90th percentile:           {weighted_quantile(pi_peak, w, 0.90):6.1%}")
    agg_w = plutocratic_weights(W, w, hh["TOTAL_SPEND"].to_numpy())
    print(
        f"  Rebuilt aggregate (CPI-style weights): {agg_w @ rel_peak - 1:6.1%}"
        "   <- sanity check vs CPI-U"
    )

    hr(f"2. Group comparison: mean household inflation, 95% CI from {N_REP} replicate weights")
    groups = label_groups(hh)
    print(f"  {'Group':<26}{'n':>6}{'Jun21-Jun22':>17}{'Dec19-Dec23':>18}")
    print(f"  {'Official CPI-U':<26}{'':>6}  {cpi_peak:6.2%}{'':>11}{cpi_cum:6.2%}")
    for dim in groups.columns:
        print(f"  -- {dim}")
        for level in sorted(groups[dim].dropna().unique(), key=str):
            if level in ("None", "nan"):
                continue
            mask = (groups[dim] == level).to_numpy()
            if mask.sum() < MIN_GROUP_SIZE:
                continue
            e1, s1 = brr_mean(pi_peak, hh, mask)
            e2, s2 = brr_mean(pi_cum, hh, mask)
            print(
                f"  {level:<26}{mask.sum():>6,}  {e1:6.2%} ±{196 * s1:4.2f}pp"
                f"   {e2:6.2%} ±{196 * s2:4.2f}pp"
            )

    hr("3. What drives the rural-urban gap? (basket effect by category, peak year)")
    area = groups["Area"]
    if area.notna().any():
        rural, urban = (area == "Rural").to_numpy(), (area == "Urban").to_numpy()
        wr, wu = group_mean_shares(W, rural, w), group_mean_shares(W, urban, w)
        contrib = basket_effect(wr, wu, rel_peak)
        print(
            f"  Rural minus urban (basket effect): {contrib.sum() * 100:+.2f} pp."
            " Largest contributions:"
        )
        for i in np.argsort(-np.abs(contrib))[:5]:
            print(
                f"    {CATEGORIES[i]:<24} share {wr[i]:5.1%} vs {wu[i]:5.1%}   "
                f"price {rel_peak[i] - 1:+6.1%}   -> {contrib[i] * 100:+.2f} pp"
            )

    if not args.quick:
        hr("4. Same numbers, three algorithms (H households x 16 categories x T months)")
        r = benchmark.run(W, groups["Income quintile"].to_numpy(), w)
        print(f"  A1 naive Python loops      {r['a1'] * 1000:9.1f} ms")
        print(
            f"  A2 dense matrix product    {r['a2'] * 1000:9.2f} ms"
            f"   ({r['a1'] / r['a2']:,.0f}x faster)"
        )
        print(f"  A4 aggregate first (G=5)   {r['a4'] * 1000:9.2f} ms   (group results only)")
        print(f"  Max difference A1 vs A2:   {r['diff_a1_a2']:.1e}")
        print(
            f"  Max difference A4 vs A2:   {r['diff_a4_a2']:.1e}   (identical: (A W) R = A (W R))"
        )

    hr()
    print("  Prototype: 2023 spending shares, 16 categories, national prices.")
    print("  Next: 2016/2019 base years, ~250 CPI items via the BLS concordance,")
    print("  regional prices, regression with replicate-weight standard errors.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

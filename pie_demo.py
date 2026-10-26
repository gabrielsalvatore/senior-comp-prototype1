"""PIE prototype demo: household-level inflation from CE microdata.

Usage:
    python3 pie_demo.py path/to/intrvw23            (full demo)
    python3 pie_demo.py path/to/intrvw23 --quick    (skip the timing section)

What it does (preliminary version for the Week 9 presentation):
  1. Loads the CE Interview household files (FMLI) for one survey year.
  2. Builds each household's spending shares over 16 CPI categories.
  3. Applies official CPI-U category price changes to get each household's
     own inflation rate (Laspeyres form: sum_i w_hi * relative_i - 1).
  4. Compares groups, with 95% confidence intervals from the 44 BLS
     replicate weights (balanced repeated replication).
  5. Times three algorithms that compute the same numbers.

Prototype limitations (stated on purpose):
  - Spending shares come from the CE year loaded (e.g. 2023), not a
    pre-period base year. The thesis uses 2016 and 2019 base years.
  - 16 broad categories built from CE summary variables. The thesis maps
    detailed UCC codes to a few hundred CPI items via the BLS concordance.
  - National CPI-U price indexes for everyone; regional prices come later.
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Apple's Accelerate BLAS raises spurious matmul warnings on M-series Macs.
warnings.filterwarnings("ignore", message=".*encountered in matmul")

# ---------------------------------------------------------------------------
# CPI-U, U.S. city average, not seasonally adjusted (BLS via FRED).
# Values: Dec 2019, Jun 2021, Jun 2022, Dec 2023.
# ---------------------------------------------------------------------------
CPI = {
    "All items":            ("CUUR0000SA0",    256.974, 271.696, 296.311, 306.746),
    "Food at home":         ("CUUR0000SAF11",  241.750, 257.412, 288.884, 303.005),
    "Food away from home":  ("CUUR0000SEFV",   288.078, 305.634, 329.033, 361.564),
    "Alcohol":              ("CUUR0000SAF116", 252.361, 262.913, 273.537, 287.873),
    "Shelter":              ("CUUR0000SAH1",   321.766, 334.222, 352.967, 389.433),
    "Utilities":            ("CUUR0000SAH2",   241.788, 260.400, 306.293, 306.768),
    "Household furnishings": ("CUUR0000SAH3",  123.462, 131.131, 143.586, 146.600),
    "Apparel":              ("CUUR0000SAA",    119.111, 120.324, 126.619, 125.794),
    "Vehicles":             ("CUUR0000SETA",    98.495, 118.941, 130.450, 125.975),
    "Gasoline":             ("CUUR0000SETB01", 228.857, 268.979, 430.142, 280.289),
    "Transport services":   ("CUUR0000SAS4",   324.209, 335.548, 365.178, 409.749),
    "Medical care":         ("CUUR0000SAM",    509.689, 522.989, 546.717, 553.485),
    "Recreation":           ("CUUR0000SAR",    121.545, 124.893, 130.677, 136.811),
    "Education":            ("CUUR0000SAE1",   268.700, 273.169, 280.562, 294.040),
    "Personal care":        ("CUUR0000SAG1",   235.310, 241.907, 257.287, 277.089),
    "Tobacco":              ("CUUR0000SEGA",    71.984,  78.667,  84.876,  93.757),
    "Other goods/services": ("CUUR0000SAG",    455.413, 473.718, 505.312, 546.784),
}

# CE summary variables (current quarter CQ + previous quarter PQ) per category.
CE_VARS = {
    "Food at home":          ["FDHOME"],
    "Food away from home":   ["FDAWAY"],
    "Alcohol":               ["ALCBEV"],
    # Shelter is built separately (renters: rent paid; owners: rental equivalence)
    "Utilities":             ["UTIL"],
    "Household furnishings": ["HOUSOP", "HOUSEQ"],
    "Apparel":               ["APPAR"],
    "Vehicles":              ["CARTKN", "CARTKU", "OTHVEH"],
    "Gasoline":              ["GASMO"],
    "Transport services":    ["VEHFIN", "MAINRP", "VEHINS", "VRNTLO", "PUBTRA"],
    "Medical care":          ["HEALTH"],
    "Recreation":            ["ENTERT", "READ"],
    "Education":             ["EDUCA"],
    "Personal care":         ["PERSCA"],
    "Tobacco":               ["TOBACC"],
    "Other goods/services":  ["MISC"],
}
CATEGORIES = ["Food at home", "Food away from home", "Alcohol", "Shelter",
              "Utilities", "Household furnishings", "Apparel", "Vehicles",
              "Gasoline", "Transport services", "Medical care", "Recreation",
              "Education", "Personal care", "Tobacco", "Other goods/services"]
N_REP = 44


def hr(title=""):
    print("\n" + "=" * 72)
    if title:
        print(title)
        print("=" * 72)


def col(df, name):
    return df[name].fillna(0).clip(lower=0) if name in df.columns else 0.0


def load_households(root):
    files = sorted(Path(root).rglob("fmli*.csv"))
    if not files:
        sys.exit(f"No fmli*.csv files under {root}")
    df = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    df.columns = [c.upper() for c in df.columns]

    missing = []
    spend = pd.DataFrame(index=df.index)
    for cat, stems in CE_VARS.items():
        total = 0.0
        for s in stems:
            for suffix in ("CQ", "PQ"):
                if f"{s}{suffix}" not in df.columns:
                    missing.append(f"{s}{suffix}")
                total = total + col(df, f"{s}{suffix}")
        spend[cat] = total

    # Shelter, valued the way the CPI does: rent for renters, rental
    # equivalence (RENTEQVX, monthly) for owners, plus other lodging.
    owner = df["CUTENURE"].isin([1, 2, 3]) if "CUTENURE" in df.columns else False
    rent = col(df, "RENDWECQ") + col(df, "RENDWEPQ")
    other_lodging = col(df, "OTHLODCQ") + col(df, "OTHLODPQ")
    own_outlays = col(df, "OWNDWECQ") + col(df, "OWNDWEPQ")
    rent_eq = 3 * col(df, "RENTEQVX")              # one interview covers 3 months
    owner_cost = np.where(rent_eq > 0, rent_eq, own_outlays)
    spend["Shelter"] = np.where(owner, owner_cost, rent) + other_lodging
    spend = spend[CATEGORIES]

    if missing:
        print(f"  note: {len(set(missing))} expected CE variables not found, treated as 0: "
              f"{sorted(set(missing))[:8]}{' ...' if len(set(missing)) > 8 else ''}")

    df = df.copy()
    df["CUID"] = df["NEWID"] // 10
    spend["CUID"] = df["CUID"]
    hh_spend = spend.groupby("CUID").sum()

    rep_cols = [f"WTREP{i:02d}" for i in range(1, N_REP + 1)]
    keep = ["FINLWT21"] + rep_cols + ["FINCBTXM"]
    df[rep_cols] = df[rep_cols].fillna(0)   # blank = not in that BRR half-sample
    agg = df.groupby("CUID")[keep].mean()
    first = df.sort_values("NEWID").groupby("CUID")[
        ["FAM_SIZE", "AGE_REF", "SEX_REF", "FAM_TYPE", "BLS_URBN", "CUTENURE", "REGION"]
    ].last()
    hh = agg.join(first)
    total = hh_spend.sum(axis=1)
    ok = total > 0
    shares = hh_spend[ok].div(total[ok], axis=0)
    hh = hh.loc[shares.index]
    hh["TOTAL_SPEND"] = total[ok]
    return df, hh, shares


def label_groups(hh):
    g = pd.DataFrame(index=hh.index)
    g["Area"] = hh["BLS_URBN"].map({1: "Urban", 2: "Rural"})
    eq_income = hh["FINCBTXM"] / np.sqrt(hh["FAM_SIZE"].clip(lower=1))
    # weighted quintile cut points
    order = eq_income.sort_values().index
    cum = hh.loc[order, "FINLWT21"].cumsum() / hh["FINLWT21"].sum()
    q = np.minimum((cum * 5).astype(int) + 1, 5)
    g["Income quintile"] = q.reindex(hh.index).map(lambda x: f"Q{x}" + (" (lowest)" if x == 1 else " (highest)" if x == 5 else ""))
    ft = hh["FAM_TYPE"]
    sex = hh["SEX_REF"]
    g["Household type"] = np.select(
        [ft.eq(7), ft.eq(6), ft.eq(8) & sex.eq(2), ft.eq(8) & sex.eq(1), ft.isin([1, 2, 3, 4, 5])],
        ["Single mother", "Single father", "Single woman", "Single man", "Married couple"],
        default=None)
    g["Age of reference person"] = pd.cut(hh["AGE_REF"], [0, 34, 64, 200],
                                          labels=["Under 35", "35-64", "65+"]).astype(str)
    g["Housing tenure"] = hh["CUTENURE"].map({1: "Owner, mortgage", 2: "Owner, no mortgage",
                                              4: "Renter"})
    return g


def brr_mean(values, hh, mask):
    """Weighted mean and BRR standard error (BLS formula: var = sum((t_r - t)^2) / 44)."""
    w = hh.loc[mask, "FINLWT21"]
    v = values[mask]
    est = np.average(v, weights=w)
    reps = np.array([np.average(v, weights=hh.loc[mask, f"WTREP{r:02d}"].clip(lower=0) + 1e-12)
                     for r in range(1, N_REP + 1)])
    se = np.sqrt(((reps - est) ** 2).sum() / N_REP)
    return est, se


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    quick = "--quick" in sys.argv

    hr("PIE: household inflation from Consumer Expenditure Survey microdata")
    t0 = time.perf_counter()
    df, hh, shares = load_households(sys.argv[1])
    W = shares.to_numpy()                                  # H x I weight matrix
    print(f"  Loaded {len(df):,} household-quarter interviews -> {len(hh):,} households")
    print(f"  Spending weights: {W.shape[0]:,} households x {W.shape[1]} CPI categories "
          f"({time.perf_counter() - t0:.2f}s)")

    # price relatives per category for two windows
    rel_peak = np.array([CPI[c][3] / CPI[c][2] for c in CATEGORIES])   # Jun21 -> Jun22
    rel_cum = np.array([CPI[c][4] / CPI[c][1] for c in CATEGORIES])    # Dec19 -> Dec23
    cpi_peak = CPI["All items"][3] / CPI["All items"][2] - 1
    cpi_cum = CPI["All items"][4] / CPI["All items"][1] - 1

    pi_peak = W @ rel_peak - 1
    pi_cum = W @ rel_cum - 1

    hr("1. Every household gets its own inflation rate (peak year, Jun 2021 -> Jun 2022)")
    w = hh["FINLWT21"].to_numpy()
    def wq(x, p):
        o = np.argsort(x); c = np.cumsum(w[o]) / w.sum()
        return x[o][np.searchsorted(c, p)]
    print(f"  Official CPI-U:                        {cpi_peak:6.1%}")
    print(f"  Households, 10th percentile:           {wq(pi_peak, .10):6.1%}")
    print(f"  Households, median:                    {wq(pi_peak, .50):6.1%}")
    print(f"  Households, 90th percentile:           {wq(pi_peak, .90):6.1%}")
    agg_w = (W * (w * hh["TOTAL_SPEND"].to_numpy())[:, None]).sum(0)
    agg_w = agg_w / agg_w.sum()
    print(f"  Rebuilt aggregate (CPI-style weights): {agg_w @ rel_peak - 1:6.1%}   <- sanity check vs CPI-U")

    hr("2. Group comparison: mean household inflation, 95% CI from 44 replicate weights")
    groups = label_groups(hh)
    print(f"  {'Group':<26}{'n':>6}{'Jun21-Jun22':>17}{'Dec19-Dec23':>18}")
    print(f"  {'Official CPI-U':<26}{'':>6}  {cpi_peak:6.2%}{'':>11}{cpi_cum:6.2%}")
    for dim in groups.columns:
        print(f"  -- {dim}")
        for level in sorted(groups[dim].dropna().unique(), key=str):
            if level in ("None", "nan"):
                continue
            mask = (groups[dim] == level).to_numpy()
            if mask.sum() < 30:
                continue
            e1, s1 = brr_mean(pi_peak, hh, mask)
            e2, s2 = brr_mean(pi_cum, hh, mask)
            print(f"  {level:<26}{mask.sum():>6,}  {e1:6.2%} ±{196*s1:4.2f}pp   {e2:6.2%} ±{196*s2:4.2f}pp")

    hr("3. What drives the rural-urban gap? (basket effect by category, peak year)")
    gm = groups["Area"]
    if gm.notna().any():
        wr = np.average(W[(gm == "Rural").to_numpy()], axis=0, weights=w[(gm == "Rural").to_numpy()])
        wu = np.average(W[(gm == "Urban").to_numpy()], axis=0, weights=w[(gm == "Urban").to_numpy()])
        contrib = (wr - wu) * (rel_peak - 1)
        print(f"  Rural minus urban (basket effect): {contrib.sum()*100:+.2f} pp. Largest contributions:")
        for i in np.argsort(-np.abs(contrib))[:5]:
            print(f"    {CATEGORIES[i]:<24} share {wr[i]:5.1%} vs {wu[i]:5.1%}   "
                  f"price {rel_peak[i]-1:+6.1%}   -> {contrib[i]*100:+.2f} pp")

    if not quick:
        hr("4. Same numbers, three algorithms (H households x 16 categories x T months)")
        rng = np.random.default_rng(0)
        T = 12
        R = 1 + rng.normal(0.005, 0.01, size=(W.shape[1], T)).cumsum(axis=1)   # monthly relatives
        # A1 naive loops
        t = time.perf_counter()
        Wl, Rl = W.tolist(), R.tolist()
        out1 = [[sum(Wl[h][i] * Rl[i][m] for i in range(len(Rl))) - 1 for m in range(T)]
                for h in range(len(Wl))]
        a1 = time.perf_counter() - t
        # A2 dense matrix product
        def best_of(fn, n=7):                      # warm up once, keep the fastest run
            fn(); best = float("inf")
            for _ in range(n):
                t = time.perf_counter(); res = fn(); best = min(best, time.perf_counter() - t)
            return res, best
        out2, a2 = best_of(lambda: W @ R - 1)
        # A4 aggregate first (group means), vs aggregating A2's output
        q = groups["Income quintile"].to_numpy()
        levels = sorted(set(q))
        A = np.array([(q == l) * w for l in levels], dtype=float)
        A = A / A.sum(axis=1, keepdims=True)                    # G x H averaging matrix
        g4, a4 = best_of(lambda: (A @ W) @ R - 1)
        g2 = A @ out2
        print(f"  A1 naive Python loops      {a1*1000:9.1f} ms")
        print(f"  A2 dense matrix product    {a2*1000:9.2f} ms   ({a1/a2:,.0f}x faster)")
        print(f"  A4 aggregate first (G=5)   {a4*1000:9.2f} ms   (group results only)")
        print(f"  Max difference A1 vs A2:   {np.abs(np.array(out1) - out2).max():.1e}")
        print(f"  Max difference A4 vs A2:   {np.abs(g4 - g2).max():.1e}   (identical: (A W) R = A (W R))")

    hr()
    print("  Prototype: 2023 spending shares, 16 categories, national prices.")
    print("  Next: 2016/2019 base years, ~250 CPI items via the BLS concordance,")
    print("  regional prices, regression with replicate-weight standard errors.")


if __name__ == "__main__":
    main()

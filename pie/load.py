"""Load CE Interview household files (FMLI) and build quarterly spending.

Missing CE *columns* raise an error instead of being silently treated as 0:
a renamed or dropped variable would otherwise bias every household's shares.
Blank *values* inside a column that exists are treated as zero spending, which
is how the CE public-use files record "no expenditure" for summary variables.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pie.config import (
    CATEGORIES,
    CE_VARS,
    DEMOGRAPHIC_VARS,
    ID_AND_WEIGHT_VARS,
    OWNER_TENURE_CODES,
    QUARTER_SUFFIXES,
    REP_COLS,
    SHELTER_VARS,
)


class MissingCEVariableError(KeyError):
    """Raised when the FMLI files lack variables the pipeline depends on."""

    def __str__(self) -> str:  # KeyError would otherwise wrap the message in quotes
        return str(self.args[0])


def required_columns() -> list[str]:
    """Every FMLI column the pipeline reads."""
    spend = [f"{s}{q}" for stems in CE_VARS.values() for s in stems for q in QUARTER_SUFFIXES]
    return sorted(set(spend + SHELTER_VARS + REP_COLS + DEMOGRAPHIC_VARS + ID_AND_WEIGHT_VARS))


def check_columns(df: pd.DataFrame) -> None:
    """Raise MissingCEVariableError listing every required column that is absent."""
    missing = [c for c in required_columns() if c not in df.columns]
    if missing:
        raise MissingCEVariableError(
            f"{len(missing)} required CE variable(s) not found in the FMLI files: {missing}"
        )


def read_fmli(root: str | Path) -> pd.DataFrame:
    """Read and stack every fmli*.csv under root, with upper-case column names."""
    files = sorted(Path(root).rglob("fmli*.csv"))
    if not files:
        raise FileNotFoundError(f"No fmli*.csv files under {root}")
    df = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    df.columns = [c.upper() for c in df.columns]
    return df


def spending_value(df: pd.DataFrame, name: str) -> pd.Series:
    """One spending column: blanks become 0 and negative values (refunds) are clipped to 0."""
    return df[name].fillna(0).clip(lower=0)


def category_spending(df: pd.DataFrame) -> pd.DataFrame:
    """Interview-level spending for each of the 16 categories (columns in CATEGORIES order)."""
    check_columns(df)
    spend = pd.DataFrame(index=df.index)
    for cat, stems in CE_VARS.items():
        spend[cat] = sum(spending_value(df, f"{s}{q}") for s in stems for q in QUARTER_SUFFIXES)
    spend["Shelter"] = shelter_spending(df)
    return spend[CATEGORIES]


def shelter_spending(df: pd.DataFrame) -> np.ndarray:
    """Shelter valued as the CPI does it: rent for renters, rental equivalence for owners.

    RENTEQVX is a monthly figure and one interview covers three months. Owners
    with no reported rental equivalence fall back to their owner outlays.
    """
    owner = df["CUTENURE"].isin(OWNER_TENURE_CODES)
    rent = spending_value(df, "RENDWECQ") + spending_value(df, "RENDWEPQ")
    other_lodging = spending_value(df, "OTHLODCQ") + spending_value(df, "OTHLODPQ")
    own_outlays = spending_value(df, "OWNDWECQ") + spending_value(df, "OWNDWEPQ")
    rent_eq = 3 * spending_value(df, "RENTEQVX")
    owner_cost = np.where(rent_eq > 0, rent_eq, own_outlays)
    return np.where(owner, owner_cost, rent) + other_lodging


def household_table(df: pd.DataFrame, spend: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse interviews to households (consumer units).

    Returns (households, spending) indexed by CUID. Weights are averaged over a
    household's interviews; demographics come from its last interview.
    """
    df = df.copy()
    df["CUID"] = df["NEWID"] // 10
    spend = spend.assign(CUID=df["CUID"])
    hh_spend = spend.groupby("CUID").sum()

    df[REP_COLS] = df[REP_COLS].fillna(0)  # blank = not in that BRR half-sample
    weights = df.groupby("CUID")[["FINLWT21", *REP_COLS, "FINCBTXM"]].mean()
    demographics = df.sort_values("NEWID").groupby("CUID")[DEMOGRAPHIC_VARS].last()
    return weights.join(demographics), hh_spend


def load_households(root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Full loading step: (interviews, households, spending by household and category)."""
    df = read_fmli(root)
    hh, hh_spend = household_table(df, category_spending(df))
    return df, hh, hh_spend

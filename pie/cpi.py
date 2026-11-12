"""Load CPI price levels from a file and turn them into price relatives."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pie.config import DEFAULT_CPI_FILE


def load_cpi(path: str | Path = DEFAULT_CPI_FILE) -> pd.DataFrame:
    """Read a CPI table: one row per category, one column per date (YYYY-MM).

    Raises ValueError if the file has no category column or no date columns.
    """
    cpi = pd.read_csv(path)
    if "category" not in cpi.columns:
        raise ValueError(f"{path}: missing required 'category' column")
    cpi = cpi.set_index("category")
    if not date_columns(cpi):
        raise ValueError(f"{path}: no date columns (expected names like 2021-06)")
    return cpi


def date_columns(cpi: pd.DataFrame) -> list[str]:
    """Columns that look like YYYY-MM dates."""
    return [c for c in cpi.columns if len(c) == 7 and c[4] == "-" and c.replace("-", "").isdigit()]


def relatives(cpi: pd.DataFrame, categories: list[str], start: str, end: str) -> np.ndarray:
    """Price relative p_end / p_start for each category, in the given order.

    Raises KeyError naming any category or date not in the CPI table.
    """
    missing_cats = [c for c in categories if c not in cpi.index]
    if missing_cats:
        raise KeyError(f"CPI table has no rows for: {missing_cats}")
    for d in (start, end):
        if d not in cpi.columns:
            raise KeyError(f"CPI table has no column for date {d}")
    sub = cpi.loc[categories]
    return (sub[end] / sub[start]).to_numpy(dtype=float)


def headline_inflation(cpi: pd.DataFrame, start: str, end: str, row: str = "All items") -> float:
    """Official inflation over the window, from the all-items row."""
    return float(cpi.loc[row, end] / cpi.loc[row, start] - 1)

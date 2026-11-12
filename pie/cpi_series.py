"""Read monthly CPI series from a BLS time-series flat file (e.g. cu.data.0.Current).

The BLS files are tab separated with columns series_id, year, period, value,
footnote_codes, padded with spaces. Monthly periods are M01-M12; M13 is the
annual average and is dropped. The benchmark uses these series to get price
relatives at realistic sizes (hundreds of items, many months).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd


def load_bls_series(path: str | Path, series_ids: Iterable[str] | None = None) -> pd.DataFrame:
    """Monthly index levels: one row per series, one column per month ("YYYY-MM")."""
    raw = pd.read_csv(path, sep="\t", dtype=str)
    raw.columns = [c.strip() for c in raw.columns]
    needed = ["series_id", "year", "period", "value"]
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        raise ValueError(f"{path}: not a BLS time-series file (missing {missing})")
    raw = raw[needed].apply(lambda col: col.str.strip())
    raw = raw[raw["period"].str.fullmatch(r"M(0[1-9]|1[0-2])")]
    if series_ids is not None:
        raw = raw[raw["series_id"].isin(set(series_ids))]
    raw = raw.assign(
        month=raw["year"] + "-" + raw["period"].str[1:],
        value=pd.to_numeric(raw["value"], errors="coerce"),
    )
    return raw.pivot(index="series_id", columns="month", values="value").sort_index(axis=1)


def complete_block(levels: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Series with a value in every month from start to end (inclusive)."""
    cols = [c for c in levels.columns if start <= c <= end]
    block = levels[cols]
    return block[block.notna().all(axis=1)]


def relatives_matrix(levels: pd.DataFrame) -> np.ndarray:
    """I x T price relatives, each series divided by its first month."""
    values = levels.to_numpy(dtype=float)
    return values / values[:, :1]

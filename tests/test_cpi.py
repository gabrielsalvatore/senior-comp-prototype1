from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pie.config import CATEGORIES
from pie.cpi import headline_inflation, load_cpi, relatives


def write_cpi(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def test_bundled_file_covers_every_category() -> None:
    cpi = load_cpi()
    assert set(CATEGORIES) | {"All items"} <= set(cpi.index)


def test_bundled_headline_inflation() -> None:
    # 296.311 / 271.696 - 1, the published Jun 2021 -> Jun 2022 CPI-U change
    assert headline_inflation(load_cpi(), "2021-06", "2022-06") == pytest.approx(0.0906, abs=5e-5)


def test_relatives_by_hand(tmp_path: Path) -> None:
    path = write_cpi(tmp_path / "c.csv", "category,2020-01,2021-01\nA,100,110\nB,50,45\n")
    np.testing.assert_allclose(
        relatives(load_cpi(path), ["B", "A"], "2020-01", "2021-01"), [0.9, 1.1]
    )


def test_unknown_category_raises() -> None:
    with pytest.raises(KeyError, match="Bananas"):
        relatives(load_cpi(), ["Bananas"], "2021-06", "2022-06")


def test_unknown_date_raises() -> None:
    with pytest.raises(KeyError, match="1999-01"):
        relatives(load_cpi(), CATEGORIES, "1999-01", "2022-06")


def test_gap_raises(tmp_path: Path) -> None:
    path = write_cpi(tmp_path / "c.csv", "category,2020-01,2021-01\nA,100,110\nB,50,\n")
    with pytest.raises(ValueError, match="B"):
        relatives(load_cpi(path), ["A", "B"], "2020-01", "2021-01")


def test_duplicate_category_raises(tmp_path: Path) -> None:
    path = write_cpi(tmp_path / "c.csv", "category,2020-01\nA,100\nA,101\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_cpi(path)


def test_missing_category_column_raises(tmp_path: Path) -> None:
    path = write_cpi(tmp_path / "c.csv", "name,2020-01\nA,100\n")
    with pytest.raises(ValueError, match="category"):
        load_cpi(path)


def test_no_date_columns_raises(tmp_path: Path) -> None:
    path = write_cpi(tmp_path / "c.csv", "category,series_id\nA,X\n")
    with pytest.raises(ValueError, match="date"):
        load_cpi(path)

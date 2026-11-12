from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pie.config import CATEGORIES
from pie.load import (
    MissingCEVariableError,
    category_spending,
    household_table,
    load_households,
    read_fmli,
)
from pie.weights import spending_shares

from conftest import make_fmli


def households(df):
    hh, hh_spend = household_table(df, category_spending(df))
    return spending_shares(hh, hh_spend)


def test_missing_variables_raise_with_their_names() -> None:
    df = make_fmli(3).drop(columns=["GASMOCQ", "RENTEQVX"])
    with pytest.raises(MissingCEVariableError) as info:
        category_spending(df)
    assert "GASMOCQ" in str(info.value) and "RENTEQVX" in str(info.value)


def test_categories_sum_both_quarters_and_all_stems() -> None:
    df = make_fmli(1, FDHOMECQ=0.0, CARTKNCQ=100.0, CARTKUPQ=50.0, OTHVEHCQ=25.0)
    assert category_spending(df).loc[0, "Vehicles"] == 175.0


def test_blank_values_are_zero_and_refunds_clipped() -> None:
    df = make_fmli(1, FDHOMECQ=float("nan"), FDHOMEPQ=-40.0, FDAWAYCQ=10.0)
    spend = category_spending(df)
    assert spend.loc[0, "Food at home"] == 0.0
    assert spend.loc[0, "Food away from home"] == 10.0


def test_shelter_renter_uses_rent() -> None:
    df = make_fmli(1, CUTENURE=4, RENDWECQ=900.0, RENDWEPQ=300.0, RENTEQVX=5000.0)
    assert category_spending(df).loc[0, "Shelter"] == 1200.0


def test_shelter_owner_uses_three_months_of_rental_equivalence() -> None:
    df = make_fmli(1, CUTENURE=1, RENTEQVX=1500.0, OWNDWECQ=9999.0, RENDWECQ=9999.0)
    assert category_spending(df).loc[0, "Shelter"] == 4500.0


def test_shelter_owner_without_rental_equivalence_uses_outlays() -> None:
    df = make_fmli(1, CUTENURE=2, RENTEQVX=0.0, OWNDWECQ=700.0, OWNDWEPQ=100.0)
    assert category_spending(df).loc[0, "Shelter"] == 800.0


def test_other_lodging_added_for_everyone() -> None:
    df = make_fmli(2, CUTENURE=[1, 4], RENTEQVX=[1000.0, 0.0], RENDWECQ=[0.0, 500.0])
    df[["OTHLODCQ", "OTHLODPQ"]] = 60.0
    np.testing.assert_allclose(category_spending(df)["Shelter"], [3120.0, 620.0])


def test_households_combine_interviews() -> None:
    # Two interviews of household 1 (NEWID 11, 12) and one of household 2 (NEWID 21).
    df = make_fmli(
        3,
        NEWID=[11, 12, 21],
        FDHOMECQ=[100.0, 300.0, 50.0],
        FDAWAYCQ=[0.0, 0.0, 50.0],
        FINLWT21=[1000.0, 3000.0, 500.0],
        AGE_REF=[40, 41, 70],
    )
    hh, shares = households(df)
    assert list(hh.index) == [1, 2]
    assert hh.loc[1, "TOTAL_SPEND"] == 400.0
    assert hh.loc[1, "FINLWT21"] == 2000.0  # mean over interviews
    assert hh.loc[1, "AGE_REF"] == 41  # latest interview
    assert shares.loc[2, "Food at home"] == 0.5


def test_shares_sum_to_one_and_zero_spenders_dropped(rng: np.random.Generator) -> None:
    df = make_fmli(20, FDHOMECQ=rng.uniform(0, 500, 20), GASMOCQ=rng.uniform(0, 200, 20))
    df.loc[3, ["FDHOMECQ", "GASMOCQ"]] = 0.0
    hh, shares = households(df)
    assert len(shares) == 19 and 4 not in shares.index  # NEWID 41 -> CUID 4
    assert list(shares.columns) == CATEGORIES
    np.testing.assert_allclose(shares.sum(axis=1), 1.0)
    assert (hh["TOTAL_SPEND"] > 0).all()


def test_blank_replicate_weight_means_zero() -> None:
    df = make_fmli(1)
    df["WTREP05"] = float("nan")
    hh, _ = households(df)
    assert hh.loc[1, "WTREP05"] == 0.0


def test_read_fmli_finds_files_and_uppercases(tmp_path: Path) -> None:
    sub = tmp_path / "intrvw"
    sub.mkdir()
    make_fmli(2).rename(columns=str.lower).to_csv(sub / "fmli231.csv", index=False)
    make_fmli(3).to_csv(sub / "fmli232.csv", index=False)
    (sub / "mtbi231.csv").write_text("ignored\n")
    df = read_fmli(tmp_path)
    assert len(df) == 5 and "FINLWT21" in df.columns


def test_read_fmli_without_files_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_fmli(tmp_path)


def test_load_households_end_to_end(tmp_path: Path) -> None:
    make_fmli(4).to_csv(tmp_path / "fmli231.csv", index=False)
    df, hh, hh_spend = load_households(tmp_path)
    assert len(df) == 4 and len(hh) == 4 and hh_spend.shape == (4, len(CATEGORIES))

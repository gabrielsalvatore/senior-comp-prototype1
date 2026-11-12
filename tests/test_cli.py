from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pie.cli import main

from conftest import make_fmli


@pytest.fixture
def ce_dir(tmp_path: Path, rng: np.random.Generator) -> Path:
    n = 200
    df = make_fmli(
        n,
        BLS_URBN=rng.choice([1, 2], n),
        FAM_TYPE=rng.choice([1, 6, 7, 8], n),
        SEX_REF=rng.choice([1, 2], n),
        AGE_REF=rng.integers(20, 90, n),
        CUTENURE=rng.choice([1, 2, 4], n),
        FINCBTXM=rng.uniform(5_000, 200_000, n),
    )
    for col in ["FDHOMECQ", "GASMOCQ", "UTILCQ", "RENDWECQ", "HEALTHCQ"]:
        df[col] = rng.uniform(0, 1000, n)
    df["RENTEQVX"] = rng.uniform(500, 2500, n)
    df.to_csv(tmp_path / "fmli231.csv", index=False)
    return tmp_path


def test_quick_report_runs(ce_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(ce_dir), "--quick"]) == 0
    out = capsys.readouterr().out
    assert "200 households" in out
    assert "Rural minus urban" in out
    assert "algorithms" not in out


def test_full_report_runs(ce_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(ce_dir)]) == 0
    assert "A4" in capsys.readouterr().out


def test_missing_variable_reports_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    make_fmli(3).drop(columns=["UTILCQ"]).to_csv(tmp_path / "fmli231.csv", index=False)
    assert main([str(tmp_path), "--quick"]) == 1
    assert "UTILCQ" in capsys.readouterr().err


def test_missing_folder_reports_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(tmp_path / "nope"), "--quick"]) == 1
    assert "fmli" in capsys.readouterr().err

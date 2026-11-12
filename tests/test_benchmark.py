from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pie import benchmark
from pie.cpi_series import complete_block, load_bls_series, relatives_matrix

BLS_SAMPLE = (
    "series_id        \tyear\tperiod\t       value\tfootnote_codes\n"
    "CUUR0000SA0      \t2020\tM01\t     100.0\t\n"
    "CUUR0000SA0      \t2020\tM02\t     102.0\t\n"
    "CUUR0000SA0      \t2020\tM13\t     101.0\t\n"
    "CUUR0000SAF11    \t2020\tM01\t      50.0\t\n"
    "CUUR0000SAF11    \t2020\tM02\t      55.0\t\n"
    "CUUR0000SETB01   \t2020\tM02\t      80.0\t\n"
)


def test_naive_loops_by_hand() -> None:
    W = np.array([[0.5, 0.5]])
    R = np.array([[1.0, 1.2], [1.1, 1.0]])
    np.testing.assert_allclose(benchmark.naive_loops(W, R), [[0.05, 0.10]])


def test_all_single_weight_algorithms_agree() -> None:
    inp = benchmark.synthetic_inputs(H=200, I=8, T=6, G=4)
    algs = benchmark.single_weight_algorithms(inp, include_loops=True)
    household = [np.asarray(algs[k]()) for k in ("A1 Python loops", "A2 dense W@R")]
    group = [np.asarray(algs[k]()) for k in algs if k.startswith(("A3", "A4"))]
    np.testing.assert_allclose(household[0], household[1])
    for g in group[1:]:
        np.testing.assert_allclose(g, group[0])


def test_replicate_algorithms_agree() -> None:
    inp = benchmark.synthetic_inputs(H=150, I=6, T=5, G=3, K=45)
    outs = [np.asarray(f()) for f in benchmark.replicate_algorithms(inp).values()]
    assert len(outs) == 3 and outs[0].shape == (45 * 3, 5)
    for out in outs[1:]:
        np.testing.assert_allclose(out, outs[0])


def test_sweep_records_every_algorithm_and_size() -> None:
    rows = benchmark.sweep("H", [50, 100], {"I": 4, "T": 3, "G": 2, "K": 1}, repeats=2)
    assert {r.H for r in rows} == {50, 100}
    assert len({r.algorithm for r in rows}) == 6  # loops included at this size
    assert all(r.mean > 0 and r.repeats == 2 for r in rows)


def test_loglog_slope_recovers_exponent() -> None:
    sizes = np.array([10, 100, 1000])
    assert benchmark.loglog_slope(sizes, 3e-6 * sizes**2) == pytest.approx(2.0)


def test_crossover_formula() -> None:
    assert benchmark.crossover_columns(45, 16) == pytest.approx(45 * 16 / 61)


def test_load_bls_series(tmp_path: Path) -> None:
    path = tmp_path / "cu.data"
    path.write_text(BLS_SAMPLE)
    levels = load_bls_series(path)
    assert list(levels.columns) == ["2020-01", "2020-02"]  # M13 annual average dropped
    assert levels.loc["CUUR0000SAF11", "2020-02"] == 55.0
    block = complete_block(levels, "2020-01", "2020-02")
    assert list(block.index) == ["CUUR0000SA0", "CUUR0000SAF11"]  # gasoline has a gap
    np.testing.assert_allclose(relatives_matrix(block), [[1.0, 1.02], [1.0, 1.1]])


def test_load_bls_series_filters_ids(tmp_path: Path) -> None:
    path = tmp_path / "cu.data"
    path.write_text(BLS_SAMPLE)
    assert list(load_bls_series(path, ["CUUR0000SA0"]).index) == ["CUUR0000SA0"]


def test_load_bls_series_rejects_other_files(tmp_path: Path) -> None:
    path = tmp_path / "x.tsv"
    path.write_text("a\tb\n1\t2\n")
    with pytest.raises(ValueError, match="BLS"):
        load_bls_series(path)

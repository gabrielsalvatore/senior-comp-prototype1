# Changelog

All notable changes are listed here. Versions follow [semantic versioning](https://semver.org/).

## [0.3.0] - unreleased

### Added
- pytest suite with hand-computed cases and Hypothesis tests of the index-number axioms.
- ruff, mypy and a GitHub Actions workflow.
- Paasche, Fisher and Törnqvist formulas.
- Sparse (A4s) and grouped-BLAS (A4g) aggregate-first algorithms; the dense version is kept as A4d.
- Stacked replicate-weight aggregation: every group estimate and BRR standard error from one product.
- `benchmarks/run_benchmarks.py`: scaling experiments over H, I, T, G and S with log-log slopes.
- Reader for BLS CPI flat files (`pie.cpi_series`).
- `docs/theory.md`: when group indexes can be computed aggregate first, the Jensen-gap bound, and the gap measured in the 2023 CE data.

### Fixed
- A household exactly on an income-quintile boundary was put in the next quintile up.
- The CLI now exits with a one-line error when the CE folder or a required variable is missing.
- The CPI loader rejects duplicate categories and gaps.

## [0.2.0] - 2026-11-12

### Changed
- Split `pie_demo.py` into the `pie` package; CPI price levels moved to `data/cpi_u.csv`.
- Missing CE variables raise an error instead of being treated as zero.

## [0.1.0] - 2026-10-26

### Added
- PIE prototype: household inflation from CE microdata (Week 9 presentation).

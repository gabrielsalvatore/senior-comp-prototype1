# Who Bore the Inflation Surge? — Senior Comp Prototype

Computational artifact for Gabriel Saccoccio's combined Computer Science and Economics senior comprehensive project at Allegheny College (CMPSC 600, Fall 2026).

The prototype calculates a personal inflation rate for every household in the U.S. Bureau of Labor Statistics Consumer Expenditure Survey (CE). It then compares groups of households, with confidence intervals from the survey's replicate weights, and compares equivalent algorithms for computing the same numbers at realistic scale, with a proof of when group indexes can be computed aggregate first.

## Research question

How did the inflation experienced by U.S. households differ across socioeconomic characteristics, and did those differences change from before COVID-19 to after it?

## What the report does

1. Loads the CE Interview household files (`fmli*.csv`) for one survey year.
2. Builds each household's spending shares over 16 CPI categories. Owner housing is valued as rental equivalence, as the CPI does.
3. Applies official CPI-U category price changes to get each household's own inflation rate, using the Laspeyres form:

   π_h = Σ_i w_{h,i} · (P_{i,t} / P_{i,0}) − 1

4. Compares groups (urban/rural, income quintile, household type, age, housing tenure). It reports 95% confidence intervals from the 44 BLS replicate weights (balanced repeated replication).
5. Breaks the rural–urban gap into contributions by category.
6. Times equivalent algorithms on the CE weights:

| Algorithm | Idea | Time complexity |
| --- | --- | --- |
| A1 | Naive Python loops | O(H · I · T), interpreted |
| A2 | Dense matrix product `W @ R` | O(H · I · T), BLAS (same arithmetic as A1; a constant-factor gap) |
| A3 | Household first, `A @ (W @ R)` | O(H · I · T + H · T) |
| A4d | Aggregate first with a dense G × H matrix `A` | O(G · H · I + G · I · T) |
| A4s | Aggregate first with sparse `A` (H nonzeros) | O(H · I + G · I · T) |
| A4g | Aggregate first, one BLAS product per group | O(H · I + G · I · T) |

   Here H is households, I is items, T is months (or columns of a sensitivity grid) and G is groups.

7. Computes every group estimate and its BRR standard error as one product with a stacked (45·G) × H matrix, and compares the two multiplication orders.
8. Measures the gap between the democratic mean of household indexes and the aggregate-share index for nonlinear (geometric, harmonic) indexes, with the bounds from `docs/theory.md`.

## Setup

Requires Python 3.10+.

```bash
pip install -r requirements.txt     # to run the report
pip install -e ".[dev]"             # to develop: adds pytest, hypothesis, ruff, mypy
```

### Get the data

The survey files are public but large (about 60 MB per year), so they are not in this repository.

1. Go to the [CE Public-Use Microdata page](https://www.bls.gov/cex/pumd_data.htm).
2. In the **2023** row, download **Interview** in the **CSV** column (`intrvw23.zip`).
3. Unzip it into this folder, so that `intrvw23/fmli232.csv` exists.

## Run

```bash
python3 -m pie intrvw23                          # full demo
python3 -m pie intrvw23 --quick                  # skip the timing section
python3 -m pie intrvw23 --cpi data/cpi_u.csv     # use a different CPI file
python3 feasibility_check.py intrvw23            # check that every needed variable exists
```

`python3 pie_demo.py intrvw23` still works and runs the same report.

If a CE variable the pipeline needs is missing from the files, the program stops
and lists the missing variables instead of treating them as zero.

## Develop

```bash
pytest                                   # unit and property-based tests
ruff check . && ruff format --check .    # lint and formatting
mypy                                     # type checks
```

GitHub Actions runs all of these on every push and pull request (`.github/workflows/ci.yml`).

## Benchmarks

```bash
python3 benchmarks/run_benchmarks.py                                  # synthetic prices
python3 benchmarks/run_benchmarks.py --cpi-series cu.data.0.Current   # real CPI series
```

The script varies H, I, T, G and the number of sensitivity columns S. For each point it reports mean ± sd over repeated runs and log-log slopes against the predicted exponents, and writes everything to `benchmarks/results/`. Real CPI series come from the BLS flat file `cu.data.0.Current` (download from the [BLS CPI database](https://download.bls.gov/pub/time.series/cu/)). `docs/benchmarks.md` explains the results.

## Theory

`docs/theory.md` proves when a group price index can be computed aggregate first. Laspeyres, Paasche and Törnqvist are all generalized means; aggregate-share indexes always commute with aggregation, but the democratic mean of household indexes does so only for Laspeyres. It bounds the error otherwise (Jensen's inequality) and measures it in the 2023 CE data.

## CPI data

Category price levels live in `data/cpi_u.csv`: one row per CPI category, with
its BLS series ID and one column per date (`YYYY-MM`). To change the time windows
or add categories, edit that file instead of the code.

## Example results (2023 CE, June 2021 → June 2022)

| Group | Mean household inflation | 95% CI |
| --- | --- | --- |
| Official CPI-U | 9.06% | — |
| Rural | 11.01% | ±0.35 pp |
| Urban | 10.00% | ±0.08 pp |
| Single mothers | 11.04% | ±0.32 pp |
| Single women | 9.22% | ±0.13 pp |
| Lowest income quintile | 10.26% | ±0.16 pp |
| Highest income quintile | 9.24% | ±0.11 pp |

Over December 2019 → December 2023 the rural–urban gap disappears (20.93% vs 21.01%), because gasoline prices fell back.

## Known limitations of the prototype

- Spending shares come from 2023, after the surge. The thesis uses 2016 and 2019 base years. This is the main reason the rebuilt aggregate (9.7%) runs above the official CPI-U (9.1%).
- 16 broad categories built from CE summary variables. The thesis maps detailed UCC codes to a few hundred CPI items using the [BLS CE–CPI concordance](https://www.bls.gov/cpi/additional-resources/ce-cpi-concordance.htm).
- National prices for every household; regional price indexes come later.
- The group comparisons are descriptive. The thesis adds a survey-weighted regression with replicate-weight standard errors.

## Data sources

- U.S. Bureau of Labor Statistics, [Consumer Expenditure Surveys Public-Use Microdata](https://www.bls.gov/cex/pumd_data.htm).
- U.S. Bureau of Labor Statistics, CPI-U category indexes (U.S. city average, not seasonally adjusted), retrieved via [FRED](https://fred.stlouisfed.org/).

## Versioning

Releases follow [semantic versioning](https://semver.org/) and are tagged `vMAJOR.MINOR.PATCH`; `CHANGELOG.md` lists the changes in each. Per the project contract, the finalized prototype for December 10, 2026 is released as 1.0.0 or later.

## Use of AI tools

Claude Code (Anthropic, Claude Opus 5.5) assisted in November 2026 with: restructuring the code into the `pie` package; writing the test suite, linting, type-checking and CI configuration; the sparse and grouped aggregation code and the benchmark script; and drafting the proofs in `docs/theory.md`. The author reviewed the changes, checked that the report's results on the 2023 CE data are unchanged, and is responsible for the content. Details are in the AI-use record in the research notebook.

## Repository contents

- `pie/`: the prototype, split into modules:
  - `config.py`: categories, CE variable names, survey constants
  - `cpi.py`: loads `data/cpi_u.csv` and computes price relatives
  - `cpi_series.py`: reads monthly CPI series from a BLS flat file
  - `load.py`: reads the CE Interview files, builds spending by category, and raises an error on missing variables
  - `weights.py`: household spending shares and CPI-style aggregate weights
  - `indexes.py`: household Laspeyres inflation; Laspeyres, Paasche, Fisher and Törnqvist formulas; weighted quantiles
  - `groups.py`: income quintile, area, household type, age and tenure groups
  - `variance.py`: standard errors from the 44 BRR replicate weights, looped or stacked
  - `aggregate.py`: sparse, dense and grouped averaging; stacked replicate matrices; basket effect
  - `theory.py`: generalized-mean indexes, Jensen gap and its bounds
  - `benchmark.py`: the algorithms and timing helpers
  - `cli.py`: the command-line report (`python3 -m pie`)
- `benchmarks/`: scaling experiments (`run_benchmarks.py`) and their results
- `tests/`: pytest suite
- `data/cpi_u.csv`: CPI-U category price levels.
- `docs/proposal.md`: the project proposal; `docs/theory.md`: proofs; `docs/benchmarks.md`: benchmark interpretation.
- `pie_demo.py`: backward-compatible entry point that runs `pie.cli`.
- `feasibility_check.py`: confirms the CE files contain every variable the project needs.

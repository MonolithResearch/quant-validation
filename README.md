# monolith-validation

[![ci](https://github.com/MonolithResearch/quant-validation/actions/workflows/ci.yml/badge.svg)](https://github.com/MonolithResearch/quant-validation/actions/workflows/ci.yml)

Reproducible research engineering for backtest validation: statistical diagnostics,
explicit evidence requirements, and tests designed to expose false certification.
The examples use synthetic data. This repository makes no claim of market alpha,
live performance, or suitability for deployment.

The library computes Probabilistic Sharpe (PSR), Deflated Sharpe (DSR), Minimum
Track Record Length (MinTRL), Probability of Backtest Overfitting (PBO) via CSCV,
Sharpe confidence intervals, and descriptive cost/lag scenario comparisons.

## Evidence before a verdict

The default research gate requires **all three**:

```text
DSR >= 0.50       PBO <= 0.20       independent trades >= 50
```

These are configurable project policy thresholds, not universal significance
levels. In particular, DSR 0.50 is not a 95% confidence requirement.

- `PASS`: all required evidence is usable and every threshold is met.
- `FAIL`: all required evidence is usable and at least one threshold fails.
- `INCONCLUSIVE`: required evidence is missing or insufficient. Known threshold
  failures are still reported alongside missing evidence.

`gate_pass` is true **only** for `PASS`. `gate_ready` means evidence is complete,
not that it passed. `missing_reasons` identifies gaps; `gate_reasons` includes
those gaps and any known failures. `gate_thresholds` records the applied policy.
Absent PBO remains `None`. Missing multi-trial dispersion makes DSR and its
expected-maximum benchmark `None`; the ordinary PSR remains available.

## Run it

Python >= 3.10. The package declares NumPy, SciPy, and pandas; the CLI itself
uses the standard library and NumPy for CSV/text input.

```bash
git clone https://github.com/MonolithResearch/quant-validation.git
cd quant-validation
python -m pip install -e ".[test]"
python -m pytest -q
python -m examples.synthetic_demo
```

If dependencies are already installed, tests, the module CLI, and the demo run
directly from the repository root without installing this package. On Windows,
the locally verified interpreter is `py -3.14`.

The demo asserts these outcomes:

```text
Returns only: INCONCLUSIVE
Declared search without dispersion: INCONCLUSIVE
Complete engineered fixture: PASS
Same fixture with too few trades: FAIL
```

The pass fixture deliberately uses a large, persistent synthetic drift. It is a
test of the software contract, not a proposed strategy.

## Python API

```python
import numpy as np
from monolith_validation import litigious_report, sharpe_ratio

returns = np.tile([0.01, 0.03], 40)  # engineered synthetic fixture
matrix = np.column_stack([returns, returns - 0.02])
report = litigious_report(
    returns,
    periods_per_year=252,
    n_trials=2,
    trial_sharpes=[sharpe_ratio(column, 252) for column in matrix.T],
    returns_matrix=matrix,
    pbo_splits=4,
    trade_count=80,  # caller-supplied independent trade count
)
print(report.gate_status)  # PASS
print(report.to_dict())
```

Input contracts:

- Returns are a finite one-dimensional series of per-period returns. Supply
  excess/net returns appropriate to the question; the library does not subtract
  cash returns, costs, or financing automatically. It never drops NaN/infinity
  or flattens multiple strategies into one series. Masked observations are rejected
  in returns, trial Sharpes, and the returns matrix, including nested masked values;
  a masked array with no masked observations remains valid.
- `n_trials` is a positive integer, never a bool or a rounded float. Effective
  count is the maximum of that declaration, the trial-Sharpe sample size, and
  the returns-matrix column count. Adding a short sample cannot reduce a declared
  search of 500 trials to two.
- `trial_sharpes` are **annualised**, converted using `periods_per_year`. Their
  sample standard deviation uses `ddof=1`. A shorter sample may estimate the
  dispersion of a larger declared search, but the caller must justify its
  representativeness and disclose the full count.
- `trial_sr_std` is a **per-period standard deviation**, not a variance or the
  selected strategy's standard error. It overrides dispersion from the sample;
  supplied sample values are still validated. Multi-trial reports require a
  positive finite dispersion. Zero is mathematically supported by the low-level
  formula but is insufficient evidence for a multi-trial gate.
- `sr_benchmark` is per-period and affects PSR/MinTRL. DSR retains the zero-skill
  expected-maximum benchmark. With one trial, DSR reduces to PSR against zero;
  a lone return series still cannot supply comparative PBO.
- `returns_matrix` has rows in time order and columns for compared trials. PBO
  needs at least two columns, an even `pbo_splits >= 2`, and at least
  `max(pbo_splits, 4)` rows. No PBO is inferred from a single strategy.
- `trade_count` is a nonnegative integer supplied by the caller, never inferred
  from the number of observations. Gate probability thresholds must be finite
  and within `[0, 1]`; `min_trades` must be a positive integer. Integer inputs
  must also be representable as finite floats for diagnostics and rendering;
  oversized integers raise `ValueError` rather than leaking an overflow.

Malformed inputs raise `ValueError`. Insufficient usable evidence returns an
inconclusive report with available descriptive statistics. At least three
observations and usable nonzero dispersion/moments are required for report-level
Sharpe inference. `to_dict()` converts unavailable/nonfinite statistics to JSON
`null`; for example, infinite MinTRL means the estimate cannot beat its benchmark
under this calculation, while absent DSR means the required inputs are unavailable.

## CLI

```bash
python -m monolith_validation.cli "0.01,0.03,0.02" --json
gate-check returns.csv --col returns --matrix trials.csv --pbo-splits 4 \
  --n-trials 20 --trial-sharpes trial_sharpes.csv --trades 80 --json
```

The first command intentionally exits **2** with `INCONCLUSIVE`. The console
entry point `gate-check` is available after installation. CSV/text files can
have headers; headerless files retain the first row. Header inference rejects
blank cells and the case-insensitive missing markers `NA`, `N/A`, `NaN`, `null`,
`None`, and `<NA>` (also signed NaN and infinity spellings), so these cannot be
column names. Malformed quoting and fields exceeding the CSV reader's limit are
input errors. Returns selection uses an
explicit `--col`, one recognised return column, or a single column. Ambiguous
columns and misspelled names are errors. Matrix files must contain only numeric
trial columns; put timestamps outside the matrix. Parquet is not supported by
this bounded CLI. Put options before `--` when passing an inline series starting
with a negative number, for example `--json -- "-0.01,0.02,0.03"`.

Exit codes: **0 PASS**, **1 FAIL**, **2 INCONCLUSIVE or input error**. JSON includes
the API status and a matching `verdict`; input errors use `ERROR` and an `error`
message. Argument-syntax errors use argparse's usage message on stderr. JSON is
strict: no NaN/Infinity tokens. Text output shows the actual configured thresholds.

## Assumptions and limits

The expected-maximum approximation assumes independent, normally distributed
trial Sharpe estimates under a zero-skill null. Raw configuration count is used
conservatively; this package does not estimate an effective count from correlations.
The Sharpe inference and square-root annualisation do not correct for serial
correlation or overlapping trades. Scenario comparisons are descriptive and do
not add a fourth gate condition.

CSCV evaluates selection within the supplied matrix; it is not a fresh temporal
holdout. The existing implementation uses equal contiguous blocks, discards a
trailing remainder when rows are not divisible by `pbo_splits`, breaks ties by
column order, and counts strictly below-median OOS ranks. Use divisible row counts
and inspect ties; the number of splits grows as `choose(S, S/2)`. The software
cannot verify completeness of the search archive, point-in-time data integrity,
matrix/strategy provenance, independence of trades, or omitted costs. Passing
these numerical thresholds does not establish those facts.

## Tests and references

Tests include known-answer formulas, the authors' published DSR numerical example,
engineered PBO outcomes, missing-evidence regressions, malformed numeric domains,
trial-count monotonicity, and CLI integration paths. Local verification uses
CPython 3.14.3, NumPy 2.4.4, SciPy 1.17.1, and pytest 9.0.3. CI is configured for
Python 3.10-3.14; configuration alone is not a claim that those remote runs passed.
See [WORK_LOG.md](WORK_LOG.md) for provenance, commands, results, and limitations.

- Bailey & Lopez de Prado (2012), [The Sharpe Ratio Efficient Frontier](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1821643).
- Bailey & Lopez de Prado (2014), [The Deflated Sharpe Ratio](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf), Eq. 2 and the numerical example on pp. 9-10.
- Bailey, Borwein, Lopez de Prado & Zhu (2015), [The Probability of Backtest Overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf).

Research companion: [strategy-graveyard](https://github.com/MonolithResearch/strategy-graveyard).
Its case studies are separate evidence records, not performance claims made here.

[MIT license](LICENSE). Methodology and synthetic examples only.

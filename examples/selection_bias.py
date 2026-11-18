"""Select the highest in-sample Sharpe from independent zero-mean trials.

Run from the repository root with python -m examples.selection_bias.
"""
import numpy as np

from monolith_validation import litigious_report, sharpe_ratio


def run_example(periods=504, trials=100, seed=42, splits=8):
    """One independent synthetic trade per day per candidate, with zero expected return."""
    rng = np.random.default_rng(seed)
    training = rng.normal(0.0, 0.01, size=(periods, trials))
    annual_sharpes = [sharpe_ratio(column, 252) for column in training.T]
    winner = int(np.argmax(annual_sharpes))

    report = litigious_report(
        training[:, winner], periods_per_year=252,
        n_trials=trials, trial_sharpes=annual_sharpes,
        returns_matrix=training, pbo_splits=splits, trade_count=periods,
    )

    # Freeze the winner before drawing any holdout returns. CSCV uses training only.
    holdout = rng.normal(0.0, 0.01, size=(periods, trials))
    return winner, report, sharpe_ratio(holdout[:, winner], 252)


def main():
    winner, report, holdout_sharpe = run_example()
    print(f"Selected column {winner} (zero-based) from {report.n_trials} zero-edge trials")
    print(f"{'Metric':<33}  {'Result':>8}")
    print(f"{'Winner IS Sharpe (annual)':<33}  {report.sharpe:8.3f}")
    print(f"{'PSR against zero':<33}  {report.psr:8.3f}")
    print(f"{'Expected maximum Sharpe (annual)':<33}  {report.expected_max_sharpe_annual:8.3f}")
    print(f"{'DSR against expected maximum':<33}  {report.deflated_sharpe:8.3f}")
    print(f"{'PBO across all trial columns':<33}  {report.pbo:8.3f}")
    print(f"{'Frozen winner holdout Sharpe':<33}  {holdout_sharpe:8.3f}")
    print(f"{'Gate on selection evidence':<33}  {report.gate_status:>8}")


if __name__ == "__main__":
    main()

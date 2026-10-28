"""Deterministic evidence-completeness demo; no market data or strategy claims.

Run from the repository root: python -m examples.synthetic_demo
The deliberately large returns make the PASS fixture easy to inspect by hand.
"""
import numpy as np

from monolith_validation import litigious_report, sharpe_ratio


def main():
    returns = np.tile([0.01, 0.03], 40)
    trials = np.column_stack([returns, returns - 0.02])
    evidence = dict(
        returns_matrix=trials,
        pbo_splits=4,
        trial_sharpes=[sharpe_ratio(column, 252) for column in trials.T],
        trade_count=80,  # Declared toy count; never inferred by the library.
    )
    cases = [
        ("Returns only", litigious_report(returns), "INCONCLUSIVE"),
        ("Declared search without dispersion",
         litigious_report(returns, returns_matrix=trials, pbo_splits=4,
                          n_trials=500, trade_count=80), "INCONCLUSIVE"),
        ("Complete engineered fixture", litigious_report(returns, **evidence), "PASS"),
        ("Same fixture with too few trades",
         litigious_report(returns, **(evidence | {"trade_count": 49})), "FAIL"),
    ]
    for label, report, expected in cases:
        assert report.gate_status == expected, report.to_dict()
        print(f"{label}: {report.gate_status}")
        for reason in report.gate_reasons:
            print(f"  {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

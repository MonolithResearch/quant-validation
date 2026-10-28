"""monolith-validation — research diagnostics from Monolith Research.

Σ  Monolith Research — Evidence over narrative

A small toolkit for investigating uncertainty and selection bias in backtests:
Probabilistic Sharpe, Deflated Sharpe,
Minimum Track Record Length, Probability of Backtest Overfitting (CSCV), a
skew/kurtosis-adjusted Sharpe confidence interval, and an implementation-
uncertainty interval — wrapped in a single report and an evidence-complete gate.

The included examples are synthetic and make no claims about trading performance.
"""
from __future__ import annotations

from .core import (
    DEFAULT_GATE,
    EULER_MASCHERONI,
    LitigiousVerdict,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    implementation_uncertainty,
    litigious_report,
    min_track_record_length,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
    sharpe_confidence_interval,
    sharpe_ratio,
    sharpe_std_error,
)

# --------------------------------------------------------------------------- #
# Brand palette — Monolith Research
# --------------------------------------------------------------------------- #
INK = "#0E0C14"     # near-black
PLUM = "#A78AA3"    # accent
PLUM_DARK = "#7A6177"
PAPER = "#F3EFE8"   # light

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_GATE",
    "EULER_MASCHERONI",
    "LitigiousVerdict",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "implementation_uncertainty",
    "litigious_report",
    "min_track_record_length",
    "probabilistic_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "sharpe_confidence_interval",
    "sharpe_ratio",
    "sharpe_std_error",
    "INK",
    "PLUM",
    "PLUM_DARK",
    "PAPER",
    "__version__",
]

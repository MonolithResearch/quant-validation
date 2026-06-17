"""Known-answer and property tests for the litigation-grade validation core.

Every statistic is pinned either to a textbook standard-normal value or to an
exact deterministic construction, so the suite defends the formulas the way a
sceptical allocator (or an FCA reviewer) would attack them.

Run:
    py -3.14 -m pytest -q
from the repository root.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.stats as ss

from monolith_validation import core as V


# --------------------------------------------------------------------------- #
# Probabilistic Sharpe Ratio (Bailey & Lopez de Prado 2012, SSRN 1821643)
# --------------------------------------------------------------------------- #
def test_psr_equals_half_when_benchmark_equals_estimate():
    # numerator is zero -> Phi(0) = 0.5 exactly, regardless of moments
    got = V.probabilistic_sharpe_ratio(0.5, 50, 0.0, 1.0, sr_benchmark=0.5)
    assert got == pytest.approx(0.5, abs=1e-12)


def test_psr_matches_standard_normal_cdf_clean_case():
    # skew=0, kurt=1 -> denominator sqrt(1) = 1 ; arg = sr * sqrt(n-1)
    # sr=0.196, n=101 -> arg = 0.196 * 10 = 1.96 -> Phi(1.96) = 0.9750021
    got = V.probabilistic_sharpe_ratio(0.196, 101, 0.0, 1.0, sr_benchmark=0.0)
    assert got == pytest.approx(0.9750021, abs=1e-4)


def test_psr_monotonic_in_sharpe():
    lo = V.probabilistic_sharpe_ratio(0.10, 100, 0.0, 3.0)
    hi = V.probabilistic_sharpe_ratio(0.20, 100, 0.0, 3.0)
    assert hi > lo


def test_psr_symmetric_about_half():
    # PSR(sr) + PSR(2*bench - sr) == 1 by symmetry of the normal CDF about the mean.
    sr, n, bench = 0.18, 200, 0.05
    lo = V.probabilistic_sharpe_ratio(sr, n, 0.0, 1.0, sr_benchmark=bench)
    hi = V.probabilistic_sharpe_ratio(2 * bench - sr, n, 0.0, 1.0, sr_benchmark=bench)
    assert lo + hi == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# Skew/kurtosis-adjusted Sharpe standard error and confidence interval
# (Mertens 2002 / Lo 2002 small-sample correction)
# --------------------------------------------------------------------------- #
def test_sharpe_std_error_normal_clean():
    # skew=0, kurt=1 -> numerator term = 1 ; n=101 -> sqrt(1/100) = 0.1
    se = V.sharpe_std_error(0.3, 101, 0.0, 1.0)
    assert se == pytest.approx(0.1, abs=1e-9)


def test_sharpe_confidence_interval_95():
    lo, hi = V.sharpe_confidence_interval(0.3, 101, 0.0, 1.0, prob=0.95)
    # SE = 0.1 ; z_{0.975} = 1.9599640 -> 0.3 +/- 0.19599640
    assert lo == pytest.approx(0.1040036, abs=1e-4)
    assert hi == pytest.approx(0.4959964, abs=1e-4)


# --------------------------------------------------------------------------- #
# Expected maximum Sharpe over N trials (deflation benchmark)
# --------------------------------------------------------------------------- #
def test_expected_max_sharpe_linear_in_std():
    a = V.expected_max_sharpe(10, 1.0)
    b = V.expected_max_sharpe(10, 2.5)
    assert b == pytest.approx(2.5 * a, rel=1e-9)


def test_expected_max_sharpe_two_trials_anchor():
    # N=2, sigma=1: (1-gamma)*Phi^-1(0.5) + gamma*Phi^-1(1 - 1/(2e)) ~= 0.5197
    val = V.expected_max_sharpe(2, 1.0)
    assert val == pytest.approx(0.5197, abs=3e-3)


def test_expected_max_sharpe_increases_with_trials():
    assert V.expected_max_sharpe(100, 1.0) > V.expected_max_sharpe(5, 1.0)


def test_expected_max_sharpe_zero_when_degenerate():
    # < 2 trials or non-positive trial dispersion -> no deflation benchmark
    assert V.expected_max_sharpe(1, 1.0) == 0.0
    assert V.expected_max_sharpe(50, 0.0) == 0.0


# --------------------------------------------------------------------------- #
# Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014, SSRN 2460551)
# DSR = PSR evaluated against the expected-maximum benchmark.
# --------------------------------------------------------------------------- #
def test_deflated_sharpe_is_psr_against_expected_max():
    sr, n, sk, ku, n_trials, sr_std = 0.20, 250, 0.0, 3.0, 30, 0.05
    emax = V.expected_max_sharpe(n_trials, sr_std)
    expected = V.probabilistic_sharpe_ratio(sr, n, sk, ku, sr_benchmark=emax)
    got = V.deflated_sharpe_ratio(sr, n, sk, ku, n_trials, sr_std)
    assert got == pytest.approx(expected, abs=1e-12)


def test_deflated_sharpe_half_when_sr_equals_expected_max():
    n_trials, sr_std = 25, 0.04
    emax = V.expected_max_sharpe(n_trials, sr_std)
    got = V.deflated_sharpe_ratio(emax, 500, 0.0, 3.0, n_trials, sr_std)
    assert got == pytest.approx(0.5, abs=1e-9)


def test_deflated_sharpe_falls_as_trials_rise():
    base = V.deflated_sharpe_ratio(0.18, 400, 0.0, 3.0, 2, 0.05)
    many = V.deflated_sharpe_ratio(0.18, 400, 0.0, 3.0, 500, 0.05)
    assert many < base


def test_deflated_sharpe_reduces_to_psr_without_deflation():
    # With a single trial the expected-max benchmark is zero, so DSR == PSR(vs 0).
    sr, n, sk, ku = 0.22, 300, 0.0, 3.0
    assert V.deflated_sharpe_ratio(sr, n, sk, ku, 1, 0.05) == pytest.approx(
        V.probabilistic_sharpe_ratio(sr, n, sk, ku, 0.0), abs=1e-12
    )


# --------------------------------------------------------------------------- #
# Minimum Track Record Length
# --------------------------------------------------------------------------- #
def test_min_track_record_length_known():
    # sr=0.5, skew=0, kurt=3, sr*=0, p=0.95 :
    # 1 + (1 + 0.5*0.25) * (1.6448536/0.5)^2 = 1 + 1.125*10.82277 = 13.1756
    val = V.min_track_record_length(0.5, 0.0, 3.0, sr_benchmark=0.0, prob=0.95)
    assert val == pytest.approx(13.176, abs=5e-2)


def test_min_track_record_length_infinite_below_benchmark():
    # An estimate that does not beat the benchmark never clears it -> infinite track.
    assert V.min_track_record_length(0.10, 0.0, 3.0, sr_benchmark=0.20) == float("inf")


# --------------------------------------------------------------------------- #
# Probability of Backtest Overfitting via CSCV
# (Bailey, Borwein, Lopez de Prado, Zhu 2015, SSRN 2326253)
# --------------------------------------------------------------------------- #
def _two_block_matrix(col0, col1):
    return np.column_stack([np.array(col0, float), np.array(col1, float)])


def test_pbo_dominant_strategy_is_zero():
    # col0 beats col1 on Sharpe in every block -> IS-best is also OOS-best -> PBO 0
    col0 = [0.03, 0.01, 0.03, 0.01]
    col1 = [0.00, 0.02, 0.00, 0.02]
    m = _two_block_matrix(col0, col1)
    assert V.probability_of_backtest_overfitting(m, n_splits=2) == pytest.approx(0.0, abs=1e-12)


def test_pbo_adversarial_strategy_is_one():
    # whichever strategy wins IS is engineered to lose OOS -> PBO 1
    col0 = [0.05, 0.03, -0.05, -0.03]
    col1 = [-0.05, -0.03, 0.05, 0.03]
    m = _two_block_matrix(col0, col1)
    assert V.probability_of_backtest_overfitting(m, n_splits=2) == pytest.approx(1.0, abs=1e-12)


def test_pbo_pure_noise_is_high():
    rng = np.random.default_rng(7)
    m = rng.standard_normal((250, 50))  # 50 strategies, none with a real edge
    pbo = V.probability_of_backtest_overfitting(m, n_splits=10)
    assert pbo > 0.4


def test_pbo_real_edge_is_low():
    rng = np.random.default_rng(7)
    noise = rng.standard_normal((250, 49)) * 0.01
    edge = 0.03 + rng.standard_normal((250, 1)) * 0.005  # dominant, persistent drift
    m = np.column_stack([edge, noise])
    pbo = V.probability_of_backtest_overfitting(m, n_splits=10)
    assert pbo < 0.10


# --------------------------------------------------------------------------- #
# Implementation uncertainty across cost/lag scenarios
# --------------------------------------------------------------------------- #
def test_implementation_uncertainty_spread():
    out = V.implementation_uncertainty({"base": 2.0, "cost_x2": 1.6, "cost_x3": 1.2, "lag1": 1.4})
    assert out["sharpe_min"] == pytest.approx(1.2)
    assert out["sharpe_max"] == pytest.approx(2.0)
    assert out["sharpe_spread"] == pytest.approx(0.8)
    assert out["base_sharpe"] == pytest.approx(2.0)


def test_implementation_uncertainty_sign_unstable_flag():
    # A scenario that flips the sign of the edge is the loudest red flag.
    out = V.implementation_uncertainty({"base": 0.8, "cost_x3": -0.3})
    assert out["sign_unstable"] is True
    assert out["worst_case_drop"] == pytest.approx(1.1)


# --------------------------------------------------------------------------- #
# Annualised Sharpe helper
# --------------------------------------------------------------------------- #
def test_sharpe_ratio_zero_variance_is_zero():
    assert V.sharpe_ratio(np.zeros(100), periods_per_year=252) == 0.0


def test_sharpe_ratio_annualises():
    rng = np.random.default_rng(3)
    r = rng.normal(0.0005, 0.01, size=1000)
    per_period = float(np.mean(r) / np.std(r))
    assert V.sharpe_ratio(r, periods_per_year=252) == pytest.approx(per_period * np.sqrt(252), rel=1e-9)


# --------------------------------------------------------------------------- #
# Top-level report: units wiring + governance gate
# --------------------------------------------------------------------------- #
def test_litigious_report_consistency_and_gate():
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, size=600) + 0.0005
    rep = V.litigious_report(r, periods_per_year=252, n_trials=1, trade_count=120)

    assert rep.sharpe == pytest.approx(V.sharpe_ratio(r, periods_per_year=252), rel=1e-9)

    pp = float(np.mean(r) / np.std(r))
    sk = float(ss.skew(r))
    ku = float(ss.kurtosis(r, fisher=False))
    assert rep.psr == pytest.approx(
        V.probabilistic_sharpe_ratio(pp, len(r), sk, ku, 0.0), abs=1e-9
    )
    assert isinstance(rep.gate_pass, bool)
    assert rep.n_obs == 600


def test_litigious_report_pbo_none_without_matrix():
    rng = np.random.default_rng(1)
    r = rng.normal(0.0005, 0.01, size=300)
    rep = V.litigious_report(r, periods_per_year=252)
    assert rep.pbo is None


def test_litigious_report_fails_gate_on_thin_trades():
    rng = np.random.default_rng(2)
    r = rng.normal(0.001, 0.01, size=300)
    rep = V.litigious_report(r, periods_per_year=252, n_trials=1, trade_count=12)
    assert rep.gate_pass is False
    assert any("trade" in reason.lower() for reason in rep.gate_reasons)

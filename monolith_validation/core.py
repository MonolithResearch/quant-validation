"""Litigation-grade statistical validation for quantitative backtests.

Σ  Monolith Research — Evidence over narrative

This module is the anti-overfitting gate used inside Monolith Research to decide
whether a backtest is real or a search artifact. It produces the statistics a
sceptical allocator (or an FCA reviewer) would demand:

  * Probabilistic Sharpe Ratio (PSR)
  * Deflated Sharpe Ratio (DSR) with explicit trial count
  * Minimum Track Record Length (MinTRL)
  * Probability of Backtest Overfitting (PBO) via CSCV
  * skew/kurtosis-adjusted Sharpe standard error + confidence interval
  * implementation-uncertainty interval across cost/lag scenarios

Design rule: every "core" statistic operates on PER-PERIOD Sharpe inputs
(``sr``, ``sr_benchmark``, ``trial_sr_std``) with ``n`` = number of observations,
exactly as written in Bailey & Lopez de Prado. Annualisation and unit conversion
live only in the report layer (:func:`litigious_report`), so the formulas can be
tested against textbook standard-normal values without annualisation noise.

References:
  * PSR ............. Bailey & Lopez de Prado (2012), SSRN 1821643
  * DSR ............. Bailey & Lopez de Prado (2014), SSRN 2460551
  * PBO / CSCV ...... Bailey, Borwein, Lopez de Prado, Zhu (2015), SSRN 2326253
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import kurtosis as _kurtosis
from scipy.stats import norm
from scipy.stats import skew as _skew

EULER_MASCHERONI = 0.5772156649015329

# Default promotion gate. A strategy that clears all three has cleared the
# Monolith Research promotion gate: deflated edge survives the search, the
# in-sample pick is not a coin flip out-of-sample, and there are enough
# independent trades for the statistics to mean anything.
DEFAULT_GATE: dict[str, float] = {"deflated_sharpe": 0.50, "pbo": 0.20, "min_trades": 50}


# --------------------------------------------------------------------------- #
# Moments
# --------------------------------------------------------------------------- #
def _clean(returns) -> np.ndarray:
    r = np.asarray(returns, dtype=float).ravel()
    return r[np.isfinite(r)]


def _moments(returns) -> tuple[int, float, float, float, float]:
    """(n, mean, std[ddof=0], skew, non-excess kurtosis). Normal kurtosis == 3."""
    r = _clean(returns)
    n = r.size
    if n == 0:
        return 0, 0.0, 0.0, 0.0, 3.0
    mean = float(r.mean())
    std = float(r.std(ddof=0))
    if n < 3 or std == 0.0:
        return n, mean, std, 0.0, 3.0
    return n, mean, std, float(_skew(r)), float(_kurtosis(r, fisher=False))


def sharpe_ratio(returns, periods_per_year: int | None = None, ddof: int = 0) -> float:
    """Annualised Sharpe ratio (0.0 if the series has no dispersion)."""
    r = _clean(returns)
    if r.size < 2:
        return 0.0
    std = r.std(ddof=ddof)
    if std == 0.0:
        return 0.0
    scale = math.sqrt(periods_per_year) if periods_per_year else 1.0
    return float(r.mean() / std * scale)


# --------------------------------------------------------------------------- #
# Probabilistic Sharpe Ratio and Sharpe confidence interval
# --------------------------------------------------------------------------- #
def _sr_variance_term(sr: float, skew: float, kurt: float) -> float:
    """1 - g3*SR + ((g4-1)/4)*SR^2 — the (n-1)-scaled variance of the SR estimator."""
    return 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr


def probabilistic_sharpe_ratio(sr, n, skew, kurt, sr_benchmark: float = 0.0) -> float:
    """P(true SR > sr_benchmark) given the estimate and its higher moments."""
    if n is None or n < 2:
        return float("nan")
    denom = math.sqrt(max(_sr_variance_term(sr, skew, kurt), 1e-300))
    z = (sr - sr_benchmark) * math.sqrt(n - 1) / denom
    return float(norm.cdf(z))


def sharpe_std_error(sr, n, skew, kurt) -> float:
    """Skew/kurtosis-aware standard error of a per-period Sharpe estimate."""
    if n is None or n < 2:
        return float("nan")
    return math.sqrt(max(_sr_variance_term(sr, skew, kurt), 1e-300) / (n - 1))


def sharpe_confidence_interval(sr, n, skew, kurt, prob: float = 0.95) -> tuple[float, float]:
    """Two-sided ``prob`` confidence interval for a per-period Sharpe estimate."""
    se = sharpe_std_error(sr, n, skew, kurt)
    z = norm.ppf(0.5 + prob / 2.0)
    return (sr - z * se, sr + z * se)


# --------------------------------------------------------------------------- #
# Expected maximum Sharpe and the Deflated Sharpe Ratio
# --------------------------------------------------------------------------- #
def expected_max_sharpe(n_trials: int, trial_sr_std: float) -> float:
    """E[max] of ``n_trials`` iid N(0, trial_sr_std^2) Sharpe estimates.

    The deflation benchmark from Bailey & Lopez de Prado (2014): the Sharpe you
    would expect to see by pure luck after running this many independent trials.
    """
    if n_trials is None or n_trials < 2 or trial_sr_std <= 0.0:
        return 0.0
    g = EULER_MASCHERONI
    a = norm.ppf(1.0 - 1.0 / n_trials)
    b = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(trial_sr_std * ((1.0 - g) * a + g * b))


def deflated_sharpe_ratio(sr, n, skew, kurt, n_trials, trial_sr_std) -> float:
    """PSR evaluated against the expected-maximum-Sharpe benchmark."""
    sr0 = expected_max_sharpe(n_trials, trial_sr_std)
    return probabilistic_sharpe_ratio(sr, n, skew, kurt, sr_benchmark=sr0)


# --------------------------------------------------------------------------- #
# Minimum Track Record Length
# --------------------------------------------------------------------------- #
def min_track_record_length(sr, skew, kurt, sr_benchmark: float = 0.0, prob: float = 0.95) -> float:
    """Observations required for SR to be > sr_benchmark at confidence ``prob``."""
    if sr <= sr_benchmark:
        return float("inf")
    z = norm.ppf(prob)
    return float(1.0 + max(_sr_variance_term(sr, skew, kurt), 0.0) * (z / (sr - sr_benchmark)) ** 2)


# --------------------------------------------------------------------------- #
# Probability of Backtest Overfitting via Combinatorially Symmetric CV
# --------------------------------------------------------------------------- #
def _column_sharpe(rows: np.ndarray) -> np.ndarray:
    """Per-column mean/std Sharpe over ``rows`` (0 where std == 0)."""
    mean = rows.mean(axis=0)
    std = rows.std(axis=0, ddof=0)
    out = np.zeros_like(mean, dtype=float)
    nz = std > 0
    out[nz] = mean[nz] / std[nz]
    return out


def probability_of_backtest_overfitting(returns_matrix, n_splits: int = 16) -> float:
    """PBO via CSCV (Bailey et al. 2015).

    ``returns_matrix`` is T x M of per-period returns (rows = time, cols = the M
    trials/configurations compared). Returns the fraction of symmetric IS/OOS
    splits in which the best in-sample strategy lands below the out-of-sample
    median — i.e. the probability the selected backtest is a false positive.
    """
    M = np.asarray(returns_matrix, dtype=float)
    if M.ndim != 2 or M.shape[1] < 2:
        return float("nan")
    T, N = M.shape
    S = int(n_splits)
    if S < 2 or S % 2 != 0:
        raise ValueError("n_splits must be an even integer >= 2")
    if T < S:
        raise ValueError(f"need at least n_splits={S} rows, got {T}")

    block_len = T // S
    M = M[: block_len * S]
    blocks = [np.arange(i * block_len, (i + 1) * block_len) for i in range(S)]

    logits: list[float] = []
    for is_ids in combinations(range(S), S // 2):
        is_set = set(is_ids)
        is_rows = np.concatenate([blocks[i] for i in is_ids])
        oos_rows = np.concatenate([blocks[i] for i in range(S) if i not in is_set])

        is_perf = _column_sharpe(M[is_rows])
        oos_perf = _column_sharpe(M[oos_rows])

        n_star = int(np.argmax(is_perf))
        order = np.argsort(oos_perf, kind="mergesort")  # ascending: worst -> best
        ranks = np.empty(N, dtype=float)
        ranks[order] = np.arange(1, N + 1)
        omega = ranks[n_star] / (N + 1.0)
        omega = min(max(omega, 1e-12), 1.0 - 1e-12)
        logits.append(math.log(omega / (1.0 - omega)))

    return float(np.mean(np.asarray(logits) < 0.0))


# --------------------------------------------------------------------------- #
# Implementation uncertainty across cost / lag scenarios
# --------------------------------------------------------------------------- #
def implementation_uncertainty(
    scenario_sharpes: Mapping[str, float], base_key: str = "base"
) -> dict[str, Any]:
    """Spread of net Sharpe when the same weights run under different frictions."""
    items = {k: float(v) for k, v in scenario_sharpes.items() if v is not None and np.isfinite(v)}
    if not items:
        return {}
    vals = np.array(list(items.values()), dtype=float)
    smin, smax = float(vals.min()), float(vals.max())
    base = float(items.get(base_key, smax))
    spread = smax - smin
    return {
        "scenarios": items,
        "base_sharpe": base,
        "sharpe_min": smin,
        "sharpe_max": smax,
        "sharpe_spread": spread,
        "worst_case_drop": base - smin,
        "divergence_ratio": float(spread / abs(base)) if base != 0 else float("inf"),
        "sign_unstable": bool(smin < 0 < smax),
    }


# --------------------------------------------------------------------------- #
# Top-level forensic report + governance gate
# --------------------------------------------------------------------------- #
@dataclass
class LitigiousVerdict:
    n_obs: int
    periods_per_year: int
    sharpe: float                       # annualised
    sharpe_per_period: float
    skew: float
    kurtosis: float                     # non-excess (normal == 3)
    psr: float                          # vs sr_benchmark (default 0)
    sharpe_se_annual: float
    sharpe_ci95_annual: tuple[float, float]
    n_trials: int
    expected_max_sharpe_annual: float
    deflated_sharpe: float
    min_track_record_length_periods: float
    min_track_record_length_years: float
    pbo: float | None
    implementation: dict | None
    trade_count: int | None
    gate_pass: bool
    gate_reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["sharpe_ci95_annual"] = list(self.sharpe_ci95_annual)
        return d


def litigious_report(
    returns,
    *,
    periods_per_year: int | None = None,
    n_trials: int = 1,
    trial_sharpes: Sequence[float] | None = None,
    trial_sr_std: float | None = None,
    returns_matrix=None,
    pbo_splits: int = 16,
    scenario_sharpes: Mapping[str, float] | None = None,
    sr_benchmark: float = 0.0,
    trade_count: int | None = None,
    gate: Mapping[str, float] | None = None,
) -> LitigiousVerdict:
    """Assemble the full forensic record for one strategy and apply the gate.

    Parameters of note:
      * ``trial_sharpes``  — annualised Sharpes of every trial in the search that
        produced this strategy (drives the DSR deflation). Converted internally.
      * ``trial_sr_std``   — per-period std of trial Sharpes; overrides the above.
      * ``returns_matrix`` — T x M per-period returns of the compared trials, for PBO.
      * ``scenario_sharpes`` — {label: annual Sharpe} under different cost/lag runs.
    """
    gate = {**DEFAULT_GATE, **(gate or {})}
    n, mean, std, sk, ku = _moments(returns)
    ppy = periods_per_year or 252
    sqrt_ppy = math.sqrt(ppy)

    pp_sr = (mean / std) if std > 0 else 0.0
    ann_sr = pp_sr * sqrt_ppy

    if n >= 2:
        psr = probabilistic_sharpe_ratio(pp_sr, n, sk, ku, sr_benchmark)
        se_pp = sharpe_std_error(pp_sr, n, sk, ku)
        ci_pp = sharpe_confidence_interval(pp_sr, n, sk, ku)
    else:
        psr = float("nan")
        se_pp = float("nan")
        ci_pp = (float("nan"), float("nan"))

    # Deflation inputs: prefer an explicit per-period std, else derive from the
    # annualised trial Sharpes, else fall back to a single pre-registered trial.
    if trial_sr_std is not None:
        sr_std_pp = float(trial_sr_std)
        n_trials_eff = max(int(n_trials), len(trial_sharpes) if trial_sharpes else int(n_trials))
    elif trial_sharpes is not None and len([s for s in trial_sharpes if np.isfinite(s)]) >= 2:
        arr = np.asarray([s for s in trial_sharpes if np.isfinite(s)], dtype=float)
        sr_std_pp = float(arr.std(ddof=1) / sqrt_ppy)
        n_trials_eff = int(arr.size)
    else:
        sr_std_pp = 0.0
        n_trials_eff = max(int(n_trials), 1)

    emax_pp = expected_max_sharpe(n_trials_eff, sr_std_pp)
    dsr = deflated_sharpe_ratio(pp_sr, n, sk, ku, n_trials_eff, sr_std_pp) if n >= 2 else float("nan")

    mintrl_pp = min_track_record_length(pp_sr, sk, ku, sr_benchmark, 0.95)
    mintrl_years = mintrl_pp / ppy if math.isfinite(mintrl_pp) else float("inf")

    pbo = None
    if returns_matrix is not None:
        pbo = probability_of_backtest_overfitting(returns_matrix, n_splits=pbo_splits)

    impl = implementation_uncertainty(scenario_sharpes) if scenario_sharpes else None

    reasons: list[str] = []
    if not math.isfinite(dsr) or dsr < gate["deflated_sharpe"]:
        reasons.append(f"deflated Sharpe {dsr:.3f} < gate {gate['deflated_sharpe']:.2f}")
    if pbo is not None and pbo > gate["pbo"]:
        reasons.append(f"PBO {pbo:.3f} > gate {gate['pbo']:.2f}")
    if trade_count is not None and trade_count < gate["min_trades"]:
        reasons.append(f"independent trades {trade_count} < gate {int(gate['min_trades'])}")

    return LitigiousVerdict(
        n_obs=n,
        periods_per_year=ppy,
        sharpe=ann_sr,
        sharpe_per_period=pp_sr,
        skew=sk,
        kurtosis=ku,
        psr=psr,
        sharpe_se_annual=se_pp * sqrt_ppy if math.isfinite(se_pp) else float("nan"),
        sharpe_ci95_annual=(ci_pp[0] * sqrt_ppy, ci_pp[1] * sqrt_ppy),
        n_trials=n_trials_eff,
        expected_max_sharpe_annual=emax_pp * sqrt_ppy,
        deflated_sharpe=dsr,
        min_track_record_length_periods=mintrl_pp,
        min_track_record_length_years=mintrl_years,
        pbo=pbo,
        implementation=impl,
        trade_count=trade_count,
        gate_pass=(len(reasons) == 0),
        gate_reasons=reasons,
    )

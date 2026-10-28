"""Statistical diagnostics and an evidence-complete gate for quantitative backtests.

Σ  Monolith Research — Evidence over narrative

This module computes research diagnostics for selection bias and uncertainty.
The gate checks explicit evidence and thresholds; it does not establish alpha,
verify the provenance of supplied evidence, or authorize deployment:

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
from dataclasses import asdict, dataclass, field
from itertools import combinations
from numbers import Integral, Real
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import kurtosis as _kurtosis
from scipy.stats import norm
from scipy.stats import skew as _skew

EULER_MASCHERONI = 0.5772156649015329

# Research policy thresholds, not universal statistical significance levels.
DEFAULT_GATE: dict[str, float] = {"deflated_sharpe": 0.50, "pbo": 0.20, "min_trades": 50}


def _count(value, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum} (not bool)")
    _finite(value, name)  # Counts must also fit the floating-point diagnostics.
    return int(value)


def _finite(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number (not bool)")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _reject_masked(values, name: str) -> None:
    """Inspect masks before NumPy can discard them, including nested array values."""
    if np.ma.isMaskedArray(values) and np.any(np.ma.getmask(values)):
        raise ValueError(f"{name} contains masked observations")
    if isinstance(values, np.ndarray):
        if not values.dtype.hasobject:
            return
        values = values.flat
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        pass
    elif hasattr(values, "__array__"):
        _reject_masked(np.asanyarray(values), name)
        return
    else:
        return
    for value in values:
        _reject_masked(value, name)


def _array(values, name: str, ndim: int) -> np.ndarray:
    try:
        _reject_masked(values, name)
        raw = np.asanyarray(values)  # Preserve masks exposed by the array protocol.
        _reject_masked(raw, name)
        if raw.dtype.kind not in "iuf":
            raise ValueError("expected real numeric values")
        result = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError(f"{name} must contain unmasked real numeric values") from exc
    if result.ndim != ndim or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    return result


def _splits(value, name: str = "n_splits") -> int:
    result = _count(value, name, 2)
    if result % 2:
        raise ValueError(f"{name} must be even")
    return result


def _gate_config(gate) -> dict[str, float]:
    if gate is not None and not isinstance(gate, Mapping):
        raise ValueError("gate must be a mapping")
    result = {**DEFAULT_GATE, **({} if gate is None else gate)}
    if set(result) - set(DEFAULT_GATE):
        raise ValueError("unknown gate key; use deflated_sharpe, pbo, min_trades")
    for key in ("deflated_sharpe", "pbo"):
        result[key] = _finite(result[key], f"gate {key}")
        if not 0 <= result[key] <= 1:
            raise ValueError(f"gate {key} must be in [0, 1]")
    result["min_trades"] = _count(result["min_trades"], "gate min_trades", 1)
    return result


# --------------------------------------------------------------------------- #
# Moments
# --------------------------------------------------------------------------- #
def _clean(returns) -> np.ndarray:
    """Validate without dropping observations or flattening multiple strategies."""
    return _array(returns, "returns", 1)


def _moments(returns) -> tuple[int, float, float, float, float]:
    """(n, mean, std[ddof=0], skew, non-excess kurtosis). Normal kurtosis == 3."""
    r = _clean(returns)
    n = r.size
    if n == 0:
        return 0, 0.0, 0.0, 0.0, 3.0
    # Scale before moments to avoid overflow/underflow for finite extreme inputs.
    scale = float(np.max(np.abs(r))) or 1.0
    scaled = r / scale
    mean = float(scaled.mean() * scale)
    std = float(scaled.std(ddof=0) * scale)
    if np.all(r == r[0]):
        return n, float(r[0]), 0.0, 0.0, 3.0
    if n < 3 or std == 0.0:
        return n, mean, std, 0.0, 3.0
    if scaled.std() <= 10 * np.finfo(float).eps * abs(scaled.mean()):
        return n, mean, std, float("nan"), float("nan")
    return n, mean, std, float(_skew(scaled)), float(_kurtosis(scaled, fisher=False))


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
    n_trials = _count(n_trials, "n_trials", 1)
    trial_sr_std = _finite(trial_sr_std, "trial_sr_std")
    if trial_sr_std < 0:
        raise ValueError("trial_sr_std must be nonnegative")
    if n_trials < 2 or trial_sr_std == 0.0:
        return 0.0
    g = EULER_MASCHERONI
    try:
        tail = 1.0 / n_trials
    except OverflowError as exc:
        raise ValueError("n_trials is too large for floating-point evaluation") from exc
    if tail / math.e == 0:
        raise ValueError("n_trials is too large for floating-point evaluation")
    # isf avoids rounding 1 - 1/N to exactly 1 for large N.
    a = norm.isf(tail)
    b = norm.isf(tail / math.e)
    return trial_sr_std * float((1.0 - g) * a + g * b)


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
    scale = np.max(np.abs(rows), axis=0)
    scaled = rows / np.where(scale > 0, scale, 1.0)
    mean = scaled.mean(axis=0)
    std = scaled.std(axis=0, ddof=0)
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
    M = _array(returns_matrix, "returns_matrix", 2)
    S = _splits(n_splits)
    if M.shape[1] < 2:
        return float("nan")
    T, N = M.shape
    if T < max(S, 4):
        raise ValueError(f"need at least max(n_splits={S}, 4) rows, got {T}")

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
    if not isinstance(scenario_sharpes, Mapping):
        raise ValueError("scenario_sharpes must be a mapping of labels to finite Sharpes")
    items = {k: _finite(v, f"scenario_sharpes[{k}]") for k, v in scenario_sharpes.items()}
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
    expected_max_sharpe_annual: float | None
    deflated_sharpe: float | None
    min_track_record_length_periods: float
    min_track_record_length_years: float
    pbo: float | None
    implementation: dict | None
    trade_count: int | None
    gate_pass: bool
    gate_reasons: list[str]
    gate_ready: bool = False
    gate_status: str = "INCONCLUSIVE"
    missing_reasons: list[str] = field(default_factory=list)
    gate_thresholds: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_GATE))

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe record: undefined/nonfinite statistics become null, never NaN."""
        def serializable(value):
            if isinstance(value, dict):
                return {key: serializable(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [serializable(item) for item in value]
            if isinstance(value, (float, np.floating)):
                return float(value) if math.isfinite(value) else None
            return value
        return serializable(asdict(self))


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
    """Compute diagnostics and apply an evidence-complete research gate.

    Parameters of note:
      * ``n_trials`` — declared search count, never reduced by supplied evidence.
        The effective count is max(n_trials, number of trial Sharpes, matrix columns).
      * ``trial_sharpes`` — annualised trial Sharpes; sample std (ddof=1) is
        converted to per-period units. If this is a sample of a larger search,
        declare the full count and justify the sample's representativeness.
      * ``trial_sr_std``   — per-period std of trial Sharpes; overrides the above.
      * ``returns_matrix`` — T x M per-period returns of the compared trials, for PBO.
      * ``scenario_sharpes`` — {label: annual Sharpe} under different cost/lag runs.
      * ``sr_benchmark`` — per-period benchmark for PSR/MinTRL; DSR retains its
        zero-skill expected-maximum benchmark.

    PASS requires usable DSR, PBO, and trade count meeting all thresholds.
    Missing evidence gives INCONCLUSIVE even if another check is known to fail;
    gate_reasons includes both missing evidence and known threshold failures.
    Malformed/nonfinite inputs raise ValueError. A single-trial DSR equals PSR
    against zero, but a single series alone cannot provide comparative PBO.
    """
    gate = _gate_config(gate)
    n_trials = _count(n_trials, "n_trials", 1)
    ppy = 252 if periods_per_year is None else _count(periods_per_year, "periods_per_year", 1)
    pbo_splits = _splits(pbo_splits, "pbo_splits")
    if trade_count is not None:
        trade_count = _count(trade_count, "trade_count", 0)
    sr_benchmark = _finite(sr_benchmark, "sr_benchmark")
    arr = _array(trial_sharpes, "trial_sharpes", 1) if trial_sharpes is not None else np.array([])
    matrix = _array(returns_matrix, "returns_matrix", 2) if returns_matrix is not None else None
    if trial_sr_std is not None:
        trial_sr_std = _finite(trial_sr_std, "trial_sr_std")
        if trial_sr_std < 0:
            raise ValueError("trial_sr_std must be nonnegative")
    n, mean, std, sk, ku = _moments(returns)
    sqrt_ppy = math.sqrt(ppy)
    missing: list[str] = []
    moments_ready = n >= 3 and std > 0 and all(math.isfinite(x) for x in (mean, std, sk, ku))
    if not moments_ready:
        missing.append("returns need at least 3 observations and usable nonzero dispersion/moments")

    pp_sr = (mean / std) if std > 0 else 0.0
    ann_sr = pp_sr * sqrt_ppy

    if moments_ready:
        psr = probabilistic_sharpe_ratio(pp_sr, n, sk, ku, sr_benchmark)
        se_pp = sharpe_std_error(pp_sr, n, sk, ku)
        ci_pp = sharpe_confidence_interval(pp_sr, n, sk, ku)
    else:
        psr = float("nan")
        se_pp = float("nan")
        ci_pp = (float("nan"), float("nan"))

    n_trials_eff = max(n_trials, int(arr.size), matrix.shape[1] if matrix is not None else 0)
    if trial_sr_std is not None:
        sr_std_pp = trial_sr_std
    elif arr.size >= 2:
        scale = float(np.max(np.abs(arr))) or 1.0
        sr_std_pp = float((arr / scale).std(ddof=1)) * (scale / sqrt_ppy)
    else:
        sr_std_pp = 0.0

    spread_ready = n_trials_eff == 1 or (math.isfinite(sr_std_pp) and sr_std_pp > 0)
    if not spread_ready:
        missing.append(f"deflation needs positive finite trial-Sharpe dispersion for {n_trials_eff} trials")
    emax_pp = expected_max_sharpe(n_trials_eff, sr_std_pp) if spread_ready else None
    if emax_pp is not None and not math.isfinite(emax_pp * sqrt_ppy):
        missing.append("deflation benchmark is not finite at the supplied scale")
        emax_pp = None
        spread_ready = False
    dsr = probabilistic_sharpe_ratio(pp_sr, n, sk, ku, emax_pp) if moments_ready and spread_ready else None
    if dsr is not None and (not math.isfinite(dsr) or not 0 <= dsr <= 1):
        missing.append("DSR is not a finite probability in [0, 1]")
        dsr = None

    mintrl_pp = min_track_record_length(pp_sr, sk, ku, sr_benchmark, 0.95) if moments_ready else float("nan")
    mintrl_years = mintrl_pp / ppy

    pbo = None
    if matrix is None:
        missing.append("PBO requires returns_matrix for the compared trials")
    elif matrix.shape[1] < 2 or matrix.shape[0] < max(pbo_splits, 4):
        missing.append("PBO needs at least 2 trial columns and max(pbo_splits, 4) rows")
    else:
        pbo = probability_of_backtest_overfitting(matrix, n_splits=pbo_splits)
        if not math.isfinite(pbo) or not 0 <= pbo <= 1:
            missing.append("PBO is not a finite probability in [0, 1]")
            pbo = None
    if trade_count is None:
        missing.append("independent trade_count is required")

    impl = implementation_uncertainty(scenario_sharpes) if scenario_sharpes is not None else None

    reasons = list(missing)
    if dsr is not None and dsr < gate["deflated_sharpe"]:
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
        expected_max_sharpe_annual=emax_pp * sqrt_ppy if emax_pp is not None else None,
        deflated_sharpe=dsr,
        min_track_record_length_periods=mintrl_pp,
        min_track_record_length_years=mintrl_years,
        pbo=pbo,
        implementation=impl,
        trade_count=trade_count,
        gate_pass=(len(reasons) == 0),
        gate_reasons=reasons,
        gate_ready=not missing,
        gate_status="INCONCLUSIVE" if missing else ("FAIL" if reasons else "PASS"),
        missing_reasons=missing,
        gate_thresholds=gate,
    )

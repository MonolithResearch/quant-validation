"""Regressions for false certification, numeric domains, and search accounting."""
import json
import math

import numpy as np
import pytest

from monolith_validation import core as V


@pytest.fixture
def strong():
    return np.tile([0.01, 0.03], 40)


@pytest.fixture
def complete(strong):
    # Identical variability; column 0 strictly dominates in every CSCV block.
    return dict(
        returns_matrix=np.column_stack([strong, strong - 0.02]),
        pbo_splits=4, n_trials=2, trial_sr_std=0.1, trade_count=80,
    )


def test_missing_evidence_never_passes_but_keeps_descriptive_statistics(strong):
    report = V.litigious_report(strong)
    assert report.gate_pass is False
    assert report.gate_ready is False
    assert report.gate_status == "INCONCLUSIVE"
    assert report.pbo is None
    assert report.sharpe == pytest.approx(2 * math.sqrt(252))
    assert report.psr > 0.99
    assert any("PBO" in reason for reason in report.missing_reasons)
    assert any("trade" in reason for reason in report.missing_reasons)


@pytest.mark.parametrize("omitted", ["returns_matrix", "trade_count", "trial_sr_std"])
def test_removing_any_required_evidence_cannot_preserve_pass(strong, complete, omitted):
    assert V.litigious_report(strong, **complete).gate_pass
    del complete[omitted]
    report = V.litigious_report(strong, **complete)
    assert report.gate_pass is False
    assert report.gate_status == "INCONCLUSIVE"
    assert report.missing_reasons


def test_missing_spread_is_not_labelled_as_deflated_sharpe(strong):
    report = V.litigious_report(strong, n_trials=500)
    assert report.deflated_sharpe is None
    assert report.expected_max_sharpe_annual is None
    assert report.psr > 0.99
    assert any("dispersion" in reason for reason in report.missing_reasons)


@pytest.mark.parametrize("spread", [0.0, None])
def test_unusable_spread_stays_inconclusive(strong, complete, spread):
    complete["trial_sr_std"] = spread
    report = V.litigious_report(strong, **complete)
    assert report.gate_pass is False
    assert report.gate_status == "INCONCLUSIVE"


def test_complete_pass_and_independent_failure_conditions(strong, complete):
    passed = V.litigious_report(strong, **complete)
    assert passed.gate_status == "PASS"
    assert passed.gate_ready is True
    assert passed.gate_pass is True
    assert passed.gate_reasons == passed.missing_reasons == []
    assert passed.pbo == 0.0
    for changes, reason in [
        ({"trade_count": 49}, "trades"),
        ({"trial_sr_std": 10.0}, "deflated Sharpe"),
        ({"returns_matrix": np.column_stack([
            np.tile([.05, .03], 20).tolist() + np.tile([-.05, -.03], 20).tolist(),
            np.tile([-.05, -.03], 20).tolist() + np.tile([.05, .03], 20).tolist(),
        ]), "pbo_splits": 2}, "PBO"),
    ]:
        failed = V.litigious_report(strong, **(complete | changes))
        assert failed.gate_ready is True
        assert failed.gate_status == "FAIL"
        assert failed.gate_pass is False
        assert any(reason in item for item in failed.gate_reasons)


def test_incomplete_takes_precedence_but_preserves_known_failures(strong):
    report = V.litigious_report(strong, trade_count=0)
    assert report.gate_status == "INCONCLUSIVE"
    assert any("PBO" in item for item in report.missing_reasons)
    assert any("trades" in item for item in report.gate_reasons)


def test_declared_search_count_is_never_replaced_by_shorter_sharpe_list(strong):
    small = V.litigious_report(strong, n_trials=2, trial_sharpes=[0.0, 2.0])
    large = V.litigious_report(strong, n_trials=500, trial_sharpes=[0.0, 2.0])
    assert large.n_trials == 500
    assert large.expected_max_sharpe_annual > small.expected_max_sharpe_annual
    assert large.deflated_sharpe <= small.deflated_sharpe


def test_array_trial_sharpes_and_override_use_same_conservative_count(strong):
    sharpes = np.array([0.0, 1.0, 2.0])
    derived = V.litigious_report(strong, n_trials=20, trial_sharpes=sharpes)
    explicit = V.litigious_report(
        strong, n_trials=20, trial_sharpes=sharpes, trial_sr_std=1 / math.sqrt(252),
    )
    assert derived.n_trials == explicit.n_trials == 20
    assert derived.deflated_sharpe == pytest.approx(explicit.deflated_sharpe)
    inferred = V.litigious_report(strong, trial_sharpes=sharpes)
    assert inferred.n_trials == 3


def test_matrix_does_not_allow_a_multi_trial_search_to_claim_one_trial(strong, complete):
    complete.update(n_trials=1, trial_sr_std=None)
    report = V.litigious_report(strong, **complete)
    assert report.n_trials == 2
    assert report.gate_status == "INCONCLUSIVE"


def test_zero_variance_trial_list_is_not_usable_deflation_evidence(strong):
    report = V.litigious_report(strong, trial_sharpes=[2.0, 2.0])
    assert report.deflated_sharpe is None
    assert any("dispersion" in reason for reason in report.missing_reasons)


@pytest.mark.parametrize("returns", [[], [1.0], [1.0, 2.0], [0.01] * 20])
def test_insufficient_return_moments_cannot_clear_gate(returns):
    report = V.litigious_report(returns, trade_count=80)
    assert report.gate_status == "INCONCLUSIVE"
    assert report.deflated_sharpe is None
    assert any("returns" in reason for reason in report.missing_reasons)
    json.dumps(report.to_dict(), allow_nan=False)


@pytest.mark.parametrize("matrix", [np.ones((20, 1)), np.empty((0, 2))])
def test_insufficient_matrix_does_not_fabricate_pbo(strong, matrix):
    report = V.litigious_report(strong, returns_matrix=matrix, trade_count=80)
    assert report.pbo is None
    assert report.gate_status == "INCONCLUSIVE"
    assert any("PBO" in reason for reason in report.missing_reasons)


@pytest.mark.parametrize("field,bad", [
    ("n_trials", x) for x in [True, False, 1.5, 2.0, 0, -1, np.nan, np.inf]
] + [
    ("trade_count", x) for x in [True, False, 50.5, 50.0, -1, np.nan, np.inf]
] + [
    ("periods_per_year", x) for x in [True, 0, -1, 252.5, np.nan, np.inf]
] + [
    ("trial_sr_std", x) for x in [True, -0.1, np.nan, np.inf, -np.inf]
] + [
    ("sr_benchmark", x) for x in [True, np.nan, np.inf, -np.inf]
] + [("pbo_splits", x) for x in [True, 0, 3, 4.5]])
def test_invalid_scalar_inputs_raise_instead_of_being_coerced(strong, field, bad):
    with pytest.raises(ValueError, match=field):
        V.litigious_report(strong, **{field: bad})


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("field", ["returns", "trial_sharpes", "returns_matrix", "scenario_sharpes"])
def test_nonfinite_arrays_are_not_silently_filtered(strong, field, bad):
    args = {"returns": strong}
    args[field] = {"base": bad} if field == "scenario_sharpes" else (
        [[.01, bad], [.02, .03]] if field == "returns_matrix" else [.01, bad, .03]
    )
    with pytest.raises(ValueError, match=field):
        V.litigious_report(**args)


@pytest.mark.parametrize("gate", [
    {"pbo": x} for x in [-.1, 1.1, np.nan, np.inf, True]
] + [{"deflated_sharpe": x} for x in [-.1, 1.1, np.nan, np.inf, True]] + [
    {"min_trades": x} for x in [0, -1, 50.1, np.nan, True]
] + [{"typo": .5}])
def test_invalid_gate_configuration_rejected(strong, gate):
    with pytest.raises(ValueError):
        V.litigious_report(strong, gate=gate)


def test_numpy_integer_counts_are_supported(strong, complete):
    complete.update(n_trials=np.int64(2), trade_count=np.int64(80))
    report = V.litigious_report(strong, periods_per_year=np.int64(252), **complete)
    assert report.gate_pass
    json.dumps(report.to_dict(), allow_nan=False)


@pytest.mark.parametrize("trials,std", [
    (True, .1), (1.5, .1), (0, .1), (2, -.1), (2, np.nan), (2, np.inf),
])
def test_expected_max_rejects_invalid_trial_inputs(trials, std):
    with pytest.raises(ValueError):
        V.expected_max_sharpe(trials, std)


def test_expected_max_is_finite_for_large_counts_without_cdf_cancellation():
    value = V.expected_max_sharpe(10**20, .1)
    assert math.isfinite(value)
    assert value > V.expected_max_sharpe(10**10, .1)


def test_dsr_matches_authors_published_numerical_example():
    # Bailey & Lopez de Prado (2014), numerical example, pp. 9-10:
    # annual SR 2.5; 250 days/year; T=1250; N=100; variance=1/500;
    # skew=-3, Pearson kurtosis=10. Published rounded DSR=0.9004.
    assert V.deflated_sharpe_ratio(
        2.5 / math.sqrt(250), 1250, -3, 10, 100, math.sqrt(1 / 500),
    ) == pytest.approx(.9004, abs=.0001)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_pbo_rejects_nonfinite_matrix_before_ranking(bad):
    matrix = np.tile([[.01, .02], [.03, .01]], (4, 1))
    matrix[0, 0] = bad
    with pytest.raises(ValueError, match="returns_matrix"):
        V.probability_of_backtest_overfitting(matrix, n_splits=2)


@pytest.mark.parametrize("splits", [True, 2.5, 3, 0])
def test_pbo_split_count_is_not_truncated(splits):
    with pytest.raises(ValueError):
        V.probability_of_backtest_overfitting(np.ones((8, 2)), n_splits=splits)


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_finite_rescaling_cannot_change_pbo_or_gate(strong, complete, scale):
    complete["returns_matrix"] = complete["returns_matrix"] * scale
    report = V.litigious_report(strong * scale, **complete)
    assert report.pbo == 0.0
    assert report.gate_status == "PASS"
    assert report.sharpe_per_period == pytest.approx(2.0)


def test_overflowed_deflation_benchmark_is_unusable_even_with_zero_gate(strong, complete):
    complete.update(n_trials=500, trial_sr_std=1e308)
    report = V.litigious_report(strong, gate={"deflated_sharpe": 0.0}, **complete)
    assert report.gate_pass is False
    assert report.gate_status == "INCONCLUSIVE"
    assert report.expected_max_sharpe_annual is None
    assert report.deflated_sharpe is None


def test_nonzero_psr_benchmark_does_not_change_dsr(strong, complete):
    reference = V.litigious_report(strong, **complete)
    changed = V.litigious_report(strong, sr_benchmark=2.0, **complete)
    assert changed.psr == pytest.approx(.5)
    assert changed.psr < reference.psr
    assert changed.deflated_sharpe == reference.deflated_sharpe


def test_exact_gate_thresholds_are_inclusive(strong, complete):
    reference = V.litigious_report(strong, **complete)
    report = V.litigious_report(strong, gate={
        "deflated_sharpe": reference.deflated_sharpe, "pbo": 0.0, "min_trades": 80,
    }, **complete)
    assert report.gate_status == "PASS"


@pytest.mark.parametrize("field,value", [
    ("returns", [[.01, .02], [.02, .01]]),
    ("trial_sharpes", [[1.0, 2.0]]),
    ("returns_matrix", [.01, .02]),
    ("scenario_sharpes", [1.0, 2.0]),
])
def test_malformed_array_shapes_and_scenario_mapping_raise(strong, field, value):
    arguments = {"returns": strong, field: value}
    with pytest.raises(ValueError, match=field):
        V.litigious_report(**arguments)


@pytest.mark.parametrize("field", ["returns", "returns_matrix", "trial_sharpes"])
@pytest.mark.parametrize("representation", ["all", "partial", "nested", "scalar"])
def test_masked_observations_cannot_certify(strong, complete, field, representation):
    arguments = {"returns": strong, **complete, "trial_sharpes": np.array([0., 2.])}
    data = arguments[field]
    mask = np.ones(data.shape, dtype=bool) if representation == "all" else np.zeros(data.shape, dtype=bool)
    mask.flat[0] = True
    masked = np.ma.array(data, mask=mask)
    if representation == "nested":
        masked = list(masked) if data.ndim == 2 else [np.ma.array(value, mask=missing) for value, missing in zip(data, mask)]
    elif representation == "scalar":
        masked = data.tolist()
        if data.ndim == 2:
            masked[0][0] = np.ma.masked
        else:
            masked[0] = np.ma.masked
    arguments[field] = masked
    with pytest.raises(ValueError, match=field):
        V.litigious_report(**arguments)


@pytest.mark.parametrize("field", ["returns", "returns_matrix", "trial_sharpes"])
@pytest.mark.parametrize("nested", [False, True])
def test_unmasked_array_evidence_remains_valid(strong, complete, field, nested):
    arguments = {"returns": strong, **complete, "trial_sharpes": np.array([0., 2.])}
    unmasked = np.ma.array(arguments[field], mask=False)
    arguments[field] = list(unmasked) if nested else unmasked
    report = V.litigious_report(**arguments)
    assert report.gate_status == "PASS"
    assert report.n_obs == 80
    assert report.n_trials == 2


def test_masked_trial_sample_is_rejected_when_used_for_dispersion(strong, complete):
    del complete["trial_sr_std"]
    with pytest.raises(ValueError, match="trial_sharpes"):
        V.litigious_report(strong, trial_sharpes=np.ma.array([0., 2.], mask=True), **complete)


def test_direct_pbo_rejects_nested_masked_rows(strong, complete):
    matrix = list(np.ma.array(complete["returns_matrix"], mask=True))
    with pytest.raises(ValueError, match="returns_matrix"):
        V.probability_of_backtest_overfitting(matrix, n_splits=4)


@pytest.mark.parametrize("field", [
    "periods_per_year", "n_trials", "pbo_splits", "trade_count",
    "trial_sr_std", "sr_benchmark",
])
def test_unrepresentable_scalar_inputs_raise_value_error(strong, complete, field):
    arguments = {**complete, field: 10**309}
    with pytest.raises(ValueError, match=field):
        V.litigious_report(strong, **arguments)


def test_unrepresentable_min_trades_is_rejected(strong, complete):
    with pytest.raises(ValueError, match="min_trades"):
        V.litigious_report(strong, gate={"min_trades": 10**309}, **complete)


@pytest.mark.parametrize("field", ["returns", "returns_matrix", "trial_sharpes"])
@pytest.mark.parametrize("nested", [False, True])
def test_array_protocol_cannot_strip_masked_evidence(strong, complete, field, nested):
    arguments = {"returns": strong, **complete, "trial_sharpes": np.array([0., 2.])}
    data = arguments[field]

    class MaskedArrayLike:
        def __init__(self, value):
            self.value = value

        def __array__(self, dtype=None, copy=None):
            return np.ma.array(self.value, dtype=dtype, mask=True)

    arguments[field] = [MaskedArrayLike(value) for value in data] if nested else MaskedArrayLike(data)
    with pytest.raises(ValueError, match=field):
        V.litigious_report(**arguments)

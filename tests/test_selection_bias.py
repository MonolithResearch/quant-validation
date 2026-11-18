import numpy as np
import pytest

from examples.selection_bias import run_example


def test_best_of_noise_accounts_for_the_whole_search():
    periods, trials, seed = 128, 32, 42
    winner, report, holdout_sharpe = run_example(periods=periods, trials=trials, seed=seed)
    rng = np.random.default_rng(seed)
    train = rng.normal(0.0, 0.01, (periods, trials))
    scores = train.mean(axis=0) / train.std(axis=0)
    assert winner == scores.argmax()
    assert report.sharpe == pytest.approx(scores.max() * np.sqrt(252))
    assert report.n_trials == trials
    assert report.psr > 0.95
    assert report.deflated_sharpe < report.psr
    assert report.pbo > report.gate_thresholds["pbo"]
    assert report.gate_status == "FAIL"

    held_out = rng.normal(0.0, 0.01, (periods, trials))[:, winner]
    assert holdout_sharpe == pytest.approx(held_out.mean() / held_out.std() * np.sqrt(252))

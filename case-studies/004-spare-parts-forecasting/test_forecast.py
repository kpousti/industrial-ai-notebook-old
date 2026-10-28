"""Tests: Croston learns the right rate, ignores empty weeks, MAE is a trap, and
calmer forecasts need less stock for the same service."""

import numpy as np

from forecast import croston_sba, evaluate, ses


def test_croston_learns_rate_on_regular_demand():
    y = np.tile([3, 0, 0, 0], 50).astype(float)   # 3 units every 4 weeks = 0.75/week
    f = croston_sba(y)
    assert abs(f[-1] - 0.75 * (1 - 0.1 / 2)) < 0.01  # SBA deflates by (1 - alpha/2)


def test_croston_holds_steady_in_empty_weeks_but_ses_decays():
    y = np.array([0] * 11 + [4] + [0] * 6, dtype=float)
    c, s = croston_sba(y), ses(y)
    assert np.allclose(c[13:], c[13])               # no demand -> no change
    assert np.all(np.diff(s[13:]) < 0)              # SES keeps sliding toward zero


def test_mae_prefers_the_useless_zero_forecast():
    results = evaluate()
    assert results["MAE"].idxmin() == "zero"
    assert results.loc["zero", "bias_%"] == -100.0


def test_croston_needs_less_stock_than_spike_chasers():
    stock = evaluate()["units_for_95%_fill"]
    assert stock["Croston (SBA)"] < stock["moving average (12w)"]
    assert stock["Croston (SBA)"] < stock["exp. smoothing"]

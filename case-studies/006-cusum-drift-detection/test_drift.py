"""Tests: CUSUM math, load correction, and the fleet-level comparison."""

import numpy as np

from drift import BASELINE, DETECTORS, calibrate, cusum, evaluate, make_fleet, residuals


def test_cusum_resets_on_noise_and_accumulates_small_shift():
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1, 2000)
    # Pure noise: S keeps falling back to 0, so it stays small
    assert cusum(noise).max() < 15
    # A shift of just 1 std dev is invisible per reading but CUSUM grows ~0.5/hour
    shifted = np.concatenate([noise[:1000], noise[1000:] + 1.0])
    assert cusum(shifted)[-1] > 300


def test_load_correction_removes_daily_cycle():
    m = make_fleet(n_healthy=1, n_drifting=0, seed=1)[0]
    r = residuals(m)
    assert abs(np.corrcoef(m["load"], m["temp"])[0, 1]) > 0.5  # raw temp follows load
    assert abs(np.corrcoef(m["load"], r)[0, 1]) < 0.1          # residual does not
    assert abs(r[:BASELINE].std() - 1) < 1e-9                  # standardized on baseline


def test_cusum_catches_every_drift_faster_than_fixed_limit():
    calib = make_fleet(n_healthy=40, n_drifting=0, seed=6)
    test = make_fleet(n_healthy=20, n_drifting=20, seed=106)
    results = {name: evaluate(fn, calibrate(fn, calib), test) for name, fn in DETECTORS.items()}
    cus, fixed = results["CUSUM on load-corrected"], results["Fixed temperature limit"]
    assert cus["caught"] == 20
    assert cus["median_delay_days"] < fixed["median_delay_days"]
    assert cus["false_alarm_machines"] <= 2  # same calibration budget, few false alarms

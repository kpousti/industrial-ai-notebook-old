"""Case study 006: catching slow equipment drift early with CUSUM.

Follow-up to 001. A bearing that is slowly wearing out heats up a fraction of a
degree per day. Single-reading alarms (a fixed limit, a rolling z-score) are
built for sudden jumps and see this late or never. CUSUM adds up small, steady
excesses until they become undeniable.

All four detectors get the same false-alarm budget: each threshold is set on a
separate fleet of healthy machines, so the comparison is about speed, not about
who is allowed to alarm more often.

All data is synthetic. Run:  python drift.py
"""

import numpy as np
import pandas as pd

HOURS = 30 * 24          # 30 days of hourly readings per machine
BASELINE = 7 * 24        # first week is assumed healthy and used to learn "normal"
CUSUM_K = 0.5            # CUSUM slack, in std devs: ignore excesses smaller than this


def make_fleet(n_healthy: int, n_drifting: int, seed: int) -> list[dict]:
    """Generate hourly bearing temperatures for a fleet of machines.

    Temperature = machine offset + load effect + autocorrelated noise (+ drift).
    Drifting machines start a slow linear rise somewhere between day 10 and 20.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(HOURS)
    fleet = []
    for i in range(n_healthy + n_drifting):
        # Daily load cycle with a random phase plus some random load changes
        load = 0.5 + 0.3 * np.sin(2 * np.pi * (t + rng.integers(24)) / 24) + rng.normal(0, 0.05, HOURS)
        # AR(1) noise: real sensor noise is correlated hour to hour
        noise = np.zeros(HOURS)
        for h in range(1, HOURS):
            noise[h] = 0.6 * noise[h - 1] + rng.normal(0, 0.8)
        temp = rng.uniform(62, 76) + rng.uniform(8, 14) * load + noise

        drift_start = None
        if i >= n_healthy:
            drift_start = int(rng.integers(10 * 24, 20 * 24))
            slope_per_day = rng.uniform(0.3, 1.5)  # degrees C per day: slow
            ramp = np.clip(t - drift_start, 0, None) / 24 * slope_per_day
            temp = temp + ramp
        fleet.append({"temp": temp, "load": load, "drift_start": drift_start})
    return fleet


def residuals(machine: dict) -> np.ndarray:
    """Remove each machine's own load effect, learned on its baseline week.

    Fit temp = a + b * load on the baseline, then return the standardized
    leftover. A healthy machine's residual is ~N(0, 1) whatever its load.
    """
    temp, load = machine["temp"], machine["load"]
    b, a = np.polyfit(load[:BASELINE], temp[:BASELINE], 1)
    resid = temp - (a + b * load)
    return resid / resid[:BASELINE].std()


# --- Detector statistics: each returns one number per hour; alarm when > threshold ---

def stat_fixed_limit(m: dict) -> np.ndarray:
    """The classic alarm: raw temperature above a fixed limit."""
    return m["temp"]


def stat_rolling_z(m: dict, window: int = 48) -> np.ndarray:
    """001's baseline: distance from the trailing 48-hour mean, in std devs."""
    s = pd.Series(m["temp"])
    mean = s.rolling(window).mean().shift(1)
    std = s.rolling(window).std().shift(1)
    return ((s - mean) / std).fillna(0).to_numpy()


def stat_residual(m: dict) -> np.ndarray:
    """Load-corrected reading: 'hotter than this machine should be right now'."""
    return residuals(m)


def cusum(r: np.ndarray, k: float = CUSUM_K) -> np.ndarray:
    """One-sided CUSUM: S_t = max(0, S_{t-1} + r_t - k).

    Each hour adds how far the residual sits above the slack k. Noise averages
    out and S falls back to 0; a persistent small rise keeps adding up.
    """
    s = np.zeros(len(r))
    for h in range(1, len(r)):
        s[h] = max(0.0, s[h - 1] + r[h] - k)
    return s


def stat_cusum(m: dict) -> np.ndarray:
    return cusum(residuals(m))


DETECTORS = {
    "Fixed temperature limit": stat_fixed_limit,
    "Rolling z-score (001)": stat_rolling_z,
    "Load-corrected limit": stat_residual,
    "CUSUM on load-corrected": stat_cusum,
}


def calibrate(stat_fn, healthy: list[dict]) -> float:
    """Threshold = highest value seen on healthy machines after the baseline week.

    So on the calibration fleet every detector raises exactly zero alarms.
    """
    return max(stat_fn(m)[BASELINE:].max() for m in healthy)


def evaluate(stat_fn, threshold: float, fleet: list[dict]) -> dict:
    """Score a detector on a test fleet: false alarms, detections, delay."""
    false_alarm_machines, delays, n_drift = 0, [], 0
    for m in fleet:
        alarms = np.flatnonzero(stat_fn(m)[BASELINE:] > threshold) + BASELINE
        start = m["drift_start"]
        if start is None:
            false_alarm_machines += int(len(alarms) > 0)
            continue
        n_drift += 1
        if (alarms < start).any():
            false_alarm_machines += 1  # alarmed before anything was wrong
        after = alarms[alarms >= start]
        if len(after):
            delays.append((after[0] - start) / 24)  # delay in days
    return {
        "caught": len(delays),
        "n_drift": n_drift,
        "false_alarm_machines": false_alarm_machines,
        "median_delay_days": float(np.median(delays)) if delays else float("nan"),
    }


def run(seed: int = 6) -> pd.DataFrame:
    calib = make_fleet(n_healthy=40, n_drifting=0, seed=seed)       # sets thresholds
    test = make_fleet(n_healthy=20, n_drifting=20, seed=seed + 100)  # scores them
    rows = []
    for name, fn in DETECTORS.items():
        res = evaluate(fn, calibrate(fn, calib), test)
        rows.append({"detector": name, **res})
    return pd.DataFrame(rows)


def robustness(seeds=range(1, 9)) -> pd.DataFrame:
    """Repeat the whole experiment on fresh fleets so one lucky seed can't fool us."""
    runs = pd.concat([run(s).assign(seed=s) for s in seeds])
    return runs.groupby("detector", sort=False).agg(
        caught_pct=("caught", lambda c: 100 * c.sum() / (20 * len(seeds))),
        false_alarm_machines=("false_alarm_machines", "sum"),
        avg_median_delay_days=("median_delay_days", "mean"),
    )


if __name__ == "__main__":
    df = run()
    print("Test fleet: 20 healthy + 20 slowly drifting machines, 30 days hourly\n")
    for _, r in df.iterrows():
        print(f"{r['detector']:<26} caught {r['caught']:>2}/{r['n_drift']}  "
              f"false-alarm machines {r['false_alarm_machines']:>2}  "
              f"median delay {r['median_delay_days']:.1f} days")

    print("\nSame experiment on 8 fresh fleets (seeds 1-8, 160 drifting + 160 healthy machines)\n")
    for name, r in robustness().iterrows():
        print(f"{name:<26} caught {r['caught_pct']:>5.1f}%  "
              f"false-alarm machines {r['false_alarm_machines']:>2.0f}  "
              f"avg median delay {r['avg_median_delay_days']:.1f} days")

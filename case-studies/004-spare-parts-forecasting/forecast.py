"""Case study 004: forecasting spare-parts demand that is mostly zeros.

A maintenance storeroom stocks many spare parts. Most weeks a part isn't used
at all, then someone needs three at once. We compare five forecasts:
  1. Zero              - always predict 0 (a trap: it looks best on MAE)
  2. Fixed average     - the training-period average, never updated
  3. Moving average    - mean of the last 12 weeks
  4. Exp. smoothing    - classic simple exponential smoothing (SES)
  5. Croston (SBA)     - forecast "how much" and "how often" separately

then judge them the way the storeroom would: feed each forecast into the same
reorder rule and look at fill rate and how much stock sits on the shelf.

All data is synthetic. Run:  python forecast.py
"""

import numpy as np
import pandas as pd

N_PARTS, N_WEEKS, TEST_WEEKS = 60, 104, 26
WARMUP = 12          # first weeks used only to start the forecasts
ALPHA = 0.1          # smoothing weight for SES and Croston
LEAD_TIME = 2        # weeks from placing an order to having the part
Z = 1.65             # safety factor, set in advance (about 95% if errors were normal)
TARGET_FILL = 0.95   # share of demanded units the shelf should serve right away


# ---------- synthetic data ----------

def make_demand(seed: int = 4) -> np.ndarray:
    """Weekly demand, one row per part. Each part has its own chance of being
    needed in a week (5-40%) and its own typical quantity when it is (1-6)."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.05, 0.40, (N_PARTS, 1))
    size = rng.uniform(1, 6, (N_PARTS, 1))
    occurs = rng.random((N_PARTS, N_WEEKS)) < p
    qty = 1 + rng.poisson(size - 1, (N_PARTS, N_WEEKS))
    return occurs * qty


# ---------- forecasters: f[t] is the forecast made BEFORE week t is known ----------

def zero(y: np.ndarray) -> np.ndarray:
    f = np.zeros(len(y))
    f[:WARMUP] = np.nan
    return f


def fixed_average(y: np.ndarray) -> np.ndarray:
    f = np.full(len(y), np.nan)
    f[WARMUP:] = y[:N_WEEKS - TEST_WEEKS].mean()   # set once from training weeks
    return f


def moving_average(y: np.ndarray, window: int = 12) -> np.ndarray:
    f = np.full(len(y), np.nan)
    for t in range(WARMUP, len(y)):
        f[t] = y[t - window:t].mean()
    return f


def ses(y: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """The level moves a little toward every new week, zeros included. After each
    demand it jumps up and then decays, so it chases the spikes."""
    f = np.full(len(y), np.nan)
    level = y[:WARMUP].mean()
    for t in range(WARMUP, len(y)):
        f[t] = level
        level += alpha * (y[t] - level)
    return f


def croston_sba(y: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """Track two things, updated only in weeks with demand:
    size = typical quantity when it happens, gap = typical weeks between demands.
    Weekly forecast = size / gap, times (1 - alpha/2) to remove a known upward
    bias (the Syntetos-Boylan Approximation)."""
    f = np.full(len(y), np.nan)
    hits = np.flatnonzero(y[:WARMUP])
    size = y[hits].mean() if len(hits) else 1.0
    gap = WARMUP / max(len(hits), 1)
    since = 1                                   # weeks since the last demand
    for t in range(WARMUP, len(y)):
        f[t] = (1 - alpha / 2) * size / gap
        if y[t] > 0:
            size += alpha * (y[t] - size)
            gap += alpha * (since - gap)
            since = 1
        else:
            since += 1
    return f


METHODS = {"zero": zero, "fixed average": fixed_average, "moving average (12w)": moving_average,
           "exp. smoothing": ses, "Croston (SBA)": croston_sba}


# ---------- the storeroom: same reorder rule for every forecast ----------

def simulate_stock(y, f, sigma, start, z=Z) -> dict:
    """Weekly review, order-up-to policy. Each week we top up the stock position
    (on hand + on order) to S = forecast x (lead time + 1 week) + z x sigma
    safety stock. Demand the shelf can't serve that week is a shortfall."""
    cover = LEAD_TIME + 1
    level = lambda t: np.ceil(f[t] * cover + z * sigma * np.sqrt(cover))
    on_hand, pipeline = level(start), [0.0] * LEAD_TIME   # start the test "full"
    served = shelf = 0.0
    for t in range(start, len(y)):
        on_hand += pipeline.pop(0)              # the order placed 2 weeks ago arrives
        sold = min(on_hand, y[t])
        served += sold
        on_hand -= sold
        shelf += on_hand
        pipeline.append(max(0.0, level(t) - on_hand - sum(pipeline)))
    return {"served": served, "demanded": y[start:].sum(), "avg_on_hand": shelf / (len(y) - start)}


def run_storeroom(demand, forecasts, sigmas, start, z=Z):
    """Fill rate and average units on the shelf, summed over all parts."""
    sims = [simulate_stock(y, f, s, start, z) for y, f, s in zip(demand, forecasts, sigmas)]
    fill = sum(r["served"] for r in sims) / sum(r["demanded"] for r in sims)
    return fill, sum(r["avg_on_hand"] for r in sims)


def stock_for_target(demand, forecasts, sigmas, start, target=TARGET_FILL):
    """Fair comparison: raise z (one setting for all parts) until the storeroom
    serves `target` of units, and report the stock that takes."""
    for z in np.arange(0, 4.01, 0.05):
        fill, shelf = run_storeroom(demand, forecasts, sigmas, start, z)
        if fill >= target:
            return shelf
    return np.nan


def evaluate(seed: int = 4) -> pd.DataFrame:
    demand = make_demand(seed)
    start = N_WEEKS - TEST_WEEKS
    rows = {}
    for name, method in METHODS.items():
        forecasts = np.array([method(y) for y in demand])
        err = demand[:, start:] - forecasts[:, start:]
        row = {"MAE": round(np.abs(err).mean(), 2),
               "bias_%": round(-100 * err.sum() / demand[:, start:].sum(), 1)}
        if name != "zero":   # zero is only here to show the MAE trap
            # Safety stock uses each method's own one-week error on training weeks
            sigmas = np.sqrt(np.nanmean((demand - forecasts)[:, WARMUP:start] ** 2, axis=1))
            fill, shelf = run_storeroom(demand, forecasts, sigmas, start)
            row |= {"fill_rate": round(fill, 3), "units_on_shelf": round(shelf),
                    "units_for_95%_fill": round(stock_for_target(demand, forecasts, sigmas, start))}
        rows[name] = row
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    demand = make_demand()
    print(f"parts={N_PARTS}  weeks={N_WEEKS} (last {TEST_WEEKS} = test)  "
          f"weeks with zero demand={100 * (demand == 0).mean():.0f}%  "
          f"test demand={demand[:, -TEST_WEEKS:].sum()} units\n")
    print(evaluate().to_string())

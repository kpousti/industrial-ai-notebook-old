# 006 · Catching slow equipment drift early with CUSUM

## Problem
A wearing bearing doesn't fail all at once: it runs a fraction of a degree hotter each day for weeks. Most alarms judge one reading at a time (a fixed temperature limit, or 001's rolling z-score), so a slow rise is caught late or never. Case study 001 ended with "add a drift-specific detector and measure *time to detection*". This is that follow-up.

## Inputs
Hourly bearing temperature and load for machines over 30 days. **All synthetic**, generated in `drift.py`:
- Each machine has its own normal temperature (62–76 °C), its own sensitivity to load, a daily load cycle, and hour-to-hour correlated noise
- **Calibration fleet:** 40 healthy machines, used only to set thresholds
- **Test fleet:** 20 healthy machines + 20 that start drifting between day 10 and 20, rising 0.3–1.5 °C per day

## Approach
1. **Load correction.** For each machine, fit `temp = a + b × load` on its first week (assumed healthy). The standardized leftover (residual) means "hotter than this machine should be right now", in std devs.
2. **Four detectors:**

| Detector | Alarms when |
|---|---|
| Fixed temperature limit | Raw temperature is above one limit |
| Rolling z-score (001) | A reading is far from its trailing 48-hour mean |
| Load-corrected limit | The residual for a single hour is too high |
| CUSUM on load-corrected | The running sum `S = max(0, S + residual − 0.5)` gets too high |

3. **Same false-alarm budget.** Each detector's threshold is the highest value it ever reached on the 40 healthy calibration machines, so every detector raises zero alarms there. The test then compares speed fairly.

## Results
Test fleet (seed 6):

| Detector | Drifts caught | Machines with a false alarm | Median delay |
|---|---|---|---|
| Fixed temperature limit | 10 / 20 | 0 | 9.3 days |
| Rolling z-score (001) | 0 / 20 | 0 | – |
| Load-corrected limit | 20 / 20 | 0 | 2.8 days |
| CUSUM on load-corrected | 20 / 20 | 1 | 2.4 days |

Repeated on 8 fresh fleets (seeds 1–8; 160 drifting and 160 healthy machines; false alarms counted over all 320):

| Detector | Drifts caught | Machines with a false alarm | Avg median delay (of drifts caught) |
|---|---|---|---|
| Fixed temperature limit | 65.6% | 8 | 7.4 days |
| Rolling z-score (001) | 18.1% | 9 | 8.7 days |
| Load-corrected limit | 100% | 7 | 3.0 days |
| CUSUM on load-corrected | 100% | 5 | 2.4 days |

## Takeaway
The biggest win wasn't CUSUM, it was **load correction**. Machines run at different temperatures and swing with load, so a fixed limit has to sit high enough for the hottest healthy machine at full load. It only fires once a cool machine has drifted a long way, and it missed a third of the drifts entirely. Comparing each machine to its own load-adjusted normal cut the delay from about 7 days to 3.

CUSUM then took off another ~0.6 days (about 20%), with slightly fewer false alarms. It adds up many small excesses, so it notices a steady rise before any single reading looks extreme. The rolling z-score was almost useless here, and that's expected: its 48-hour window keeps learning the drift as the new normal.

One honest caveat: the noise is correlated hour to hour, which makes CUSUM wander more than textbook examples. Setting its threshold on real healthy data (not the textbook `h = 5`) is what kept false alarms low.

## Next
- Estimate *when* the drift started (the last time S was 0) so maintenance can look at what changed then
- Feed the load-corrected residual into 001's Isolation Forest and run both: CUSUM for slow drift, the forest for sudden multi-sensor faults

## Run
```bash
pip install -r ../../requirements.txt
python drift.py
python -m pytest
```

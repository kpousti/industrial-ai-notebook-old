# 005 · Spotting server incidents in plain-text logs

## Problem
Servers write thousands of log lines an hour, and most incidents show up there first: a disk starts throwing errors, someone hammers the login page, a monitoring agent quietly stops. Nobody reads logs line by line, so the usual alert is "too many lines" or a keyword search for `error`. Both miss problems where the *kind* of messages changes but the amount doesn't.

## Inputs
116,659 log lines from one application server over 7 days, grouped into 1,008 ten-minute windows. **All synthetic**, generated in `logwatch.py`. Eleven normal message types (logins, API calls, database queries, heartbeats, cron jobs) follow a daily load cycle. Five incidents are injected in the last 4 days (24 windows in total):

| Incident | What changes in the logs |
|---|---|
| Disk write errors | A new message type appears, ~25 lines per window |
| Brute-force logins | Failed-password lines jump from under 1 to ~12 per window |
| Monitoring agent silent | Heartbeat lines stop completely |
| Database timeouts | 40% of query-ok lines become timeout lines, same total |
| App crash loop | A new "worker exited, restarting" message, a few lines per window |

## Approach
1. **Templates.** Mask the variable parts of each line (IPs, numbers, device names) so `user7 from 10.0.3.17` and `user42 from 10.0.1.200` become one template. HTTP status codes are kept, since a 200 and a 404 are different events. This gives 13 templates.
2. **Count matrix.** One row per window, one column per template.
3. **Detectors**, trained on the first 3 days (assumed incident-free):

| Detector | Idea |
|---|---|
| Volume z-score | Alert if a window's total line count is more than 3 std devs from normal for that time of day |
| Template PCA + unseen rule | PCA (3 components) learns the normal mix of templates on log counts. A window is flagged if PCA reconstructs it badly (error above the 99.5th percentile of training), or if it contains a template never seen in training |

## Results
On the 4 test days (576 windows, 24 incident windows):

| Detector | Precision | Recall | False alarms | Incidents caught |
|---|---|---|---|---|
| Volume z-score | 0.125 | 0.042 | 7 | 1 / 5 |
| Template PCA + unseen rule | 0.857 | 1.000 | 4 | 5 / 5 |

The volume baseline only caught the disk errors, and it raised more false alarms than real ones. The template detector flagged every incident window, with about one false alarm per day.

## Takeaway
The two parts of the template detector cover different failures. PCA alone caught the brute-force logins, the silent agent and the database timeouts in every window, but it scored the two *new* message types as normal (it can only judge templates it has seen). The unseen-template rule caught those right away. Neither part needs labeled incidents, only a few clean days of logs.

The silent agent is the best example: total volume dropped by under 10%, well inside normal noise, but a template that always appears suddenly being at zero is very unusual. Counting *what* is logged beats counting *how much*.

One practical lesson: the masking rule decides what the model can see. A first version also masked HTTP status codes, which merged 200s and 404s into one template, so a wave of 404s would have been invisible.

## Next
- Tell the operator *which* templates drove each alert (the largest reconstruction errors), so an alert reads "heartbeats missing" instead of "score 34"
- Replace hand-written masking with an automatic template miner (e.g. the Drain algorithm) for messy, real-world log formats

## Run
```bash
pip install -r ../../requirements.txt
python logwatch.py
python -m pytest
```

"""Case study 005: spotting server incidents in plain-text logs.

Turns raw log lines into template counts per 10-minute window, then compares:
  1. Volume z-score - alert when the total number of lines is unusual for that time of day
  2. Template PCA   - learn the normal *mix* of message types and alert when a window
                      can't be explained by it (plus a rule for never-seen messages)

All data is synthetic and generated in code with a fixed seed. Run:  python logwatch.py
"""

import re

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

WINDOW_MIN = 10
WINDOWS_PER_DAY = 24 * 60 // WINDOW_MIN  # 144
N_DAYS = 7
TRAIN_DAYS = 3  # first 3 days are assumed incident-free and used to learn "normal"

# Normal message types: (format string, mean lines per window, follows daily load?)
NORMAL = [
    ("sshd: Accepted publickey for user{u} from 10.0.{a}.{b} port {p}", 3, True),
    ("sshd: Failed password for user{u} from 10.0.{a}.{b} port {p}", 0.6, True),
    ("app: GET /api/orders/{n} 200 in {ms}ms", 60, True),
    ("app: POST /api/workorders 201 in {ms}ms", 15, True),
    ("app: GET /api/orders/{n} 404 in {ms}ms", 2, True),
    ("db: query ok rows={n} time={ms}ms", 40, True),
    ("db: connection timeout after {ms}ms", 0.4, True),
    ("cron: job report-export finished in {s}s", 0.5, False),
    ("agent: heartbeat ok cpu={c}% mem={c}%", 10, False),
    ("disk: /var usage at {c}%", 1, False),
    ("kernel: eth0 link up speed 1000Mbps", 0.05, False),
]

# Injected incidents (test period only): (name, start window, length in windows)
INCIDENTS = [
    ("disk write errors", 3 * 144 + 100, 4),
    ("brute-force logins", 4 * 144 + 30, 6),
    ("monitoring agent silent", 4 * 144 + 90, 6),
    ("database timeouts", 5 * 144 + 60, 5),
    ("app crash loop (new message)", 6 * 144 + 20, 3),
]


def _fill(fmt: str, rng) -> str:
    """Fill a format string with random numbers, like a real log line would have."""
    vals = dict(u=rng.integers(1, 60), a=rng.integers(0, 8), b=rng.integers(2, 250),
                p=rng.integers(30000, 65000), n=rng.integers(1, 99999), ms=rng.integers(2, 900),
                s=rng.integers(20, 300), c=rng.integers(5, 95))
    return fmt.format(**vals)


def make_logs(seed: int = 5) -> tuple[pd.DataFrame, pd.Series]:
    """Return (log lines with their window number, true incident label per window)."""
    rng = np.random.default_rng(seed)
    n_win = N_DAYS * WINDOWS_PER_DAY
    tod = np.arange(n_win) % WINDOWS_PER_DAY
    # Daily load: quiet at night, busy during the day shift
    load = 0.6 + 0.5 * np.sin(2 * np.pi * (tod - 36) / WINDOWS_PER_DAY).clip(0) + 0.1

    rates = np.array([[r * (load[w] if daily else 1.0) for _, r, daily in NORMAL] for w in range(n_win)])
    labels = pd.Series(0, index=range(n_win), name="is_incident")
    extra: dict[int, list[tuple[str, float]]] = {}

    def at(name):
        return next((s, n) for nm, s, n in INCIDENTS if nm == name)

    # Each incident changes the *mix* of messages; only some change the total volume much
    s, n = at("disk write errors")
    for w in range(s, s + n):
        extra.setdefault(w, []).append(("disk: write error on /dev/sdb sector {n}", 25))
    s, n = at("brute-force logins")
    rates[s:s + n, 1] += 12  # failed passwords jump from <1 to ~12 per window
    s, n = at("monitoring agent silent")
    rates[s:s + n, 8] = 0  # heartbeats stop; total volume barely moves
    s, n = at("database timeouts")
    rates[s:s + n, 5] *= 0.6  # some queries fail instead of succeeding...
    rates[s:s + n, 6] += 0.4 * 40 * load[s:s + n]  # ...so volume stays about the same
    s, n = at("app crash loop (new message)")
    for w in range(s, s + n):
        extra.setdefault(w, []).append(("app: worker {n} exited with code 139, restarting", 4))
    for _, s, n in INCIDENTS:
        labels[s:s + n] = 1

    rows = []
    for w in range(n_win):
        sources = [(fmt, rates[w, i]) for i, (fmt, _, _) in enumerate(NORMAL)] + extra.get(w, [])
        for fmt, rate in sources:
            for _ in range(rng.poisson(rate)):
                rows.append((w, _fill(fmt, rng)))
    return pd.DataFrame(rows, columns=["window", "line"]), labels


def to_template(line: str) -> str:
    """Mask the variable parts (IPs, numbers, device names) so similar lines share a template."""
    line = re.sub(r"\b\d+\.\d+\.\d+\.\d+\b", "<IP>", line)
    line = re.sub(r"/dev/\w+", "<DEV>", line)
    tokens = line.split()
    for i, tok in enumerate(tokens):
        # Keep HTTP status codes ("... 404 in 12ms"): a 200 and a 404 are different events
        if tok.isdigit() and i + 1 < len(tokens) and tokens[i + 1] == "in":
            continue
        tokens[i] = re.sub(r"\d+", "<N>", tok)
    return " ".join(tokens)


def count_matrix(logs: pd.DataFrame, n_win: int) -> pd.DataFrame:
    """Rows = windows, columns = templates, values = how many lines of that template."""
    logs = logs.assign(template=logs["line"].map(to_template))
    return pd.crosstab(logs["window"], logs["template"]).reindex(range(n_win), fill_value=0)


def volume_zscore_flags(counts: pd.DataFrame, train_end: int, threshold: float = 3.0) -> pd.Series:
    """Baseline: compare each window's total lines to the same time of day in training."""
    total = counts.sum(axis=1)
    tod = counts.index % WINDOWS_PER_DAY
    # Average over a 7-window neighbourhood of time-of-day, so each slot has enough samples
    train = pd.DataFrame({"tod": tod[:train_end], "total": total[:train_end]})
    mean = np.zeros(WINDOWS_PER_DAY)
    std = np.zeros(WINDOWS_PER_DAY)
    for t in range(WINDOWS_PER_DAY):
        near = np.abs((train["tod"] - t + 72) % WINDOWS_PER_DAY - 72) <= 3
        mean[t], std[t] = train.loc[near, "total"].mean(), train.loc[near, "total"].std()
    z = (total - mean[tod]) / std[tod]
    return z.abs() > threshold


def pca_flags(counts: pd.DataFrame, train_end: int, n_components: int = 3,
              quantile: float = 0.995) -> tuple[pd.Series, pd.Series]:
    """Learn the normal template mix with PCA; flag windows it reconstructs badly.

    Templates never seen in training can't be in the model, so any window containing one
    is flagged by a separate, simple rule. Returns (flags, residual score).
    """
    known = counts.columns[(counts.iloc[:train_end] > 0).any()]
    unseen = counts.drop(columns=known).sum(axis=1) > 0

    x = np.log1p(counts[known].to_numpy(float))  # log: a 2x jump matters equally for any template
    scaler = StandardScaler().fit(x[:train_end])
    z = scaler.transform(x)
    pca = PCA(n_components=n_components, random_state=0).fit(z[:train_end])
    recon = pca.inverse_transform(pca.transform(z))
    score = pd.Series(((z - recon) ** 2).sum(axis=1), index=counts.index)  # squared prediction error

    threshold = np.quantile(score[:train_end], quantile)
    return (score > threshold) | unseen, score


def evaluate(seed: int = 5) -> pd.DataFrame:
    """Score both detectors on the 4 test days, by window and by incident."""
    logs, labels = make_logs(seed)
    n_win = len(labels)
    train_end = TRAIN_DAYS * WINDOWS_PER_DAY
    counts = count_matrix(logs, n_win)

    detectors = {
        "volume z-score": volume_zscore_flags(counts, train_end),
        "template PCA + unseen rule": pca_flags(counts, train_end)[0],
    }
    test = slice(train_end, n_win)
    y = labels[test]
    out = {}
    for name, flags in detectors.items():
        f = flags[test]
        tp, fp = int((f & (y == 1)).sum()), int((f & (y == 0)).sum())
        caught = [nm for nm, s, n in INCIDENTS if flags[s:s + n].any()]
        out[name] = {
            "precision": tp / max(tp + fp, 1),
            "recall": tp / int(y.sum()),
            "false_alarms": fp,
            "incidents_caught": len(caught),
            "missed": ", ".join(nm for nm, _, _ in INCIDENTS if nm not in caught) or "-",
        }
    return pd.DataFrame(out).T


if __name__ == "__main__":
    logs, labels = make_logs()
    counts = count_matrix(logs, len(labels))
    print(f"{len(logs):,} log lines -> {counts.shape[1]} templates over {len(labels)} windows "
          f"({labels.sum()} incident windows, all in the last {N_DAYS - TRAIN_DAYS} days)")
    print("Example:", logs["line"].iloc[1], "->", to_template(logs["line"].iloc[1]))
    res = evaluate()
    with pd.option_context("display.width", 140, "display.max_colwidth", 60):
        print(res.round(3).to_string())

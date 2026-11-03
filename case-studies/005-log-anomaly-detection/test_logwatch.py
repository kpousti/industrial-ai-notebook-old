"""Tests: templating groups the right lines, and each detector part earns its place."""

import numpy as np

from logwatch import INCIDENTS, count_matrix, evaluate, make_logs, pca_flags, to_template


def test_template_masks_variables_but_keeps_status_codes():
    a = to_template("sshd: Failed password for user7 from 10.0.3.17 port 51234")
    b = to_template("sshd: Failed password for user42 from 10.0.1.200 port 40001")
    assert a == b == "sshd: Failed password for user<N> from <IP> port <N>"
    # Same endpoint, different outcome: must stay separate templates
    assert to_template("app: GET /api/orders/5 200 in 9ms") != to_template("app: GET /api/orders/5 404 in 9ms")


def test_pca_catches_mix_changes_that_volume_misses():
    logs, labels = make_logs()
    counts = count_matrix(logs, len(labels))
    _, score = pca_flags(counts, train_end=432)
    threshold = np.quantile(score[:432], 0.995)
    s, n = next((s, n) for name, s, n in INCIDENTS if name == "monitoring agent silent")
    assert (score[s:s + n] > threshold).all()  # heartbeats vanish -> every window flagged


def test_combined_detector_catches_every_incident_with_few_false_alarms():
    res = evaluate()
    combined = res.loc["template PCA + unseen rule"]
    assert combined["incidents_caught"] == len(INCIDENTS)
    assert combined["false_alarms"] <= 6  # about 1 per day over 4 test days


def test_volume_baseline_misses_most_incidents():
    res = evaluate()
    assert res.loc["volume z-score", "incidents_caught"] <= 2
    assert res.loc["template PCA + unseen rule", "recall"] > res.loc["volume z-score", "recall"]

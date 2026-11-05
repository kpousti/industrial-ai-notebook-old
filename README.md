# Industrial AI Notebook

![tests](https://github.com/kpousti/industrial-ai-notebook/actions/workflows/tests.yml/badge.svg)

Small, working prototypes of how AI and automation can fix real problems in industrial IT: equipment monitoring, ticket triage, asset tracking, document processing and more.

Each case study follows the same format: **problem → inputs → approach → results → takeaway**, with runnable code and tests.

> **Personal project.** Built on my own time with my own equipment. All data is synthetic and generated in code. Nothing here comes from, describes, or is affiliated with any employer or company, and no confidential or proprietary information is used.

## Case studies

| # | Topic | Techniques | Status |
|---|---|---|---|
| [001](case-studies/001-telemetry-anomaly-detection) | Catching failing equipment early from sensor telemetry | Rolling z-score, Isolation Forest | ✅ |
| [002](case-studies/002-it-ticket-triage) | Routing IT tickets to the right support queue | Keyword rules, TF-IDF + logistic regression, confidence threshold | ✅ |
| [003](case-studies/003-asset-deduplication) | Finding duplicate devices across asset inventories | Blocking, similarity features, logistic regression, union-find | ✅ |
| [005](case-studies/005-log-anomaly-detection) | Spotting server incidents in plain-text logs | Log templating, PCA reconstruction error, unseen-template rule | ✅ |
| [006](case-studies/006-cusum-drift-detection) | Catching slow equipment drift early | Load-corrected residuals, CUSUM, threshold calibration on healthy machines | ✅ |

## Run any case study

```bash
pip install -r requirements.txt
cd case-studies/001-telemetry-anomaly-detection
python detect.py
python -m pytest
```

## About

Built by [Kia Pousti](https://github.com/kpousti), a software developer in San Diego exploring applied AI. Views and code here are my own.

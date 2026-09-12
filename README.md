<div align="center">

# 🔍 ML Model Monitoring Service

**A production-shaped API that catches ML model drift before it silently tanks your accuracy.**

![Python](https://img.shields.io/badge/Python-3.11-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.115-teal) ![scikit--learn](https://img.shields.io/badge/scikit--learn-1.5-orange) ![Docker](https://img.shields.io/badge/Docker-ready-2496ED) ![Tests](https://img.shields.io/badge/tests-14%20passing-brightgreen) ![License](https://img.shields.io/badge/license-MIT-lightgrey)

[**Live Demo**](https://ml-model-monitoring-service.onrender.com/docs) · [**API Docs**](https://ml-model-monitoring-service.onrender.com/docs) · [Report Bug](https://github.com/ABHISHEKTU/ml-model-monitoring-service/issues)

</div>

---

## 📌 The Problem

A deployed ML model's accuracy can silently degrade in production — long before you have new labels to measure it directly — because the data it sees drifts away from what it was trained on. Most teams find out only after something visibly breaks downstream.

This service is the early-warning layer: it statistically compares live production data against training-time data, across **inputs, feature relationships, and model outputs**, and flags exactly what shifted and by how much.

## ✨ Key Features

| Capability | What it catches |
|---|---|
| **Univariate drift** (KS test + PSI) | A single feature's distribution shifted — mean, spread, or shape |
| **Multivariate drift** (domain classifier) | Features individually look fine, but their *relationship* changed (e.g. age/income correlation flips) |
| **Prediction drift** | The model's *output* distribution shifted, even when inputs look stable — a red flag for pipeline/model bugs |
| **History tracking** | Every check is timestamped and queryable — trend drift over time, not just a single snapshot |
| **Fail-soft error handling** | One malformed feature never crashes a whole report; typed exceptions map cleanly to HTTP codes |
| **Restart-safe persistence** | Reference data *and* its schema survive redeploys — no silent state loss |

## 🏗️ Architecture

```
Client
  │
  ▼
FastAPI routes (main.py)  ──▶  validates input, maps errors to HTTP
  │
  ▼
DriftMonitor (drift/monitor.py)  ──▶  orchestrates per-feature checks, fail-soft
  ├── ks_test.py         KS two-sample test (statistical significance)
  ├── psi.py             Population Stability Index (magnitude)
  └── multivariate.py    domain classifier (joint-relationship drift)
  │
  ▼
FileStore / HistoryStore (storage.py, history_store.py)  ──▶  persistence, swappable backend
```

Each layer has exactly one job. Drift logic is pure `numpy`/`pandas`/`sklearn` — zero FastAPI dependency, fully unit-testable without a running server. Storage sits behind a small interface so a file-backed store can later be swapped for S3/Postgres without touching any statistics code.

## 🧠 The Statistics — Briefly

<details>
<summary><b>Kolmogorov-Smirnov (KS) test</b> — is the difference statistically real?</summary>
<br>

Compares the empirical CDFs of two samples. Statistic `D` = the largest vertical gap between them. Distribution-free — no normality assumption, catches any shape/location/scale change. `p-value < 0.05` → the two samples are unlikely to come from the same distribution → drift. Weakness: with large batches, even trivial differences become "significant," which is why it's paired with PSI.

</details>

<details>
<summary><b>Population Stability Index (PSI)</b> — how big is the shift?</summary>
<br>

Bins the reference distribution (deciles for numeric, categories for categorical), then measures how much the population share per bin moved: `PSI = Σ (cur% − ref%) · ln(cur% / ref%)`. Industry thresholds (from credit scoring): `<0.10` none, `0.10–0.25` moderate, `≥0.25` major. Unlike a p-value, PSI isn't sample-size sensitive — it measures magnitude, which is what you actually want on an alerting dashboard.

</details>

<details>
<summary><b>Domain classifier</b> — did feature <i>relationships</i> change?</summary>
<br>

Label reference rows `0`, current rows `1`, train a classifier on all features jointly, measure AUC. If reference and current are the same distribution, the classifier can't beat random guessing (`AUC ≈ 0.5`). A high AUC means the classifier found a real joint pattern separating old from new data — drift that no single-feature test could see.

</details>

## 🛠️ Engineering Challenges I Hit (and Fixed)

**1. State that didn't survive a restart.** Feature schema (numeric vs. categorical columns) was initially kept in an in-memory dict. It worked until the server restarted — the dict reset, but the reference *data* stayed on disk, so drift checks failed with a false "no reference set" error. **Fix:** persisted the schema as a JSON sidecar file alongside the reference data, so both survive restarts together.

**2. A linear model that couldn't see an obvious pattern.** The multivariate domain classifier initially used `LogisticRegression`. On a deliberately crafted test (age/income correlation flipped between reference and current), it scored **AUC 0.46** — essentially random — because the two correlation patterns formed an X-shape no straight decision boundary can separate. **Fix:** switched to `RandomForestClassifier`, which splits non-linearly; AUC jumped to **0.99** on the same data.

## 📡 API Reference

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness probe |
| `POST` | `/reference` | Upload baseline dataset (optionally with `reference_predictions`) |
| `POST` | `/monitor/drift` | Compare a live batch to baseline → per-feature KS+PSI, `multivariate_drift`, `prediction_drift` |
| `GET` | `/monitor/history/{model_id}` | Past drift reports for a model, most recent last |
| `GET` | `/reference/{model_id}/exists` | Check if a baseline is stored |

Full interactive docs (request/response schemas, try-it-out): **[`/docs`](https://ml-model-monitoring-service.onrender.com/docs)**

## 🚀 Quick Start

```bash
git clone https://github.com/ABHISHEKTU/ml-model-monitoring-service.git
cd ml-model-monitoring-service
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Run the test suite:
```bash
python -m pytest tests/ -v
```

Or run it fully containerized:
```bash
docker build -t model-monitor .
docker run -p 8000:8000 model-monitor
```

## 🌐 Live Demo

**[https://ml-model-monitoring-service.onrender.com/docs](https://ml-model-monitoring-service.onrender.com/docs)**

> ⚠️ Runs on a free-tier host with an **ephemeral filesystem** — uploaded reference data/history may be wiped on redeploy or after the instance sleeps from inactivity, since `FileStore`/`HistoryStore` write to local disk. For real production use, swap these for a persistent backend (S3, Postgres) behind the same interface — the storage layer was deliberately kept abstracted for exactly this kind of swap. The instance may also take 10–30s to respond on first request after being idle.

## 🗺️ Roadmap

- [ ] Label drift once ground-truth outcomes arrive (accuracy/calibration over time, not just distributions)
- [ ] Scheduled monitoring + Slack/email alerting instead of on-demand checks
- [ ] Dashboard visualizing PSI/AUC trend from `/monitor/history`
- [ ] Auth + rate limiting for multi-tenant deployment

## 📄 License

This project is licensed under the [MIT License](LICENSE) — free to use, modify, and learn from.

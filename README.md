\# ML Model Monitoring Service


![Python](https://img.shields.io/badge/Python-3.11-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.115-teal) ![scikit--learn](https://img.shields.io/badge/scikit--learn-1.5-orange) ![Docker](https://img.shields.io/badge/Docker-ready-2496ED) ![Tests](https://img.shields.io/badge/tests-14%20passing-brightgreen)



A lightweight, production-shaped API that detects \*\*data drift\*\* between a model's

training (reference) data and its live (current) production data, using multiple

complementary statistical tests — univariate (KS, PSI), multivariate (domain

classifier), and prediction drift.



\## Why this exists



A model's accuracy can silently degrade in production even without new labels to

measure it directly, because the input data it sees drifts away from what it was

trained on. This service is the early-warning layer that catches that before

accuracy visibly drops.



\## Features



\- \*\*Univariate drift\*\* — Kolmogorov-Smirnov test (statistical significance) + 

&#x20; Population Stability Index (magnitude), for numeric and categorical features,

&#x20; including detection of unseen categories in live traffic.

\- \*\*Multivariate drift\*\* — a domain classifier (random forest) trained to

&#x20; distinguish reference vs. current rows using all features jointly. Catches

&#x20; cases where individual feature distributions look unchanged but the

&#x20; \*relationship between features\* has shifted (e.g. age/income correlation

&#x20; flipping) — something no single-feature test can see.

\- \*\*Prediction drift\*\* — monitors the model's \*output\* distribution separately

&#x20; from its inputs, using the same KS/PSI machinery.

\- \*\*History tracking\*\* — every drift check is persisted (JSONL) and queryable

&#x20; via `/monitor/history/{model\_id}`, so drift can be tracked over time.

\- \*\*Robust error handling\*\* — typed exceptions (`SchemaMismatchError`,

&#x20; `InsufficientDataError`, `ReferenceNotSetError`) mapped centrally to clean

&#x20; HTTP responses; a single malformed feature never crashes a whole report

&#x20; (fail-soft design).

\- \*\*Persistent, restart-safe storage\*\* — reference data \*and\* its feature

&#x20; config survive server restarts (file-backed, not in-memory).

\- Fully tested (14 pytest tests, unit + API integration) and containerized.



\## Architecture



Client → FastAPI routes (main.py) → DriftMonitor (drift/monitor.py)

├── ks\_test.py (univariate significance)

├── psi.py (univariate magnitude)

└── multivariate.py (joint relationship drift)

↓ ↓

FileStore (storage.py) HistoryStore (history\_store.py)





Each layer has one job: routes validate and translate errors to HTTP; drift

logic is pure numpy/pandas/sklearn with zero FastAPI knowledge (fully unit

testable without a server); storage is swappable (file-backed now, could be

S3/Postgres later) behind a tiny interface.



\## The statistics, briefly



\- \*\*KS test\*\*: compares empirical CDFs of two samples; answers "is this

&#x20; difference statistically real, not noise?"

\- \*\*PSI\*\*: bins the reference distribution, measures how much the population

&#x20; share per bin shifted; answers "how big is the shift, in a way I can put a

&#x20; stable alert threshold on?" Not sample-size sensitive the way a p-value is.

\- \*\*Domain classifier (multivariate)\*\*: label reference=0/current=1, train a

&#x20; classifier on all features jointly, measure AUC. AUC near 0.5 = distributions

&#x20; indistinguishable; AUC well above 0.5 = classifier found a real joint pattern

&#x20; that separates them, even if no single feature moved.



\## A real bug I hit and fixed



Initially, the mapping of "which columns are numeric vs. categorical" for each

model was kept in an in-memory Python dict. It worked fine until I restarted the

server (or `--reload` triggered a restart) — the dict reset to empty, so

`/monitor/drift` started failing with a false "no reference set" error, even

though the actual reference \*data\* was safely persisted to disk. Root cause:

two pieces of state tied to the same model\_id, only one of which was durable.

Fixed by persisting the feature config as a JSON sidecar file next to the

reference parquet file, so both survive restarts.



I also initially used `LogisticRegression` for the multivariate domain

classifier and it failed to detect an intentionally crafted correlation-flip

scenario (AUC \~0.46, near random) — because the two correlation patterns formed

an X-shape that no straight decision boundary can separate. Switching to

`RandomForestClassifier` (which can split non-linearly) fixed it (AUC \~0.99).



\## API



\- `GET /health`

\- `POST /reference` — upload baseline dataset (optionally with

&#x20; `reference\_predictions`)

\- `POST /monitor/drift` — check a live batch against the baseline; returns

&#x20; per-feature KS+PSI, `multivariate\_drift`, and (if predictions supplied)

&#x20; `prediction\_drift`

\- `GET /monitor/history/{model\_id}` — past drift reports for a model

\- `GET /reference/{model\_id}/exists`



Interactive docs at `/docs`.



\## Run it



```bash

pip install -r requirements.txt

uvicorn app.main:app --reload

python -m pytest tests/ -v

```



Or with Docker:

```bash

docker build -t model-monitor .

docker run -p 8000:8000 model-monitor

```
## Deployment note

**Live demo**: https://ml-model-monitoring-service.onrender.com/docs

The hosted demo above runs on a free-tier host with an **ephemeral filesystem** —
uploaded reference data and history may be wiped on redeploy/restart (or after
the free instance sleeps from inactivity), since `FileStore`/`HistoryStore`
write to local disk. For real production use, swap these for a persistent
backend (S3, Postgres) behind the same interface — the storage layer was
deliberately kept abstracted for exactly this kind of swap.

\## Possible next steps



\- Scheduled monitoring + alerting (Slack/email) instead of on-demand only

\- Label drift once ground-truth outcomes arrive (accuracy over time, not just

&#x20; input/output distribution)

\- Dashboard visualizing PSI/AUC trend from `/monitor/history`

\- Auth + rate limiting for multi-tenant deployment


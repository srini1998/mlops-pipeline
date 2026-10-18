# MLOps Pipeline: event-driven retraining on real drifting data

**Current public website:** [current-ml.vercel.app](https://current-ml.vercel.app). The interface is English-only and its main task requires no AI-weight download. See [current interface, datasets and execution boundaries](site/INTERFACE.md). The backend experiments and historical measurements below are separate from the current public UI.

Upload a new week of data and the pipeline validates it, versions it, scores the model currently
in production on that week, trains a challenger, and promotes it **only if it passes the gates**.
A prediction service always serves the registry's champion and hot-swaps on promotion or rollback.

Everything is free and runs locally: scikit-learn, **MLflow** (tracking + model registry on SQLite),
FastAPI, GitHub Actions. No cloud account or billing. *(This project began as an AWS SageMaker
design; that plan is preserved in `docs/phase1/`. The cloud mapping is below.)*

## The data: real concept drift

**Elec2**, the New South Wales electricity market (Harries 1999; Gama), OpenML #151, licence
"Public". It has 45,312 half-hour records, and the label is whether the price went **UP or DOWN**
relative to the last 24 hours. Demand, seasons and market rules change over the 2.5 years, so a
model trained once goes stale. Uploads are a replay of that real history, one week per file
(`data/source/electricity.arff`, checksum-pinned).

## Flow

```
landing/week_042.csv ──► trigger (new + size-stable + not seen) ──► validate ──► content-addressed store
                                                                                   │
            ┌──────────────────────────────────────────────────────────────────────┘
            ▼
   score current champion on the new week (prequential "live" accuracy - it never saw this week)
            ▼
   train challenger on the last 8 weeks ─► holdout = newest 2 days ─► gates ─► MLflow registry
            ▼                                                               │
   promote (alias "champion") ◄── only if gates pass                          ▼
            ▼                                                       lineage: data hashes, params,
   FastAPI /predict hot-swaps to the new champion                    metrics, gate decision + reason
```

**Gates**

| Gate | Rule |
|---|---|
| Data validation | Schema, no nulls, normalized ranges, plausible label share, half-hour ordering |
| Duplicate | Identical bytes already admitted → no retrain |
| No-skill floor | Holdout accuracy ≥ max(0.60, majority-class rate + 0.05) |
| Champion comparison | Challenger must beat the champion on the same holdout by ≥ 0.005 |

Rollback: `mlops rollback --reason "..."` points the alias back at the version the champion
replaced. The service picks it up within 5 s.

## Results: 100 real weeks with two injected faults (`docs/simulation.json`)

| Live accuracy on unseen weeks 16–99 | |
|---|---|
| **Pipeline (champion at the time)** | **88.2%** |
| Persistence baseline (repeat the last direction) | 85.8% |
| Static model (trained once on weeks 0–7) | 79.5% |

Over the 100 weeks: 53 promotions, 45 champion kept, 2 blocked by the floor, 1 upload rejected,
1 duplicate.

- **Corrupt upload** (week 33, prices un-normalized, as if a feed broke): rejected by
  validation before touching any model.
- **Broken challenger** (week 66, labels misaligned as by a training bug): holdout 0.667 <
  floor 0.685, so it never served.
- A **real** week also failed the floor (week 65, 0.865 < 0.873 on a skewed week); the champion
  stayed.

### Things the real data taught this pipeline

1. **A naive baseline beat the first version.** Without lag features the pipeline scored 77%
   against persistence's 86%. Adding the previous half-hour's direction and price change, both
   known at prediction time, fixed it. Shipping without a baseline would have hidden this.
2. **The published `date` field goes backwards at 5 points.** A "dates must increase" rule
   rejected a genuine week. Ordering is now checked on the half-hour sequence, and a regression
   test covers week 75.
3. **A fixed 0.60 floor is too weak.** A no-skill model clears it on skewed weeks, so the floor is
   now relative to the majority class.

## Run it

```bash
uv sync
uv run mlops simulate --weeks 100            # full replay with injected faults -> docs/simulation.json
uv run mlops replay --first 10 --weeks 3     # drop real weekly files into state/landing
uv run mlops watch --once                    # or `mlops watch` to keep polling
uv run mlops status                          # registry history, champion, gate decisions
uv run mlops rollback --reason "bad deploy"
uv run mlops serve                           # POST /predict, GET /health
uv run mlflow ui --backend-store-uri sqlite:///state/mlflow.db   # browse runs and lineage
uv run pytest
```

Models are saved with **skops**, not pickle. Only the one sklearn tree type our training code
produces is trusted on load.

## Cloud mapping (not deployed)

| Here | AWS equivalent in the original design |
|---|---|
| landing directory + polling trigger | S3 `ObjectCreated` → EventBridge/Lambda |
| content-addressed data store | S3 + DVC |
| local training | SageMaker training job |
| MLflow registry + `champion` alias | SageMaker Model Registry approval status |
| FastAPI service with hot-swap | SageMaker endpoint |

## Limits

- Single machine. The trigger polls a directory rather than receiving real S3 events.
- The holdout is two days (96 rows), so per-week gate decisions are noisy. The live-accuracy
  comparison over 84 weeks is the reliable number.
- Normalized features come from the published dataset. This predicts direction, not prices.

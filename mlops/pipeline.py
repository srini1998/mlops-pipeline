"""Batch arrival -> validate -> version -> monitor current champion -> train challenger -> gate -> promote.

Each batch is one week. Before retraining, the serving champion is scored on the new
week it has never seen: that prequential accuracy is the honest "how did production do"
number. The challenger is then trained on a sliding window and must beat the champion on
a shared holdout (the newest two days) to be promoted.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

from . import registry
from .data import FEATURES, DataStore, ValidationError, add_features, validate

logger = logging.getLogger("mlops.pipeline")


@dataclass
class GateConfig:
    window_weeks: int = 8
    holdout_rows: int = 96          # newest two days of the arriving week
    min_accuracy: float = 0.60      # absolute floor; a model below this never serves
    min_skill: float = 0.05         # ...and it must beat always-predicting-the-majority-class by this much
    min_gain: float = 0.005         # challenger must beat the champion by at least this
    seed: int = 7


@dataclass
class BatchResult:
    batch: str
    data_sha256: str | None = None
    status: str = "pending"         # rejected | duplicate | promoted | kept_champion | failed_floor
    reason: str = ""
    champion_before: str | None = None
    champion_after: str | None = None
    champion_live_accuracy: float | None = None
    challenger_version: str | None = None
    challenger_holdout_accuracy: float | None = None
    champion_holdout_accuracy: float | None = None
    extra: dict = field(default_factory=dict)


def train_model(df: pd.DataFrame, seed: int, shuffle_labels: bool = False):
    y = df["label"].sample(frac=1, random_state=seed).to_numpy() if shuffle_labels else df["label"].to_numpy()
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed)
    model.fit(df[FEATURES], y)
    return model


class Pipeline:
    def __init__(self, state: Path, cfg: GateConfig | None = None):
        self.state = state
        self.cfg = cfg or GateConfig()
        self.store = DataStore(state / "data")
        self.client = registry.configure(state)
        self.accepted = state / "accepted.jsonl"   # ordered list of batches admitted to training
        self.results = state / "results.jsonl"

    def _accepted(self) -> list[dict]:
        if not self.accepted.exists():
            return []
        return [json.loads(line) for line in self.accepted.read_text().splitlines() if line.strip()]

    def _record(self, result: BatchResult) -> BatchResult:
        with self.results.open("a") as f:
            f.write(json.dumps(asdict(result)) + "\n")
        logger.info("%s -> %s (%s)", result.batch, result.status, result.reason)
        return result

    def process(self, path: Path, *, shuffle_labels: bool = False) -> BatchResult:
        """``shuffle_labels`` deliberately trains a broken challenger; used to prove the gate works."""
        res = BatchResult(batch=path.name, champion_before=registry.champion_version(self.client))
        try:
            batch = pd.read_csv(path)
            validate(batch)
        except (ValidationError, pd.errors.ParserError, ValueError) as exc:
            res.status, res.reason = "rejected", f"validation: {exc}"
            res.champion_after = res.champion_before
            return self._record(res)

        version, is_new = self.store.put(path)
        res.data_sha256 = version.sha256
        if not is_new and any(a["sha256"] == version.sha256 for a in self._accepted()):
            res.status, res.reason = "duplicate", "identical bytes were already processed"
            res.champion_after = res.champion_before
            return self._record(res)

        # 1) Monitoring: how did the serving model do on this week it has never seen?
        if res.champion_before is not None:
            champ = registry.load(res.champion_before)
            feats = add_features(batch)
            res.champion_live_accuracy = round(accuracy_score(feats["label"], champ.predict(feats[FEATURES])), 4)

        with self.accepted.open("a") as f:
            f.write(json.dumps({"sha256": version.sha256, "batch": path.name}) + "\n")
        window = [a["sha256"] for a in self._accepted()][-self.cfg.window_weeks:]
        data = pd.concat([add_features(self.store.frame([sha])) for sha in window], ignore_index=True)
        train, holdout = data.iloc[: -self.cfg.holdout_rows], data.iloc[-self.cfg.holdout_rows:]

        # 2) Train the challenger and log everything needed to reproduce it.
        with mlflow.start_run(run_name=f"train-{path.stem}") as run:
            model = train_model(train, self.cfg.seed, shuffle_labels=shuffle_labels)
            acc = accuracy_score(holdout["label"], model.predict(holdout[FEATURES]))
            res.challenger_holdout_accuracy = round(acc, 4)
            mlflow.log_params({**asdict(self.cfg), "window_batches": len(window), "train_rows": len(train),
                               "shuffle_labels": shuffle_labels})
            mlflow.log_dict({"data_versions": window}, "lineage.json")
            mlflow.set_tags({"trained_through": path.name, "data_sha256": version.sha256})
            mlflow.log_metric("challenger_holdout_accuracy", acc)
            if res.champion_live_accuracy is not None:
                mlflow.log_metric("champion_live_accuracy", res.champion_live_accuracy)

            champ_acc = None
            if res.champion_before is not None:
                champ = registry.load(res.champion_before)
                champ_acc = accuracy_score(holdout["label"], champ.predict(holdout[FEATURES]))
                res.champion_holdout_accuracy = round(champ_acc, 4)
                mlflow.log_metric("champion_holdout_accuracy", champ_acc)

            # 3) Gate.
            majority = max(holdout["label"].mean(), 1 - holdout["label"].mean())
            floor = max(self.cfg.min_accuracy, majority + self.cfg.min_skill)
            mlflow.log_metric("holdout_majority_rate", majority)
            if acc < floor:
                decision, reason = "failed_floor", (f"holdout accuracy {acc:.3f} < floor {floor:.3f} "
                                                    f"(majority class {majority:.3f} + {self.cfg.min_skill})")
            elif champ_acc is not None and acc < champ_acc + self.cfg.min_gain:
                decision, reason = "kept_champion", f"challenger {acc:.3f} did not beat champion {champ_acc:.3f} + {self.cfg.min_gain}"
            else:
                decision = "promoted"
                reason = "first model" if champ_acc is None else f"challenger {acc:.3f} >= champion {champ_acc:.3f} + {self.cfg.min_gain}"
            mlflow.set_tags({"gate_decision": decision, "gate_reason": reason})

            if decision != "failed_floor":
                info = mlflow.sklearn.log_model(model, name="model", registered_model_name=registry.MODEL_NAME,
                                                input_example=holdout[FEATURES].astype(float).head(2),
                                                skops_trusted_types=registry.TRUSTED_TYPES)
                res.challenger_version = str(info.registered_model_version)
            if decision == "promoted":
                registry.promote(self.client, res.challenger_version, reason)

        res.status, res.reason = decision, reason
        res.champion_after = registry.champion_version(self.client)
        res.extra["mlflow_run_id"] = run.info.run_id
        res.extra["shuffle_labels"] = shuffle_labels
        return self._record(res)

"""Model registry on MLflow (SQLite backend, local artifacts): versions, champion alias, rollback."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402
from mlflow import MlflowClient  # noqa: E402

MODEL_NAME = "elec2-updown"
CHAMPION = "champion"
# Models are serialized with skops (not pickle). Only this one sklearn type, produced by our own
# training code, is trusted on load; anything else in a model file is refused.
TRUSTED_TYPES = ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]


def configure(state: Path) -> MlflowClient:
    state.mkdir(parents=True, exist_ok=True)
    uri = f"sqlite:///{(state / 'mlflow.db').resolve()}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    exp = mlflow.get_experiment_by_name("elec2-retraining")
    if exp is None:
        mlflow.create_experiment("elec2-retraining", artifact_location=(state / "artifacts").resolve().as_uri())
    mlflow.set_experiment("elec2-retraining")
    return MlflowClient()


def champion_version(client: MlflowClient) -> str | None:
    try:
        return str(client.get_model_version_by_alias(MODEL_NAME, CHAMPION).version)
    except mlflow.exceptions.MlflowException:
        return None


def load(version: str):
    return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version}")


def promote(client: MlflowClient, version: str, reason: str) -> None:
    previous = champion_version(client)
    client.set_registered_model_alias(MODEL_NAME, CHAMPION, version)
    client.set_model_version_tag(MODEL_NAME, version, "promoted_reason", reason)
    if previous is not None:
        client.set_model_version_tag(MODEL_NAME, version, "replaced_version", previous)


def rollback(client: MlflowClient, reason: str) -> tuple[str, str]:
    """Point the champion alias back at the version the current champion replaced."""
    current = champion_version(client)
    if current is None:
        raise RuntimeError("no champion to roll back")
    mv = client.get_model_version(MODEL_NAME, current)
    previous = mv.tags.get("replaced_version")
    if not previous:
        raise RuntimeError(f"champion v{current} has no earlier version to roll back to")
    client.set_registered_model_alias(MODEL_NAME, CHAMPION, previous)
    client.set_model_version_tag(MODEL_NAME, current, "rolled_back_reason", reason)
    return current, previous


def history(client: MlflowClient) -> list[dict]:
    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    except mlflow.exceptions.MlflowException:
        return []
    champ = champion_version(client)
    out = []
    for mv in sorted(versions, key=lambda m: int(m.version)):
        run = client.get_run(mv.run_id)
        out.append({
            "version": str(mv.version),
            "champion": str(mv.version) == champ,
            "trained_through": run.data.tags.get("trained_through"),
            "holdout_accuracy": run.data.metrics.get("challenger_holdout_accuracy"),
            "decision": run.data.tags.get("gate_decision"),
            "tags": dict(mv.tags),
        })
    return out

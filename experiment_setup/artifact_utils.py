"""Utility functions for working with MLflow-stored training artifacts."""

import pickle
import tempfile
from datetime import datetime
from typing import Any

import mlflow


def run_id_from_result(result: dict[str, Any]) -> str:
    """Get the MLflow run ID from a training result dictionary."""
    run_id = result.get("mlflow_run_id")
    if not run_id:
        raise ValueError("result does not contain 'mlflow_run_id'")
    return run_id


def load_artifacts_from_run(run_id: str) -> dict[str, Any]:
    """Download and deserialise artifacts.pkl from an MLflow run."""
    with tempfile.TemporaryDirectory() as tmp:
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="artifacts.pkl", dst_path=tmp
        )
        with open(local_path, "rb") as f:
            return pickle.load(f)


def update_mlflow_artifacts(run_id: str, updates: dict[str, Any]) -> None:
    """Download, merge updates into, and re-upload artifacts.pkl for a run."""
    artifacts = load_artifacts_from_run(run_id)
    artifacts.update(updates)
    artifacts.setdefault("retrain_history", [])
    artifacts["retrain_history"].append(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            **updates.get("retrain_metadata", {}),
        }
    )
    client = mlflow.tracking.MlflowClient()
    with tempfile.TemporaryDirectory() as tmp:
        pkl_path = f"{tmp}/artifacts.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(artifacts, f)
        client.log_artifact(run_id, pkl_path)

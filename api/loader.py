"""Load exported training runs from MLflow for inference."""

from __future__ import annotations

import pickle
import tempfile
from typing import Any

import mlflow
import mlflow.pytorch
import torch

from ._label_utils import label_encoder_from_class_dict


def load_model(run_id: str) -> dict[str, Any]:
    """Load a trained model and its preprocessing artifacts from an MLflow run.

    Args:
        run_id: MLflow run ID produced by any runner with ``save=True``.

    Returns:
        Dict containing ``model``, ``model_type``, ``energy_model``,
        ``artifacts``, ``label_enc``, ``class_dict``, ``config``,
        ``device``, ``num_classes``, and ``mlflow_run_id``.
    """
    run = mlflow.get_run(run_id)

    with tempfile.TemporaryDirectory() as tmp:
        try:
            local_path = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path="artifacts.pkl", dst_path=tmp
            )
        except Exception as exc:
            raise ValueError(
                f"Run {run_id!r} has no artifacts.pkl — it was likely saved before "
                "artifact logging was added and cannot be used for inference. "
                "Retrain or pick a different run."
            ) from exc
        with open(local_path, "rb") as f:
            artifacts: dict[str, Any] = pickle.load(f)

    model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    model_type = str(artifacts.get("model_type", run.data.params.get("model_type", ""))).strip()
    energy_model = bool(artifacts.get("energy_model", False))

    label_enc = _get_label_enc(artifacts, energy_model)
    class_dict = {int(k): str(v) for k, v in label_enc.id_to_label.items()} if label_enc else {}
    num_classes = label_enc.num_classes if label_enc else 0

    config = {
        "run_name": run.info.run_name,
        "mlflow_run_id": run_id,
        **run.data.params,
    }

    return {
        "model": model,
        "model_type": model_type,
        "energy_model": energy_model,
        "artifacts": artifacts,
        "label_enc": label_enc,
        "class_dict": class_dict,
        "config": config,
        "device": device,
        "num_classes": num_classes,
        "mlflow_run_id": run_id,
    }


def _get_label_enc(artifacts: dict[str, Any], energy_model: bool):
    """Extract the label encoder from an artifacts dict regardless of architecture."""
    if "label_enc" in artifacts:
        return artifacts["label_enc"]
    if energy_model and "energy_enc" in artifacts:
        return artifacts["energy_enc"]
    if not energy_model and "damage_enc" in artifacts:
        return artifacts["damage_enc"]
    raw = artifacts.get("class_dict")
    if raw:
        return label_encoder_from_class_dict(raw)
    return None

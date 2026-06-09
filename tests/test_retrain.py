"""Tests for retrain / fine-tuning functionality.

Drop this file into the repository as:

    tests/test_retrain.py

These tests assume the retrain implementation exposes:

    api.retrain.retrain(...)
    api.retrain.register_retrainer(...)
    api.retrain._RETRAINERS

The TF-IDF tests use the existing raw_csv fixture from tests/conftest.py on the
dev branch. The BERT test is dispatcher-level and mocked, so it does not
download Hugging Face weights or run a real BERT training loop.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
import torch


# ---------------------------------------------------------------------------
# Base model fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def base_tfidf_energy_model(raw_csv):
    """Train a small TF-IDF base model for retrain tests."""
    from api.train import train

    result = train(
        train_path=str(raw_csv["train"]),
        valid_path=str(raw_csv["valid"]),
        test_path=str(raw_csv["test"]),
        model_type="energy",
        architecture="tf_idf",
        train_config={
            "epochs": 1,
            "patience": 1,
            "run_name": "pytest_base_tfidf_energy",
            "verbose": False,
        },
    )
    return result, result["mlflow_run_id"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_mlflow_artifacts(run_id: str) -> dict[str, Any]:
    from experiment_setup.artifact_utils import load_artifacts_from_run
    return load_artifacts_from_run(run_id)


def _fake_embedding_matrix(
    vocab: dict[str, int],
    *,
    model_name: str = "pytest/fake",
    device: str | torch.device | None = None,
    verbose: bool = False,
) -> torch.Tensor:
    """Small deterministic embedding matrix used to avoid Hugging Face downloads."""
    size = max(vocab.values()) + 1 if vocab else 1
    embed_dim = 4
    matrix = torch.arange(size * embed_dim, dtype=torch.float32).reshape(size, embed_dim)
    if device is not None:
        matrix = matrix.to(device)
    return matrix


def _get_saved_embedding_matrix(artifacts: dict[str, Any]) -> torch.Tensor | None:
    """Accept a few reasonable key names for saved TF-IDF embedding artifacts."""
    for key in ("embedding_matrix", "tfidf_embedding_matrix", "embeddings"):
        if key in artifacts:
            value = artifacts[key]
            if isinstance(value, torch.Tensor):
                return value
            return torch.as_tensor(value)
    return None


# ---------------------------------------------------------------------------
# TF-IDF refresh retraining
# ---------------------------------------------------------------------------

def test_tfidf_refresh_retrains_and_updates_embedding_artifacts(
    raw_csv,
    base_tfidf_energy_model,
    monkeypatch,
):
    """Refresh mode should rebuild TF-IDF artifacts and save dense embeddings."""
    from api.loader import load_model
    from api.retrain import retrain
    import modules.embedding.safety_bert_static as safety_static

    _, old_run_id = base_tfidf_energy_model

    monkeypatch.setattr(
        safety_static,
        "get_safety_bert_embedding_matrix",
        _fake_embedding_matrix,
    )

    result = retrain(
        run_id=old_run_id,
        train_path=str(raw_csv["train"]),
        valid_path=str(raw_csv["valid"]),
        test_path=str(raw_csv["test"]),
        mode="refresh",
        train_config={
            "epochs": 1,
            "patience": 1,
            "run_name": "pytest_tfidf_refresh_embed",
            "feature_representation": "tfidf_embed_avg",
            "embedding_model_name": "pytest/fake",
            "verbose": False,
        },
    )

    new_run_id = result["mlflow_run_id"]
    assert new_run_id
    assert new_run_id != old_run_id

    artifacts = _load_mlflow_artifacts(new_run_id)

    assert artifacts["energy_model"] is True
    assert artifacts["feature_representation"] == "tfidf_embed_avg"
    assert artifacts["embedding_model_name"] == "pytest/fake"
    assert "vectorizer" in artifacts
    assert "label_enc" in artifacts

    embedding_matrix = _get_saved_embedding_matrix(artifacts)
    assert embedding_matrix is not None, (
        "Expected TF-IDF embedding-average retraining to save the embedding "
        "matrix in artifacts.pkl under one of: embedding_matrix, "
        "tfidf_embedding_matrix, embeddings."
    )
    assert embedding_matrix.ndim == 2
    assert embedding_matrix.shape[1] == 4

    saved_input_dim = artifacts.get("input_dim")
    if saved_input_dim is not None:
        assert int(saved_input_dim) == int(embedding_matrix.shape[1])

        bundle = load_model(new_run_id)
        assert bundle["model_type"].lower() == "tf_idf"
        assert bundle["num_classes"] == artifacts["label_enc"].num_classes


# ---------------------------------------------------------------------------
# TF-IDF continue retraining
# ---------------------------------------------------------------------------

def test_tfidf_continue_keeps_existing_vectorizer_and_saves_new_artifacts(
    raw_csv,
    base_tfidf_energy_model,
):
    """Continue mode should train from the previous TF-IDF checkpoint."""
    from api.retrain import retrain

    _, old_run_id = base_tfidf_energy_model
    old_artifacts = _load_mlflow_artifacts(old_run_id)
    old_vocab = dict(old_artifacts["vectorizer"].vocab)

    result = retrain(
        run_id=old_run_id,
        train_path=str(raw_csv["train"]),
        valid_path=str(raw_csv["valid"]),
        test_path=str(raw_csv["test"]),
        mode="continue",
        train_config={
            "epochs": 1,
            "patience": 1,
            "run_name": "pytest_tfidf_continue",
            "verbose": False,
        },
    )

    new_run_id = result["mlflow_run_id"]
    assert new_run_id
    assert new_run_id != old_run_id

    new_artifacts = _load_mlflow_artifacts(new_run_id)

    assert dict(new_artifacts["vectorizer"].vocab) == old_vocab
    assert new_artifacts["label_enc"].id_to_label == old_artifacts["label_enc"].id_to_label
    assert new_artifacts.get("retrain_mode") == "continue"


def test_tfidf_continue_rejects_unseen_labels(
    raw_csv,
    base_tfidf_energy_model,
    tmp_path,
):
    """Continue mode should not silently add new output classes."""
    from api.retrain import retrain

    _, old_run_id = base_tfidf_energy_model

    bad_train = pd.read_csv(raw_csv["train"])
    bad_train.loc[0, "Energy Type"] = "Chemical"

    bad_train_path = tmp_path / "bad_train.csv"
    bad_train.to_csv(bad_train_path, index=False)

    with pytest.raises(ValueError, match="Unknown label"):
        retrain(
            run_id=old_run_id,
            train_path=str(bad_train_path),
            valid_path=str(raw_csv["valid"]),
            test_path=str(raw_csv["test"]),
            mode="continue",
            train_config={
                "epochs": 1,
                "patience": 1,
                "run_name": "pytest_tfidf_continue_bad_label",
                "verbose": False,
            },
        )


# ---------------------------------------------------------------------------
# BERT dispatch without slow model downloads
# ---------------------------------------------------------------------------

def test_retrain_auto_dispatches_bert_to_continue_without_huggingface(
    raw_csv,
    monkeypatch,
):
    """mode='auto' should route BERT models to continue-mode retraining."""
    import api.retrain as retrain_module

    captured: dict[str, Any] = {}

    class DummyLabelEncoder:
        num_classes = 2
        id_to_label = {0: "Electrical", 1: "Vehicular"}

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)

        def forward(self, x):
            return self.linear(x)

    def fake_load_model(run_id: str) -> dict[str, Any]:
        return {
            "model": DummyModel(),
            "model_type": "bert",
            "energy_model": True,
            "artifacts": {
                "energy_model": True,
                "text_col": "description",
                "model_name": "bert-base-uncased",
                "tokenizer_name": "bert-base-uncased",
                "max_length": 16,
                "pooling": "mean",
                "fine_tune": True,
                "batch_size": 2,
            },
            "label_enc": DummyLabelEncoder(),
            "config": {
                "mlflow_run_id": run_id,
                "run_name": "old_bert_run",
                "model_type": "BERT",
            },
            "device": torch.device("cpu"),
            "num_classes": 2,
            "class_dict": {0: "Electrical", 1: "Vehicular"},
            "mlflow_run_id": run_id,
        }

    def fake_bert_retrainer(
        bundle: dict[str, Any],
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        test_df: pd.DataFrame,
        text_col: str,
        train_config: dict[str, Any],
    ) -> dict[str, Any]:
        captured["bundle"] = bundle
        captured["train_rows"] = len(train_df)
        captured["valid_rows"] = len(valid_df)
        captured["test_rows"] = len(test_df)
        captured["text_col"] = text_col
        captured["train_config"] = train_config
        return {
            "mlflow_run_id": "mock_run_id",
            "best_metric_value": 0.5,
        }

    monkeypatch.setattr(retrain_module, "load_model", fake_load_model)
    monkeypatch.setitem(retrain_module._RETRAINERS, "bert", fake_bert_retrainer)

    result = retrain_module.retrain(
        run_id="mock_old_bert_run_id",
        train_path=str(raw_csv["train"]),
        valid_path=str(raw_csv["valid"]),
        test_path=str(raw_csv["test"]),
        mode="auto",
        train_config={"epochs": 1, "learning_rate": 1e-5},
    )

    assert result["best_metric_value"] == 0.5
    assert captured["bundle"]["model_type"] == "bert"
    assert captured["text_col"] == "description"
    assert captured["train_config"]["epochs"] == 1
    assert captured["train_config"]["learning_rate"] == 1e-5
    assert captured["train_rows"] > 0
    assert captured["valid_rows"] > 0
    assert captured["test_rows"] > 0


# ---------------------------------------------------------------------------
# Extensibility / future architectures
# ---------------------------------------------------------------------------

def test_retrain_refresh_delegates_to_existing_train_api_for_future_models(
    raw_csv,
    monkeypatch,
):
    """Refresh mode should be architecture-agnostic."""
    import api.retrain as retrain_module

    captured: dict[str, Any] = {}

    def fake_load_model(run_id: str) -> dict[str, Any]:
        return {
            "model_type": "looped_transformer",
            "energy_model": False,
            "artifacts": {"text_col": "description"},
            "config": {
                "mlflow_run_id": run_id,
                "run_name": "old_looped_damage",
            },
            "mlflow_run_id": run_id,
        }

    def fake_train_from_splits(
        train_path: str,
        valid_path: str,
        test_path: str,
        model_type: str,
        architecture: str,
        train_config: dict | None = None,
        text_col: str = "description",
    ) -> dict[str, Any]:
        captured.update(
            {
                "train_path": train_path,
                "valid_path": valid_path,
                "test_path": test_path,
                "model_type": model_type,
                "architecture": architecture,
                "train_config": train_config or {},
                "text_col": text_col,
            }
        )
        return {
            "mlflow_run_id": "mock_refreshed_run_id",
            "best_metric_value": 0.25,
        }

    monkeypatch.setattr(retrain_module, "load_model", fake_load_model)
    monkeypatch.setattr(retrain_module, "train_from_splits", fake_train_from_splits)

    result = retrain_module.retrain(
        run_id="mock_old_looped_run_id",
        train_path=str(raw_csv["train"]),
        valid_path=str(raw_csv["valid"]),
        test_path=str(raw_csv["test"]),
        mode="refresh",
        train_config={"epochs": 1},
    )

    assert result["best_metric_value"] == 0.25
    assert captured["model_type"] == "damage"
    assert captured["architecture"] == "looped_transformer"
    assert captured["text_col"] == "description"
    assert captured["train_config"]["epochs"] == 1

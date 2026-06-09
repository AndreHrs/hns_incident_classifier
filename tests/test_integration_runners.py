"""Integration tests for experiment runner artifact-saving behaviour.

Each test verifies that a runner logs the correct artifacts to MLflow and that
the artifacts dict contains the expected keys.

Fast (default): TF-IDF, BiGRU
Slow (opt-in):  BERT, LoopedTransformer — run with: pytest -m slow

Run with (from project root): pytest tests/test_integration_runners.py -v
"""

from __future__ import annotations

import pytest

from api.loader import load_model


# ---------------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------------

class TestTFIDFRunnerArtifacts:
    def test_energy_has_run_id(self, tfidf_energy_model):
        _, run_id = tfidf_energy_model
        assert run_id

    def test_energy_artifacts_have_vectorizer(self, tfidf_energy_model):
        _, run_id = tfidf_energy_model
        bundle = load_model(run_id)
        assert "vectorizer" in bundle["artifacts"]

    def test_energy_artifacts_have_label_enc(self, tfidf_energy_model):
        _, run_id = tfidf_energy_model
        bundle = load_model(run_id)
        assert bundle["label_enc"] is not None

    def test_energy_artifacts_flag_true(self, tfidf_energy_model):
        _, run_id = tfidf_energy_model
        bundle = load_model(run_id)
        assert bundle["energy_model"] is True

    def test_energy_result_has_best_metric(self, tfidf_energy_model):
        result, _ = tfidf_energy_model
        assert isinstance(result.get("best_metric_value"), float)

    def test_damage_artifacts_flag_false(self, tfidf_damage_model):
        _, run_id = tfidf_damage_model
        bundle = load_model(run_id)
        assert bundle["energy_model"] is False

    def test_damage_has_run_id(self, tfidf_damage_model):
        _, run_id = tfidf_damage_model
        assert run_id


# ---------------------------------------------------------------------------
# BiGRU
# ---------------------------------------------------------------------------

class TestBiGRURunnerArtifacts:
    def test_has_run_id(self, bigru_energy_model):
        _, run_id = bigru_energy_model
        assert run_id

    def test_artifacts_have_seq_enc(self, bigru_energy_model):
        _, run_id = bigru_energy_model
        bundle = load_model(run_id)
        assert "seq_enc" in bundle["artifacts"]

    def test_artifacts_have_vocab_enc(self, bigru_energy_model):
        _, run_id = bigru_energy_model
        bundle = load_model(run_id)
        assert "vocab_enc" in bundle["artifacts"]

    def test_artifacts_have_max_len(self, bigru_energy_model):
        _, run_id = bigru_energy_model
        bundle = load_model(run_id)
        assert "max_len" in bundle["artifacts"]

    def test_artifacts_have_energy_enc(self, bigru_energy_model):
        _, run_id = bigru_energy_model
        bundle = load_model(run_id)
        assert "energy_enc" in bundle["artifacts"]

    def test_artifacts_have_damage_enc(self, bigru_energy_model):
        _, run_id = bigru_energy_model
        bundle = load_model(run_id)
        assert "damage_enc" in bundle["artifacts"]

    def test_artifacts_flag_true(self, bigru_energy_model):
        _, run_id = bigru_energy_model
        bundle = load_model(run_id)
        assert bundle["energy_model"] is True

    def test_best_metric_is_float(self, bigru_energy_model):
        result, _ = bigru_energy_model
        assert isinstance(result.get("best_metric_value"), float)


# ---------------------------------------------------------------------------
# BERT  (slow — needs HuggingFace weights download)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestBERTRunnerArtifacts:
    def test_has_run_id(self, bert_energy_model):
        _, run_id = bert_energy_model
        assert run_id

    def test_artifacts_have_label_enc(self, bert_energy_model):
        _, run_id = bert_energy_model
        bundle = load_model(run_id)
        assert bundle["label_enc"] is not None

    def test_artifacts_have_max_length(self, bert_energy_model):
        _, run_id = bert_energy_model
        bundle = load_model(run_id)
        assert "max_length" in bundle["artifacts"]

    def test_artifacts_flag_true(self, bert_energy_model):
        _, run_id = bert_energy_model
        bundle = load_model(run_id)
        assert bundle["energy_model"] is True


# ---------------------------------------------------------------------------
# LoopedTransformer  (slow — needs HuggingFace tokeniser)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestLoopedTransformerRunnerArtifacts:
    def test_has_run_id(self, looped_energy_model):
        _, run_id = looped_energy_model
        assert run_id

    def test_artifacts_have_label_enc(self, looped_energy_model):
        _, run_id = looped_energy_model
        bundle = load_model(run_id)
        assert bundle["label_enc"] is not None

    def test_artifacts_have_vocab_size(self, looped_energy_model):
        _, run_id = looped_energy_model
        bundle = load_model(run_id)
        assert "vocab_size" in bundle["artifacts"]

    def test_artifacts_have_d_model(self, looped_energy_model):
        _, run_id = looped_energy_model
        bundle = load_model(run_id)
        assert "d_model" in bundle["artifacts"]

    def test_artifacts_flag_true(self, looped_energy_model):
        _, run_id = looped_energy_model
        bundle = load_model(run_id)
        assert bundle["energy_model"] is True

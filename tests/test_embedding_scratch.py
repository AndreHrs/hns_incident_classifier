"""
Unit tests for the scratch embedding backend in modules/embedding.

Run with (from the project root):
    pytest tests/test_embedding_scratch.py -v
"""

import pytest
import torch

from modules.embedding import ScratchEmbeddingConfig, TrainableEmbeddingBackend


# ── ScratchEmbeddingConfig ───────────────────────────────────────────────────

class TestScratchEmbeddingConfig:
    def test_valid_config_passes(self):
        config = ScratchEmbeddingConfig(
            vocab_size=100,
            embedding_dim=32,
            pad_idx=0,
            dropout=0.1,
        )
        config.validate()  # should not raise

    def test_invalid_vocab_size_raises(self):
        config = ScratchEmbeddingConfig(
            vocab_size=0,
            embedding_dim=32,
            pad_idx=0,
            dropout=0.1,
        )
        with pytest.raises(ValueError):
            config.validate()

    def test_invalid_embedding_dim_raises(self):
        config = ScratchEmbeddingConfig(
            vocab_size=100,
            embedding_dim=0,
            pad_idx=0,
            dropout=0.1,
        )
        with pytest.raises(ValueError):
            config.validate()

    def test_invalid_pad_idx_raises(self):
        config = ScratchEmbeddingConfig(
            vocab_size=100,
            embedding_dim=32,
            pad_idx=100,
            dropout=0.1,
        )
        with pytest.raises(ValueError):
            config.validate()

    def test_invalid_dropout_raises(self):
        config = ScratchEmbeddingConfig(
            vocab_size=100,
            embedding_dim=32,
            pad_idx=0,
            dropout=1.0,
        )
        with pytest.raises(ValueError):
            config.validate()


# ── TrainableEmbeddingBackend ────────────────────────────────────────────────

class TestTrainableEmbeddingBackend:
    @pytest.fixture
    def config(self):
        return ScratchEmbeddingConfig(
            vocab_size=50,
            embedding_dim=16,
            pad_idx=0,
            dropout=0.0,
        )

    @pytest.fixture
    def backend(self, config):
        return TrainableEmbeddingBackend(config)

    @pytest.fixture
    def token_ids(self):
        return torch.tensor(
            [
                [2, 5, 7, 0, 0],
                [3, 4, 8, 9, 0],
            ],
            dtype=torch.long,
        )

    def test_output_shapes(self, backend, token_ids):
        output = backend(token_ids)

        assert output.token_embeddings.shape == (2, 5, 16)
        assert output.attention_mask.shape == (2, 5)
        assert output.sentence_embedding.shape == (2, 16)

    def test_attention_mask_respects_padding(self, backend, token_ids):
        output = backend(token_ids)

        expected = torch.tensor(
            [
                [1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0],
            ],
            dtype=torch.long,
        )
        assert torch.equal(output.attention_mask, expected)

    def test_pad_row_is_zeroed_on_initialization(self, backend):
        pad_vector = backend.embedding.weight[backend.config.pad_idx]
        assert torch.allclose(pad_vector, torch.zeros_like(pad_vector))

    def test_get_output_dim(self, backend):
        assert backend.get_output_dim() == 16

    def test_forward_requires_long_dtype(self, backend):
        token_ids = torch.tensor([[1.0, 2.0, 0.0]], dtype=torch.float32)
        with pytest.raises(TypeError):
            backend(token_ids)

    def test_sentence_embedding_is_masked_mean(self, backend):
        # Make embeddings deterministic for easier testing
        with torch.no_grad():
            backend.embedding.weight.zero_()
            backend.embedding.weight[2].fill_(2.0)
            backend.embedding.weight[5].fill_(4.0)
            backend.embedding.weight[7].fill_(6.0)

        token_ids = torch.tensor([[2, 5, 7, 0]], dtype=torch.long)
        output = backend(token_ids)

        # Mean of [2, 4, 6] = 4 for every embedding dimension
        expected = torch.full((1, 16), 4.0)
        assert torch.allclose(output.sentence_embedding, expected)

    def test_freeze_sets_requires_grad_false(self, backend):
        backend.freeze()
        assert backend.embedding.weight.requires_grad is False

    def test_unfreeze_sets_requires_grad_true(self, backend):
        backend.freeze()
        backend.unfreeze()
        assert backend.embedding.weight.requires_grad is True
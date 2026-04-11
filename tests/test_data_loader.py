"""
Unit tests for modules/data_loader.

Run with (from the project root): pytest tests/test_data_loader.py -v
"""

import pandas as pd
import pytest
import torch

from modules.data_loader import df_to_dataloader


@pytest.fixture
def sample_df():
    """Minimal DataFrame with integer-encoded labels and token-id lists."""
    return pd.DataFrame(
        {
            "tokens": [[1, 2, 3], [4, 5], [6, 7, 8, 9]],
            "energy": [0, 1, 2],
            "risk": [1, 0, 1],
        }
    )


class TestDfToDataloader:
    def _make_dl(self, df, **kwargs):
        defaults = dict(
            tokens_col="tokens",
            energy_col="energy",
            risk_col="risk",
            batch_size=8,
            shuffle=False,
        )
        defaults.update(kwargs)
        return df_to_dataloader(df, **defaults)

    def test_returns_dataloader(self, sample_df):
        dl = self._make_dl(sample_df)
        assert isinstance(dl, torch.utils.data.DataLoader)

    def test_batch_has_four_tensors(self, sample_df):
        dl = self._make_dl(sample_df)
        batch = next(iter(dl))
        assert len(batch) == 4

    def test_batch_order_D_DL_Energy_Risk(self, sample_df):
        dl = self._make_dl(sample_df)
        D, DL, Energy, Risk = next(iter(dl))
        assert D.shape[0] == len(sample_df)  # batch dimension
        assert DL.shape == (len(sample_df),)
        assert Energy.shape == (len(sample_df),)
        assert Risk.shape == (len(sample_df),)

    def test_all_tensors_are_long(self, sample_df):
        dl = self._make_dl(sample_df)
        for tensor in next(iter(dl)):
            assert tensor.dtype == torch.long

    def test_D_is_padded_to_max_length(self, sample_df):
        # longest sequence has 4 tokens
        dl = self._make_dl(sample_df)
        D, *_ = next(iter(dl))
        assert D.shape[1] == 4

    def test_lengths_match_original_sequence_lengths(self, sample_df):
        dl = self._make_dl(sample_df)
        _, DL, *_ = next(iter(dl))
        expected = torch.tensor([3, 2, 4], dtype=torch.long)
        assert torch.equal(DL, expected)

    def test_padding_uses_pad_id(self, sample_df):
        pad_id = 99
        dl = self._make_dl(sample_df, pad_id=pad_id)
        D, DL, *_ = next(iter(dl))
        # row with length 2 should have pad_id in remaining positions
        short_row_idx = DL.tolist().index(2)
        assert D[short_row_idx, 2].item() == pad_id
        assert D[short_row_idx, 3].item() == pad_id

    def test_labels_match_dataframe_values(self, sample_df):
        dl = self._make_dl(sample_df)
        _, _, Energy, Risk = next(iter(dl))
        assert torch.equal(Energy, torch.tensor([0, 1, 2], dtype=torch.long))
        assert torch.equal(Risk, torch.tensor([1, 0, 1], dtype=torch.long))

    def test_batch_size_splits_correctly(self, sample_df):
        dl = self._make_dl(sample_df, batch_size=2)
        batches = list(iter(dl))
        assert len(batches) == 2  # 3 rows → batches of 2 and 1
        assert batches[0][0].shape[0] == 2
        assert batches[1][0].shape[0] == 1

    def test_raises_on_string_labels(self, sample_df):
        df = sample_df.copy()
        df["energy"] = ["low", "medium", "high"]
        with pytest.raises((ValueError, RuntimeError, TypeError)):
            dl = self._make_dl(df)
            next(iter(dl))

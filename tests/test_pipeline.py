"""
Pipeline integration tests.

Covers:
    - CSV input loading (tests/test_data/csv_dataset.csv)
    - End-to-end: CSV → DataFrame → DataLoader (batch format verified)

Preprocessing is covered separately in test_pre_processing.py.
Encoding is a stub and is therefore excluded here.

Run with (from the project root): pytest tests/test_pipeline.py -v
"""

from pathlib import Path

import pandas as pd
import pytest
import torch

from modules.data_loader import df_to_dataloader

CSV_PATH = Path(__file__).parent / "test_data" / "csv_dataset.csv"
EXPECTED_COLUMNS = {"reference", "datetime", "description", "energy_type", "potential_damage"}


@pytest.fixture
def raw_df():
    return pd.read_csv(CSV_PATH)


@pytest.fixture
def encoded_df(raw_df):
    """Simulate post-encoding state: token lists added, labels already integers."""
    df = raw_df.copy()
    df["description_tokens"] = [[1, 2, 3], [4, 5], [6, 7, 8, 9], [1, 3], [2, 4, 5, 6, 7]]
    return df


# ── CSV loading ───────────────────────────────────────────────────────────────

class TestCSVLoading:
    def test_loads_without_error(self):
        df = pd.read_csv(CSV_PATH)
        assert df is not None

    def test_expected_columns_present(self, raw_df):
        assert EXPECTED_COLUMNS.issubset(set(raw_df.columns))

    def test_row_count(self, raw_df):
        assert len(raw_df) == 5

    def test_label_columns_are_integer(self, raw_df):
        assert raw_df["energy_type"].dtype in (int, "int64")
        assert raw_df["potential_damage"].dtype in (int, "int64")

    def test_no_null_descriptions(self, raw_df):
        assert raw_df["description"].notna().all()


# ── End-to-end: CSV → DataLoader ─────────────────────────────────────────────

class TestCSVToDataloader:
    def _make_dl(self, df, **kwargs):
        defaults = dict(
            tokens_col="description_tokens",
            energy_col="energy_type",
            risk_col="potential_damage",
            batch_size=8,
            shuffle=False,
        )
        defaults.update(kwargs)
        return df_to_dataloader(df, **defaults)

    def test_produces_dataloader(self, encoded_df):
        dl = self._make_dl(encoded_df)
        assert isinstance(dl, torch.utils.data.DataLoader)

    def test_batch_tuple_length(self, encoded_df):
        dl = self._make_dl(encoded_df)
        batch = next(iter(dl))
        assert len(batch) == 4

    def test_batch_covers_all_rows(self, encoded_df):
        dl = self._make_dl(encoded_df)
        D, DL, Energy, Risk = next(iter(dl))
        assert D.shape[0] == len(encoded_df)

    def test_lengths_reflect_token_lists(self, encoded_df):
        dl = self._make_dl(encoded_df)
        _, DL, *_ = next(iter(dl))
        expected_lengths = torch.tensor(
            [len(seq) for seq in encoded_df["description_tokens"]],
            dtype=torch.long,
        )
        assert torch.equal(DL, expected_lengths)

    def test_labels_match_csv_values(self, encoded_df):
        dl = self._make_dl(encoded_df)
        _, _, Energy, Risk = next(iter(dl))
        assert torch.equal(Energy, torch.tensor(encoded_df["energy_type"].tolist(), dtype=torch.long))
        assert torch.equal(Risk,   torch.tensor(encoded_df["potential_damage"].tolist(), dtype=torch.long))
"""
Unit tests for modules/encoding.

Run with (from the project root):
    pytest tests/test_encoding.py -v
"""

import pytest

from modules.encoding import VocabEncoder, SequenceEncoder, LabelEncoder


# ── VocabEncoder ─────────────────────────────────────────────────────────────

class TestVocabEncoder:
    @pytest.fixture
    def token_sequences(self):
        return [
            ["worker", "fell", "from", "ladder"],
            ["worker", "reported", "pain"],
            ["vehicle", "hit", "worker"],
        ]

    @pytest.fixture
    def vocab(self, token_sequences):
        vocab = VocabEncoder(min_freq=1)
        vocab.fit(token_sequences)
        return vocab

    def test_special_tokens_present(self, vocab):
        assert vocab.token_to_id["<pad>"] == 0
        assert vocab.token_to_id["<unk>"] == 1

    def test_vocab_size_greater_than_special_tokens(self, vocab):
        assert vocab.vocab_size > 2

    def test_encode_known_token(self, vocab):
        token_id = vocab.encode_token("worker")
        assert isinstance(token_id, int)
        assert token_id != vocab.unk_id

    def test_encode_unknown_token_returns_unk(self, vocab):
        token_id = vocab.encode_token("nonexistent_token")
        assert token_id == vocab.unk_id

    def test_encode_decode_roundtrip(self, vocab):
        tokens = ["worker", "fell", "from", "ladder"]
        ids = vocab.encode_tokens(tokens)
        decoded = vocab.decode_ids(ids)
        assert decoded == tokens

    def test_decode_unknown_id_returns_unk_token(self, vocab):
        decoded = vocab.decode_token(999999)
        assert decoded == "<unk>"

    def test_fit_with_min_freq_filters_rare_tokens(self, token_sequences):
        vocab = VocabEncoder(min_freq=2)
        vocab.fit(token_sequences)

        # "worker" appears twice and should stay
        assert vocab.encode_token("worker") != vocab.unk_id

        # "ladder" appears once and should be filtered out
        assert vocab.encode_token("ladder") == vocab.unk_id

    def test_access_before_fit_raises(self):
        vocab = VocabEncoder()
        with pytest.raises(ValueError):
            _ = vocab.vocab_size

    def test_invalid_min_freq_raises(self):
        with pytest.raises(ValueError):
            VocabEncoder(min_freq=0)


# ── SequenceEncoder ──────────────────────────────────────────────────────────

class TestSequenceEncoder:
    @pytest.fixture
    def vocab(self):
        token_sequences = [
            ["worker", "fell", "from", "ladder"],
            ["vehicle", "hit", "worker"],
        ]
        vocab = VocabEncoder(min_freq=1)
        vocab.fit(token_sequences)
        return vocab

    def test_encode_sequence_pads_to_max_length(self, vocab):
        encoder = SequenceEncoder(vocab_encoder=vocab, max_length=6)
        encoded = encoder.encode_sequence(["worker", "fell"])

        assert len(encoded) == 6
        assert encoded[-1] == vocab.pad_id
        assert encoded[-2] == vocab.pad_id

    def test_encode_sequence_truncates_post(self, vocab):
        encoder = SequenceEncoder(
            vocab_encoder=vocab,
            max_length=3,
            truncating="post",
        )
        encoded = encoder.encode_sequence(["worker", "fell", "from", "ladder"])
        decoded = vocab.decode_ids(encoded)

        assert len(encoded) == 3
        assert decoded == ["worker", "fell", "from"]

    def test_encode_sequence_truncates_pre(self, vocab):
        encoder = SequenceEncoder(
            vocab_encoder=vocab,
            max_length=3,
            truncating="pre",
        )
        encoded = encoder.encode_sequence(["worker", "fell", "from", "ladder"])
        decoded = vocab.decode_ids(encoded)

        assert len(encoded) == 3
        assert decoded == ["fell", "from", "ladder"]

    def test_padding_pre(self, vocab):
        encoder = SequenceEncoder(
            vocab_encoder=vocab,
            max_length=5,
            padding="pre",
        )
        encoded = encoder.encode_sequence(["worker", "fell"])

        assert len(encoded) == 5
        assert encoded[0] == vocab.pad_id
        assert encoded[1] == vocab.pad_id
        assert encoded[2] == vocab.pad_id

    def test_build_attention_mask_marks_non_pad_as_one(self, vocab):
        encoder = SequenceEncoder(vocab_encoder=vocab, max_length=5)
        encoded = encoder.encode_sequence(["worker", "fell"])
        mask = encoder.build_attention_mask(encoded)

        assert mask == [1, 1, 0, 0, 0]

    def test_encode_sequences_batch(self, vocab):
        encoder = SequenceEncoder(vocab_encoder=vocab, max_length=4)
        sequences = [["worker", "fell"], ["vehicle", "hit", "worker"]]
        encoded_batch = encoder.encode_sequences(sequences)

        assert len(encoded_batch) == 2
        assert all(len(seq) == 4 for seq in encoded_batch)

    def test_invalid_max_length_raises(self, vocab):
        with pytest.raises(ValueError):
            SequenceEncoder(vocab_encoder=vocab, max_length=0)

    def test_invalid_padding_raises(self, vocab):
        with pytest.raises(ValueError):
            SequenceEncoder(vocab_encoder=vocab, max_length=5, padding="middle")

    def test_invalid_truncating_raises(self, vocab):
        with pytest.raises(ValueError):
            SequenceEncoder(vocab_encoder=vocab, max_length=5, truncating="middle")


# ── LabelEncoder ─────────────────────────────────────────────────────────────

class TestLabelEncoder:
    @pytest.fixture
    def labels(self):
        return ["Vehicular", "Object", "Vehicular", "Other"]

    @pytest.fixture
    def encoder(self, labels):
        encoder = LabelEncoder()
        encoder.fit(labels)
        return encoder

    def test_num_classes(self, encoder):
        assert encoder.num_classes == 3

    def test_encode_known_label(self, encoder):
        encoded = encoder.encode("Vehicular")
        assert isinstance(encoded, int)

    def test_encode_many(self, encoder):
        encoded = encoder.encode_many(["Vehicular", "Object"])
        assert len(encoded) == 2
        assert all(isinstance(x, int) for x in encoded)

    def test_decode_roundtrip(self, encoder):
        label = "Object"
        encoded = encoder.encode(label)
        decoded = encoder.decode(encoded)
        assert decoded == label

    def test_unknown_label_raises(self, encoder):
        with pytest.raises(ValueError):
            encoder.encode("UnknownLabel")

    def test_unknown_label_id_raises(self, encoder):
        with pytest.raises(ValueError):
            encoder.decode(999)

    def test_access_before_fit_raises(self):
        encoder = LabelEncoder()
        with pytest.raises(ValueError):
            _ = encoder.num_classes
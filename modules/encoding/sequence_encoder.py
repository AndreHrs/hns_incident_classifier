"""Pad and truncate token-id sequences using a fitted vocabulary."""

from typing import Iterable

from .vocab_encoder import VocabEncoder


class SequenceEncoder:
    """Encode token sequences into padded/truncated integer sequences.

    Uses a fitted :class:`VocabEncoder` to convert tokens into ids, then applies
    optional padding and truncation to a fixed maximum length.

    :param vocab_encoder: Fitted vocabulary encoder.
    :type vocab_encoder: VocabEncoder
    :param max_length: Fixed output length for each sequence.
    :type max_length: int
    :param padding: Padding strategy. Supported values:
        - ``"post"``: pad at the end
        - ``"pre"``: pad at the beginning
    :type padding: str
    :param truncating: Truncation strategy. Supported values:
        - ``"post"``: truncate from the end
        - ``"pre"``: truncate from the beginning
    :type truncating: str
    """

    def __init__(
        self,
        vocab_encoder: VocabEncoder,
        max_length: int,
        padding: str = "post",
        truncating: str = "post",
    ):
        """Store vocabulary reference and fixed-length padding/truncation options."""
        if max_length <= 0:
            raise ValueError("max_length must be > 0")
        if padding not in {"post", "pre"}:
            raise ValueError("padding must be 'post' or 'pre'")
        if truncating not in {"post", "pre"}:
            raise ValueError("truncating must be 'post' or 'pre'")

        self.vocab_encoder = vocab_encoder
        self.max_length = max_length
        self.padding = padding
        self.truncating = truncating

    def encode_sequence(self, tokens: list[str]) -> list[int]:
        """Encode and pad/truncate a single token sequence.

        :param tokens: Input token list.
        :type tokens: list[str]
        :returns: Fixed-length list of token ids.
        :rtype: list[int]
        """
        token_ids = self.vocab_encoder.encode_tokens(tokens)
        return self._pad_or_truncate(token_ids)

    def encode_sequences(self, token_sequences: Iterable[list[str]]) -> list[list[int]]:
        """Encode and pad/truncate multiple token sequences.

        :param token_sequences: Iterable of token lists.
        :type token_sequences: Iterable[list[str]]
        :returns: List of fixed-length token-id sequences.
        :rtype: list[list[int]]
        """
        return [self.encode_sequence(seq) for seq in token_sequences]

    def build_attention_mask(self, encoded_sequence: list[int]) -> list[int]:
        """Build an attention mask for a padded encoded sequence.

        Non-padding ids are marked as 1, padding ids as 0.

        :param encoded_sequence: Padded encoded sequence.
        :type encoded_sequence: list[int]
        :returns: Binary attention mask.
        :rtype: list[int]
        """
        return [
            0 if token_id == self.vocab_encoder.pad_id else 1
            for token_id in encoded_sequence
        ]

    def build_attention_masks(
        self,
        encoded_sequences: Iterable[list[int]],
    ) -> list[list[int]]:
        """Build attention masks for multiple encoded sequences.

        :param encoded_sequences: Iterable of padded encoded sequences.
        :type encoded_sequences: Iterable[list[int]]
        :returns: List of binary attention masks.
        :rtype: list[list[int]]
        """
        return [self.build_attention_mask(seq) for seq in encoded_sequences]

    def _pad_or_truncate(self, token_ids: list[int]) -> list[int]:
        """Apply fixed-length padding/truncation to a token-id sequence.

        :param token_ids: Variable-length token-id sequence.
        :type token_ids: list[int]
        :returns: Fixed-length token-id sequence.
        :rtype: list[int]
        """
        if len(token_ids) > self.max_length:
            if self.truncating == "post":
                token_ids = token_ids[:self.max_length]
            else:
                token_ids = token_ids[-self.max_length:]

        pad_needed = self.max_length - len(token_ids)
        if pad_needed > 0:
            pad_values = [self.vocab_encoder.pad_id] * pad_needed
            if self.padding == "post":
                token_ids = token_ids + pad_values
            else:
                token_ids = pad_values + token_ids

        return token_ids
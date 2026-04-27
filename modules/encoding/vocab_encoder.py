"""Token vocabulary construction and id mapping for sequence models."""

from collections import Counter
from typing import Iterable


class VocabEncoder:
    """Build and manage a token vocabulary for sequence encoding.

    The vocabulary is built from tokenized training data only. Two special
    tokens are always included:

    - ``<pad>`` at index 0
    - ``<unk>`` at index 1

    Tokens can be filtered by minimum frequency and optional maximum
    vocabulary size.

    :param min_freq: Minimum token frequency required to enter the vocabulary.
        Defaults to 1.
    :type min_freq: int
    :param max_vocab_size: Maximum vocabulary size including special tokens.
        If ``None``, all eligible tokens are kept. Defaults to ``None``.
    :type max_vocab_size: int | None
    """

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"

    def __init__(self, min_freq: int = 1, max_vocab_size: int | None = None):
        """Initialize with frequency and size limits; vocabulary is filled in ``fit``."""
        if min_freq < 1:
            raise ValueError("min_freq must be >= 1")

        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size

        self.token_to_id: dict[str, int] = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
        }
        self.id_to_token: dict[int, str] = {
            0: self.PAD_TOKEN,
            1: self.UNK_TOKEN,
        }
        self._fitted = False

    def fit(self, token_sequences: Iterable[list[str]]) -> None:
        """Build the vocabulary from a collection of token sequences.

        :param token_sequences: Iterable of token lists.
        :type token_sequences: Iterable[list[str]]
        """
        counter = Counter()

        for seq in token_sequences:
            counter.update(seq)

        valid_tokens = [
            token for token, freq in counter.items()
            if freq >= self.min_freq
        ]

        # Sort by descending frequency, then alphabetically for reproducibility
        valid_tokens.sort(key=lambda tok: (-counter[tok], tok))

        if self.max_vocab_size is not None:
            remaining_slots = max(self.max_vocab_size - len(self.token_to_id), 0)
            valid_tokens = valid_tokens[:remaining_slots]

        next_id = len(self.token_to_id)
        for token in valid_tokens:
            if token not in self.token_to_id:
                self.token_to_id[token] = next_id
                self.id_to_token[next_id] = token
                next_id += 1

        self._fitted = True

    def encode_token(self, token: str) -> int:
        """Encode a single token into its integer id.

        Unknown tokens are mapped to ``<unk>``.

        :param token: Input token.
        :type token: str
        :returns: Integer token id.
        :rtype: int
        """
        self._check_is_fitted()
        return self.token_to_id.get(token, self.unk_id)

    def decode_token(self, token_id: int) -> str:
        """Decode a token id back to its token string.

        Unknown ids are mapped to ``<unk>``.

        :param token_id: Encoded token id.
        :type token_id: int
        :returns: Token string.
        :rtype: str
        """
        self._check_is_fitted()
        return self.id_to_token.get(token_id, self.UNK_TOKEN)

    def encode_tokens(self, tokens: list[str]) -> list[int]:
        """Encode a token list into a list of token ids.

        :param tokens: Token list.
        :type tokens: list[str]
        :returns: Encoded token ids.
        :rtype: list[int]
        """
        self._check_is_fitted()
        return [self.encode_token(token) for token in tokens]

    def decode_ids(self, token_ids: list[int]) -> list[str]:
        """Decode a list of token ids back into token strings.

        :param token_ids: List of token ids.
        :type token_ids: list[int]
        :returns: Decoded token strings.
        :rtype: list[str]
        """
        self._check_is_fitted()
        return [self.decode_token(token_id) for token_id in token_ids]

    @property
    def vocab_size(self) -> int:
        """Return the number of tokens in the vocabulary."""
        self._check_is_fitted()
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        """Return the padding token id."""
        return self.token_to_id[self.PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        """Return the unknown token id."""
        return self.token_to_id[self.UNK_TOKEN]

    def _check_is_fitted(self) -> None:
        if not self._fitted:
            raise ValueError("VocabEncoder has not been fitted yet.")
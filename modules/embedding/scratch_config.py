from dataclasses import dataclass


@dataclass
class ScratchEmbeddingConfig:
    """Configuration for a scratch-trained token embedding backend.

    :param vocab_size: Size of the vocabulary including special tokens.
    :type vocab_size: int
    :param embedding_dim: Dimensionality of each token embedding vector.
    :type embedding_dim: int
    :param pad_idx: Padding token index used by the vocabulary.
    :type pad_idx: int
    :param dropout: Dropout applied after embedding lookup.
    :type dropout: float
    """

    vocab_size: int
    embedding_dim: int
    pad_idx: int
    dropout: float = 0.0

    def validate(self) -> None:
        """Validate configuration values.

        :raises ValueError: If any field is invalid.
        """
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be > 0")
        if self.pad_idx < 0 or self.pad_idx >= self.vocab_size:
            raise ValueError("pad_idx must be within [0, vocab_size)")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in the range [0.0, 1.0)")
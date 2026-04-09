from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class EmbeddingOutput:
    """Standardized output container for all embedding backends.

    This dataclass provides a shared interface so downstream models can work
    with either scratch embeddings or transformer-based embeddings without
    changing their calling code.

    :param token_embeddings: Token-level embeddings of shape
        ``[batch_size, sequence_length, hidden_dim]``.
    :type token_embeddings: torch.Tensor
    :param attention_mask: Attention or validity mask of shape
        ``[batch_size, sequence_length]`` where 1 marks valid tokens and
        0 marks padding.
    :type attention_mask: torch.Tensor
    :param sentence_embedding: Optional pooled sequence representation of shape
        ``[batch_size, hidden_dim]``.
    :type sentence_embedding: torch.Tensor | None
    """

    token_embeddings: torch.Tensor
    attention_mask: torch.Tensor
    sentence_embedding: Optional[torch.Tensor] = None
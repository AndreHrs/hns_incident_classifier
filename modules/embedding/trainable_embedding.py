"""Trainable token embedding backend using ``torch.nn.Embedding``."""

import torch
import torch.nn as nn

from .base import BaseEmbeddingBackend
from .embedding_output import EmbeddingOutput
from .scratch_config import ScratchEmbeddingConfig


class TrainableEmbeddingBackend(nn.Module, BaseEmbeddingBackend):
    """Scratch-trained token embedding backend based on ``torch.nn.Embedding``.

    This backend consumes integer token-id tensors produced by encoding pipeline and returns token-level embeddings in the shared
    :class:`EmbeddingOutput` format.

    Input:
        ``token_ids`` of shape ``[batch_size, sequence_length]``

    Output:
        ``EmbeddingOutput`` with:
        - ``token_embeddings``: ``[batch_size, sequence_length, embedding_dim]``
        - ``attention_mask``: ``[batch_size, sequence_length]``
        - ``sentence_embedding``: masked mean-pooled representation
          ``[batch_size, embedding_dim]``

    :param config: Scratch embedding configuration.
    :type config: ScratchEmbeddingConfig
    """

    def __init__(self, config: ScratchEmbeddingConfig) -> None:
        """Build the embedding layer from ``config`` and initialize weights."""
        super().__init__()
        config.validate()
        self.config = config

        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_dim,
            padding_idx=config.pad_idx,
        )
        self.dropout = nn.Dropout(config.dropout)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize embedding weights and zero out the padding row."""
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.embedding.weight[self.config.pad_idx].fill_(0.0)

    def forward(self, token_ids: torch.Tensor) -> EmbeddingOutput:
        """Embed a batch of token-id sequences.

        :param token_ids: Tensor of token ids with shape
            ``[batch_size, sequence_length]`` and dtype ``torch.long``.
        :type token_ids: torch.Tensor
        :returns: Standardized embedding output.
        :rtype: EmbeddingOutput
        """
        if token_ids.dtype != torch.long:
            raise TypeError(
                f"TrainableEmbeddingBackend expects torch.long token ids, "
                f"got {token_ids.dtype}."
            )

        token_embeddings = self.embedding(token_ids)
        token_embeddings = self.dropout(token_embeddings)

        attention_mask = (token_ids != self.config.pad_idx).long()
        sentence_embedding = self._masked_mean_pool(
            token_embeddings=token_embeddings,
            attention_mask=attention_mask,
        )

        return EmbeddingOutput(
            token_embeddings=token_embeddings,
            attention_mask=attention_mask,
            sentence_embedding=sentence_embedding,
        )

    def _masked_mean_pool(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute masked mean pooling over token embeddings.

        :param token_embeddings: Embedded tokens of shape
            ``[batch_size, sequence_length, embedding_dim]``.
        :type token_embeddings: torch.Tensor
        :param attention_mask: Valid-token mask of shape
            ``[batch_size, sequence_length]``.
        :type attention_mask: torch.Tensor
        :returns: Mean-pooled sequence embeddings of shape
            ``[batch_size, embedding_dim]``.
        :rtype: torch.Tensor
        """
        mask = attention_mask.unsqueeze(-1).float()
        summed = (token_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts

    def get_output_dim(self) -> int:
        """Return the embedding hidden size."""
        return self.config.embedding_dim

    def freeze(self) -> None:
        """Freeze embedding parameters."""
        self.embedding.weight.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze embedding parameters."""
        self.embedding.weight.requires_grad = True
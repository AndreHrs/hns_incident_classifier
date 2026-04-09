from abc import ABC, abstractmethod

from .embedding_output import EmbeddingOutput


class BaseEmbeddingBackend(ABC):
    """Abstract base class for interchangeable embedding backends.

    All embedding backends should return an :class:`EmbeddingOutput` so that
    downstream models can remain agnostic to whether the source representation
    came from a scratch-trained embedding layer or a pretrained transformer.
    """

    @abstractmethod
    def get_output_dim(self) -> int:
        """Return the dimensionality of the embedding output.

        :returns: Hidden dimension size.
        :rtype: int
        """
        raise NotImplementedError

    @abstractmethod
    def freeze(self) -> None:
        """Freeze trainable parameters of the backend."""
        raise NotImplementedError

    @abstractmethod
    def unfreeze(self) -> None:
        """Unfreeze trainable parameters of the backend."""
        raise NotImplementedError
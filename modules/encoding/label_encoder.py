"""String label to integer id encoding (fit/encode/decode)."""

from typing import Iterable


class LabelEncoder:
    """Encode string class labels into integer ids and decode them back.

    Labels are sorted alphabetically during fitting for reproducibility.

    """

    def __init__(self):
        """Create an unfitted encoder with empty label mappings."""
        self.label_to_id: dict[str, int] = {}
        self.id_to_label: dict[int, str] = {}
        self._fitted = False

    def fit(self, labels: Iterable[str]) -> None:
        """Fit the encoder on a collection of labels.

        :param labels: Iterable of class labels.
        :type labels: Iterable[str]
        """
        unique_labels = sorted(set(labels))

        self.label_to_id = {
            label: idx for idx, label in enumerate(unique_labels)
        }
        self.id_to_label = {
            idx: label for label, idx in self.label_to_id.items()
        }
        self._fitted = True

    def encode(self, label: str) -> int:
        """Encode a single label into its integer id.

        :param label: Label string.
        :type label: str
        :returns: Encoded label id.
        :rtype: int
        """
        self._check_is_fitted()
        if label not in self.label_to_id:
            raise ValueError(f"Unknown label: {label}")
        return self.label_to_id[label]

    def encode_many(self, labels: Iterable[str]) -> list[int]:
        """Encode multiple labels into integer ids.

        :param labels: Iterable of label strings.
        :type labels: Iterable[str]
        :returns: Encoded label ids.
        :rtype: list[int]
        """
        self._check_is_fitted()
        return [self.encode(label) for label in labels]

    def decode(self, label_id: int) -> str:
        """Decode a label id back to its string label.

        :param label_id: Encoded label id.
        :type label_id: int
        :returns: Decoded label string.
        :rtype: str
        """
        self._check_is_fitted()
        if label_id not in self.id_to_label:
            raise ValueError(f"Unknown label id: {label_id}")
        return self.id_to_label[label_id]

    def decode_many(self, label_ids: Iterable[int]) -> list[str]:
        """Decode multiple label ids into string labels.

        :param label_ids: Iterable of encoded label ids.
        :type label_ids: Iterable[int]
        :returns: Decoded label strings.
        :rtype: list[str]
        """
        self._check_is_fitted()
        return [self.decode(label_id) for label_id in label_ids]

    @property
    def num_classes(self) -> int:
        """Return the number of fitted classes."""
        self._check_is_fitted()
        return len(self.label_to_id)

    def _check_is_fitted(self) -> None:
        if not self._fitted:
            raise ValueError("LabelEncoder has not been fitted yet.")
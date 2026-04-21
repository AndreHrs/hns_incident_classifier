"""Configurable loss functions for training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.

    Reduces the loss contribution from easy examples and focuses
    on hard misclassified examples.

    Args:
        gamma: Focusing parameter. Higher values focus more on hard examples.
        weight: Class weights tensor. Same as CrossEntropyLoss weight.
        reduction: 'mean' or 'sum'.
    """

    def __init__(self, gamma=2.0, weight=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        return focal_loss.sum()


def get_loss_function(criterion_type="cross_entropy", weight=None, **kwargs):
    """Return a loss function based on the criterion_type string.

    Args:
        criterion_type: One of 'cross_entropy', 'focal'.
        weight: Optional class weights tensor.
        **kwargs: Additional arguments passed to the loss function.

    Returns:
        A loss function (nn.Module).
    """
    criterion_type = criterion_type.lower()

    if criterion_type == "cross_entropy":
        return nn.CrossEntropyLoss(weight=weight)
    elif criterion_type == "focal":
        gamma = kwargs.get("gamma", 2.0)
        return FocalLoss(gamma=gamma, weight=weight)
    else:
        raise ValueError(
            f"Unknown criterion_type: '{criterion_type}'. "
            "Choose from: 'cross_entropy', 'focal'."
        )
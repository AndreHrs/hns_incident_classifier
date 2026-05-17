"""
Unit tests for modules/optimisation functions.

Covers pure/side-effect-free functions that don't require a full training run:
    - get_loss_function
    - FocalLoss
    - compute_class_weights
    - make_weighted_sampler
    - create_scheduler
    - normalise_scheduler_config
    - step_scheduler

Run with (from the project root): pytest tests/test_optimisation.py -v
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim


from modules.optimisation import (
                                    get_loss_function, 
                                    FocalLoss, 
                                    make_weighted_sampler, 
                                    create_scheduler, 
                                    normalise_scheduler_config,
                                 )
from modules.optimisation.imbalance import compute_class_weights
from modules.optimisation.scheduler import step_scheduler



# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_model(seq_len=8, out_features=3, use_length=False):
    """Minimal linear model compatible with _unpack_batch.
    Expects D of shape (batch, seq_len) and returns logits of shape (batch, out_features).
    """
    if use_length:
        class _WithLength(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(seq_len, out_features)

            def forward(self, x, lengths):
                return self.fc(x.float())

        return _WithLength()
    
    else:
        class _NoLength(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(seq_len, out_features)

            def forward(self, x):
                return self.fc(x.float())

        return _NoLength()


# ── get_loss_function ─────────────────────────────────────────────────────────


class TestGetLossFunction:
    def test_cross_entropy_returns_correct_type(self):
        criterion = get_loss_function("cross_entropy")
        assert isinstance(criterion, nn.CrossEntropyLoss)

    def test_focal_returns_correct_type(self):
        criterion = get_loss_function("focal")
        assert isinstance(criterion, FocalLoss)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            get_loss_function("unknown")

    def test_cross_entropy_with_weights(self):
        weights = torch.tensor([1.0, 2.0, 3.0])
        criterion = get_loss_function("cross_entropy", weight=weights)
        assert isinstance(criterion, nn.CrossEntropyLoss)

    def test_focal_forward_runs(self):
        criterion = get_loss_function("focal")
        logits = torch.randn(4, 3)
        targets = torch.tensor([0, 1, 2, 0])
        loss = criterion(logits, targets)
        assert loss.item() > 0


# ── Imbalance ─────────────────────────────────────────────────────────────────

class TestComputeClassWeights:
    def test_returns_correct_num_classes(self):
        labels = torch.tensor([0, 0, 0, 1, 2])
        weights = compute_class_weights(labels, num_classes=3)
        assert len(weights) == 3

    def test_minority_class_gets_higher_weight(self):
        labels = torch.tensor([0, 0, 0, 0, 1])
        weights = compute_class_weights(labels, num_classes=2)
        assert weights[1] > weights[0]

    def test_infers_num_classes(self):
        labels = torch.tensor([0, 1, 2])
        weights = compute_class_weights(labels)
        assert len(weights) == 3


class TestMakeWeightedSampler:
    def test_returns_sampler(self):
        from torch.utils.data import WeightedRandomSampler
        labels = torch.tensor([0, 0, 0, 1, 2])
        sampler = make_weighted_sampler(labels, num_classes=3)
        assert isinstance(sampler, WeightedRandomSampler)

    def test_sampler_length_matches_labels(self):
        from torch.utils.data import WeightedRandomSampler
        labels = torch.tensor([0, 0, 1, 1, 2])
        sampler = make_weighted_sampler(labels, num_classes=3)
        assert len(sampler) == len(labels)


# ── Scheduler Utilities ────────────────────────────────────────────────────────────────

class TestSchedulerUtilities:
    def test_false_scheduler_disables_scheduler(self):
        config = normalise_scheduler_config(
            scheduler=False,
            scheduler_step_per_batch=False,
            best_metric="loss",
            best_metric_mode="min",
        )
        assert config["name"] is None

    def test_none_scheduler_defaults_to_step_lr(self):
        config = normalise_scheduler_config(
            scheduler=None,
            scheduler_step_per_batch=False,
            best_metric="loss",
            best_metric_mode="min",
        )
        assert config["name"] == "StepLR"
        assert config["step_size"] == 1
        assert config["gamma"] == pytest.approx(0.95)

    def test_create_step_lr_scheduler(self):
        model = _make_model()
        opt = optim.Adam(model.parameters(), lr=1e-3)

        scheduler_config = {
            "name": "StepLR",
            "step_size": 1,
            "gamma": 0.5,
        }

        scheduler = create_scheduler(opt, scheduler_config)
        assert isinstance(scheduler, optim.lr_scheduler.StepLR)

    def test_step_lr_updates_learning_rate(self):
        model = _make_model()
        opt = optim.Adam(model.parameters(), lr=1e-3)

        scheduler_config = {
            "name": "StepLR",
            "step_size": 1,
            "gamma": 0.5,
        }

        scheduler = create_scheduler(opt, scheduler_config)
        opt.step()
        step_scheduler(scheduler, scheduler_config)

        assert opt.param_groups[0]["lr"] == pytest.approx(5e-4)

    def test_create_reduce_on_plateau_scheduler(self):
        model = _make_model()
        opt = optim.Adam(model.parameters(), lr=1e-3)

        scheduler_config = {
            "name": "ReduceLROnPlateau",
            "monitor": "f1_macro",
            "mode": "max",
            "factor": 0.5,
            "patience": 1,
        }

        scheduler = create_scheduler(opt, scheduler_config)
        assert isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau)

    def test_reduce_on_plateau_requires_monitor_metric(self):
        model = _make_model()
        opt = optim.Adam(model.parameters(), lr=1e-3)

        scheduler_config = {
            "name": "ReduceLROnPlateau",
            "monitor": "f1_macro",
            "mode": "max",
        }

        scheduler = create_scheduler(opt, scheduler_config)

        with pytest.raises(ValueError):
            step_scheduler(
                scheduler=scheduler,
                scheduler_config=scheduler_config,
                metrics={"loss": 1.0},
            )

    def test_create_cosine_scheduler(self):
        model = _make_model()
        opt = optim.Adam(model.parameters(), lr=1e-3)

        scheduler_config = {
            "name": "CosineAnnealingLR",
            "T_max": 10,
            "eta_min": 1e-6,
        }

        scheduler = create_scheduler(opt, scheduler_config)
        assert isinstance(scheduler, optim.lr_scheduler.CosineAnnealingLR)
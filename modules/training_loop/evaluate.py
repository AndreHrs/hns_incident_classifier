"""Training loop module for evaluation."""

import json
import torch
from pathlib import Path

from .metrics import _compute_classification_metrics
from .utility import _unpack_batch
from modules.inference import run_inference

FATAL_CLASSES = ["Single Fatality", "Multiple Fatality"]


def evaluate(config):
    """Evaluation function for evaluating the model on the test set.

    Extends the validation function with additional evaluation metrics
    based on project success criteria.

    Args:
        config (dict): A configuration dictionary containing the following keys:
            - model: The trained model to be evaluated.
            - test_dl: The test dataloader.
            - criterion: The loss function.
            - device: The device to run the evaluation on.
            - num_classes: The number of classes in the classification task.
            - class_names: List of class name strings.
            - save_dir: Directory where the model was saved (passed from RunSaver).
            - threshold: Confidence threshold for auto-classification. Defaults to 0.80.
            - temperature: Temperature value for scaling logits. Defaults to 1.5.
            - use_temperature: Whether to use temperature scaling. Defaults to True.

    Returns:
        metrics (dict): A dictionary containing:
            - accuracy, precision_macro, recall_macro, f1_macro
            - precision_weighted, recall_weighted, f1_weighted
            - loss
            - auto_classification_rate: proportion of predictions above threshold
            - meets_requirement: whether auto_classification_rate >= 0.70
            - fatal_flag_count: number of fatal predictions flagged
            - fatal_flag_rate: proportion of fatal predictions flagged
    """
    criterion = config["criterion"]
    class_names = config.get("class_names", [])
    threshold = config.get("threshold", 0.80)
    save_dir = config.get("save_dir", None)

    inference_result = run_inference(config)
    if inference_result is None:
        return {}

    all_preds = inference_result["all_preds"]
    all_targets = inference_result["all_targets"]
    all_probs = inference_result["all_probs"]
    total_examples = inference_result["total_examples"]

    # Recompute loss over test set (run_inference doesn't track loss)
    total_loss = 0.0
    with torch.no_grad():
        for batch in config["test_dl"]:
            logits, targets = _unpack_batch(batch, config)
            loss = criterion(logits, targets)
            total_loss += loss.item() * targets.size(0)
    avg_loss = total_loss / max(total_examples, 1)

    # Compute base classification metrics from metrics.py
    metrics = _compute_classification_metrics(
        y_true=all_targets,
        y_pred=all_preds,
        num_classes=config["num_classes"],
    )
    metrics["loss"] = avg_loss

    # Threshold analysis - auto classification rate
    max_probs = all_probs.max(dim=1).values
    high_confidence = (max_probs > threshold).sum().item()
    metrics["auto_classification_rate"] = high_confidence / max(total_examples, 1)
    metrics["meets_requirement"] = metrics["auto_classification_rate"] >= 0.70
    metrics["threshold_used"] = threshold

    # Fatal category flagging - 100% of fatal predictions must be flagged
    if class_names:
        fatal_indices = [
            class_names.index(c) for c in FATAL_CLASSES if c in class_names
        ]
        if fatal_indices:
            fatal_mask = torch.isin(all_preds, torch.tensor(fatal_indices))
            metrics["fatal_flag_count"] = fatal_mask.sum().item()
            metrics["fatal_flag_rate"] = fatal_mask.sum().item() / max(total_examples, 1)

    # Save evaluation results to save_dir
    if save_dir is not None:
        save_path = Path(save_dir) / "evaluation_metrics.json"
        serialisable_metrics = {
            k: v.tolist() if hasattr(v, "tolist") else v
            for k, v in metrics.items()
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(serialisable_metrics, f, indent=2)

    return metrics

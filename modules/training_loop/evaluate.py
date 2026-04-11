"""Training loop module for evaluation."""

import torch
import torch.nn.functional as F
from pathlib import Path
import json

from .metrics import _compute_classification_metrics
from .utility import _unpack_batch

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
    model = config["model"]
    criterion = config["criterion"]
    class_names = config.get("class_names", [])
    threshold = config.get("threshold", 0.80)
    temperature = config.get("temperature", 1.0)
    use_temperature = config.get("use_temperature", False)
    save_dir = config.get("save_dir", None)

    model.eval()
    total_loss, total_examples = 0.0, 0
    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for batch in config["test_dl"]:
            logits, targets = _unpack_batch(batch, config)
            loss = criterion(logits, targets)

            # Apply temperature scaling or standard softmax
            if use_temperature:
                scaled = logits / temperature
                probs = F.softmax(scaled, dim=1)
            else:
                probs = F.softmax(logits, dim=1)

            preds = probs.argmax(dim=1)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size
            all_preds.append(preds.detach().cpu())
            all_targets.append(targets.detach().cpu())
            all_probs.append(probs.detach().cpu())

    avg_loss = total_loss / max(total_examples, 1)
    all_preds = torch.cat(all_preds) if all_preds else torch.tensor([])
    all_targets = torch.cat(all_targets) if all_targets else torch.tensor([])
    all_probs = torch.cat(all_probs) if all_probs else torch.tensor([])

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
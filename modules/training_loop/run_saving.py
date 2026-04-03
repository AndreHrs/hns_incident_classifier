import json
import torch
from pathlib import Path

"""
RUN SAVING UTILITIES 
Includes:
    _initialise_history: Create a new history dictionary with empty lists for each metric.
    _append_metrics: Append the metrics for the current epoch to the history dictionary.
    _save_run_artifacts: Save the best model state dict and run summary to disk.
"""

# HISTORY DICTIONARY // Create a new history dictionary with empty lists for each metric
def _initialise_history():
    return {
        "train": {
            "loss": [],
            "accuracy": [],
            "precision_macro": [],
            "recall_macro": [],
            "f1_macro": [],
            "precision_weighted": [],
            "recall_weighted": [],
            "f1_weighted": [],
            "lr": [],
        },
        "val": {
            "loss": [],
            "accuracy": [],
            "precision_macro": [],
            "recall_macro": [],
            "f1_macro": [],
            "precision_weighted": [],
            "recall_weighted": [],
            "f1_weighted": [],
        },
        "epoch_time_sec": [],
    }

# APPEND METRICS TO HISTORY // Append the metrics for the current epoch to the history dictionary
def _append_metrics(history_section, metrics):
    for key in history_section.keys():
        if key in metrics:
            history_section[key].append(metrics[key])


# SAVE RUN ARTIFACTS // Save the best model state dict and run summary to disk
def _save_run_artifacts(config, run_summary):
    save_dir = Path(config["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = save_dir / config["save_name"]
    summary_path = save_dir / f"{Path(config['save_name']).stem}_run_summary.json"

    torch.save(run_summary["best_model_state_dict"], model_path)

    serialisable_config = {
        k: _serialise_value(v)
        for k, v in config.items()
        if k not in {"model", "train_dl", "valid_dl", "optimiser", "scheduler", "criterion"}
    }
    serialisable_config["metadata"] = {
        **serialisable_config.get("metadata", {}),
        "optimiser_defaults": _serialise_value(config["optimiser"].defaults if config["optimiser"] is not None else None),
    }

    serialisable_summary = {
        "config": serialisable_config,
        "best_epoch": run_summary["best_epoch"],
        "best_metric_name": run_summary["best_metric_name"],
        "best_metric_value": run_summary["best_metric_value"],
        "training_time_sec": run_summary["training_time_sec"],
        "history": _serialise_value(run_summary["history"]),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(serialisable_summary, f, indent=2)

    return str(model_path), str(summary_path)
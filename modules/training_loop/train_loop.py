"""Main training loop with early stopping and MLflow artifact logging."""

import copy
import os
import tempfile
import time

import mlflow
import mlflow.pytorch
import torch
from dotenv import load_dotenv

load_dotenv()
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:8080"))

from .one_epoch import train_one_epoch
from .validation import validate
from .run_saving import RunSaver
from .utility import _is_better
from .evaluate import evaluate
from ..optimisation.scheduler import step_scheduler


#  MAIN TRAINING LOOP // ensures all control variables are consistent // compatible with Dataloader-based pipelines
def train_model_loop(
    config,
):  # all top-level inputs stored in config dictionary, which is passed around to all functions that need it
    """Manage the entire training process, including early stopping.

    Args:
        config: Dictionary containing model, dataloaders, optimizer, scheduler,
            criterion, epochs, device, patience, and all other training parameters.

    Returns:
        Dictionary containing training history, best epoch, best metric value,
        best model state dict, and total training time.
    """
    run_saver = RunSaver()

    patience_counter = 0
    best_metric_value = None
    best_epoch = None
    best_model_state_dict = None

    verbose = config.get("verbose", True)

    experiment_name = (
        config.get("mlflow_experiment")
        or config.get("metadata", {}).get("target")
        or "hns_classification"
    )
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=config["run_name"]) as active_run:
        mlflow_run_id = active_run.info.run_id
        if verbose:
            print("=" * 120)
            print(f"Training the {config['model_type']} model")
            print(f"Run name: {config['run_name']}")
            print(
                f"Best model tracked by: {config['best_metric']} ({config['best_metric_mode']})"
            )
            print("=" * 120)

        training_start_time = time.time()

        for epoch in range(1, config["epochs"] + 1):
            epoch_start_time = time.time()

            train_metrics = train_one_epoch(config)
            val_metrics = validate(config)

            if config["scheduler"] is not None and not config["scheduler_step_per_batch"]:
                step_scheduler(
                    scheduler=config["scheduler"],
                    scheduler_config=config.get("scheduler_config"),
                    metrics=val_metrics,
                )

            run_saver.history["training"]["epoch_time_sec"].append(time.time() - epoch_start_time)

            run_saver.append_metrics("train", train_metrics)
            run_saver.append_metrics("val", val_metrics)

            mlflow.log_metrics(
                {
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "train_f1_macro": train_metrics["f1_macro"],
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_f1_macro": val_metrics["f1_macro"],
                    "val_f1_weighted": val_metrics["f1_weighted"],
                    "val_precision_macro": val_metrics["precision_macro"],
                    "val_recall_macro": val_metrics["recall_macro"],
                },
                step=epoch,
            )

            current_metric_value = val_metrics[config["best_metric"]]

            if verbose:
                print("-" * 120)
                print(
                    f"| Epoch {epoch:03d} "
                    f"| Time: {run_saver.history['training']['epoch_time_sec'][-1]:7.2f}s "
                    f"| Train Loss: {train_metrics['loss']:.4f} "
                    f"| Train Acc: {train_metrics['accuracy'] * 100:.2f}% "
                    f"| Val Loss: {val_metrics['loss']:.4f} "
                    f"| Val Acc: {val_metrics['accuracy'] * 100:.2f}% "
                    f"| Val F1 Macro: {val_metrics['f1_macro']:.4f} "
                    f"| Val F1 Weighted: {val_metrics['f1_weighted']:.4f} "
                    f"| Best {config['best_metric']}: {current_metric_value:.4f} |"
                )
                print("-" * 120)

            if _is_better(
                current_metric_value, best_metric_value, config["best_metric_mode"]
            ):
                best_metric_value = current_metric_value
                best_epoch = epoch
                patience_counter = 0
                best_model_state_dict = copy.deepcopy(config["model"].state_dict())
            else:
                patience_counter += 1
                if patience_counter >= config["patience"]:
                    if verbose:
                        print(f"Early stopping triggered at epoch {epoch}.")
                    break

        total_train_time = time.time() - training_start_time

        if best_model_state_dict is not None:
            config["model"].load_state_dict(best_model_state_dict)

        test_metrics = evaluate(config)
        run_saver.append_metrics("test", test_metrics, training=False)

        run_summary = {
            "config": config,
            "history": run_saver.history,
            "best_epoch": best_epoch,
            "best_metric_name": config["best_metric"],
            "best_metric_value": best_metric_value,
            "best_model_state_dict": best_model_state_dict,
            "training_time_sec": total_train_time,
            "mlflow_run_id": mlflow_run_id,
        }

        if config["save"] and best_model_state_dict is not None:
            # --- Params ---
            optimiser = config.get("optimiser")
            try:
                lr = optimiser.defaults.get("lr") if optimiser is not None else None
            except Exception:
                lr = None
            metadata = config.get("metadata", {})

            mlflow.log_params(
                {
                    "model_type": config.get("model_type"),
                    "model_class": metadata.get("model_class"),
                    "optimiser_class": metadata.get("optimiser_class"),
                    "scheduler_class": metadata.get("scheduler_class"),
                    "criterion_class": metadata.get("criterion_class"),
                    "criterion_type": config.get("criterion_type"),
                    "lr": lr,
                    "epochs_max": config.get("epochs"),
                    "patience": config.get("patience"),
                    "clip_grad_max_norm": config.get("clip_grad_max_norm"),
                    "scheduler_step_per_batch": config.get("scheduler_step_per_batch"),
                    "energy_model": config.get("energy_model"),
                    "target": experiment_name,
                    "search_parameters": str(config.get("parameters", {})),
                    "is_hyperparameter_search": config.get("is_hyperparameter_search", False),
                }
            )

            # --- Summary metrics at best epoch and on test set ---
            val_history = run_saver.history.get("training", {}).get("val", {})
            test_history = run_saver.history.get("test", {})
            idx = best_epoch - 1

            def _at_best(metric):
                vals = val_history.get(metric, [])
                return vals[idx] if idx < len(vals) else None

            def _last_test(metric):
                vals = test_history.get(metric, [])
                return vals[-1] if vals else None

            summary_metrics = {
                "best_epoch": best_epoch,
                "best_metric_value": best_metric_value,
                "training_time_sec": total_train_time,
            }
            for m in ("loss", "accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"):
                v = _at_best(m)
                if v is not None:
                    summary_metrics[f"best_val_{m}"] = v
            for m in (
                "loss", "accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro",
                "auto_classification_rate", "fatal_flag_rate", "fatal_accuracy",
                "confidence_high_rate", "confidence_medium_rate", "confidence_low_rate",
            ):
                v = _last_test(m)
                if v is not None:
                    summary_metrics[f"test_{m}"] = v
            for flag in ("req_high_confidence_met", "req_fatal_accuracy_met", "req_all_f1_targets_met"):
                v = _last_test(flag)
                if v is not None:
                    summary_metrics[flag] = int(v) if isinstance(v, bool) else v
            mlflow.log_metrics(summary_metrics)

            # --- Model and plots ---
            mlflow.pytorch.log_model(config["model"], "model")

            with tempfile.TemporaryDirectory() as tmp_dir:
                run_saver.plot_history(best_epoch, config, plot_dir=tmp_dir)
                mlflow.log_artifacts(tmp_dir, artifact_path="plots")

            if verbose:
                print(f"MLflow run logged: experiment='{experiment_name}', run='{config['run_name']}'")
                print(f"Total training time: {total_train_time:.4f}s")
                print(f"Best epoch: {best_epoch}")
                print(f"Best {config['best_metric']}: {best_metric_value:.6f}")

    return run_summary

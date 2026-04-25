"""Training loop module exposing the top-level training entry point."""

from .config import _build_train_config
from .train_loop import train_model_loop

__all__ = ["training", "_build_train_config", "train_model_loop"]

"""
TRAINING LOOP MODULE // Files related to the main training loop and its components
- config: 
        Function to build the training configuration dictionary, which includes all necessary parameters and objects for training
- one_epoch: 
        Function that trains the model for one epoch and returns the training loss and accuracy
- validation: 
        Function that evaluates the model on the validation set and returns the validation loss and accuracy
- metrics: 
        Function for computing classification metrics (e.g., accuracy, precision, recall, F1) based on the true labels and predicted labels
- run_saving: 
        Functions for initializing the training history, appending metrics to the history, and saving the best model and run summary to disk at the end of training
- utility: 
        Utility functions for 
                - safely getting the class name of an object 
                - converting various types of values (including tensors) to a format that can be easily saved in JSON or similar formats
                - unpacking batches from the dataloader and preparing them for model input
                - getting the learning rates from the optimizer's parameter groups
                - comparing the current metric value to the best metric value based on the specified mode (min or max)


Functionality notes (): 
 - Support for different loss functions 
    -> currently supports any loss function that can be called as `criterion(preds, labels)` and returns a scalar loss value. 
        -> Could add support for more complex loss functions that require additional inputs (e.g., class weights, sample weights, etc.)
 - More detailed model saving 
    -> model saving is now more detailed
 - More detailed logging (e.g., using TensorBoard or a logging library instead of print statements)
    -> currently uses print statements for logging, but could be extended to use a more robust logging framework or TensorBoard for better visualization of training progress and metrics
 - Support for resuming training from a checkpoint
    -> currently does not support resuming training from a checkpoint, 
        -> could be extended to allow loading a saved model state dict and training history to continue training from where it left off
 - More flexible learning rate scheduling (e.g., support for different schedulers, or custom scheduling logic)
    -> currently supports a default StepLR scheduler, or use of a custom scheduler if provided. 
        -> could be extended to support a wider range of built-in schedulers, or allow for custom scheduling logic to be implemented by the user
 - ....etc.
"""


# EXPOSED TRAINING FUNCTION // main function to call for training a model, which builds the config and calls the main training loop
def training(
    model,
    optimiser=None,
    train_dl=None,
    valid_dl=None,
    epochs=10,
    device="cpu",
    patience=3,
    criterion_weights=None,
    model_type="Simple",
    save=True,
    scheduler=None,  # need to be defined outside
    criterion_type="cross_entropy",
    criterion_args={}, 
    need_length=False,
    energy_model=False,
    best_metric="loss",  # must be in: "loss", "accuracy", "precision_macro", "recall_macro", "f1_macro", "precision_weighted", "recall_weighted", "f1_weighted"
    best_metric_mode=None,
    clip_grad_max_norm=1.0,
    scheduler_step_per_batch=False,
    save_dir="trained_models",
    run_name=None,
    compute_train_metrics=False,
    parameters=None,
    num_classes=None,
    extra_config=None,
    **_,
):
    """Build training config and run the full training loop.

    Args:
        model: PyTorch model to train.
        optimiser: Optimizer instance. Defaults to Adam with lr=1e-3.
        train_dl: DataLoader for training data.
        valid_dl: DataLoader for validation data.
        epochs: Number of training epochs.
        device: Device string, e.g. 'cpu' or 'cuda'.
        patience: Early stopping patience in epochs.
        criterion_weights: Optional class weights for the loss function.
        model_type: Label used for saving and logging.
        save: Whether to save model artifacts after training.
        scheduler: Learning rate scheduler. Defaults to StepLR.
        criterion_type: Loss function type. One of 'cross_entropy', 'focal'. Defaults to 'cross_entropy'.
        criterion_args: Dictionary of additional arguments for the loss function.
        need_length: Whether the model expects sequence lengths as input.
        energy_model: If True, predict energy type; otherwise predict risk.
        best_metric: Metric used to select the best model checkpoint.
        best_metric_mode: 'min' or 'max'. Inferred from best_metric if None.
        clip_grad_max_norm: Max norm for gradient clipping.
        scheduler_step_per_batch: Step scheduler per batch instead of per epoch.
        save_dir: Directory to save model artifacts.
        run_name: Optional name for the run.
        compute_train_metrics: Whether to compute metrics on the training set.
        parameters: The training parameters,
        num_classes: Number of output classes.
        extra_config: Optional dict of additional config keys to merge.

    Returns:
        Run summary dictionary with history, best epoch, and best metric value.
    """
    # All top-level inputs are collected into `config` and that config is passed everywhere else.
    # This keeps the function signatures clean and makes it easy to add new parameters without needing to change a lot of function signatures.
    train_config = _build_train_config(
        model=model,
        train_dl=train_dl,
        valid_dl=valid_dl,
        epochs=epochs,
        device=device,
        patience=patience,
        criterion_weights=criterion_weights,
        model_type=model_type,
        save=save,
        optimiser=optimiser,
        scheduler=scheduler,
        criterion_type=criterion_type,
        criterion_args=criterion_args,
        need_length=need_length,
        energy_model=energy_model,
        best_metric=best_metric,
        best_metric_mode=best_metric_mode,
        clip_grad_max_norm=clip_grad_max_norm,
        scheduler_step_per_batch=scheduler_step_per_batch,
        save_dir=save_dir,
        run_name=run_name,
        compute_train_metrics=compute_train_metrics,
        parameters=parameters,
        num_classes=num_classes,
        extra_config=extra_config,
    )

    return train_model_loop(train_config)

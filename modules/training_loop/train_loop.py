'''
TRAINING LOOP MODULE // MAIN TRAINING FUNCTION
- train_model_loop: Main training loop function that manages the entire training process, including early stopping
    - train_one_epoch: Function that trains the model for one epoch and returns the training loss and accuracy  
    - validate: Function that evaluates the model on the validation set and returns the validation loss and accuracy    

-> this should be adjusted so that different criterions can be used for energy and risk models, but for now we will just use the same criterion with different weights.

Functionality to add: 
 - Support for different loss functions 
 - More detailed model saving (e.g., saving optimizer state, epoch number, etc.)
 - More detailed logging (e.g., using TensorBoard or a logging library instead of print statements)
 - Support for resuming training from a checkpoint
 - More flexible learning rate scheduling (e.g., support for different schedulers, or custom scheduling logic)
 - ....
'''

import time
import torch
import torch.nn as nn
import torch.optim as optim
from .validation import validate
from .one_epoch import train_one_epoch

#  MAIN TRAINING LOOP // ensures all control variables are consistent // compatible with Dataloader-based pipelines
def train_model_loop(model, optimiser, train_dl, valid_dl, epochs, device, patience, criterion_weights, model_type='Simple', save=True):
    best_val_loss = float("inf")
    best_model = None
    patience_counter = 0
    epochs_history = {}
    epochs_history["train_acc"] = []
    epochs_history["val_acc"] = []
    epochs_history["train_loss"] = []
    epochs_history["val_loss"] = []

    scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=1, gamma=0.95)
    criterion_weights = criterion_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=criterion_weights)


    print("="*114)
    print(f'Training the {model_type} model')
    print("="*114)

    # Total training time
    training_start_time = time.time()

    # Begin training
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()

        tr_loss, tr_acc = train_one_epoch(model, train_dl, optimiser, scheduler, criterion, device, need_length, energy_model)
        val_loss, val_acc = validate(model, valid_dl, criterion, device, need_length, energy_model)
        epochs_history["train_acc"].append(tr_acc)
        epochs_history["val_acc"].append(val_acc)
        epochs_history["train_loss"].append(tr_loss)
        epochs_history["val_loss"].append(val_loss)

        print("-" * 114)
        print(f"| End of Epoch {epoch:3d} "
              f"| Time: {time.time() - epoch_start_time:6.2f}s "
              f"| Train Loss: {tr_loss:6.3f} "
              f"| Train Acc: {tr_acc:6.2f} "
              f"| Val Loss: {val_loss:6.3f} "
              f"| Val Acc: {val_acc:6.2f} |")
        print("-" * 114)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model = model.state_dict()  # or should i just use model
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break

        scheduler.step()

    total_train_time = time.time() - training_start_time

    # Save best model to device
    # REPLACE WITH MORE DETAILED MODEL SAVING FUNCTION - ISSUE #24
    if best_model and save:
        model.load_state_dict(best_model)
        torch.save(model.state_dict(), f"{model_type}_model.pt")
        print(f"Total training time: {total_train_time:.4f}s")
        print(f"\tBest validation loss: {best_val_loss:.4f}")
        print(f"\tModel saved to '{model_type}_model.pt'")

    return epochs_history
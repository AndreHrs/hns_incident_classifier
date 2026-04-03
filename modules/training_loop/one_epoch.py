import torch
import torch.nn as nn
import torch.optim as optim

# SINGLE EPOCH TRAINING LOOP // returning training accuracy and loss of one training epoch
def train_one_epoch(model, dataloader, optimiser, scheduler, criterion, device, need_length, energy_model):
    model.train()
    total_loss, total_correct, total_examples = 0.0, 0, 0

    for batch in dataloader:
        optimiser.zero_grad(set_to_none=True)  # does not zero // efficiency and clarity

        # With length -> Simple and BiDAF models
        if need_length == True:
            D, DL, Energy, Risk = batch
            D, DL, Energy, Risk = D.to(device), DL.to(device), Energy.to(device), Risk.to(device)
            logits = model(D, DL)
        # Without length -> Transformer model
        else:
            D, _, Energy, Risk = batch
            D, Energy, Risk = D.to(device), Energy.to(device), Risk.to(device)
            logits = model(D) #if 'Transformer' in type(model).__name__ else model(D, Energy)

        if energy_model:
            Y = Energy
        else:                
            Y = Risk

        # Compute loss & backpropagation
        loss = criterion(logits, Y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()

        # Performance tracking
        preds = logits.argmax(1)
        total_correct += (preds == Y).sum().item()
        total_loss += loss.item() * Y.size(0)
        total_examples += Y.size(0)

    avg_loss = total_loss / total_examples
    avg_acc = total_correct / total_examples * 100
    return avg_loss, avg_acc
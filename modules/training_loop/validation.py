import torch
import torch.nn as nn
import torch.optim as optim

# VALIDATION FUNCTION // Evaluation funciton used for validation
def validate(model, dataloader, criterion, device, need_length, energy_model): # add true/false statement for generating the f1, etc. stats?
    model.eval()
    total_loss, total_correct, total_examples = 0.0, 0, 0

    with torch.no_grad():
        for batch in dataloader:
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
            loss = criterion(logits, Y)
            preds = logits.argmax(1)
            total_correct += (preds == Y).sum().item()
            total_loss += loss.item() * Y.size(0)
            total_examples += Y.size(0)

    # Average the performance (accuracy & loss)
    avg_loss = total_loss / total_examples
    avg_acc = total_correct / total_examples * 100

    return avg_loss, avg_acc
"""Inference module for running model predictions on a dataset."""

import torch
import torch.nn.functional as F
from modules.training_loop.utility import _unpack_batch

def run_inference(config, dataloader=None):
    """Run model inference on a dataset, collecting predictions and probabilities.
    Can be called as a standalone eval step (no dataloader) or as a full inference pass that returns collected tensors for downstream metric computation.
    
    Args:
    
        config (dict): Configuration dict containing:
            - model: The PyTorch model to run.
            - device: Device string (e.g. 'cpu', 'cuda').
            - temperature (optional): Temperature for scaling logits. Default 1.0.
            - use_temperature (optional): Whether to apply temperature scaling. Default False.
            - test_dl (optional): Default dataloader if none passed explicitly.
        dataloader: Dataloader to run inference on. If None, falls back to config["test_dl"]. If still None, only model.eval() is called.
        
    Returns:
    
        dict with keys all_preds, all_targets, all_probs, total_examples if a dataloder is available, otherwise None.
    """
    model = config["model"]
    device = config.get("device", "cpu")
    temperature = config.get("temperature", 1.0)
    use_temperature = config.get("use_temperature", False)

    if use_temperature and temperature <= 0:
        raise ValueError("temperature must be > 0 when use_temperature=True")

    model.to(device)
    model.eval()

    dl = dataloader or config.get("test_dl")
    if dl is None:
        return None

    all_preds, all_targets, all_probs = [], [], []
    total_examples = 0

    with torch.no_grad():
        for batch in dl:
            logits, targets = _unpack_batch(batch, config)

            if use_temperature:
                logits = logits / temperature

            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            batch_size = targets.size(0)
            total_examples += batch_size

            all_preds.append(preds.detach().cpu())
            all_targets.append(targets.detach().cpu())
            all_probs.append(probs.detach().cpu())

    return {
        "all_preds": torch.cat(all_preds) if all_preds else torch.tensor([], dtype=torch.long),
        "all_targets": torch.cat(all_targets) if all_targets else torch.tensor([], dtype=torch.long),
        "all_probs": torch.cat(all_probs) if all_probs else torch.empty((0, config["num_classes"])),
        "total_examples": total_examples,
    }

# def run_inference(config, dataloader=None):
#     """Run model inference on a dataset, collecting predictions and probabilities.

#     Can be called as a standalone eval step (no dataloader) or as a full
#     inference pass that returns collected tensors for downstream metric computation.

#     Args:
#         config (dict): Configuration dict containing:
#             - model: The PyTorch model to run.
#             - device: Device string (e.g. 'cpu', 'cuda').
#             - need_length: Whether the model expects (D, DL) input.
#             - energy_model: Whether the target label is Energy (True) or Risk (False).
#             - temperature (optional): Temperature for scaling logits. Default 1.0.
#             - use_temperature (optional): Whether to apply temperature scaling. Default False.
#             - test_dl (optional): Default dataloader if none passed explicitly.
#         dataloader: Dataloader to run inference on. If None, falls back to
#                     config["test_dl"]. If still None, only model.eval() is called.

#     Returns:
#         dict with keys all_preds, all_targets, all_probs, total_examples
#         if a dataloader is available, otherwise None.
#     """
#     model = config["model"]
#     device = config.get("device", "cpu")
#     need_length = config.get("need_length", False)
#     energy_model = config.get("energy_model", True)
#     temperature = config.get("temperature", 1.0)
#     use_temperature = config.get("use_temperature", False)

#     model.eval()

#     dl = dataloader or config.get("test_dl")
#     if dl is None:
#         return None

#     all_preds, all_targets, all_probs = [], [], []
#     total_examples = 0

#     with torch.no_grad():
#         for batch in dl:
#             if need_length:
#                 D, DL, Energy, Risk = batch
#                 D, DL = D.to(device), DL.to(device)
#                 Energy, Risk = Energy.to(device), Risk.to(device)
#                 logits = model(D, DL)
#             else:
#                 D, _, Energy, Risk = batch
#                 D = D.to(device)
#                 Energy, Risk = Energy.to(device), Risk.to(device)
#                 logits = model(D)

#             targets = Energy if energy_model else Risk

#             if use_temperature:
#                 probs = F.softmax(logits / temperature, dim=1)
#             else:
#                 probs = F.softmax(logits, dim=1)

#             preds = probs.argmax(dim=1)

#             batch_size = targets.size(0)
#             total_examples += batch_size
#             all_preds.append(preds.detach().cpu())
#             all_targets.append(targets.detach().cpu())
#             all_probs.append(probs.detach().cpu())

#     return {
#         "all_preds": torch.cat(all_preds) if all_preds else torch.tensor([]),
#         "all_targets": torch.cat(all_targets) if all_targets else torch.tensor([]),
#         "all_probs": torch.cat(all_probs) if all_probs else torch.tensor([]),
#         "total_examples": total_examples,
#     }

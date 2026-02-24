"""
Engine for training and evaluating models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader



def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None, # None -> full training
) -> dict:
    
    model.train() # train mode

    total_loss = 0.0
    correct = 0
    total = 0

    # iterate over batches
    for batch_idx, (xb, yb, _paths) in enumerate(loader):
        
        # for faster training during development
        if max_batches is not None and batch_idx >= max_batches:
            break
        
        if batch_idx == 0: # debug
            print("[DEBUG] xb stats:", xb.min().item(), xb.max().item(), xb.mean().item(), xb.std().item())
            print("[DEBUG] yb stats:", yb.min().item(), yb.max().item())

        # forward
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)

        # backward + step
        optimizer.zero_grad()
        loss.backward()
        #print("grad mean:", model.classifier.weight.grad.abs().mean().item())
        optimizer.step()

        # stats
        bs = yb.size(0) # current batch size
        total += bs
        total_loss += loss.item() * bs  # weight by batch size

        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()

    return {
        "loss": total_loss / max(total, 1),
        "acc": correct / max(total, 1),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None # None -> full evaluation
) -> dict:
    
    model.eval() # eval mode

    total_loss = 0.0
    correct = 0
    total = 0

    # iterate over batches
    for batch_idx, (xb, yb, _paths) in enumerate(loader):
        
        # for faster evaluation during development
        if max_batches is not None and batch_idx >= max_batches:
            break
        
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)

        bs = yb.size(0) # current batch size
        total += bs
        total_loss += loss.item() * bs

        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()

    return {
        "loss": total_loss / max(total, 1),
        "acc": correct / max(total, 1),
    }
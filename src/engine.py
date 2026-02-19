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
) -> dict: 
    
    model.train() # train mode

    total_loss = 0.0
    correct = 0
    total = 0

    # iterate over batches
    for xb, yb, _paths in loader:
        # forward
        logits = model(xb)
        loss = criterion(logits, yb)

        # backward + step
        optimizer.zero_grad()
        loss.backward()
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
) -> dict:
    
    model.eval() # eval mode

    total_loss = 0.0
    correct = 0
    total = 0
    
    # iterate over batches
    for xb, yb, _paths in loader:
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
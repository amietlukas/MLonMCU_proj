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

        if batch_idx == 0:
            # keep CPU copies for debug
            xb_cpu = xb
            yb_cpu = yb
            paths0 = [str(p) for p in _paths]

            print("[DEBUG] xb stats:", xb_cpu.min().item(), xb_cpu.max().item(), xb_cpu.mean().item(), xb_cpu.std().item())
            print("[DEBUG] yb stats:", yb_cpu.min().item(), yb_cpu.max().item())

            print("[DEBUG] unique paths in batch:", len(set(paths0)), "/", len(paths0))
            print("[DEBUG] first 5 paths+labels:")
            for i in range(min(5, len(paths0))):
                print("   ", paths0[i], "label", int(yb_cpu[i].item()))

            diff01 = (xb_cpu[0] - xb_cpu[1]).abs().mean().item()
            diff02 = (xb_cpu[0] - xb_cpu[2]).abs().mean().item()
            print("[DEBUG] mean|x0-x1|:", diff01, "mean|x0-x2|:", diff02)

            counts_y = torch.bincount(yb_cpu, minlength=6)
            print("[DEBUG] label counts in batch:", counts_y.tolist())


        # forward
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)

        if batch_idx == 0:
            print("[DEBUG] logits mean/std:", logits.mean().item(), logits.std().item())
            probs = logits.softmax(dim=1)
            print("[DEBUG] probs mean/std:", probs.mean().item(), probs.std().item())
            print("[DEBUG] probs[0]:", probs[0].detach().cpu().tolist())
            preds0 = logits.argmax(dim=1).detach().cpu()
            counts = torch.bincount(preds0, minlength=probs.shape[1])
            print("[DEBUG] pred counts:", counts.tolist())
        # debug-------

        # backward + step
        optimizer.zero_grad()
        loss.backward()

        if batch_idx == 0:
            w = model.classifier.weight
            b = model.classifier.bias
            print("[DEBUG] loss:", loss.item())
            print("[DEBUG] head grad |w| mean:", w.grad.abs().mean().item(), "max:", w.grad.abs().max().item())
            if b is not None and b.grad is not None:
                print("[DEBUG] head grad |b| mean:", b.grad.abs().mean().item(), "max:", b.grad.abs().max().item())

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
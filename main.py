import torch
import torch.optim as optim
import torch.nn as nn
from pathlib import Path

from src.config import load_config
from src.data import build_dataloaders
from src.model import BaselineCNN
from src.engine import train_one_epoch, evaluate


def main():
    project_root = Path(__file__).resolve().parent  # IMPORTANT: main.py needs to be in the project root
    cfg = load_config(project_root / "config.yaml", project_root)  # load config and check global assumptions

    # ========== build dataloaders ==========
    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    # check dataloader
    xb, yb, _ = next(iter(train_loader))
    assert xb.ndim == 4 and xb.shape[1] == 1, xb.shape
    assert yb.ndim == 1, yb.shape

    # ========== init model ==========
    model = BaselineCNN(cfg)

    # ========== loss + optimizer ==========
    lr = float(cfg["train"]["lr"])
    epochs = int(cfg["train"]["epochs"])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ========== train loop ==========
    for epoch in range(1, epochs + 1):
        # train one epoch
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion)
        # evaluete on val set after each epoch
        val_metrics = evaluate(model, val_loader, criterion)

        print(
            f"epoch {epoch:02d}/{epochs} | "
            f"train loss {train_metrics['loss']:.4f} acc {train_metrics['acc']:.3f} | "
            f"val loss {val_metrics['loss']:.4f} acc {val_metrics['acc']:.3f}"
        )


if __name__ == "__main__":
    main()
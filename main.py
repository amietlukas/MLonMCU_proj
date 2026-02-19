"""
The orchestrator of the full pipeline.
From project root, run this file, together with a name.
"""

import argparse
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn as nn

from src.config import load_config
from src.data import build_dataloaders
from src.model import BaselineCNN
from src.engine import train_one_epoch, evaluate
from src.run import make_run_info

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", type=str, required=True, help="Run name, e.g. --name baseline128")
    return p.parse_args()

############## Pipeline starts here ##############
def main():
    args = parse_args()

    project_root = Path(__file__).resolve().parent  # IMPORTANT: main.py needs to be in the project root
    cfg = load_config(project_root / "config.yaml", project_root)  # load config and check global assumptions

    run = make_run_info(project_root, args.name)
    print(f"Run: {run.run_id}")

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
    best_val_acc = -1.0
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

        val_acc = float(val_metrics["acc"])
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # save current best version
            torch.save(
                {
                    "run_id": run.run_id,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "cfg": cfg,
                    "val_metrics": val_metrics,
                    "train_metrics": train_metrics,
                },
                run.checkpoint_path,
            )
            print(f"  saved best -> {run.checkpoint_path} (val acc {best_val_acc:.3f})")


if __name__ == "__main__":
    main()
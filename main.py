"""
The orchestrator of the full pipeline.
From project root, run this file: python main.py --name <name>
"""

import argparse
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn as nn

from src.config import load_config
from src.data import build_dataloaders, CLASS_NAMES
from src.model import BaselineCNN
from src.engine import train_one_epoch, evaluate
from src.run import make_run_info, save_config_snapshot
from src.checkpoint import save_checkpoint
from src.metrics import init_metrics_csv, append_metrics_csv
from src.model_utils import save_model_summary
from src.per_class_metrics import (
    compute_confusion_matrix,
    compute_per_class_metrics,
    save_confusion_matrix_csv,
    save_per_class_metrics_csv,
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", type=str, required=True, help="Run name, e.g. --name baseline128")
    return p.parse_args()


############## Pipeline starts here ##############
def main():
    args = parse_args()

    project_root = Path(__file__).resolve().parent  # IMPORTANT: main.py needs to be in the project root
    cfg = load_config(project_root / "config.yaml", project_root)  # load config and check global assumptions
    # for quick debugging, you can set max_train_batches and max_val_batches in config.yaml
    max_train_batches = cfg["train"].get("max_train_batches", None)
    max_val_batches = cfg["train"].get("max_val_batches", None)

    run = make_run_info(project_root, args.name)
    save_config_snapshot(cfg, run.config_snapshot_path)
    init_metrics_csv(run.metrics_csv_path)

    print(f"Run: {run.run_id}")
    print(f"Run dir:        {run.run_dir}")
    print(f"Config snapshot:{run.config_snapshot_path}")
    print(f"Metrics CSV:    {run.metrics_csv_path}")
    print(f"Best ckpt:      {run.best_ckpt_path}")

    # ========== build dataloaders ==========
    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    # check dataloader
    xb, yb, _ = next(iter(train_loader))
    assert xb.ndim == 4 and xb.shape[1] == 1, xb.shape
    assert yb.ndim == 1, yb.shape

    # ========== init model ==========
    model = BaselineCNN(cfg)
    save_model_summary(run.model_summary_path, model, cfg)
    print(f"Model summary:  {run.model_summary_path}", "\n")

    # ========== loss + optimizer ==========
    lr = float(cfg["train"]["lr"])
    epochs = int(cfg["train"]["epochs"])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ========== train loop ==========
    best_val_acc = -1.0
    for epoch in range(1, epochs + 1):

        # train one epoch
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, max_batches=max_train_batches
        )
        # evaluete on val set after each epoch
        val_metrics = evaluate(
            model, val_loader, criterion, max_batches=max_val_batches
        )

        print(
            f"epoch {epoch:02d}/{epochs} | "
            f"train loss {train_metrics['loss']:.4f}; acc {train_metrics['acc']:.3f} | "
            f"val loss {val_metrics['loss']:.4f}; acc {val_metrics['acc']:.3f}"
        )

        # save metrics
        append_metrics_csv(
            run.metrics_csv_path,
            epoch=epoch,
            train_loss=float(train_metrics["loss"]),
            train_acc=float(train_metrics["acc"]),
            val_loss=float(val_metrics["loss"]),
            val_acc=float(val_metrics["acc"]),
        )

        # new best model?
        val_acc = float(val_metrics["acc"])
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # save current best version
            save_checkpoint(
                run.best_ckpt_path,
                run_id=run.run_id,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
            )
            print(f" ->  NEW saved best: val acc {best_val_acc:.3f}")
        else:
            print(f" ->  NO NEW BEST; current best: {best_val_acc:.3f}")
    
    # ===== per-class metrics on BEST checkpoint (val set) =====
    ckpt = torch.load(run.best_ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])

    cm = compute_confusion_matrix(model, val_loader, num_classes=int(cfg["data"]["num_classes"]))
    save_confusion_matrix_csv(run.log_dir / "confusion_matrix.csv", cm, CLASS_NAMES)

    per_class = compute_per_class_metrics(cm)
    save_per_class_metrics_csv(run.log_dir / "per_class_metrics.csv", per_class, CLASS_NAMES)

    print("\n", f"Saved confusion matrix -> {run.log_dir / 'confusion_matrix.csv'}")
    print(f"Saved per-class metrics -> {run.log_dir / 'per_class_metrics.csv'}")


    print("\n", "------ FINISHED ------", "\n", "\n")

if __name__ == "__main__":
    main()
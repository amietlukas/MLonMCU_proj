"""
1) saves the metrics in a .csv file.
2) plot this csv file.
"""

from __future__ import annotations

import csv
from pathlib import Path
import matplotlib.pyplot as plt

# =========== CSV Metrics Logging ===========
def init_metrics_csv(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        return  # don't overwrite

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

def append_metrics_csv(
    path: str | Path,
    *,
    epoch: int,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
) -> None:
    path = Path(path)
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([epoch, train_loss, train_acc, val_loss, val_acc])



# =========== CSV Metrics plotting ===========
def plot_loss_acc(metrics_csv_path: str | Path) -> None:
    """
    Plot training + validation loss over epochs from the metrics.csv and save it.
    """

    metrics_csv_path = Path(metrics_csv_path)

    if not metrics_csv_path.exists():
        raise FileNotFoundError(f"metrics.csv not found: {metrics_csv_path}")

    epochs, train_loss, train_acc, val_loss, val_acc = [], [], [], [], []
    
    with metrics_csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            train_acc.append(float(row["train_acc"]))
            val_loss.append(float(row["val_loss"]))
            val_acc.append(float(row["val_acc"]))

    out_path = metrics_csv_path.parent / "loss_acc_curves.png"

    if len(epochs) == 0:
        # nothing to plot yet
        return

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # upper left: train loss
    axes[0, 0].plot(epochs, train_loss)
    axes[0, 0].set_title("Train Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True)

    # upper left: val loss
    axes[0, 1].plot(epochs, val_loss)
    axes[0, 1].set_title("Val Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].grid(True)

    # lower left: train acc
    axes[1, 0].plot(epochs, train_acc)
    axes[1, 0].set_title("Train Accuracy")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].grid(True)

    # lower right: val acc
    axes[1, 1].plot(epochs, val_acc)
    axes[1, 1].set_title("Validation Accuracy")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].grid(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

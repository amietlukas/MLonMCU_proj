from __future__ import annotations

import csv
from pathlib import Path


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
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def compute_confusion_matrix(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Returns confusion matrix cm with shape [C, C] where:
      rows = true class, cols = predicted class
    """
    model.eval()
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    for xb, yb, _paths in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        preds = logits.argmax(dim=1)

        for t, p in zip(yb.view(-1), preds.view(-1)):
            cm[int(t), int(p)] += 1

    return cm


def compute_per_class_metrics(cm: torch.Tensor) -> List[Dict[str, float]]:
    """
    Computes per-class precision/recall/F1 from confusion matrix.
    """
    C = cm.size(0)
    cm_f = cm.to(torch.float64)

    support = cm_f.sum(dim=1)     # true count per class
    predicted = cm_f.sum(dim=0)   # predicted count per class
    tp = cm_f.diag()

    out: List[Dict[str, float]] = []
    for c in range(C):
        supp = support[c].item()
        pred = predicted[c].item()
        tpc = tp[c].item()

        recall = tpc / supp if supp > 0 else 0.0
        precision = tpc / pred if pred > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        out.append(
            {
                "class_idx": float(c),
                "support": supp,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return out


def save_confusion_matrix_csv(path: str | Path, cm: torch.Tensor, class_names: List[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + class_names)
        for i, name in enumerate(class_names):
            w.writerow([name] + cm[i].tolist())


def save_per_class_metrics_csv(path: str | Path, rows: List[Dict[str, float]], class_names: List[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "support", "precision", "recall", "f1"])
        for r in rows:
            idx = int(r["class_idx"])
            w.writerow([class_names[idx], int(r["support"]), r["precision"], r["recall"], r["f1"]])
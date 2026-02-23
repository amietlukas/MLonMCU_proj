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
from src.data import build_dataloaders, build_datasets, build_transform, CLASS_NAMES
from src.model import BaselineCNN
from src.engine import train_one_epoch, evaluate
from src.run import make_run_info, save_config_snapshot
from src.checkpoint import save_checkpoint
from src.metrics import init_metrics_csv, append_metrics_csv
from src.model_utils import save_model_summary
from src.viz import save_transform_preview
from src.per_class_metrics import (
    compute_confusion_matrix,
    compute_per_class_metrics,
    save_confusion_matrix_csv,
    save_per_class_metrics_csv,
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", type=str, required=True, help="Run name, e.g. --name baseline128")
    p.add_argument("--checkpoint", type=str, default=None, help="Checkpoint dir name to resume from, e.g. --checkpoint baseline128-20260221-192155")
    return p.parse_args()


############## Pipeline starts here ##############
def main():
    args = parse_args()

    project_root = Path(__file__).resolve().parent  # IMPORTANT: main.py needs to be in the project root
    cfg = load_config(project_root / "config.yaml", project_root)  # load config and check global assumptions
    # for quick debugging, you can set max_train_batches and max_val_batches in config.yaml
    max_train_batches = cfg["train"].get("max_train_batches", None)
    max_val_batches = cfg["train"].get("max_val_batches", None)
    print("[DEBUG] max_train_batches:", max_train_batches, "max_val_batches:", max_val_batches)

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

    # ===== DEBUG: Check class distribution =====
    from collections import Counter
    train_labels = []
    for _, labels, _ in train_loader:
        train_labels.extend(labels.tolist())
    train_dist = Counter(train_labels)
    print("[DEBUG] Train set class distribution:", dict(sorted(train_dist.items())))
    
    val_labels = []
    for _, labels, _ in val_loader:
        val_labels.extend(labels.tolist())
    val_dist = Counter(val_labels)
    print("[DEBUG] Val set class distribution:", dict(sorted(val_dist.items())))

    # ===== debug: plot transformed images =====
    if cfg.get("debug", {}).get("plot_images", False):
        # build datasets so we can access raw paths + transform deterministically
        train_ds, _, _ = build_datasets(cfg)

        n_per_class = int(cfg.get("debug", {}).get("n_plot_per_class", 5))
        out_path = run.log_dir / "transform_preview_train.png"

        # IMPORTANT: use the same transform as training
        save_transform_preview(
            train_samples=train_ds.samples,
            class_names=CLASS_NAMES,
            transform=train_ds.transform,
            out_path=out_path,
            n_per_class=n_per_class,
            seed=int(cfg.get("seed", 0)),
        )
        print(f"[DEBUG] saved transform preview -> {out_path}")

    # ========== device setup ==========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # ========== init model ==========
    model = BaselineCNN(cfg)
    model = model.to(device)
    save_model_summary(run.model_summary_path, model, cfg)
    print(f"Model summary:  {run.model_summary_path}", "\n")

    # ========== loss + optimizer ==========
    lr = float(cfg["train"]["lr"])
    epochs = int(cfg["train"]["epochs"])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ========== load checkpoint if specified ==========
    start_epoch = 1
    best_val_acc = -1.0
    
    if args.checkpoint is not None:
        checkpoint_dir = project_root / "runs" / args.checkpoint
        checkpoint_path = checkpoint_dir / "checkpoints" / "best.pt"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading checkpoint from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        
        # Get metrics from checkpoint
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_acc = ckpt.get("val_metrics", {}).get("acc", -1.0)
        
        print(f"Resuming from epoch {start_epoch}, best val acc: {best_val_acc:.3f}\n")
        
        # Set start_epoch to 1
        start_epoch = 1

    # ========== train loop ==========
    for epoch in range(start_epoch, epochs + 1):

        # train one epoch
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device=device, max_batches=max_train_batches
        )
        # evaluete on val set after each epoch
        val_metrics = evaluate(
            model, val_loader, criterion, device=device, max_batches=max_val_batches
        )

        print(
            f"epoch {epoch:02d}/{epochs} | "
            f"TRAIN: loss {train_metrics['loss']:.4f}, acc {train_metrics['acc']:.3f} | "
            f"VAL: loss {val_metrics['loss']:.4f}, acc {val_metrics['acc']:.3f}"
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
            print(f" ->  new best: val acc {best_val_acc:.3f}")
        else:
            print(f" ->  no improvement, current best: {best_val_acc:.3f}")
    
    # ===== per-class metrics on BEST checkpoint (val set) =====
    ckpt = torch.load(run.best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    cm = compute_confusion_matrix(model, val_loader, num_classes=int(cfg["data"]["num_classes"]), device=device)
    save_confusion_matrix_csv(run.log_dir / "confusion_matrix.csv", cm, CLASS_NAMES)

    # ===== DEBUG: Print confusion matrix details =====
    print("\n[DEBUG] Confusion Matrix:")
    print(cm)
    print("[DEBUG] Predictions per class (column sums):", cm.sum(dim=0).tolist())
    print("[DEBUG] True instances per class (row sums):", cm.sum(dim=1).tolist())

    per_class = compute_per_class_metrics(cm)
    save_per_class_metrics_csv(run.log_dir / "per_class_metrics.csv", per_class, CLASS_NAMES)

    print("\n",f"Saved confusion matrix -> {run.log_dir / 'confusion_matrix.csv'}")
    print(f"Saved per-class metrics -> {run.log_dir / 'per_class_metrics.csv'}")

    # ===== Test set evaluation =====
    # NOTE: Commented out to prevent data leakage during development
    # Uncomment when final model evaluation is needed
    # print("\n", "="*50)
    # print("EVALUATING ON TEST SET")
    # print("="*50)
    # 
    # test_metrics = evaluate(model, test_loader, criterion, device=device, max_batches=None)
    # print(f"TEST: loss {test_metrics['loss']:.4f}, acc {test_metrics['acc']:.3f}")
    # 
    # # Confusion matrix for test set
    # cm_test = compute_confusion_matrix(model, test_loader, num_classes=int(cfg["data"]["num_classes"]), device=device)
    # save_confusion_matrix_csv(run.log_dir / "confusion_matrix_test.csv", cm_test, CLASS_NAMES)
    # 
    # per_class_test = compute_per_class_metrics(cm_test)
    # save_per_class_metrics_csv(run.log_dir / "per_class_metrics_test.csv", per_class_test, CLASS_NAMES)
    # 
    # print(f"Saved test confusion matrix -> {run.log_dir / 'confusion_matrix_test.csv'}")
    # print(f"Saved test per-class metrics -> {run.log_dir / 'per_class_metrics_test.csv'}")

    print("\n", "------ FINISHED ------", "\n", "\n")

if __name__ == "__main__":
    main()
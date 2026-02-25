"""
The orchestrator of the full pipeline.
From project root, run this file: python main.py --name <name>
"""

import argparse
from pathlib import Path
from collections import Counter

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # since only one GPU is supported
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn

from src.config import load_config
from src.data import build_dataloaders, build_datasets, build_transform, CLASS_NAMES
from src.model import BaselineCNN, vprint
from src.engine import train_one_epoch, evaluate
from src.run import make_run_info, save_config_snapshot, save_code_snapshot
from src.checkpoint import save_checkpoint
from src.metrics import init_metrics_csv, append_metrics_csv, plot_loss_acc
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
    vprint(f"[DEBUG] max_train_batches: {max_train_batches}, max_val_batches: {max_val_batches}", cfg)

    run = make_run_info(project_root, args.name)
    save_config_snapshot(cfg, run.config_snapshot_path)
    save_code_snapshot(project_root, run.code_snapshot_path)
    init_metrics_csv(run.metrics_csv_path)

    print(f"Run: {run.run_id}")
    print(f"Run dir:        {run.run_dir}")
    print(f"Config snapshot:{run.config_snapshot_path}")
    print(f"Code snapshot:{run.code_snapshot_path}")
    print(f"Metrics CSV:    {run.metrics_csv_path}")
    print(f"Best ckpt:      {run.best_ckpt_path}")

    # ========== build dataloaders ==========
    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    # debug--------
    from collections import Counter

    # dataset label sanity (works for your custom dataset)
    train_counts = Counter([y for _, y in train_loader.dataset.samples])
    val_counts   = Counter([y for _, y in val_loader.dataset.samples])

    vprint(f"[DEBUG] train labels unique: {sorted(train_counts.keys())}", cfg)
    vprint(f"[DEBUG] val labels unique: {sorted(val_counts.keys())}", cfg)
    vprint(f"[DEBUG] num_classes cfg: {cfg['data']['num_classes']}, len(CLASS_NAMES): {len(CLASS_NAMES)}", cfg)
    vprint(f"[DEBUG] CLASS_NAMES: {CLASS_NAMES}", cfg)

    from collections import Counter, defaultdict

    import random


    def infer_mapping_from_samples(samples, n=5000, seed=0):
        rng = random.Random(seed)
        idxs = rng.sample(range(len(samples)), min(n, len(samples)))
        m = defaultdict(Counter)
        for i in idxs:
            p, y = samples[i]
            cls = Path(p).parent.name  # uses the Path you already imported at file top
            m[int(y)][cls] += 1
        return {y: m[y].most_common(3) for y in sorted(m.keys())}

    vprint(f"[DEBUG] train label->folder top3: {infer_mapping_from_samples(train_loader.dataset.samples, seed=0)}", cfg)
    vprint(f"[DEBUG] val   label->folder top3: {infer_mapping_from_samples(val_loader.dataset.samples, seed=1)}", cfg)
    # show a few sample paths per label (does the path name match the label meaning?)
    from collections import defaultdict
    examples = defaultdict(list)
    for path, y in train_loader.dataset.samples[:5000]:  # scan first 5k only
        if len(examples[y]) < 2:
            examples[y].append(path)

    for y in range(cfg["data"]["num_classes"]):
        print(f"[DEBUG] label {y} -> {CLASS_NAMES[y]} examples:", examples[y])
    # debug--------
    # check dataloader
    xb, yb, _ = next(iter(train_loader))
    assert xb.ndim == 4 and (xb.shape[1] == 1 or xb.shape[1] == 3), xb.shape
    assert yb.ndim == 1, yb.shape
    print(f"[INFO] Train batch shape: {xb.shape}, labels shape: {yb.shape}")

    # ===== DEBUG: Check class distribution =====
    train_labels = []
    for _, labels, _ in train_loader:
        train_labels.extend(labels.tolist())
    train_dist = Counter(train_labels)
    print("[INFO] Train set class distribution:", dict(sorted(train_dist.items())))
    
    val_labels = []
    for _, labels, _ in val_loader:
        val_labels.extend(labels.tolist())
    val_dist = Counter(val_labels)
    print("[INFO] Val set class distribution:", dict(sorted(val_dist.items())))

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
    print(f"[INFO] Using device: {device}\n")
    vprint(f"[DEBUG] torch.cuda.is_available(): {torch.cuda.is_available()}", cfg)
    vprint(f"[DEBUG] cuda device count: {torch.cuda.device_count()}", cfg)
    if torch.cuda.is_available():
        d = torch.cuda.current_device()
        vprint(f"[DEBUG] current cuda device: {d}", cfg)
        vprint(f"[DEBUG] cuda device name: {torch.cuda.get_device_name(d)}", cfg)

    # ========== init model ==========
    model = BaselineCNN(cfg)
    model = model.to(device)
    save_model_summary(run.model_summary_path, model, cfg)
    print(f"Model summary:  {run.model_summary_path}", "\n")
    vprint(f"[DEBUG] model param device: {next(model.parameters()).device}", cfg)

    # ========== loss + LRScheduler + optimizer ==========
    lr = float(cfg["LR_scheduler"]["lr"])
    epochs = int(cfg["train"]["epochs"])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler_type = cfg["LR_scheduler"]["type"]

    if scheduler_type == "CosineAnnealingLR":
        eta_min = float(cfg["LR_scheduler"].get("eta_min", 0))
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, eta_min=eta_min, T_max=epochs)
        print(
            f"[INFO] Scheduler: {scheduler.__class__.__name__} | "
            f"initial_lr={optimizer.param_groups[0]['lr']:.3e} | "
            f"T_max={scheduler.T_max} | eta_min={scheduler.eta_min}"
        )

    else:
        raise ValueError(f"Unsupported LR scheduler type: {scheduler_type}")

    # TODO: currently wrong since new scheduler -> adapt if needed correctly
    # # ========== load checkpoint if specified ==========
    # start_epoch = 1
    # best_val_acc = -1.0
    
    # if args.checkpoint is not None:
    #     checkpoint_dir = project_root / "runs" / args.checkpoint
    #     checkpoint_path = checkpoint_dir / "checkpoints" / "best.pt"
        
    #     if not checkpoint_path.exists():
    #         raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
    #     print(f"Loading checkpoint from: {checkpoint_path}")
    #     ckpt = torch.load(checkpoint_path, map_location=device)
        
    #     model.load_state_dict(ckpt["model_state_dict"])
    #     optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        
    #     # Get metrics from checkpoint
    #     start_epoch = ckpt.get("epoch", 0) + 1
    #     best_val_acc = ckpt.get("val_metrics", {}).get("acc", -1.0)
        
    #     print(f"Resuming from epoch {start_epoch}, best val acc: {best_val_acc:.3f}\n")
        
    # Set start_epoch to 1
    start_epoch = 1
    best_val_acc = -1.0

    # early stopping config
    is_es = bool(cfg["early_stopping"]["enabled"])
    patience = int(cfg["early_stopping"]["patience"])
    min_delta = float(cfg["early_stopping"]["min_delta"])
    if is_es:
        print(f"[INFO] EarlyStopping enabled | patience={patience} | min_delta={min_delta}")
    else:
        print("[INFO] EarlyStopping disabled")

    best_val_loss = float("inf")
    bad_epochs = 0

    # ========== train loop ==========
    # debug----------
    import math

    def param_norm(m):
        s = 0.0
        for p in m.parameters():
            s += p.detach().float().norm().item() ** 2
        return math.sqrt(s)
    # debug----------

    for epoch in range(start_epoch, epochs + 1):
        # debug----------
        w0 = param_norm(model)
        # debug----------
        ## train one epoch
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device=device, max_batches=max_train_batches
        )
        ## evaluete on val set after each epoch
        val_metrics = evaluate(
            model, val_loader, criterion, device=device, max_batches=max_val_batches
        )
        # debug----------
        w1 = param_norm(model)
        print(f"[DEBUG] param_norm: {w0:.6f} -> {w1:.6f}")
        # debug----------

        lr_used = optimizer.param_groups[0]["lr"]  # LR used for this epoch

        # print metrics
        print(
            f"epoch {epoch:02d}/{epochs} | "
            f"lr_used {lr_used:.3e} | "
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

        # plot metrics
        plot_loss_acc(run.metrics_csv_path)

        ## new best model?
        # evaluate based on validation accuracy!
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
        
        ## early stopping
        # evaluate based on validation loss!
        val_loss = float(val_metrics["loss"])
        if is_es:
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                bad_epochs = 0
                print(f" -> earlystop: new best val_loss {best_val_loss:.4f}")
            else:
                bad_epochs += 1
                print(f" -> earlystop: no val_loss improvement ({bad_epochs}/{patience})")

            if bad_epochs >= patience:
                print(f"[EARLY STOP] Stop at epoch {epoch} | best val_loss {best_val_loss:.4f}")
                break
        

        ## update LR for next epoch
        scheduler.step()
        print("\n")



    # ========== validation on Best Model ==========
    # per-class metrics on best checkpoint (val set) =====
    ckpt = torch.load(run.best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    cm = compute_confusion_matrix(model, val_loader, num_classes=int(cfg["data"]["num_classes"]), device=device)
    save_confusion_matrix_csv(run.log_dir / "confusion_matrix.csv", cm, CLASS_NAMES)

    # ===== DEBUG: Print confusion matrix details =====
    print("\n[DEBUG] Confusion Matrix:")
    print(cm)
    vprint(f"[DEBUG] Predictions per class (column sums): {cm.sum(dim=0).tolist()}", cfg)
    vprint(f"[DEBUG] True instances per class (row sums): {cm.sum(dim=1).tolist()}", cfg)

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
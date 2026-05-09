import argparse
import sys
import torch
from pathlib import Path
import numpy as np
import onnxruntime as ort
import yaml
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.data import build_dataloaders, _parse_norm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, required=True, help="Path to ONNX file")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file")
    parser.add_argument("--fraction", type=float, default=0.1, help="Fraction of set to test")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test", "train"], help="Dataset split to test on")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # For fast testing, we just load data
    print("[INFO] Constructing dataloaders...")
    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    
    if args.split == "val":
        loader = val_loader
    elif args.split == "test":
        loader = test_loader
    else:
        loader = train_loader

    # Parse normalization from config to undo the dataloader's ToTensor and Normalize
    in_ch = int(cfg.get("model", {}).get("in_channels", 1))
    norm_cfg = cfg.get("preprocess", {}).get("normalize", {})
    norm_mean = _parse_norm(norm_cfg, in_ch)[0]
    norm_std = _parse_norm(norm_cfg, in_ch)[1]

    mu = torch.tensor(norm_mean).view(1, -1, 1, 1)
    sigma = torch.tensor(norm_std).view(1, -1, 1, 1)

    print(f"[INFO] Loading ONNX model from {args.onnx} ...")
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    num_batches = len(loader)
    test_batches = max(1, int(num_batches * args.fraction))
    print(f"[INFO] Testing on {args.fraction*100:.0f}% of {args.split} set ({test_batches}/{num_batches} batches).")

    correct = 0
    total = 0

    for i, (xb, yb, _) in enumerate(loader):
        if i >= test_batches:
            break
        
        # Datloader provides `(x - mu)/std` where `x` is in [0, 1].
        # We need to reverse: `x_raw_255 = (xb * std + mu) * 255.0`
        # Because our newly fused ONNX model explicitly expects [0..255] float!
        xb_raw = (xb * sigma + mu) * 255.0
        
        # Run inference one by one because model is exported with fixed batch_size=1
        xb_np = xb_raw.numpy().astype(np.float32)
        logits_list = []
        for j in range(xb_np.shape[0]):
            out = sess.run(None, {input_name: xb_np[j:j+1]})[0]
            logits_list.append(out)
        
        logits = np.concatenate(logits_list, axis=0)
        preds = np.argmax(logits, axis=1)
        labels = yb.numpy()
        
        correct += (preds == labels).sum()
        total += len(labels)
        
        if (i+1) % 10 == 0:
            print(f"  Batch {i+1}/{test_batches} - Acc: {correct/total:.4f}")

    final_acc = correct / total
    print(f"\n[RESULT] Final ONNX Accuracy on {args.fraction*100:.0f}% {args.split.capitalize()}: {final_acc:.4f} ({correct}/{total})")

if __name__ == "__main__":
    main()

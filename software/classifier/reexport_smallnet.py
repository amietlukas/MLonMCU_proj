"""
Re-export the smallnet_greyscale baseline checkpoint with the current
checkpoint.py logic, which fuses the (pixel/255 - mu)/sigma normalization into
the first Conv. The resulting ONNX expects raw [0, 255] float pixels, matching
the bignet / bignet_pruned export convention so the MCU can do a single
`u - 128` shift instead of float normalization.

Run from /mnt/core/MLonMCU_proj/software:
    python classifier/reexport_smallnet.py
"""

from __future__ import annotations

import sys
from pathlib import Path

SOFTWARE_ROOT = Path(__file__).resolve().parent.parent
if str(SOFTWARE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOFTWARE_ROOT))

import torch

from utils.config import load_config
from utils.checkpoint import export_best_model_fp32_and_int8_qdq
from classifier.src.data import build_dataloaders
from classifier.src.model import BaselineCNN


RUN_DIR = SOFTWARE_ROOT / "classifier" / "runs" / "smallnet_greyscale-20260504-184641_BASELINE_SMALL"
RUN_ID  = "smallnet_greyscale-20260504-184641"


def main():
    ckpt_path   = RUN_DIR / "checkpoints" / "best.pt"
    config_path = RUN_DIR / "config_snapshot.yaml"
    onnx_out    = RUN_DIR / "checkpoints" / "onnx"
    onnx_out.mkdir(parents=True, exist_ok=True)

    cfg = load_config(config_path, RUN_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")
    print(f"[INFO] checkpoint: {ckpt_path}")
    print(f"[INFO] config:     {config_path}")

    model = BaselineCNN(cfg).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    best_epoch = int(ckpt.get("epoch", -1))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[INFO] loaded best epoch = {best_epoch}")

    _train_loader, val_loader, _test_loader = build_dataloaders(cfg)

    export_best_model_fp32_and_int8_qdq(
        model=model,
        cfg=cfg,
        device=device,
        run_id=RUN_ID,
        out_dir=onnx_out,
        best_epoch=best_epoch,
        calibration_loader=val_loader,
        calibration_batches=32,
    )
    print("[INFO] done.")


if __name__ == "__main__":
    main()

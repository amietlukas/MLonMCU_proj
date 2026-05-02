from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch.nn as nn
from torchinfo import summary


def save_model_summary(path: str | Path, model: nn.Module, cfg: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    size_cfg = cfg.get("data", {}).get("input_size", 128)
    in_ch = int(cfg["model"].get("in_channels", 1))
    
    if isinstance(size_cfg, (list, tuple)):
        h, w = size_cfg
    else:
        h = w = int(size_cfg)

    # This produces the "Layer / Output Shape / Param #" table
    s = summary(
        model,
        input_size=(1, in_ch, h, w),  # batch=1
        col_names=("output_size", "num_params"),
        depth=10,
        verbose=1,
    )

    path.write_text(str(s) + "\n", encoding="utf-8")
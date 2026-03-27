from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch.nn as nn
from torchinfo import summary


def save_model_summary(path: str | Path, model: nn.Module, cfg: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    input_size = int(cfg["data"]["input_size"])
    in_ch = int(cfg["model"].get("in_channels", 1))

    # This produces the "Layer / Output Shape / Param #" table
    s = summary(
        model,
        input_size=(1, in_ch, input_size, input_size),  # batch=1
        col_names=("output_size", "num_params"),
        depth=10,
        verbose=1,
    )

    path.write_text(str(s) + "\n", encoding="utf-8")
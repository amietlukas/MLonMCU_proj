from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from ball_detection.src.models.styolo.backbone import STYOLONanoBackbone
from ball_detection.src.models.styolo.head import STYOLOHead
from ball_detection.src.models.styolo.neck import STYOLONeck


class BallSTYOLONano(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        in_ch = int(cfg["input"]["channels"])
        width_mult = float(cfg["model"].get("width_mult", 1.0))
        neck_out_ch = int(cfg["model"].get("neck_out_ch", 96))
        if neck_out_ch <= 0:
            raise ValueError("model.neck_out_ch must be > 0")

        self.strides = tuple(int(s) for s in cfg["model"].get("strides", [8, 16, 32]))
        if self.strides != (8, 16, 32):
            raise ValueError("Current BallSTYOLONano implementation expects strides [8,16,32]")

        self.backbone = STYOLONanoBackbone(in_channels=in_ch, width_mult=width_mult)
        self.neck = STYOLONeck(in_channels=self.backbone.out_channels, out_ch=neck_out_ch)
        self.head = STYOLOHead(feat_ch=neck_out_ch)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)
        fused = self.neck(feats)
        out = self.head(fused)
        return out

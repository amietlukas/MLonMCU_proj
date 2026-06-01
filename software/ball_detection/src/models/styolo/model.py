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
        
        valid_strides = (8, 16, 32)
        if not set(self.strides).issubset(valid_strides):
            raise ValueError(f"strides {self.strides} must be a subset of {valid_strides}")

        self.stride_indices = [valid_strides.index(s) for s in self.strides]

        self.backbone = STYOLONanoBackbone(in_channels=in_ch, width_mult=width_mult)
        
        neck_in_channels = tuple(self.backbone.out_channels[i] for i in self.stride_indices)
        self.neck = STYOLONeck(in_channels=neck_in_channels, out_ch=neck_out_ch)
        self.head = STYOLOHead(feat_ch=neck_out_ch, num_heads=len(self.strides))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        feats = self.backbone(x)
        # Select only the features corresponding to requested strides
        feats = tuple(feats[i] for i in self.stride_indices)
        
        fused = self.neck(feats)
        out = self.head(fused)
        return out

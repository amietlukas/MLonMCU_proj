from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from ball_detection.src.models.styolo.blocks import ConvBNReLU


class STYOLOHead(nn.Module):
    """Single-class objectness head: [tx, ty, tw, th, objectness]."""

    def __init__(self, feat_ch: int = 96, num_heads: int = 3):
        super().__init__()

        self.stems = nn.ModuleList([
            nn.Sequential(ConvBNReLU(feat_ch, feat_ch), ConvBNReLU(feat_ch, feat_ch))
            for _ in range(num_heads)
        ])
        
        self.preds = nn.ModuleList([
            nn.Conv2d(feat_ch, 5, kernel_size=1, stride=1, padding=0)
            for _ in range(num_heads)
        ])

    def forward(
        self,
        feats: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, ...]:
        return tuple(pred(stem(f)) for f, stem, pred in zip(feats, self.stems, self.preds))

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from ball_detection.src.models.styolo.blocks import ConvBNReLU


class STYOLOHead(nn.Module):
    """Single-class objectness head: [tx, ty, tw, th, objectness]."""

    def __init__(self, feat_ch: int = 96):
        super().__init__()

        self.stem8 = nn.Sequential(ConvBNReLU(feat_ch, feat_ch), ConvBNReLU(feat_ch, feat_ch))
        self.stem16 = nn.Sequential(ConvBNReLU(feat_ch, feat_ch), ConvBNReLU(feat_ch, feat_ch))
        self.stem32 = nn.Sequential(ConvBNReLU(feat_ch, feat_ch), ConvBNReLU(feat_ch, feat_ch))

        self.pred8 = nn.Conv2d(feat_ch, 5, kernel_size=1, stride=1, padding=0)
        self.pred16 = nn.Conv2d(feat_ch, 5, kernel_size=1, stride=1, padding=0)
        self.pred32 = nn.Conv2d(feat_ch, 5, kernel_size=1, stride=1, padding=0)

    def forward(
        self,
        feats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p8, p16, p32 = feats
        o8 = self.pred8(self.stem8(p8))
        o16 = self.pred16(self.stem16(p16))
        o32 = self.pred32(self.stem32(p32))
        return o8, o16, o32

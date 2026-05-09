from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from ball_detection.src.models.styolo.blocks import ConvBNReLU


class STYOLONeck(nn.Module):
    """Simple FPN/PAN-lite neck using nearest upsample and concat."""

    def __init__(self, in_channels: tuple[int, int, int], out_ch: int = 96):
        super().__init__()
        c3, c4, c5 = in_channels

        self.lat_c5 = ConvBNReLU(c5, out_ch, k=1, s=1, p=0)
        self.lat_c4 = ConvBNReLU(c4, out_ch, k=1, s=1, p=0)
        self.lat_c3 = ConvBNReLU(c3, out_ch, k=1, s=1, p=0)

        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        self.fuse_p4 = ConvBNReLU(out_ch * 2, out_ch, k=3, s=1)
        self.fuse_p3 = ConvBNReLU(out_ch * 2, out_ch, k=3, s=1)

        self.down_p3 = ConvBNReLU(out_ch, out_ch, k=3, s=2)
        self.fuse_n4 = ConvBNReLU(out_ch * 2, out_ch, k=3, s=1)

        self.down_n4 = ConvBNReLU(out_ch, out_ch, k=3, s=2)
        self.fuse_n5 = ConvBNReLU(out_ch * 2, out_ch, k=3, s=1)

    def forward(
        self,
        feats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c3, c4, c5 = feats

        p5 = self.lat_c5(c5)
        p4 = self.fuse_p4(torch.cat([self.up(p5), self.lat_c4(c4)], dim=1))
        p3 = self.fuse_p3(torch.cat([self.up(p4), self.lat_c3(c3)], dim=1))

        n4 = self.fuse_n4(torch.cat([self.down_p3(p3), p4], dim=1))
        n5 = self.fuse_n5(torch.cat([self.down_n4(n4), p5], dim=1))

        return p3, n4, n5

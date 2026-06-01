from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from ball_detection.src.models.styolo.blocks import ConvBNReLU


class STYOLONeck(nn.Module):
    """Simple FPN/PAN-lite neck using nearest upsample and concat."""

    def __init__(self, in_channels: tuple[int, ...], out_ch: int = 96):
        super().__init__()
        self.num_levels = len(in_channels)
        if self.num_levels < 2:
            raise ValueError("STYOLONeck requires at least 2 input feature maps")

        # 1x1 laterals for each input feature map
        self.lat = nn.ModuleList([
            ConvBNReLU(c, out_ch, k=1, s=1, p=0) for c in in_channels
        ])

        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        # Top-down FPN fusions
        self.fuse_up = nn.ModuleList([
            ConvBNReLU(out_ch * 2, out_ch, k=3, s=1) 
            for _ in range(self.num_levels - 1)
        ])

        # Bottom-up PAN branches
        self.down = nn.ModuleList([
            ConvBNReLU(out_ch, out_ch, k=3, s=2) 
            for _ in range(self.num_levels - 1)
        ])
        self.fuse_down = nn.ModuleList([
            ConvBNReLU(out_ch * 2, out_ch, k=3, s=1) 
            for _ in range(self.num_levels - 1)
        ])

    def forward(
        self,
        feats: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, ...]:
        
        # 1. Project inputs to target channels
        lat_feats = [lat(f) for lat, f in zip(self.lat, feats)]

        # 2. Top-down FPN (start from deepest/lowest-res feature)
        fpn_feats = [lat_feats[-1]]
        for i in range(self.num_levels - 2, -1, -1):
            up_feat = self.up(fpn_feats[-1])
            fused = self.fuse_up[i](torch.cat([up_feat, lat_feats[i]], dim=1))
            fpn_feats.append(fused)
        
        # Reverse to get order: [highest-res, ..., lowest-res]
        fpn_feats = fpn_feats[::-1]

        # 3. Bottom-up PAN
        pan_feats = [fpn_feats[0]]
        for i in range(self.num_levels - 1):
            down_feat = self.down[i](pan_feats[-1])
            fused = self.fuse_down[i](torch.cat([down_feat, fpn_feats[i + 1]], dim=1))
            pan_feats.append(fused)

        return tuple(pan_feats)

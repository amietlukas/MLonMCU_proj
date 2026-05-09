from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from ball_detection.src.models.styolo.blocks import ConvBNReLU, StageBlock


class STYOLONanoBackbone(nn.Module):
    """Compact Conv-BN-ReLU backbone with residual bottlenecks.

    Outputs feature maps at strides 8, 16, 32.
    """

    def __init__(self, in_channels: int = 3, width_mult: float = 1.0):
        super().__init__()

        def c(x: int) -> int:
            return max(8, int(round(x * width_mult / 8.0)) * 8)

        c2, c4, c8, c16, c32 = c(32), c(48), c(64), c(128), c(192)

        self.stem = ConvBNReLU(in_channels, c2, k=3, s=2)  # stride 2
        self.stage2 = StageBlock(c2, c4, num_blocks=1)  # stride 4
        self.stage3 = StageBlock(c4, c8, num_blocks=2)  # stride 8
        self.stage4 = StageBlock(c8, c16, num_blocks=2)  # stride 16
        self.stage5 = StageBlock(c16, c32, num_blocks=2)  # stride 32

        self.out_channels = (c8, c16, c32)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.stage2(x)
        c3 = self.stage3(x)  # stride 8
        c4 = self.stage4(c3)  # stride 16
        c5 = self.stage5(c4)  # stride 32
        return c3, c4, c5

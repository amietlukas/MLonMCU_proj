from __future__ import annotations

import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int | None = None):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ResidualBottleneck(nn.Module):
    def __init__(self, ch: int, hidden_ratio: float = 0.5):
        super().__init__()
        hidden = max(8, int(ch * hidden_ratio))
        self.conv1 = ConvBNReLU(ch, hidden, k=1, s=1, p=0)
        self.conv2 = ConvBNReLU(hidden, hidden, k=3, s=1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden, ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv3(self.conv2(self.conv1(x)))
        return self.act(x + y)


class StageBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_blocks: int):
        super().__init__()
        layers: list[nn.Module] = [ConvBNReLU(in_ch, out_ch, k=3, s=2)]
        for _ in range(num_blocks):
            layers.append(ResidualBottleneck(out_ch))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

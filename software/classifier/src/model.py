"""
Here we build a simple CNN baseline model for image (hand gesture) classification. 
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List


# build one stage / block
def make_stage(in_ch: int, out_ch: int, k: int, blocks: int, dropout_rate: float = 0.0) -> nn.Sequential:
    layers = []
    pad = k // 2  # keeps spatial size for stride=1

    # first conv (bias=False because BatchNorm has its own bias)
    layers += [nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=pad, bias=False),
               nn.BatchNorm2d(out_ch),
               nn.ReLU(inplace=True)]
    if dropout_rate > 0:
        layers += [nn.Dropout2d(dropout_rate)]
    
    # optional extra convs in the same stage (if model.blocks_per_stage > 1)
    for _ in range(blocks - 1):
        layers += [nn.Conv2d(out_ch, out_ch, kernel_size=k, padding=pad, bias=False),
                   nn.BatchNorm2d(out_ch),
                   nn.ReLU(inplace=True)]
        if dropout_rate > 0:
            layers += [nn.Dropout2d(dropout_rate)]

    # downsample
    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*layers)



# =========== Baseline CNN Model ===========
class BaselineCNN(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()

        m = cfg["model"]

        self.type = m.get("type", "baseline_cnn")
        self.in_channels = int(m.get("in_channels", 1))
        self.num_classes = int(m.get("num_classes", cfg["data"]["num_classes"]))
        self.channels: List[int] = list(m["channels"])
        self.blocks_per_stage = int(m.get("blocks_per_stage", 1))
        self.kernel_size = int(m.get("kernel_size", 3))
        self.dropout_rate = float(m.get("dropout_rate", 0.0))
        
        # checks
        if self.type != "baseline_cnn":
            raise ValueError(f"BaselineCNN called but model.type is {m.get('type')}")
        if self.blocks_per_stage < 1:
            raise ValueError("model.blocks_per_stage must be >= 1")
        if len(self.channels) == 0:
            raise ValueError("model.channels must be a non-empty list")
        if self.kernel_size % 2 == 0:
            raise ValueError("model.kernel_size should be odd (e.g., 3, 5)")
        
        stages = []
        in_ch = self.in_channels
        for out_ch in self.channels:
            stages.append(make_stage(in_ch, out_ch, self.kernel_size, self.blocks_per_stage, self.dropout_rate))
            in_ch = out_ch
        
        self.features = nn.Sequential(*stages)
        self.classifier = nn.Linear(in_ch, self.num_classes)
        if self.dropout_rate > 0:
            self.dropout_fc = nn.Dropout(self.dropout_rate)
        else:
            self.dropout_fc = None
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)                  # in: [B, C, H, W] out: [B, C', H', W']
        x = x.mean(dim=(2, 3))                # global average pooling -> [B, C']
        if self.dropout_fc is not None:
            x = self.dropout_fc(x)            # apply dropout before classifier
        logits = self.classifier(x)           # [B, num_classes]
        return logits

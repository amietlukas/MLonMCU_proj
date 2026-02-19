"""
Here we build a simple CNN baseline model for image (hand gesture) classification. 
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List

# build one stage / block
def make_stage(in_ch: int, out_ch: int, k: int, blocks: int) -> nn.Sequential:
    layers = []
    pad = k // 2  # keeps spatial size for stride=1

    # first conv
    layers += [nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=pad, bias=True),
               nn.ReLU(inplace=True)]
    
    # optional extra convs in the same stage (if model.blocks_per_stage > 1)
    for _ in range(blocks - 1):
        layers += [nn.Conv2d(out_ch, out_ch, kernel_size=k, padding=pad, bias=True),
                   nn.ReLU(inplace=True)]

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
            stages.append(make_stage(in_ch, out_ch, self.kernel_size, self.blocks_per_stage))
            in_ch = out_ch
        
        self.features = nn.Sequential(*stages)
        self.classifier = nn.Linear(in_ch, self.num_classes)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)                  # in: [B, C, H, W] out: [B, C', H', W']
        x = x.mean(dim=(2, 3))                # global average pooling -> [B, C']
        logits = self.classifier(x)           # [B, num_classes]
        return logits

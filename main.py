import torch
from pathlib import Path
from src.config import load_config
from src.data import build_dataloaders
from src.model import BaselineCNN


def main():
    project_root = Path(__file__).resolve().parent # IMPORTANT: main.py needs to be in the project root
    cfg = load_config(project_root / "config.yaml", project_root) # load config and check global assumptions

    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    model = BaselineCNN(cfg)
    model.eval()

    xb, yb, pb = next(iter(train_loader))

    with torch.no_grad():
        logits = model(xb)

    print("x:", xb.shape)
    print("logits:", logits.shape, logits.dtype)
    print("y:", yb.shape, yb.dtype, "min/max:", yb.min().item(), yb.max().item())


if __name__ == "__main__":
    main()
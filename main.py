from pathlib import Path

from src.config import load_config
from src.data import build_dataloaders


def main():
    project_root = Path(__file__).resolve().parent # IMPORTANT: main.py needs to be in the project root
    cfg = load_config(project_root / "config.yaml", project_root) # load config and check global assumptions

    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    xb, yb, pb = next(iter(train_loader))

    print("train batch:")
    print("  x:", xb.shape, xb.dtype)
    print("  y:", yb.shape, yb.dtype, "min/max:", yb.min().item(), yb.max().item())
    print("  paths:", type(pb), len(pb))
    print("  example path:", pb[0])


if __name__ == "__main__":
    main()
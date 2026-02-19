# config.py
# checks global assumptions

from pathlib import Path
from typing import Any, Dict
import yaml



def load_config(config_path: str | Path,
                project_root: str | Path) -> Dict[str, Any]:
    
    # read config.yaml
    config_path =  Path(config_path) # make sure Path obj. to be platform independent
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML to dict.")
    
    # make paths absolute
    dataset_root = Path(cfg["data"]["dataset_root"])
    if not dataset_root.is_absolute():
        dataset_root = (project_root / dataset_root).resolve()
    cfg["data"]["dataset_root"] = dataset_root

    # structure checks
    for d in ["train", "val", "test"]:
        p = dataset_root / d
        if not p.exists():
            raise FileNotFoundError(f"Missing dataset folder: {p}")
    ann = dataset_root / "annotations"
    if not ann.exists():
        print(f"[WARN] annotations folder not found at: {ann}")

    return cfg





    
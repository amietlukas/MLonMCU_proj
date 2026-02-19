from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
import yaml



@dataclass(frozen=True)
class RunInfo:
    name: str           # user-provided name
    timestamp: str      # YYYYMMDD-HHMMSS
    run_id: str         # name-timestamp
    run_dir: Path
    ckpt_dir: Path
    log_dir: Path
    best_ckpt_path: Path
    config_snapshot_path: Path
    metrics_csv_path: Path



def make_run_info(project_root: Path, name: str) -> RunInfo:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{name}-{timestamp}"

    run_dir = project_root / "runs" / run_id
    ckpt_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt_path = ckpt_dir / "best.pt"
    config_snapshot_path = run_dir / "config_snapshot.yaml"
    metrics_csv_path = log_dir / "metrics.csv"

    return RunInfo(
        name=name,
        timestamp=timestamp,
        run_id=run_id,
        run_dir=run_dir,
        ckpt_dir=ckpt_dir,
        log_dir=log_dir,
        best_ckpt_path=best_ckpt_path,
        config_snapshot_path=config_snapshot_path,
        metrics_csv_path=metrics_csv_path,
    )



def save_config_snapshot(cfg: Dict[str, Any], path: Path) -> None:
    # cfg contains Path objects (dataset_root). YAML can't serialize Path nicely.
    def _to_serializable(x):
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, dict):
            return {k: _to_serializable(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_to_serializable(v) for v in x]
        return x

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(_to_serializable(cfg), f, sort_keys=False)
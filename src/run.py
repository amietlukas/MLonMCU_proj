from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class RunInfo:
    name: str          # user-provided name
    timestamp: str     # YYYYMMDD-HHMMSS
    run_id: str        # name-timestamp
    checkpoint_path: Path


def make_run_info(project_root: Path, name: str) -> RunInfo:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{name}-{timestamp}"
    ckpt_dir = project_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"best_checkpoint_{run_id}.pt"
    return RunInfo(name=name, timestamp=timestamp, run_id=run_id, checkpoint_path=ckpt_path)
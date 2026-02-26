from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
import yaml
import zipfile
import sys


@dataclass(frozen=True)
class RunInfo:
    name: str           # user-provided name
    timestamp: str      # YYYYMMDD-HHMMSS
    run_id: str         # name-timestamp
    run_dir: Path
    model_summary_path: Path # model architecture
    ckpt_dir: Path
    log_dir: Path
    best_ckpt_path: Path
    config_snapshot_path: Path
    code_snapshot_path: Path 
    metrics_csv_path: Path
    terminal_log_path: Path



def make_run_info(project_root: Path, name: str) -> RunInfo:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{name}-{timestamp}"

    run_dir = project_root / "runs" / run_id
    ckpt_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    model_summary_path = run_dir / "model_summary.txt"

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt_path = ckpt_dir / "best.pt"
    config_snapshot_path = run_dir / "config_snapshot.yaml"
    code_snapshot_path = run_dir / "code_snapshot.zip"
    metrics_csv_path = log_dir / "metrics.csv"
    terminal_log_path = log_dir / "terminal.log"

    return RunInfo(
        name=name,
        timestamp=timestamp,
        run_id=run_id,
        run_dir=run_dir,
        model_summary_path=model_summary_path,
        ckpt_dir=ckpt_dir,
        log_dir=log_dir,
        best_ckpt_path=best_ckpt_path,
        config_snapshot_path=config_snapshot_path,
        code_snapshot_path=code_snapshot_path,
        metrics_csv_path=metrics_csv_path,
        terminal_log_path=terminal_log_path,
    )


# ===== save the config =====
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


# ===== save the code snapshot =====
def save_code_snapshot(project_root: Path, out_zip_path: Path) -> None:
    """
    Zip only:
      - main.py
      - src/ (entire folder)
    """
    project_root = Path(project_root).resolve()
    out_zip_path = Path(out_zip_path)
    out_zip_path.parent.mkdir(parents=True, exist_ok=True)

    main_py = project_root / "main.py"
    src_dir = project_root / "src"

    if not main_py.exists():
        raise FileNotFoundError(f"Not found: {main_py}")
    if not src_dir.exists():
        raise FileNotFoundError(f"Not found: {src_dir}")

    with zipfile.ZipFile(out_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # add main.py
        zf.write(main_py, arcname="main.py")

        # add src/ recursively
        for fp in src_dir.rglob("*"):
            if fp.is_dir():
                continue
            arcname = fp.relative_to(project_root).as_posix()  # keeps "src/..."
            zf.write(fp, arcname=arcname)



# ===== save the terminal log =====
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

def setup_terminal_logging(log_path: Path):
    """
    Redirect stdout+stderr to log_path while keeping terminal output visible.
    Returns the opened file handle (keep it alive, close at end).
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "a", buffering=1, encoding="utf-8")  # line-buffered

    sys.stdout = Tee(sys.__stdout__, f)
    sys.stderr = Tee(sys.__stderr__, f)

    return f
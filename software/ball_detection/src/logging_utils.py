from __future__ import annotations

import logging
from pathlib import Path


def setup_logger(log_file: Path, debug: bool = False) -> logging.Logger:
    logger = logging.getLogger("ball_detection")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(message)s")

    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG if debug else logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.propagate = False
    return logger

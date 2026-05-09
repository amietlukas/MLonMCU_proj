from __future__ import annotations

from typing import Any, Dict

from ball_detection.src.models.styolo.model import BallSTYOLONano


MODEL_REGISTRY = {
    "ball_styolo_nano": BallSTYOLONano,
}


def build_model(cfg: Dict[str, Any]):
    model_name = str(cfg["model"]["name"]).lower()
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{model_name}'. Available: {sorted(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name](cfg)

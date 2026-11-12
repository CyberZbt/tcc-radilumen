"""Utilitários compartilhados do pipeline TCC."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

CODIGO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = CODIGO_DIR / "configs" / "default.yaml"
WEIGHTS_DIR = CODIGO_DIR / "weights"


def load_yaml(path: Path | str) -> dict[str, Any]:
    p = Path(path).resolve()
    with p.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_device(device: str | int | None = "auto") -> str | int:
    if device is None or str(device).lower() == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    if str(device).lower() == "cpu":
        return "cpu"
    return device


def resolve_model_path(model: str) -> str:
    """Prioriza weights/ local; Ultralytics baixa se não existir."""
    local = WEIGHTS_DIR / model
    if local.is_file():
        return str(local)
    codigo_local = CODIGO_DIR / model
    if codigo_local.is_file():
        return str(codigo_local)
    return model


def resolve_data_path(data: str) -> str:
    p = Path(data)
    if p.is_file():
        return str(p.resolve())
    candidate = (CODIGO_DIR / data).resolve()
    if candidate.is_file():
        return str(candidate)
    return data

"""
Treino YOLO (Ultralytics) — GRAZPEDWRI-DX.

Exemplo:
  python -m src.train
  python -m src.train --config configs/default.yaml --epochs 1 --batch 2 --device cpu
  python -m src.train --data dataset/yolo_8bit/data.yaml --name smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

from src.utils import (
    CODIGO_DIR,
    DEFAULT_CONFIG,
    load_yaml,
    resolve_data_path,
    resolve_device,
    resolve_model_path,
)


def build_train_kwargs(cfg: dict, overrides: dict) -> dict:
    keys = [
        "epochs",
        "imgsz",
        "batch",
        "patience",
        "workers",
        "seed",
        "lr0",
        "lrf",
        "momentum",
        "weight_decay",
        "optimizer",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "degrees",
        "translate",
        "scale",
        "shear",
        "flipud",
        "fliplr",
        "mosaic",
        "mixup",
    ]
    train_kwargs = {k: overrides.get(k, cfg.get(k)) for k in keys if overrides.get(k, cfg.get(k)) is not None}
    return train_kwargs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Treinar YOLO no GRAZPEDWRI-DX")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML de hiperparâmetros")
    parser.add_argument("--model", help="Peso inicial (ex.: yolov8n.pt)")
    parser.add_argument("--data", help="data.yaml do dataset")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--device", help="auto | cpu | 0")
    parser.add_argument("--name", default="grazpedwri", help="Nome da run em runs/detect/")
    parser.add_argument("--project", default=str(CODIGO_DIR / "runs" / "detect"))
    args = parser.parse_args(argv)

    cfg = load_yaml(args.config)
    overrides = {k: v for k, v in vars(args).items() if v is not None and k not in ("config", "name", "project")}

    model_name = resolve_model_path(overrides.get("model", cfg.get("model", "yolov8n.pt")))
    data_yaml = resolve_data_path(overrides.get("data", cfg.get("data", "dataset/yolo/data.yaml")))
    device = resolve_device(overrides.get("device", cfg.get("device", "auto")))

    train_kwargs = build_train_kwargs(cfg, overrides)
    train_kwargs.update(
        {
            "data": data_yaml,
            "device": device,
            "project": args.project,
            "name": args.name,
            "exist_ok": True,
        }
    )

    print(f"Modelo: {model_name}")
    print(f"Data: {data_yaml}")
    print(f"Device: {device}")
    print(f"Train kwargs: {train_kwargs}")

    model = YOLO(model_name)
    model.train(**train_kwargs)
    return 0


if __name__ == "__main__":
    sys.exit(main())

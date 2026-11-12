"""Funções reutilizáveis de treino e avaliação."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO

from src.train import build_train_kwargs
from src.utils import CODIGO_DIR, load_yaml, resolve_data_path, resolve_device, resolve_model_path


def patch_data_split(data_yaml: str, split: str) -> Path:
    src = Path(data_yaml).resolve()
    raw = yaml.safe_load(src.read_text(encoding="utf-8"))
    if split == "test" and "test" in raw:
        raw["val"] = raw["test"]
    tmp = CODIGO_DIR / "runs" / "_eval_data.yaml"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return tmp


def train_model(
    config_path: str | Path,
    *,
    name: str | None = None,
    epochs: int | None = None,
    batch: int | None = None,
    device: str | None = None,
    project: str | Path | None = None,
) -> Path:
    """Treina YOLO e retorna caminho de best.pt."""
    cfg = load_yaml(config_path)
    overrides: dict[str, Any] = {}
    if epochs is not None:
        overrides["epochs"] = epochs
    if batch is not None:
        overrides["batch"] = batch
    if device is not None:
        overrides["device"] = device

    run_name = name or cfg.get("name", "experiment")
    model_name = resolve_model_path(cfg.get("model", "yolov8n.pt"))
    data_yaml = resolve_data_path(cfg.get("data", "dataset/yolo/data.yaml"))
    device_resolved = resolve_device(overrides.get("device", cfg.get("device", "auto")))

    train_kwargs = build_train_kwargs(cfg, overrides)
    train_kwargs.update(
        {
            "data": data_yaml,
            "device": device_resolved,
            "project": str(project or CODIGO_DIR / "runs" / "detect"),
            "name": run_name,
            "exist_ok": True,
            "pretrained": cfg.get("pretrained", True),
        }
    )

    print(f"[train] {run_name} | model={model_name} | data={data_yaml}")
    model = YOLO(model_name)
    model.train(**train_kwargs)

    best = Path(train_kwargs["project"]) / run_name / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"Peso não encontrado após treino: {best}")
    return best


def evaluate_model(
    weights: str | Path,
    config_path: str | Path,
    *,
    split: str = "test",
    device: str | None = None,
) -> dict[str, Any]:
    """Avalia pesos e retorna métricas principais."""
    cfg = load_yaml(config_path)
    data_yaml = resolve_data_path(cfg.get("data", "dataset/yolo/data.yaml"))
    device_resolved = resolve_device(device or cfg.get("device", "auto"))
    conf = cfg.get("conf", 0.25)
    iou = cfg.get("iou", 0.5)
    imgsz = cfg.get("imgsz", 640)
    experiment = cfg.get("experiment", cfg.get("name", "unknown"))

    eval_data = patch_data_split(data_yaml, split)
    model = YOLO(str(resolve_model_path(str(weights))))

    metrics = model.val(
        data=str(eval_data),
        split="val",
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device_resolved,
        plots=True,
        project=str(CODIGO_DIR / "runs" / "val"),
        name=f"eval_{experiment}_{split}",
        exist_ok=True,
    )

    result = {
        "experiment": experiment,
        "description": cfg.get("description", ""),
        "weights": str(Path(weights).resolve()),
        "data": data_yaml,
        "split": split,
        "conf": conf,
        "iou": iou,
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return result


def save_metrics(result: dict[str, Any], out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return out

"""
Avaliação no conjunto de teste (mAP, IoU via Ultralytics).

Exemplo:
  python -m src.evaluate --weights runs/detect/grazpedwri/weights/best.pt
  python -m src.evaluate --weights best.pt --data dataset/yolo_8bit/data.yaml --split test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

from src.utils import CODIGO_DIR, DEFAULT_CONFIG, load_yaml, resolve_data_path, resolve_device, resolve_model_path


def patch_data_split(data_yaml: str, split: str) -> Path:
    """Cria YAML temporário apontando val ao split desejado (test/val)."""
    src = Path(data_yaml).resolve()
    raw = yaml.safe_load(src.read_text(encoding="utf-8"))
    if split == "test" and "test" in raw:
        raw["val"] = raw["test"]
    tmp = CODIGO_DIR / "runs" / "_eval_data.yaml"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return tmp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Avaliar YOLO no GRAZPEDWRI-DX")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", help="data.yaml")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args(argv)

    cfg = load_yaml(args.config)
    data_yaml = resolve_data_path(args.data or cfg.get("data", "dataset/yolo/data.yaml"))
    device = resolve_device(args.device)
    conf = args.conf if args.conf is not None else cfg.get("conf", 0.25)
    iou = args.iou if args.iou is not None else cfg.get("iou", 0.5)
    weights = resolve_model_path(args.weights)

    eval_data = patch_data_split(data_yaml, args.split)
    model = YOLO(weights)

    metrics = model.val(
        data=str(eval_data),
        split="val",
        conf=conf,
        iou=iou,
        imgsz=args.imgsz,
        device=device,
        plots=True,
        project=str(CODIGO_DIR / "runs" / "val"),
        name=f"eval_{args.split}",
        exist_ok=True,
    )

    print(f"Split: {args.split}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Inferência YOLO com pré-processamento e visualização.

Exemplo:
  python -m src.predict --weights runs/detect/grazpedwri/weights/best.pt --source archive/images_part1
  python -m src.predict --weights yolov8n.pt --source path/to/xray.png --show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

from src.preprocess import preprocess_image
from src.utils import CODIGO_DIR, DEFAULT_CONFIG, load_yaml, resolve_device, resolve_model_path


def collect_sources(source: str) -> list[Path]:
    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(f"Fonte não encontrada: {source}")
    if p.is_file():
        return [p]
    files: list[Path] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.dcm"):
        files.extend(sorted(p.rglob(ext)))
    if not files:
        raise FileNotFoundError(f"Nenhuma imagem em: {source}")
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inferência YOLO em radiografias")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--weights", required=True, help="Pesos treinados ou yolov8n.pt")
    parser.add_argument("--source", required=True, help="Imagem, pasta ou glob")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Aplicar 16-bit→8-bit + CLAHE antes da inferência",
    )
    parser.add_argument(
        "--out",
        default=str(CODIGO_DIR / "runs" / "predict"),
        help="Pasta de saída",
    )
    parser.add_argument("--name", default="exp", help="Subpasta em --out")
    parser.add_argument("--show", action="store_true", help="Abrir visualização (se suportado)")
    args = parser.parse_args(argv)

    cfg = load_yaml(args.config)
    device = resolve_device(args.device)
    conf = args.conf if args.conf is not None else cfg.get("conf", 0.25)
    iou = args.iou if args.iou is not None else cfg.get("iou", 0.5)
    weights = resolve_model_path(args.weights)

    sources = collect_sources(args.source)
    out_dir = Path(args.out) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(weights)

    for src in sources:
        if args.preprocess:
            img = preprocess_image(src)
            predict_source = img
        else:
            predict_source = str(src)

        results = model.predict(
            source=predict_source,
            conf=conf,
            iou=iou,
            imgsz=args.imgsz,
            device=device,
            verbose=False,
        )
        if not results:
            continue
        annotated = results[0].plot()
        out_path = out_dir / f"{src.stem}_pred.png"
        cv2.imwrite(str(out_path), annotated)
        if args.show:
            results[0].show()

    print(f"Predições salvas em: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

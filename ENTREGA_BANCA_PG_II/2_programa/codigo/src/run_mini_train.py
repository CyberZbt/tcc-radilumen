"""
Treino rápido em subset (~1600) + avaliação + log na tabela de comparação.

Exemplo:
  python -m src.run_mini_train --profile proposed --epochs 30 --device 0
  python -m src.run_mini_train --profile baseline --epochs 30 --device 0
  python -m src.run_mini_train --profile proposed --epochs 50 --name mini_prop_e50 --device 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ultralytics import YOLO

from src.log_metrics import append_row, rebuild_tex
from src.make_subset import main as make_subset_main
from src.utils import CODIGO_DIR, resolve_device

PROFILES = {
    "baseline": {
        "model": "yolov8n.yaml",
        "pretrained": False,
        "augmentation": False,
        "train_kwargs": {
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "degrees": 0.0,
            "translate": 0.0,
            "scale": 0.0,
            "shear": 0.0,
            "flipud": 0.0,
            "fliplr": 0.0,
            "mosaic": 0.0,
            "mixup": 0.0,
        },
    },
    "proposed": {
        "model": "yolov8n.pt",
        "pretrained": True,
        "augmentation": True,
        "train_kwargs": {
            "hsv_h": 0.015,
            "hsv_s": 0.4,
            "hsv_v": 0.4,
            "degrees": 10.0,
            "translate": 0.1,
            "scale": 0.3,
            "shear": 0.0,
            "flipud": 0.0,
            "fliplr": 0.5,
            "mosaic": 1.0,
            "mixup": 0.0,
        },
    },
}


def ensure_subset(src: Path, out: Path, train_images: int, val_images: int, test_images: int) -> Path:
    data_yaml = out / "data_fracture.yaml"
    if data_yaml.is_file():
        return data_yaml
    rc = make_subset_main(
        [
            "--src",
            str(src),
            "--out",
            str(out),
            "--train-images",
            str(train_images),
            "--val-images",
            str(val_images),
            "--test-images",
            str(test_images),
        ]
    )
    if rc != 0:
        raise RuntimeError("Falha ao criar subset")
    return data_yaml


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Treino mini (~1600) com log de métricas")
    parser.add_argument("--profile", choices=list(PROFILES), default="proposed")
    parser.add_argument("--src-dataset", default="")
    parser.add_argument("--subset-dir", default="")
    parser.add_argument("--train-images", type=int, default=1600)
    parser.add_argument("--val-images", type=int, default=200)
    parser.add_argument("--test-images", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--name", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument("--skip-subset", action="store_true")
    args = parser.parse_args(argv)

    profile = PROFILES[args.profile]
    src = Path(args.src_dataset).resolve() if args.src_dataset else (
        CODIGO_DIR / "dataset" / "yolo_8bit" / "fracture_only"
    )
    # Colab local extract fallback
    if not src.is_dir() and Path("/content/fracture_only").is_dir():
        src = Path("/content/fracture_only")

    subset = Path(args.subset_dir).resolve() if args.subset_dir else (
        src.parent / f"{src.name}_1600" if src.name != "fracture_only_1600" else src
    )
    if args.skip_subset:
        data_yaml = subset / "data_fracture.yaml"
        if not data_yaml.is_file():
            raise FileNotFoundError(data_yaml)
    else:
        data_yaml = ensure_subset(
            src, subset, args.train_images, args.val_images, args.test_images
        )

    run_name = args.name or f"mini_{args.profile}_e{args.epochs}_n{args.train_images}"
    device = resolve_device(args.device)
    project = CODIGO_DIR / "runs" / "detect"

    model = YOLO(profile["model"])
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch,
        device=device,
        workers=2,
        patience=15,
        seed=42,
        optimizer="AdamW",
        pretrained=profile["pretrained"],
        project=str(project),
        name=run_name,
        exist_ok=True,
        **profile["train_kwargs"],
    )

    best = project / run_name / "weights" / "best.pt"
    if not best.is_file():
        # fallback last.pt
        best = project / run_name / "weights" / "last.pt"
    eval_model = YOLO(str(best))
    metrics = eval_model.val(
        data=str(data_yaml),
        split="test",
        imgsz=640,
        batch=args.batch,
        device=device,
        conf=0.001,
        iou=0.7,
        plots=True,
        project=str(project),
        name=f"test_{run_name}",
        exist_ok=True,
    )

    row = {
        "timestamp": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "run": run_name,
        "n_train": args.train_images,
        "epochs": args.epochs,
        "pretrained": str(profile["pretrained"]),
        "augmentation": str(profile["augmentation"]),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "notes": args.notes or f"profile={args.profile}; subset={subset}",
    }

    out_json = CODIGO_DIR / "runs" / "experiments" / run_name / "metrics_test.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")

    append_row(row)
    tex = rebuild_tex()

    print("\n=== RESULTADO ===")
    print(f"Run: {run_name}")
    print(f"Precisão: {row['precision']:.4f}")
    print(f"Recall:   {row['recall']:.4f}")
    print(f"mAP50:    {row['map50']:.4f}")
    print(f"mAP50-95: {row['map50_95']:.4f}")
    print(f"JSON: {out_json}")
    print(f"CSV:  {CODIGO_DIR / 'runs' / 'comparisons' / 'runs_log.csv'}")
    print(f"TeX:  {tex}")

    target = 0.70
    hit = row["map50"] >= target or row["recall"] >= target or row["precision"] >= target
    print(f"Meta ~70%: {'ATINGIDA' if hit else 'ainda abaixo'} (olhar mAP50/Recall/Precisão)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

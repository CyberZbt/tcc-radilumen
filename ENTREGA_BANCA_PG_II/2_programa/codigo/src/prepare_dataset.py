"""
Prepara o GRAZPEDWRI-DX para treino com Ultralytics YOLO.

Origem (padrão): ../archive relativo a codigo/
  archive/images_part{1..4}/*.png
  archive/folder_structure/yolov5/labels/*.txt
  archive/dataset.csv

Destino: codigo/dataset/yolo/
  images/{train,val,test}/
  labels/{train,val,test}/

Imagens são vinculadas por hardlink (sem duplicar ~GB). Labels são copiados
(texto leve). Use --copy-images se hardlink falhar no seu sistema.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.preprocess import PreprocessConfig, save_preprocessed

# Índices oficiais do meta.yaml do dataset
ALL_CLASS_NAMES = [
    "boneanomaly",
    "bonelesion",
    "foreignbody",
    "fracture",
    "metal",
    "periostealreaction",
    "pronatorsign",
    "softtissue",
    "text",
]
FRACTURE_CLASS_ID = 3


def resolve_archive_root(codigo_dir: Path, archive_arg: str | None) -> Path:
    if archive_arg:
        root = Path(archive_arg).resolve()
    else:
        root = (codigo_dir.parent / "archive").resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Pasta archive não encontrada: {root}")
    return root


def index_images(archive_root: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for part in sorted(archive_root.glob("images_part*")):
        if not part.is_dir():
            continue
        for png in part.glob("*.png"):
            mapping[png.stem] = png.resolve()
    return mapping


def index_labels(archive_root: Path) -> dict[str, Path]:
    labels_dir = archive_root / "folder_structure" / "yolov5" / "labels"
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Labels YOLO não encontrados: {labels_dir}")
    return {p.stem: p.resolve() for p in labels_dir.glob("*.txt")}


def patient_splits(
    df: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, set[int]]:
    patients = sorted(df["patient_id"].unique())
    train_ids, temp_ids = train_test_split(
        patients,
        test_size=val_ratio + test_ratio,
        random_state=seed,
    )
    relative_test = test_ratio / (val_ratio + test_ratio)
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=relative_test,
        random_state=seed,
    )
    return {
        "train": set(train_ids),
        "val": set(val_ids),
        "test": set(test_ids),
    }


def filter_label_lines(
    label_path: Path,
    class_filter: set[int] | None,
    remap_to_zero: bool,
) -> list[str]:
    lines: list[str] = []
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return lines
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        if class_filter is not None and cls not in class_filter:
            continue
        if remap_to_zero:
            parts[0] = "0"
        lines.append(" ".join(parts))
    return lines


def place_image(
    src: Path,
    dst: Path,
    use_copy: bool,
    preprocess: bool,
    preprocess_cfg: PreprocessConfig | None,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if preprocess:
        save_preprocessed(src, dst, preprocess_cfg)
        return
    if use_copy:
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_label(
    src_label: Path,
    dst_label: Path,
    class_filter: set[int] | None,
    remap_to_zero: bool,
) -> None:
    dst_label.parent.mkdir(parents=True, exist_ok=True)
    lines = filter_label_lines(src_label, class_filter, remap_to_zero)
    dst_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def build_split_lists(
    df: pd.DataFrame,
    images: dict[str, Path],
    labels: dict[str, Path],
    splits: dict[str, set[int]],
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {k: [] for k in splits}
    missing_img = 0
    missing_label = 0
    for row in df.itertuples(index=False):
        filestem = row.filestem
        patient_id = int(row.patient_id)
        split_name = None
        for name, ids in splits.items():
            if patient_id in ids:
                split_name = name
                break
        if split_name is None:
            continue
        if filestem not in images:
            missing_img += 1
            continue
        if filestem not in labels:
            missing_label += 1
            continue
        result[split_name].append(filestem)
    if missing_img:
        print(f"Aviso: {missing_img} entradas do CSV sem imagem correspondente.")
    if missing_label:
        print(f"Aviso: {missing_label} entradas do CSV sem label correspondente.")
    return result


def write_data_yaml(
    out_root: Path,
    class_names: list[str],
    yaml_path: Path,
) -> None:
    data = {
        "path": str(out_root.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": class_names,
    }
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preparar GRAZPEDWRI-DX para YOLO")
    parser.add_argument(
        "--archive",
        help="Pasta archive (padrão: ../archive relativo a codigo/)",
    )
    parser.add_argument(
        "--out",
        help="Saída YOLO (padrão: codigo/dataset/yolo)",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--classes",
        choices=["all", "fracture"],
        default="all",
        help="all = 9 classes oficiais; fracture = apenas fraturas (classe única)",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copiar imagens em vez de hardlink",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Apenas mostra contagens do split, sem criar arquivos",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Aplicar 16-bit→8-bit + CLAHE ao salvar imagens (em vez de hardlink)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Limitar imagens por split (0 = todas; útil para smoke test)",
    )
    parser.add_argument(
        "--config",
        help="YAML com parâmetros de pré-processamento (padrão: configs/default.yaml)",
    )
    args = parser.parse_args(argv)

    codigo_dir = Path(__file__).resolve().parents[1]
    archive_root = resolve_archive_root(codigo_dir, args.archive)
    out_root = Path(args.out).resolve() if args.out else codigo_dir / "dataset" / "yolo"
    if args.preprocess and args.out is None:
        out_root = codigo_dir / "dataset" / "yolo_8bit"

    preprocess_cfg: PreprocessConfig | None = None
    if args.preprocess:
        cfg_path = Path(args.config).resolve() if args.config else codigo_dir / "configs" / "default.yaml"
        if cfg_path.is_file():
            raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            pp = raw.get("preprocess", {})
            preprocess_cfg = PreprocessConfig(
                clahe=pp.get("clahe", True),
                clip_limit=float(pp.get("clip_limit", 2.0)),
                tile_grid_size=int(pp.get("tile_grid_size", 8)),
                normalize=pp.get("normalize", "percentile"),
                percentile_low=float(pp.get("percentile_low", 1.0)),
                percentile_high=float(pp.get("percentile_high", 99.0)),
            )
        else:
            preprocess_cfg = PreprocessConfig()

    csv_path = archive_root / "dataset.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"dataset.csv não encontrado: {csv_path}")

    df = pd.read_csv(csv_path)
    images = index_images(archive_root)
    labels = index_labels(archive_root)

    print(f"Archive: {archive_root}")
    print(f"Imagens indexadas: {len(images)}")
    print(f"Labels indexados: {len(labels)}")
    print(f"CSV: {len(df)} linhas")

    splits = patient_splits(df, args.val_ratio, args.test_ratio, args.seed)
    filestems_by_split = build_split_lists(df, images, labels, splits)

    for name in ("train", "val", "test"):
        n_patients = len(splits[name])
        n_images = len(filestems_by_split[name])
        print(f"Split {name}: {n_patients} pacientes, {n_images} imagens")

    if args.dry_run:
        return 0

    class_filter: set[int] | None = None
    remap_to_zero = False
    if args.classes == "fracture":
        class_filter = {FRACTURE_CLASS_ID}
        remap_to_zero = True
        class_names = ["fracture"]
        out_root = out_root / "fracture_only"
        yaml_name = "data_fracture.yaml"
    else:
        class_names = ALL_CLASS_NAMES
        yaml_name = "data.yaml"

    for split_name, stems in filestems_by_split.items():
        if args.max_images > 0:
            stems = stems[:args.max_images]
        for stem in stems:
            img_src = images[stem]
            lbl_src = labels[stem]
            img_dst = out_root / "images" / split_name / f"{stem}.png"
            lbl_dst = out_root / "labels" / split_name / f"{stem}.txt"
            place_image(img_src, img_dst, args.copy_images, args.preprocess, preprocess_cfg)
            write_label(lbl_src, lbl_dst, class_filter, remap_to_zero)

    write_data_yaml(out_root, class_names, out_root / yaml_name)
    print(f"Dataset YOLO em: {out_root}")
    print(f"Config: {out_root / yaml_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

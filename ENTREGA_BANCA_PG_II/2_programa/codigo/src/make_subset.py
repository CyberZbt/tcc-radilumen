"""
Cria subset YOLO com ~N imagens de treino (hardlink/symlink ou cópia).

Uso (a partir de dataset já preparado):
  python -m src.make_subset --src dataset/yolo_8bit/fracture_only --train-images 1600
  python -m src.make_subset --src /content/fracture_only --train-images 1600 --out /content/fracture_only_1600
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

import yaml

from src.utils import CODIGO_DIR


def list_stems(images_dir: Path) -> list[str]:
    if not images_dir.is_dir():
        return []
    return sorted(p.stem for p in images_dir.glob("*.png"))


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        try:
            os.symlink(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def place_split(
    src_root: Path,
    dst_root: Path,
    split: str,
    stems: list[str],
    mode: str,
) -> None:
    for stem in stems:
        img_src = src_root / "images" / split / f"{stem}.png"
        lbl_src = src_root / "labels" / split / f"{stem}.txt"
        if not img_src.is_file():
            raise FileNotFoundError(img_src)
        link_or_copy(img_src, dst_root / "images" / split / f"{stem}.png", mode)
        if lbl_src.is_file():
            link_or_copy(lbl_src, dst_root / "labels" / split / f"{stem}.txt", mode)
        else:
            (dst_root / "labels" / split / f"{stem}.txt").write_text("", encoding="utf-8")


def sample_stems(stems: list[str], n: int, seed: int) -> list[str]:
    if n <= 0 or n >= len(stems):
        return list(stems)
    rng = random.Random(seed)
    picked = stems[:]
    rng.shuffle(picked)
    return sorted(picked[:n])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Subset YOLO (~1600 treino)")
    parser.add_argument(
        "--src",
        default=str(CODIGO_DIR / "dataset" / "yolo_8bit" / "fracture_only"),
        help="Dataset YOLO completo (fracture_only)",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Destino (padrão: <src>_1600 ao lado do src)",
    )
    parser.add_argument("--train-images", type=int, default=1600)
    parser.add_argument("--val-images", type=int, default=200)
    parser.add_argument("--test-images", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        choices=["hardlink", "copy"],
        default="hardlink",
        help="hardlink (rápido) ou copy",
    )
    args = parser.parse_args(argv)

    src = Path(args.src).resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"Dataset fonte não encontrado: {src}")

    out = Path(args.out).resolve() if args.out else src.parent / f"{src.name}_1600"
    if out.exists():
        shutil.rmtree(out)

    yaml_src = src / "data_fracture.yaml"
    if not yaml_src.is_file():
        yaml_src = src / "data.yaml"
    if not yaml_src.is_file():
        raise FileNotFoundError(f"data.yaml ausente em {src}")

    counts = {}
    for split, n in (
        ("train", args.train_images),
        ("val", args.val_images),
        ("test", args.test_images),
    ):
        stems = list_stems(src / "images" / split)
        if not stems:
            raise FileNotFoundError(f"Sem imagens em {src / 'images' / split}")
        chosen = sample_stems(stems, n, args.seed + hash(split) % 10_000)
        place_split(src, out, split, chosen, args.mode)
        counts[split] = len(chosen)
        print(f"{split}: {len(chosen)} / {len(stems)} imagens")

    raw = yaml.safe_load(yaml_src.read_text(encoding="utf-8"))
    raw["path"] = str(out)
    raw["train"] = "images/train"
    raw["val"] = "images/val"
    raw["test"] = "images/test"
    out_yaml = out / "data_fracture.yaml"
    out_yaml.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    meta = {
        "source": str(src),
        "out": str(out),
        "seed": args.seed,
        "counts": counts,
        "total": sum(counts.values()),
    }
    (out / "subset_meta.json").write_text(
        __import__("json").dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Subset em: {out}")
    print(f"YAML: {out_yaml}")
    print(f"Total imagens: {meta['total']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

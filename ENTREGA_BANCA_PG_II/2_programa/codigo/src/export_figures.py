"""
Exporta figuras de resultados para o TCC (LaTeX/Figuras).

No Colab (após treino / com pesos no Drive):
  python -m src.export_figures --weights runs/detect/proposed/weights/final_proposed.pt
  python -m src.export_figures --weights runs/detect/mini_proposed_e30_n1600/weights/best.pt --data /content/fracture_only_1600/data_fracture.yaml

Gera em latex/Figuras/ (ou --out):
  - resultados_deteccoes.png
  - resultados_pr_curve.png (se existir na run)
  - resultados_confusion.png (se existir)
  - resultados_curves.png (se existir)
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.utils import CODIGO_DIR


def find_plot(run_dir: Path, names: list[str]) -> Path | None:
    for name in names:
        for path in run_dir.rglob(name):
            if path.is_file():
                return path
    return None


def copy_if_found(src: Path | None, dst: Path) -> bool:
    if src is None or not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"OK: {dst.name} <- {src}")
    return True


def list_test_images(data_yaml: Path, limit: int, seed: int = 42) -> list[Path]:
    import yaml

    raw = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(str(raw.get("path", data_yaml.parent))).expanduser()
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()

    candidates = []
    for key in ("test", "val"):
        rel = raw.get(key)
        if not rel:
            continue
        p = Path(rel)
        candidates.append(p if p.is_absolute() else root / p)
    candidates.extend(
        [
            root / "images" / "test",
            root / "images" / "val",
            data_yaml.parent / "images" / "test",
            data_yaml.parent / "images" / "val",
        ]
    )

    test_dir = next((d for d in candidates if d.is_dir()), None)
    if test_dir is None:
        raise FileNotFoundError(
            "Sem pasta de imagens test/val. YAML="
            f"{data_yaml} path={root} tentativas={candidates}"
        )

    imgs = sorted(test_dir.glob("*.png")) + sorted(test_dir.glob("*.jpg"))
    if not imgs:
        raise FileNotFoundError(f"Sem imagens em {test_dir}")
    print(f"Usando {min(limit, len(imgs))} imagens de {test_dir} (total={len(imgs)})")
    rng = random.Random(seed)
    pick = imgs[:]
    rng.shuffle(pick)
    return pick[:limit]


def _letterbox(img: np.ndarray, height: int, width: int) -> np.ndarray:
    """Redimensiona mantendo proporção e preenche com branco até (height, width)."""
    h, w = img.shape[:2]
    scale = min(height / h, width / w)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    y0 = (height - nh) // 2
    x0 = (width - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def make_detection_grid(
    weights: Path,
    images: list[Path],
    out_path: Path,
    conf: float = 0.25,
    cols: int = 3,
) -> Path:
    model = YOLO(str(weights))
    tiles: list[np.ndarray] = []
    cell_h, cell_w = 320, 320

    for img_path in images:
        results = model.predict(source=str(img_path), conf=conf, verbose=False)
        plotted = results[0].plot()  # BGR
        tiles.append(_letterbox(plotted, cell_h, cell_w))

    # completa a última linha com células brancas
    while len(tiles) % cols:
        tiles.append(np.full((cell_h, cell_w, 3), 255, dtype=np.uint8))

    rows = [np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)]
    grid = np.vstack(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)
    print(f"OK: grade de detecções -> {out_path}")
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Exportar figuras para o TCC")
    parser.add_argument("--weights", required=True, help="Caminho do .pt")
    parser.add_argument(
        "--data",
        default="",
        help="data_fracture.yaml (padrão: yolo_8bit fracture_only ou /content/...)",
    )
    parser.add_argument("--out", default="", help="Pasta Figuras (padrão: latex/Figuras)")
    parser.add_argument("--n", type=int, default=6, help="Nº de imagens na grade")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--run-dir", default="", help="Pasta da run Ultralytics para copiar plots")
    args = parser.parse_args(argv)

    weights = Path(args.weights)
    if not weights.is_file():
        # tentar relativo ao CODIGO_DIR
        alt = CODIGO_DIR / args.weights
        if alt.is_file():
            weights = alt
        else:
            raise FileNotFoundError(f"Pesos não encontrados: {args.weights}")

    out_dir = Path(args.out) if args.out else CODIGO_DIR.parent / "latex" / "Figuras"
    out_dir.mkdir(parents=True, exist_ok=True)

    data_yaml = None
    if args.data:
        data_yaml = Path(args.data)
    else:
        candidates = [
            Path("/content/fracture_only/data_fracture.yaml"),
            Path("/content/fracture_only_1600/data_fracture.yaml"),
            CODIGO_DIR / "dataset" / "yolo_8bit" / "fracture_only" / "data_fracture.yaml",
        ]
        data_yaml = next((p for p in candidates if p.is_file()), None)
    if data_yaml is None or not data_yaml.is_file():
        raise FileNotFoundError("Informe --data com o data_fracture.yaml")

    # 1) grade de detecções
    imgs = list_test_images(data_yaml, args.n)
    make_detection_grid(weights, imgs, out_dir / "resultados_deteccoes.png", conf=args.conf)

    # 2) plots da run (se houver)
    run_dir = Path(args.run_dir) if args.run_dir else weights.parents[1]  # .../name/weights/best.pt → name
    if run_dir.name == "weights":
        run_dir = run_dir.parent
    copy_if_found(
        find_plot(run_dir, ["PR_curve.png", "BoxPR_curve.png"]),
        out_dir / "resultados_pr_curve.png",
    )
    copy_if_found(
        find_plot(run_dir, ["confusion_matrix.png", "confusion_matrix_normalized.png"]),
        out_dir / "resultados_confusion.png",
    )
    copy_if_found(
        find_plot(run_dir, ["results.png"]),
        out_dir / "resultados_curves.png",
    )

    print(f"Figuras em: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

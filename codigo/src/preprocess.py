"""
Pré-processamento de radiografias GRAZPEDWRI-DX (PNG 16-bit → 8-bit).

Pipeline alinhado à metodologia do TCC: escala de cinza, CLAHE e normalização
para compatibilizar a entrada com YOLO e reduzir variabilidade entre exames.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

NormalizeMode = Literal["minmax", "percentile"]


@dataclass(frozen=True)
class PreprocessConfig:
    clahe: bool = True
    clip_limit: float = 2.0
    tile_grid_size: int = 8
    normalize: NormalizeMode = "percentile"
    percentile_low: float = 1.0
    percentile_high: float = 99.0
    output_bgr: bool = True


def read_grayscale(path: Path | str) -> np.ndarray:
    """Lê PNG (8 ou 16 bits) como matriz 2D."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Não foi possível ler a imagem: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def to_uint8(
    img: np.ndarray,
    mode: NormalizeMode = "percentile",
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> np.ndarray:
    """Converte matriz de intensidade para uint8."""
    if img.dtype == np.uint8:
        return img

    work = img.astype(np.float32)
    if mode == "minmax":
        lo, hi = float(work.min()), float(work.max())
    else:
        lo, hi = np.percentile(work, [p_low, p_high])
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((work - lo) / (hi - lo), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def apply_clahe(
    img_uint8: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
) -> np.ndarray:
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_grid_size, tile_grid_size),
    )
    return clahe.apply(img_uint8)


def preprocess_array(img: np.ndarray, config: PreprocessConfig | None = None) -> np.ndarray:
    """Aplica normalização + CLAHE opcional em matriz 2D."""
    cfg = config or PreprocessConfig()
    gray = to_uint8(
        img,
        mode=cfg.normalize,
        p_low=cfg.percentile_low,
        p_high=cfg.percentile_high,
    )
    if cfg.clahe:
        gray = apply_clahe(gray, cfg.clip_limit, cfg.tile_grid_size)
    if cfg.output_bgr:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return gray


def preprocess_image(
    path: Path | str,
    config: PreprocessConfig | None = None,
) -> np.ndarray:
    """Lê e pré-processa uma radiografia."""
    return preprocess_array(read_grayscale(path), config)


def save_preprocessed(
    src: Path | str,
    dst: Path | str,
    config: PreprocessConfig | None = None,
) -> Path:
    """Pré-processa e salva PNG 8-bit em canal único para economizar espaço."""
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = config or PreprocessConfig()
    out = preprocess_image(src, replace(cfg, output_bgr=False))
    cv2.imwrite(str(dst_path), out)
    return dst_path

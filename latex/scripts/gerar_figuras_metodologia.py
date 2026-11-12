"""Gera figuras ilustrativas para o capítulo de Metodologia do TCC."""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "Figuras"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "figure.dpi": 150,
})


def salvar(fig, nome):
    path = OUT / nome
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"OK: {path} ({path.stat().st_size} bytes)")


def fig_grazpedwri_fluxo():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    etapas = [
        ("PACS hospitalar\n(DICOM)", "#4C78A8"),
        ("De-identificação\ne conversão PNG", "#72B7B2"),
        ("Anotação por\nradiologistas", "#F58518"),
        ("Exportação\nVOC / YOLO", "#E45756"),
    ]
    xs = [0.6, 3.0, 5.4, 7.8]
    for x, (txt, cor) in zip(xs, etapas):
        box = FancyBboxPatch(
            (x, 1.2), 2.0, 1.6, boxstyle="round,pad=0.08,rounding_size=0.12",
            linewidth=1.2, edgecolor="#333333", facecolor=cor, alpha=0.85,
        )
        ax.add_patch(box)
        ax.text(x + 1.0, 2.0, txt, ha="center", va="center", color="white", fontweight="bold")

    for i in range(len(xs) - 1):
        ax.annotate(
            "", xy=(xs[i + 1], 2.0), xytext=(xs[i] + 2.0, 2.0),
            arrowprops=dict(arrowstyle="->", lw=1.8, color="#333333"),
        )

    ax.text(5.0, 3.35, "Construção do dataset GRAZPEDWRI-DX", ha="center", fontsize=12, fontweight="bold")
    ax.text(5.0, 0.45, "20.327 imagens | 67.771 objetos anotados | formatos VOC e YOLO", ha="center", fontsize=9, color="#555555")
    salvar(fig, "grazpedwri_fluxo.png")


def fig_yolo_grade():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    ax = axes[0]
    ax.set_title("Radiografia de entrada", fontweight="bold")
    img = np.zeros((120, 120))
    img[20:100, 40:80] = 0.55
    img[30:90, 50:70] = 0.75
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")

    ax = axes[1]
    ax.set_title("Grade YOLO e predição", fontweight="bold")
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    for i in range(0, 121, 30):
        ax.axhline(i - 0.5, color="#00BFFF", lw=0.8, alpha=0.8)
        ax.axvline(i - 0.5, color="#00BFFF", lw=0.8, alpha=0.8)
    rect = Rectangle((48, 38), 28, 34, linewidth=2.2, edgecolor="#FF4136", facecolor="none")
    ax.add_patch(rect)
    ax.text(62, 72, "fratura\np_c = 0.87", color="#FF4136", ha="center", fontsize=9, fontweight="bold")
    ax.axis("off")

    fig.suptitle("Detecção YOLO: divisão em células e regressão de bounding box", fontsize=12, fontweight="bold", y=1.02)
    salvar(fig, "yolo_grade_deteccao.png")


def fig_data_augmentation():
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.8))
    rng = np.random.default_rng(42)
    base = np.zeros((80, 80))
    base[15:65, 30:50] = 0.6
    base[25:55, 35:45] = 0.82
    base += rng.normal(0, 0.03, base.shape)
    base = np.clip(base, 0, 1)

    variantes = [
        ("Original", base),
        ("Rotação", np.rot90(base, 1)),
        ("Espelhamento", np.fliplr(base)),
        ("Contraste", np.clip(base * 1.35 - 0.15, 0, 1)),
    ]

    for ax, (titulo, data) in zip(axes, variantes):
        ax.imshow(data, cmap="gray", vmin=0, vmax=1)
        ax.set_title(titulo, fontweight="bold")
        ax.axis("off")

    fig.suptitle("Exemplos de Data Augmentation aplicados no treinamento", fontsize=12, fontweight="bold", y=1.05)
    salvar(fig, "data_augmentation_metodologia.png")


def fig_bounding_box():
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ax.imshow(np.ones((80, 100)) * 0.92, cmap="gray", extent=[1, 9, 1.5, 7], alpha=0.35)
    rect = Rectangle((3.2, 3.0), 3.6, 2.8, linewidth=2.5, edgecolor="#E45756", facecolor="none")
    ax.add_patch(rect)
    ax.plot(4.9, 4.4, "o", color="#E45756", markersize=8)
    ax.text(4.9, 4.0, r"$(b_x, b_y)$", ha="center", fontsize=11, color="#E45756", fontweight="bold")
    ax.annotate("", xy=(6.8, 3.0), xytext=(3.2, 3.0), arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.5))
    ax.text(5.0, 2.55, r"$b_w$", ha="center", fontsize=11)
    ax.annotate("", xy=(6.8, 5.8), xytext=(6.8, 3.0), arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.5))
    ax.text(7.25, 4.4, r"$b_h$", va="center", fontsize=11)
    ax.text(5.0, 7.35, r"Vetor de saída: $\hat{y} = [p_c, b_x, b_y, b_w, b_h]$", ha="center", fontsize=12, fontweight="bold")
    ax.text(5.0, 0.8, "Representação esquemática da caixa delimitadora prevista pelo modelo", ha="center", fontsize=9, color="#555555")
    salvar(fig, "bounding_box_esquema.png")


if __name__ == "__main__":
    fig_grazpedwri_fluxo()
    fig_yolo_grade()
    fig_data_augmentation()
    fig_bounding_box()

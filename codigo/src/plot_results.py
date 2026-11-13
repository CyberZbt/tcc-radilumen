"""
Gera gráficos de resultados para o TCC a partir do CSV de comparações
(ou valores oficiais embutidos).

Uso:
  python -m src.plot_results
  python -m src.plot_results --out ../latex/Figuras
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils import CODIGO_DIR

# Estilo limpo para monografia
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "figure.dpi": 160,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# Dados oficiais (fallback se CSV ausente)
DEFAULT_ROWS = [
    {
        "run": "mini_baseline_e30_n1600",
        "label": "Baseline\n(exploratório)",
        "n_train": 1600,
        "precision": 0.6715,
        "recall": 0.4699,
        "map50": 0.5802,
        "map50_95": 0.2789,
    },
    {
        "run": "mini_proposed_e30_n1600",
        "label": "Proposta\n(exploratório)",
        "n_train": 1600,
        "precision": 0.9082,
        "recall": 0.7377,
        "map50": 0.8443,
        "map50_95": 0.4508,
    },
    {
        "run": "proposed_full_test",
        "label": "Proposta\n(completo)",
        "n_train": 16211,
        "precision": 0.9221,
        "recall": 0.8637,
        "map50": 0.9294,
        "map50_95": 0.5387,
    },
]

LABEL_MAP = {
    "mini_baseline_e30_n1600": "Baseline\n(exploratório)",
    "mini_proposed_e30_n1600": "Proposta\n(exploratório)",
    "proposed_full_test": "Proposta\n(completo)",
}


def load_rows(csv_path: Path) -> list[dict]:
    if not csv_path.is_file():
        return DEFAULT_ROWS
    with csv_path.open(encoding="utf-8") as f:
        raw = list(csv.DictReader(f))
    if not raw:
        return DEFAULT_ROWS
    rows = []
    for r in raw:
        key = r.get("run", "")
        rows.append(
            {
                "run": key,
                "label": LABEL_MAP.get(key, key),
                "n_train": int(float(r["n_train"])) if r.get("n_train") else 0,
                "precision": float(r["precision"]),
                "recall": float(r["recall"]),
                "map50": float(r["map50"]),
                "map50_95": float(r["map50_95"]) if r.get("map50_95") else 0.0,
            }
        )
    # ordem acadêmica
    order = list(LABEL_MAP.keys())
    rows.sort(key=lambda x: order.index(x["run"]) if x["run"] in order else 99)
    return rows


def _pct(values: list[float]) -> list[float]:
    return [v * 100 for v in values]


def plot_subset_comparison(rows: list[dict], out: Path) -> None:
    """Barras agrupadas: baseline vs proposta (subset)."""
    base = next(r for r in rows if "baseline" in r["run"])
    prop = next(r for r in rows if "mini_proposed" in r["run"])
    metrics = ["Precisão", "Recall", "mAP50", "mAP50-95"]
    b = _pct([base["precision"], base["recall"], base["map50"], base["map50_95"]])
    p = _pct([prop["precision"], prop["recall"], prop["map50"], prop["map50_95"]])

    x = np.arange(len(metrics))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    bars_b = ax.bar(x - width / 2, b, width, label="Baseline (N=1.600)", color="#6c757d")
    bars_p = ax.bar(x + width / 2, p, width, label="Proposta (N=1.600)", color="#0d6efd")
    ax.axhline(70, color="#dc3545", linestyle="--", linewidth=1.2, label="Meta 70%")
    ax.set_ylabel("Valor (%)")
    ax.set_title("Comparação exploratória — Baseline vs Proposta (subset)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    for bars in (bars_b, bars_p):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.savefig(out)
    plt.close(fig)
    print(f"OK: {out}")


def plot_evolution(rows: list[dict], out: Path) -> None:
    """Evolução mAP50 e Recall ao longo dos experimentos."""
    labels = [r["label"].replace("\n", " ") for r in rows]
    map50 = _pct([r["map50"] for r in rows])
    recall = _pct([r["recall"] for r in rows])
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.plot(x, map50, "o-", color="#0d6efd", linewidth=2, markersize=8, label="mAP50")
    ax.plot(x, recall, "s--", color="#198754", linewidth=2, markersize=8, label="Recall")
    ax.axhline(70, color="#dc3545", linestyle="--", linewidth=1.2, label="Meta 70%")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Valor (%)")
    ax.set_title("Evolução dos treinos (conjunto de teste)")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    for i, (m, r) in enumerate(zip(map50, recall)):
        ax.annotate(f"{m:.1f}%", (i, m), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
        ax.annotate(f"{r:.1f}%", (i, r), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=8, color="#198754")

    fig.savefig(out)
    plt.close(fig)
    print(f"OK: {out}")


def plot_full_metrics(rows: list[dict], out: Path) -> None:
    """Barras das métricas oficiais da proposta completa."""
    full = next(r for r in rows if r["run"] == "proposed_full_test")
    metrics = ["Precisão", "Recall", "mAP50", "mAP50-95"]
    values = _pct([full["precision"], full["recall"], full["map50"], full["map50_95"]])
    colors = ["#0d6efd", "#198754", "#6610f2", "#fd7e14"]

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    bars = ax.bar(metrics, values, color=colors, width=0.65)
    ax.axhline(70, color="#dc3545", linestyle="--", linewidth=1.2, label="Meta 70%")
    ax.set_ylabel("Valor (%)")
    ax.set_title("Métricas oficiais — Proposta (conjunto de teste, N=2.026)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    for bar, val in zip(bars, values):
        ax.annotate(
            f"{val:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.savefig(out)
    plt.close(fig)
    print(f"OK: {out}")


def plot_gain(rows: list[dict], out: Path) -> None:
    """Ganho da proposta sobre o baseline no subset (pontos percentuais)."""
    base = next(r for r in rows if "baseline" in r["run"])
    prop = next(r for r in rows if "mini_proposed" in r["run"])
    metrics = ["Precisão", "Recall", "mAP50", "mAP50-95"]
    gains = [
        (prop["precision"] - base["precision"]) * 100,
        (prop["recall"] - base["recall"]) * 100,
        (prop["map50"] - base["map50"]) * 100,
        (prop["map50_95"] - base["map50_95"]) * 100,
    ]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    colors = ["#198754" if g >= 0 else "#dc3545" for g in gains]
    bars = ax.bar(metrics, gains, color=colors, width=0.65)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Ganho (pontos percentuais)")
    ax.set_title("Ganho da proposta sobre o baseline (subset N=1.600)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    for bar, g in zip(bars, gains):
        ax.annotate(
            f"{g:+.1f} pp",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3 if g >= 0 else -12),
            textcoords="offset points",
            ha="center",
            va="bottom" if g >= 0 else "top",
            fontsize=9,
        )
    fig.savefig(out)
    plt.close(fig)
    print(f"OK: {out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gerar gráficos de resultados do TCC")
    parser.add_argument(
        "--csv",
        default=str(CODIGO_DIR / "runs" / "comparisons" / "runs_log.csv"),
    )
    parser.add_argument(
        "--out",
        default=str(CODIGO_DIR.parent / "latex" / "Figuras"),
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(Path(args.csv))

    plot_subset_comparison(rows, out_dir / "grafico_baseline_vs_proposta.png")
    plot_evolution(rows, out_dir / "grafico_evolucao_treinos.png")
    plot_full_metrics(rows, out_dir / "grafico_metricas_oficiais.png")
    plot_gain(rows, out_dir / "grafico_ganho_proposta.png")

    print(f"Gráficos em: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

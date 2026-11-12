"""
Registra métricas de cada treino em CSV + JSONL + tabela LaTeX (acumulativa).

Uso:
  # após avaliar um modelo
  python -m src.log_metrics --run mini_proposed_e30 --map50 0.71 --precision 0.68 --recall 0.72 --map50-95 0.41

  # ou a partir de metrics_test.json
  python -m src.log_metrics --from-json runs/experiments/proposed/metrics_test.json --run proposed_full

  # regenerar só a tabela LaTeX
  python -m src.log_metrics --rebuild-tex
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils import CODIGO_DIR

COMPARE_DIR = CODIGO_DIR / "runs" / "comparisons"
CSV_PATH = COMPARE_DIR / "runs_log.csv"
JSONL_PATH = COMPARE_DIR / "runs_log.jsonl"
TEX_PATH = CODIGO_DIR.parent / "latex" / "resultados" / "tabela_comparacao_treinos.tex"

FIELDNAMES = [
    "timestamp",
    "run",
    "n_train",
    "epochs",
    "pretrained",
    "augmentation",
    "precision",
    "recall",
    "map50",
    "map50_95",
    "notes",
]


def _pct(value: float | None) -> str:
    if value is None:
        return "---"
    return f"{value * 100:.2f}".replace(".", ",")


def append_row(row: dict[str, Any]) -> None:
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not CSV_PATH.is_file()
    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in FIELDNAMES})
    with JSONL_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_rows() -> list[dict[str, str]]:
    if not CSV_PATH.is_file():
        return []
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_latex(rows: list[dict[str, str]]) -> str:
    """Tabela compacta (nomes curtos) para caber na página ABNT."""
    label_map = {
        "proposed_full_test": "Proposta (completo)",
        "mini_proposed_e30_n1600": "Proposta (exploratório)",
        "mini_baseline_e30_n1600": "Baseline (exploratório)",
    }
    # Ordem preferida na monografia
    preferred = [
        "mini_baseline_e30_n1600",
        "mini_proposed_e30_n1600",
        "proposed_full_test",
    ]
    by_run = {r.get("run"): r for r in rows}
    ordered = [by_run[k] for k in preferred if k in by_run]
    for r in rows:
        if r.get("run") not in preferred:
            ordered.append(r)

    lines = [
        "% Gerado automaticamente — codigo/src/log_metrics.py",
        "% Tabela compacta de evolução dos treinos",
        "\\begin{table}[htb]",
        "    \\centering",
        "    \\footnotesize",
        "    \\caption{EVOLUÇÃO DOS TREINOS (TESTE)}",
        "    \\label{tab:comparacao_treinos}",
        "    \\begin{tabular}{lccc}",
        "        \\toprule",
        "        \\textbf{Experimento} & \\textbf{N} & \\textbf{mAP50} & \\textbf{Recall} \\\\",
        "        \\midrule",
    ]
    if not ordered:
        lines.append("        --- & --- & --- & --- \\\\")
    else:
        for r in ordered:
            try:
                rec = _pct(float(r["recall"])) if r.get("recall") not in ("", None) else "---"
                m50 = _pct(float(r["map50"])) if r.get("map50") not in ("", None) else "---"
                if rec != "---":
                    rec = rec + "\\%"
                if m50 != "---":
                    m50 = m50 + "\\%"
            except ValueError:
                rec = m50 = "---"
            key = r.get("run") or "---"
            name = label_map.get(key, key.replace("_", "\\_"))
            n_train = r.get("n_train") or "---"
            if str(n_train).isdigit():
                n_train = f"{int(n_train):,}".replace(",", ".")
            lines.append(f"        {name} & {n_train} & {m50} & {rec} \\\\")
    lines.extend(
        [
            "        \\bottomrule",
            "    \\end{tabular}",
            "    \\fonte{Elaborado pelo autor, 2026}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def rebuild_tex() -> Path:
    rows = load_rows()
    TEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEX_PATH.write_text(build_latex(rows), encoding="utf-8")
    return TEX_PATH


def row_from_json(path: Path, run: str, extras: dict[str, Any]) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "timestamp": data.get("timestamp") or datetime.now(timezone.utc).isoformat(),
        "run": run or data.get("experiment") or path.stem,
        "n_train": extras.get("n_train", ""),
        "epochs": extras.get("epochs", ""),
        "pretrained": extras.get("pretrained", ""),
        "augmentation": extras.get("augmentation", ""),
        "precision": data.get("precision"),
        "recall": data.get("recall"),
        "map50": data.get("map50"),
        "map50_95": data.get("map50_95"),
        "notes": extras.get("notes", data.get("notes", "")),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Log acumulativo de métricas dos treinos")
    parser.add_argument("--run", default="", help="Nome do experimento / run")
    parser.add_argument("--from-json", help="metrics_test.json para importar")
    parser.add_argument("--precision", type=float)
    parser.add_argument("--recall", type=float)
    parser.add_argument("--map50", type=float)
    parser.add_argument("--map50-95", type=float, dest="map50_95")
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--pretrained", default="")
    parser.add_argument("--augmentation", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument("--rebuild-tex", action="store_true")
    parser.add_argument("--no-tex", action="store_true")
    args = parser.parse_args(argv)

    if args.rebuild_tex and not args.from_json and args.map50 is None:
        out = rebuild_tex()
        print(f"Tabela LaTeX: {out}")
        print(f"Linhas: {len(load_rows())}")
        return 0

    extras = {
        "n_train": args.n_train if args.n_train is not None else "",
        "epochs": args.epochs if args.epochs is not None else "",
        "pretrained": args.pretrained,
        "augmentation": args.augmentation,
        "notes": args.notes,
    }

    if args.from_json:
        row = row_from_json(Path(args.from_json), args.run, extras)
    else:
        if args.map50 is None or args.precision is None or args.recall is None:
            print("Informe --from-json ou --precision/--recall/--map50/--map50-95")
            return 1
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run": args.run or "run",
            **extras,
            "precision": args.precision,
            "recall": args.recall,
            "map50": args.map50,
            "map50_95": args.map50_95 if args.map50_95 is not None else "",
        }

    append_row(row)
    print(f"Registrado: {row['run']} | mAP50={row['map50']} | P={row['precision']} | R={row['recall']}")
    print(f"CSV: {CSV_PATH}")

    if not args.no_tex:
        out = rebuild_tex()
        print(f"Tabela LaTeX: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

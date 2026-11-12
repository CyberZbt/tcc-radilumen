"""
Exporta métricas dos experimentos para tabela LaTeX.

Exemplo:
  python -m src.export_metrics
  python -m src.export_metrics --out ../latex/resultados/tabela_metricas.tex
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.utils import CODIGO_DIR


def load_metrics(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}".replace(".", ",")


def build_latex_table(baseline: dict | None, proposed: dict | None) -> str:
    b_map = format_pct(baseline["map50"]) if baseline else "---"
    p_map = format_pct(proposed["map50"]) if proposed else "---"
    b_map95 = format_pct(baseline["map50_95"]) if baseline else "---"
    p_map95 = format_pct(proposed["map50_95"]) if proposed else "---"
    b_prec = format_pct(baseline["precision"]) if baseline else "---"
    p_prec = format_pct(proposed["precision"]) if proposed else "---"
    b_rec = format_pct(baseline["recall"]) if baseline else "---"
    p_rec = format_pct(proposed["recall"]) if proposed else "---"

    return f"""% Gerado automaticamente — codigo/src/export_metrics.py
\\begin{{table}}[htb]
    \\centering
    \\caption{{COMPARAÇÃO DE MÉTRICAS NO CONJUNTO DE TESTE (IOU = 0,5)}}
    \\label{{tab:metricas_experimentos}}
    \\begin{{tabular}}{{lcc}}
        \\toprule
        \\textbf{{Métrica}} & \\textbf{{Baseline}} & \\textbf{{Proposta}} \\\\
        \\midrule
        mAP@0,5 (\\%) & {b_map} & {p_map} \\\\
        mAP@0,5:0,95 (\\%) & {b_map95} & {p_map95} \\\\
        Precisão (\\%) & {b_prec} & {p_prec} \\\\
        Sensibilidade / Recall (\\%) & {b_rec} & {p_rec} \\\\
        \\bottomrule
    \\end{{tabular}}
    \\fonte{{Elaborado pelo autor, 2026}}
\\end{{table}}
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Exportar métricas para LaTeX")
    parser.add_argument(
        "--out",
        default=str(CODIGO_DIR.parent / "latex" / "resultados" / "tabela_metricas.tex"),
    )
    args = parser.parse_args(argv)

    baseline = load_metrics(CODIGO_DIR / "runs" / "experiments" / "baseline" / "metrics_test.json")
    proposed = load_metrics(CODIGO_DIR / "runs" / "experiments" / "proposed" / "metrics_test.json")

    if not baseline and not proposed:
        print("Nenhuma metrica encontrada. Rode: python -m src.run_experiments")

    latex = build_latex_table(baseline, proposed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(latex, encoding="utf-8")
    print(f"Tabela LaTeX: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

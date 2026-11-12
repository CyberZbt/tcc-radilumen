"""Extrai e lista anotações/comentários de PDFs de correção (professora, revisores).

Uso:
  python scripts/extrair_anotacoes_pdf.py "C:\\caminho\\correcao.pdf"
  python scripts/extrair_anotacoes_pdf.py "C:\\caminho\\correcao.pdf" --salvar correcoes.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyMuPDF não instalado. Rode: pip install -r requirements.txt"
    ) from exc

TIPOS_UTEIS = {
    fitz.PDF_ANNOT_TEXT,
    fitz.PDF_ANNOT_FREE_TEXT,
    fitz.PDF_ANNOT_HIGHLIGHT,
    fitz.PDF_ANNOT_UNDERLINE,
    fitz.PDF_ANNOT_STRIKE_OUT,
    fitz.PDF_ANNOT_SQUIGGLY,
    fitz.PDF_ANNOT_INK,
    fitz.PDF_ANNOT_SQUARE,
    fitz.PDF_ANNOT_CIRCLE,
    fitz.PDF_ANNOT_LINE,
    fitz.PDF_ANNOT_POLYGON,
    fitz.PDF_ANNOT_POLY_LINE,
    fitz.PDF_ANNOT_STAMP,
    fitz.PDF_ANNOT_CARET,
    fitz.PDF_ANNOT_POPUP,
}


def tipo_anotacao(annot: fitz.Annot) -> str:
    try:
        return annot.type[1]
    except Exception:
        return str(annot.type)


def texto_anotacao(annot: fitz.Annot) -> str:
    partes: list[str] = []
    for chave in ("content", "subject", "title"):
        valor = annot.info.get(chave, "").strip()
        if valor:
            partes.append(valor)
    if not partes:
        try:
            texto = annot.get_text().strip()
            if texto:
                partes.append(texto)
        except Exception:
            pass
    return " | ".join(dict.fromkeys(partes))


def extrair(pdf_path: Path) -> list[dict]:
    doc = fitz.open(pdf_path)
    itens: list[dict] = []

    for pagina in doc:
        for annot in pagina.annots() or []:
            if annot.type[0] not in TIPOS_UTEIS:
                continue
            rect = annot.rect
            itens.append(
                {
                    "pagina": pagina.number + 1,
                    "tipo": tipo_anotacao(annot),
                    "texto": texto_anotacao(annot),
                    "x0": round(rect.x0, 1),
                    "y0": round(rect.y0, 1),
                    "x1": round(rect.x1, 1),
                    "y1": round(rect.y1, 1),
                }
            )

    doc.close()
    return itens


def formatar(itens: list[dict]) -> str:
    if not itens:
        return "Nenhuma anotação encontrada neste PDF."

    linhas = [f"Total de anotações: {len(itens)}", ""]
    for i, item in enumerate(itens, 1):
        linhas.append(f"--- Anotação {i} ---")
        linhas.append(f"Página: {item['pagina']}")
        linhas.append(f"Tipo: {item['tipo']}")
        linhas.append(
            f"Posição (pt): ({item['x0']}, {item['y0']}) -> ({item['x1']}, {item['y1']})"
        )
        linhas.append(f"Texto: {item['texto'] or '(sem texto)'}")
        linhas.append("")
    return "\n".join(linhas).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="Caminho do PDF com anotações")
    parser.add_argument(
        "--salvar",
        type=Path,
        default=None,
        help="Salva a listagem em arquivo .txt",
    )
    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"Arquivo não encontrado: {args.pdf}", file=sys.stderr)
        return 1

    itens = extrair(args.pdf)
    saida = formatar(itens)
    print(saida)

    if args.salvar:
        args.salvar.write_text(saida, encoding="utf-8")
        print(f"Salvo em: {args.salvar.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

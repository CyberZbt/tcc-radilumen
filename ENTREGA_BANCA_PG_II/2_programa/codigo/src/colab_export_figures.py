"""
Pipeline Colab: garante dataset local + exporta figuras para o Drive.

Uso no Colab (após montar o Drive e estar em CODIGO_DIR):
  python -m src.colab_export_figures

Ou:
  python -m src.colab_export_figures --n 6 --conf 0.25
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

from src.export_figures import main as export_figures_main
from src.utils import CODIGO_DIR


def ensure_dataset(codigo_dir: Path) -> Path:
    """Garante /content/fracture_only e retorna data_fracture.yaml."""
    local = Path("/content/fracture_only")
    data_yaml = local / "data_fracture.yaml"
    drive_tar = codigo_dir / "fracture_only_colab.tar"

    if not local.exists():
        if not drive_tar.is_file():
            raise FileNotFoundError(
                f"Dataset ausente e TAR não encontrado:\n  {drive_tar}\n"
                "Coloque fracture_only_colab.tar na pasta do codigo no Drive."
            )
        print(f"Copiando TAR: {drive_tar}")
        subprocess.run(
            ["cp", str(drive_tar), "/content/fracture_only_colab.tar"],
            check=True,
        )
        print("Extraindo TAR em /content ...")
        subprocess.run(
            ["tar", "-xf", "/content/fracture_only_colab.tar", "-C", "/content"],
            check=True,
        )
    else:
        print(f"Dataset já presente: {local}")

    if not data_yaml.is_file():
        raise FileNotFoundError(f"YAML não encontrado após extração: {data_yaml}")

    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    data["path"] = str(local)
    data_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    print(f"data.yaml OK -> path={local}")
    return data_yaml


def resolve_weights(codigo_dir: Path) -> Path:
    candidates = [
        codigo_dir / "runs/detect/proposed/weights/final_proposed.pt",
        codigo_dir / "runs/detect/proposed/weights/safe_checkpoints/safe_epoch_100.pt",
        codigo_dir / "weights/final_proposed.pt",
        codigo_dir / "runs/detect/mini_proposed_e30_n1600/weights/best.pt",
        codigo_dir / "runs/detect/proposed/weights/best.pt",
    ]
    for path in candidates:
        if path.is_file():
            print(f"Pesos: {path}")
            return path
    raise FileNotFoundError(
        "Nenhum peso encontrado. Esperado final_proposed.pt em "
        "runs/detect/proposed/weights/"
    )


def resolve_run_dir(weights: Path) -> Path:
    """Pasta da run para copiar PR_curve/results.png."""
    run_dir = weights.parent
    if run_dir.name == "weights":
        run_dir = run_dir.parent
    if run_dir.name == "safe_checkpoints":
        run_dir = run_dir.parent.parent  # .../proposed
    # também procurar em pastas irmãs de validação
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Colab: dataset + exportar figuras TCC")
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--out",
        default="",
        help="Destino (padrão: <CODIGO_DIR>/Figuras_export)",
    )
    args = parser.parse_args(argv)

    codigo_dir = CODIGO_DIR
    print(f"CODIGO_DIR = {codigo_dir}")

    data_yaml = ensure_dataset(codigo_dir)
    weights = resolve_weights(codigo_dir)
    run_dir = resolve_run_dir(weights)
    out = Path(args.out) if args.out else codigo_dir / "Figuras_export"
    out.mkdir(parents=True, exist_ok=True)

    # também varrer proposed e test_final_* por plots
    extra_roots = [
        run_dir,
        codigo_dir / "runs" / "detect" / "proposed",
        codigo_dir / "runs" / "detect" / "test_final_proposed",
        codigo_dir / "runs" / "val",
    ]
    # export_figures aceita um run-dir; copiamos plots extras depois se precisar
    rc = export_figures_main(
        [
            "--weights",
            str(weights),
            "--data",
            str(data_yaml),
            "--out",
            str(out),
            "--n",
            str(args.n),
            "--conf",
            str(args.conf),
            "--run-dir",
            str(run_dir),
        ]
    )

    # fallback: procurar plots em outras pastas de runs
    from src.export_figures import copy_if_found, find_plot

    for root in extra_roots:
        if not root.is_dir():
            continue
        copy_if_found(find_plot(root, ["PR_curve.png", "BoxPR_curve.png"]), out / "resultados_pr_curve.png")
        copy_if_found(
            find_plot(root, ["confusion_matrix.png", "confusion_matrix_normalized.png"]),
            out / "resultados_confusion.png",
        )
        copy_if_found(find_plot(root, ["results.png"]), out / "resultados_curves.png")

    print("Arquivos gerados:")
    for f in sorted(out.glob("*.png")):
        print(f" - {f.name} ({f.stat().st_size // 1024} KB)")
    print(f"Pasta: {out}")
    print("Baixe Figuras_export do Drive e copie para latex/Figuras/")
    return rc


if __name__ == "__main__":
    sys.exit(main())

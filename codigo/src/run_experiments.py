"""
Executa experimentos baseline vs. proposta (Cap. 5 do TCC).

Exemplo:
  python -m src.run_experiments --only proposed --epochs 1 --device cpu
  python -m src.run_experiments --epochs 100 --device 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.experiment import evaluate_model, save_metrics, train_model
from src.utils import CODIGO_DIR, load_yaml


CONFIGS = {
    "baseline": CODIGO_DIR / "configs" / "baseline.yaml",
    "proposed": CODIGO_DIR / "configs" / "proposed.yaml",
}


def ensure_dataset(cfg_path: Path) -> None:
    cfg = load_yaml(cfg_path)
    data_yaml = (CODIGO_DIR / cfg["data"]).resolve()
    train_dir = data_yaml.parent / "images" / "train"
    if not data_yaml.is_file():
        preprocess = "8bit" in str(data_yaml)
        prepare_command = "python -m src.prepare_dataset --classes fracture"
        if preprocess:
            prepare_command += " --preprocess"
        raise FileNotFoundError(
            f"Dataset não encontrado: {data_yaml}\n"
            f"Rode: {prepare_command}"
        )
    n_train = len(list(train_dir.glob("*.png"))) if train_dir.is_dir() else 0
    if n_train < 100:
        print(
            f"Aviso: apenas {n_train} imagens em {train_dir}. "
            "Para treino completo, rode prepare_dataset sem --max-images."
        )


def run_one(
    key: str,
    *,
    epochs: int | None,
    batch: int | None,
    device: str | None,
    skip_train: bool,
    skip_eval: bool,
) -> Path | None:
    cfg_path = CONFIGS[key]
    ensure_dataset(cfg_path)
    cfg = load_yaml(cfg_path)
    run_name = cfg.get("name", key)
    out_dir = CODIGO_DIR / "runs" / "experiments" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics_test.json"

    best = CODIGO_DIR / "runs" / "detect" / run_name / "weights" / "best.pt"

    if not skip_train:
        best = train_model(
            cfg_path,
            name=run_name,
            epochs=epochs,
            batch=batch,
            device=device,
        )
    elif not best.is_file():
        raise FileNotFoundError(f"Peso ausente: {best}. Remova --skip-train.")

    if not skip_eval:
        result = evaluate_model(best, cfg_path, split="test", device=device)
        save_metrics(result, metrics_path)
        print(f"[{key}] mAP50={result['map50']:.4f} | recall={result['recall']:.4f}")
        print(f"Métricas: {metrics_path}")

    return metrics_path if metrics_path.is_file() else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Experimentos baseline vs. proposta")
    parser.add_argument("--only", choices=["baseline", "proposed"], help="Executar um experimento")
    parser.add_argument("--epochs", type=int, help="Sobrescreve epochs do YAML")
    parser.add_argument("--batch", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args(argv)

    keys = [args.only] if args.only else ["baseline", "proposed"]
    for key in keys:
        print(f"\n=== Experimento: {key} ===")
        run_one(
            key,
            epochs=args.epochs,
            batch=args.batch,
            device=args.device,
            skip_train=args.skip_train,
            skip_eval=args.skip_eval,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

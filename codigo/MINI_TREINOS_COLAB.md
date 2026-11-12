# Treinos rápidos (~1600 imagens) — pedido do orientador

Objetivo: fazer **vários treinos**, sempre **guardando as métricas**, até chegar perto de **≥ 70%** (mAP50 / recall / precisão), sem esperar o dataset completo (~16k).

## O que foi preparado no PC

| Arquivo | Função |
|---------|--------|
| `src/make_subset.py` | Cria subset com 1600 treino + 200 val + 200 test |
| `src/run_mini_train.py` | Treina 1 perfil, avalia no test e **grava na tabela** |
| `src/log_metrics.py` | Acrescenta linha no CSV e gera LaTeX |
| `runs/comparisons/runs_log.csv` | Histórico de todos os treinos |
| `latex/resultados/tabela_comparacao_treinos.tex` | Tabela para o TCC |

## No Colab (GPU T4)

### Célula 1 — GPU + Drive + dataset + subset

```python
# =============================================================================
# CÉLULA 1 — GPU + Drive + dataset + subset 1600
# =============================================================================
import os, sys, subprocess, yaml
from pathlib import Path

print("=" * 60)
print("1) VERIFICANDO GPU")
print("=" * 60)
try:
    gpu_check = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
except FileNotFoundError:
    raise RuntimeError(
        "GPU OFF. Vá em: Ambiente de execução → Alterar tipo → GPU (T4) → Salvar\n"
        "Depois rode esta célula de novo."
    ) from None
if gpu_check.returncode != 0:
    raise RuntimeError("GPU OFF. Ative GPU (T4) no menu e rode de novo.")
print(gpu_check.stdout)

!pip install -q ultralytics pyyaml

import torch
assert torch.cuda.is_available(), "PyTorch sem CUDA — ative GPU e reinicie"
print(f"OK GPU: {torch.cuda.get_device_name(0)}")

print("=" * 60)
print("2) MONTANDO GOOGLE DRIVE")
print("=" * 60)
from google.colab import drive
drive.mount("/content/drive", force_remount=True)

drive_root = Path("/content/drive/MyDrive")
CODIGO_DIR = next(
    p for p in drive_root.iterdir()
    if (p / "src").is_dir() and (p / "configs").is_dir()
)

# Importante: !python NÃO herda bem pastas com espaço — usamos cwd= sempre
os.chdir(CODIGO_DIR)
sys.path.insert(0, str(CODIGO_DIR))
print(f"CODIGO_DIR = {CODIGO_DIR}")
print("Arquivos em src:")
for f in sorted((CODIGO_DIR / "src").glob("*.py")):
    print(" -", f.name)

required = ["make_subset.py", "run_mini_train.py", "log_metrics.py", "__init__.py"]
missing = [r for r in required if not (CODIGO_DIR / "src" / r).is_file()]
if missing:
    raise FileNotFoundError(
        f"Faltam no Drive em {CODIGO_DIR/'src'}: {missing}\n"
        "Envie de novo a pasta src completa do PC."
    )

print("=" * 60)
print("3) DATASET (TAR) + SUBSET 1600")
print("=" * 60)
DRIVE_TAR = CODIGO_DIR / "fracture_only_colab.tar"
LOCAL = Path("/content/fracture_only")
if not LOCAL.exists():
    if not DRIVE_TAR.is_file():
        raise FileNotFoundError(f"TAR não encontrado: {DRIVE_TAR}")
    print("Copiando TAR do Drive...")
    subprocess.run(["cp", str(DRIVE_TAR), "/content/fracture_only_colab.tar"], check=True)
    print("Extraindo...")
    subprocess.run(["tar", "-xf", "/content/fracture_only_colab.tar", "-C", "/content"], check=True)
else:
    print("Dataset já extraído em /content/fracture_only")

data = yaml.safe_load((LOCAL / "data_fracture.yaml").read_text())
data["path"] = str(LOCAL)
(LOCAL / "data_fracture.yaml").write_text(yaml.safe_dump(data, sort_keys=False))

print("Criando subset 1600...")
subprocess.run(
    [
        sys.executable, "-m", "src.make_subset",
        "--src", "/content/fracture_only",
        "--out", "/content/fracture_only_1600",
        "--train-images", "1600",
        "--val-images", "200",
        "--test-images", "200",
    ],
    cwd=str(CODIGO_DIR),
    check=True,
)

print("=" * 60)
print("PRONTO. Agora rode a Célula 2.")
print("=" * 60)
```

### Célula 2 — treino proposta (30 épocas)

```python
import os, sys, subprocess
from pathlib import Path

drive_root = Path("/content/drive/MyDrive")
CODIGO_DIR = next(p for p in drive_root.iterdir() if (p/"src").is_dir() and (p/"configs").is_dir())
os.chdir(CODIGO_DIR)
sys.path.insert(0, str(CODIGO_DIR))
print("CODIGO_DIR =", CODIGO_DIR)

subprocess.run(
    [
        sys.executable, "-m", "src.run_mini_train",
        "--profile", "proposed",
        "--src-dataset", "/content/fracture_only",
        "--subset-dir", "/content/fracture_only_1600",
        "--skip-subset",
        "--epochs", "30",
        "--batch", "16",
        "--device", "0",
        "--notes", "mini proposta 30 epocas",
    ],
    cwd=str(CODIGO_DIR),
    check=True,
)
print("Treino finalizado. Veja a célula 'Ler métricas' abaixo.")
```

### Célula — ler métricas salvas (use depois de cada treino)

```python
import csv, json
from pathlib import Path

drive_root = Path("/content/drive/MyDrive")
CODIGO_DIR = next(p for p in drive_root.iterdir() if (p/"src").is_dir() and (p/"configs").is_dir())
csv_path = CODIGO_DIR / "runs" / "comparisons" / "runs_log.csv"
print("CSV:", csv_path)
print("Existe?", csv_path.is_file())

if csv_path.is_file():
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    print(f"\nTotal de runs: {len(rows)}\n")
    for r in rows:
        print(
            f"{r['run']:30} | N={r['n_train']:5} | ep={r['epochs']:3} | "
            f"P={float(r['precision']):.4f} | R={float(r['recall']):.4f} | "
            f"mAP50={float(r['map50']):.4f} | mAP50-95={float(r['map50_95']):.4f}"
        )
    last = rows[-1]
    print("\n=== ÚLTIMO RUN ===")
    print("Nome:", last["run"])
    print("Precisão:", float(last["precision"]))
    print("Recall:  ", float(last["recall"]))
    print("mAP50:   ", float(last["map50"]))
    print("mAP50-95:", float(last["map50_95"]))
    meta = 0.70
    ok = any(float(last[k]) >= meta for k in ("precision", "recall", "map50"))
    print("Meta >= 70%:", "SIM" if ok else "ainda não")
else:
    # fallback: procurar metrics_test.json
    exps = sorted((CODIGO_DIR / "runs" / "experiments").glob("*/metrics_test.json"))
    print("JSONs encontrados:", len(exps))
    for p in exps:
        data = json.loads(p.read_text())
        print(p.parent.name, data)
```

### Célula 3 — treino baseline (30 épocas)

```python
import os, sys, subprocess
from pathlib import Path

drive_root = Path("/content/drive/MyDrive")
CODIGO_DIR = next(p for p in drive_root.iterdir() if (p/"src").is_dir() and (p/"configs").is_dir())
os.chdir(CODIGO_DIR)
sys.path.insert(0, str(CODIGO_DIR))

subprocess.run(
    [
        sys.executable, "-m", "src.run_mini_train",
        "--profile", "baseline",
        "--src-dataset", "/content/fracture_only",
        "--subset-dir", "/content/fracture_only_1600",
        "--skip-subset",
        "--epochs", "30",
        "--batch", "16",
        "--device", "0",
        "--notes", "mini baseline 30 epocas",
    ],
    cwd=str(CODIGO_DIR),
    check=True,
)
print("Baseline finalizado. Rode a célula 'Ler métricas'.")
```

### Célula 4 — mais tentativas (ex.: 50 épocas)

```python
import os, sys, subprocess
from pathlib import Path

drive_root = Path("/content/drive/MyDrive")
CODIGO_DIR = next(p for p in drive_root.iterdir() if (p/"src").is_dir() and (p/"configs").is_dir())
os.chdir(CODIGO_DIR)
sys.path.insert(0, str(CODIGO_DIR))

subprocess.run(
    [
        sys.executable, "-m", "src.run_mini_train",
        "--profile", "proposed",
        "--subset-dir", "/content/fracture_only_1600",
        "--skip-subset",
        "--epochs", "50",
        "--batch", "16",
        "--device", "0",
        "--name", "mini_prop_e50",
        "--notes", "tentativa 50 epocas",
    ],
    cwd=str(CODIGO_DIR),
    check=True,
)
```

Cada execução grava em: `CODIGO_DIR/runs/comparisons/runs_log.csv`

## Conferir se o `src` no Drive está completo

No Drive, em `TCC tcc__Copy codigo./src/`, devem existir pelo menos:

- `make_subset.py`
- `run_mini_train.py`
- `log_metrics.py`
- `__init__.py`
- `utils.py`, `preprocess.py`, etc.

Se a Célula 1 listar os `.py` e **faltar** `make_subset.py` / `run_mini_train.py`, apague a `src` do Drive e envie de novo a pasta `codigo/src` do PC.

## Depois no PC

```powershell
cd codigo
.\venv\Scripts\Activate.ps1
python -m src.log_metrics --rebuild-tex
cd ..
.\compilar.ps1
```

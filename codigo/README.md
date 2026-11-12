# Ambiente Python — TCC (YOLO / GRAZPEDWRI-DX)

## 1. Ativar o venv (se ainda não estiver)

```powershell
cd C:\Users\lzabot\Downloads\tcc__Copy\codigo
python -m venv venv
.\venv\Scripts\Activate.ps1
```

## 2. Instalar PyTorch (antes do requirements.txt)

### GPU NVIDIA (CUDA 12.4 — recomendado para treinar)

```powershell
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Se falhar, tente CUDA 12.1:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Apenas CPU (sem GPU)

```powershell
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## 3. Instalar o restante das dependências

```powershell
pip install -r requirements.txt
```

## 4. Registrar kernel Jupyter

```powershell
python -m ipykernel install --user --name=tcc-yolo --display-name="TCC YOLO (venv)"
```

## 5. Verificar instalação

```powershell
python -c "import torch; from ultralytics import YOLO; print('PyTorch', torch.__version__); print('CUDA disponivel:', torch.cuda.is_available()); print('Ultralytics OK')"
```

## 6. Dataset GRAZPEDWRI-DX

Imagens em `../archive/`. Detalhes em `dataset/README.md`.

```powershell
# Split YOLO (hardlink 16-bit — rápido)
python -m src.prepare_dataset

# Com pré-processamento 16-bit→8-bit + CLAHE e classe fracture
python -m src.prepare_dataset --classes fracture --preprocess

# Smoke test (10 imagens por split)
python -m src.prepare_dataset --classes fracture --preprocess --max-images 10
```

## 7. Treinar YOLO

Hiperparâmetros em `configs/default.yaml`. Experimentos do Cap. 5:

| Config | Perfil |
|--------|--------|
| `configs/baseline.yaml` | 16-bit, sem aug, from scratch |
| `configs/proposed.yaml` | 8-bit+CLAHE, transfer learning, aug |

```powershell
# Experimento completo (baseline + proposta)
python -m src.run_experiments --epochs 100 --device 0

# Só proposta, smoke test CPU
python -m src.run_experiments --only proposed --epochs 1 --batch 2 --device cpu

# Exportar tabela LaTeX após treino
python -m src.export_metrics
```

### Treino individual

```powershell
python -m src.train --config configs/proposed.yaml
```

Pesos iniciais: coloque `yolov8n.pt` em `weights/` ou deixe Ultralytics baixar.

## 8. Inferência e avaliação

```powershell
python -m src.predict --weights weights/final_proposed.pt --source ../archive/images_part1 --preprocess --name demo

python -m src.evaluate --weights weights/final_proposed.pt --split test
```

## 8.1 Interface Streamlit (demonstração)

```powershell
cd codigo
.\venv\Scripts\Activate.ps1
pip install streamlit
# Baixe final_proposed.pt do Drive para codigo/weights/
streamlit run app/streamlit_app.py
```

## 9. Google Colab (GPU)

Guia completo: **[COLAB.md](COLAB.md)** · Baseline: **[BASELINE_COLAB.md](BASELINE_COLAB.md)**

Notebook: `notebooks/colab_train.ipynb`

Resumo:

1. Copie `tcc__Copy` (ou `archive` + `codigo`) para o Google Drive
2. Abra `colab_train.ipynb` em [colab.research.google.com](https://colab.research.google.com)
3. Runtime → **GPU**
4. Ajuste `PROJECT_ROOT` na primeira célula
5. Execute as células em ordem
6. Para o baseline (comparação Cap. 5), use `BASELINE_COLAB.md` ou `EXPERIMENT = "baseline"` no notebook

## Estrutura

```
codigo/
  app/          # Streamlit (demonstração)
  configs/      # baseline.yaml, proposed.yaml
  dataset/      # yolo/, yolo_8bit/
  notebooks/    # Colab + EDA
  runs/         # artefatos de treino (gitignored)
  src/          # pipeline completo
  weights/      # final_proposed.pt
  requirements.txt
  venv/
```

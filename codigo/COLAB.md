# Google Colab — Treino YOLO (TCC UTP)

Guia rápido para treinar no GPU gratuito do Colab.

## 1. Preparar o Google Drive

Escolha **uma** opção:

### Opção A — Projeto inteiro (recomendado)

Copie `tcc__Copy` para o Drive:

```
Meu Drive/TCC/tcc__Copy/
  archive/          ← imagens + labels (obrigatório se vai preprocessar no Colab)
  codigo/           ← código Python
```

No Windows, pode arrastar a pasta no [drive.google.com](https://drive.google.com). O upload do `archive/` demora (vários GB).

### Opção B — Só dataset já processado (mais rápido)

Se você já rodou no PC:

```powershell
python -m src.prepare_dataset --preprocess
```

Copie para o Drive apenas:

```
Meu Drive/TCC/tcc__Copy/codigo/dataset/yolo_8bit/
```

E também a pasta `codigo/src/`, `codigo/configs/` e o notebook.

### Opção C — Zip no Drive

1. Compacte `archive` → `archive.zip`
2. Envie o zip ao Drive
3. No Colab, descompacte antes de `prepare_dataset`

## 2. Abrir o notebook no Colab

1. Envie `codigo/notebooks/colab_train.ipynb` ao Drive **ou**
2. No GitHub: File → Upload notebook
3. Abra o arquivo no [colab.research.google.com](https://colab.research.google.com)

## 3. Configurar GPU

**Runtime → Change runtime type → T4 GPU** (ou melhor se disponível).

## 4. Ajustar caminho

Na primeira célula de código, edite:

```python
PROJECT_ROOT = "/content/drive/MyDrive/TCC/tcc__Copy"
```

Use o caminho real da sua pasta no Drive.

## 5. Executar células em ordem

| Etapa | O que faz |
|-------|-----------|
| Montar Drive | `drive.mount` |
| Instalar pacotes | PyTorch CUDA + Ultralytics |
| `prepare_dataset` | Só se não tem `yolo_8bit` |
| `train` | Treino principal (~horas) |
| `evaluate` | mAP no test |
| Salvar no Drive | Copia `best.pt` |

## 6. Parâmetros úteis

| Problema | Solução |
|----------|---------|
| CUDA out of memory | `BATCH = 8` ou `IMGSZ = 512` |
| Sessão expirou | Pesos em `colab_outputs/` no Drive |
| Treino lento | Confirme GPU em Runtime settings |
| Só testar pipeline | `PREPARE_FULL = False` e `EPOCHS = 1` |

## 7. Trazer pesos para o PC

Baixe do Drive:

```
tcc__Copy/colab_outputs/grazpedwri_colab/best.pt
```

No PC:

```powershell
python -m src.predict --weights best.pt --source ../archive/images_part1 --preprocess
```

## Tamanho aproximado

| Item | Tamanho |
|------|---------|
| `archive/` (16-bit PNG) | ~15–25 GB |
| `dataset/yolo_8bit/` (8-bit) | ~8–15 GB |
| `best.pt` (yolov8n) | ~6 MB |

## Baseline (comparação Cap. 5)

Com a **proposta** já validada no teste, rode o baseline no Colab:

→ Guia completo: **[BASELINE_COLAB.md](BASELINE_COLAB.md)**

Resumo: `YOLO("yolov8n.yaml")` + `pretrained=False` + augmentation zerada + mesmo `fracture_only` 8-bit. Ao terminar, avalie no `split="test"` e envie os 4 números (P, R, mAP50, mAP50-95).

## Sem Drive (upload direto)

Não recomendado para o dataset completo. Use só smoke test com `--max-images 10`.

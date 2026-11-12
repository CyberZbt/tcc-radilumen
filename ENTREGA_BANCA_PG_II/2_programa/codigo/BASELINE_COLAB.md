# Baseline no Google Colab (comparação Cap. 5)

O modelo **proposta** já está validado no teste. Agora falta o **baseline**.

## Diferença científica

| Item | Baseline | Proposta |
|------|----------|----------|
| Pesos iniciais | do zero (`yolov8n.yaml`) | pré-treinados COCO |
| Augmentation | desligada | ligada |
| Pré-processamento | mesmo dataset 8-bit (prático no Colab) | CLAHE + 8-bit |
| Objetivo | referência bruta | técnica completa do TCC |

> No Colab usamos o mesmo `fracture_only` 8-bit por limite de espaço. A comparação principal fica em **transfer learning + augmentation**.

## Célula única (após GPU T4 + recuperação do Drive)

```python
from google.colab import drive
drive.mount("/content/drive", force_remount=True)

!pip install -q ultralytics pyyaml

import os
import yaml
import subprocess
from pathlib import Path
from ultralytics import YOLO
import torch

assert torch.cuda.is_available(), "Ative T4 GPU"

drive_root = Path("/content/drive/MyDrive")
CODIGO_DIR = next(
    p for p in drive_root.iterdir()
    if (p / "src").exists() and (p / "configs").exists()
)

DRIVE_TAR = CODIGO_DIR / "fracture_only_colab.tar"
LOCAL_TAR = Path("/content/fracture_only_colab.tar")
LOCAL_DATASET = Path("/content/fracture_only")

if not LOCAL_DATASET.exists():
    subprocess.run(["cp", str(DRIVE_TAR), str(LOCAL_TAR)], check=True)
    subprocess.run(["tar", "-xf", str(LOCAL_TAR), "-C", "/content"], check=True)

DATA_YAML = LOCAL_DATASET / "data_fracture.yaml"
data = yaml.safe_load(DATA_YAML.read_text())
data["path"] = str(LOCAL_DATASET)
DATA_YAML.write_text(yaml.safe_dump(data, sort_keys=False))

os.chdir(CODIGO_DIR)
RUN = CODIGO_DIR / "runs/detect/baseline"
RUN.mkdir(parents=True, exist_ok=True)

# Treino baseline: from scratch, sem augmentation
model = YOLO("yolov8n.yaml")  # arquitetura sem pesos COCO
model.train(
    data=str(DATA_YAML),
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    workers=2,
    patience=20,
    seed=42,
    optimizer="AdamW",
    pretrained=False,
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
    degrees=0.0,
    translate=0.0,
    scale=0.0,
    shear=0.0,
    flipud=0.0,
    fliplr=0.0,
    mosaic=0.0,
    mixup=0.0,
    project=str(CODIGO_DIR / "runs/detect"),
    name="baseline",
    exist_ok=True,
)
```

## Após o treino — avaliar no TEST

```python
best = CODIGO_DIR / "runs/detect/baseline/weights/best.pt"
model = YOLO(str(best))

metrics = model.val(
    data=str(DATA_YAML),
    split="test",
    imgsz=640,
    batch=16,
    device=0,
    conf=0.001,
    iou=0.7,
    plots=True,
    project=str(CODIGO_DIR / "runs/detect"),
    name="test_final_baseline",
    exist_ok=True,
)

print("Baseline TEST")
print("Precisão:", metrics.box.mp)
print("Recall:", metrics.box.mr)
print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)
```

Envie os 4 números. Atualizaremos a tabela do TCC e a comparação completa.

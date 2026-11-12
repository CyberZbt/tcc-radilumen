# Dataset GRAZPEDWRI-DX

As imagens originais ficam em `../archive/` (não duplicadas aqui por padrão).

## Preparar para YOLO

```powershell
cd C:\Users\lzabot\Downloads\tcc__Copy\codigo
.\venv\Scripts\Activate.ps1
python -m src.prepare_dataset --dry-run
python -m src.prepare_dataset --classes fracture
```

Isso cria `dataset/yolo/fracture_only/` com:

- `images/{train,val,test}/` — hardlinks às PNG em `archive/images_part*`
- `labels/{train,val,test}/` — classe única `fracture` (ID remapeado para 0)
- `data_fracture.yaml` — config Ultralytics

Split por **paciente** (`patient_id` no `dataset.csv`): ~80% / 10% / 10%.

### Com pré-processamento (8-bit + CLAHE) — recomendado

```powershell
python -m src.prepare_dataset --classes fracture --preprocess
```

Gera `dataset/yolo_8bit/fracture_only/` (PNG 8-bit + labels + `data_fracture.yaml`).

Parâmetros de CLAHE/normalização: `configs/default.yaml` → `preprocess:`.

### Smoke test (subset)

```powershell
python -m src.prepare_dataset --classes fracture --preprocess --max-images 10
```

### Se hardlink falhar no Windows (sem --preprocess)

```powershell
python -m src.prepare_dataset --copy-images
```

## Origem (`archive/`)

| Pasta | Conteúdo |
|-------|----------|
| `images_part1` … `images_part4` | 20.327 PNG 16-bit |
| `folder_structure/yolov5/labels/` | Labels YOLO (.txt) |
| `folder_structure/yolov5/meta.yaml` | 9 classes |
| `dataset.csv` | Metadados (paciente, idade, fratura, etc.) |

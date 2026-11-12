# Exportar figuras (Colab → Drive → PC)

## 1) Envie estes arquivos ao Drive (`src/`)

Do PC: `codigo/src/`

- `export_figures.py`
- `colab_export_figures.py`  ← **novo**

## 2) Célula Colab (substitua a antiga)

```python
from google.colab import drive
from pathlib import Path
import os, sys

!pip install -q ultralytics pyyaml opencv-python-headless

drive.mount("/content/drive", force_remount=True)
drive_root = Path("/content/drive/MyDrive")
CODIGO_DIR = next(p for p in drive_root.iterdir() if (p/"src").is_dir() and (p/"configs").is_dir())
os.chdir(CODIGO_DIR)
sys.path.insert(0, str(CODIGO_DIR))
print("CODIGO_DIR =", CODIGO_DIR)

# Extrai dataset (se precisar) + gera Figuras_export/
!python -m src.colab_export_figures --n 6 --conf 0.25
```

## 3) Baixar no PC (célula rápida no Colab)

```python
from google.colab import files
from pathlib import Path
import shutil

src = Path("/content/drive/MyDrive")
CODIGO_DIR = next(p for p in src.iterdir() if (p/"src").is_dir() and (p/"Figuras_export").is_dir())
export = CODIGO_DIR / "Figuras_export"
zip_path = Path("/content/Figuras_export")
shutil.make_archive(str(zip_path), "zip", export)
print("Baixando", zip_path.with_suffix(".zip"))
files.download(str(zip_path.with_suffix(".zip")))
```

## 4) No PC

1. Extraia o ZIP  
2. Copie estes 4 PNG para `latex/Figuras/`:
   - `resultados_deteccoes.png`
   - `resultados_pr_curve.png`
   - `resultados_confusion.png`
   - `resultados_curves.png`
3. Avise aqui — eu compilo o PDF

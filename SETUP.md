# Setup — TCC UTP (LaTeX + YOLO / GRAZPEDWRI-DX)

Guia para replicar o projeto em **outro PC** ou **outra conta do Cursor**.

> **Agente Cursor (IA):** leia também **`AGENTS.md`** na raiz. A regra `.cursor/rules/tcc-onboarding-setup.mdc` manda ler estes dois arquivos antes de qualquer setup ou execução.

---

## Para o agente Cursor

Se você é o assistente neste repositório:

1. Leia **`AGENTS.md`** (comandos de verificação e comportamento).
2. Leia o restante deste **`SETUP.md`** (instalação completa).
3. Confirme: `archive/`, `codigo/venv/`, `codigo/dataset/yolo_8bit/`, MiKTeX.
4. Execute verificações do `AGENTS.md` se o usuário pedir para rodar o projeto.
5. Respeite `.cursor/rules/` — especialmente `tcc-protecao-geral.mdc`.

**Prompt útil para o usuário colar no chat:**

```
Leia AGENTS.md e SETUP.md, verifique o que falta nesta máquina e me diga os passos para rodar o projeto (LaTeX + Python/YOLO).
```

---

## 1. Copiar o projeto

Copie a pasta **`tcc__Copy`** inteira. Estrutura esperada:

```
tcc__Copy/
├── AGENTS.md            ← agente Cursor: ler primeiro
├── SETUP.md             ← setup humano + checklist
├── .cursor/rules/       ← regras do Cursor (obrigatório para o agente)
├── .vscode/             ← LaTeX Workshop
├── archive/             ← dataset GRAZPEDWRI-DX (~15–25 GB)
├── codigo/              ← Python, YOLO, notebooks
├── latex/               ← monografia (trabalho.tex, PDF, bib)
├── compilar.ps1         ← atalho para compilar PDF
└── SETUP.md             ← este arquivo
```

> **Sem `archive/`** o código instala, mas treino e EDA não rodam.

---

## 2. Abrir no Cursor

1. **File → Open Folder** → selecione `tcc__Copy`
2. As regras em `.cursor/rules/` carregam automaticamente (inclui `tcc-onboarding-setup.mdc`)
3. No chat, cole: *Leia AGENTS.md e SETUP.md e verifique o que falta para rodar o projeto.*
4. Instale a extensão **LaTeX Workshop** (`james-yu.latex-workshop`)

### Regras do Cursor (`.cursor/rules/`)

| Arquivo | Função |
|---------|--------|
| `tcc-protecao-geral.mdc` | Regra global — escopo mínimo, índice |
| `tcc-compilacao-latex.mdc` | Compilar PDF, ciclo bibtex |
| `tcc-abnt-configuracao.mdc` | ABNT, `abnt-opcoes.bib`, o que não patchar |
| `tcc-citacoes-bibliografia.mdc` | `\cite`, `\citeonline`, `referencias.bib` |
| `tcc-resumo-palavras-chave.mdc` | Resumo 1,5, palavras-chave |
| `tcc-figuras-abnt.mdc` | `\legendaf`, `\fonte`, figuras ABNT |
| `tcc-equacoes-tabelas.mdc` | Equações, tabelas |
| `tcc-trabalhos-relacionados.mdc` | Cap. trabalhos relacionados |
| `tcc-novas-referencias.mdc` | Checklist novas referências |
| `tcc-codigo-yolo.mdc` | Python, YOLO, `codigo/` |

---

## 3. LaTeX (monografia)

### Software

- **MiKTeX** ou TeX Live
- `pdflatex` e `bibtex` no PATH
- Extensão **LaTeX Workshop** no Cursor/VS Code

### VS Code / Cursor (`.vscode/settings.json`)

- Root: `latex/trabalho.tex`
- Auto-build ao salvar
- Receita: `pdflatex → bibtex → pdflatex × 2`

### Configuração ABNT (não quebrar)

**`latex/trabalho.tex`:**

```latex
\OnehalfSpacing
\usepackage{packages/abnt-UTP}
\citeoption{abnt-emphasize=bf}
\citeoption{abnt-cite-style=AuthorYEAR}
...
\nocite{deeplearningbook2024,gil2022,...}   % 21 chaves explícitas
\bibliography{abnt-opcoes,referencias}      % abnt-opcoes PRIMEIRO
```

**`latex/abnt-opcoes.bib`:** `AuthorYEAR`, `abnt-etal-text="\textit{et al.}"`, etc.

**Proibido patchar:** `\bibciteIMPL`, `\authorcapstyle` com `\edef` — quebra hyperref.

### Compilar PDF

```powershell
cd tcc__Copy
.\compilar.ps1
```

Nunca só um `pdflatex` após mudar `.bib` ou citações.

**Validação:** PDF ~40 páginas, 21 refs em REFERÊNCIAS, sem `??` no texto.

---

## 4. Python (YOLO)

### Software

- **Python 3.10+**
- GPU NVIDIA opcional (sem GPU = treino só CPU, muito lento)

### Ambiente virtual

```powershell
cd tcc__Copy\codigo
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
```

**Com GPU NVIDIA (CUDA 12.4):**

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**Só CPU:**

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Demais dependências:**

```powershell
pip install -r requirements.txt
python -m ipykernel install --user --name=tcc-yolo --display-name="TCC YOLO (venv)"
```

**Verificar:**

```powershell
python -c "import torch; from ultralytics import YOLO; print('PyTorch', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

### Pacotes principais (`requirements.txt`)

- ultralytics, opencv-python, pydicom, albumentations
- numpy>=1.26.0 (sem teto superior — evita build no Windows)
- pandas, scikit-learn, matplotlib, jupyterlab, pyyaml

### Código (`codigo/src/`)

| Script | Função |
|--------|--------|
| `prepare_dataset.py` | Monta layout YOLO a partir de `archive/` |
| `preprocess.py` | 16-bit → 8-bit + CLAHE |
| `train.py` | Treino Ultralytics |
| `predict.py` | Inferência |
| `evaluate.py` | mAP no test/val |

### Hiperparâmetros (`codigo/configs/default.yaml`)

| Parâmetro | Valor |
|-----------|-------|
| model | yolov8n.pt |
| data | dataset/yolo_8bit/fracture_only/data_fracture.yaml |
| epochs | 100 |
| imgsz | 640 |
| batch | 16 |
| patience | 20 |
| conf / iou | 0.25 / 0.5 |
| preprocess | CLAHE + normalização percentil 1–99 |

---

## 5. Dataset GRAZPEDWRI-DX

### Origem (`archive/`)

```
archive/
  images_part1 … images_part4/              # 20.327 PNG 16-bit
  folder_structure/yolov5/labels/           # labels YOLO (.txt)
  folder_structure/yolov5/meta.yaml         # 9 classes
  dataset.csv                               # metadados
```

### Destino (gerado)

```
codigo/dataset/yolo/          # hardlink 16-bit (rápido, sem duplicar)
codigo/dataset/yolo_8bit/     # 8-bit + CLAHE (recomendado para treino)
  images/{train,val,test}/
  labels/{train,val,test}/
  data.yaml
```

**Split:** por `patient_id` — ~80% train / 10% val / 10% test (seed=42).

### Comandos

```powershell
cd codigo
.\venv\Scripts\Activate.ps1

# Ver contagens (sem criar arquivos)
python -m src.prepare_dataset --dry-run

# Layout YOLO com hardlink (rápido)
python -m src.prepare_dataset

# 8-bit + CLAHE (recomendado para treino)
python -m src.prepare_dataset --classes fracture --preprocess

# Apenas fraturas (classe única)
python -m src.prepare_dataset --classes fracture --preprocess

# Smoke test (10 imgs/split)
python -m src.prepare_dataset --classes fracture --preprocess --max-images 10
```

Detalhes: `codigo/dataset/README.md`

---

## 6. Treino, avaliação e inferência

```powershell
cd codigo
.\venv\Scripts\Activate.ps1

# Experimentos Cap. 5 (baseline + proposta)
python -m src.run_experiments --epochs 100 --device 0

# Smoke test
python -m src.run_experiments --only proposed --epochs 1 --batch 2 --device cpu

# Tabela LaTeX → latex/resultados/tabela_metricas.tex
python -m src.export_metrics

# Inferência
python -m src.predict --weights runs/detect/proposed/weights/best.pt --source ../archive/images_part1 --preprocess
```

Artefatos ignorados pelo git: `codigo/runs/`, `codigo/weights/`, `*.pt`, `venv/`

---

## 7. Google Colab (GPU)

Guia completo: **`codigo/COLAB.md`**

Notebook: **`codigo/notebooks/colab_train.ipynb`**

1. Copie `tcc__Copy` (ou `archive` + `codigo`) para Google Drive
2. Abra o notebook em [colab.research.google.com](https://colab.research.google.com)
3. **Runtime → GPU**
4. Ajuste `PROJECT_ROOT = "/content/drive/MyDrive/TCC/tcc__Copy"`
5. Execute células em ordem
6. Pesos salvos em `colab_outputs/` no Drive

---

## 8. Notebooks locais

| Arquivo | Uso |
|---------|-----|
| `codigo/notebooks/01_eda_dataset.ipynb` | EDA do dataset |
| `codigo/notebooks/colab_train.ipynb` | Pipeline Colab |

Kernel Jupyter: **TCC YOLO (venv)**

---

## 9. Checklist — nova máquina / conta Cursor

```
[ ] Copiar tcc__Copy (inclui .cursor/, .vscode/, archive/, codigo/, latex/)
[ ] Abrir pasta no Cursor (Open Folder)
[ ] Instalar extensão LaTeX Workshop
[ ] Instalar MiKTeX — pdflatex e bibtex no PATH
[ ] Testar LaTeX: .\compilar.ps1 → PDF ~40 pág., 21 refs, sem ??
[ ] Instalar Python 3.10+
[ ] cd codigo → venv → pip torch + requirements.txt
[ ] Testar: python -c "from ultralytics import YOLO; print('OK')"
[ ] Dataset: archive/ presente OU yolo_8bit/ já gerado
[ ] Se necessário: python -m src.prepare_dataset --classes fracture --preprocess
[ ] (Opcional) Registrar kernel: ipykernel install --name=tcc-yolo
```

---

## 10. Estado validado (referência)

| Item | Status |
|------|--------|
| PDF LaTeX | ~40 páginas, 21 referências |
| Citações ABNT | AuthorYEAR, `et al.` em itálico |
| Resumo | Espaçamento 1,5 |
| Dataset | 20.327 imagens = labels = CSV |
| Classes YOLO | 9 (ou 1 com `--classes fracture`) |
| Smoke train CPU | `runs/detect/smoke/weights/best.pt` |

---

## 11. Problemas comuns

| Problema | Solução |
|----------|---------|
| Citações `??` no PDF | Rodar `.\compilar.ps1` (ciclo completo) |
| `numpy` não instala no Windows | Usar `numpy>=1.26.0` sem teto (já no requirements) |
| Hardlink falha | `python -m src.prepare_dataset --copy-images` |
| CUDA out of memory (Colab) | `BATCH = 8` ou `IMGSZ = 512` |
| Treino lento no PC | Usar Colab (`colab_train.ipynb`) |
| Regras Cursor não aparecem | Confirmar `.cursor/rules/` na raiz do projeto aberto |

---

## 12. Documentação adicional

| Arquivo | Conteúdo |
|---------|----------|
| `AGENTS.md` | Instruções para o agente Cursor |
| `SETUP.md` | Setup humano + checklist |
| `codigo/README.md` | Ambiente Python, comandos |
| `codigo/COLAB.md` | Google Colab passo a passo |
| `codigo/dataset/README.md` | Dataset e splits |
| `latex/compilar.ps1` | Script de compilação LaTeX |

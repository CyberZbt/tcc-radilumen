# Instruções para o agente Cursor — TCC UTP

**Leia este arquivo e `SETUP.md` na primeira interação** neste repositório (nova máquina, nova conta ou usuário pedindo “rodar o projeto”).

## O que é este projeto

Monografia LaTeX (UTP) + pipeline Python YOLO para detecção de fraturas no dataset **GRAZPEDWRI-DX**.

```
tcc__Copy/
  SETUP.md          ← guia completo de instalação e comandos (LEIA PRIMEIRO)
  AGENTS.md         ← este arquivo (onboarding do agente)
  .cursor/rules/    ← regras por tema (LaTeX ABNT, YOLO, etc.)
  latex/            ← trabalho.tex, PDF, bibliografia
  codigo/           ← Python, venv, treino YOLO
  archive/          ← imagens e labels do dataset (obrigatório para ML)
  compilar.ps1      ← compila o PDF
```

## Ao iniciar neste workspace

1. **Ler** `SETUP.md` (checklist, caminhos, comandos).
2. **Confirmar** que a pasta raiz aberta é `tcc__Copy` (contém `latex/`, `codigo/`, `.cursor/`).
3. **Verificar** presença de `archive/` e, se existir, `codigo/dataset/yolo_8bit/`.
4. **Consultar** `.cursor/rules/tcc-protecao-geral.mdc` e regras específicas antes de editar LaTeX ou código.
5. **Executar** os comandos de verificação abaixo (não assumir que venv/GPU já existem).

## Comandos de verificação (Windows / PowerShell)

### LaTeX

```powershell
cd <RAIZ_DO_PROJETO>
.\compilar.ps1
```

Esperado: `OK: citacoes resolvidas.` e `latex/trabalho.pdf`.

### Python

```powershell
cd <RAIZ_DO_PROJETO>\codigo
.\venv\Scripts\Activate.ps1
python -c "import torch; from ultralytics import YOLO; print('CUDA:', torch.cuda.is_available())"
```

Se `venv` não existir:

```powershell
cd codigo
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Dataset

```powershell
cd codigo
.\venv\Scripts\Activate.ps1
python -m src.prepare_dataset --dry-run
```

Se `archive/` existir e `yolo_8bit/fracture_only/` não:

```powershell
python -m src.prepare_dataset --classes fracture --preprocess
```

Smoke test treino (CPU):

```powershell
python -m src.prepare_dataset --classes fracture --preprocess --max-images 10
python -m src.train --epochs 1 --batch 2 --device cpu --name smoke --data dataset/yolo_8bit/fracture_only/data_fracture.yaml --workers 0
```

## Regras de comportamento do agente

- **LaTeX:** compilar sempre com `.\compilar.ps1` após mudar `.bib` ou citações. Não patchar `\bibciteIMPL` / `\authorcapstyle`.
- **Código:** ficar em `codigo/`; não misturar Python em `latex/`.
- **Escopo:** alterar só o que o usuário pediu; não refatorar `trabalho.tex` ou `referencias.bib` “por limpeza”.
- **Dataset:** origem em `archive/`; destino YOLO em `codigo/dataset/yolo_8bit/` (treino recomendado).
- **GPU:** se `torch.cuda.is_available()` for `False`, avisar que treino full é lento; sugerir Colab (`codigo/notebooks/colab_train.ipynb`, `codigo/COLAB.md`).
- **Commits:** só criar commit se o usuário pedir explicitamente.

## Documentação por tarefa

| Tarefa | Arquivo |
|--------|---------|
| Setup geral | `SETUP.md` |
| Python / YOLO | `codigo/README.md`, `.cursor/rules/tcc-codigo-yolo.mdc` |
| Colab GPU | `codigo/COLAB.md`, `codigo/notebooks/colab_train.ipynb` |
| Dataset | `codigo/dataset/README.md` |
| LaTeX / ABNT | `.cursor/rules/tcc-compilacao-latex.mdc`, `tcc-abnt-configuracao.mdc` |
| Hiperparâmetros | `codigo/configs/default.yaml` |

## Mensagem sugerida para o usuário (primeira sessão)

Se o usuário abrir o projeto sem contexto, responda com:

1. Confirmação de que leu `SETUP.md` / `AGENTS.md`
2. O que foi encontrado (`archive/`, `venv/`, GPU, `yolo_8bit/`)
3. Próximo passo concreto (compilar PDF, criar venv, preparar dataset ou treinar)

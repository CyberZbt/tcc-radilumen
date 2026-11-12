# RADILÚMEN — TCC UTP

Detecção e localização de fraturas de punho em radiografias com **YOLOv8n** (Ultralytics), no dataset público [GRAZPEDWRI-DX](https://doi.org/10.1038/s41597-022-01328-z).

Trabalho de Conclusão de Curso de **Lucas Rodrigues Zabot**, orientado pelo Prof. Sérgio Marques. Universidade Tuiuti do Paraná, 2026.

**Isto é apoio à decisão. Não substitui avaliação médica.**

## Demo no navegador

https://viewdicom-tcc.web.app

## O que enviar à banca

Pasta pronta (PDF + código + pesos + apresentação):

[`ENTREGA_BANCA_PG_II/`](ENTREGA_BANCA_PG_II/LEIA-ME.txt)

- Monografia: `ENTREGA_BANCA_PG_II/1_monografia/trabalho.pdf`
- Como rodar o programa: `ENTREGA_BANCA_PG_II/2_programa/COMO_RODAR.txt`
- Apresentação: `ENTREGA_BANCA_PG_II/3_apresentacao/`

## Resultado oficial (teste, 2.026 imagens)

| Métrica | Valor |
|---------|-------|
| mAP@0,5 | 92,94% |
| Precisão | 92,21% |
| Sensibilidade | 86,37% |

No subset exploratório (1.600 imagens): baseline 58,02% → proposta 84,43% de mAP@0,5.

## Rodar localmente

```powershell
cd ENTREGA_BANCA_PG_II\2_programa\codigo
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Envie `exemplos/punho_fratura.jpg` na interface.

O dataset original (`archive/`) **não** está neste repositório (~20 GB). Os pesos treinados estão em `ENTREGA_BANCA_PG_II/2_programa/codigo/weights/final_proposed.pt`.

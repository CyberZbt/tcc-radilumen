# Demonstrações do software para o TCC

Cada caso processado no Streamlit (botão **Salvar pacote completo**) cria uma pasta:

```
caso_01/
  original.png       ← imagem original
  deteccao.png       ← com bounding box
  coordenadas.csv    ← tabela x1,y1,x2,y2,confiança
  painel_tcc.png     ← painel pronto para LaTeX (original | detecção | tabela)
  readme.txt
```

## Como gerar
1. Abra o Streamlit
2. Envie uma radiografia
3. Clique em **Salvar pacote completo na pasta do TCC**
4. Avise o agente para incluir `painel_tcc.png` no `trabalho.tex`

## Coordenadas
Não há eixo Z. A caixa 2D usa:
- `x1, y1` — canto superior esquerdo
- `x2, y2` — canto inferior direito
(em pixels da imagem)

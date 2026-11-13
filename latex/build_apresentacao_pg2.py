# Apresentação RADILÚMEN — layout livre, clínico, sem modelo UTP e sem caixas.
from __future__ import annotations

import shutil
from pathlib import Path

from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt, Emu

ROOT = Path(r"d:\tcc__Copy")
FIG = ROOT / "latex" / "Figuras"
PUNHO = ROOT / "web-detector" / "public" / "exemplos" / "punho_fratura.jpg"
LOGO = FIG / "logo_sentinela.jpg"
OUT_LATEX = ROOT / "latex" / "Apresentacao_PG_II_Lucas_Zabot.pptx"
OUT_DL = Path(r"C:\Users\Lucas R Zabot\Downloads\Apresentacao_PG_II_Lucas_Zabot.pptx")
OUT_DL2 = Path(r"C:\Users\Lucas R Zabot\Downloads\Apresentacao_PG_II_visual.pptx")

SW, SH = 13.333, 7.5
P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"

NIGHT = RGBColor(7, 12, 20)
NAVY = RGBColor(10, 22, 36)
CYAN = RGBColor(86, 212, 197)
WHITE = RGBColor(244, 248, 251)
MUTED = RGBColor(156, 176, 194)
SOFT = RGBColor(210, 222, 232)
CORAL = RGBColor(255, 138, 122)
MINT = RGBColor(167, 243, 208)


def _run(run, size, bold=False, color=WHITE, italic=False):
    run.font.name = "Calibri"
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    rPr = run._r.get_or_add_rPr()
    for tag in ("latin", "cs", "ea"):
        el = rPr.find(qn(f"a:{tag}"))
        if el is None:
            el = etree.SubElement(rPr, qn(f"a:{tag}"))
        el.set("typeface", "Calibri")


def txt(slide, l, t, w, h, text, size=18, bold=False, color=WHITE,
        align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, italic=False):
    box = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    try:
        tf._txBody.bodyPr.set(
            "anchor",
            {MSO_ANCHOR.TOP: "t", MSO_ANCHOR.MIDDLE: "ctr", MSO_ANCHOR.BOTTOM: "b"}[anchor],
        )
    except Exception:
        pass
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    _run(run, size, bold, color, italic)
    return box


def solid(slide, l, t, w, h, color):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(l), Inches(t), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = color
    sh.line.fill.background()
    return sh


def overlay(slide, l, t, w, h, color, alpha=70000):
    """alpha: 0–100000 (100000 = opaco)."""
    sh = solid(slide, l, t, w, h, color)
    srgb = sh._element.find(f".//{{{A_NS}}}srgbClr")
    if srgb is not None:
        al = etree.SubElement(srgb, f"{{{A_NS}}}alpha")
        al.set("val", str(alpha))
    return sh


def night(slide):
    solid(slide, 0, 0, SW, SH, NIGHT)


def fade(slide):
    el = slide._element
    for old in el.findall(f"{{{P_NS}}}transition"):
        el.remove(old)
    tr = etree.Element(f"{{{P_NS}}}transition")
    tr.set("spd", "slow")
    etree.SubElement(tr, f"{{{P_NS}}}fade")
    el.append(tr)


def fill_photo(slide, path):
    im = Image.open(path)
    iw, ih = im.size
    img_aspect = iw / ih
    slide_aspect = SW / SH
    if img_aspect > slide_aspect:
        h = SH
        w = h * img_aspect
        left, top = (SW - w) / 2, 0
    else:
        w = SW
        h = w / img_aspect
        left, top = 0, (SH - h) / 2
    slide.shapes.add_picture(str(path), Inches(left), Inches(top), Inches(w), Inches(h))


def photo_right(slide, path, width=5.6):
    im = Image.open(path)
    aspect = im.width / max(im.height, 1)
    h = SH
    w = max(width, h * aspect)
    left = SW - w
    slide.shapes.add_picture(str(path), Inches(left), Inches(0), Inches(w), Inches(h))


def photo_fit(slide, path, l, t, h, max_w=None):
    im = Image.open(path)
    aspect = im.width / max(im.height, 1)
    w = h * aspect
    if max_w and w > max_w:
        w = max_w
        h = w / aspect
    slide.shapes.add_picture(str(path), Inches(l), Inches(t), Inches(w), Inches(h))
    return w


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def left_crop(src, dest, frac=0.42):
    dest.parent.mkdir(parents=True, exist_ok=True)
    im = Image.open(src).convert("RGB")
    w, h = im.size
    im.crop((0, 0, int(w * frac), h)).save(dest, quality=92)
    return dest


def kicker(slide, text):
    txt(slide, 0.7, 0.38, 12, 0.28, text.upper(), 11, True, CYAN)


def title(slide, text, y=0.68):
    txt(slide, 0.7, y, 12, 0.55, text, 32, True, WHITE)


def build():
    prs = Presentation()
    prs.slide_width = Inches(SW)
    prs.slide_height = Inches(SH)
    blank = prs.slide_layouts[6]

    # 1 Capa — título em cima, logo no centro, nome embaixo, raio-X de fundo
    s = prs.slides.add_slide(blank)
    fill_photo(s, PUNHO)
    overlay(s, 0, 0, SW, SH, NIGHT, 60000)
    txt(s, 0.8, 0.42, 11.7, 1.15,
        "Aplicação de Redes Neurais Convolucionais para Auxílio ao Diagnóstico e Localização de Fraturas Ósseas em Imagens Médicas DICOM",
        16, True, SOFT, PP_ALIGN.CENTER)
    logo = 3.05
    s.shapes.add_picture(str(LOGO), Inches((SW - logo) / 2), Inches(1.85), Inches(logo), Inches(logo))
    txt(s, 0.5, 5.05, 12.3, 0.55, "RADILÚMEN", 36, True, CYAN, PP_ALIGN.CENTER)
    txt(s, 0.5, 5.58, 12.3, 0.35, "A IA ilumina a fratura. O médico decide.", 16, False, MUTED, PP_ALIGN.CENTER, italic=True)
    txt(s, 0.5, 6.85, 12.3, 0.35, "Modo apresentador (Alt+F5): o roteiro de cada slide está nas notas.", 12, False, MUTED, PP_ALIGN.CENTER)
    fade(s)
    notes(s, """O QUE FALAR
Bom dia. Meu nome é Lucas Rodrigues Zabot. Este é o TCC de Ciência da Computação, orientado pelo professor Sérgio Marques.
O título oficial está no topo. O sistema que construí se chama RADILÚMEN: a inteligência artificial ilumina o ponto da fratura na radiografia, e o médico decide.
Não é um diagnóstico automático. É uma segunda opinião visual.
A imagem de fundo é um raio-X de punho — exatamente o tipo de exame que o modelo vê.""")

    # 2 Problema + imagem médica
    s = prs.slides.add_slide(blank)
    night(s)
    photo_right(s, PUNHO, 6.1)
    overlay(s, 6.4, 0, 7, SH, NIGHT, 35000)
    kicker(s, "O cenário")
    title(s, "O problema")
    rows = [
        ("01", "Plantão cheio", "Muita radiografia, pouco tempo."),
        ("02", "Fratura sutil", "O olho cansa e a lesão passa."),
        ("03", "Falta um alerta", "O médico precisa ver o ponto — não de um diagnóstico automático."),
    ]
    for i, (n, h, b) in enumerate(rows):
        y = 1.55 + i * 1.55
        txt(s, 0.7, y, 1.1, 0.4, n, 20, True, CYAN)
        txt(s, 1.9, y, 5.2, 0.4, h, 22, True, WHITE)
        txt(s, 1.9, y + 0.42, 5.2, 0.7, b, 16, False, MUTED)
    fade(s)
    notes(s, """O QUE FALAR
O problema não é falta de raio-X. É o contrário: o volume no plantão cresceu e o profissional lê muitos exames, cansado, com pouco tempo.
Fratura de punho, principalmente a sutil, é fácil de passar. Isso gera alta indevida e custo depois.
O que faltava era um alerta visual: uma caixa dizendo “olhe aqui”. Não um robô fechando o caso.
Quem usaria: médico de urgência e radiologista. O hospital inteiro, o PACS, fica fora deste TCC.""")

    # 3 O que entrega
    s = prs.slides.add_slide(blank)
    night(s)
    kicker(s, "Proposta")
    title(s, "O que o RADILÚMEN faz")
    items = [("Detecta", "há fratura no exame?"), ("Localiza", "caixa em cima do osso"), ("Apoia", "o médico decide")]
    for i, (h, b) in enumerate(items):
        x = 0.8 + i * 4.1
        txt(s, x, 2.0, 3.7, 0.85, h, 36, True, CYAN)
        txt(s, x, 2.9, 3.7, 0.9, b, 18, False, MUTED)
    txt(s, 0.8, 4.5, 11.5, 1.5,
        "Segunda opinião visual para quem lê raio-X de punho.\nNão é diagnóstico. Não entra no PACS. Não prova que o plantão erra menos.",
        18, False, SOFT)
    fade(s)
    notes(s, """O QUE FALAR
O RADILÚMEN faz três coisas, e só três.
Detecta: tem fratura nesta imagem?
Localiza: desenha a caixa em cima do osso, com uma confiança.
Apoia: o médico confirma ou descarta. Eu não substituo o profissional.
Se a banca perguntar “então é um diagnóstico?”: não. É filtro de segurança contra fadiga. Segunda opinião.""")

    # 4 Objetivos
    s = prs.slides.add_slide(blank)
    night(s)
    kicker(s, "Caminho")
    title(s, "Objetivos")
    txt(s, 0.7, 1.4, 12, 0.7, "Encontrar a fratura no punho e mostrar a caixa na imagem.", 20, False, SOFT)
    objs = [
        ("01  Treinar", "YOLO na base pública, em PNG, sem nome de paciente."),
        ("02  Medir", "Precisão, sensibilidade e mAP no teste por paciente."),
        ("03  Mostrar", "Software: envia a imagem, vê a caixa, lê a confiança."),
    ]
    for i, (h, b) in enumerate(objs):
        y = 2.3 + i * 1.05
        txt(s, 0.7, y, 12, 0.38, h, 20, True, CYAN)
        txt(s, 0.7, y + 0.38, 12, 0.5, b, 16, False, MUTED)
    fade(s)
    notes(s, """O QUE FALAR
Objetivo geral: achar a fratura no punho e mostrar a caixa.
Específicos, na ordem em que fiz:
Um — treinar a YOLO numa base pública, em PNG, sem nome de paciente.
Dois — medir precisão, sensibilidade e mAP no teste. O teste é separado por paciente, para não vazar a mesma pessoa.
Três — um software em que se envia a imagem e se vê o resultado.
No próximo slide eu separo o que está pronto do que é trabalho futuro.""")

    # 5 Feito / ainda não — tipografia, sem caixa
    s = prs.slides.add_slide(blank)
    night(s)
    kicker(s, "Escopo real")
    title(s, "O que foi testado")
    txt(s, 0.7, 1.7, 5.8, 0.4, "Feito", 14, True, MINT)
    txt(s, 0.7, 2.15, 5.8, 1.8, "Treino e teste em PNG.\nSe chegar DICOM, o programa converte e apaga o cabeçalho.", 20, False, WHITE)
    txt(s, 7.1, 1.7, 5.5, 0.4, "Ainda não", 14, True, CORAL)
    txt(s, 7.1, 2.15, 5.5, 1.8, "Gravar a saída de volta em DICOM.\nIntegrar ao sistema do hospital.", 20, False, WHITE)
    txt(s, 0.7, 4.6, 12, 1.2, "O título fala DICOM porque o exame nasce assim.\nO pipeline comprovado opera em PNG — só a imagem, sem dados pessoais.", 16, False, MUTED)
    fade(s)
    notes(s, """O QUE FALAR
A banca vai olhar o título e perguntar: “onde está o DICOM?”
Resposta honesta:
O exame no hospital nasce em DICOM. Por isso o título.
O que eu treinei e medi foi PNG de 8 bits, já sem nome de paciente.
Se alguém mandar um DICOM no programa, ele converte, descarta o cabeçalho e trabalha só na imagem.
O que ainda não fiz: gravar a caixa de volta no arquivo DICOM, nem ligar no PACS do hospital. Isso é trabalho futuro, não falha do modelo.""")

    # 6 Relacionados
    s = prs.slides.add_slide(blank)
    night(s)
    kicker(s, "Contexto")
    title(s, "O que já existia")
    cols = [
        ("Classificar", "A rede diz sim ou não.\nNão aponta o osso."),
        ("Segmentar", "Contorno pixel a pixel.\nMais detalhe, mais peso."),
        ("RADILÚMEN", "Caixa no punho, rápida.\nTela que o médico vê."),
    ]
    for i, (h, b) in enumerate(cols):
        x = 0.7 + i * 4.15
        txt(s, x, 1.7, 3.8, 0.55, h, 24, True, CYAN if i == 2 else WHITE)
        txt(s, x, 2.4, 3.8, 1.6, b, 18, False, MUTED)
    txt(s, 0.7, 4.6, 12, 1.2, "Aqui a tarefa é localizar a fratura e guiar o olhar — não classificar o exame inteiro nem desenhar cada pixel.", 18, False, SOFT)
    fade(s)
    notes(s, """O QUE FALAR
Não inventei visão computacional em raio-X. Já existiam três famílias:
Classificar: a rede diz “tem fratura” ou “não tem”. Não aponta onde. O médico ainda precisa varrer o filme inteiro.
Segmentar: desenha o contorno pixel a pixel. Mais preciso, mais pesado, mais lento para uma triagem.
Detectar com caixa — o que o RADILÚMEN faz: uma caixa no punho, rápida, na tela.
A minha contribuição não é a YOLO em si. É treinar, medir e entregar essa caixa num fluxo que o médico consegue ver.""")

    # 7 Como a rede vê — imagem médica grande
    s = prs.slides.add_slide(blank)
    night(s)
    kicker(s, "Inteligência")
    title(s, "Como a rede encontra a fratura")
    photo_fit(s, FIG / "yolo_grade_deteccao.png", 0.45, 1.55, 5.2, max_w=6.5)
    txt(s, 7.4, 1.8, 5.3, 0.4, "CNN", 14, True, CYAN)
    txt(s, 7.4, 2.2, 5.3, 1.3, "Aprende sozinha: bordas, depois o osso, depois o intervalo da fratura.", 18, False, SOFT)
    txt(s, 7.4, 3.7, 5.3, 0.4, "YOLO — uma passada", 14, True, CYAN)
    txt(s, 7.4, 4.15, 5.3, 1.6, "Olha a imagem inteira uma vez e devolve a caixa com a confiança. Versão leve, para a triagem ser rápida.", 18, False, SOFT)
    fade(s)
    notes(s, """O QUE FALAR
Como a rede “enxerga”:
CNN: camadas. Primeiro bordas, depois o formato do osso, depois o intervalo da fratura — o modelo aprende isso sozinho, eu não desenhei a regra.
YOLO: You Only Look Once. Uma passada na imagem inteira, devolve a caixa e um número de confiança.
Usei a versão leve, YOLOv8n, porque a triagem precisa ser rápida.
A figura à esquerda mostra a grade: a rede varre a imagem e escolhe a célula onde está a fratura.""")

    # 8 Pipeline
    s = prs.slides.add_slide(blank)
    night(s)
    photo_right(s, PUNHO, 5.4)
    overlay(s, 0, 0, 9.2, SH, NIGHT, 82000)
    kicker(s, "Método")
    title(s, "Fluxo automático")
    steps = [("DICOM", "exame"), ("PNG", "sem nome"), ("YOLO", "caixa"), ("Tela", "médico")]
    for i, (h, b) in enumerate(steps):
        x = 0.7 + i * 1.9
        txt(s, x, 1.7, 1.7, 0.5, h, 22, True, CYAN)
        txt(s, x, 2.2, 1.7, 0.5, b, 14, False, MUTED)
        if i < 3:
            txt(s, x + 1.45, 1.72, 0.4, 0.4, "→", 18, False, MUTED)
    mets = [("20 mil", "imagens"), ("80/10/10", "por paciente"), ("2 fases", "1.600 e 16 mil"), ("GPU", "Colab")]
    for i, (h, b) in enumerate(mets):
        y = 3.2 + i * 0.75
        txt(s, 0.7, y, 2.6, 0.45, h, 22, True, WHITE)
        txt(s, 3.4, y, 4.5, 0.45, b, 18, False, MUTED)
    fade(s)
    notes(s, """O QUE FALAR
O fluxo, da esquerda para a direita:
O exame chega em DICOM. Vira PNG. O nome do paciente some. A YOLO desenha a caixa. O médico vê na tela.
Números para memorizar:
Cerca de 20 mil imagens, base GRAZPEDWRI-DX, punho, de Graz.
Corte 80 treino, 10 validação, 10 teste — por paciente, não por arquivo. Assim a mesma criança não aparece no treino e no teste.
Duas fases: primeiro um recorte de cerca de 1.600 imagens para acertar o método. Depois a base completa, cerca de 16 mil no treino.
A GPU foi no Google Colab. No meu PC não dava para treinar a versão grande.""")

    # 9 Duas receitas
    s = prs.slides.add_slide(blank)
    night(s)
    kicker(s, "Experimento")
    title(s, "Duas receitas")
    txt(s, 0.7, 1.6, 5.8, 0.4, "Baseline", 14, True, CORAL)
    txt(s, 0.7, 2.05, 5.8, 2.2, "A mesma YOLO, do zero.\nSem pesos prontos, sem aumentar os dados.\n\nSó para perguntar: e se não fizéssemos nada especial?", 18, False, SOFT)
    txt(s, 7.1, 1.6, 5.6, 0.4, "RADILÚMEN", 14, True, CYAN)
    txt(s, 7.1, 2.05, 5.6, 2.2, "Contraste (CLAHE), rede já treinada, giros e zoom.\n\nPython, YOLO, Colab.\nEntrada PNG — DICOM vira PNG e o nome some.", 18, False, SOFT)
    fade(s)
    notes(s, """O QUE FALAR
Duas receitas, mesma rede, mesmo conjunto, para ser justo.
Baseline: YOLO do zero. Sem pesos prontos, sem aumentar os dados. É a pergunta “e se eu não fizesse nada especial?”
RADILÚMEN: três escolhas.
Um — CLAHE, contraste local, porque fratura sutil some no cinza.
Dois — transfer learning: a rede já vinha treinada em fotos do mundo, eu só especializei no osso.
Três — augmentação: giro, zoom, para não decorar a pose.
Ferramentas: Python, Ultralytics YOLO, Colab.
Se perguntarem “por que não ResNet classificando?”: porque eu precisava da caixa, não só do sim/não.""")
    s = prs.slides.add_slide(blank)
    night(s)
    fill_photo(s, FIG / "demo_software_pipeline.png")
    overlay(s, 0, 0, SW, SH, NIGHT, 42000)
    overlay(s, 0, 0, SW, 1.55, NIGHT, 80000)
    overlay(s, 0, 5.85, SW, 1.65, NIGHT, 85000)
    kicker(s, "Protótipo")
    title(s, "O programa em funcionamento")
    txt(s, 0.7, 6.05, 12, 0.85, "Original  ·  imagem limpa  ·  caixa na fratura\nNão substitui o médico    ·    viewdicom-tcc.web.app", 16, False, SOFT)
    fade(s)
    notes(s, """O QUE FALAR
Isto é o programa rodando. Três painéis: original, imagem limpa, caixa na fratura.
Não substitui o médico. O médico confirma ou descarta.
A demo ao vivo está no ar: viewdicom-tcc.web.app
Se a banca pedir, abro o site, envio um PNG de punho e mostro a caixa.
O modelo roda no navegador, em ONNX — não depende do meu notebook ligado.
Se o Wi-Fi falhar, tenho a mesma tela no PDF e no OneDrive.""")

    # 11 Subset — números grandes
    s = prs.slides.add_slide(blank)
    night(s)
    kicker(s, "Teste rápido  ·  1.600 imagens")
    title(s, "Passamos da meta")
    nums = [("58%", "baseline", CORAL), ("70%", "meta", MUTED), ("84%", "RADILÚMEN", CYAN)]
    for i, (n, c, col) in enumerate(nums):
        x = 0.8 + i * 4.1
        txt(s, x, 2.1, 3.8, 1.2, n, 60, True, col)
        txt(s, x, 3.35, 3.8, 0.5, c, 18, False, MUTED)
    txt(s, 0.8, 4.6, 11.5, 1.2, "Mesmo conjunto, mesmas épocas.\nO ganho vem do contraste, do transfer learning e da augmentação.", 18, False, SOFT)
    fade(s)
    notes(s, """O QUE FALAR
Primeiro experimento: recorte de cerca de 1.600 imagens, mesmas épocas, mesma YOLO.
Baseline, do zero: 58% de mAP. Abaixo da meta de 70% que eu tinha no projeto.
Com CLAHE, transfer e augmentação: 84%. Passamos da meta.
O ganho não é “mágica da YOLO”. É contraste + pesos prontos + variação das imagens.
Este número ainda não é o resultado oficial. É a prova de que a receita funciona. O oficial vem no próximo slide.""")
    s = prs.slides.add_slide(blank)
    night(s)
    kicker(s, "Resultado oficial  ·  base completa")
    title(s, "O que a banca precisa ver")
    nums = [("92,9%", "mAP"), ("92,2%", "precisão"), ("86,4%", "sensibilidade")]
    for i, (n, c) in enumerate(nums):
        x = 0.8 + i * 4.1
        txt(s, x, 1.7, 3.8, 1.15, n, 52, True, CYAN)
        txt(s, x, 2.9, 3.8, 0.45, c, 18, False, MUTED)
    txt(s, 0.8, 4.0, 11.5, 1.8, "Esses números valem na base pública de punho.\nNão medem se o médico do plantão erra menos com o sistema ligado.", 20, False, SOFT)
    fade(s)
    notes(s, """O QUE FALAR
Resultado oficial, base completa, teste separado por paciente, 2.026 imagens de teste.
Memorize estes três:
mAP a IoU 0,5: 92,9%. É o quanto a caixa acerta o lugar da fratura.
Precisão: 92,2%. Quando o sistema marca, na maior parte das vezes acertou.
Sensibilidade, recall: 86,4%. Ainda perde cerca de 14% das fraturas. Por isso o médico não pode desligar o olho.
Caveat, fale em voz alta: estes números valem nesta base pública de punho. Não medem se o plantão erra menos com o sistema ligado. Isso seria um estudo clínico, e eu não fiz.""")

    # 13 Erros — duas radiografias
    s = prs.slides.add_slide(blank)
    night(s)
    kicker(s, "Honestidade")
    title(s, "O sistema também erra")
    crop_dir = FIG / "_slides"
    fp = left_crop(FIG / "demo_software_fp.png", crop_dir / "fp.jpg", 0.40)
    fn = left_crop(FIG / "demo_software_fn.png", crop_dir / "fn.jpg", 0.40)
    photo_fit(s, fp, 0.55, 1.45, 4.7)
    photo_fit(s, fn, 7.05, 1.45, 4.7)
    txt(s, 0.55, 6.22, 5.8, 0.45, "Marcou o metal", 16, True, CORAL)
    txt(s, 7.05, 6.22, 5.8, 0.45, "Não marcou nada", 16, True, CYAN)
    fade(s)
    notes(s, """O QUE FALAR
O sistema também erra. Mostre as duas fotos.
Esquerda: falso positivo. Marcou o metal, o material de síntese, como se fosse fratura. Contraste alto engana a rede.
Direita: falso negativo. Fratura sutil, não marcou nada. O médico precisaria olhar sozinho.
Por isso a caixa é um aviso, não um laudo. O médico fecha o caso.
Se perguntarem o limiar: usei confiança 0,40 nestes recortes para mostrar o erro de verdade, não esconder.""")

    # 14 Conclusão
    s = prs.slides.add_slide(blank)
    night(s)
    kicker(s, "Fechamento")
    title(s, "Conclusão")
    bits = [
        ("Objetivos", "Modelo, métricas acima de 70%, tela com a caixa."),
        ("Importa", "Filtro contra o cansaço. Segunda opinião — nunca sozinho."),
        ("Dificuldades", "GPU, PNG sem nome, checkpoint, Python em outro PC."),
        ("Depois", "Ruído, limiar, metal. Um dia: DICOM de saída e estudo com médico."),
    ]
    for i, (h, b) in enumerate(bits):
        x = 0.7 + (i % 2) * 6.2
        y = 1.55 + (i // 2) * 2.15
        txt(s, x, y, 5.8, 0.4, h, 16, True, CYAN)
        txt(s, x, y + 0.45, 5.8, 1.3, b, 18, False, SOFT)
    fade(s)
    notes(s, """O QUE FALAR
Feche em quatro frases, sem ler o slide:
Objetivos cumpridos: treinei o modelo, as métricas passaram de 70%, existe uma tela com a caixa.
O que importa na prática: filtro contra o cansaço. Segunda opinião. Nunca sozinho.
Dificuldades reais: GPU, PNG sem nome, checkpoint, Python que no PC da banca pode não abrir — por isso o site.
Depois: ruído, limiar, metal. Um dia gravar DICOM de saída e medir com médico no plantão.
Não prometa hospital implantado. Prometa o que está no ar hoje.""")

    # Limitações — dizer em voz alta
    s = prs.slides.add_slide(blank)
    night(s)
    kicker(s, "Honestidade")
    title(s, "O que este trabalho não prova")
    lims = [
        ("Teste N = 2.026", "O 16.211 é o treino. As métricas oficiais são do teste."),
        ("Pacote, não ablação", "8-bit + CLAHE + transfer + aug juntos. Sem isolar cada um."),
        ("Baseline só no subset", "58% → 92,9% mistura técnica com mais dados e 100 épocas."),
        ("Limiar", "Tabelas: conf 0,001 (mAP). Software: 0,25. Não é o mesmo número."),
        ("Uma semente", "Sem desvio padrão. Sem dataset externo. Sem Grad-CAM."),
    ]
    for i, (h, b) in enumerate(lims):
        y = 1.45 + i * 0.95
        txt(s, 0.7, y, 12, 0.32, h, 18, True, CYAN)
        txt(s, 0.7, y + 0.32, 12, 0.5, b, 16, False, SOFT)
    fade(s)
    notes(s, """O QUE FALAR
Diga isto antes da banca perguntar. É o slide que sobe a nota.
Um — as métricas oficiais são do teste, 2.026 imagens. Dezesseis mil é o treino. Foi erro de legenda, já corrigido no PDF.
Dois — não fiz ablação. O ganho de 26 pontos no subset é do pacote inteiro.
Três — não treinei baseline no dataset completo. Por isso não comparo 58 com 92,9 como se fosse só a técnica.
Quatro — 86,37% de sensibilidade é no protocolo de mAP, conf 0,001. No software o médico vê 0,25. Não medi o par P/R nesse limiar.
Cinco — uma rodada, semente 42. Sem GRAZPEDWRI-DX de outros autores no texto: o capítulo de relacionados foi validado em março e não reabri.
Se perguntarem Ju e Cai 2023: “existem trabalhos no mesmo dataset; não os incluí e não cito números de memória.”""")

    s = prs.slides.add_slide(blank)
    fill_photo(s, PUNHO)
    overlay(s, 0, 0, SW, SH, NIGHT, 64000)
    lg = 1.35
    s.shapes.add_picture(str(LOGO), Inches((SW - lg) / 2), Inches(1.15), Inches(lg), Inches(lg))
    txt(s, 0.5, 2.65, 12.3, 0.4, "RADILÚMEN", 18, True, CYAN, PP_ALIGN.CENTER)
    txt(s, 0.5, 3.15, 12.3, 0.7, "Obrigado", 44, True, WHITE, PP_ALIGN.CENTER)
    txt(s, 0.5, 3.9, 12.3, 0.4, "Dúvidas", 18, False, MUTED, PP_ALIGN.CENTER)
    txt(s, 0.5, 4.45, 12.3, 0.35, "Lucas Rodrigues Zabot  ·  Prof. Sérgio Marques", 14, False, MUTED, PP_ALIGN.CENTER)
    txt(s, 0.5, 5.15, 12.3, 0.5, "viewdicom-tcc.web.app", 22, True, CYAN, PP_ALIGN.CENTER)
    fade(s)
    notes(s, """O QUE FALAR
Obrigado. Dúvidas.
Se pedirem o programa: abra viewdicom-tcc.web.app, envie um punho, mostre a caixa.
Se pedirem o código e o PDF: OneDrive da disciplina.
Se perguntarem “pode diagnosticar sozinho?”: não. Apoio visual.
Se perguntarem “funciona em outro osso?”: treinei só punho. Outro sítio exigiria outro treino.
Se perguntarem o nome: RADILÚMEN — radiografia + lúmen, a luz no ponto da fratura.
Agradeça o professor Sérgio Marques e a banca.""")

    prs.save(str(OUT_LATEX))
    roteiro = []
    for i, slide in enumerate(prs.slides, 1):
        body = slide.notes_slide.notes_text_frame.text.strip()
        roteiro.append(f"{'=' * 72}\nSLIDE {i}\n{'=' * 72}\n{body}\n")
    roteiro_path = ROOT / "latex" / "Roteiro_apresentacao_PG_II.txt"
    roteiro_path.write_text("\n".join(roteiro), encoding="utf-8")
    for dest in (OUT_DL, OUT_DL2):
        try:
            shutil.copy2(OUT_LATEX, dest)
            print("saved", dest)
        except PermissionError:
            print("locked", dest)
    try:
        shutil.copy2(roteiro_path, Path(r"C:\Users\Lucas R Zabot\Downloads\Roteiro_apresentacao_PG_II.txt"))
    except PermissionError:
        pass
    print("slides", len(prs.slides))
    print(OUT_LATEX)
    print(roteiro_path)


if __name__ == "__main__":
    build()

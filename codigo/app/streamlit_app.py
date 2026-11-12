"""
Interface Streamlit — demonstração do detector de fraturas (punho).

Uso:
  cd codigo
  .\\venv\\Scripts\\Activate.ps1
  streamlit run app/streamlit_app.py

Salva pacotes para o TCC em:
  latex/Figuras/demonstracao/caso_XX/
    original.png
    deteccao.png
    coordenadas.csv
    painel_tcc.png   (original | detecção + tabela)
"""

from __future__ import annotations

import csv
import io
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

CODIGO_DIR = Path(__file__).resolve().parents[1]
if str(CODIGO_DIR) not in sys.path:
    sys.path.insert(0, str(CODIGO_DIR))

from src.preprocess import PreprocessConfig, preprocess_array

DEFAULT_WEIGHTS = [
    CODIGO_DIR / "weights" / "final_proposed.pt",
    CODIGO_DIR / "runs" / "detect" / "proposed" / "weights" / "final_proposed.pt",
    CODIGO_DIR / "runs" / "detect" / "proposed" / "weights" / "safe_checkpoints" / "safe_epoch_100.pt",
]

DEMO_ROOT = CODIGO_DIR.parent / "latex" / "Figuras" / "demonstracao"


@st.cache_resource
def load_model(weights_path: str) -> YOLO:
    return YOLO(weights_path)


def resolve_weights() -> Path | None:
    for path in DEFAULT_WEIGHTS:
        if path.is_file():
            return path
    return None


def draw_detections(image_bgr: np.ndarray, result, conf_min: float) -> np.ndarray:
    out = image_bgr.copy()
    if result.boxes is None or len(result.boxes) == 0:
        return out

    h, w = out.shape[:2]
    for box in result.boxes:
        conf = float(box.conf[0])
        if conf < conf_min:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 2)
        label = f"fracture {conf:.2f}"
        cv2.putText(
            out,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )
    return out


def collect_rows(result, conf_min: float) -> list[dict]:
    rows: list[dict] = []
    if result.boxes is None:
        return rows
    for box in result.boxes:
        conf = float(box.conf[0])
        if conf < conf_min:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        rows.append(
            {
                "classe": "fracture",
                "confianca": round(conf, 4),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )
    return rows


def _as_bgr(img: np.ndarray) -> np.ndarray:
    """Garante BGR de 3 canais (PNG com alfa chega como BGRA)."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _fit_height(img: np.ndarray, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = height / max(h, 1)
    return cv2.resize(img, (max(1, int(w * scale)), height), interpolation=cv2.INTER_AREA)


def make_table_panel(rows: list[dict], width: int, height: int) -> np.ndarray:
    """Desenha a tabela de coordenadas como imagem (para o TCC)."""
    panel = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(
        panel,
        "Coordenadas da deteccao (caixa)",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    headers = ["#", "classe", "conf", "x1", "y1", "x2", "y2"]
    y = 90
    cv2.putText(
        panel,
        "  ".join(f"{h:>6}" for h in headers),
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    y += 28
    cv2.line(panel, (20, y - 16), (width - 20, y - 16), (180, 180, 180), 1)

    if not rows:
        cv2.putText(
            panel,
            "Nenhuma deteccao acima do limiar.",
            (20, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 180),
            2,
            cv2.LINE_AA,
        )
        return panel

    for i, r in enumerate(rows, start=1):
        line = (
            f"{i:>6}  {r['classe']:>6}  {r['confianca']:>6.2f}  "
            f"{r['x1']:>6}  {r['y1']:>6}  {r['x2']:>6}  {r['y2']:>6}"
        )
        cv2.putText(
            panel,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
        y += 32
        if y > height - 30:
            break

    nota = "Obs.: x1,y1 = canto superior esquerdo; x2,y2 = canto inferior direito (pixels)."
    cv2.putText(
        panel,
        nota,
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (80, 80, 80),
        1,
        cv2.LINE_AA,
    )
    return panel


def make_tcc_panel(original_bgr: np.ndarray, annotated_bgr: np.ndarray, rows: list[dict]) -> np.ndarray:
    """Monta painel: original | detecção | tabela."""
    h = 520
    left = _fit_height(_as_bgr(original_bgr), h)
    mid = _fit_height(_as_bgr(annotated_bgr), h)
    table = make_table_panel(rows, width=520, height=h)

    def labeled(img: np.ndarray, title: str) -> np.ndarray:
        canvas = np.full((img.shape[0] + 36, img.shape[1], 3), 255, dtype=np.uint8)
        canvas[36:, :] = img
        cv2.putText(
            canvas,
            title,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        return canvas

    left = labeled(left, "Original")
    mid = labeled(mid, "Deteccao (bounding box)")
    table = labeled(table, "Tabela de coordenadas")
    # equalize heights
    H = max(left.shape[0], mid.shape[0], table.shape[0])

    def pad(img: np.ndarray) -> np.ndarray:
        if img.shape[0] == H:
            return img
        out = np.full((H, img.shape[1], 3), 255, dtype=np.uint8)
        out[: img.shape[0], :] = img
        return out

    return np.hstack([pad(left), pad(mid), pad(table)])


def next_case_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    existing = sorted(root.glob("caso_*"))
    n = 1
    if existing:
        try:
            n = max(int(p.name.split("_")[1]) for p in existing) + 1
        except Exception:
            n = len(existing) + 1
    out = root / f"caso_{n:02d}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_tcc_pack(
    case_dir: Path,
    stem: str,
    original_bgr: np.ndarray,
    annotated_bgr: np.ndarray,
    rows: list[dict],
) -> Path:
    cv2.imwrite(str(case_dir / "original.png"), original_bgr)
    cv2.imwrite(str(case_dir / "deteccao.png"), annotated_bgr)
    panel = make_tcc_panel(original_bgr, annotated_bgr, rows)
    cv2.imwrite(str(case_dir / "painel_tcc.png"), panel)

    csv_path = case_dir / "coordenadas.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["classe", "confianca", "x1", "y1", "x2", "y2"],
        )
        writer.writeheader()
        writer.writerows(rows)

    meta = case_dir / "readme.txt"
    meta.write_text(
        f"Caso gerado em {datetime.now().isoformat(timespec='seconds')}\n"
        f"Arquivo fonte: {stem}\n"
        f"Deteccoes: {len(rows)}\n"
        "Arquivos:\n"
        "- original.png\n"
        "- deteccao.png\n"
        "- coordenadas.csv  (x1,y1,x2,y2 em pixels)\n"
        "- painel_tcc.png   (usar no LaTeX)\n",
        encoding="utf-8",
    )
    return case_dir


def main() -> None:
    st.set_page_config(
        page_title="TCC — Detecção de Fraturas (Punho)",
        page_icon="🦴",
        layout="wide",
    )

    st.title("Detector de fraturas — punho")
    st.caption(
        "Demonstração acadêmica (YOLO + GRAZPEDWRI-DX). "
        "Não substitui avaliação médica. "
        "Coordenadas da caixa: x1,y1 (canto sup. esq.) e x2,y2 (canto inf. dir.)."
    )

    weights = resolve_weights()
    uploaded_weights = st.sidebar.file_uploader("Ou carregue um .pt", type=["pt"])
    conf = st.sidebar.slider("Confiança mínima", 0.05, 0.95, 0.25, 0.05)
    use_preprocess = st.sidebar.checkbox("Pré-processar (8-bit + CLAHE)", value=True)

    if uploaded_weights is not None:
        tmp = CODIGO_DIR / "weights"
        tmp.mkdir(parents=True, exist_ok=True)
        weights_path = tmp / "uploaded_model.pt"
        weights_path.write_bytes(uploaded_weights.getvalue())
    elif weights is not None:
        weights_path = weights
        st.sidebar.success(f"Modelo: {weights_path.name}")
    else:
        st.error(
            "Modelo não encontrado. Baixe `final_proposed.pt` do Drive e coloque em "
            "`codigo/weights/final_proposed.pt`."
        )
        st.stop()

    model = load_model(str(weights_path))

    file = st.file_uploader(
        "Envie uma radiografia (PNG, JPG ou DICOM)",
        type=["png", "jpg", "jpeg", "dcm", "dicom"],
    )
    if file is None:
        st.info("Aguardando imagem de teste.")
        st.markdown(
            f"Pacotes salvos para o TCC ficam em: `{DEMO_ROOT}`"
        )
        return

    raw = np.frombuffer(file.getvalue(), dtype=np.uint8)
    if file.name.lower().endswith((".dcm", ".dicom")):
        import pydicom

        ds = pydicom.dcmread(io.BytesIO(file.getvalue()))
        gray = ds.pixel_array.astype(np.float32)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        original_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        decoded = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
        if decoded is None:
            st.error("Não foi possível ler a imagem.")
            return
        if decoded.ndim == 2:
            original_bgr = cv2.cvtColor(decoded, cv2.COLOR_GRAY2BGR)
            gray = decoded
        else:
            original_bgr = _as_bgr(decoded)
            gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)

    if use_preprocess:
        processed = preprocess_array(gray, PreprocessConfig(output_bgr=True))
        infer_img = processed
    else:
        if original_bgr.dtype != np.uint8:
            infer_img = cv2.normalize(original_bgr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            infer_img = original_bgr
        processed = infer_img

    t0 = time.perf_counter()
    results = model.predict(source=infer_img, conf=conf, verbose=False)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    result = results[0]
    annotated = draw_detections(infer_img, result, conf)
    rows = collect_rows(result, conf)

    n_det = len(rows)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Original")
        st.image(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col2:
        st.subheader("Pré-processada")
        st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col3:
        st.subheader("Detecção")
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

    m1, m2 = st.columns(2)
    m1.metric("Detecções", n_det)
    m2.metric("Tempo de inferência", f"{elapsed_ms:.1f} ms")

    st.subheader("Tabela de coordenadas (x1, y1, x2, y2)")
    if rows:
        st.dataframe(rows, use_container_width=True)
    else:
        st.warning("Nenhuma detecção acima do limiar de confiança.")

    panel = make_tcc_panel(original_bgr, annotated, rows)
    st.subheader("Painel para o TCC (original + detecção + tabela)")
    st.image(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    ok_ann, buf_ann = cv2.imencode(".png", annotated)
    ok_panel, buf_panel = cv2.imencode(".png", panel)
    ok_orig, buf_orig = cv2.imencode(".png", original_bgr)

    with c1:
        if ok_orig:
            st.download_button(
                "Baixar original",
                data=buf_orig.tobytes(),
                file_name=f"{Path(file.name).stem}_original.png",
                mime="image/png",
            )
    with c2:
        if ok_ann:
            st.download_button(
                "Baixar detecção",
                data=buf_ann.tobytes(),
                file_name=f"{Path(file.name).stem}_pred.png",
                mime="image/png",
            )
    with c3:
        if ok_panel:
            st.download_button(
                "Baixar painel TCC",
                data=buf_panel.tobytes(),
                file_name=f"{Path(file.name).stem}_painel_tcc.png",
                mime="image/png",
            )

    if st.button("Salvar pacote completo na pasta do TCC", type="primary"):
        case_dir = next_case_dir(DEMO_ROOT)
        save_tcc_pack(case_dir, Path(file.name).stem, original_bgr, annotated, rows)
        st.success(f"Salvo em: {case_dir}")
        st.code(str(case_dir), language=None)


if __name__ == "__main__":
    main()

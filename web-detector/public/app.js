/* Detector YOLOv8n no navegador — saída ONNX [1, 5, 8400] = cx,cy,w,h,score */
const IMGSZ = 640;
const PAD = 114 / 255;

const statusEl = document.getElementById("status");
const fileEl = document.getElementById("file");
const exemploEl = document.getElementById("exemplo");
const confEl = document.getElementById("conf");
const confVal = document.getElementById("confVal");
const origCv = document.getElementById("orig");
const detCv = document.getElementById("det");
const tabelaEl = document.getElementById("tabela");

let session = null;
let lastImg = null;

function setStatus(msg) {
  statusEl.textContent = msg;
}

confEl.addEventListener("input", () => {
  confVal.textContent = Number(confEl.value).toFixed(2).replace(".", ",");
  if (lastImg) infer(lastImg);
});

async function loadModel() {
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
  session = await ort.InferenceSession.create("models/final_proposed.onnx", {
    executionProviders: ["wasm"],
  });
  setStatus("Modelo pronto. Envie uma radiografia ou use o exemplo.");
}

function letterbox(imgData, srcW, srcH) {
  const scale = Math.min(IMGSZ / srcW, IMGSZ / srcH);
  const newW = Math.round(srcW * scale);
  const newH = Math.round(srcH * scale);
  const padX = (IMGSZ - newW) / 2;
  const padY = (IMGSZ - newH) / 2;

  const c = document.createElement("canvas");
  c.width = IMGSZ;
  c.height = IMGSZ;
  const ctx = c.getContext("2d");
  ctx.fillStyle = `rgb(114,114,114)`;
  ctx.fillRect(0, 0, IMGSZ, IMGSZ);

  const tmp = document.createElement("canvas");
  tmp.width = srcW;
  tmp.height = srcH;
  tmp.getContext("2d").putImageData(imgData, 0, 0);
  ctx.drawImage(tmp, padX, padY, newW, newH);

  const boxed = ctx.getImageData(0, 0, IMGSZ, IMGSZ);
  const chw = new Float32Array(3 * IMGSZ * IMGSZ);
  for (let i = 0; i < IMGSZ * IMGSZ; i++) {
    const r = boxed.data[i * 4] / 255;
    const g = boxed.data[i * 4 + 1] / 255;
    const b = boxed.data[i * 4 + 2] / 255;
    chw[i] = r;
    chw[IMGSZ * IMGSZ + i] = g;
    chw[2 * IMGSZ * IMGSZ + i] = b;
  }
  return { tensor: chw, scale, padX, padY };
}

function nms(boxes, iouTh = 0.5) {
  boxes.sort((a, b) => b.score - a.score);
  const keep = [];
  for (const b of boxes) {
    let ok = true;
    for (const k of keep) {
      const xx1 = Math.max(b.x1, k.x1);
      const yy1 = Math.max(b.y1, k.y1);
      const xx2 = Math.min(b.x2, k.x2);
      const yy2 = Math.min(b.y2, k.y2);
      const inter = Math.max(0, xx2 - xx1) * Math.max(0, yy2 - yy1);
      const union = (b.x2 - b.x1) * (b.y2 - b.y1) + (k.x2 - k.x1) * (k.y2 - k.y1) - inter;
      if (union > 0 && inter / union > iouTh) {
        ok = false;
        break;
      }
    }
    if (ok) keep.push(b);
  }
  return keep;
}

function decode(output, scale, padX, padY, srcW, srcH, confMin) {
  const data = output.data;
  const n = output.dims[2];
  const raw = [];
  for (let i = 0; i < n; i++) {
    const score = data[4 * n + i];
    if (score < confMin) continue;
    const cx = data[i];
    const cy = data[n + i];
    const w = data[2 * n + i];
    const h = data[3 * n + i];
    let x1 = (cx - w / 2 - padX) / scale;
    let y1 = (cy - h / 2 - padY) / scale;
    let x2 = (cx + w / 2 - padX) / scale;
    let y2 = (cy + h / 2 - padY) / scale;
    x1 = Math.max(0, Math.min(srcW - 1, x1));
    y1 = Math.max(0, Math.min(srcH - 1, y1));
    x2 = Math.max(0, Math.min(srcW - 1, x2));
    y2 = Math.max(0, Math.min(srcH - 1, y2));
    raw.push({ x1, y1, x2, y2, score });
  }
  return nms(raw);
}

function drawImage(canvas, img) {
  canvas.width = img.width;
  canvas.height = img.height;
  canvas.getContext("2d").drawImage(img, 0, 0);
}

function drawDetections(canvas, img, boxes) {
  drawImage(canvas, img);
  const ctx = canvas.getContext("2d");
  ctx.strokeStyle = "#1ed760";
  ctx.lineWidth = Math.max(2, img.width / 250);
  ctx.font = `${Math.max(14, img.width / 32)}px Segoe UI`;
  ctx.fillStyle = "#1ed760";
  for (const b of boxes) {
    ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
    const label = `fracture ${b.score.toFixed(2)}`;
    ctx.fillText(label, b.x1, Math.max(16, b.y1 - 6));
  }
}

function renderTable(boxes) {
  if (!boxes.length) {
    tabelaEl.innerHTML = "Nenhuma detecção acima do limiar.";
    return;
  }
  const rows = boxes
    .map(
      (b, i) =>
        `<tr><td>${i + 1}</td><td>fracture</td><td>${b.score.toFixed(2)}</td><td>${Math.round(b.x1)}</td><td>${Math.round(b.y1)}</td><td>${Math.round(b.x2)}</td><td>${Math.round(b.y2)}</td></tr>`
    )
    .join("");
  tabelaEl.innerHTML = `<table><thead><tr><th>#</th><th>classe</th><th>conf</th><th>x1</th><th>y1</th><th>x2</th><th>y2</th></tr></thead><tbody>${rows}</tbody></table>`;
}

async function infer(img) {
  if (!session) return;
  lastImg = img;
  const confMin = Number(confEl.value);
  setStatus("Detectando…");
  const t0 = performance.now();

  const src = document.createElement("canvas");
  src.width = img.width;
  src.height = img.height;
  const sctx = src.getContext("2d");
  sctx.drawImage(img, 0, 0);
  const imgData = sctx.getImageData(0, 0, img.width, img.height);
  const { tensor, scale, padX, padY } = letterbox(imgData, img.width, img.height);
  const input = new ort.Tensor("float32", tensor, [1, 3, IMGSZ, IMGSZ]);
  const feeds = {};
  feeds[session.inputNames[0]] = input;
  const out = await session.run(feeds);
  const output = out[session.outputNames[0]];
  const boxes = decode(output, scale, padX, padY, img.width, img.height, confMin);

  drawImage(origCv, img);
  drawDetections(detCv, img, boxes);
  renderTable(boxes);
  const ms = (performance.now() - t0).toFixed(0);
  setStatus(`${boxes.length} detecção(ões) · ${ms} ms no navegador`);
}

function loadFromUrl(url) {
  const img = new Image();
  img.onload = () => infer(img);
  img.onerror = () => setStatus("Não foi possível ler a imagem.");
  img.src = url;
}

fileEl.addEventListener("change", (ev) => {
  const f = ev.target.files && ev.target.files[0];
  if (!f) return;
  const url = URL.createObjectURL(f);
  loadFromUrl(url);
});

exemploEl.addEventListener("click", () => {
  loadFromUrl("exemplos/punho_fratura.jpg");
});

loadModel().catch((err) => {
  console.error(err);
  setStatus("Falha ao carregar o modelo. Recarregue a página.");
});

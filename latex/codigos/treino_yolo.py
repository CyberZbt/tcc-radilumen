# Trecho essencial — src/train.py
from ultralytics import YOLO
from src.utils import load_yaml, resolve_data_path, resolve_device, resolve_model_path
from src.train import build_train_kwargs

cfg = load_yaml("configs/proposed.yaml")
model_name = resolve_model_path(cfg.get("model", "yolov8n.pt"))
data_yaml = resolve_data_path(cfg["data"])
device = resolve_device(cfg.get("device", "auto"))

train_kwargs = build_train_kwargs(cfg, {})
train_kwargs.update({
    "data": data_yaml,
    "device": device,
    "project": "runs/detect",
    "name": cfg.get("name", "proposed"),
    "exist_ok": True,
})

model = YOLO(model_name)
model.train(**train_kwargs)

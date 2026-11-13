# Trecho essencial - avaliacao no conjunto de teste
from ultralytics import YOLO

model = YOLO("runs/detect/proposed/weights/final_proposed.pt")
metrics = model.val(
    data="dataset/yolo_8bit/fracture_only/data_fracture.yaml",
    split="test",
    imgsz=640,
    conf=0.001,  # curva de mAP (padrao Ultralytics); o software usa 0.25
    iou=0.7,
)
print("P:", metrics.box.mp)
print("R:", metrics.box.mr)
print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)

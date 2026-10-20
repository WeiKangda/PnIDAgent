"""高分辨率切块检测器: 从 soup5 (yolo11x) 微调, 640 切块, 原分辨率. batch 6 (和正在跑的重训链共用显卡).
  python train_tile.py <A|A1|A2> <gpu> [epochs]"""
import os, sys
R = "/home/suxinqi666/code/PnIDAgent_project"; os.chdir(R)
from ultralytics import YOLO
name, dev = sys.argv[1], int(sys.argv[2]); ep = int(sys.argv[3]) if len(sys.argv) > 3 else 40
m = YOLO(f"{R}/runs/detect/unsupervised_symbol_recognition/runs/soup5/weights/best.pt")
m.train(data=f"{R}/datasets/exp65_tile_{name}/data.yaml", imgsz=640, epochs=ep, batch=6, device=dev, optimizer="SGD", lr0=0.002, cos_lr=True,
        warmup_epochs=2, patience=1000, single_cls=True, mosaic=0.5, close_mosaic=5, scale=0.2, translate=0.1, degrees=0, fliplr=0.5, flipud=0.5,
        hsv_h=0, hsv_s=0, hsv_v=0.2, workers=6, project=f"{R}/runs/exp65", name=f"tile_{name}", exist_ok=True, verbose=False, plots=False)
print("TILE_DONE", name, flush=True)

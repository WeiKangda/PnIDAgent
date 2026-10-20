"""大设备专用检测器: 只学 A 半里 >=150px 的大符号, 整页缩到 1280 输入, yolo11s (占显存小, 能和正在跑的训练共用一张卡)."""
import sys, os
os.chdir("/home/suxinqi666/code/PnIDAgent_project")
from ultralytics import YOLO
m = YOLO("yolo11s.pt")
NM = sys.argv[2] if len(sys.argv) > 2 else "A"
m.train(data=f"datasets/exp65_big_{NM}/data.yaml", imgsz=1280, epochs=150, batch=4, device=int(sys.argv[1]), optimizer="AdamW", lr0=0.001,
        cos_lr=True, patience=1000, single_cls=True, mosaic=0.5, scale=0.3, degrees=0, fliplr=0.5, flipud=0.5, hsv_h=0, hsv_s=0, hsv_v=0.3,
        workers=4, project="runs/exp65", name=f"big_{NM}", exist_ok=True, verbose=False, plots=False)
print("BIG_DONE")

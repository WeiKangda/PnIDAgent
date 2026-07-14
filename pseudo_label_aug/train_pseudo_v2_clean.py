"""迭代伪标签(半监督): 用更强教师(only_pseudo)重打的标签 + scale0.9 微调
"""
from ultralytics import YOLO
YOLO("runs/pid_combined/y11x_heavyaug_final/weights/best.pt").train(
    data="/home/suxinqi666/code/PnIDAgent_project/datasets/yolo_pseudo_v2/data.yaml",
    imgsz=1280, epochs=50, batch=2, device=0,
    project="/home/suxinqi666/code/PnIDAgent_project/runs/pid_combined",
    name="pseudo_v2_clean", exist_ok=True,
    optimizer="auto", lr0=0.0005, patience=15, seed=0,
    close_mosaic=10, mosaic=0.5, scale=0.3,
    verbose=False, plots=False,
)
print("[DONE] pseudo_v2")

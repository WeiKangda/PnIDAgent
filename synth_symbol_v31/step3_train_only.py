import os,sys,time
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # -> project root
_D='unsupervised_symbol_recognition'
def log(*a): print('[v31rt]',*a,flush=True)
from ultralytics import YOLO
log('从 v28b 训练(复用已合成 yolo_real31_repro), GPU1 ...')
m=YOLO(f'runs/detect/{_D}/runs/v28b_rebalance/weights/best.pt')
m.train(data='datasets/yolo_joint31_repro/data.yaml', imgsz=1280, epochs=12, batch=4, device=0,
        project=f'{_D}/runs', name='v31_repro', exist_ok=True, verbose=False,
        optimizer='SGD', lr0=0.0004, cos_lr=True, patience=5, single_cls=True,
        mosaic=0.3, close_mosaic=3, scale=0.25, degrees=2, workers=8)
log('train done')
del m; import torch,gc; gc.collect(); torch.cuda.empty_cache()
sys.path.insert(0,'gpt_detect/llm_baseline'); import config, eval as ev
mm=YOLO(f'runs/detect/{_D}/runs/v31_repro/weights/best.pt')
per=[]
for s0 in config.SAMPLES:
    gt=ev.load_gt_from_pnid_json(config.get_gt_path(s0))
    rr=mm.predict(str(config.get_image_path(s0)),imgsz=1280,conf=0.10,iou=0.5,verbose=False,max_det=400,device=0)[0]
    preds=[{"bbox":b.tolist(),"score":float(cc)} for b,cc in zip(rr.boxes.xyxy.cpu().numpy(),rr.boxes.conf.cpu().numpy())]
    keep=[{"bbox":p["bbox"],"score":.9} for p in preds if p["score"]>=0.65]
    per.append(ev.evaluate(keep,gt,iou_thresholds=[0.5])["by_iou"]["iou_0.5"]["F1"])
log(f'*** v31r 标准口径 macro-F1 = {sum(per)/4:.4f}  逐张 {[round(v,3) for v in per]}  (原v31=0.883 soup5=0.888) ***')
log('ALLDONE')

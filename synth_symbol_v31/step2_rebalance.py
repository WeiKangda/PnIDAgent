"""v28b: 联合训练 Surry 加权版 — yolo_quality 过采样 2x, 其余同 v27
起点 v25 (Surry最强). 验证: 配比是不是 CVCS_Surry 没回血的原因
"""
import os, sys, time, traceback
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # -> project root
_D = 'unsupervised_symbol_recognition'
LOGF = open(f'{_D}/overnight_v10.log', 'a')
def log(*a):
    m = ' '.join(str(x) for x in a)
    print(m, flush=True); LOGF.write(time.strftime('[%H:%M:%S][v28b] ') + m + '\n'); LOGF.flush()
import torch
if torch.cuda.is_available(): torch.zeros(1, device='cuda')
from ultralytics import YOLO

def main():
    R = os.path.abspath('datasets')
    yaml = f"""train:
  - {R}/yolo_quality/images/train
  - {R}/yolo_quality_x2/images/train
  - {R}/yolo_apr26/images/train
  - {R}/yolo_dpid_nuke/images/train
  - {R}/yolo_pseudo_v3_union/images/train
val:
  - {R}/yolo_quality/images/val
  - {R}/yolo_apr26/images/val
names:
  0: item
"""
    os.makedirs('datasets/yolo_joint28b', exist_ok=True)
    with open('datasets/yolo_joint28b/data.yaml','w') as fh: fh.write(yaml)
    m = YOLO(f'runs/detect/{_D}/runs/v25_quality/weights/best.pt')
    log('v28b start (quality x2 oversample, from v25)')
    m.train(data='datasets/yolo_joint28b/data.yaml', imgsz=1280, epochs=12, batch=4, device=0,
            project=f'{_D}/runs', name='v28b_rebalance', exist_ok=True, verbose=False,
            optimizer='SGD', lr0=0.0004, cos_lr=True, patience=5, single_cls=True,
            mosaic=0.3, close_mosaic=3, scale=0.25, degrees=2, workers=8)
    log('v28b train done')
    sys.path.insert(0, 'gpt_detect/llm_baseline')
    import config, eval as ev
    for c in (f'runs/detect/{_D}/runs/v28b_rebalance/weights/best.pt', f'{_D}/runs/v28b_rebalance/weights/best.pt'):
        if os.path.exists(c):
            mm = YOLO(c)
            for conf in (0.5, 0.55, 0.6, 0.65):
                f1s = []
                for s0 in config.SAMPLES:
                    gt = ev.load_gt_from_pnid_json(config.get_gt_path(s0))
                    r = mm.predict(str(config.get_image_path(s0)), imgsz=1280, conf=conf, iou=0.5, verbose=False)[0]
                    preds = [{"bbox": b.tolist(), "score": float(cc)} for b, cc in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())]
                    f1s.append(ev.evaluate(preds, gt, iou_thresholds=[0.5])["by_iou"]["iou_0.5"]["F1"])
                log(f'v28b real4 conf{conf}: {sum(f1s)/4:.3f} {[round(v,3) for v in f1s]} (v25=0.866 v27=0.864, 目标0.90)')
            break
    log('v28b end')

if __name__ == '__main__':
    try: main()
    except Exception: log('v28b FAIL\n' + traceback.format_exc())

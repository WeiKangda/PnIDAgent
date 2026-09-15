"""v16: 他们的四级火箭 × 我们的图例视野 = 联合教师伪标签 v3
1) 65 张真实页: pseudo_v2_clean(conf0.5@1280) ∪ 图例合成检测器(peak 2880/conf0.4) 并集打标
2) 学生从 y11x_heavyaug_final 底座, 按其阶段3 clean 配方重训
3) real4 官方协议评测 vs 0.821
零人工标注不变(两个教师血统均无人工框)。
"""
import os, sys, json, glob, time, traceback
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))  # -> project root
_D = 'unsupervised_symbol_recognition'
LOGF = open(f'{_D}/overnight_v10.log', 'a')
def log(*a):
    m = ' '.join(str(x) for x in a)
    print(m, flush=True); LOGF.write(time.strftime('[%H:%M:%S][v16] ') + m + '\n'); LOGF.flush()
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from ultralytics import YOLO

T1 = 'runs/pid_combined/pseudo_v2_clean/weights/best.pt'          # 强教师(通用符号)
T2 = 'runs/detect/unsupervised_symbol_recognition/runs/synth_det_v2/weights/best.pt'  # 图例教师(盲区补漏)
BASE = 'runs/pid_combined/y11x_heavyaug_final/weights/best.pt'    # 学生底座
SRC = 'datasets/yolo_pseudo_v2'
DS = 'datasets/yolo_pseudo_v3_union'

def iou(A,B):
    x0,y0=max(A[0],B[0]),max(A[1],B[1]); x1,y1=min(A[2],B[2]),min(A[3],B[3])
    if x1<=x0 or y1<=y0: return 0
    i=(x1-x0)*(y1-y0); return i/((A[2]-A[0])*(A[3]-A[1])+(B[2]-B[0])*(B[3]-B[1])-i)

def main(dev=1):
    t1, t2 = YOLO(T1), YOLO(T2)
    imgs = sorted(glob.glob(f'{SRC}/images/train/*'))
    log('v16: train imgs', len(imgs))
    for sp in ('images/train','labels/train'):
        os.makedirs(f'{DS}/{sp}', exist_ok=True)
    tot1, tot2, totu = 0, 0, 0
    for p in imgs:
        im = Image.open(p).convert('RGB'); W, H = im.size
        r1 = t1.predict(p, imgsz=1280, conf=0.5, iou=0.5, verbose=False, max_det=600, device=dev)[0]
        b1 = [] if r1.boxes is None else [(list(map(float,b)), float(c)) for b,c in zip(r1.boxes.xyxy.cpu().numpy(), r1.boxes.conf.cpu().numpy())]
        # 图例教师用其峰值配置, 阈值稍高保精度
        b2 = []
        for isz in (2880,):
            r2 = t2.predict(p, imgsz=isz, conf=0.45, iou=0.5, verbose=False, max_det=900, device=dev)[0]
            if r2.boxes is not None:
                b2 += [(list(map(float,b)), float(c)) for b,c in zip(r2.boxes.xyxy.cpu().numpy(), r2.boxes.conf.cpu().numpy())]
        # 并集: 教师1 全收; 教师2 只补教师1 没有的 (IoU<0.4)
        merged = [b for b,_ in b1]
        added = 0
        for b, c in sorted(b2, key=lambda t:-t[1]):
            if all(iou(b, m) < 0.4 for m in merged):
                merged.append(b); added += 1
        tot1 += len(b1); tot2 += len(b2); totu += len(merged)
        base = os.path.splitext(os.path.basename(p))[0]
        if not os.path.exists(f'{DS}/images/train/{base}.png'):
            os.symlink(os.path.abspath(p), f'{DS}/images/train/{base}' + os.path.splitext(p)[1])
        with open(f'{DS}/labels/train/{base}.txt','w') as fh:
            for b in merged:
                fh.write(f'0 {(b[0]+b[2])/2/W:.5f} {(b[1]+b[3])/2/H:.5f} {(b[2]-b[0])/W:.5f} {(b[3]-b[1])/H:.5f}\n')
    log(f'v16 pseudo-v3: teacher1 {tot1}, teacher2补 {totu-tot1}, 合计 {totu} (原伪v2=2793)')
    # val 沿用他们的(real4 GT 仅监控)
    with open(f'{DS}/data.yaml','w') as fh:
        srcy = open(f'{SRC}/data.yaml').read()
        val_line = [l for l in srcy.splitlines() if l.startswith('val')]
        fh.write('path: ' + os.path.abspath(DS) + '\ntrain: images/train\n' +
                 (val_line[0].replace('val: ', 'val: ' + os.path.abspath(SRC) + '/') if val_line and not val_line[0].split(':',1)[1].strip().startswith('/') else (val_line[0] if val_line else 'val: images/train')) + '\nnames:\n  0: item\n')
    m = YOLO(BASE)
    m.train(data=f'{DS}/data.yaml', epochs=50, imgsz=1280, batch=2, lr0=0.0005, lrf=0.05,
            cos_lr=True, warmup_epochs=2, patience=15, single_cls=True, amp=False,
            close_mosaic=10, mosaic=0.5, mixup=0.0, copy_paste=0.0,
            hsv_h=0.015, hsv_s=0.5, hsv_v=0.4, fliplr=0.5, flipud=0.05,
            translate=0.05, scale=0.3, degrees=3, shear=1, erasing=0.2,
            device=dev, project=f'{_D}/runs', name='v16_union', exist_ok=True, verbose=False, seed=0)
    log('v16 train done')
    sys.path.insert(0, 'gpt_detect/llm_baseline')
    import config, eval as ev
    for c in (f'runs/detect/{_D}/runs/v16_union/weights/best.pt', f'{_D}/runs/v16_union/weights/best.pt'):
        if not os.path.exists(c): continue
        mm = YOLO(c)
        for conf in (0.4, 0.5, 0.6):
            f1s = []
            for s0 in config.SAMPLES:
                gt = ev.load_gt_from_pnid_json(config.get_gt_path(s0))
                r = mm.predict(str(config.get_image_path(s0)), imgsz=1280, conf=conf, iou=0.5, verbose=False)[0]
                preds = [{"bbox": b.tolist(), "score": float(cc)} for b, cc in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())]
                f1s.append(ev.evaluate(preds, gt, iou_thresholds=[0.5])["by_iou"]["iou_0.5"]["F1"])
            log(f'v16 real4 conf{conf}: {sum(f1s)/4:.3f} {[round(v,3) for v in f1s]} (监督0.821)')
        break
    log('v16 end')

if __name__ == '__main__':
    try: main()
    except Exception: log('v16 FAIL\n' + traceback.format_exc())

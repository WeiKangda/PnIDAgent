"""v25: 高质量同族靶向合成 (v24 重生+质量修正)
- 符号源: sym600 高清 + 65页真实裁剪 + 同族兄弟页挖掘; 墨迹过滤+对比度增强
- 放置: 背景上探测真实管线 -> 符号嵌到线上(断线) + 逐符号位号
- 背景: 同族兄弟PDF 300dpi + organized 页
- 从 v16 温和微调 -> real4 评测(不含任何评测图素材)
"""
import os, sys, json, glob, random, time, traceback
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # -> project root
_D = 'unsupervised_symbol_recognition'
LOGF = open(f'{_D}/overnight_v10.log', 'a')
def log(*a):
    m = ' '.join(str(x) for x in a)
    print(m, flush=True); LOGF.write(time.strftime('[%H:%M:%S][v25] ') + m + '\n'); LOGF.flush()
# cv2/fitz 在 torch 之前导入会毁掉本机 CUDA 上下文创建 — torch 必须最先初始化
import torch
if torch.cuda.is_available(): torch.zeros(1, device='cuda')
from ultralytics import YOLO
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageOps
Image.MAX_IMAGE_PIXELS = None
random.seed(25); np.random.seed(25)
import fitz

BAN = ('CVCS - APR1400', 'CVCS - (Surry)', 'CVCS2 - (Surry)', 'AFW - (Surry)', 'North ANA', 'NorthANA', 'Legend')
W = 1280

def iou_(A,B):
    xx0,yy0=max(A[0],B[0]),max(A[1],B[1]); xx1,yy1=min(A[2],B[2]),min(A[3],B[3])
    if xx1<=xx0 or yy1<=yy0: return 0
    ii=(xx1-xx0)*(yy1-yy0); return ii/((A[2]-A[0])*(A[3]-A[1])+(B[2]-B[0])*(B[3]-B[1])-ii)

def enhance(sym):
    """高清符号预处理: 自动对比度 + 墨迹过滤"""
    g = ImageOps.autocontrast(sym.convert('L'), cutoff=1)
    a = np.array(g)
    ink = (a < 160).mean()
    if ink < 0.02:      # 太淡/太空
        return None
    return g

def find_lines(a_bin):
    """背景中的真实管线段: 返回 [(orient, fixed, lo, hi)]"""
    segs = []
    h = cv2.morphologyEx(a_bin, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(60,1)))
    v = cv2.morphologyEx(a_bin, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(1,60)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(h, 8)
    for i in range(1, n):
        x,y,w,hh,area = st[i]
        if w >= 120 and hh <= 6: segs.append(('h', y+hh//2, x, x+w))
    n, lab, st, _ = cv2.connectedComponentsWithStats(v, 8)
    for i in range(1, n):
        x,y,w,hh,area = st[i]
        if hh >= 120 and w <= 6: segs.append(('v', x+w//2, y, y+hh))
    return segs

def main():
    T = f'runs/detect/{_D}/runs/v16_union/weights/best.pt'
    det = YOLO(T)
    import torch
    for attempt in range(8):
        try:
            det.predict(Image.new('RGB', (64, 64), 'white'), imgsz=64, verbose=False, device=0)
            log('v25 CUDA warmup ok')
            break
        except Exception as e:
            log('v25 warmup retry', attempt, str(e)[:80])
            torch.cuda.empty_cache(); time.sleep(30)
    else:
        raise RuntimeError('GPU context unavailable after 8 retries')
    # ---- 符号池 (三源, 全部 enhance 过滤) ----
    pool = []
    L2 = 'PID_merged/Symbol - Legend/extracted'
    for jf in glob.glob(f'{L2}/*_p*.json'):
        j = json.load(open(jf))
        for e in j['entries']:
            for key in ('sym_file','sym_file2'):
                sf = e.get(key)
                if not sf: continue
                p6 = f"{L2}/sym600/{os.path.basename(sf)}"
                if os.path.exists(p6):
                    im0 = Image.open(p6)
                    if 12 < im0.width < 900 and 12 < im0.height < 900:
                        g = enhance(im0)
                        if g is not None: pool.append(g)
    n_leg = len(pool)
    for f in glob.glob(f'{_D}/realcrop_pool/*.png'):
        g = enhance(Image.open(f))
        if g is not None: pool.append(g)
    log(f'v25 pool: legend600 {n_leg} + realcrop {len(pool)-n_leg} = {len(pool)}')
    # ---- 背景: 同族PDF 300dpi + organized ----
    bgs = []
    fam = [p for p in glob.glob('All PID Diagram/*.pdf') if not any(b in p for b in BAN)]
    for p in fam:
        try:
            d = fitz.open(p)
            for pi in range(min(2, len(d))):
                pix = d[pi].get_pixmap(dpi=300)
                im = Image.frombytes('RGB', (pix.width, pix.height), pix.samples).convert('L')
                if im.width >= 1600: bgs.append(im)
            d.close()
        except Exception: pass
    for p in glob.glob('PID_merged_organized/png/*/*.png'):
        if any(b in p for b in BAN): continue
        try:
            im = Image.open(p).convert('L')
            if im.width >= 1400: bgs.append(im)
        except Exception: pass
    log('v25 backgrounds:', len(bgs))
    # ---- 合成 ----
    DS = 'datasets/yolo_quality'
    for sp in ('images/train','labels/train','images/val','labels/val'):
        os.makedirs(f'{DS}/{sp}', exist_ok=True)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
    made = 0
    while made < 1600:
        full = random.choice(bgs)
        if full.width <= 1600 or full.height <= 1600:
            tile = full.resize((W, W), Image.LANCZOS)
        else:
            x0 = random.randint(0, full.width-1600); y0 = random.randint(0, full.height-1600)
            tile = full.crop((x0, y0, x0+1600, y0+1600)).resize((W, W), Image.LANCZOS)
        a_bin = (np.array(tile) < 190).astype(np.uint8)
        if a_bin.mean() < 0.004: continue
        segs = find_lines(a_bin)
        if len(segs) < 3: continue
        # 背景伪标签(带质量过滤: 踢黑实心块与超扁框)
        r = det.predict(tile.convert('RGB'), imgsz=1280, conf=0.45, verbose=False, max_det=300, device=0)[0]
        boxes = []
        if r.boxes is not None:
            for b in r.boxes.xyxy.cpu().numpy():
                x0b,y0b,x1b,y1b = map(int, b)
                w0, h0 = x1b-x0b, y1b-y0b
                if w0 < 12 or h0 < 12: continue
                if w0/h0 > 6 or h0/w0 > 6: continue
                win = np.array(tile.crop((x0b,y0b,x1b,y1b)))
                if (win < 100).mean() > 0.55: continue    # 黑实心块(涂改/齿孔) 踢
                boxes.append([x0b,y0b,x1b,y1b])
        d = ImageDraw.Draw(tile)
        added, tries = 0, 0
        target = random.randint(6, 14)
        while added < target and tries < 120:
            tries += 1
            sym = random.choice(pool)
            tgt = random.uniform(26, 48)
            s2 = tgt / max(sym.size)
            sym2 = sym.resize((max(10,int(sym.width*s2)), max(10,int(sym.height*s2))), Image.LANCZOS)
            if random.random() < .4: sym2 = sym2.rotate(90*random.choice([1,2,3]), expand=True, fillcolor=255)
            w2, h2 = sym2.size
            # 80% 嵌到真实管线上
            if segs and random.random() < .8:
                o, fixed, lo, hi = random.choice(segs)
                if hi - lo < w2 + 20: continue
                c = random.randint(lo+w2//2+6, hi-w2//2-6)
                if o == 'h': px, py = c-w2//2, fixed-h2//2
                else: px, py = fixed-w2//2, c-h2//2
            else:
                px, py = random.randint(0, W-w2), random.randint(0, W-h2)
            if px < 0 or py < 0 or px+w2 > W or py+h2 > W: continue
            nb = [px, py, px+w2, py+h2]
            if any(iou_(nb, b) > 0.05 for b in boxes): continue
            # 断线嵌入: 符号心区先抹白(留4px接口)
            d.rectangle((px+4, py+4, px+w2-4, py+h2-4), fill=255)
            tile.paste(Image.fromarray(np.minimum(np.array(tile.crop(tuple(nb))), np.array(sym2))), (px, py))
            boxes.append(nb); added += 1
            if random.random() < .6:
                t = random.choice(['XV{}','CV-{}','MOV-{}','V{}A']).format(random.randint(1,999))
                ty = py + h2 + 2 if random.random() < .7 else py - 15
                if 0 <= ty < W-14: d.text((max(0,px), ty), t, fill=0, font=font)
        if added < 4: continue
        a = np.array(tile).astype(np.float32)
        if random.random() < .3: a = cv2.GaussianBlur(a, (3,3), random.uniform(0.3,0.8))
        if random.random() < .25: a = 255 - (255-a)*random.uniform(0.5, 1.0)
        tile = Image.fromarray(np.clip(a + np.random.randn(W,W)*random.uniform(0,4),0,255).astype(np.uint8))
        split = 'val' if made % 12 == 11 else 'train'
        tile.convert('RGB').save(f'{DS}/images/{split}/q{made:05d}.png')
        with open(f'{DS}/labels/{split}/q{made:05d}.txt','w') as fh:
            for b in boxes:
                fh.write(f'0 {(b[0]+b[2])/2/W:.5f} {(b[1]+b[3])/2/W:.5f} {(b[2]-b[0])/W:.5f} {(b[3]-b[1])/W:.5f}\n')
        made += 1
        if made % 200 == 0: log('v25 tiles', made)
    log('v25 data done', made)
    with open(f'{DS}/data.yaml','w') as fh:
        fh.write(f'path: {os.path.abspath(DS)}\ntrain: images/train\nval: images/val\nnames:\n  0: item\n')
    m = YOLO(T)
    m.train(data=f'{DS}/data.yaml', imgsz=1280, epochs=12, batch=4, device=0,
            project=f'{_D}/runs', name='v25_quality', exist_ok=True, verbose=False,
            optimizer='SGD', lr0=0.0004, cos_lr=True, patience=6, single_cls=True,
            mosaic=0.3, close_mosaic=3, scale=0.25, degrees=2, workers=8)
    log('v25 train done')
    sys.path.insert(0, 'gpt_detect/llm_baseline')
    import config, eval as ev
    for c in (f'runs/detect/{_D}/runs/v25_quality/weights/best.pt', f'{_D}/runs/v25_quality/weights/best.pt'):
        if os.path.exists(c):
            mm = YOLO(c)
            for conf in (0.45, 0.5, 0.55):
                f1s = []
                for s0 in config.SAMPLES:
                    gt = ev.load_gt_from_pnid_json(config.get_gt_path(s0))
                    r = mm.predict(str(config.get_image_path(s0)), imgsz=1280, conf=conf, iou=0.5, verbose=False)[0]
                    preds = [{"bbox": b.tolist(), "score": float(cc)} for b, cc in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())]
                    f1s.append(ev.evaluate(preds, gt, iou_thresholds=[0.5])["by_iou"]["iou_0.5"]["F1"])
                log(f'v25 real4 conf{conf}: {sum(f1s)/4:.3f} {[round(v,3) for v in f1s]} (v16=0.855, 看CVCS两位)')
            break
    log('v25 end')

if __name__ == '__main__':
    try: main()
    except Exception: log('v25 FAIL\n' + traceback.format_exc())

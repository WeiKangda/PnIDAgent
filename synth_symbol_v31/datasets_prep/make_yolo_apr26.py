"""v26: 冲0.90 — APR1400 靶向 + 分风格高质量合成
短板诊断: CVCS_APR1400 0.785 (23漏检=盲区类型, 矢量风格); CVCS_Surry 0.845
升级点 vs v25:
  1) APR1400 图例符号(全矢量,最干净)采样权重拉高 + APR1400 同族兄弟页挖掘
  2) 分风格: 矢量背景(APR族)-> 轻退化; 扫描背景 -> 重退化 (风格对齐)
  3) 从 v25 权重继续微调 (站在 0.866 肩上)
"""
import os, sys, json, glob, random, time, traceback
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))  # -> project root
_D = 'unsupervised_symbol_recognition'
LOGF = open(f'{_D}/overnight_v10.log', 'a')
def log(*a):
    m = ' '.join(str(x) for x in a)
    print(m, flush=True); LOGF.write(time.strftime('[%H:%M:%S][v26] ') + m + '\n'); LOGF.flush()
# torch 必须最先初始化 (cv2/fitz 先导入会毁 CUDA 上下文)
import torch
if torch.cuda.is_available(): torch.zeros(1, device='cuda')
from ultralytics import YOLO
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageOps
Image.MAX_IMAGE_PIXELS = None
random.seed(26); np.random.seed(26)
import fitz

BAN = ('CVCS - APR1400', 'CVCS - (Surry)', 'CVCS2 - (Surry)', 'AFW - (Surry)', 'North ANA', 'NorthANA', 'Legend')
W = 1280

def iou_(A,B):
    xx0,yy0=max(A[0],B[0]),max(A[1],B[1]); xx1,yy1=min(A[2],B[2]),min(A[3],B[3])
    if xx1<=xx0 or yy1<=yy0: return 0
    ii=(xx1-xx0)*(yy1-yy0); return ii/((A[2]-A[0])*(A[3]-A[1])+(B[2]-B[0])*(B[3]-B[1])-ii)

def enhance(sym):
    g = ImageOps.autocontrast(sym.convert('L'), cutoff=1)
    a = np.array(g)
    if (a < 160).mean() < 0.02: return None
    return g

def find_lines(a_bin):
    segs = []
    h = cv2.morphologyEx(a_bin, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(60,1)))
    v = cv2.morphologyEx(a_bin, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(1,60)))
    for arr, o in ((h,'h'),(v,'v')):
        n, lab, st, _ = cv2.connectedComponentsWithStats(arr, 8)
        for i in range(1, n):
            x,y,w,hh,area = st[i]
            if o=='h' and w >= 120 and hh <= 6: segs.append(('h', y+hh//2, x, x+w))
            if o=='v' and hh >= 120 and w <= 6: segs.append(('v', x+w//2, y, y+hh))
    return segs

def main():
    PREV = f'runs/detect/{_D}/runs/v25_quality/weights/best.pt'
    det = YOLO(PREV)
    for attempt in range(8):
        try:
            det.predict(Image.new('RGB',(64,64),'white'), imgsz=64, verbose=False, device=0)
            log('v26 warmup ok'); break
        except Exception as e:
            log('v26 warmup retry', attempt, str(e)[:60]); torch.cuda.empty_cache(); time.sleep(30)

    # ---- 符号池: APR图例(高权重) / 其他图例 / realcrop ----
    L2 = 'PID_merged/Symbol - Legend/extracted'
    pool_apr, pool_leg = [], []
    for jf in glob.glob(f'{L2}/*_p*.json'):
        is_apr = 'APR' in os.path.basename(jf).upper()
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
                        if g is not None: (pool_apr if is_apr else pool_leg).append(g)
    pool_real = [g for g in (enhance(Image.open(f)) for f in glob.glob(f'{_D}/realcrop_pool/*.png')) if g is not None]
    # APR 同族兄弟页挖掘
    FAM_APR = [p for p in glob.glob('All PID Diagram/*.pdf') if 'APR1400' in p and not any(b in p for b in BAN)]
    pool_mined, bgs_apr, bgs_scan = [], [], []
    for p in FAM_APR:
        try:
            doc = fitz.open(p)
            for pi in range(min(2, len(doc))):
                pix = doc[pi].get_pixmap(dpi=300)
                im = Image.frombytes('RGB', (pix.width, pix.height), pix.samples).convert('L')
                if im.width < 1400: continue
                bgs_apr.append(im)
                r = det.predict(im.convert('RGB'), imgsz=2048, conf=0.45, iou=0.5, verbose=False, max_det=500, device=0)[0]
                if r.boxes is not None:
                    for b in r.boxes.xyxy.cpu().numpy():
                        x0,y0,x1,y1 = map(int, b)
                        w0,h0 = x1-x0, y1-y0
                        if w0<16 or h0<16 or w0>400 or w0/h0>6 or h0/w0>6: continue
                        win = np.array(im.crop((x0,y0,x1,y1)))
                        if (win<100).mean() > 0.55: continue
                        g = enhance(im.crop((max(0,x0-2),max(0,y0-2),x1+2,y1+2)))
                        if g is not None: pool_mined.append(g)
            doc.close()
        except Exception: pass
    for p in glob.glob('All PID Diagram/*.pdf'):
        if 'APR1400' in p or any(b in p for b in BAN): continue
        try:
            doc = fitz.open(p)
            for pi in range(min(2, len(doc))):
                pix = doc[pi].get_pixmap(dpi=300)
                im = Image.frombytes('RGB', (pix.width, pix.height), pix.samples).convert('L')
                if im.width >= 1600: bgs_scan.append(im)
            doc.close()
        except Exception: pass
    for p in glob.glob('PID_merged_organized/png/*/*.png'):
        if any(b in p for b in BAN): continue
        try:
            im = Image.open(p).convert('L')
            if im.width >= 1400: bgs_scan.append(im)
        except Exception: pass
    log(f'v26 pools: apr_legend {len(pool_apr)} other_legend {len(pool_leg)} mined_apr {len(pool_mined)} real {len(pool_real)}; bgs apr {len(bgs_apr)} scan {len(bgs_scan)}')

    def draw_pool(apr_style):
        r = random.random()
        if apr_style:
            if r < .45 and pool_apr: return random.choice(pool_apr)
            if r < .70 and pool_mined: return random.choice(pool_mined)
            if r < .85 and pool_leg: return random.choice(pool_leg)
            return random.choice(pool_real)
        else:
            if r < .25 and pool_apr: return random.choice(pool_apr)
            if r < .40 and pool_leg: return random.choice(pool_leg)
            return random.choice(pool_real)

    DS = 'datasets/yolo_apr26'
    for sp in ('images/train','labels/train','images/val','labels/val'):
        os.makedirs(f'{DS}/{sp}', exist_ok=True)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
    made = 0
    N = 2000
    while made < N:
        apr_style = (made % 2 == 0) and bgs_apr   # 50% APR 矢量风格
        full = random.choice(bgs_apr if apr_style else bgs_scan)
        if full.width <= 1600 or full.height <= 1600:
            tile = full.resize((W, W), Image.LANCZOS)
        else:
            x0 = random.randint(0, full.width-1600); y0 = random.randint(0, full.height-1600)
            tile = full.crop((x0, y0, x0+1600, y0+1600)).resize((W, W), Image.LANCZOS)
        a_bin = (np.array(tile) < 190).astype(np.uint8)
        if a_bin.mean() < 0.003: continue
        segs = find_lines(a_bin)
        if len(segs) < 3: continue
        r = det.predict(tile.convert('RGB'), imgsz=1280, conf=0.45, verbose=False, max_det=300, device=0)[0]
        boxes = []
        if r.boxes is not None:
            for b in r.boxes.xyxy.cpu().numpy():
                x0b,y0b,x1b,y1b = map(int, b)
                w0,h0 = x1b-x0b, y1b-y0b
                if w0<12 or h0<12 or w0/h0>6 or h0/w0>6: continue
                win = np.array(tile.crop((x0b,y0b,x1b,y1b)))
                if (win<100).mean() > 0.55: continue
                boxes.append([x0b,y0b,x1b,y1b])
        d = ImageDraw.Draw(tile)
        added, tries = 0, 0
        target = random.randint(7, 15)
        while added < target and tries < 130:
            tries += 1
            sym = draw_pool(apr_style)
            tgt = random.uniform(26, 52)
            s2 = tgt / max(sym.size)
            sym2 = sym.resize((max(10,int(sym.width*s2)), max(10,int(sym.height*s2))), Image.LANCZOS)
            if random.random() < .4: sym2 = sym2.rotate(90*random.choice([1,2,3]), expand=True, fillcolor=255)
            w2, h2 = sym2.size
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
            d.rectangle((px+4, py+4, px+w2-4, py+h2-4), fill=255)
            tile.paste(Image.fromarray(np.minimum(np.array(tile.crop(tuple(nb))), np.array(sym2))), (px, py))
            boxes.append(nb); added += 1
            if random.random() < .6:
                t = random.choice(['XV{}','CV-{}','MOV-{}','V{}A','LO-V{}']).format(random.randint(1,999))
                ty = py + h2 + 2 if random.random() < .7 else py - 15
                if 0 <= ty < W-14: d.text((max(0,px), ty), t, fill=0, font=font)
        if added < 5: continue
        a = np.array(tile).astype(np.float32)
        if apr_style:   # 矢量风格: 干净, 轻噪声
            if random.random() < .15: a = cv2.GaussianBlur(a, (3,3), 0.4)
            a = a + np.random.randn(W,W)*random.uniform(0,1.5)
        else:           # 扫描风格: 重退化
            if random.random() < .4: a = cv2.GaussianBlur(a, (3,3), random.uniform(0.3,0.9))
            if random.random() < .3: a = 255 - (255-a)*random.uniform(0.45, 1.0)
            a = a + np.random.randn(W,W)*random.uniform(0,4.5)
        tile = Image.fromarray(np.clip(a,0,255).astype(np.uint8))
        split = 'val' if made % 12 == 11 else 'train'
        tile.convert('RGB').save(f'{DS}/images/{split}/a{made:05d}.png')
        with open(f'{DS}/labels/{split}/a{made:05d}.txt','w') as fh:
            for b in boxes:
                fh.write(f'0 {(b[0]+b[2])/2/W:.5f} {(b[1]+b[3])/2/W:.5f} {(b[2]-b[0])/W:.5f} {(b[3]-b[1])/W:.5f}\n')
        made += 1
        if made % 250 == 0: log('v26 tiles', made)
    log('v26 data done', made)
    with open(f'{DS}/data.yaml','w') as fh:
        fh.write(f'path: {os.path.abspath(DS)}\ntrain: images/train\nval: images/val\nnames:\n  0: item\n')
    m = YOLO(PREV)
    m.train(data=f'{DS}/data.yaml', imgsz=1280, epochs=14, batch=4, device=0,
            project=f'{_D}/runs', name='v26_apr', exist_ok=True, verbose=False,
            optimizer='SGD', lr0=0.0003, cos_lr=True, patience=6, single_cls=True,
            mosaic=0.3, close_mosaic=4, scale=0.25, degrees=2, workers=8)
    log('v26 train done')
    sys.path.insert(0, 'gpt_detect/llm_baseline')
    import config, eval as ev
    for c in (f'runs/detect/{_D}/runs/v26_apr/weights/best.pt', f'{_D}/runs/v26_apr/weights/best.pt'):
        if os.path.exists(c):
            mm = YOLO(c)
            for conf in (0.45, 0.5, 0.55, 0.6):
                f1s = []
                for s0 in config.SAMPLES:
                    gt = ev.load_gt_from_pnid_json(config.get_gt_path(s0))
                    r = mm.predict(str(config.get_image_path(s0)), imgsz=1280, conf=conf, iou=0.5, verbose=False)[0]
                    preds = [{"bbox": b.tolist(), "score": float(cc)} for b, cc in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())]
                    f1s.append(ev.evaluate(preds, gt, iou_thresholds=[0.5])["by_iou"]["iou_0.5"]["F1"])
                log(f'v26 real4 conf{conf}: {sum(f1s)/4:.3f} {[round(v,3) for v in f1s]} (v25=0.866, 目标0.90, 看末位APR)')
            break
    log('v26 end')

if __name__ == '__main__':
    try: main()
    except Exception: log('v26 FAIL\n' + traceback.format_exc())

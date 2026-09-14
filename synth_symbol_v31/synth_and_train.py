"""v31: 真实感合成 — 治"贴上去一眼假"
1) 笔画归一: 符号二值化 + 按背景线宽膨胀 -> 纯黑, 和背景同粗细
2) 退化同炉: 符号跟整页一起退化 (不再先灰后贴)
3) 纸色填充: 断线抹白用本页纸色, 不留白补丁
4) 接口补桩: 贴完补画两端线桩接回管线
5) 方向对齐: 横线横贴 / 竖线转90°; 仅浮空符号随机转
生成 datasets/yolo_real31 1600页 -> 从 v28b 联合微调 -> real4
"""
import os, sys, json, glob, random, time, traceback
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # -> project root
_D = 'unsupervised_symbol_recognition'
LOGF = open(f'{_D}/overnight_v10.log', 'a')
def log(*a):
    m = ' '.join(str(x) for x in a)
    print(m, flush=True); LOGF.write(time.strftime('[%H:%M:%S][v31] ') + m + '\n'); LOGF.flush()
import torch
if torch.cuda.is_available(): torch.zeros(1, device='cuda')
from ultralytics import YOLO
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageOps
Image.MAX_IMAGE_PIXELS = None
random.seed(31); np.random.seed(31)
import fitz

BAN = ('CVCS - APR1400', 'CVCS - (Surry)', 'CVCS2 - (Surry)', 'AFW - (Surry)', 'North ANA', 'NorthANA', 'Legend')
W = 1280

def iou_(A,B):
    xx0,yy0=max(A[0],B[0]),max(A[1],B[1]); xx1,yy1=min(A[2],B[2]),min(A[3],B[3])
    if xx1<=xx0 or yy1<=yy0: return 0
    ii=(xx1-xx0)*(yy1-yy0); return ii/((A[2]-A[0])*(A[3]-A[1])+(B[2]-B[0])*(B[3]-B[1])-ii)

def enhance(sym):
    g = ImageOps.autocontrast(sym.convert('L'), cutoff=1)
    ink = (np.array(g) < 160).mean()
    if ink < 0.02 or ink > 0.45: return None
    # 结构过滤: 字母/碎片 = 单连通域 + 无孔洞 + 笔画简单 -> 踢
    s64 = np.array(g.resize((64, 64), Image.LANCZOS))
    m = (s64 < 180).astype(np.uint8)
    ncomp = cv2.connectedComponents(m, 8)[0] - 1
    inv = (1 - m).astype(np.uint8)
    nhole = max(0, cv2.connectedComponents(inv, 4)[0] - 2)   # 背景扣掉外圈
    if ncomp <= 1 and nhole == 0:
        return None
    # 文字行过滤: 2-4 个等高组件横排 = 字母串 -> 踢
    if 2 <= ncomp <= 4:
        n2, _, st2, _ = cv2.connectedComponentsWithStats(m, 8)
        cs = [st2[i] for i in range(1, n2) if st2[i][4] > 8]
        if len(cs) >= 2:
            hs = [c[3] for c in cs]
            ys = [c[1] for c in cs]
            if max(hs) and min(hs)/max(hs) > 0.6 and (max(ys)-min(ys)) < 0.3*max(hs):
                xs = sorted((c[0], c[0]+c[2]) for c in cs)
                overlap = any(xs[i+1][0] < xs[i][1] - 3 for i in range(len(xs)-1))
                if not overlap:
                    return None
    return g

def find_lines_w(a_bin):
    """返回线段 + 背景线宽估计"""
    segs, widths = [], []
    h = cv2.morphologyEx(a_bin, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(60,1)))
    v = cv2.morphologyEx(a_bin, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(1,60)))
    for arr, o in ((h,'h'),(v,'v')):
        n, lab, st, _ = cv2.connectedComponentsWithStats(arr, 8)
        for i in range(1, n):
            x,y,w,hh,area = st[i]
            if o=='h' and w>=120 and hh<=6: segs.append(('h', y+hh//2, x, x+w)); widths.append(hh)
            if o=='v' and hh>=120 and w<=6: segs.append(('v', x+w//2, y, y+hh)); widths.append(w)
    lw = int(np.median(widths)) if widths else 2
    return segs, max(1, min(4, lw))

def render_symbol(sym, w2, h2, bg_lw):
    """符号 -> 与背景同笔画粗细的纯黑二值图"""
    s = sym.resize((w2, h2), Image.LANCZOS)
    m = (np.array(s) < 200).astype(np.uint8)
    # 600DPI 缩到 30px 后笔画≈1px; 膨胀到背景线宽
    it = max(0, bg_lw - 1)
    if it:
        m2 = cv2.dilate(m, np.ones((3,3),np.uint8), iterations=min(it,2))
        if m2.mean() <= 0.32: m = m2          # 膨胀限幅: 防墨疙瘩
    out = np.full((h2, w2), 255, np.uint8)
    out[m > 0] = 0
    return Image.fromarray(out), m

def main():
    det = YOLO(f'runs/detect/{_D}/runs/v28b_rebalance/weights/best.pt')
    for attempt in range(6):
        try:
            det.predict(Image.new('RGB',(64,64),'white'), imgsz=64, verbose=False, device=0)
            log('v31 warmup ok'); break
        except Exception as e:
            log('v31 warmup retry', str(e)[:50]); torch.cuda.empty_cache(); time.sleep(20)

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
    log(f'v31 pools: apr {len(pool_apr)} leg {len(pool_leg)} real {len(pool_real)}')

    bgs_apr, bgs_scan = [], []
    for p in glob.glob('All PID Diagram/*.pdf'):
        if any(b in p for b in BAN): continue
        try:
            doc = fitz.open(p)
            for pi in range(min(2, len(doc))):
                pix = doc[pi].get_pixmap(dpi=300)
                im = Image.frombytes('RGB', (pix.width, pix.height), pix.samples).convert('L')
                if im.width >= 1400: (bgs_apr if 'APR1400' in p else bgs_scan).append(im)
            doc.close()
        except Exception: pass
    for p in glob.glob('PID_merged_organized/png/*/*.png'):
        if any(b in p for b in BAN): continue
        try:
            im = Image.open(p).convert('L')
            if im.width >= 1400: bgs_scan.append(im)
        except Exception: pass
    log(f'v31 bgs: apr {len(bgs_apr)} scan {len(bgs_scan)}')

    def draw_pool(apr_style):
        r = random.random()
        if apr_style:
            if r < .40 and pool_apr: return random.choice(pool_apr)
            if r < .55 and pool_leg: return random.choice(pool_leg)
            return random.choice(pool_real)
        else:
            if r < .20 and pool_leg: return random.choice(pool_leg)
            return random.choice(pool_real)

    DS = 'datasets/yolo_real31'
    for sp in ('images/train','labels/train','images/val','labels/val'):
        os.makedirs(f'{DS}/{sp}', exist_ok=True)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
    made = 0
    while made < 1600:
        apr_style = (made % 2 == 0) and bgs_apr
        full = random.choice(bgs_apr if apr_style else bgs_scan)
        if full.width <= 1600 or full.height <= 1600:
            tile = full.resize((W, W), Image.LANCZOS)
        else:
            x0 = random.randint(0, full.width-1600); y0 = random.randint(0, full.height-1600)
            tile = full.crop((x0, y0, x0+1600, y0+1600)).resize((W, W), Image.LANCZOS)
        ta = np.array(tile)
        a_bin = (ta < 190).astype(np.uint8)
        if a_bin.mean() < 0.003: continue
        segs, bg_lw = find_lines_w(a_bin)
        if len(segs) < 3: continue
        paper = int(np.percentile(ta, 90))          # 本页纸色
        r = det.predict(tile.convert('RGB'), imgsz=1280, conf=0.45, verbose=False, max_det=300, device=0)[0]
        boxes = []
        if r.boxes is not None:
            for b in r.boxes.xyxy.cpu().numpy():
                x0b,y0b,x1b,y1b = map(int, b)
                w0,h0 = x1b-x0b, y1b-y0b
                if w0<12 or h0<12 or w0/h0>6 or h0/w0>6: continue
                win = ta[y0b:y1b, x0b:x1b]
                if win.size and (win<100).mean() > 0.55: continue
                boxes.append([x0b,y0b,x1b,y1b])
        d = ImageDraw.Draw(tile)
        added, tries = 0, 0
        target = random.randint(7, 14)
        while added < target and tries < 130:
            tries += 1
            sym = draw_pool(apr_style)
            tgt = random.uniform(26, 52)
            s2 = tgt / max(sym.size)
            w2, h2 = max(10,int(sym.width*s2)), max(10,int(sym.height*s2))
            on_line = segs and random.random() < .95
            if on_line:
                o, fixed, lo, hi = random.choice(segs)
                # 方向对齐: 符号长轴顺着线
                if o == 'v' and w2 > h2:
                    sym_r = sym.rotate(90, expand=True, fillcolor=255); w2, h2 = h2, w2
                elif o == 'h' and h2 > w2 and random.random() < .7:
                    sym_r = sym.rotate(90, expand=True, fillcolor=255); w2, h2 = h2, w2
                else:
                    sym_r = sym
                span = w2 if o=='h' else h2
                if hi - lo < span + 24: continue
                c = random.randint(lo+span//2+8, hi-span//2-8)
                if o=='h': px, py = c-w2//2, fixed-h2//2
                else: px, py = fixed-w2//2, c-h2//2
            else:
                sym_r = sym.rotate(90*random.choice([0,1,2,3]), expand=True, fillcolor=255)
                if sym_r.width != sym.width: w2, h2 = h2, w2
                px, py = random.randint(0, W-w2), random.randint(0, W-h2)
            if px<0 or py<0 or px+w2>W or py+h2>W: continue
            nb = [px,py,px+w2,py+h2]
            if any(iou_(nb,b)>0.05 for b in boxes): continue
            symbw, _m = render_symbol(sym_r, w2, h2, bg_lw)
            # 纸色抹底 (不再留白补丁)
            d.rectangle((px+2, py+2, px+w2-2, py+h2-2), fill=paper)
            tile.paste(Image.fromarray(np.minimum(np.array(tile.crop(tuple(nb))), np.array(symbw))), (px, py))
            if on_line:   # 接口补桩: 两端短线接回管线
                if o=='h':
                    cy = py + h2//2
                    d.line([(px-2, cy), (px+max(3,w2//6), cy)], fill=0, width=bg_lw)
                    d.line([(px+w2-max(3,w2//6), cy), (px+w2+2, cy)], fill=0, width=bg_lw)
                else:
                    cx = px + w2//2
                    d.line([(cx, py-2), (cx, py+max(3,h2//6))], fill=0, width=bg_lw)
                    d.line([(cx, py+h2-max(3,h2//6)), (cx, py+h2+2)], fill=0, width=bg_lw)
            boxes.append(nb); added += 1
            if random.random() < .6:
                t = random.choice(['XV{}','CV-{}','MOV-{}','V{}A']).format(random.randint(1,999))
                if on_line and o=='v':
                    tx, ty = px+w2+3, py + h2//2 - 6
                else:
                    tx, ty = max(0,px), (py+h2+2 if random.random() < .7 else py-15)
                if 0 <= ty < W-14 and tx < W-60: d.text((tx, ty), t, fill=0, font=font)
        if added < 5: continue
        a = np.array(tile).astype(np.float32)
        if apr_style:
            if random.random() < .15: a = cv2.GaussianBlur(a,(3,3),0.4)
            a = a + np.random.randn(W,W)*random.uniform(0,1.5)
        else:
            if random.random() < .45: a = cv2.GaussianBlur(a,(3,3),random.uniform(0.4,1.0))
            if random.random() < .3: a = 255-(255-a)*random.uniform(0.5,1.0)
            a = a + np.random.randn(W,W)*random.uniform(0,4.5)
        tile = Image.fromarray(np.clip(a,0,255).astype(np.uint8))
        split = 'val' if made % 12 == 11 else 'train'
        tile.convert('RGB').save(f'{DS}/images/{split}/r{made:05d}.png')
        with open(f'{DS}/labels/{split}/r{made:05d}.txt','w') as fh:
            for b in boxes:
                fh.write(f'0 {(b[0]+b[2])/2/W:.5f} {(b[1]+b[3])/2/W:.5f} {(b[2]-b[0])/W:.5f} {(b[3]-b[1])/W:.5f}\n')
        made += 1
        if made == 12: log('v31 first-12 saved (可抽查)')
        if made % 250 == 0: log('v31 tiles', made)
    log('v31 data done', made)

    R = os.path.abspath('datasets')
    yaml = f"""train:
  - {R}/yolo_real31/images/train
  - {R}/yolo_quality/images/train
  - {R}/yolo_apr26/images/train
  - {R}/yolo_dpid_nuke/images/train
  - {R}/yolo_pseudo_v3_union/images/train
val:
  - {R}/yolo_real31/images/val
  - {R}/yolo_quality/images/val
names:
  0: item
"""
    os.makedirs('datasets/yolo_joint31', exist_ok=True)
    with open('datasets/yolo_joint31/data.yaml','w') as fh: fh.write(yaml)
    m = YOLO(f'runs/detect/{_D}/runs/v28b_rebalance/weights/best.pt')
    m.train(data='datasets/yolo_joint31/data.yaml', imgsz=1280, epochs=12, batch=4, device=0,
            project=f'{_D}/runs', name='v31_realism', exist_ok=True, verbose=False,
            optimizer='SGD', lr0=0.0004, cos_lr=True, patience=5, single_cls=True,
            mosaic=0.3, close_mosaic=3, scale=0.25, degrees=2, workers=8)
    log('v31 train done')
    sys.path.insert(0, 'gpt_detect/llm_baseline')
    import config, eval as ev
    for c in (f'runs/detect/{_D}/runs/v31_realism/weights/best.pt', f'{_D}/runs/v31_realism/weights/best.pt'):
        if os.path.exists(c):
            mm = YOLO(c)
            for conf in (0.5, 0.55, 0.6, 0.65):
                f1s = []
                for s0 in config.SAMPLES:
                    gt = ev.load_gt_from_pnid_json(config.get_gt_path(s0))
                    rr = mm.predict(str(config.get_image_path(s0)), imgsz=1280, conf=conf, iou=0.5, verbose=False)[0]
                    preds = [{"bbox": b.tolist(), "score": float(cc)} for b, cc in zip(rr.boxes.xyxy.cpu().numpy(), rr.boxes.conf.cpu().numpy())]
                    f1s.append(ev.evaluate(preds, gt, iou_thresholds=[0.5])["by_iou"]["iou_0.5"]["F1"])
                log(f'v31 real4 conf{conf}: {sum(f1s)/4:.3f} {[round(v,3) for v in f1s]} (v28b=0.877, 目标0.90)')
            break
    log('v31 end')

if __name__ == '__main__':
    try: main()
    except Exception: log('v31 FAIL\n' + traceback.format_exc())

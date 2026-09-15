"""v27gen: dpid 式程序化整页合成 — 核电版 (纯CPU, 白纸生成, 标签天然精确)
参考 Dataset-P&ID (Paliwal et al., arXiv:2109.03794) 的版面语法:
正交管线网 + 跳线弧 + 结点圆点 + 虚线备用线 + 内嵌符号断线 + 位号语法 + 仪表圈 + NOTES/标题块 + 灰底噪声
符号源 = 我们的 962 库 (sym600 高清) + realcrop; 尺寸 26-52px 甜区 (px-law)
"""
import os, sys, json, glob, random, time, traceback
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))  # -> project root
_D = 'unsupervised_symbol_recognition'
LOGF = open(f'{_D}/overnight_v10.log', 'a')
def log(*a):
    m = ' '.join(str(x) for x in a)
    print(m, flush=True); LOGF.write(time.strftime('[%H:%M:%S][v27gen] ') + m + '\n'); LOGF.flush()
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
Image.MAX_IMAGE_PIXELS = None
random.seed(27); np.random.seed(27)

W = 1280
def enhance(sym):
    g = ImageOps.autocontrast(sym.convert('L'), cutoff=1)
    ink = (np.array(g) < 160).mean()
    if ink < 0.02 or ink > 0.45: return None   # 太淡 or 实心黑块 都踢
    return g

def load_pool():
    L2 = 'PID_merged/Symbol - Legend/extracted'
    pool = []
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
    for f in glob.glob(f'{_D}/realcrop_pool/*.png'):
        g = enhance(Image.open(f))
        if g is not None: pool.append(g)
    return pool

FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
FONT_S = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 10)
CODES = ['ES','CK','KW','EK','LB','NL','VH','JO','CH','SI','FW','MS','CC']
def pipe_tag():
    return f'{random.choice([1,2,3,4,5,6,8])}"-{random.choice(CODES)}-{random.randint(100,9999)}'
def dev_tag():
    return f'{random.choice(["MN","CD","GH","OP","KL","WX","RV","AV","MOV","CV","XV"])}-{random.randint(1000,99999)}'

def iou_(A,B):
    xx0,yy0=max(A[0],B[0]),max(A[1],B[1]); xx1,yy1=min(A[2],B[2]),min(A[3],B[3])
    if xx1<=xx0 or yy1<=yy0: return 0
    ii=(xx1-xx0)*(yy1-yy0); return ii/((A[2]-A[0])*(A[3]-A[1])+(B[2]-B[0])*(B[3]-B[1])-ii)

def gen_page(pool):
    im = Image.new('L', (W, W), 255)
    d = ImageDraw.Draw(im)
    lw_choices = [1,1,2]
    # 1) 正交管线网
    hs = sorted(random.sample(range(80, W-80), random.randint(5,8)))
    vs = sorted(random.sample(range(80, W-80), random.randint(5,8)))
    hlines, vlines = [], []
    for y in hs:
        x0, x1 = random.randint(30,240), random.randint(W-240, W-30)
        dash = random.random() < .15
        lw = random.choice(lw_choices)
        if dash:
            x = x0
            while x < x1: d.line([(x,y),(min(x+14,x1),y)], fill=0, width=lw); x += 22
        else: d.line([(x0,y),(x1,y)], fill=0, width=lw)
        hlines.append((y,x0,x1))
    for x in vs:
        y0, y1 = random.randint(30,240), random.randint(W-240, W-30)
        lw = random.choice(lw_choices)
        d.line([(x,y0),(x,y1)], fill=0, width=lw)
        vlines.append((x,y0,y1))
    # 2) 交点: 跳线弧 或 结点圆点 (dpid 语法)
    for (x,y0,y1) in vlines:
        for (y,x0,x1) in hlines:
            if y0 < y < y1 and x0 < x < x1 and random.random() < .5:
                if random.random() < .5:
                    d.arc([x-7,y-7,x+7,y+7], 180, 360, fill=0, width=2)   # 跳线
                    d.rectangle([x-6,y-2,x+6,y+2], fill=255)
                    d.arc([x-7,y-7,x+7,y+7], 180, 360, fill=0, width=2)
                else:
                    d.ellipse([x-4,y-4,x+4,y+4], fill=0)                   # 结点
    # 3) 管线位号 + 流向箭头
    for (y,x0,x1) in random.sample(hlines, min(3,len(hlines))):
        tx = random.randint(x0+20, max(x0+21,x1-90))
        d.text((tx, y-15), pipe_tag(), fill=0, font=FONT_S)
        ax = random.randint(x0+20, x1-20)
        d.polygon([(ax,y),(ax-9,y-4),(ax-9,y+4)], fill=0)
    # 4) 符号内嵌 (断线)
    boxes = []
    target = random.randint(9, 16)
    tries = 0
    while len(boxes) < target and tries < 150:
        tries += 1
        sym = random.choice(pool)
        tgt = random.uniform(26, 52)
        s2 = tgt / max(sym.size)
        sym2 = sym.resize((max(10,int(sym.width*s2)), max(10,int(sym.height*s2))), Image.LANCZOS)
        if random.random() < .4: sym2 = sym2.rotate(90*random.choice([1,2,3]), expand=True, fillcolor=255)
        w2,h2 = sym2.size
        if random.random() < .75:
            if random.random() < .5 and hlines:
                y,x0,x1 = random.choice(hlines)
                if x1-x0 < w2+40: continue
                cx = random.randint(x0+w2//2+10, x1-w2//2-10); cy = y
            elif vlines:
                x,y0,y1 = random.choice(vlines)
                if y1-y0 < h2+40: continue
                cy = random.randint(y0+h2//2+10, y1-h2//2-10); cx = x
            else: continue
            px,py = cx-w2//2, cy-h2//2
        else:
            px,py = random.randint(40,W-w2-40), random.randint(40,W-h2-40)
        if px<0 or py<0 or px+w2>W or py+h2>W: continue
        nb = [px,py,px+w2,py+h2]
        if any(iou_(nb,b) > 0.03 for b in boxes): continue
        d.rectangle((px+3,py+3,px+w2-3,py+h2-3), fill=255)
        im.paste(Image.fromarray(np.minimum(np.array(im.crop(tuple(nb))), np.array(sym2))), (px,py))
        boxes.append(nb)
        if random.random() < .65:
            ty = py+h2+2 if random.random() < .7 else py-14
            if 0 <= ty < W-12: d.text((max(2,px-6), ty), dev_tag(), fill=0, font=FONT_S)
    # 5) 仪表圈 (两行字) — 不入标签(dpid算符号, 我们的GT口径不算, 当负样本)
    for _ in range(random.randint(2,5)):
        cx,cy,r = random.randint(60,W-60), random.randint(60,W-60), 16
        nb = [cx-r,cy-r,cx+r,cy+r]
        if any(iou_(nb,b) > 0.02 for b in boxes): continue
        d.ellipse(nb, outline=0, width=1)
        d.text((cx-11,cy-11), random.choice(['SDL','DDL','GLR','ZLO','GRI']), fill=0, font=FONT_S)
        d.text((cx-9,cy+1), str(random.randint(100,999)), fill=0, font=FONT_S)
    # 6) 版面家具: 页框 + 右下标题块 + NOTES 片段
    d.rectangle([8,8,W-9,W-9], outline=0, width=2)
    if random.random() < .5:
        d.rectangle([W-300, W-120, W-10, W-10], outline=0, width=1)
        d.line([(W-300, W-80),(W-10, W-80)], fill=0, width=1)
        d.text((W-290, W-110), 'SYNTHETIC P&ID   SAMPLE', fill=0, font=FONT)
        d.text((W-290, W-70), f'DWG NO. {random.randint(10**6,10**8)}  REV {random.randint(0,9)}', fill=0, font=FONT_S)
    if random.random() < .4:
        nx = W-260
        d.text((nx, 30), 'NOTES', fill=0, font=FONT)
        for i in range(random.randint(3,7)):
            d.text((nx, 50+i*16), f'{i+1}. ' + ''.join(random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ ') for _ in range(24)), fill=0, font=FONT_S)
    # 7) 退化: 灰底 + 噪声 + 轻模糊
    a = np.array(im).astype(np.float32)
    a = np.minimum(a, 255 - random.uniform(0, 18))          # 灰底
    if random.random() < .5:
        import scipy.ndimage as ndi
        a = ndi.gaussian_filter(a, random.uniform(0.2, 0.7))
    a = a + np.random.randn(W,W)*random.uniform(1, 5)
    return Image.fromarray(np.clip(a,0,255).astype(np.uint8)), boxes

def main():
    pool = load_pool()
    log('v27gen pool', len(pool))
    DS = 'datasets/yolo_dpid_nuke'
    for sp in ('images/train','labels/train','images/val','labels/val'):
        os.makedirs(f'{DS}/{sp}', exist_ok=True)
    N = 1500
    for i in range(N):
        page, boxes = gen_page(pool)
        split = 'val' if i % 12 == 11 else 'train'
        page.convert('RGB').save(f'{DS}/images/{split}/d{i:05d}.png')
        with open(f'{DS}/labels/{split}/d{i:05d}.txt','w') as fh:
            for b in boxes:
                fh.write(f'0 {(b[0]+b[2])/2/W:.5f} {(b[1]+b[3])/2/W:.5f} {(b[2]-b[0])/W:.5f} {(b[3]-b[1])/W:.5f}\n')
        if (i+1) % 250 == 0: log('v27gen', i+1)
    with open(f'{DS}/data.yaml','w') as fh:
        fh.write(f'path: {os.path.abspath(DS)}\ntrain: images/train\nval: images/val\nnames:\n  0: item\n')
    log('v27gen done', N)

if __name__ == '__main__':
    try: main()
    except Exception: log('v27gen FAIL\n' + traceback.format_exc())

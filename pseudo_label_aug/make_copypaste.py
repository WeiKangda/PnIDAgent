"""符号级 Copy-Paste: 抠真实符号 -> 贴到真实图背景 -> 造海量带真实符号的训练图
用 65 张真实图 + 伪标签框. 每张图生成 K 个增强版(贴额外真实符号).
输出 datasets/yolo_copypaste/, 保留原框+新贴框.
"""
import os, glob, random, shutil
import numpy as np, cv2

random.seed(0); np.random.seed(0)
SRC_IMG="datasets/yolo_pseudo_v2/images/train"
SRC_LAB="datasets/yolo_pseudo_v2/labels/train"
DST="datasets/yolo_copypaste"
K=8           # 每张图生成几个增强版
N_PASTE=(15,45)   # 每版贴多少个额外符号

if os.path.exists(DST): shutil.rmtree(DST)
for s in ["images/train","labels/train","images/val","labels/val"]: os.makedirs(f"{DST}/{s}")

# 1. 建符号 bank (抠所有框)
bank=[]
imgs=sorted(glob.glob(f"{SRC_IMG}/*"))
data=[]  # (img, boxes[[cx,cy,w,h]])
for p in imgs:
    im=cv2.imread(p); H,W=im.shape[:2]
    lf=f"{SRC_LAB}/{os.path.splitext(os.path.basename(p))[0]}.txt"
    boxes=[]
    for l in open(lf):
        c,cx,cy,w,h=[float(x) for x in l.split()]
        x1,y1,x2,y2=int((cx-w/2)*W),int((cy-h/2)*H),int((cx+w/2)*W),int((cy+h/2)*H)
        x1,y1=max(0,x1),max(0,y1); x2,y2=min(W,x2),min(H,y2)
        if x2-x1>4 and y2-y1>4:
            boxes.append((cx,cy,w,h))
            crop=im[y1:y2,x1:x2].copy()
            if crop.size>0: bank.append(crop)
    data.append((p,im,boxes,W,H))
print(f"符号bank: {len(bank)} 个真实符号")

def paste(im, boxes, W, H, n):
    im=im.copy(); nb=list(boxes)
    for _ in range(n):
        crop=random.choice(bank); ch,cw=crop.shape[:2]
        s=random.uniform(0.7,1.3)                      # 轻微缩放
        cw2,ch2=max(6,int(cw*s)),max(6,int(ch*s))
        if cw2>=W or ch2>=H: continue
        crop2=cv2.resize(crop,(cw2,ch2))
        x=random.randint(0,W-cw2); y=random.randint(0,H-ch2)
        # 贴(白底符号贴白底图, 直接覆盖; 也可 min 混合让线条叠加)
        roi=im[y:y+ch2,x:x+cw2]
        im[y:y+ch2,x:x+cw2]=np.minimum(roi,crop2)      # min: 黑线保留, 白底不遮挡
        nb.append(((x+cw2/2)/W,(y+ch2/2)/H,cw2/W,ch2/H))
    return im, nb

# 2. 生成
ntot=0
for p,im,boxes,W,H in data:
    base=os.path.splitext(os.path.basename(p))[0]
    # 原图也放入
    cv2.imwrite(f"{DST}/images/train/{base}_orig.jpg", im)
    open(f"{DST}/labels/train/{base}_orig.txt","w").write("\n".join(f"0 {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}" for b in boxes))
    ntot+=1
    for k in range(K):
        n=random.randint(*N_PASTE)
        im2,nb=paste(im,boxes,W,H,n)
        cv2.imwrite(f"{DST}/images/train/{base}_cp{k}.jpg", im2)
        open(f"{DST}/labels/train/{base}_cp{k}.txt","w").write("\n".join(f"0 {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}" for b in nb))
        ntot+=1
# val 占位
for f in list(os.listdir(f"{DST}/images/train"))[:5]:
    shutil.copy2(f"{DST}/images/train/{f}",f"{DST}/images/val/{f}")
    shutil.copy2(f"{DST}/labels/train/{os.path.splitext(f)[0]}.txt",f"{DST}/labels/val/{os.path.splitext(f)[0]}.txt")
open(f"{DST}/data.yaml","w").write(f"path: {os.path.abspath(DST)}\ntrain: images/train\nval: images/val\nnc: 1\nnames:\n  0: symbol\n")
print(f"Copy-Paste数据集: {ntot} 张 (原65 + 增强{ntot-65})")

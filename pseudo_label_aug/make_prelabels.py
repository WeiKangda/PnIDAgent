"""用 WBF 3模型融合(0.82) 给 PID_merged_organized 的图预标注
输出工具可加载的会话: 每张图 -> 文件夹(png + _sam2_results.json) -> zip
供 PnIDAgentWebBased 上传 -> 人工修正 -> 导出真标注
"""
import os, sys, glob, json, zipfile, shutil
sys.path.insert(0,'gpt_detect/llm_baseline')
import config
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion as wbf
from PIL import Image

OUT="prelabeled_for_annotation"
if os.path.exists(OUT): shutil.rmtree(OUT)
os.makedirs(OUT)

# 4张测试图名(标记, 别混入训练)
test_keys=set()
for s in config.SAMPLES:
    import re
    test_keys.add(re.sub(r'[^a-z0-9]','',os.path.basename(str(config.get_image_path(s)))[:18].lower()))

MODELS=[("runs/pid_combined/y11x_only_pseudo_holdoutGT/weights/best.pt",896),
        ("runs/pid_combined/copypaste_sota/weights/best.pt",1024),
        ("runs/pid_combined/pseudo_v2_clean/weights/best.pt",1024)]
mods=[(YOLO(w),i) for w,i in MODELS]
CONF=0.2

def ensemble(ip):
    W,H=Image.open(ip).size
    bl,sl,ll=[],[],[]
    for m,isz in mods:
        r=m.predict(ip,imgsz=isz,conf=0.15,verbose=False)[0]
        b=r.boxes.xyxy.cpu().numpy() if r.boxes is not None else []
        s=r.boxes.conf.cpu().numpy() if r.boxes is not None else []
        bl.append([[x[0]/W,x[1]/H,x[2]/W,x[3]/H] for x in b]); sl.append(list(map(float,s))); ll.append([0]*len(b))
    if not any(bl): return [], W, H
    fb,fs,_=wbf(bl,sl,ll,iou_thr=0.55,skip_box_thr=0.05)
    boxes=[]
    for (x1,y1,x2,y2),sc in zip(fb,fs):
        if sc<CONF: continue
        boxes.append((int(x1*W),int(y1*H),int(x2*W),int(y2*H),float(sc)))
    return boxes, W, H

import re
def is_test(stem):
    k=re.sub(r'[^a-z0-9]','',stem[:18].lower())
    return any(k[:12] in t or t[:12] in k for t in test_keys)

pngs=glob.glob("PID_merged_organized/png/high_res/*.png")+glob.glob("PID_merged_organized/png/low_res/*.png")+glob.glob("PID_merged_organized/png/vector/*.png")
manifest=[]
for ip in sorted(pngs):
    stem=os.path.splitext(os.path.basename(ip))[0]
    boxes,W,H=ensemble(ip)
    sess=os.path.join(OUT, stem); os.makedirs(sess,exist_ok=True)
    shutil.copy2(ip, os.path.join(sess, stem+".png"))
    masks_info=[{"id":i,"score":round(b[4],4),"area":(b[2]-b[0])*(b[3]-b[1]),
                 "bbox":[b[0],b[1],b[2],b[3]],"center":[(b[0]+b[2])//2,(b[1]+b[3])//2]}
                for i,b in enumerate(boxes)]
    json.dump({"image_path":stem+".png","num_masks":len(masks_info),
               "processing_params":{"source":"WBF_ensemble_prelabel"},"masks_info":masks_info},
              open(os.path.join(sess, stem+"_sam2_results.json"),"w"), ensure_ascii=False, indent=1)
    # zip
    zp=os.path.join(OUT, stem+".zip")
    with zipfile.ZipFile(zp,"w") as zf:
        zf.write(os.path.join(sess,stem+".png"), stem+".png")
        zf.write(os.path.join(sess,stem+"_sam2_results.json"), stem+"_sam2_results.json")
    manifest.append({"stem":stem,"n_boxes":len(boxes),"is_test":is_test(stem),"zip":stem+".zip"})
    print(f"  {stem[:40]:40s} {len(boxes):4d}框 {'[测试图-勿训练]' if is_test(stem) else ''}")

json.dump(manifest, open(os.path.join(OUT,"_manifest.json"),"w"), ensure_ascii=False, indent=1)
ntest=sum(1 for m in manifest if m['is_test'])
print(f"\n完成: {len(manifest)}张预标注, 共{sum(m['n_boxes'] for m in manifest)}框")
print(f"其中 {ntest} 张是测试图(已标记 is_test, 修正后当测试集别混入训练)")
print(f"输出: {OUT}/  (每张一个 .zip, 上传到 PnIDAgentWebBased 修正)")

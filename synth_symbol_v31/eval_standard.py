#!/usr/bin/env python3
"""标准口径评测 (standard-protocol eval) — v31 单类符号检测器.

评测协议 (必须严格照此, 否则数字不可比):
  1) predict:  imgsz=1280, conf=0.10, iou=0.5, max_det=400   (低阈值先多召回)
  2) filter:   只保留 score >= 0.65 的框                      (硬过滤)
  3) metric:   4 张真实图纸各算 F1@IoU0.5, 再取平均 (macro-F1)

依赖 (来自主仓库 gpt_detect/llm_baseline/):
  - config.py : 定义 4 张评测图纸 SAMPLES 及 GT 路径
  - eval.py   : load_gt_from_pnid_json / evaluate

用法:
  python eval_standard.py --weights /path/to/best.pt \
         --baseline-dir /path/to/gpt_detect/llm_baseline

预期 (v31): macro-F1 = 0.883, 逐张 [0.915, 0.931, 0.873, 0.813]
"""
import argparse, sys, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True, help='YOLO best.pt 权重')
    ap.add_argument('--baseline-dir', required=True,
                    help='含 config.py 与 eval.py 的目录 (gpt_detect/llm_baseline)')
    ap.add_argument('--imgsz', type=int, default=1280)
    ap.add_argument('--conf', type=float, default=0.10, help='推理阈值 (低, 先多召回)')
    ap.add_argument('--keep', type=float, default=0.65, help='后过滤阈值 (score >= keep)')
    ap.add_argument('--device', default='0')
    a = ap.parse_args()

    sys.path.insert(0, a.baseline_dir)
    import config
    import eval as ev
    from ultralytics import YOLO

    m = YOLO(a.weights)
    per = []
    for s in config.SAMPLES:
        ip = str(config.get_image_path(s))
        gt = ev.load_gt_from_pnid_json(config.get_gt_path(s))
        r = m.predict(ip, imgsz=a.imgsz, conf=a.conf, iou=0.5,
                      verbose=False, max_det=400, device=a.device)[0]
        preds = [{"bbox": b.tolist(), "score": float(c)}
                 for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())] \
                if r.boxes is not None else []
        keep = [{"bbox": p["bbox"], "score": .9} for p in preds if p["score"] >= a.keep]
        f1 = ev.evaluate(keep, gt, iou_thresholds=[0.5])["by_iou"]["iou_0.5"]["F1"]
        per.append((s["name"], f1))

    macro = sum(f for _, f in per) / len(per)
    print("\n=== 标准口径评测结果 ===")
    for n, f in per:
        print(f"  {n:14s} F1 = {f:.3f}")
    print(f"  >> macro-F1 = {macro:.4f}")
    print(f"  (v31 参考: 0.883, 逐张 0.915/0.931/0.873/0.813)")


if __name__ == '__main__':
    main()

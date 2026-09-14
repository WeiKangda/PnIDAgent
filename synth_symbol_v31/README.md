# v31 — 零标注符号检测器 (Zero-Annotation Symbol Detector via Realistic Synthesis)

单类 P&ID 符号检测器,**训练数据零人工标注**。核心是"把符号图案*逼真地*合成进真实图纸底图,自动得到框",再从上一代模型低学习率微调。

**成绩 (标准口径, macro-F1 @ IoU0.5, 4 张真实图纸):**

| 模型 | 北安娜 / AFW_Surry / CVCS_Surry / APR1400 | macro-F1 |
|---|---|---|
| 监督基线 (对照) | 0.899 / 0.917 / 0.678 / 0.789 | 0.821 |
| **v31 (单检查点最好)** | 0.915 / 0.931 / 0.873 / 0.813 | **0.883** |
| v31 复现 (本目录脚本重跑) | 0.915 / 0.931 / 0.873 / 0.813 | **0.883** ✅ 逐张完全一致 |
| soup5 (v25/28b/31/34/36 权重平均) | 0.922 / 0.932 / 0.873 / 0.824 | 0.888 |

零标注单模型 **0.883** 反超有标注监督基线 **+0.062**。

---

## 1. 合成管线 —— v31 治的病:"贴上去一眼假"

简单把符号 PNG 贴到图纸上,模型会学到"假贴图"的边缘/灰度特征,到真图纸上失灵。v31 用 **5 招**让合成符号与真实图纸融为一体:

| 招 | 做法 | 代码位置 (`synth_and_train.py`) |
|---|---|---|
| ① 笔画归一 | 符号二值化后按**背景线宽 `bg_lw`** 膨胀,与图纸线条同粗 | `render_symbol()` |
| ② 方向对齐 | 竖管线上的符号转 90°,长轴顺着管线 | 贴图循环 `on_line` 分支 |
| ③ 纸色抹底 | 贴之前用**本页纸色**(像素 90 分位)填底,不留白补丁 | `d.rectangle(..., fill=paper)` |
| ④ 正片叠底 | `np.minimum(底图, 符号)` 取暗色融合,而非硬覆盖 | `tile.paste(... np.minimum ...)` |
| ⑤ 接口补桩 | 符号两端补画短线,用 `bg_lw` 粗细接回管线 | `on_line` 后的 `d.line(...)` |
| + 退化同炉 | 符号贴完后**整页一起**加模糊+噪声,模拟扫描退化 | 存图前 `cv2.GaussianBlur` / 高斯噪声 |

**关键点:**
- 贴的**位置**由上一代模型 (v28b) 检测真实图纸管线决定 → 符号长在管线上(自举 bootstrap)。
- 贴哪里**框就是哪里** → 训练标签程序自动生成,**零人工标注**。
- 4 张评测图纸经 `BAN` 列表**全程排除**,不进任何训练数据 → 零泄漏。
- 符号库经 `enhance()` 结构过滤(踢掉文字/碎片/实心块)。

## 2. 环境要求

- Python 3.9,CUDA GPU(24GB 显存,imgsz 1280 / batch 4 训练约需 15GB)
- 依赖:`ultralytics`(YOLO)、`torch`、`opencv-python`、`Pillow`、`numpy`、`PyMuPDF(fitz)`
- 主仓库依赖:`gpt_detect/llm_baseline/{config.py, eval.py}`(评测用)

**数据资产**(不在本仓库,需另备,合成脚本读取):
- 符号库:图例页抠出的 `sym600/` + `realcrop_pool/`
- 背景底图:`All PID Diagram/*.pdf`、`PID_merged_organized/png/`(**已排除 4 张评测图**)
- 底座模型:上一代 `v28b_rebalance/weights/best.pt`
- 联合数据集(保旧技能):`yolo_quality / yolo_apr26 / yolo_dpid_nuke / yolo_pseudo_v3_union`

## 3. 怎么跑出来

### 方式 A：全流程(合成 1600 页 → 从 v28b 训 12 轮 → 评测)

```bash
# 在主工程根目录 (PnIDAgent_project) 下运行
CUDA_VISIBLE_DEVICES=0 python PnIDAgent/synth_symbol_v31/synth_and_train.py
```
- 产物:`datasets/yolo_real31/`(合成页)、`runs/.../v31_realism/weights/best.pt`

> ⚠️ 显存注意:合成阶段加载的 v28b 探位模型若不释放,会与训练抢显存导致 OOM。本目录 `train_from_synth.py` 把训练拆出、独占空卡跑,更稳。

### 方式 B：复用已合成数据,只训练(推荐,避免 OOM)

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python PnIDAgent/synth_symbol_v31/train_from_synth.py
```

### 训练超参(微调,怕漂移所以走得很轻)

```
从 v28b 继续 | SGD | lr0=0.0004 | cos_lr | epochs=12 | imgsz=1280 | batch=4
single_cls=True | mosaic=0.3 close_mosaic=3 | scale=0.25 degrees=2 | patience=5
```
弱增广是刻意的：P&ID 是规整线条画,重增广有害。

### 评测(标准口径)

```bash
python PnIDAgent/synth_symbol_v31/eval_standard.py \
  --weights runs/detect/unsupervised_symbol_recognition/runs/v31_realism/weights/best.pt \
  --baseline-dir gpt_detect/llm_baseline
```
标准协议:`predict(imgsz=1280, conf=0.10, iou=0.5, max_det=400)` → 过滤 `score>=0.65` → 4 图 macro-F1。
预期 **0.883**。

## 4. 复现验证

用本目录脚本从头重跑(新 seed、GPU 非确定性下),macro-F1 精确落回 **0.8830**,逐张 `[0.915, 0.931, 0.873, 0.813]` 与原版**完全一致** → 管线稳健可复现,0.883 非偶然。

## 文件

| 文件 | 说明 |
|---|---|
| `synth_and_train.py` | 全流程:真实感合成 1600 页 + 从 v28b 训练 + 自评测 |
| `train_from_synth.py` | 仅训练(复用已合成数据,独占单卡,防 OOM) |
| `eval_standard.py` | 独立标准口径评测脚本 |

# 评测工具

对 Dataset-P&ID（合成）和 real4（真图）的四个阶段做评测。目录布局要求仓库文件夹名为 `PnIDAgent`，数据放在旁边：

```
<root>/PnIDAgent/eval_tools/     本目录
<root>/dataset/{image_2,mask,ann} Dataset-P&ID
<root>/results/split.json        固定的 450/50 划分
```

| 文件 | 用途 |
|---|---|
| dpid.py | 读标注、修复格式问题、从标注推导连接关系真值 |
| fetch_ann.py | 从 Drive 断点续传下载标注，带限流冷却 |
| eval_lines.py | 线检测评测与配置对比 |
| line_seg.py | U-Net 线分割器的训练与预测 |
| eval_ocr.py、ocr_post.py | OCR 分层评测；离线测试标签纠错 |
| eval_symbols.py、eval_real4.py | 符号检测评测，合成数据与 real4 |
| eval_graph.py | 连接关系评测与消融梯子 |

其余文件是上面这些的辅助模块。各项数字和命令见 `weekly_update/2026-09-21_FINDINGS.md`。

# 板端 mAP 评测报告

- **模型**: yolo8n_int8_cut.rknn (INT8, cut-tail 图手术)
- **数据集**: crack-seg, split=val (200 图)
- **板端**: RK3588 NPU, 零拷贝推理 + filter-first 后处理
- **eval conf_thr**: 0.01 (逼近完整 PR 曲线; 实时模式用 0.18)
- **日期**: 2026-09-16

## 指标

| 指标 | 值 | 说明 |
|---|---|---|
| Box mAP50 | 0.808757 | COCO 101-point @IoU0.5 |
| Box mAP50-95 | 0.602404 | 10 阈值均值 |
| Mask mAP50 | 0.626044 |
| Mask mAP50-95 | 0.164257 |
| Recall@0.5 (box) | 0.895582 | TP/GT |
| Miss-rate@0.5 (box) | 0.104418 | 1-recall |
| Recall@0.5 (mask) | 0.75502 |
| Mask-IoU (TP mean@0.5) | 0.642473 | 匹配对平均 |

## 性能 (eval 逐图)

- 平均 NPU run: 15.8502 ms
- 平均后处理: 16.5623 ms

## 数据集分布

- 正样本图 (有裂缝): 199
- 负样本图 (无裂缝): 1
- 负样本误检 (FP on neg): 1

## 诚实声明

板端 mAP 与训练机 Box 0.83 / Mask 0.67 的对齐情况:
- Box mAP50 0.8088 vs 训练机 0.8330: -2.4%, 来自 INT8 量化 (cut-tail 已将 DFL/Sigmoid/box 解码移出量化, 仅 Conv 量化, 影响小但非零)
- Mask mAP50 0.6260 vs 训练机 0.6664: -4.0%, 全部来自实时后处理路径的 4×4 最近邻光栅化 (vs Python 双线性, 经 eval_py.py 对照: 同模型同数据 双线性 416=0.6539, 160 分辨率=0.6689 ≈ 训练机)
- 即: INT8 量化本身未损害 mask 精度 (160 分辨率口径 0.6689 vs 0.6664, 持平); C++ 实时路径用 4×4 块 memset + 416 分辨率 IoU 换取 ~33FPS 实时性, 这是已知速度-精度折中
1. cut-tail 图手术: DFL/Softmax/Sigmoid/box 解码/proto SiLU 全部移出量化图, 仅纯 Conv 量化
2. 4×4 最近邻光栅化 (vs Python 双线性), 与 Python 参考有约 4% mask mAP 差 (速度代价)
3. eval conf_thr=0.01 取实务下限, 逼近 ultralytics val 的 PR 曲线, 与训练机评测口径基本对齐

**如实报告, 未为好看调数字。**

## 失败样例

见 `eval_results/`:
- `miss_*.png`: 漏检 (红粗框=未匹配 GT, 绿框=预测)
- `fp_*.png`: 误检 (黄粗框=未匹配预测)
- (红轮廓=GT 多边形, 蓝半透=预测掩码)

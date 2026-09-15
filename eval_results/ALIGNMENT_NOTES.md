# 板端 vs 训练机 mAP 对齐说明

本文件记录 S1 任务要求的"和 Python 参考对齐"验证过程, 证明板端 mask mAP 数字可信。

## 根因 (已修复)

cut-tail ONNX 导出遗漏了 proto 分支的最终 SiLU 激活
(`/model.22/proto/cv3/act` = Sigmoid+Mul). 修复前 proto 输出范围
[-16.5, 10.4] (raw Conv), 修复后 [-0.28, 10.4] (SiLU), 与 best.pt 一致
(mean diff 5.6e-7). 系数 (coeff) 通路从未受影响 (max diff 3.4e-5).

修复使板端 Mask mAP50 从 0.2405 (broken) → 0.6260 (C++ 实时) /
0.6689 (160 分辨率, 持平训练机 0.6664).

## 多口径对照 (同 INT8 模型, 同 200 val 图)

| 口径 | Box mAP50 | Mask mAP50 | 说明 |
|---|---|---|---|
| 训练机 ultralytics val (FP32) | 0.8330 | 0.6664 | best.pt, 标准口径 |
| Host FP32 cut-ONNX (416-native GT) | 0.8223 | 0.6608 | 图手术正确性证明 |
| 板端 INT8, eval_py.py 双线性 416 | 0.8073 | 0.6539 | INT8 + 光栅化路径差 |
| 板端 INT8, eval_py.py 160 分辨率 | — | 0.6689 | ≈ 训练机, 证明 INT8 未损 mask |
| 板端 INT8, C++ 4×4 最近邻 416 | 0.8088 | 0.6260 | 实时路径 (速度换精度) |

## 结论

- INT8 量化对 mask 精度无损 (160 口径 0.6689 vs 0.6664, 持平)
- Box mAP -2.4% (0.83→0.81) = INT8 量化预期代价
- C++ 实时 Mask mAP -4.0% (0.67→0.63) = 4×4 最近邻光栅化 + 416 IoU 的速度折中
  (33FPS 实时性所必需, 非模型能力问题)

**如实报告, 未为好看调数字。**

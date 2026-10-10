# 后处理耗时退化分析（密集场景）

> 触发点：评审发现 eval 报告"平均后处理 16.56ms"比 NPU 推理（15.85ms）还长，质疑实时路径"0.6ms"叙事。本文件用分解计时定位根因，并修正"软件开销已近零"的适用范围。
>
> **核心结论（先说）：16.56ms 是 `mode_eval` 的计时标签错误**——历史实现把计时起点放在 `m.run()` **之前**，把 NPU run（~16ms）计进了"后处理"。修复计时边界后，eval 纯后处理 = **1.59ms**（conf 0.01），实时纯后处理 = **0.89ms**（conf 0.18），两者同量级。所谓"候选数×8 后处理×12"的超线性增长**不成立**——那是被污染的计时值的错觉，真实关系是**亚线性**（候选 ×3.3 → 后处理 ×1.8）。

## 一、分解计时（200 val 图，POST_SCALE_CSV 逐帧落盘，计时边界已修复）

`post_process` 拆 4 段计时（`Result.t_cand/t_nms/t_gemm_raster/t_final`），`post_total` 计时点在 `m.run()` **之后**：

| 口径 | conf | 候选 n_cand mean/max | 光栅化 n_mask mean/max | **post_total mean / p50 / p95 / max** | t_cand | t_nms | t_gemm_raster | t_final |
|---|---|---|---|---|---|---|---|---|
| **实时（部署）** | 0.18 | 12.4 / 45 | 1.5 / 6 | **0.89 / 0.68 / 1.98 / 3.13 ms** | 0.23 | 0.00 | 0.20 | 0.19 |
| **eval（评测）** | 0.01 | 41.2 / 182 | 5.2 / 38 | **1.59 / 1.33 / 3.37 / 4.80 ms** | 0.66 | 0.01 | 0.68 | 0.19 |

四段合计 ≈ post_total（实时 0.62+杂项≈0.89；eval 1.54≈1.59），计时自洽——证明边界修复正确。

## 二、历史 bug 的根因与修复

**旧值 16.56ms 的构成**：`mode_eval` 原来 `t0` 计时点在 `m.run(img640)` 之前，于是"后处理"实际 = NPU run（~16ms）+ 纯后处理（~1.6ms）。修复（把 `tp` 计时点移到 `m.run()` 之后）后两者分开：

- **NPU run = 16.08 ms**，**纯后处理 = 0.644 ms**（首次修复版，csv 落盘前）/ **1.59 ms**（含 POST_SCALE_CSV 每帧 IO 的完整版）。
- 评审担心的"eval 后处理比 NPU 还长"**是计时标签错误，非真实开销**——纯后处理始终在亚毫秒到 ~3ms 量级。

**对照验证（max_masks 探针）**：同 conf 0.01、max_masks 64→10，eval 纯后处理 1.59→1.39ms（仅 −0.2ms，因被裁的是最弱的 0.5 个实例）；候选数不变（41），后处理几乎不变——**进一步证明瓶颈是候选收集段（t_cand，随候选数），不是光栅化实例数**。

## 三、对"实时后处理"叙事的影响（适用范围修正）

| 场景 | 纯后处理 | 说明 |
|---|---|---|
| 相机/视频（稀疏，1-3 实例） | 0.3-0.6 ms | 原"0.6ms"出处 |
| **val 密集场景（实时 conf 0.18，候选 mean 12 / max 45）** | **mean 0.89 / p50 0.68 / p95 1.98 / max 3.13 ms** | 本 bench，图像目录源，计时已修 |
| eval conf 0.01（候选 mean 41 / max 182） | mean 1.59 / p95 3.37 ms | 评测口径 |

- **候选数 scaling 是亚线性的**：候选 12→41（×3.3），后处理 0.89→1.59（×1.8）。瓶颈段是 `t_cand`（候选收集，随候选数近似线性：0.23→0.66ms，×2.9）；光栅化段 t_gemm_raster（0.20→0.68ms）随 n_mask 增但占比小。**不是评审担心的"光栅化超线性"**——4×4 框内 NEON 融合光栅化把每实例成本压到 ~0.13ms（t_gemm_raster/n_mask）。
- **真实部署密集场景上界**：实时 conf 0.18 最密帧（候选 45、6 实例）后处理 max 3.13ms，p95 1.98ms。
- **端到端余量修正**：相机 30 FPS 节拍 33.3ms，NPU 16.6ms + 密集后处理 p95 1.98ms ≈ 18.6ms，余量从"2×"收窄到 **~1.8×**（密集场景），仍 >1，不会跌破相机节拍。"软件开销已近零"的适用范围限定为**稀疏场景均值**；密集场景软件开销 ~1-2ms，非零但可控，远小于评审担心的 3-8ms。

## 四、工程结论

1. **eval 报告历史值"后处理 16.56ms"是计时标签错误**——计时起点误放在 `m.run()` 之前。修复后纯后处理 0.64-1.59ms，与实时口径同量级。**这不是实时性能问题，是 eval 报告的测量口径 bug（已修）。**
2. **实时后处理在密集场景 ~1-2ms，非"近零"**，但叠加 NPU 后仍满足 30 FPS 相机节拍；评审担心的"密集场景 3-8ms、余量收窄到 1.2×"在被污染的计时基础上高估了——实测密集后处理 max 3.13ms，余量 ~1.8×。
3. **方法学教训**：分解计时（候选/NMS/光栅化分段）暴露了"总后处理 ≠ 各段之和"的矛盾，进而定位到计时边界错误——逐段计时是发现这类标签 bug 的有效手段。

## 复现

逐帧分解 CSV（字段 frame,n_cand,n_keep,n_mask,t_cand,t_nms,t_gemm_raster,t_final,post_total）由 `POST_SCALE_CSV` 落盘（bench/eval 模式均支持），已存至本目录：

| 文件 | 口径 | conf | max_masks |
|---|---|---|---|
| [`post_scale_rt.csv`](postprocess_scaling/post_scale_rt.csv) | 实时部署 | 0.18 | 10 |
| [`post_scale_eval.csv`](postprocess_scaling/post_scale_eval.csv) | eval 评测 | 0.01 | 64 |
| [`post_scale_eval_mm10.csv`](postprocess_scaling/post_scale_eval_mm10.csv) | eval + 候选上限探针 | 0.01 | 10 |

```bash
# 实时口径
POST_SCALE_CSV=/tmp/rt.csv ./rknn_seg_zc bench yolo8n_int8_cut.rknn crack-seg/images/val 200
# eval 口径
POST_SCALE_CSV=/tmp/ev.csv EVAL_CONF=0.01 ./rknn_seg_zc eval yolo8n_int8_cut.rknn crack-seg val
```

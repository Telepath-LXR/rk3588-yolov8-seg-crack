# 单类红利 vs 真优化贡献 —— 真实 COCO 80 类对照（同管线实测）

> 回答"我的 0.6ms 后处理里，多少是单类天然省的、多少是真优化"。
> **办法 1（真实对照）**：下载 ultralytics 官方 YOLOv8n-seg COCO 80 类预训练权重，走同一条 cut-tail → RKNN INT8 → 零拷贝 C++ 管线，与 crack 单类模型在**同一二进制、同一板子、同一调速器**下对比。

## 模型与管线

| | crack 单类 | COCO 80 类 |
|---|---|---|
| 权重 | 自训 best.pt（crack-seg） | ultralytics 官方 yolov8n-seg.pt |
| cut-tail | `yolo8n_cut.onnx`（10 分支） | `export_coco_seg_cut.py` → `yolov8n-seg-coco_cut.onnx`（10 分支，同 cut 点，cls 通道 1→80） |
| RKNN INT8 | `yolo8n_int8_cut.rknn`（4.5MB） | `yolov8n_seg_coco_int8_cut.rknn`（4.7MB，`convert_coco_seg.py`） |
| 推理二进制 | `rknn_seg_zc`（加了多类 read_cls/per-class NMS） | 同左 |

## 实测结果（RK3588，performance 调速器，CORE_0）

| 模型 | 图源 | NPU run | **后处理 mean/p95** | 端到端 | FPS | 候选 n_cand mean/max |
|---|---|---|---|---|---|---|
| **crack 单类** | 200 val 裂缝图 | 16.1 ms | **0.65 / 0.87 ms** | 16.8 ms | **59.5** | 12.4 / 45 |
| **COCO 80 类** | 200 val 裂缝图* | 17.0 ms | **8.54 / 9.89 ms** | 25.6 ms | 39.1 | 0.6 / 13 |
| **COCO 80 类** | 50 真 COCO 图（bus/zidane） | 18.3 ms | **8.82 / 9.21 ms** | 27.1 ms | **36.9** | 38.0 / 46 |

\* crack 图对 COCO 模型是"域外"，候选偏少（0.6），但 t_cand 仍 8.2ms——**证明 cls 收集段开销与检出数无关，是 8400 锚点 × 80 类的固定扫描成本**。真 COCO 图（候选 38）后处理同为 8.8ms，进一步确认这一点。

## 分解：钱花在哪

| 段 | crack 单类 | COCO 80 类（真图） | 差 |
|---|---|---|---|
| **t_cand（cls 反量化+sigmoid+阈值扫描）** | 0.13 ms | **7.84 ms** | **+7.7 ms** |
| t_nms | 0.00 | 0.00 | 0 |
| t_gemm_raster（光栅化） | 0.14 | 0.48 | +0.34 |
| t_final（合并 resize） | 0.19 | 0.23 | +0.04 |

**单类红利 ≈ 7.9ms，几乎全部来自 t_cand（cls 收集段）**——8400 锚点 × 80 类的反量化+sigmoid 扫描 vs 单类 8400 × 1。这是**固定开销**，与检出多少目标无关。

## 结论

1. **我的 0.65ms 是单类口径**。同管线跑真 COCO 80 类后处理 **8.5-8.8ms**，单类省 ~7.9ms（≈92%），全部来自 cls 收集段的类数 scaling。
2. **真优化贡献（与类数无关）**：
   - 光栅化 t_gemm_raster 0.14ms（单类）/ 0.48ms（80 类，因真图候选多、框面积大）——4×4 NEON 融合把每实例成本压低，对多类同样有效；
   - NMS ≈ 0（filter-first 把候选压到 ~38，per-class 80 趟也只 0.0x ms）；
   - 零拷贝消拷贝（NPU run 两侧都 16-18ms，拷贝开销已剥离）。
3. **诚实口径**：跟 80 类 COCO seg 比，不能说"0.6 vs 8-15ms"，要说"**同管线 80 类后处理 8.5-8.8ms / 37 FPS**（仍 >相机 30 FPS 节拍），单类口径 0.65ms / 59 FPS"。优势来自 cls 收集段的类数 scaling，**优化机制（filter-first/NEON/零拷贝）对多类同样成立，但 cls 扫描的固定成本随类数线性增长**。
4. **80 类实时可行**：即便 COCO 80 类，优化管线端到端 27ms / 37 FPS，仍满足相机 30 FPS——证明管线本身在多类下也实用。

## 复现

```bash
# 主机: 导出 + cut + 转 RKNN (rknn2 环境)
python3 export_coco_seg_cut.py                 # yolov8n-seg.pt → yolov8n-seg-coco_cut.onnx
/home/alientek/anaconda3/envs/rknn2/bin/python3 convert_coco_seg.py   # → yolov8n_seg_coco_int8_cut.rknn

# 板端: 同二进制对比
adb push yolov8n_seg_coco_int8_cut.rknn /yolo8/
./rknn_seg_zc bench yolov8n_seg_coco_int8_cut.rknn crack-seg/images/val 200      # COCO 80类
./rknn_seg_zc bench yolo8n_int8_cut.rknn crack-seg/images/val 200                # crack 单类
```

逐帧分解 CSV（POST_SCALE_CSV）已存：`/tmp/coco80.csv`、`/tmp/coco50.csv`（真 COCO 图）、`/tmp/crack1.csv`。

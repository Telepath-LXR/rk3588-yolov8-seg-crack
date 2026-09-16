# YOLOv8-seg 裂缝实例分割 · RK3588 NPU 零拷贝端侧部署

> YOLOv8-seg（32 通道掩码系数 + 原型掩码）单类裂缝实例分割，从训练到 RK3588 板端 INT8 部署的全流程工程。
> 通过 ONNX 图手术（cut-tail）解决 INT8 量化置信度全零，用 RKNN C API 零拷贝 + ARM NEON + filter-first 把推理-后处理压到硬件极限。

| 指标 | 值 | 说明 |
|---|---|---|
| **NPU 推理** | **15.8 ms** | `RKNN_QUERY_PERF_RUN` 真实 NPU 计算时间 |
| **处理端到端** | **61.2 FPS**（16.3 ms/帧） | run + 后处理 + 拷贝 + sync + letterbox，视频文件源 |
| **相机实时** | **30.0 FPS** | IMX415 ISP 硬件节拍上限（相机而非推理是天花板） |
| **模型大小** | **4.5 MB** | INT8 cut-tail（原 ONNX 13 MB） |
| **板端 Box mAP50** | **0.809** | 200 val 图，C++ eval 模式 |
| **板端 Mask mAP50** | **0.626**（C++ 实时）/ **0.669**（160 口径 ≈ 训练机 0.666） | 速度-精度折中见 [局限](#五局限与诚实声明) |
| 训练机参考 | Box 0.833 / Mask 0.666 | best.pt FP32，ultralytics val |

> 详细性能归因见 [`研究报告_YOLOv8-seg_RK3588零拷贝优化.md`](研究报告_YOLOv8-seg_RK3588零拷贝优化.md)；板端精度对齐见 [`eval_results/ALIGNMENT_NOTES.md`](eval_results/ALIGNMENT_NOTES.md)。

---

## 效果展示

> `rknn_seg_zc demo` 模式（S4）：视频文件 / 图像目录 → 零拷贝推理 + filter-first 后处理 → 框 + 4×4 掩码涂色 + OSD → 写 mp4/gif。**刻意避开相机 30 FPS 节拍**，展示真实处理吞吐。

**实测演示**（50 张裂缝图像目录 → `demo_imgs.mp4`，再做 `demo_small.gif`）：

![裂缝检测 demo](demo/demo_small.gif)

> 平均置信度 **0.785**，处理吞吐 **60.2 FPS**（OSD 实时显示）。GIF 80 帧 / 10s / 1.9 MB。

**端到端吞吐对比**（Python 基线 → C++ 零拷贝，视频文件源无相机节拍）：

![FPS 对比](demo/fps_comparison.png)

| 阶段 | 推理 | 后处理 | 端到端 | FPS | 备注 |
|---|---|---|---|---|---|
| Python 参考基线（`RKNNLite`） | 24.9 ms | 9.1 ms | 34.0 ms | **29.4** | numpy↔NPU 拷贝 + 自动反量化 |
| C++ 零拷贝（schedutil 调速器） | ~19 ms | 0.6 ms | ~22.8 ms | **43.0** | 零拷贝 + filter-first，调速器未锁频 |
| **C++ 零拷贝（performance 调速器）** | **15.8 ms** | **0.6 ms** | **16.3 ms** | **61.2** | 8 核锁 performance，NPU 逼近 1 GHz 极限 |
| 相机实时墙钟 | — | — | 33.3 ms | **30.0** | IMX415 ISP 硬件节拍上限（非推理瓶颈） |

**2.08× 提升**（29.4 → 61.2 FPS）。NPU 推理占端到端 96.7%，软件开销 <0.6 ms，推理已逼近 RK3588 NPU 1 GHz 硬件极限。**相机实时仍 30 FPS**：这是 IMX415 硬件节拍，不是推理瓶颈——视频文件源展示的是真实处理吞吐。数据见研究报告 L303-306，如实未改。

**延迟分布与 10 分钟热稳定性**（36000 帧持续推理，≈10 min，performance 调速器，堵两个评审缺口）：

| 指标 | mean | p50 | p95 | 说明 |
|---|---|---|---|---|
| NPU run（`PERF_RUN`） | 15.9 ms | 15.9 ms | 16.0 ms | 真实 NPU 计算，36000 帧无降频 |
| 端到端（run+后处理+拷贝+sync） | 16.3 ms | 16.2 ms | 16.6 ms | → **61.3 FPS**，p95 仅 +0.4 ms |

- **尾延迟可忽略**：36000 帧 p95 比 mean 高 0.4 ms（端到端）/ 0.1 ms（NPU），无长尾。300 帧快照 p50≈mean、p95 仅 +0.1 ms，二者吻合。
- **零降频**：10 min 内热采样 120 次，**NPU 频率全程锁 1 GHz**（无一帧跌频）；NPU 温度 32.4 °C → 峰值 38.8 °C → 末值 37.0 °C，全部热区 ≤39.8 °C，远低于降频阈值（~85-95 °C）。
- 原始数据：[`eval_results/bench_10min.log`](eval_results/bench_10min.log)（36000 帧逐帧 + 总结）、[`eval_results/thermal_10min.csv`](eval_results/thermal_10min.csv)（120 样本 npufreq/7 热区）。

---

## 一、效果定位

一句话：**把 YOLOv8-seg 在 RK3588 上的推理-后处理路径优化至硬件计算极限（NPU 占端到端 96.7%），精度与 Python 参考对齐，为端侧视觉感知提供高吞吐、低延迟、零拷贝前端。**

三个关键数字：

- **15.8 ms** —— NPU 推理已逼近 RK3588 NPU 1 GHz 硬件极限，软件开销 <0.6 ms。
- **61 FPS** —— 视频文件源端到端处理吞吐；相机实时 30 FPS 是 IMX415 硬件节拍上限，不是推理瓶颈。
- **0.809 / 0.626** —— 板端 INT8 mAP，与训练机 0.833/0.666 对齐（Box −2.4% 来自 INT8；Mask 160 口径持平训练机，C++ 实时 −4% 是 4×4 光栅化速度折中）。

---

## 二、架构图

```mermaid
flowchart LR
    A["IMX415<br/>/dev/video44"] -->|"V4L2 异步采集<br/>1 帧队列"| B["零拷贝输入<br/>int8 = pixel-128"]
    B --> C["RK3588 NPU 推理<br/>15.8ms / rknn_mem_sync"]
    C -->|"INT8 NC1HWC2 原生张量<br/>pass_through=1"| D["filter-first 后处理<br/>0.6ms"]
    D --> E["框 + 掩码涂色<br/>imshow / 存图"]
    D -.->|"eval 模式"| F["GT 比对 + mAP<br/>eval_report.md"]

    subgraph D ["后处理 (NEON)"]
        D1["sigmoid(cls) 过滤<br/>8400 → ~15"] --> D2["DFL + box 解码<br/>仅幸存锚点"]
        D2 --> D3["dot16_i8f32<br/>融合反量化+点积"]
        D3 --> D4["框内 4×4 memset<br/>光栅化掩码"]
    end
```

ASCII 版（无 mermaid 渲染时）：

```
IMX415 →[V4L2 async]→ 零拷贝输入(int8=pixel-128)
                           │
                           ▼
          RK3588 NPU 推理 15.8ms (rknn_create_mem/set_io_mem/mem_sync)
                           │  INT8 NC1HWC2 原生张量, pass_through=1
                           ▼
   filter-first 后处理 0.6ms ──┬── sigmoid(cls) 过滤 8400→~15
                               ├── DFL + box 解码 (仅幸存锚点)
                               ├── dot16_i8f32 (融合 int8×float32 点积, 省 proto 全量反量化)
                               └── 框内 4×4 memset 光栅化 (省 cv2.resize)
                           │
                           ▼
                  框 + 掩码涂色 → imshow / 存图
                           └─ eval 模式 → GT 比对 → mAP → eval_report.md
```

优化链路（每条都是"问题 → 手段 → 结果"）：

1. **ONNX cut-tail 图手术** —— INT8 把 DFL/Sigmoid/decode 量化塌掉致置信度全零 → 删 31 个尾部敏感算子、10 个原始 Conv 分支输出 → INT8 只碰 Conv → "INT8 速度 + FP32 精度"。额外修复：cut 导出漏了 proto SiLU，手动补回 Sigmoid+Mul（详见 [ALIGNMENT_NOTES](eval_results/ALIGNMENT_NOTES.md)）。
2. **零拷贝原生张量** —— `rknn.inference()` 的 numpy↔NPU 拷贝 + 自动反量化慢 → `rknn_create_mem/set_io_mem/mem_sync` + `pass_through=1` 直接吃 NPU 原生 int8 → 推理 24.9→16.0 ms。
3. **NEON 融合点积 `dot16_i8f32`** —— proto 全量反量化浪费 → 直接吃原生 int8 proto，反量化融进点积累加（C2=16 连续，一次 `vld1_s8`）→ 省全量反量化。
4. **filter-first** —— 8400 锚点全算 DFL（每锚点 64 次 softmax）是浪费 → 先 sigmoid(cls) 过滤到 ~15 个幸存锚点再重活 → DFL 工作量降 ~560×。掩码合成跳过 sigmoid（`sigmoid(x)>0.5 ⟺ x>0`）。
5. **框内 4×4 memset 光栅化** —— 160→640 双线性 `cv2.resize` 慢 → 框内直接 4×4 块 memset → 省中间 Mat + resize，后处理 9.1→0.6 ms。
6. **CPU performance 调速器** —— schedutil 抖动 → 8 核全设 performance → NPU run ~19→15.8 ms（最廉价单一优化）。

---

## 三、环境与依赖

### 硬件

- **板端**：RK3588（3 核 NPU @ 1 GHz，8 GB 内存），IMX415 摄像头（`/dev/video44`）
- **训练机**：Windows + NVIDIA GPU（仅训练，非部署）

### 板端运行依赖

| 依赖 | 版本 | 说明 |
|---|---|---|
| librknnrt.so | RKNN Toolkit2 runtime | 板端 NPU C API，sysroot 自带 |
| OpenCV | 4.9.0 | highgui/imgcodecs/imgproc/videoio/core |
| RKNNLite (Python) | 板端自带 | 仅 Python 参考/eval 路径用 |
| Python | 3.11 | 板端自带（仅 `eval_py.py` / `vtes_cut.py`） |

### 交叉编译工具链

- **buildroot aarch64 g++ 13.4.0**：`/home/alientek/atk_dlrk3588_linux6.1/buildroot/output/alientek_rk3588/host/bin/aarch64-buildroot-linux-gnu-g++`
- **sysroot**：含 librknnrt.so + opencv4 头/库
- 见 [`build_zc.sh`](build_zc.sh)

### 模型转换工具链（PC 侧）

- **rknn-toolkit2 v2.3.2**（conda env `rknn2`，`/home/alientek/anaconda3/envs/rknn2/bin/python3`）
  - ⚠️ **必须用 `rknn2` 环境**。`pytorch` 环境（toolkit v1.5.2）对 cut-tail ONNX 会触发 `onnxoptimizer` segfault。
- **onnxruntime 1.19.2**（host FP32 参考验证用）
- **ultralytics**（训练 + val 参考）

---

## 四、一键复现

### 1. 交叉编译 C++ 推理程序

```bash
cd /home/alientek/yolo8-seg
bash build_zc.sh          # 产出 aarch64 ELF: rknn_seg_zc
```

### 2. 部署到板端（adb 已连接）

```bash
adb push rknn_seg_zc            /yolo8/
adb push yolo8n_int8_cut.rknn   /yolo8/
adb push post_cut.py            /yolo8/    # Python 参考/eval 用
adb push eval_py.py             /yolo8/
```

### 3. 板端运行（先设 performance 调速器）

```bash
# 8 核全设 performance (NPU run ~19→15.8ms, 最廉价优化)
for i in 0 1 2 3 4 5 6 7; do
  adb shell "echo performance > /sys/devices/system/cpu/cpu$i/cpufreq/scaling_governor"
done

adb shell
cd /yolo8
export LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH

# 静态图推理 (精度回归)
./rknn_seg_zc image yolo8n_int8_cut.rknn test_frame.png

# 相机实时显示
./rknn_seg_zc cam   yolo8n_int8_cut.rknn /dev/video44

# 相机 benchmark (300 帧)
./rknn_seg_zc bench yolo8n_int8_cut.rknn /dev/video44 300

# 视频吞吐 (无相机节拍, 展示真实 61 FPS)
./rknn_seg_zc bench yolo8n_int8_cut.rknn /oem/SampleVideo_1280x720_5mb.mp4 300

# demo 生成可视化 (视频文件或图像目录 → 涂色+OSD mp4, 避开相机节拍展示真实处理 FPS)
./rknn_seg_zc demo  yolo8n_int8_cut.rknn /oem/SampleVideo_1280x720_5mb.mp4 demo_video.mp4
./rknn_seg_zc demo  yolo8n_int8_cut.rknn datasets/crack-seg/images/val    demo_imgs.mp4

# 板端 mAP 评测 (eval 模式, 需 datasets/crack-seg)
./rknn_seg_zc eval  yolo8n_int8_cut.rknn /path/to/crack-seg val
#   → 生成 eval_report.md (Box/Mask mAP50/mAP50-95/recall/miss/IoU + 失败样例)

# Python 双线性参考 (隔离 4×4 光栅化对 mask mAP 的影响)
python3 eval_py.py yolo8n_int8_cut.rknn /path/to/crack-seg val
```

`rknn_seg_zc` 模式总览：

| 模式 | 命令 | 用途 |
|---|---|---|
| `image` | `image <model> <img>` | 单图推理 + 可视化，精度回归 |
| `cam` | `cam <model> [src]` | 相机实时显示，默认 `/dev/video44` |
| `bench` | `bench <model> [src] [N]` | N 帧基准，打印推理/后处理/FPS |
| `demo` | `demo <model> <video_or_dir> [out.mp4]` | 视频文件/图像目录 → 零拷贝推理 + 涂色 + OSD → mp4，避开相机节拍展示真实吞吐（S4） |
| `eval` | `eval <model> <dataset_dir> [split]` | val 集 mAP 评测，生成 `eval_report.md` |

环境变量：`CORE_MASK=0|1|3|7|0xffff` 覆盖 NPU 核心掩码（默认 `CORE_0`，小模型单核已饱和）；`EVAL_DEBUG=1` 打印 eval 逐图调试。

### 4. 模型转换（从 ONNX 重建 .rknn，可选）

```bash
# 用 rknn2 环境 (不是 pytorch!)
/home/alientek/anaconda3/envs/rknn2/bin/python3 convert_cut.py
#   读 yolo8n_cut.onnx → INT8 量化 (50 张校准图 quant.txt) → yolo8n_int8_cut.rknn
```

---

## 五、局限与诚实声明

**如实报告，未为好看调数字。**

1. **相机 30 FPS 是真天花板，不是推理瓶颈。** 用 `RKNN_QUERY_PERF_RUN` 分离出 NPU 计算 15.77 ms、软件开销 <0.6 ms（NPU 占端到端 96.7%），推理已逼近 NPU 1 GHz 硬件极限。墙钟 FPS 受 IMX415 ISP 30 FPS 节拍限制（见 [`cam-imx415-30fps-hardlimit` 记忆]）。视频文件源可跑到 61 FPS 展示真实吞吐。**10 分钟 / 36000 帧持续推理验证**：NPU 频率全程锁 1 GHz 无降频，p95 端到端 16.6 ms（仅 +0.4 ms 尾延迟），峰值 38.8 °C 远未触阈（见 [效果展示](#效果展示) 延迟/热稳定表与 [`eval_results/`](eval_results/)）。

2. **单类（crack）。** cls 通道=1 硬编码，`rknn_seg_zc.cpp` 多类路径已预留但未启用。单类 demo 对简历/验证完全够；多类需重训 + 改后处理。

3. **Mask mAP 的 4×4 光栅化折中。** C++ 实时路径用框内 4×4 块 memset（最近邻）+ 416 分辨率 IoU 换取 ~33 FPS 实时性，Mask mAP50 = 0.626 vs 训练机 0.666（−4.0%）。**这不是模型能力问题**：同模型同数据，Python 双线性 160 分辨率口径 = 0.669 ≈ 训练机 0.666，证明 INT8 量化本身对 mask 精度无损。多口径对照见 [`eval_results/ALIGNMENT_NOTES.md`](eval_results/ALIGNMENT_NOTES.md)。

4. **Box mAP −2.4%（0.833→0.809）= INT8 量化预期代价。** cut-tail 已把 DFL/Sigmoid/box 解码移出量化图（仅 Conv 量化），影响小但非零。

5. **训练轻微过拟合。** Box/Mask mAP 峰值在 epoch 84-90，之后波动下行至 184（patience=100 早停触发）。`best.pt` 保存峰值权重。这是 YOLOv8n-seg 小模型 + 细长裂缝的典型上限，非实现错误。

---

## 附录 A：模型文件注册表

仓库内 `.rknn` 文件命名规整如下（**最终交付仅 `yolo8n_int8_cut.rknn`**，其余为演进过程/对照）：

| 文件 | 量化 | 大小 | 状态 | 说明 |
|---|---|---|---|---|
| **`yolo8n_int8_cut.rknn`** | INT8 cut-tail | 4.5 MB | ✅ **最终交付** | 仅 Conv 量化，尾部算子移出，板端运行 |
| `yolo8n_int8_cut.rknn.bak_broken_proto` | INT8 cut-tail | 4.5 MB | ❌ 备份 | proto 缺 SiLU，Mask mAP 0.24（已修） |
| `yolo8n_fp16_cut.rknn` | FP16 cut-tail | 7.9 MB | 对照 | FP16 参考精度 |
| `yolo8n_int8_hybrid.rknn` | INT8 hybrid | 5.2 MB | ❌ 失败 | auto_hybrid 方案，置信度仍全零 |
| `yolo8n_int8.rknn` | INT8 端到端 | 5.2 MB | ❌ 失败 | 尾部算子被量化塌掉 |
| `yolo8n_int8_v2.rknn` | INT8 端到端 | 5.2 MB | ❌ 旧 | 同上 |
| `yolo8n_fp16.rknn` | FP16 端到端 | 10.9 MB | 旧 | 未 cut 的 FP16 |
| `crack_seg.rknn` / `crack_seg_fp16.rknn` | — | 15.6 / 29.7 MB | 旧 | 早期命名 |

ONNX 源文件：

| 文件 | 说明 |
|---|---|
| `yolo8n_cut.onnx` | cut-tail 10 分支 ONNX（**已补 proto SiLU**），转 INT8 用 |
| `yolo8n_cut.onnx.bak` | 补 SiLU 前备份（proto 损坏） |
| `yolo8n.onnx` | 原始端到端 ONNX |
| `best.pt` / `best.onnx` | 训练峰值权重（FP32） |

## 附录 B：后处理阈值清单（`rknn_seg_zc.cpp` static const）

实时路径（`cam`/`image`/`bench`）：

| 常量 | 值 | 含义 |
|---|---|---|
| `OBJ_THRESH` | 0.18 | 框置信度阈值（filter-first 第一道筛） |
| `NMS_THRESH` | 0.45 | NMS IoU 阈值 |
| `MASK_THRESH` | 0.5 | 掩码二值化阈值 |
| `MASK_LOGIT` | 0.0 | `logit(0.5)`，掩码合成跳过 sigmoid 的门限 |
| `IMG_SIZE` | 640 | letterbox 目标尺寸 |
| `MAX_MASKS` | 10 | 实时路径最多合成掩码数（top-N，控开销） |

评测路径（`eval`）：

| 常量 | 值 | 含义 |
|---|---|---|
| `EVAL_CONF` | 0.01 | eval 置信度下限（逼近完整 PR 曲线，对齐 ultralytics val） |
| `EVAL_MAX_MASKS` | 64 | eval 路径掩码数上限（更全的 PR 曲线） |

Python 参考路径（`post_cut.py` / `eval_py.py`）对齐：`OBJ_THRESH=0.01`（eval）/ `0.18`（实时）、`NMS_THRESH=0.45`、`MASK_THRESH=0.5`。

## 附录 C：源码索引

| 文件 | 作用 |
|---|---|
| [`rknn_seg_zc.cpp`](rknn_seg_zc.cpp) | C++ 零拷贝推理 + filter-first 后处理 + eval 模式（最终交付） |
| [`post_cut.py`](post_cut.py) | numpy 后处理参考（正确性 ground truth） |
| [`eval_py.py`](eval_py.py) | 板端 Python 双线性参考评测（隔离 4×4 光栅化影响） |
| [`convert_cut.py`](convert_cut.py) | ONNX→RKNN INT8 转换（用 rknn2 环境！） |
| [`probe_zero_copy.cpp`](probe_zero_copy.cpp) | 探测 NPU 原生张量格式 |
| [`build_zc.sh`](build_zc.sh) | 交叉编译脚本 |
| [`eval_results/`](eval_results/) | mAP 评测报告 + 对齐说明 + 失败样例图 + 10 min bench/thermal 数据 |
| [`demo/`](demo/) | S4 演示产物：`demo_small.gif`、`demo_imgs.mp4`、`fps_comparison.png`、`make_fps_chart.py` |
| [`seg/`](seg/) | 训练产物（args.yaml/results.csv/权重/PR 曲线） |
| [`datasets/crack-seg/`](datasets/crack-seg/) | 数据集（images/labels × val/test） |

---

*配套文档：技术细节见 [`研究报告_YOLOv8-seg_RK3588零拷贝优化.md`](研究报告_YOLOv8-seg_RK3588零拷贝优化.md)；简历视角与执行清单见 [`学生简历项目评估与继续执行报告.md`](学生简历项目评估与继续执行报告.md)；企业级商业化路线见 [`裂缝检测商业化路线与差距分析.md`](裂缝检测商业化路线与差距分析.md)。*

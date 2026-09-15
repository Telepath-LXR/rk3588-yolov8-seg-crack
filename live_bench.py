"""
板端实时分割显示 + 全程指标采集 (INT8 / FP16 通用)
================================================================
功能:
  1. 全屏自适应屏幕实时显示 (绿框 + 半透明红掩码 + OSD)
  2. 全程采集性能/硬件指标
  3. 按 Ctrl+C 或 q/ESC 退出时, 把所有指标写入报告文件
     - 文本报告  bench_report_<tag>_<时间>.txt   (人读)
     - JSON 报告 bench_report_<tag>_<时间>.json   (机器读/对比)

用法 (板子, 经 bench_run.sh 启动):
  ./bench_run.sh int8                 # INT8 模型 + 摄像头
  ./bench_run.sh fp16                 # FP16 模型 + 摄像头
  ./bench_run.sh int8 /dev/video0     # 指定摄像头
  ./bench_run.sh fp16 /yolo8/x.mp4   # 测视频文件

指标采集内容:
  模型规格: 文件大小 / 输入输出张量(dtype/scale/zp/shape) / toolkit+runtime版本
  效率:   推理ms 后处理ms 端到端ms FPS (mean/p50/p95/min/max)
  检测:   命中帧数 平均检测数 平均最高置信度 平均掩码像素
  硬件:   NPU负载(3核) NPU频率 DDR频率 温度(npu/gpu/bigcore) 内存RSS/MemAvailable
          (每隔 ~1s 采样一次, 取均值/峰值)
"""
import sys, os, time, json, signal, threading, subprocess, gc, statistics
import cv2
import numpy as np
from rknnlite.api import RKNNLite
try:
    from rknnlite.api import rknn_perf
except Exception:
    rknn_perf = None
from post_cut import post_process

# ---------- 模型选择 ----------
MODELS = {
    "int8": "./yolo8n_int8_cut.rknn",
    "fp16": "./yolo8n_fp16_cut.rknn",
}
CAMERA_DEV = "/dev/video44"
MODEL_SIZE = (640, 640)
REPORT_DIR = "."     # 报告写到板子当前目录

# ---------- 解析参数 ----------
if len(sys.argv) < 2 or sys.argv[1] not in MODELS:
    print("用法: python3 live_bench.py <int8|fp16> [视频源]")
    print("  视频源: /dev/videoX  或  视频文件路径  (默认 %s)" % CAMERA_DEV)
    sys.exit(1)
TAG = sys.argv[1]
RKNN_MODEL = MODELS[TAG]
src = sys.argv[2] if len(sys.argv) > 2 else CAMERA_DEV
use_cam = src.startswith("/dev/video") or not os.path.splitext(src)[1]

# =========================================================
#  指标容器 (线程安全由 GIL 保证简单累加)
# =========================================================
class Stats:
    def __init__(self):
        self.t0 = time.time()
        self.infer_ms = []
        self.post_ms = []
        self.e2e_ms = []          # 推理+后处理
        self.frame_count = 0
        self.det_count_list = []
        self.max_conf_list = []
        self.mask_px_list = []
        # 硬件采样 (后台线程)
        self.npu_load = {"core0": [], "core1": [], "core2": []}
        self.npu_freq = []
        self.gpu_freq = []
        self.ddr_freq = []
        self.temp = {}
        self.rss_kb = []
        self.mem_avail_kb = []
        self.last_run_perf_us = []   # rknn get_run_perf 单帧 us
        self.memory_detail = None    # 模型内存占用
        self.stop = False

S = Stats()

# =========================================================
#  硬件指标读取函数
# =========================================================
def _read_int(path):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except Exception:
        return None

def _read_npu_load():
    """读 /sys/kernel/debug/rknpu/load, 返回 (c0,c1,c2) 百分比"""
    try:
        with open("/sys/kernel/debug/rknpu/load") as f:
            txt = f.read()
        import re
        m = re.findall(r"Core\d:\s*(\d+)%", txt)
        if len(m) >= 3:
            return int(m[0]), int(m[1]), int(m[2])
    except Exception:
        pass
    return None, None, None

def _read_temps():
    """thermal_zone → {type: 摄氏度}"""
    out = {}
    import glob
    for tz in glob.glob("/sys/class/thermal/thermal_zone*"):
        try:
            t = open(tz + "/type").read().strip()
            v = int(open(tz + "/temp").read().strip())
            out[t] = round(v / 1000.0, 1)
        except Exception:
            pass
    return out

def _rss_kb():
    """本进程 RSS (kB)"""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS"):
                    return int(line.split()[1])
    except Exception:
        pass
    return None

def _mem_avail_kb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable"):
                    return int(line.split()[1])
    except Exception:
        pass
        return None
    return None

def hw_sampler():
    """后台线程: 每 1s 采一次硬件指标"""
    while not S.stop:
        c0, c1, c2 = _read_npu_load()
        if c0 is not None:
            S.npu_load["core0"].append(c0)
            S.npu_load["core1"].append(c1)
            S.npu_load["core2"].append(c2)
        f = _read_int("/sys/class/devfreq/fdab0000.npu/cur_freq")
        if f is not None: S.npu_freq.append(f)
        g = _read_int("/sys/class/devfreq/fb000000.gpu/cur_freq")
        if g is not None: S.gpu_freq.append(g)
        d = _read_int("/sys/class/devfreq/dmc/cur_freq")
        if d is not None: S.ddr_freq.append(d)
        for k, v in _read_temps().items():
            S.temp.setdefault(k, []).append(v)
        r = _rss_kb()
        if r is not None: S.rss_kb.append(r)
        ma = _mem_avail_kb()
        if ma is not None: S.mem_avail_kb.append(ma)
        time.sleep(1.0)

# =========================================================
#  模型加载
# =========================================================
print("=" * 56)
print("  模式: %s   模型: %s" % (TAG.upper(), RKNN_MODEL))
print("  输入: %s" % src)
print("=" * 56)
if not os.path.exists(RKNN_MODEL):
    print("模型文件不存在:", RKNN_MODEL); sys.exit(1)

rknn = RKNNLite()
if rknn.load_rknn(RKNN_MODEL) != 0:
    print("load_rknn 失败"); sys.exit(1)
ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
if ret != 0:
    print("init_runtime 失败:", ret); sys.exit(1)
print("模型加载成功, 输入源:", src)

rt = rknn.rknn_runtime
TYPE_MAP = {0: "float32", 1: "float16", 2: "int8", 3: "uint8",
            4: "int16", 5: "uint16", 6: "int32", 7: "uint32"}

# ---------- 收集模型规格 ----------
model_info = {"tag": TAG, "path": RKNN_MODEL,
              "size_mb": round(os.path.getsize(RKNN_MODEL) / 1e6, 2),
              "core_mask": "NPU_CORE_0_1_2"}
try:
    sdk_ver = rknn.get_sdk_version()
    model_info["sdk_version"] = sdk_ver
except Exception:
    pass
try:
    n_in, n_out = rt.get_in_out_num()
    model_info["n_inputs"] = n_in
    model_info["n_outputs"] = n_out
    outs = []
    for i in range(n_out):
        a = rt.get_tensor_attr(i, True)   # True=输出
        dims = [a.dims[j] for j in range(a.n_dims)]
        zp = a.zp
        if zp > 2147483647: zp -= 4294967296
        outs.append({"index": i, "name": a.name.decode() if isinstance(a.name, bytes) else a.name,
                     "dims": dims, "dtype": TYPE_MAP.get(a.type, a.type),
                     "qnt_type": a.qnt_type, "scale": a.scale, "zp": zp, "fl": a.fl})
    model_info["outputs"] = outs
    print("输出张量属性已采集: %d 个输出, dtype=%s" % (n_out, outs[0]["dtype"]))
except Exception as e:
    print("采集输出属性失败:", repr(e))

# 模型内存占用 (INT8 才有量化内存信息, FP16 通常 0)
if rknn_perf is not None:
    try:
        # 先跑一帧 warmup + 拿 perf 字符串
        _warm = np.expand_dims(np.zeros((640, 640, 3), np.uint8), 0)
        rknn.inference([_warm])
        perf_str = rt.get_run_perf()
        md = rknn_perf.collect_memory_detail(perf_str)
        S.memory_detail = json.loads(json.dumps(md))   # OrderedDict -> dict
        print("模型内存占用采集成功")
    except Exception as e:
        print("模型内存占用采集失败:", repr(e))

# =========================================================
#  输入源 / 窗口
# =========================================================
cap = cv2.VideoCapture(src, cv2.CAP_V4L2 if use_cam else 0)
if not cap.isOpened():
    print("打开输入失败:", src); sys.exit(1)

# 屏幕尺寸
def get_screen_size():
    for p in ("/sys/class/drm/card0-DSI-1/modes", "/sys/class/drm/card0-HDMI-A-1/modes"):
        try:
            line = open(p).readline().strip()
            if "x" in line:
                w, h = map(int, line.split("x"))
                return max(w, h), min(w, h)
        except Exception:
            pass
    return 1080, 1920
scr_w, scr_h = get_screen_size()
print("屏幕尺寸(横放): %dx%d" % (scr_w, scr_h))

WIN = "crack_seg_bench"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# =========================================================
#  画面处理
# =========================================================
def letter_box(im, new_shape=640):
    shape = im.shape[:2]
    r = min(new_shape / shape[0], new_shape / shape[1])
    nu = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape - nu[0], new_shape - nu[1]; dw /= 2; dh /= 2
    if shape[::-1] != nu:
        im = cv2.resize(im, nu)
    t, b = int(round(dh - 0.1)), int(round(dh + 0.1))
    l, rr = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(im, t, b, l, rr, cv2.BORDER_CONSTANT, value=(0, 0, 0))

def draw(frame, boxes, confs, mask):
    out = frame.copy()
    if mask is not None and mask.size and (mask > 0).any():
        red = np.zeros_like(out)
        red[mask > 0] = (0, 0, 255)
        out = cv2.addWeighted(out, 1.0, red, 0.45, 0)
    for k in range(len(boxes)):
        x1, y1, x2, y2 = boxes[k].astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, "crack %.2f" % confs[k], (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out

def fit_screen(frame, sw, sh):
    fh, fw = frame.shape[:2]
    s = min(sw / fw, sh / fh)
    nw, nh = int(fw * s), int(fh * s)
    canvas = np.zeros((sh, sw, 3), np.uint8)
    canvas[(sh - nh)//2:(sh - nh)//2 + nh,
           (sw - nw)//2:(sw - nw)//2 + nw] = cv2.resize(frame, (nw, nh))
    return canvas

# =========================================================
#  Ctrl+C / 退出处理 → 写报告
# =========================================================
def summarize():
    S.stop = True
    t1 = time.time()
    dur = t1 - S.t0
    rep = {
        "model": model_info,
        "run": {"tag": TAG, "source": src,
                "duration_s": round(dur, 2),
                "frames": S.frame_count,
                "fps_mean": round(S.frame_count / dur, 2) if dur > 0 else 0},
    }
    # 效率统计
    def stat(arr):
        if not arr: return None
        arr2 = sorted(arr)
        n = len(arr2)
        return {"mean": round(statistics.mean(arr), 2),
                "p50":  round(arr2[n//2], 2),
                "p95":  round(arr2[int(n*0.95)] if n > 1 else arr2[-1], 2),
                "min":  round(min(arr), 2),
                "max":  round(max(arr), 2),
                "n":    len(arr)}
    rep["latency"] = {
        "infer_ms": stat(S.infer_ms),
        "post_ms":  stat(S.post_ms),
        "e2e_ms":   stat(S.e2e_ms),
    }
    # 检测统计
    rep["detection"] = {
        "det_frames": sum(1 for x in S.det_count_list if x > 0),
        "total_frames": S.frame_count,
        "hit_rate": round(100.0 * sum(1 for x in S.det_count_list if x > 0) / max(1, S.frame_count), 1),
        "avg_obj_count": round(statistics.mean(S.det_count_list), 2) if S.det_count_list else 0,
        "avg_max_conf": round(statistics.mean(S.max_conf_list), 3) if S.max_conf_list else 0,
        "avg_mask_px": int(statistics.mean(S.mask_px_list)) if S.mask_px_list else 0,
    }
    # 硬件统计
    def hs(arr, scale=1.0, unit=""):
        if not arr: return None
        return {"mean": round(statistics.mean(arr) * scale, 2),
                "peak": round(max(arr) * scale, 2),
                "n": len(arr), "unit": unit}
    rep["hardware"] = {
        "npu_load_pct": {
            "core0": hs(S.npu_load["core0"]),
            "core1": hs(S.npu_load["core1"]),
            "core2": hs(S.npu_load["core2"]),
        },
        "npu_freq_mhz":   hs(S.npu_freq, 1e-6, "MHz"),
        "gpu_freq_mhz":   hs(S.gpu_freq, 1e-6, "MHz"),
        "ddr_freq_mhz":   hs(S.ddr_freq, 1e-6, "MHz"),
        "temperature_c":  {k: hs(v) for k, v in S.temp.items()},
        "process_rss_mb": hs(S.rss_kb, 1/1024, "MB"),
        "mem_available_mb": hs(S.mem_avail_kb, 1/1024, "MB"),
        "model_memory_detail": S.memory_detail,
        "rknn_get_run_perf_us_last": S.last_run_perf_us[-1] if S.last_run_perf_us else None,
        "rknn_get_run_perf_us_mean": round(statistics.mean(S.last_run_perf_us)) if S.last_run_perf_us else None,
    }
    return rep

def write_reports(rep):
    ts = time.strftime("%Y%m%d_%H%M%S")
    base = os.path.join(REPORT_DIR, "bench_report_%s_%s" % (TAG, ts))
    # JSON
    with open(base + ".json", "w") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    # 文本
    lines = []
    lines.append("=" * 60)
    lines.append("  RKNN 板端性能报告  |  模型: %s (%s)" % (TAG.upper(), rep["model"]["path"]))
    lines.append("  生成时间: %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    lines.append("=" * 60)
    m = rep["model"]
    lines.append("\n[模型规格]")
    lines.append("  类型:           %s" % TAG.upper())
    lines.append("  文件大小:       %.2f MB" % m["size_mb"])
    lines.append("  SDK 版本:       %s" % m.get("sdk_version", "?"))
    lines.append("  输入/输出:      %d / %d" % (m.get("n_inputs", 0), m.get("n_outputs", 0)))
    if m.get("outputs"):
        lines.append("  输出张量:")
        for o in m["outputs"]:
            lines.append("    out%-2d %-22s %s %-18s scale=%s zp=%s" %
                         (o["index"], str(o["dims"]), o["dtype"],
                          o["name"].split("/")[-1], o["scale"], o["zp"]))
    r = rep["run"]
    lines.append("\n[运行概况]")
    lines.append("  输入源:         %s" % r["source"])
    lines.append("  运行时长:       %.1f s" % r["duration_s"])
    lines.append("  总帧数:         %d" % r["frames"])
    lines.append("  实测 FPS:       %.2f" % r["fps_mean"])
    L = rep["latency"]
    for name, d in (("推理 infer", L["infer_ms"]), ("后处理 post", L["post_ms"]), ("端到端 e2e", L["e2e_ms"])):
        if d:
            lines.append("  %s:  mean=%6.2f  p50=%6.2f  p95=%6.2f  min=%6.2f  max=%6.2f ms  (n=%d)" %
                         (name, d["mean"], d["p50"], d["p95"], d["min"], d["max"], d["n"]))
    D = rep["detection"]
    lines.append("\n[检测效果]")
    lines.append("  命中帧数:       %d / %d (%.1f%%)" % (D["det_frames"], D["total_frames"], D["hit_rate"]))
    lines.append("  平均目标数:     %.2f" % D["avg_obj_count"])
    lines.append("  平均最高置信度: %.3f" % D["avg_max_conf"])
    lines.append("  平均掩码像素:   %d" % D["avg_mask_px"])
    H = rep["hardware"]
    lines.append("\n[硬件指标] (后台每秒采样)")
    npl = H["npu_load_pct"]
    for c in ("core0", "core1", "core2"):
        d = npl[c]
        if d: lines.append("  NPU %s 负载:  mean=%5.1f%%  peak=%5.1f%%" % (c, d["mean"], d["peak"]))
    for k, lab in (("npu_freq_mhz","NPU 频率"),("gpu_freq_mhz","GPU 频率"),("ddr_freq_mhz","DDR 频率")):
        d = H[k]
        if d: lines.append("  %-9s     mean=%7.1f  peak=%7.1f %s" % (lab, d["mean"], d["peak"], d["unit"]))
    for k, d in H["temperature_c"].items():
        if d: lines.append("  温度 %-14s mean=%5.1f  peak=%5.1f C" % (k, d["mean"], d["peak"]))
    d = H["process_rss_mb"]
    if d: lines.append("  进程 RSS:       mean=%7.1f  peak=%7.1f MB" % (d["mean"], d["peak"]))
    d = H["mem_available_mb"]
    if d: lines.append("  系统可用内存:   mean=%7.1f  peak=%7.1f MB" % (d["mean"], d["peak"]))
    if H["model_memory_detail"]:
        md = H["model_memory_detail"]
        lines.append("  模型内存占用:")
        for cat in ("system_memory", "npu_memory", "total_memory"):
            if cat in md and md[cat]:
                lines.append("    %-14s 最大=%d  总计=%d" % (cat, md[cat].get("maximum_allocation",0), md[cat].get("total_allocation",0)))
    if H["rknn_get_run_perf_us_mean"]:
        lines.append("  rknn 单帧耗时(get_run_perf): mean=%d us" % H["rknn_get_run_perf_us_mean"])
    lines.append("\n" + "=" * 60)
    txt = "\n".join(lines)
    with open(base + ".txt", "w") as f:
        f.write(txt)
    print("\n" + txt)
    print("\n报告已写入:")
    print("  %s.txt" % base)
    print("  %s.json" % base)
    return base

# 注册 Ctrl+C
def on_sigint(signum, frame):
    raise KeyboardInterrupt
signal.signal(signal.SIGINT, on_sigint)

# =========================================================
#  主循环
# =========================================================
threading.Thread(target=hw_sampler, daemon=True).start()
print("\n实时显示已启动, 采样中...  (Ctrl+C 或 q/ESC 退出)\n")
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            if not use_cam:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
            break
        h, w = frame.shape[:2]
        img = letter_box(frame.copy(), MODEL_SIZE[0])
        inp = np.expand_dims(img, axis=0)

        t1 = time.time()
        outputs = rknn.inference([inp])
        t_infer = time.time() - t1

        t2 = time.time()
        res = post_process(outputs, w, h)
        t_post = time.time() - t2

        # rknn 单帧 perf (轻量, 偶尔采以免拖慢)
        if S.frame_count % 30 == 0:
            try:
                p = rt.get_run_perf()
                S.last_run_perf_us.append(int(p))
            except Exception:
                pass

        # 记录指标
        S.infer_ms.append(t_infer * 1000)
        S.post_ms.append(t_post * 1000)
        S.e2e_ms.append((t_infer + t_post) * 1000)
        S.frame_count += 1
        if res is None:
            anno = frame
            S.det_count_list.append(0)
            S.max_conf_list.append(0)
            S.mask_px_list.append(0)
            info = "no detect"
        else:
            boxes, confs, mask = res
            S.det_count_list.append(len(confs))
            S.max_conf_list.append(float(confs.max()))
            S.mask_px_list.append(int((mask > 0).sum()))
            info = "%d obj  max=%.2f  mask=%d" % (len(confs), confs.max(), int((mask>0).sum()))
            anno = draw(frame, boxes, confs, mask)

        show = fit_screen(anno, scr_w, scr_h)
        fps = 1.0 / (t_infer + t_post) if (t_infer + t_post) > 0 else 0
        cv2.putText(show, "[%s] infer %.0fms post %.0fms  %.1fFPS  %s"
                    % (TAG.upper(), t_infer*1000, t_post*1000, fps, info),
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(WIN, show)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        del outputs, inp, img, show
        if S.frame_count % 20 == 0:
            gc.collect()
except KeyboardInterrupt:
    print("\n[Ctrl+C] 正在汇总指标...")
finally:
    cap.release()
    cv2.destroyAllWindows()
    rknn.release()
    rep = summarize()
    write_reports(rep)

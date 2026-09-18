# live_spool.py
"""
相机实时灰区采样（端侧，Python）—— rknn_seg_zc.cpp::cloud_sample 的 Python 实时镜像。

读 /dev/videoX 或视频文件 + RKNNLite 加载 yolo8n_int8_cut.rknn + post_cut.post_process，
对灰区 [OBJ_THRESH, CLOUD_HI) 的检测做与 C++ cloud_sample 完全同口径的采样：
  外扩 30% / resize 长边 512 / JPEG q85 / 5s+20px 去重 / <ts>_<frame_id> 命名 / 原子写
产物落入 spool/outbox/，供 cloud_bridge.py 上传到云端 review.py 复核。

用法（板端，从仓库根目录运行以解析 post_cut 路径）：
  python3 cloud/live_spool.py <model.rknn> [src] [N]
    model  yolo8n_int8_cut.rknn 等
    src    /dev/video44 或视频文件 (默认 /dev/video44)
    N      采样帧数上限, 0=无限直到 Ctrl+C (默认 0)
环境变量:
  SPOOL_DIR      outbox 根 (默认 spool, 从 cloud/ 运行解析为 cloud/spool)
  CLOUD_HI       灰区上界 (默认 0.45, 与 C++ CLOUD_CONF_HI 对齐)
  OBJ_THRESH     置信度下界 (默认 0.18, 与 C++/post_cut 对齐)
  DEVICE_ID      写入 meta.source 的设备标识 (默认 live-spool-01)
  HEADLESS=1     不开显示窗口
  CORE_MASK      0|1|3|7|0xffff (默认 NPU_CORE_0_1_2)
"""
import os
import sys
import time
import json
import random
import signal
from pathlib import Path

import cv2
import numpy as np

# 让 cloud/ 子目录能 import 根目录的 post_cut
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from post_cut import post_process, OBJ_THRESH, IMG_SIZE  # noqa: E402
try:
    from rknnlite.api import RKNNLite
except Exception:
    RKNNLite = None  # PC 无 rknnlite 时只做语法检查

# ---- 配置 ----
SPOOL = os.environ.get("SPOOL_DIR", "spool")
OUTBOX = os.path.join(SPOOL, "outbox")
CLOUD_HI = float(os.environ.get("CLOUD_HI", "0.45"))
OBJ_T = float(os.environ.get("OBJ_THRESH", str(OBJ_THRESH)))
DEVICE_ID = os.environ.get("DEVICE_ID", "live-spool-01")
HEADLESS = os.environ.get("HEADLESS", "0") == "1"
MODEL_VERSION = "yolov8n-seg-int8-cut"
DEDUP_WINDOW_MS = 5000
DEDUP_DIST_PX = 20
CORE_MASK_ENV = os.environ.get("CORE_MASK", "0_1_2")
MODEL_SIZE = IMG_SIZE  # (w,h) 640

os.makedirs(OUTBOX, exist_ok=True)

# ---- 解析参数 ----
if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
    print(__doc__); sys.exit(0)
RKNN_MODEL = sys.argv[1]
src = sys.argv[2] if len(sys.argv) > 2 else "/dev/video44"
max_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 0
use_cam = src.startswith("/dev/video") or not os.path.splitext(src)[1]

# ---- 去重 ----
recent = []  # [(cx, cy, t_ms)]

def is_duplicate(cx, cy, now_ms):
    for d in recent:
        if now_ms - d[2] < DEDUP_WINDOW_MS and \
           abs(d[0] - cx) < DEDUP_DIST_PX and abs(d[1] - cy) < DEDUP_DIST_PX:
            return True
    recent.append((cx, cy, now_ms))
    if len(recent) > 64:
        recent.pop(0)
    return False

def spool_paused():
    return os.path.exists(os.path.join(SPOOL, "PAUSE"))

def sample(frame, bbox, conf, frame_id):
    """与 C++ cloud_sample 同口径：外扩30% / 512 / q85 / 原子写。"""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return False
    rx1 = max(0, int(x1 - w * 0.3))
    ry1 = max(0, int(y1 - h * 0.3))
    rx2 = min(frame.shape[1], int(x2 + w * 0.3))
    ry2 = min(frame.shape[0], int(y2 + h * 0.3))
    crop = frame[ry1:ry2, rx1:rx2]
    if crop.size == 0:
        return False
    scale = 512.0 / max(crop.shape[:2])
    if scale < 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    ts_ms = int(time.time() * 1000) + random.randint(0, 999)
    event_name = f"{ts_ms}_{frame_id}"
    tmp_dir = Path(OUTBOX) / f".{event_name}.tmp"
    final_dir = Path(OUTBOX) / event_name
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(tmp_dir / "crop.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
    meta = {
        "conf": float(conf),
        "bbox": [int(x1), int(y1), int(x2), int(y2)],
        "frame_id": frame_id,
        "source": DEVICE_ID,
        "model_version": MODEL_VERSION,
        "mono_ms": int(time.monotonic() * 1000),
        "ts": time.time(),
    }
    with open(tmp_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    tmp_dir.rename(final_dir)
    return True

# ---- letterbox / 显示 ----
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

def draw(frame, boxes, confs, gray_idx=()):
    out = frame.copy()
    for k in range(len(boxes)):
        x1, y1, x2, y2 = boxes[k].astype(int)
        if k in gray_idx:
            color, tag = (0, 165, 255), "gray→cloud"   # 橙：灰区采样
        else:
            color, tag = (0, 255, 0), "crack"          # 绿：高置信
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, "%s %.2f" % (tag, confs[k]), (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out

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

def fit_screen(frame, sw, sh):
    fh, fw = frame.shape[:2]
    s = min(sw / fw, sh / fh)
    nw, nh = int(fw * s), int(fh * s)
    canvas = np.zeros((sh, sw, 3), np.uint8)
    canvas[(sh - nh)//2:(sh - nh)//2 + nh,
           (sw - nw)//2:(sw - nw)//2 + nw] = cv2.resize(frame, (nw, nh))
    return canvas

# ---- 模型加载 ----
if RKNNLite is None:
    print("[live_spool] 缺 rknnlite，仅支持语法检查/PC 联调，无法真跑。"); sys.exit(1)
if not os.path.exists(RKNN_MODEL):
    print("[live_spool] 模型不存在:", RKNN_MODEL); sys.exit(1)

rknn = RKNNLite()
if rknn.load_rknn(RKNN_MODEL) != 0:
    print("[live_spool] load_rknn 失败"); sys.exit(1)
core = getattr(RKNNLite, "NPU_CORE_" + CORE_MASK_ENV, RKNNLite.NPU_CORE_0_1_2)
if rknn.init_runtime(core_mask=core) != 0:
    print("[live_spool] init_runtime 失败"); sys.exit(1)
print(f"[live_spool] 模型 {RKNN_MODEL} 已加载，输入源 {src}，outbox={OUTBOX}")
print(f"[live_spool] 灰区 [{OBJ_T}, {CLOUD_HI})，去重 {DEDUP_WINDOW_MS}ms+{DEDUP_DIST_PX}px")

# ---- 输入源 / 窗口 ----
cap = cv2.VideoCapture(src, cv2.CAP_V4L2 if use_cam else 0)
if not cap.isOpened():
    print("[live_spool] 打开输入失败:", src); sys.exit(1)
scr_w, scr_h = get_screen_size()
WIN = "live_spool"
if not HEADLESS:
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

frame_id = 0
sampled = 0
t_start = time.time()
paused_note = ""

def on_sigint(*_):
    raise KeyboardInterrupt
signal.signal(signal.SIGINT, on_sigint)

print("[live_spool] 实时采样中... (Ctrl+C 或 q/ESC 退出)\n")
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
        t_infer = (time.time() - t1) * 1000

        t2 = time.time()
        res = post_process(outputs, w, h)
        t_post = (time.time() - t2) * 1000
        frame_id += 1

        gray_idx = set()
        if res is not None:
            boxes, confs, _ = res
            if spool_paused():
                paused_note = " [PAUSE]"
            else:
                paused_note = ""
                for k in range(len(confs)):
                    c = float(confs[k])
                    if not (OBJ_T <= c < CLOUD_HI):
                        continue
                    x1, y1, x2, y2 = boxes[k]
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    now_ms = int(time.monotonic() * 1000)
                    if is_duplicate(cx, cy, now_ms):
                        continue
                    if sample(frame, (x1, y1, x2, y2), c, frame_id):
                        sampled += 1
                        gray_idx.add(k)
            anno = draw(frame, boxes, confs, gray_idx)
            info = "%d obj  gray=%d  sampled=%d%s" % (len(confs), len(gray_idx), sampled, paused_note)
        else:
            anno = frame
            info = "no detect  sampled=%d%s" % (sampled, paused_note)

        show = fit_screen(anno, scr_w, scr_h)
        fps = 1000.0 / (t_infer + t_post) if (t_infer + t_post) > 0 else 0
        cv2.putText(show, "[live_spool] infer %.0fms post %.0fms  %.1fFPS  %s"
                    % (t_infer, t_post, fps, info),
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if not HEADLESS:
            cv2.imshow(WIN, show)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
        if max_frames and frame_id >= max_frames:
            print("\n[live_spool] 达到帧数限制 %d" % max_frames)
            break
except KeyboardInterrupt:
    print("\n[live_spool] Ctrl+C 退出...")
finally:
    cap.release()
    if not HEADLESS:
        cv2.destroyAllWindows()
    rknn.release()
    dur = time.time() - t_start
    print("[live_spool] 结束: %d 帧 / %.1fs, 采样灰区 %d 个 → %s"
          % (frame_id, dur, sampled, OUTBOX))

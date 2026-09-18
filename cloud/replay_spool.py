# replay_spool.py
"""用 best.pt 对 val 集跑推理，模拟 C++ 的灰区采样逻辑。

输出到 spool/outbox/，供 cloud_bridge.py 上传。
这是 rknn_seg_zc.cpp::cloud_sample 的 Python 离线参考实现，
参数（30% 外扩、512 长边、q85、5s/20px 去重、<ts>_<frame_id> 命名）
与 C++ 逐项对齐，便于在无板子时打通云端链路。

用法:
  python3 replay_spool.py [--model best.pt] [--val datasets/crack-seg/images/val]
                          [--max 50] [--spool spool/outbox]
环境变量 SPOOL_DIR 也可覆盖 outbox 目录。
"""
import argparse
import cv2
import json
import os
import time
import random
from pathlib import Path
from ultralytics import YOLO

ap = argparse.ArgumentParser(description="离线回放灰区采样")
ap.add_argument("--model", default="best.pt", help="YOLO 训练权重 (默认 best.pt)")
ap.add_argument("--val", default="datasets/crack-seg/images/val", help="val 集目录")
ap.add_argument("--max", type=int, default=50, help="最多处理图片数")
ap.add_argument("--spool", default=os.environ.get("SPOOL_DIR", "spool/outbox"),
                help="outbox 目录 (默认 spool/outbox, 或 env SPOOL_DIR)")
ap.add_argument("--obj", type=float, default=0.18, help="置信度下界 (与 C++ OBJ_THRESH 一致)")
ap.add_argument("--hi", type=float, default=0.45, help="灰区上界 CLOUD_HI")
ap.add_argument("--model-version", default="yolov8n-seg-int8-cut",
                help="写入 meta 的 model_version 字段")
args = ap.parse_args()

MODEL_PATH = args.model
VAL_DIR = args.val
SPOOL_DIR = args.spool
OBJ_THRESH = args.obj
CLOUD_HI = args.hi
MODEL_VERSION = args.model_version
MAX_IMAGES = args.max

os.makedirs(SPOOL_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

# ===== 去重状态 =====
recent = []  # [(cx, cy, t_ms)]
DEDUP_WINDOW_MS = 5000
DEDUP_DIST_PX = 20

def is_duplicate(cx, cy, now_ms):
    for d in recent:
        if now_ms - d[2] < DEDUP_WINDOW_MS and \
           abs(d[0] - cx) < DEDUP_DIST_PX and abs(d[1] - cy) < DEDUP_DIST_PX:
            return True
    recent.append((cx, cy, now_ms))
    if len(recent) > 64:
        recent.pop(0)
    return False

def sample(frame, bbox, conf, frame_id, source_name):
    """模拟 C++ cloud_sample：外扩30%、resize 512、JPEG q85、原子写。"""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return

    # 外扩 30%
    rx1 = max(0, int(x1 - w * 0.3))
    ry1 = max(0, int(y1 - h * 0.3))
    rx2 = min(frame.shape[1], int(x2 + w * 0.3))
    ry2 = min(frame.shape[0], int(y2 + h * 0.3))
    crop = frame[ry1:ry2, rx1:rx2]
    if crop.size == 0:
        return

    # resize 长边 512
    scale = 512.0 / max(crop.shape[:2])
    if scale < 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # 原子写
    ts_ms = int(time.time() * 1000) + random.randint(0, 999)
    event_name = f"{ts_ms}_{frame_id}"
    tmp_dir = Path(SPOOL_DIR) / f".{event_name}.tmp"
    final_dir = Path(SPOOL_DIR) / event_name
    tmp_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(tmp_dir / "crop.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
    meta = {
        "conf": float(conf),
        "bbox": [int(x1), int(y1), int(x2), int(y2)],
        "frame_id": frame_id,
        "source": source_name,
        "model_version": MODEL_VERSION,
        "mono_ms": int(time.monotonic() * 1000),
        "ts": time.time(),
    }
    with open(tmp_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    tmp_dir.rename(final_dir)
    print(f"  [采样] conf={conf:.3f} bbox={bbox} → {final_dir.name}")

# ===== 主循环 =====
if not os.path.isdir(VAL_DIR):
    raise SystemExit(f"val 目录不存在: {VAL_DIR}")
files = sorted([f for f in os.listdir(VAL_DIR) if f.lower().endswith((".jpg", ".png"))])[:MAX_IMAGES]
print(f"回放 {len(files)} 张 val 图，模型 {MODEL_PATH}，outbox={SPOOL_DIR}")

total_sampled = 0
for idx, fname in enumerate(files):
    img_path = os.path.join(VAL_DIR, fname)
    frame = cv2.imread(img_path)
    if frame is None:
        continue

    # YOLO 推理
    results = model(frame, verbose=False, conf=OBJ_THRESH)[0]

    if results.boxes is None or len(results.boxes) == 0:
        continue

    for i, box in enumerate(results.boxes):
        conf = float(box.conf[0])
        if not (OBJ_THRESH <= conf < CLOUD_HI):
            continue  # 非灰区

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        now_ms = int(time.monotonic() * 1000)

        if is_duplicate(cx, cy, now_ms):
            continue  # 同目标去重

        sample(frame, (x1, y1, x2, y2), conf, idx, fname)
        total_sampled += 1

    if (idx + 1) % 10 == 0:
        print(f"[{idx+1}/{len(files)}] 已采样 {total_sampled} 个灰区")

print(f"\n完成：共采样 {total_sampled} 个灰区 → {SPOOL_DIR}")

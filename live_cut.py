"""
板端实时分割显示: yolo8n_int8_cut.rknn + post_cut, 摄像头实时画面 + 掩码叠加, 全屏自适应屏幕
用法 (板子):
    # 方式1: 用启动器(自动配Wayland), 默认摄像头
    ./live_run.sh
    # 指定摄像头
    ./live_run.sh /dev/video0
    # 测试视频文件
    ./live_run.sh /path/to/x.mp4

    # 方式2: 手动配环境
    export XDG_RUNTIME_DIR=/var/run WAYLAND_DISPLAY=wayland-0 QT_QPA_PLATFORM=wayland
    python3 live_cut.py
"""
import sys, cv2, time, gc
import numpy as np
from rknnlite.api import RKNNLite
from post_cut import post_process

RKNN_MODEL = "./yolo8n_int8_cut.rknn"
CAMERA_DEV = "/dev/video44"
MODEL_SIZE = (640, 640)
# 屏幕物理尺寸 (板子 DSI 屏是 1080x1920 竖屏); 运行时自动读 fb 尺寸
SCREEN_W, SCREEN_H = 1080, 1920


def get_screen_size():
    """优先读 DRM/wayland 屏幕尺寸, 读不到就用默认 1080x1920"""
    for p in ("/sys/class/drm/card0-DSI-1/modes", "/sys/class/drm/card0-HDMI-A-1/modes"):
        try:
            with open(p) as f:
                line = f.readline().strip()
            if "x" in line:
                w, h = map(int, line.split("x"))
                return max(w, h), min(w, h)   # 取横放 (长边做宽)
        except Exception:
            pass
    return SCREEN_W, SCREEN_H


# ---------- 参数 ----------
src = sys.argv[1] if len(sys.argv) > 1 else CAMERA_DEV
use_cam = src.startswith("/dev/video") or src == ""
src = CAMERA_DEV if src == "" else src

# ---------- 模型 ----------
rknn = RKNNLite()
if rknn.load_rknn(RKNN_MODEL) != 0:
    print("Load rknn failed"); sys.exit(1)
if rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) != 0:
    print("Init runtime failed"); sys.exit(1)
print("模型加载成功:", RKNN_MODEL)

# ---------- 输入 ----------
cap = cv2.VideoCapture(src if not use_cam else src, cv2.CAP_V4L2 if use_cam else 0)
if not cap.isOpened():
    print("打开输入失败:", src); sys.exit(1)
print("输入源:", src)

# ---------- 窗口 (全屏自适应) ----------
scr_w, scr_h = get_screen_size()
print("屏幕尺寸(横放): %dx%d" % (scr_w, scr_h))
WIN = "crack_seg_live"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def letter_box(im, new_shape=640):
    shape = im.shape[:2]
    r = min(new_shape / shape[0], new_shape / shape[1])
    nu = (int(round(shape[1] * r)), int(round(shape[0] * r)))   # (w,h)
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
        out = cv2.addWeighted(out, 1.0, red, 0.45, 0)        # 半透明红掩码
    for k in range(len(boxes)):
        x1, y1, x2, y2 = boxes[k].astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, "crack %.2f" % confs[k], (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out


def fit_screen(frame, scr_w, scr_h):
    """保持比例缩放到屏幕, 居中黑边填充"""
    fh, fw = frame.shape[:2]
    s = min(scr_w / fw, scr_h / fh)
    nw, nh = int(fw * s), int(fh * s)
    resized = cv2.resize(frame, (nw, nh))
    canvas = np.zeros((scr_h, scr_w, 3), np.uint8)
    x = (scr_w - nw) // 2
    y = (scr_h - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


# ---------- 主循环 ----------
t_infer_sum = t_post_sum = 0.0
n = 0
print("实时显示已启动, 按 q 或 ESC 退出\n")
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            if not use_cam:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # 视频循环
                continue
            break
        n += 1
        h, w = frame.shape[:2]
        img = letter_box(frame.copy(), MODEL_SIZE[0])
        inp = np.expand_dims(img, axis=0)

        t1 = time.time()
        outputs = rknn.inference([inp])
        t_infer = time.time() - t1

        t2 = time.time()
        res = post_process(outputs, w, h)
        t_post = time.time() - t2

        t_infer_sum += t_infer
        t_post_sum += t_post

        if res is None:
            anno = frame
            info = "no detect"
        else:
            boxes, confs, mask = res
            info = "%d obj, max %.2f" % (len(confs), confs.max())
            anno = draw(frame, boxes, confs, mask)

        # 画面缩放到全屏
        show = fit_screen(anno, scr_w, scr_h)

        # 左上角 OSD
        ms = (t_infer + t_post) * 1000
        fps = 1.0 / (t_infer + t_post) if (t_infer + t_post) > 0 else 0
        cv2.putText(show, "infer %.0fms | post %.0fms | %.0fFPS | %s"
                    % (t_infer * 1000, t_post * 1000, fps, info),
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow(WIN, show)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):   # q 或 ESC 退出
            break

        del outputs, inp, img, show
        if n % 20 == 0:
            gc.collect()
except KeyboardInterrupt:
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()
    rknn.release()
    if n > 0:
        print("\n=== 汇总 (%d 帧) ===" % n)
        print("平均推理:   %6.1f ms" % (t_infer_sum / n * 1000))
        print("平均后处理: %6.1f ms" % (t_post_sum / n * 1000))
        print("平均合计:   %6.1f ms (%.1f FPS)" %
              ((t_infer_sum + t_post_sum) / n * 1000, n / (t_infer_sum + t_post_sum)))

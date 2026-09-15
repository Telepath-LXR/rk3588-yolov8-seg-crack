"""
板端视频测试: yolo8n_int8_cut.rknn + post_cut 单类后处理, 并保存标注结果视频
用法 (板子, 非 conda):
    # 测试视频文件, 结果存 video_result.mp4
    python3 video_cut.py /path/to/xxx.mp4

    # 测试视频文件, 指定输出名
    python3 video_cut.py /path/to/xxx.mp4 out.mp4

    # 用摄像头 (默认 /dev/video44), 不存视频只看实时打印
    python3 video_cut.py cam

    # 摄像头 + 存视频
    python3 video_cut.py cam cam_result.mp4
"""
import sys, cv2, time, gc
import numpy as np
from rknnlite.api import RKNNLite
from post_cut import post_process

RKNN_MODEL = "./yolo8n_int8_cut.rknn"
CAMERA_DEV = "/dev/video44"
MODEL_SIZE = (640, 640)
OBJ_DRAW_MIN = 0.25          # 画框/掩码的最低置信度

# ---------- 参数解析 ----------
src = sys.argv[1] if len(sys.argv) > 1 else "cam"
out_path = sys.argv[2] if len(sys.argv) > 2 else ("cam_result.mp4" if src == "cam" else "video_result.mp4")
use_cam = (src == "cam")

# ---------- 模型 ----------
rknn = RKNNLite()
if rknn.load_rknn(RKNN_MODEL) != 0:
    print("Load rknn failed"); sys.exit(1)
if rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) != 0:
    print("Init runtime failed"); sys.exit(1)
print("模型加载成功:", RKNN_MODEL)

# ---------- 输入源 ----------
if use_cam:
    cap = cv2.VideoCapture(CAMERA_DEV, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
else:
    cap = cv2.VideoCapture(src)
if not cap.isOpened():
    print("打开输入失败:", src); sys.exit(1)

total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not use_cam else -1
fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
print("输入: %s  %dx%d  fps=%.1f  帧数=%s" % (src, fw, fh, fps_in, total if total > 0 else "摄像头(连续)"))
print("输出视频:", out_path, "\n")


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
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return out


# ---------- 输出视频 ----------
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
vw = cv2.VideoWriter(out_path, fourcc, fps_in, (fw, fh))
print("VideoWriter isOpened:", vw.isOpened())

# ---------- 主循环 ----------
t_infer_sum = t_post_sum = 0.0
n = 0
det_frames = 0
max_conf_all = 0.0
while True:
    ret, frame = cap.read()
    if not ret:
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
        info = "无检测"
    else:
        boxes, confs, mask = res
        info = "检测 %d 个, 最高 %.3f, 掩码 %d" % (len(confs), confs.max(), int((mask > 0).sum()))
        det_frames += 1
        max_conf_all = max(max_conf_all, float(confs.max()))
        anno = draw(frame, boxes, confs, mask)

    vw.write(anno)

    if n % 10 == 0 or n <= 3:
        print("[%5d] infer=%5.1fms post=%5.1fms  %s" % (n, t_infer * 1000, t_post * 1000, info))

    del outputs, inp, img
    if n % 20 == 0:
        gc.collect()

cap.release()
vw.release()
rknn.release()

print("\n=== 汇总 (%d 帧) ===" % n)
print("平均推理:   %6.1f ms" % (t_infer_sum / n * 1000))
print("平均后处理: %6.1f ms" % (t_post_sum / n * 1000))
print("平均合计:   %6.1f ms  (%.1f FPS)" % ((t_infer_sum + t_post_sum) / n * 1000,
                                            n / (t_infer_sum + t_post_sum)))
print("检测到目标的帧: %d / %d (%.0f%%)" % (det_frames, n, 100.0 * det_frames / n))
print("全局最高置信度: %.3f" % max_conf_all)
print("结果视频已存:", out_path)

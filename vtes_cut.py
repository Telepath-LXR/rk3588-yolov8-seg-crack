"""
板端测试: yolo8n_int8_cut.rknn (10 分支输出) + post_cut 单类后处理
用法 (板子, 非 conda):
    cd /yolo8 && python3 vtes_cut.py
"""
import cv2, time, gc
import numpy as np
from rknnlite.api import RKNNLite
from post_cut import post_process

RKNN_MODEL = "./yolo8n_int8_cut.rknn"
CAMERA_DEV = "/dev/video44"
MODEL_SIZE = (640, 640)
SAVE_IMG = "./cut_board.png"   # 首个检测帧存图, 方便 adb pull 回看

rknn = RKNNLite()
ret = rknn.load_rknn(RKNN_MODEL)
if ret != 0:
    print("Load rknn failed:", ret); exit(1)
ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
if ret != 0:
    print("Init runtime failed:", ret); exit(1)
print("模型加载成功:", RKNN_MODEL)

cap = cv2.VideoCapture(CAMERA_DEV, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)


def letter_box(im, new_shape=640):
    shape = im.shape[:2]
    r = min(new_shape / shape[0], new_shape / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # (w, h)
    dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
    dw /= 2; dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(im, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(0, 0, 0))


def draw(frame, boxes, confs, mask):
    out = frame.copy()
    if mask is not None and mask.size:
        out[mask > 0] = (0, 0, 255)               # 红色掩码叠加
    for k in range(len(boxes)):
        x1, y1, x2, y2 = boxes[k].astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, "%.2f" % confs[k], (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return out


print("测试 50 帧 (含后处理)\n")
t_infer_sum = t_post_sum = 0.0
N = 50
saved = False
for i in range(N):
    cap.grab()
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    img = letter_box(frame.copy(), MODEL_SIZE[0])
    inp = np.expand_dims(img, axis=0)

    t1 = time.time()
    outputs = rknn.inference([inp])
    t_infer = time.time() - t1

    if i == 0:                                     # 首帧诊断: 确认板端输出 dtype
        print("首帧输出 dtype/shape/min/max:")
        for j, o in enumerate(outputs):
            print("  out[%d]: %s %s min=%.3f max=%.3f" % (j, o.dtype, o.shape, o.min(), o.max()))

    t2 = time.time()
    res = post_process(outputs, w, h)
    t_post = time.time() - t2

    t_infer_sum += t_infer
    t_post_sum += t_post

    if res is None:
        info = "无检测"
    else:
        boxes, confs, mask = res
        info = "检测 %d 个, 最高分 %.3f, 掩码像素 %d" % (len(confs), confs.max(), int((mask > 0).sum()))
        if not saved:
            cv2.imwrite(SAVE_IMG, draw(frame, boxes, confs, mask))
            saved = True
            print("  已存可视化:", SAVE_IMG)
    print("[%3d] infer=%5.1fms  post=%5.1fms  %s" % (i + 1, t_infer * 1000, t_post * 1000, info))

    del outputs, inp, img
    if i % 10 == 0:
        gc.collect()

print("\n=== 平均 (%d 帧) ===" % N)
print("推理:   %6.1f ms" % (t_infer_sum / N * 1000))
print("后处理: %6.1f ms" % (t_post_sum / N * 1000))
print("合计:   %6.1f ms" % ((t_infer_sum + t_post_sum) / N * 1000))
print("端到端 FPS: %.1f" % (N / (t_infer_sum + t_post_sum)))

cap.release()
rknn.release()

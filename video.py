import cv2
import time
import gc
from rknnlite.api import RKNNLite
import numpy as np

# ===== 配置 =====
RKNN_MODEL = "./yolo8n_fp16.rknn"
CAMERA_DEV = "/dev/video44"
RESULT_PATH = "./output_video"

OBJ_THRESH = 0.15
NMS_THRESH = 0.60
MASK_THRESH = 0.2
MIN_AREA = 80

MODEL_SIZE = (640, 640)
CAM_W = 320
CAM_H = 240
DETECT_INTERVAL = 3

# ===== 显示配置 =====
DISPLAY_W = 1280       # 屏幕宽（根据你的实际屏幕调整）
DISPLAY_H = 720        # 屏幕高
FULLSCREEN = True      # 是否全屏


def letter_box(im, new_shape, pad_color=(0, 0, 0)):
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)


def post_process(outputs, img_w, img_h):
    pred = np.squeeze(outputs[0])
    proto = np.squeeze(outputs[1])
    if pred.shape[0] != 37 and pred.shape[1] == 37:
        pred = pred.T

    box_xywh = pred[:4, :]
    scores = np.squeeze(pred[4, :])
    mask_coeff = pred[5:37, :]

    indices = np.where(scores > OBJ_THRESH)[0]
    if len(indices) == 0:
        return None

    boxes_640 = []
    confs = []
    for idx in indices:
        x = float(np.squeeze(box_xywh[0, idx]))
        y = float(np.squeeze(box_xywh[1, idx]))
        bw = float(np.squeeze(box_xywh[2, idx]))
        bh = float(np.squeeze(box_xywh[3, idx]))
        boxes_640.append([x - bw / 2, y - bh / 2, bw, bh])
        confs.append(float(scores[idx]))

    nms_idx = cv2.dnn.NMSBoxes(boxes_640, confs, OBJ_THRESH, NMS_THRESH)
    if len(nms_idx) == 0:
        return None
    nms_idx = nms_idx.flatten()
    if len(nms_idx) > 3:
        nms_idx = nms_idx[:3]

    proto_flat = proto.reshape(32, -1)
    combined = np.zeros((img_h, img_w), dtype=np.uint8)

    scale_x = img_w / MODEL_SIZE[0]
    scale_y = img_h / MODEL_SIZE[1]

    for i in nms_idx:
        orig_idx = indices[i]
        coeff = np.squeeze(mask_coeff[:, orig_idx])
        m = coeff @ proto_flat
        m = 1.0 / (1.0 + np.exp(-m))
        m = m.reshape(160, 160)
        m_up = cv2.resize(m, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        m_bin = (m_up > MASK_THRESH).astype(np.uint8) * 255

        x, y, bw, bh = boxes_640[i]
        x1 = max(0, int(x * scale_x))
        y1 = max(0, int(y * scale_y))
        x2 = min(img_w, int((x + bw) * scale_x))
        y2 = min(img_h, int((y + bh) * scale_y))

        crop = np.zeros_like(m_bin)
        crop[y1:y2, x1:x2] = m_bin[y1:y2, x1:x2]
        combined = cv2.bitwise_or(combined, crop)

    return combined


def draw_fast(image, combined_mask):
    if combined_mask is None:
        return image, 0

    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed)
    valid = [i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= MIN_AREA]

    if len(valid) == 0:
        return image, 0

    overlay = image.copy()
    overlay[closed > 0] = (0, 0, 255)
    result = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)

    for k, label_id in enumerate(valid):
        cx, cy = centroids[label_id]
        cv2.putText(result, f"#{k+1}", (int(cx) - 15, int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return result, len(valid)


def make_display(original, result, crack_count, fps, is_fresh):
    """
    把 320x240 的检测结果放大，并叠加大字体 HUD，组成适合全屏的显示画面。
    """
    # 放大到屏幕尺寸
    big_original = cv2.resize(original, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_LINEAR)
    big_result = cv2.resize(result, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_LINEAR)

    # 用 result 作为主画面
    display = big_result.copy()

    # ===== 顶部信息条 =====
    bar_h = 80
    cv2.rectangle(display, (0, 0), (DISPLAY_W, bar_h), (0, 0, 0), -1)

    # 标题
    title = "Crack Detection (FP16)"
    cv2.putText(display, title, (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    # 状态：有裂缝/无裂缝
    if crack_count > 0:
        status = f"Detected: {crack_count} crack(s)"
        status_color = (0, 0, 255)   # 红色
    else:
        status = "No crack"
        status_color = (0, 255, 0)    # 绿色
    cv2.putText(display, status, (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)

    # FPS（右上角）
    fps_text = f"FPS: {fps:.1f}"
    (tw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.putText(display, fps_text, (DISPLAY_W - tw - 20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # 检测新鲜度（右上角，第二行）
    fresh_text = "LIVE" if is_fresh else "last frame"
    fresh_color = (0, 255, 0) if is_fresh else (128, 128, 128)
    (tw2, _), _ = cv2.getTextSize(fresh_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(display, fresh_text, (DISPLAY_W - tw2 - 20, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, fresh_color, 2)

    # ===== 底部信息条 =====
    bar_h2 = 50
    cv2.rectangle(display, (0, DISPLAY_H - bar_h2), (DISPLAY_W, DISPLAY_H), (0, 0, 0), -1)
    hint = "Press 'q' to quit | 's' to save snapshot"
    cv2.putText(display, hint, (20, DISPLAY_H - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    return display


if __name__ == '__main__':
    rknn_lite = RKNNLite()
    ret = rknn_lite.load_rknn(RKNN_MODEL)
    if ret != 0:
        print('Load RKNN model failed'); exit(ret)
    ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    if ret != 0:
        print('Init runtime failed'); exit(ret)
    print("✅ 模型加载成功")

    cap = cv2.VideoCapture(CAMERA_DEV, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_DEV)
    if not cap.isOpened():
        print(f"无法打开摄像头 {CAMERA_DEV}"); exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    print(f"摄像头: {int(cap.get(3))}x{int(cap.get(4))}")

    # ===== 创建全屏窗口 =====
    win_name = "Crack Detection"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    if FULLSCREEN:
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(win_name, DISPLAY_W, DISPLAY_H)

    frame_count = 0
    infer_count = 0
    last_result = None
    last_crack_count = 0
    last_original = None
    t_infer_sum = 0
    t_start_all = time.time()

    print(f"\n实时检测已启动（全屏 {DISPLAY_W}x{DISPLAY_H}）")
    print("按 'q' 退出\n")

    try:
        while True:
            cap.grab()
            ret, frame = cap.read()
            if not ret:
                continue
            frame_count += 1

            is_fresh = False

            if frame_count % DETECT_INTERVAL == 0:
                h, w = frame.shape[:2]
                img = letter_box(frame.copy(), (MODEL_SIZE[1], MODEL_SIZE[0]))
                inp = np.expand_dims(img, axis=0)

                t1 = time.time()
                outputs = rknn_lite.inference([inp])
                t_infer_sum += time.time() - t1
                infer_count += 1

                combined = post_process(outputs, w, h)
                result, cracks = draw_fast(frame.copy(), combined)
                last_result = result
                last_crack_count = cracks
                last_original = frame.copy()
                is_fresh = True

                del outputs, inp, img
                if frame_count % 25 == 0:
                    gc.collect()
            else:
                if last_result is None:
                    last_result = frame.copy()
                    last_original = frame.copy()
                result = last_result
                cracks = last_crack_count

            # FPS
            t_elapsed = time.time() - t_start_all
            real_fps = frame_count / t_elapsed if t_elapsed > 0 else 0

            # ===== 构造显示画面 =====
            display = make_display(last_original, result, last_crack_count, real_fps, is_fresh)

            cv2.imshow(win_name, display)

            if frame_count % 50 == 0:
                infer_avg = t_infer_sum / infer_count * 1000 if infer_count > 0 else 0
                print(f"[{frame_count:5d}] FPS={real_fps:.1f} | 推理均={infer_avg:.0f}ms | 裂缝={cracks}")

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:   # 27 = ESC
                break
            elif key == ord('s'):
                snap = f"{RESULT_PATH}/snap_{frame_count:05d}.jpg"
                cv2.imwrite(snap, display)
                print(f"截图: {snap}")

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        rknn_lite.release()
        t_total = time.time() - t_start_all
        print(f"\n总帧数: {frame_count}, 总时间: {t_total:.1f}s")
        print(f"平均 FPS: {frame_count/t_total:.2f}")


# ===== 先确认屏幕分辨率 =====
# 在开发板上执行以下命令查看屏幕分辨率：
#   cat /sys/class/graphics/fb0/virtual_size
# 输出格式如 "1280,720"，把这两个数字填入 DISPLAY_W 和 DISPLAY_H

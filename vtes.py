import cv2, time, gc
from rknnlite.api import RKNNLite
import numpy as np

RKNN_MODEL = "./yolo8n_int8_hybrid.rknn"
CAMERA_DEV = "/dev/video44"

OBJ_THRESH = 0.25
NMS_THRESH = 0.50
MASK_THRESH = 0.3
MIN_AREA = 80
MODEL_SIZE = (640, 640)

rknn = RKNNLite()
rknn.load_rknn(RKNN_MODEL)
rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
print("模型加载成功")

cap = cv2.VideoCapture(CAMERA_DEV, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

def letter_box(im, new_shape):
    shape = im.shape[:2]
    r = min(new_shape[0]/shape[0], new_shape[1]/shape[1])
    new_unpad = int(round(shape[1]*r)), int(round(shape[0]*r))
    dw, dh = new_shape[1]-new_unpad[0], new_shape[0]-new_unpad[1]
    dw/=2; dh/=2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad)
    top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
    left, right = int(round(dw-0.1)), int(round(dw+0.1))
    return cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))

def post_process(outputs, img_w, img_h):
    pred = outputs[0][0]
    proto = outputs[1][0]
    box_xywh = pred[:4, :]
    scores = pred[4, :]
    mask_coeff = pred[5:37, :]
    indices = np.where(scores > OBJ_THRESH)[0]
    print(f"  [post] 高于阈值的框: {len(indices)}, 最高分: {scores.max():.3f}") 
    if len(indices) == 0:
        return None
    boxes_640 = []
    confs = []
    for idx in indices:
        x, y, bw, bh = box_xywh[:, idx]
        boxes_640.append([float(x-bw/2), float(y-bh/2), float(bw), float(bh)])
        confs.append(float(scores[idx]))
    nms_idx = cv2.dnn.NMSBoxes(boxes_640, confs, OBJ_THRESH, NMS_THRESH)
    if len(nms_idx) == 0:
        return None
    nms_idx = nms_idx.flatten()
    if len(nms_idx) > 3:
        nms_idx = nms_idx[:3]
    proto_flat = proto.reshape(32, -1)
    combined = np.zeros((img_h, img_w), dtype=np.uint8)
    for i in nms_idx:
        orig_idx = indices[i]
        coeff = mask_coeff[:, orig_idx]
        m = coeff @ proto_flat
        m = 1.0/(1.0+np.exp(-m))
        m = m.reshape(160,160)
        m_up = cv2.resize(m, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        m_bin = (m_up > MASK_THRESH).astype(np.uint8)*255
        x, y, bw, bh = box_xywh[:, orig_idx]
        x1 = max(0, int((x-bw/2)*img_w/MODEL_SIZE[0]))
        y1 = max(0, int((y-bh/2)*img_h/MODEL_SIZE[1]))
        x2 = min(img_w, int((x+bw/2)*img_w/MODEL_SIZE[0]))
        y2 = min(img_h, int((y+bh/2)*img_h/MODEL_SIZE[1]))
        crop = np.zeros_like(m_bin)
        crop[y1:y2, x1:x2] = m_bin[y1:y2, x1:x2]
        combined = cv2.bitwise_or(combined, crop)
    return combined

print("测试 50 帧（含后处理）\n")
t_infer_sum = 0
t_post_sum = 0
N = 50

for i in range(N):
    cap.grab()
    ret, frame = cap.read()
    if not ret: break
    h, w = frame.shape[:2]
    img = letter_box(frame.copy(), (640, 640))
    inp = np.expand_dims(img, axis=0)

    t1 = time.time()
    outputs = rknn.inference([inp])
    t_infer = time.time() - t1

    t2 = time.time()
    combined = post_process(outputs, w, h)
    t_post = time.time() - t2

    t_infer_sum += t_infer
    t_post_sum += t_post
    del outputs, inp, img
    if i % 10 == 0:
        gc.collect()
    if (i+1) % 10 == 0:
        print(f"[{i+1:3d}] infer={t_infer*1000:5.1f}ms  post={t_post*1000:5.1f}ms")

print(f"\n=== 平均（{N} 帧）===")
print(f"推理:   {t_infer_sum/N*1000:6.1f} ms")
print(f"后处理: {t_post_sum/N*1000:6.1f} ms")
print(f"合计:   {(t_infer_sum+t_post_sum)/N*1000:6.1f} ms")
print(f"端到端 FPS: {N/(t_infer_sum+t_post_sum):.1f}")

cap.release()
rknn.release()

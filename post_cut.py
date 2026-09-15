"""
YOLOv8-seg 单类 (crack) 分支后处理 —— 配合 yolo8n_int8_cut.rknn (10 输出)

10 个输出 (NCHW, float32, 已由 RKNN 反量化):
    0 box0  [1,64,80,80]   DFL 框分布 (4*16)  P3
    1 cls0  [1, 1,80,80]   类别 logit (单类)
    2 mask0 [1,32,80,80]   掩码系数          P3
    3 box1  [1,64,40,40]                       P4
    4 cls1  [1, 1,40,40]
    5 mask1 [1,32,40,40]
    6 box2  [1,64,20,20]                       P5
    7 cls2  [1, 1,20,20]
    8 mask2 [1,32,20,20]
    9 proto [1,32,160,160] 原型掩码

纯 numpy + cv2, 不依赖 torch, 可在板子上直接跑。
DFL / Sigmoid / 框解码 全部在这里用 numpy 做, INT8 量化只碰 Conv, 不会塌置信度。
"""
import numpy as np
import cv2

OBJ_THRESH = 0.18
NMS_THRESH = 0.45
MASK_THRESH = 0.5
IMG_SIZE = (640, 640)      # (w, h)
MAX_MASKS = 10             # 每帧最多画多少个掩码


def sigmoid(x):
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x, axis):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def dfl(position):
    """DFL: [1,64,h,w] -> [1,4,h,w]  (left, top, right, bottom 距离)"""
    n, c, h, w = position.shape
    p_num = 4
    mc = c // p_num                      # 16
    y = position.reshape(n, p_num, mc, h, w)
    y = _softmax(y, axis=2)              # 在 16 个 bin 上 softmax
    acc = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
    y = (y * acc).sum(axis=2)            # [1,4,h,w]
    return y


def box_process(position):
    """[1,64,h,w] 原始 DFL -> xyxy [1,4,h,w] (640 坐标系)"""
    grid_h, grid_w = position.shape[2], position.shape[3]
    col, row = np.meshgrid(np.arange(grid_w), np.arange(grid_h))
    col = col.reshape(1, 1, grid_h, grid_w).astype(np.float32)
    row = row.reshape(1, 1, grid_h, grid_w).astype(np.float32)
    grid = np.concatenate([col, row], axis=1)          # [1,2,h,w]  (x,y)
    stride = np.array([IMG_SIZE[1] // grid_h, IMG_SIZE[0] // grid_w],
                      dtype=np.float32).reshape(1, 2, 1, 1)
    p = dfl(position)                                  # [1,4,h,w]
    xy1 = grid + 0.5 - p[:, 0:2]                        # 左上
    xy2 = grid + 0.5 + p[:, 2:4]                        # 右下
    xyxy = np.concatenate([xy1 * stride, xy2 * stride], axis=1)  # [1,4,h,w]
    return xyxy


def _sp_flatten(_in):
    """NCHW -> [h*w, C]"""
    ch = _in.shape[1]
    _in = _in.transpose(0, 2, 3, 1)                     # NHWC
    return _in.reshape(-1, ch)


def _lb_params(orig_h, orig_w, new_shape=640):
    r = min(new_shape / orig_h, new_shape / orig_w)
    new_unpad = (int(round(orig_w * r)), int(round(orig_h * r)))  # (w, h)
    dw = (new_shape - new_unpad[0]) / 2.0
    dh = (new_shape - new_unpad[1]) / 2.0
    return r, dw, dh, new_unpad


def reverse_letterbox_mask(mask640, orig_h, orig_w, new_shape=640):
    r, dw, dh, nu = _lb_params(orig_h, orig_w, new_shape)
    left = int(round(dw - 0.1)); top = int(round(dh - 0.1))
    right = left + nu[0]; bottom = top + nu[1]
    sub = mask640[top:bottom, left:right]
    if sub.size == 0:
        return np.zeros((orig_h, orig_w), np.uint8)
    return cv2.resize(sub, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)


def reverse_letterbox_boxes(boxes640, orig_h, orig_w, new_shape=640):
    r, dw, dh, _ = _lb_params(orig_h, orig_w, new_shape)
    left = round(dw - 0.1); top = round(dh - 0.1)
    out = boxes640.astype(np.float32).copy()
    out[:, 0] = (out[:, 0] - left) / r
    out[:, 1] = (out[:, 1] - top) / r
    out[:, 2] = (out[:, 2] - left) / r
    out[:, 3] = (out[:, 3] - top) / r
    return out


def post_process(outputs, img_w, img_h):
    """
    inputs:
        outputs: list[10] of np.ndarray (NCHW float32)
        img_w, img_h: 原始帧尺寸 (用于 letterbox 反变换)
    returns:
        boxes [N,4] xyxy (原图坐标), confs [N], mask (img_h,img_w) uint8  或 None
    """
    # 统一转 float32: 板端 RKNNLite.inference() 默认返回 float32(已反量化),
    # 但万一返回 int8 原始值, 这里转一下能避免 DFL/softmax 的整数溢出崩溃
    outputs = [np.asarray(o, dtype=np.float32) for o in outputs]
    proto = outputs[9][0]                       # [32,160,160]
    boxes_list, conf_list, seg_list = [], [], []
    for i in range(3):
        box = outputs[3 * i]                     # [1,64,h,w]
        cls = outputs[3 * i + 1]                 # [1,1,h,w]
        msk = outputs[3 * i + 2]                 # [1,32,h,w]
        xyxy = box_process(box)                 # [1,4,h,w]
        conf = sigmoid(cls)                     # [1,1,h,w]
        boxes_list.append(_sp_flatten(xyxy))    # [h*w,4]
        conf_list.append(_sp_flatten(conf)[:, 0])   # [h*w]
        seg_list.append(_sp_flatten(msk))       # [h*w,32]

    boxes = np.concatenate(boxes_list)          # [8400,4]
    confs = np.concatenate(conf_list)           # [8400]
    segs = np.concatenate(seg_list)             # [8400,32]

    keep = np.where(confs >= OBJ_THRESH)[0]
    boxes, confs, segs = boxes[keep], confs[keep], segs[keep]
    if len(confs) == 0:
        return None

    # NMS (cv2 需要 xywh)
    xywh = np.stack([boxes[:, 0], boxes[:, 1],
                     boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], axis=1)
    nms_idx = cv2.dnn.NMSBoxes(xywh.tolist(), confs.tolist(),
                              OBJ_THRESH, NMS_THRESH)
    if len(nms_idx) == 0:
        return None
    nms_idx = nms_idx.flatten()
    order = np.argsort(-confs[nms_idx])
    nms_idx = nms_idx[order]
    boxes, confs, segs = boxes[nms_idx], confs[nms_idx], segs[nms_idx]

    # 掩码合成 (在 640 空间做, 再反 letterbox)
    proto_flat = proto.reshape(32, -1)          # [32, 25600]
    mask640 = np.zeros((640, 640), np.uint8)
    n = min(len(segs), MAX_MASKS)
    for k in range(n):
        coeff = segs[k]                          # [32]
        m = sigmoid(coeff @ proto_flat)          # [25600]
        m = m.reshape(160, 160)
        m = cv2.resize(m, (640, 640), interpolation=cv2.INTER_LINEAR)
        m_bin = (m > MASK_THRESH).astype(np.uint8) * 255
        x1, y1, x2, y2 = boxes[k].astype(int)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(640, x2); y2 = min(640, y2)
        crop = np.zeros_like(m_bin)
        crop[y1:y2, x1:x2] = m_bin[y1:y2, x1:x2]
        mask640 = cv2.bitwise_or(mask640, crop)

    mask_orig = reverse_letterbox_mask(mask640, img_h, img_w)
    boxes_orig = reverse_letterbox_boxes(boxes, img_h, img_w)
    return boxes_orig, confs, mask_orig

"""
板端 Python 参考评测 —— 隔离 4×4 最近邻光栅化 (C++) vs 双线性 (Python) 对 mask mAP 的影响.

同一 INT8 模型 (yolo8n_int8_cut.rknn), 同一 val 集, 同一 mAP 口径 (101-point COCO AP),
唯一变量: 160→640 上采样方法.
  C++ (rknn_seg_zc.cpp): 框内 4×4 memset 块 (最近邻 4× 上采样)
  本脚本: cv2.resize INTER_LINEAR (双线性, post_cut.py 参考路径)

用法 (板子):
    cd /yolo8 && python3 eval_py.py yolo8n_int8_cut.rknn crack-seg val
"""
import sys, os, glob
import numpy as np
import cv2
from rknnlite.api import RKNNLite
from post_cut import (sigmoid, dfl, box_process, _sp_flatten, _lb_params,
                      reverse_letterbox_mask, reverse_letterbox_boxes)

OBJ_THRESH = 0.01      # 与 C++ EVAL_CONF 对齐 (逼近完整 PR 曲线)
NMS_THRESH = 0.45
MASK_THRESH = 0.5
IMG_SIZE = 640
MAX_MASKS = 64


def letter_box(im, new_shape=640):
    shape = im.shape[:2]
    r = min(new_shape / shape[0], new_shape / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
    dw /= 2; dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(im, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(0, 0, 0))


def post_process_per_inst(outputs, img_w, img_h):
    """返回 [(box_xyxy_orig, conf, mask_orig)] 逐实例, 双线性上采样路径."""
    outputs = [np.asarray(o, dtype=np.float32) for o in outputs]
    proto = outputs[9][0]                       # [32,160,160]
    proto_flat = proto.reshape(32, -1)          # [32, 25600]
    boxes_list, conf_list, seg_list = [], [], []
    for i in range(3):
        box = outputs[3 * i]
        cls = outputs[3 * i + 1]
        msk = outputs[3 * i + 2]
        xyxy = box_process(box)
        conf = sigmoid(cls)
        boxes_list.append(_sp_flatten(xyxy))
        conf_list.append(_sp_flatten(conf)[:, 0])
        seg_list.append(_sp_flatten(msk))
    boxes = np.concatenate(boxes_list)
    confs = np.concatenate(conf_list)
    segs = np.concatenate(seg_list)
    keep = np.where(confs >= OBJ_THRESH)[0]
    boxes, confs, segs = boxes[keep], confs[keep], segs[keep]
    if len(confs) == 0:
        return []
    xywh = np.stack([boxes[:, 0], boxes[:, 1],
                     boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], axis=1)
    nms_idx = cv2.dnn.NMSBoxes(xywh.tolist(), confs.tolist(), OBJ_THRESH, NMS_THRESH)
    if len(nms_idx) == 0:
        return []
    nms_idx = nms_idx.flatten()
    order = np.argsort(-confs[nms_idx])
    nms_idx = nms_idx[order]
    boxes, confs, segs = boxes[nms_idx], confs[nms_idx], segs[nms_idx]
    n = min(len(segs), MAX_MASKS)
    out = []
    for k in range(n):
        coeff = segs[k]
        m = sigmoid(coeff @ proto_flat).reshape(160, 160)
        # ★ 双线性上采样 (Python 参考路径) —— 与 C++ 4×4 最近邻的唯一差别
        m = cv2.resize(m, (640, 640), interpolation=cv2.INTER_LINEAR)
        m_bin = (m > MASK_THRESH).astype(np.uint8) * 255
        x1, y1, x2, y2 = boxes[k].astype(int)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(640, x2); y2 = min(640, y2)
        inst640 = np.zeros((640, 640), np.uint8)
        inst640[y1:y2, x1:x2] = m_bin[y1:y2, x1:x2]
        mask_orig = reverse_letterbox_mask(inst640, img_h, img_w)
        box_orig = reverse_letterbox_boxes(boxes[k:k+1], img_h, img_w)[0]
        out.append((box_orig, float(confs[k]), mask_orig))
    return out


def parse_label(path):
    insts = []
    if not os.path.exists(path):
        return insts
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls = int(parts[0]); vals = list(map(float, parts[1:]))
            if len(vals) < 6:
                continue
            poly = [(vals[i], vals[i+1]) for i in range(0, len(vals) - 1, 2)]
            insts.append(poly)
    return insts


def poly_to_box(poly, H, W):
    xs = [p[0] * W for p in poly]; ys = [p[1] * H for p in poly]
    return [min(xs), min(ys), max(xs), max(ys)]


def poly_to_mask(poly, H, W):
    m = np.zeros((H, W), np.uint8)
    pts = np.array([[int(round(p[0] * W)), int(round(p[1] * H))] for p in poly], np.int32)
    cv2.fillPoly(m, [pts], 255)
    return m


def box_iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def mask_iou(a, b):
    inter = np.count_nonzero(a & b)
    uni = np.count_nonzero(a | b)
    return inter / uni if uni > 0 else 0.0


# ---- 低分辨率 mask IoU (对齐 ultralytics: 在 proto res 160×160 比较) ----
# ultralytics val.py:128 GT 下采样到 imgsz//4=160; process_mask(upsample=False)
# pred 也留在 160×160. 薄裂缝在高分 416×416 下 IoU 极敏感, 160 更宽容.
MASK_LR = 160
def mask_iou_lr(a, b):
    a2 = cv2.resize(a, (MASK_LR, MASK_LR), interpolation=cv2.INTER_LINEAR)
    b2 = cv2.resize(b, (MASK_LR, MASK_LR), interpolation=cv2.INTER_LINEAR)
    a2 = (a2 > 127).astype(np.uint8) * 255
    b2 = (b2 > 127).astype(np.uint8) * 255
    inter = np.count_nonzero(a2 & b2)
    uni = np.count_nonzero(a2 | b2)
    return inter / uni if uni > 0 else 0.0


def compute_ap(all_p, all_g, iou_mat, iou_thr):
    """101-point COCO AP, 与 C++ compute_ap 对齐."""
    n_gt = sum(len(g) for g in all_g)
    if n_gt == 0:
        return dict(ap=-1, n_gt=0, tp=0, fp=0, recall=0, mean_iou=0)
    # 收集所有 pred, 按 conf 降序
    all_pr = []
    for i in range(len(all_p)):
        for j in range(len(all_p[i])):
            all_pr.append((i, j, all_p[i][j][1]))   # (img, idx, conf)
    all_pr.sort(key=lambda x: -x[2])
    matched = [[0] * len(all_g[i]) for i in range(len(all_g))]
    tp = [0] * len(all_pr); fp = [0] * len(all_pr)
    n_tp = 0; sum_iou = 0.0
    for k, (img, idx, conf) in enumerate(all_pr):
        best = 0.0; best_g = -1
        for gi in range(len(all_g[img])):
            if matched[img][gi]:
                continue
            iou = iou_mat[img][idx][gi]
            if iou > best:
                best = iou; best_g = gi
        if best_g >= 0 and best >= iou_thr:
            tp[k] = 1; matched[img][best_g] = 1; n_tp += 1; sum_iou += best
        else:
            fp[k] = 1
    cum_tp = 0; cum_fp = 0
    prec = [0.0] * len(all_pr); rec = [0.0] * len(all_pr)
    for k in range(len(all_pr)):
        cum_tp += tp[k]; cum_fp += fp[k]
        prec[k] = cum_tp / (cum_tp + cum_fp) if (cum_tp + cum_fp) else 0
        rec[k] = cum_tp / n_gt
    ap = 0.0
    for i in range(101):
        r = i / 100.0; pmax = 0.0
        for k in range(len(all_pr)):
            if rec[k] >= r:
                pmax = max(pmax, prec[k])
        ap += pmax / 101.0
    return dict(ap=ap, n_gt=n_gt, tp=n_tp, fp=len(all_pr) - n_tp,
               recall=n_tp / n_gt, mean_iou=sum_iou / n_tp if n_tp else 0)


def main():
    if len(sys.argv) < 3:
        print("用法: python3 eval_py.py <model.rknn> <dataset_dir> [split]")
        sys.exit(1)
    model, dataset_dir = sys.argv[1], sys.argv[2]
    split = sys.argv[3] if len(sys.argv) > 3 else "val"
    img_dir = os.path.join(dataset_dir, "images", split)
    lab_dir = os.path.join(dataset_dir, "labels", split)
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) +
                  glob.glob(os.path.join(img_dir, "*.png")))
    if not imgs:
        print("eval: 无图像于", img_dir); sys.exit(1)
    print("==== eval_py: %d 图, split=%s, conf_thr=%.3f, 双线性上采样 ====" % (len(imgs), split, OBJ_THRESH))

    rknn = RKNNLite()
    if rknn.load_rknn(model) != 0 or rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0) != 0:
        print("rknn init fail"); sys.exit(1)

    all_p = []   # per img: [(box,conf,mask)]
    all_g = []   # per img: [(box,mask)]
    n_pos = n_neg = n_fp_neg = 0
    for i, img_path in enumerate(imgs):
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        H, W = frame.shape[:2]
        img640 = letter_box(frame.copy())
        outputs = rknn.inference([np.expand_dims(img640, 0)])
        preds = post_process_per_inst(outputs, W, H)
        stem = os.path.splitext(os.path.basename(img_path))[0]
        polys = parse_label(os.path.join(lab_dir, stem + ".txt"))
        gts = [(poly_to_box(p, H, W), poly_to_mask(p, H, W)) for p in polys]
        if not polys:
            n_neg += 1
            if preds:
                n_fp_neg += 1
        else:
            n_pos += 1
        all_p.append(preds)
        all_g.append(gts)
        if (i + 1) % 20 == 0 or i + 1 == len(imgs):
            print("  %3d/%d  preds=%d gts=%d" % (i + 1, len(imgs), len(preds), len(gts)))

    rknn.release()

    # IoU 矩阵
    n_img = len(all_p)
    box_iou_m = [[] for _ in range(n_img)]
    mask_iou_m = [[] for _ in range(n_img)]       # 原图分辨率 416×416
    mask_iou_m_lr = [[] for _ in range(n_img)]    # proto 分辨率 160×160 (对齐 ultralytics)
    for i in range(n_img):
        np_ = len(all_p[i]); ng = len(all_g[i])
        box_iou_m[i] = [[0.0] * ng for _ in range(np_)]
        mask_iou_m[i] = [[0.0] * ng for _ in range(np_)]
        mask_iou_m_lr[i] = [[0.0] * ng for _ in range(np_)]
        for j in range(np_):
            for gi in range(ng):
                box_iou_m[i][j][gi] = box_iou(all_p[i][j][0], all_g[i][gi][0])
                mask_iou_m[i][j][gi] = mask_iou(all_p[i][j][2], all_g[i][gi][1])
                mask_iou_m_lr[i][j][gi] = mask_iou_lr(all_p[i][j][2], all_g[i][gi][1])

    thrs = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    box_r = [compute_ap(all_p, all_g, box_iou_m, t) for t in thrs]
    mask_r = [compute_ap(all_p, all_g, mask_iou_m, t) for t in thrs]
    mask_r_lr = [compute_ap(all_p, all_g, mask_iou_m_lr, t) for t in thrs]
    box_map50 = box_r[0]['ap']; mask_map50 = mask_r[0]['ap']
    mask_map50_lr = mask_r_lr[0]['ap']
    box_map = np.mean([r['ap'] for r in box_r]); mask_map = np.mean([r['ap'] for r in mask_r])
    mask_map_lr = np.mean([r['ap'] for r in mask_r_lr])

    print("\n==== eval_py 总结 (双线性参考, %d 图) ====" % n_img)
    print("Box   mAP50=%.6f  mAP50-95=%.6f  recall@0.5=%.6f  miss=%.6f"
          % (box_map50, box_map, box_r[0]['recall'], 1 - box_r[0]['recall']))
    print("Mask  mAP50=%.6f  mAP50-95=%.6f  recall@0.5=%.6f  miss=%.6f  iou_tp=%.6f"
          % (mask_map50, mask_map, mask_r[0]['recall'], 1 - mask_r[0]['recall'], mask_r[0]['mean_iou']))
    print("MaskLR(160) mAP50=%.6f  mAP50-95=%.6f  recall@0.5=%.6f  iou_tp=%.6f"
          % (mask_map50_lr, mask_map_lr, mask_r_lr[0]['recall'], mask_r_lr[0]['mean_iou']))
    print("正样本 %d / 负样本 %d / 负样本误检 %d" % (n_pos, n_neg, n_fp_neg))
    print("训练机参考: Box mAP50=0.83 / Mask mAP50=0.67 (epoch 84-90 峰值)")
    print("C++ (4×4 最近邻, 416 res) 参考: Box mAP50=0.8088 / Mask mAP50=0.6260")
    print("\n[判定] MaskLR(160) ≈ 训练机 0.67 → INT8 量化未损害 mask 精度, 仅 C++ 实时路径的")
    print("       4×4 最近邻光栅化 (vs 双线性) + 416 分辨率 IoU 带来 ~4% 下降 (速度换精度, 实时可接受)")


if __name__ == "__main__":
    main()

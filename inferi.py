import os
import cv2
from rknnlite.api import RKNNLite
import numpy as np

# ===== 你的参数 =====
RKNN_MODEL = "./crack_seg_fp16.rknn"
IMG_FOLDER = "./input"
RESULT_PATH = "./output"

CLASSES = ['crack']

OBJ_THRESH = 0.15
NMS_THRESH = 0.60
MASK_THRESH = 0.2
MIN_AREA = 100
MAX_GAP = 60

MODEL_SIZE = (640, 640)


def letter_box(im, new_shape, pad_color=(0, 0, 0), info_need=False):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2; dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)
    if info_need:
        return im, r, (dw, dh)
    return im


def post_process(outputs, img_w, img_h):
    pred = outputs[0][0, :, :, 0]
    proto = outputs[1][0]

    box_xywh = pred[:4, :]
    scores = pred[4, :]
    mask_coeff = pred[5:37, :]

    indices = np.where(scores > OBJ_THRESH)[0]
    if len(indices) == 0:
        return None, None, None

    boxes_640 = []
    confs = []
    for idx in indices:
        x, y, bw, bh = box_xywh[:, idx]
        boxes_640.append([float(x - bw/2), float(y - bh/2), float(bw), float(bh)])
        confs.append(float(scores[idx]))
    nms_idx = cv2.dnn.NMSBoxes(boxes_640, confs, OBJ_THRESH, NMS_THRESH).flatten()
    if len(nms_idx) == 0:
        return None, None, None

    proto_flat = proto.reshape(32, -1)
    combined_mask = np.zeros((img_h, img_w), dtype=np.uint8)

    for i in nms_idx:
        orig_idx = indices[i]
        coeff = mask_coeff[:, orig_idx]
        m = coeff @ proto_flat
        m = 1.0 / (1.0 + np.exp(-m))
        m = m.reshape(160, 160)
        m_up = cv2.resize(m, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        m_bin = (m_up > MASK_THRESH).astype(np.uint8) * 255

        x, y, bw, bh = box_xywh[:, orig_idx]
        x1 = max(0, int((x - bw/2) * img_w / MODEL_SIZE[0]))
        y1 = max(0, int((y - bh/2) * img_h / MODEL_SIZE[1]))
        x2 = min(img_w, int((x + bw/2) * img_w / MODEL_SIZE[0]))
        y2 = min(img_h, int((y + bh/2) * img_h / MODEL_SIZE[1]))

        mask_crop = np.zeros_like(m_bin)
        mask_crop[y1:y2, x1:x2] = m_bin[y1:y2, x1:x2]
        combined_mask = cv2.bitwise_or(combined_mask, mask_crop)

    return None, None, combined_mask


def connect_nearby_components(mask, max_gap=60, line_thickness=2):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    valid = [i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= MIN_AREA]

    label_points = {}
    for lb in valid:
        ys, xs = np.where(labels == lb)
        label_points[lb] = np.column_stack([xs, ys])

    mask_out = mask.copy()
    n = len(valid)
    for i in range(n):
        for j in range(i+1, n):
            li, lj = valid[i], valid[j]
            pi = label_points[li]
            pj = label_points[lj]
            si = pi[::max(1, len(pi)//200)]
            sj = pj[::max(1, len(pj)//200)]
            diff = si[:, None, :] - sj[None, :, :]
            dists = np.sqrt((diff ** 2).sum(-1))
            min_dist = dists.min()
            if min_dist < max_gap:
                idx = np.unravel_index(np.argmin(dists), dists.shape)
                p1 = tuple(si[idx[0]])
                p2 = tuple(sj[idx[1]])
                cv2.line(mask_out, p1, p2, 255, line_thickness)
    return mask_out


def draw(image, combined_mask):
    overlay = image.copy()
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0),
              (0, 255, 255), (255, 0, 255), (255, 255, 0)]

    connected = connect_nearby_components(combined_mask, max_gap=MAX_GAP, line_thickness=2)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(connected)

    valid_labels = []
    total_area = 0
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < MIN_AREA:
            continue
        valid_labels.append(label_id)
        total_area += area
        color = colors[(len(valid_labels) - 1) % len(colors)]
        overlay[labels == label_id] = color

    result = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)

    for k, label_id in enumerate(valid_labels):
        cx, cy = centroids[label_id]
        cv2.putText(result, f"#{k+1}", (int(cx)-15, int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return result, len(valid_labels), total_area


if __name__ == '__main__':
    rknn_lite = RKNNLite()
    print('--> Load RKNN model')
    ret = rknn_lite.load_rknn(RKNN_MODEL)
    if ret != 0:
        print('Load RKNN model failed')
        exit(ret)
    print('done')

    print('--> Init runtime environment')
    ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH)

    # ===== 获取所有图片，按名称排序 =====
    img_list = sorted([f for f in os.listdir(IMG_FOLDER)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    total_imgs = len(img_list)

    print("\n" + "=" * 60)
    print(f"开始批量检测：共 {total_imgs} 张图片")
    print("=" * 60)

    # ===== 汇总列表 =====
    summary = []

    for idx, img_name in enumerate(img_list, 1):
        img_path = os.path.join(IMG_FOLDER, img_name)
        img_src = cv2.imread(img_path)
        if img_src is None:
            print(f"\n[{idx}/{total_imgs}] {img_name}: 读取失败，跳过")
            summary.append((img_name, 0, 0, "读取失败"))
            continue

        h, w = img_src.shape[:2]
        img = letter_box(im=img_src.copy(), new_shape=(MODEL_SIZE[1], MODEL_SIZE[0]), pad_color=(0, 0, 0))
        input = np.expand_dims(img, axis=0)

        outputs = rknn_lite.inference([input])
        _, _, combined_mask = post_process(outputs, w, h)

        print(f"\n[{idx}/{total_imgs}] 处理: {img_name}  ({w}x{h})")

        if combined_mask is not None:
            result, crack_count, crack_area = draw(img_src.copy(), combined_mask)
            status = f"{crack_count} 条"
            print(f"  ✅ 检测到 {crack_count} 条裂缝，总面积 {crack_area} 像素")
            summary.append((img_name, crack_count, crack_area, status))
        else:
            result = img_src.copy()
            print(f"  ⚪ 未检测到裂缝")
            summary.append((img_name, 0, 0, "无裂缝"))

        result_path = os.path.join(RESULT_PATH, img_name)
        cv2.imwrite(result_path, result)

    # ===== 打印汇总报告 =====
    print("\n" + "=" * 60)
    print("批量检测汇总报告")
    print("=" * 60)
    print(f"{'序号':<6}{'图片名':<20}{'裂缝数':<10}{'面积(px)':<12}")
    print("-" * 60)

    total_cracks = 0
    for i, (name, count, area, _) in enumerate(summary, 1):
        print(f"{i:<6}{name:<20}{count:<10}{area:<12}")
        total_cracks += count

    print("-" * 60)
    print(f"共处理 {total_imgs} 张图片，检测到 {total_cracks} 条裂缝")

    # 统计有多少张图检测到裂缝
    imgs_with_crack = sum(1 for _, c, _, _ in summary if c > 0)
    print(f"其中 {imgs_with_crack} 张图检测到裂缝，{total_imgs - imgs_with_crack} 张未检测到")

    print("=" * 60)
    print(f"结果图片已保存到 {RESULT_PATH}/")
    print("=" * 60)

    rknn_lite.release()

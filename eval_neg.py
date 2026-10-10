#!/usr/bin/env python3
"""
负样本误检率评测 —— 裂缝检测模型在无裂缝表面上的误报量化.

输入: eval_results/negative/images/*.jpg  (SDNET2018 UW 无裂缝墙面子集)
口径: 与部署实时路径一致的 conf 阈值 OBJ_THRESH=0.18（非 eval 的 0.01）。
      用 eval_py.post_process_per_inst 双线性掩码路径做检测,
      统计每张负样本的误检框数 (FP/图) 与误检掩码像素, 并按 30FPS 折算 FP/小时。

用法 (板子):
    python3 eval_neg.py yolo8n_int8_cut.rknn eval_results/negative/images
"""
import sys, os, glob
import numpy as np
import cv2
from rknnlite.api import RKNNLite

# 复用 eval_py 的双线性逐实例后处理, 仅把置信阈值换成部署口径 0.18
import eval_py
eval_py.OBJ_THRESH = 0.18          # 部署实时阈值 (rknn_seg_zc.cpp OBJ_THRESH)
from eval_py import post_process_per_inst, letter_box

FPS_REALTIME = 30.0                # 相机实时节拍, 用于 FP/小时折算


def main():
    if len(sys.argv) < 3:
        print("用法: python3 eval_neg.py <model.rknn> <negative_img_dir>")
        sys.exit(1)
    model, img_dir = sys.argv[1], sys.argv[2]
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) +
                  glob.glob(os.path.join(img_dir, "*.png")))
    if not imgs:
        print("无图像于", img_dir); sys.exit(1)
    print("==== 负样本误检评测: %d 图, conf_thr=%.2f (部署实时口径) ===="
          % (len(imgs), eval_py.OBJ_THRESH))

    rknn = RKNNLite()
    if rknn.load_rknn(model) != 0 or rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0) != 0:
        print("rknn init fail"); sys.exit(1)

    fp_per_img = []          # 每图误检框数
    fp_mask_px = []          # 每图误检掩码总像素
    fp_confs = []            # 所有误检框的置信度 (看灰区分布)
    for i, p in enumerate(imgs):
        frame = cv2.imread(p)
        if frame is None:
            continue
        H, W = frame.shape[:2]
        img640 = letter_box(frame.copy())
        outputs = rknn.inference([np.expand_dims(img640, 0)])
        preds = post_process_per_inst(outputs, W, H)   # 负样本: 所有 pred 都是 FP
        fp_per_img.append(len(preds))
        mpx = 0
        for (_, conf, mask) in preds:
            fp_confs.append(float(conf))
            mpx += int(np.count_nonzero(mask))
        fp_mask_px.append(mpx)
        if (i + 1) % 20 == 0 or i + 1 == len(imgs):
            print("  %3d/%d  累计FP=%d" % (i + 1, len(imgs), sum(fp_per_img)))
    rknn.release()

    n = len(fp_per_img)
    total_fp = sum(fp_per_img)
    img_with_fp = sum(1 for x in fp_per_img if x > 0)
    fp_per_img_mean = total_fp / n if n else 0
    # FP/小时: 每图视作一帧, 30FPS → 每秒 30 帧
    fp_per_hour = fp_per_img_mean * FPS_REALTIME * 3600
    conf_arr = np.array(fp_confs) if fp_confs else np.array([0.0])

    print("\n==== 负样本误检结果 (conf_thr=%.2f) ====" % eval_py.OBJ_THRESH)
    print("负样本图数:            %d" % n)
    print("总误检框数:            %d" % total_fp)
    print("含误检的图:            %d / %d (%.1f%%)" % (img_with_fp, n, 100.0*img_with_fp/n if n else 0))
    print("FP/图 (mean):          %.3f" % fp_per_img_mean)
    print("FP/图 (max):           %d" % (max(fp_per_img) if fp_per_img else 0))
    print("误检掩码像素 (mean):   %.0f" % (sum(fp_mask_px)/n if n else 0))
    print("FP/小时 (@%dFPS):      %.0f" % (FPS_REALTIME, fp_per_hour))
    if len(fp_confs):
        print("误检置信度:            mean=%.3f p50=%.3f max=%.3f"
              % (conf_arr.mean(), np.median(conf_arr), conf_arr.max()))
        # 灰区 [0.18,0.45) 占比 —— 这部分会送云端复核
        grey = np.sum((conf_arr >= 0.18) & (conf_arr < 0.45))
        print("灰区[0.18,0.45)误检:   %d (%.1f%%) → 送云端复核" % (grey, 100.0*grey/len(fp_confs)))


if __name__ == "__main__":
    main()

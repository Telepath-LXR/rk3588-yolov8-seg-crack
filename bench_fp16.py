"""
板端 FP16 / INT8 同口径 bench —— 用 RKNNLite (自动反量化) + post_cut 后处理.
用于测 FP16 cut 模型的真实 NPU 推理 + 后处理耗时 (零拷贝 C++ 路径只支持 int8, 对 fp16 不适用).

用法 (板子):
  python3 bench_fp16.py <model.rknn> <图像目录或视频> [帧数]
  python3 bench_fp16.py yolo8n_fp16_cut.rknn crack-seg/images/val 200
"""
import sys, os, glob, time, gc
import numpy as np
import cv2
from rknnlite.api import RKNNLite
from post_cut import post_process

def letter_box(im, new_shape=640):
    shape = im.shape[:2]
    r = min(new_shape / shape[0], new_shape / shape[1])
    nu = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape - nu[0], new_shape - nu[1]; dw /= 2; dh /= 2
    if shape[::-1] != nu: im = cv2.resize(im, nu)
    t, b = int(round(dh - 0.1)), int(round(dh + 0.1))
    l, rr = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(im, t, b, l, rr, cv2.BORDER_CONSTANT, value=(0, 0, 0))

def main():
    if len(sys.argv) < 3:
        print("用法: python3 bench_fp16.py <model.rknn> <图像目录|视频> [帧数]"); sys.exit(1)
    model, src = sys.argv[1], sys.argv[2]
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 200

    rknn = RKNNLite()
    if rknn.load_rknn(model) != 0 or rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0) != 0:
        print("rknn init fail"); sys.exit(1)
    rt = rknn.rknn_runtime

    # 输入源
    imgs = []
    if os.path.isdir(src):
        imgs = sorted(glob.glob(os.path.join(src, "*.jpg")) + glob.glob(os.path.join(src, "*.png")))[:N]
        print("图像目录: %d 张" % len(imgs))
    else:
        print("视频源:", src)

    infer_l, post_l = [], []
    n = 0; det = 0
    if imgs:
        for p in imgs:
            f = cv2.imread(p)
            if f is None: continue
            H, W = f.shape[:2]
            img = letter_box(f.copy())
            inp = np.expand_dims(img, 0)
            t1 = time.time(); outputs = rknn.inference([inp]); ti = time.time() - t1
            t2 = time.time(); res = post_process(outputs, W, H); tp = time.time() - t2
            infer_l.append(ti * 1000); post_l.append(tp * 1000); n += 1
            if res is not None: det += 1
            if n % 20 == 0: print("  %d/%d infer=%.1f post=%.1f" % (n, len(imgs), ti*1000, tp*1000))
    rknn.release()

    a = np.array(infer_l); b = np.array(post_l); e = a + b
    def st(x): return "mean=%.1f p50=%.1f p95=%.1f" % (x.mean(), np.percentile(x,50), np.percentile(x,95))
    print("\n=== %s (%d 帧) ===" % (os.path.basename(model), n))
    print("NPU 推理 (RKNNLite, 含拷贝):  %s ms" % st(a))
    print("后处理 (post_cut numpy):     %s ms" % st(b))
    print("端到端:                      %s ms → %.1f FPS" % (st(e), 1000.0/e.mean()))
    print("命中: %d/%d" % (det, n))

if __name__ == "__main__":
    main()

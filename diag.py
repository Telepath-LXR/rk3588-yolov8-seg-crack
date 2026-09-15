import numpy as np
import cv2
from rknn.api import RKNN

ONNX = './yolo8n.onnx'
PLATFORM = 'rk3588'
IMG = open('./quant.txt').readline().strip()


def make_input():
    img = cv2.imread(IMG)
    img = cv2.resize(img, (640, 640))
    return np.expand_dims(img, axis=0)


def build_and_run(do_quant, label):
    print('\n' + '=' * 60)
    print(f'  {label}  (do_quantization={do_quant})')
    print('=' * 60)
    rknn = RKNN(verbose=False)
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                target_platform=PLATFORM)
    rknn.load_onnx(model=ONNX)
    ret = rknn.build(do_quantization=do_quant, dataset='./quant.txt')
    if ret != 0:
        print('build failed', ret); rknn.release(); return
    # 不传 target = 模拟器
    ret = rknn.init_runtime()
    if ret != 0:
        print('init_runtime failed', ret); rknn.release(); return

    inp = make_input()
    out = rknn.inference(inputs=[inp])
    print('输出个数:', len(out))
    for i, o in enumerate(out):
        print(f'  output[{i}] shape={o.shape} dtype={o.dtype} '
              f'min={o.min():.4f} max={o.max():.4f} mean={o.mean():.4f}')

    pred = out[0]
    print('\n  pred(out[0]) shape=', pred.shape)
    # 尝试两种布局
    if pred.shape[1] == 37:
        p = pred[0]            # [37,8400]
        layout = '[1,37,8400]'
    elif pred.shape[2] == 37:
        p = pred[0].T          # 转成 [37,8400]
        layout = '[1,8400,37] -> 已转置'
    else:
        p = pred[0]
        layout = '未知布局'
    print('  布局判定:', layout)
    scores = p[4, :]
    print(f'  置信度行 p[4,:]  min={scores.min():.4f} max={scores.max():.4f} '
          f'mean={scores.mean():.4f}')
    print(f'  >0.25 的数量: {(scores > 0.25).sum()}')
    print(f'  >0.001 的数量: {(scores > 0.001).sum()}')
    print(f'  top5: {np.sort(scores)[-5:][::-1]}')
    rknn.release()
    return out


o_fp = build_and_run(False, 'FP16 对照组')
o_i8 = build_and_run(True, 'INT8 量化组')

# 直接逐元素对比置信度行
if o_fp is not None and o_i8 is not None:
    print('\n' + '=' * 60)
    print('  置信度行 对比')
    print('=' * 60)
    pf = o_fp[0][0] if o_fp[0][0].shape[0] == 37 else o_fp[0][0].T
    pi = o_i8[0][0] if o_i8[0][0].shape[0] == 37 else o_i8[0][0].T
    sf, si = pf[4, :], pi[4, :]
    print(f'  FP16 top5: {np.sort(sf)[-5:][::-1]}')
    print(f'  INT8 top5: {np.sort(si)[-5:][::-1]}')
    print(f'  FP16>0.25 数: {(sf > 0.25).sum()}   INT8>0.25 数: {(si > 0.25).sum()}')

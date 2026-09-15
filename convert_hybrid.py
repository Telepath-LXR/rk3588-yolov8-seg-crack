import os
import sys
import numpy as np
from rknn.api import RKNN

# ===== 配置（按你的实际路径） =====
ONNX_MODEL = './yolo8n.onnx'
DATASET_PATH = './quant.txt'
RKNN_PATH = './yolo8n_int8_hybrid.rknn'
PLATFORM = 'rk3588'

if __name__ == '__main__':
    rknn = RKNN(verbose=True)

    # Pre-process config
    # mean=0, std=255 等价于 input/255，和官方 convert.py 一致
    print('--> Config model')
    rknn.config(mean_values=[[0, 0, 0]],
                std_values=[[255, 255, 255]],
                target_platform=PLATFORM)
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed! ret=%d' % ret)
        exit(ret)
    print('done')

    # Build model —— 关键：auto_hybrid=True
    # 它会按 cos 相似度自动把量化后掉点严重的层（你的 Sigmoid/Softmax/解码那串）回退 FP16
    print('--> Building model (INT8 with auto_hybrid)')
    ret = rknn.build(do_quantization=True, dataset=DATASET_PATH, auto_hybrid=True)
    if ret != 0:
        print('Build model failed! ret=%d' % ret)
        exit(ret)
    print('done')

    # Export
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_PATH)
    if ret != 0:
        print('Export rknn model failed! ret=%d' % ret)
        exit(ret)
    print('done')

    rknn.release()
    print('\n==> 输出文件:', os.path.abspath(RKNN_PATH))

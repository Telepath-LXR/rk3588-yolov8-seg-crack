import os
from rknn.api import RKNN

ONNX_MODEL = './yolo8n_cut.onnx'
DATASET_PATH = './quant.txt'
RKNN_PATH = './yolo8n_int8_cut.rknn'
PLATFORM = 'rk3588'

if __name__ == '__main__':
    rknn = RKNN(verbose=True)

    print('--> Config model')
    # mean=0, std=255：rknn 内部做 (img-0)/255 = img/255
    rknn.config(mean_values=[[0, 0, 0]],
                std_values=[[255, 255, 255]],
                target_platform=PLATFORM)
    print('done')

    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed! ret=%d' % ret); exit(ret)
    print('done')

    print('--> Building model (INT8, 端到端敏感算子已剪到后处理)')
    # 注意这里 do_quantization=True, 不加 auto_hybrid——
    # 因为 DFL/Softmax/Sigmoid/box解码 全部不在图里了，INT8 量化的是纯 Conv，不会塌
    ret = rknn.build(do_quantization=True, dataset=DATASET_PATH)
    if ret != 0:
        print('Build model failed! ret=%d' % ret); exit(ret)
    print('done')

    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_PATH)
    if ret != 0:
        print('Export rknn model failed! ret=%d' % ret); exit(ret)
    print('done')

    rknn.release()
    print('\n==> 输出:', os.path.abspath(RKNN_PATH))

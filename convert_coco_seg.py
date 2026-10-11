"""YOLOv8n-seg COCO 80类 cut-tail ONNX → RKNN INT8. 用 rknn2 环境."""
import os
from rknn.api import RKNN

ONNX_MODEL = './yolov8n-seg-coco_cut.onnx'
DATASET_PATH = './quant.txt'          # 复用 crack 的 50 张校准图 (仅 INT8 标定用, 域不敏感)
RKNN_PATH = './yolov8n_seg_coco_int8_cut.rknn'
PLATFORM = 'rk3588'

if __name__ == '__main__':
    rknn = RKNN(verbose=False)
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform=PLATFORM)
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0: print('Load failed', ret); exit(ret)
    ret = rknn.build(do_quantization=True, dataset=DATASET_PATH)
    if ret != 0: print('Build failed', ret); exit(ret)
    ret = rknn.export_rknn(RKNN_PATH)
    if ret != 0: print('Export failed', ret); exit(ret)
    rknn.release()
    print('==> 输出:', os.path.abspath(RKNN_PATH))

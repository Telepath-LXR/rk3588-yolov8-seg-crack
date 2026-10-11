"""
YOLOv8n-seg COCO 80类 → cut-tail ONNX (10 分支输出, 与 crack 单类同结构).
在主机 pytorch 环境跑 (有 ultralytics + onnx).

cut 点 (与 crack 版 yolo8n_cut.onnx 完全一致, 仅 cls 通道 1→80):
  out0-2: /model.22/cv2.{0,1,2}/...Conv_output_0   box DFL (64ch)  3 scales
  out3-5: /model.22/cv3.{0,1,2}/...Conv_output_0   cls (80ch)      3 scales
  out6-8: /model.22/cv4.{0,1,2}/...Conv_output_0   mask coeff(32ch) 3 scales
  out9  : /model.22/proto/cv3/act/Mul_output_0     proto (32x160x160, 含SiLU)

用法:
  python3 export_coco_seg_cut.py
产出: yolov8n-seg-coco_cut.onnx
"""
from ultralytics import YOLO
import onnx

PT = "yolov8n-seg.pt"
ONNX_FULL = "yolov8n-seg.onnx"
ONNX_CUT = "yolov8n-seg-coco_cut.onnx"

# 1. 导出完整 ONNX
print("==> 导出完整 ONNX (opset 12, 与 crack 版一致)")
m = YOLO(PT)
m.export(format="onnx", opset=12, simplify=True, imgsz=640)

# 2. cut-tail: 截到 10 个分支输出
print("==> cut-tail 到 10 分支")
model = onnx.load(ONNX_FULL)

KEEP = [
    "/model.22/cv2.0/cv2.0.2/Conv_output_0",
    "/model.22/cv3.0/cv3.0.2/Conv_output_0",
    "/model.22/cv4.0/cv4.0.2/Conv_output_0",
    "/model.22/cv2.1/cv2.1.2/Conv_output_0",
    "/model.22/cv3.1/cv3.1.2/Conv_output_0",
    "/model.22/cv4.1/cv4.1.2/Conv_output_0",
    "/model.22/cv2.2/cv2.2.2/Conv_output_0",
    "/model.22/cv3.2/cv3.2.2/Conv_output_0",
    "/model.22/cv4.2/cv4.2.2/Conv_output_0",
    "/model.22/proto/cv3/act/Mul_output_0",
]

# 找到这些名字对应的 value_info (中间张量), 设为输出
name2vi = {}
for vi in list(model.graph.output) + list(model.graph.value_info):
    name2vi[vi.name] = vi

# 先跑一遍 shape inference 补全中间张量的 value_info
from onnx import shape_inference
model = shape_inference.infer_shapes(model)
for vi in model.graph.value_info:
    name2vi[vi.name] = vi

new_outputs = []
missing = []
for nm in KEEP:
    if nm in name2vi:
        new_outputs.append(name2vi[nm])
    else:
        missing.append(nm)

if missing:
    print("!! 以下 cut 点没找到 (可能图结构不同, 需检查):")
    for nm in missing:
        print("   ", nm)
    # 列出所有 model.22 下的 Conv/Mul 输出供参考
    print("\n可用的 model.22 节点输出:")
    for node in model.graph.node:
        if "/model.22/" in node.name or any("/model.22/" in o for o in node.output):
            for o in node.output:
                print("   ", node.op_type, o)
    raise SystemExit(1)

del model.graph.output[:]
model.graph.output.extend(new_outputs)
onnx.checker.check_model(model)
onnx.save(model, ONNX_CUT)
print("==> 输出:", ONNX_CUT)
print("    输出数:", len(model.graph.output))
for i, o in enumerate(model.graph.output):
    print("    out%d %s" % (i, o.name))

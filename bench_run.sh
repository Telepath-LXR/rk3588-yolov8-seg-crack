#!/bin/bash
# 板端实时显示+指标采集启动器 (自动配 Wayland)
# 用法:
#   ./bench_run.sh int8          # 跑 INT8 模型, 摄像头
#   ./bench_run.sh fp16          # 跑 FP16 模型, 摄像头
#   ./bench_run.sh int8 /dev/video0
#   ./bench_run.sh fp16 /yolo8/test.mp4
cd "$(dirname "$0")"
export XDG_RUNTIME_DIR=/var/run
export WAYLAND_DISPLAY=wayland-0
export QT_QPA_PLATFORM=wayland
export QT_QPA_FONTDIR=/usr/share/fonts
exec python3 live_bench.py "$@"

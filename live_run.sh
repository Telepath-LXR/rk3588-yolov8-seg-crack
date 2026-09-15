#!/bin/bash
# 板端实时显示启动器: 自动设置 Wayland 环境变量, 然后 python3 跑 live_cut.py
# 用法 (板子上):  ./live_run.sh            # 用默认摄像头 /dev/video44
#                 ./live_run.sh /dev/video0
#                 ./live_run.sh /path/x.mp4 # 改测视频文件
cd "$(dirname "$0")"
export XDG_RUNTIME_DIR=/var/run
export WAYLAND_DISPLAY=wayland-0
export QT_QPA_PLATFORM=wayland
export QT_QPA_FONTDIR=/usr/share/fonts
# 触摸屏/无键盘时也能正常刷帧
export QT_IM_MODULE=qtvirtualkeyboard
exec python3 live_cut.py "$@"

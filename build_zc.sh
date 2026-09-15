#!/bin/bash
# build_zc.sh — 交叉编译 rknn_seg_zc.cpp (零拷贝 + C++ 后处理)
# 用 buildroot 工具链, 链接 sysroot 里的 librknnrt.so + opencv4
set -e
cd "$(dirname "$0")"

BR=/home/alientek/atk_dlrk3588_linux6.1/buildroot/output/alientek_rk3588
SYSROOT=$BR/host/aarch64-buildroot-linux-gnu/sysroot
CXX=$BR/host/bin/aarch64-buildroot-linux-gnu-g++

$CXX -O2 -std=c++17 \
  --sysroot=$SYSROOT \
  -I$SYSROOT/usr/include -I$SYSROOT/usr/include/opencv4 -I$SYSROOT/usr/include/rknn \
  rknn_seg_zc.cpp \
  -L$SYSROOT/usr/lib \
  -lrknnrt -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_videoio -lopencv_core \
  -Wl,-rpath,/usr/lib \
  -o rknn_seg_zc

echo "==> 产物:"
file rknn_seg_zc

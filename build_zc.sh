#!/bin/bash
set -e
cd "$(dirname "$0")"

BR=/home/alientek/atk_dlrk3588_linux6.1/buildroot/output/alientek_rk3588
SYSROOT=$BR/host/aarch64-buildroot-linux-gnu/sysroot
CXX=$BR/host/bin/aarch64-buildroot-linux-gnu-g++

RTSP="${RTSP:-1}"
FFMPEG_SRC=/home/alientek/ffmpeg-rockchip
FFMPEG_LIBS=/opt/ffmpeg-rkmpp-libs/lib

SRCS="rknn_seg_zc.cpp"
RTSP_FLAGS=""
RTSP_INCLUDES=""
RTSP_LIBS=""
if [ "$RTSP" = "1" ]; then
  SRCS="$SRCS rtsp_source.cpp rga_convert.cpp"
  RTSP_FLAGS="-DWITH_RTSP -DWITH_PUSH"
  RTSP_INCLUDES="-I$FFMPEG_SRC -I$SYSROOT/usr/include/rga -I$SYSROOT/usr/include/libdrm"
  RTSP_LIBS="-L$FFMPEG_LIBS -lavcodec -lavformat -lavutil -lswscale -lswresample -L$SYSROOT/usr/lib -lrga -lrockchip_mpp -ldrm"
  echo "==> RTSP 支持: ON"
else
  echo "==> RTSP 支持: OFF"
fi

$CXX -O2 -std=c++17 \
  --sysroot=$SYSROOT \
  $RTSP_INCLUDES \
  -I$SYSROOT/usr/include -I$SYSROOT/usr/include/opencv4 -I$SYSROOT/usr/include/rknn \
  $RTSP_FLAGS \
  $SRCS \
  $RTSP_LIBS \
  -L$SYSROOT/usr/lib \
  -lrknnrt -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_videoio -lopencv_core \
  -Wl,-rpath-link,$FFMPEG_LIBS \
  -Wl,-rpath,/opt/ffmpeg-rkmpp/lib \
  -Wl,-rpath,/usr/lib \
  -o rknn_seg_zc

echo "==> 产物:"
file rknn_seg_zc

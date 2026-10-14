// rga_convert.h — RGA 硬件缩放 + 格式转换 (NV12 dma-buf → BGR 640x640)
//
// 用 RGA importbuffer_fd 导入 FFmpeg rkmpp 硬解出的 NV12 dma-buf,
// imcvtcolor + imresize 一步缩放到 letterbox 后的 640x640 BGR 图.
// 输出到 CPU 内存 (cv::Mat), 供 rknn_seg_zc 的 m.run() 直接吃.
#pragma once
#include <opencv2/core.hpp>

// 把 NV12 dma-buf 帧 (fd, 宽 w, 高 h) 经 RGA 转成 640x640 BGR letterbox 图.
//   fd:      NV12 帧的 dma-buf fd (来自 AVDRMFrameDescriptor.objects[0].fd)
//   w,h:     源帧分辨率
//   out:     输出 640x640 CV_8UC3 BGR (letterbox: 等比缩放 + 黑边补齐)
// 返回 true 成功. 内部已含 letterbox 等比缩放 + 黑边 (与 rknn_seg_zc.cpp letterbox() 一致).
bool rga_dma_to_bgr_letterbox(int fd, int w, int h, cv::Mat& out640);

bool rga_dma_to_bgr(int fd, int w, int h, cv::Mat& out);

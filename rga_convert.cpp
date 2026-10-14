// rga_convert.cpp — RGA 硬件缩放 + 格式转换 (NV12 dma-buf → BGR 640x640)
#include "rga_convert.h"
#include <cstdio>
#include <cstring>
#include <opencv2/imgproc.hpp>

#include <rga/im2d.h>
#include <rga/rga.h>
#include <rga/im2d_buffer.h>

// letterbox 参数 (与 rknn_seg_zc.cpp lb_params 一致)
struct LB { float r; int left, top, nu_w, nu_h; };
static LB lb_params(int oh, int ow, int ns = 640) {
    LB b; b.r = std::min((float)ns / oh, (float)ns / ow);
    b.nu_w = (int)std::round(ow * b.r); b.nu_h = (int)std::round(oh * b.r);
    b.left = (int)std::round((ns - b.nu_w) / 2.0f - 0.1f);
    b.top = (int)std::round((ns - b.nu_h) / 2.0f - 0.1f);
    return b;
}

bool rga_dma_to_bgr_letterbox(int fd, int w, int h, cv::Mat& out640) {
    const int NS = 640;
    LB b = lb_params(h, w, NS);

    // 1. 导入 NV12 dma-buf 为 RGA handle
    //    NV12: Y 平面 w*h, UV 平面 w*h/2, 总 size = w*h*3/2
    int nv12_size = w * h * 3 / 2;
    rga_buffer_handle_t src_handle = importbuffer_fd(fd, nv12_size);
    if (src_handle <= 0) { fprintf(stderr, "[rga] importbuffer_fd fail fd=%d\n", fd); return false; }

    rga_buffer_t src_img = wrapbuffer_handle(src_handle, w, h, RK_FORMAT_YCbCr_420_SP);

    // 2. 中间: NV12 → BGR_888 (原始尺寸), 再到 letterbox 缩放后的 nu_w x nu_h
    //    RGA 一次 imcvtcolor 只能改格式, 缩放单独 imresize. 为省一次拷贝, 分两步:
    //    a) NV12 -> BGR (原始分辨率), 到一个临时 CPU buffer
    cv::Mat bgr_full(h, w, CV_8UC3);
    rga_buffer_handle_t mid_handle = importbuffer_virtualaddr(bgr_full.data, w * h * 3);
    if (mid_handle <= 0) { fprintf(stderr, "[rga] import mid fail\n"); releasebuffer_handle(src_handle); return false; }
    rga_buffer_t mid_img = wrapbuffer_handle(mid_handle, w, h, RK_FORMAT_BGR_888);

    int ret = imcvtcolor(src_img, mid_img, RK_FORMAT_YCbCr_420_SP, RK_FORMAT_BGR_888);
    if (ret != IM_STATUS_SUCCESS) {
        fprintf(stderr, "[rga] imcvtcolor fail %d\n", ret);
        releasebuffer_handle(src_handle); releasebuffer_handle(mid_handle);
        return false;
    }

    // 3. BGR(原始) -> BGR(nu_w x nu_h) 等比缩放
    cv::Mat bgr_scaled(b.nu_h, b.nu_w, CV_8UC3);
    rga_buffer_handle_t scl_handle = importbuffer_virtualaddr(bgr_scaled.data, b.nu_w * b.nu_h * 3);
    rga_buffer_t scl_img = wrapbuffer_handle(scl_handle, b.nu_w, b.nu_h, RK_FORMAT_BGR_888);
    ret = imresize(mid_img, scl_img);
    if (ret != IM_STATUS_SUCCESS) {
        fprintf(stderr, "[rga] imresize fail %d\n", ret);
        releasebuffer_handle(src_handle); releasebuffer_handle(mid_handle); releasebuffer_handle(scl_handle);
        return false;
    }

    // 4. 黑边补齐到 640x640 (CPU copyMakeBorder, 边界像素少, 开销可忽略)
    out640 = cv::Mat::zeros(NS, NS, CV_8UC3);
    bgr_scaled.copyTo(out640(cv::Rect(b.left, b.top, b.nu_w, b.nu_h)));

    releasebuffer_handle(src_handle);
    releasebuffer_handle(mid_handle);
    releasebuffer_handle(scl_handle);
    return true;
}

// 从 DMA-BUF 直接转 BGR，保持原始尺寸（不做 letterbox）
bool rga_dma_to_bgr(int fd, int w, int h, cv::Mat& out) {
    out.create(h, w, CV_8UC3);
    rga_buffer_t src = wrapbuffer_fd(fd, w, h, RK_FORMAT_YCbCr_420_SP);
    rga_buffer_t dst = wrapbuffer_virtualaddr(out.data, w, h, RK_FORMAT_BGR_888);
    IM_STATUS st = imcvtcolor(src, dst, RK_FORMAT_YCbCr_420_SP, RK_FORMAT_BGR_888);
    return st == IM_STATUS_SUCCESS;
}

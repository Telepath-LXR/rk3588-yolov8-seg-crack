// rtsp_source.h — FFmpeg RTSP 拉流 + rkmpp 硬解 (零拷贝 dma-buf) 输入源
//
// 用 ffmpeg-rockchip 的 h264_rkmpp/hevc_rkmpp 硬解, 输出 AV_PIX_FMT_DRM_PRIME
// 硬解帧, 经 rga_convert 转成 640x640 BGR Mat 喂给 NPU 推理.
// 用法:
//   RtspSource src;
//   if(!src.open("rtsp://...")) return -1;
//   cv::Mat f; while(src.read(f)) { ... f 是 640x640 BGR ... }
//   src.close();
#pragma once
#include <string>
#include <opencv2/core.hpp>

class RtspSource {
public:
    RtspSource() = default;
    ~RtspSource();

    // 打开 RTSP 流. 自动选 h264_rkmpp/hevc_rkmpp 硬解. 返回 true 成功.
    bool open(const std::string& url);
    // 读一帧, 输出 640x640 BGR letterbox 图. 返回 false=流结束/出错.
    bool read(cv::Mat& out640);
    void close();
    bool is_open() const { return opened_; }
    int width()  const { return w_; }
    int height() const { return h_; }

private:
    bool opened_ = false;
    int w_ = 0, h_ = 0;
    // FFmpeg 上下文 (void* 避免头文件依赖外泄)
    void* fmt_ctx_ = nullptr;    // AVFormatContext*
    void* dec_ctx_ = nullptr;    // AVCodecContext*
    void* hw_dev_drm_ = nullptr; // AVBufferRef* (DRM)
    void* hw_dev_rkmpp_ = nullptr; // AVBufferRef* (RKMPP)
    void* frame_ = nullptr;      // AVFrame*
    void* pkt_ = nullptr;        // AVPacket*
    int vstream_ = -1;
};

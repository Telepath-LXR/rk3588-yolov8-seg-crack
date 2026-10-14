// rtsp_source.cpp — FFmpeg RTSP 拉流 + rkmpp 硬解 (零拷贝 dma-buf)
#include "rtsp_source.h"
#include "rga_convert.h"
#include <cstdio>
#include <cstring>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_drm.h>
#include <libavutil/pixdesc.h>
#include <libavutil/opt.h>
}

RtspSource::~RtspSource() { close(); }

bool RtspSource::open(const std::string& url) {
    close();
    fprintf(stderr, "[rtsp] opening %s\n", url.c_str()); fflush(stderr);
    avformat_network_init();   // RTSP/网络流必须初始化网络子系统
    AVFormatContext* fmt = nullptr;
    // 简化: 不传自定义 option, 用 RTSP 默认 (UDP; 海康默认支持), 避免 av_dict 干扰 codecpar 填充
    AVDictionary* opts = nullptr;
    av_dict_set(&opts, "rtsp_transport", "tcp", 0);
    av_dict_set(&opts, "fflags", "nobuffer", 0);
    av_dict_set(&opts, "flags", "low_delay", 0);
    av_dict_set(&opts, "max_delay", "500000", 0);
    av_dict_set(&opts, "reorder_queue_size", "0", 0);
    av_dict_set(&opts, "probesize", "32", 0);
    av_dict_set(&opts, "analyzeduration", "0", 0);
    int ret = avformat_open_input(&fmt, url.c_str(), nullptr, &opts);
    av_dict_free(&opts);
    fprintf(stderr, "[rtsp] avformat_open_input ret=%d\n", ret); fflush(stderr);
    if (ret < 0) { fprintf(stderr, "[rtsp] open fail %d\n", ret); return false; }
    fprintf(stderr, "[rtsp] before find_stream_info\n"); fflush(stderr);
    if (avformat_find_stream_info(fmt, nullptr) < 0) {
        fprintf(stderr, "[rtsp] find_stream_info fail\n");
        avformat_close_input(&fmt); return false;
    }
    fprintf(stderr, "[rtsp] after find_stream_info, nb_streams=%u\n", fmt->nb_streams); fflush(stderr);

    // 找视频流 (兼容 ffmpeg-rockchip: 用 av_find_best_stream, 比手遍历 codecpar 稳)
    int vstream = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    fprintf(stderr, "[rtsp] av_find_best_stream -> vstream=%d\n", vstream); fflush(stderr);
    if (vstream < 0) { fprintf(stderr, "[rtsp] no video stream\n"); avformat_close_input(&fmt); return false; }
    fprintf(stderr, "[rtsp] about to read codecpar\n"); fflush(stderr);
    AVStream* vst = fmt->streams[vstream];
    fprintf(stderr, "[rtsp] vst=%p index=%d id=%d\n", (void*)vst, vst?vst->index:-1, vst?vst->id:-1); fflush(stderr);
    AVCodecParameters* cp = vst->codecpar;
    fprintf(stderr, "[rtsp] cp=%p codec_id=%d %dx%d\n", (void*)cp, cp?cp->codec_id:-1, cp?cp->width:0, cp?cp->height:0); fflush(stderr);

    // rkmpp 硬解需要显式给 decoder 一个 DRM 设备上下文 (它基于 DRM 硬解).
    // 先建 DRM hwdevice, 再选 rkmpp decoder. 顺序: hwdevice 必须在 avcodec_open2 前挂上.
    AVBufferRef* hw_drm = nullptr;
    int hw_ret = av_hwdevice_ctx_create(&hw_drm, AV_HWDEVICE_TYPE_DRM, "/dev/dri/card0", nullptr, 0);
    fprintf(stderr, "[rtsp] hwdevice DRM create ret=%d\n", hw_ret); fflush(stderr);
    if (hw_ret < 0) {
        hw_ret = av_hwdevice_ctx_create(&hw_drm, AV_HWDEVICE_TYPE_DRM, nullptr, nullptr, 0);
        fprintf(stderr, "[rtsp] hwdevice DRM default ret=%d\n", hw_ret); fflush(stderr);
    }
    if (hw_ret < 0) { fprintf(stderr, "[rtsp] hwdevice DRM fail\n"); avformat_close_input(&fmt); return false; }

    // 选 rkmpp 硬解
    const char* dec_name = nullptr;
    if (cp->codec_id == AV_CODEC_ID_H264) dec_name = "h264_rkmpp";
    else if (cp->codec_id == AV_CODEC_ID_HEVC) dec_name = "hevc_rkmpp";
    else if (cp->codec_id == AV_CODEC_ID_AV1) dec_name = "av1_rkmpp";
    if (!dec_name) { fprintf(stderr, "[rtsp] unsupported codec %d\n", cp->codec_id); av_buffer_unref(&hw_drm); avformat_close_input(&fmt); return false; }
    const AVCodec* dec = avcodec_find_decoder_by_name(dec_name);
    fprintf(stderr, "[rtsp] decoder %s -> %p\n", dec_name, (void*)dec); fflush(stderr);
    if (!dec) { fprintf(stderr, "[rtsp] decoder %s not found\n", dec_name); av_buffer_unref(&hw_drm); avformat_close_input(&fmt); return false; }

    AVCodecContext* dc = avcodec_alloc_context3(dec);
    fprintf(stderr, "[rtsp] alloc_context -> %p\n", (void*)dc); fflush(stderr);
    if (!dc) { av_buffer_unref(&hw_drm); avformat_close_input(&fmt); return false; }
    if (avcodec_parameters_to_context(dc, cp) < 0) { avcodec_free_context(&dc); av_buffer_unref(&hw_drm); avformat_close_input(&fmt); return false; }
    fprintf(stderr, "[rtsp] params_to_context ok\n"); fflush(stderr);

    // 挂 hwdevice + 强制 drm_prime 输出
    dc->hw_device_ctx = av_buffer_ref(hw_drm);
    dc->pix_fmt = AV_PIX_FMT_DRM_PRIME;
    dc->extra_hw_frames = 0;
    fprintf(stderr, "[rtsp] before avcodec_open2\n"); fflush(stderr);
    dc->flags |= AV_CODEC_FLAG_LOW_DELAY;
    dc->flags2 |= AV_CODEC_FLAG2_FAST;
    dc->thread_count = 1;
    int oret = avcodec_open2(dc, dec, nullptr);
    fprintf(stderr, "[rtsp] avcodec_open2 ret=%d\n", oret); fflush(stderr);
    if (oret < 0) {
        fprintf(stderr, "[rtsp] avcodec_open2 fail %d\n", oret);
        av_buffer_unref(&hw_drm); avcodec_free_context(&dc); avformat_close_input(&fmt); return false;
    }

    fmt_ctx_ = fmt; dec_ctx_ = dc; hw_dev_drm_ = hw_drm;
    frame_ = av_frame_alloc(); pkt_ = av_packet_alloc();
    vstream_ = vstream; w_ = cp->width; h_ = cp->height;
    opened_ = true;
    fprintf(stderr, "[rtsp] opened %dx%d, decoder=%s, hwaccel drm_prime\n", w_, h_, dec_name);
    return true;
}

bool RtspSource::read(cv::Mat& out640) {
    if (!opened_) return false;
    AVFormatContext* fmt = (AVFormatContext*)fmt_ctx_;
    AVCodecContext* dc = (AVCodecContext*)dec_ctx_;
    AVFrame* frame = (AVFrame*)frame_;
    AVPacket* pkt = (AVPacket*)pkt_;

    while (true) {
        int ret = av_read_frame(fmt, pkt);
        if (ret < 0) {
            // 流结束/网络断, flush  decoder
            avcodec_send_packet(dc, nullptr);
            ret = avcodec_receive_frame(dc, frame);
            if (ret < 0) return false;
        } else {
            if (pkt->stream_index != vstream_) { av_packet_unref(pkt); continue; }
            ret = avcodec_send_packet(dc, pkt);
            av_packet_unref(pkt);
            if (ret < 0) { if (ret == AVERROR(EAGAIN)) continue; else return false; }
            ret = avcodec_receive_frame(dc, frame);
            if (ret == AVERROR(EAGAIN)) continue;
            if (ret < 0) return false;
        }

        // 硬解帧: AV_PIX_FMT_DRM_PRIME, data[0] 指向 AVDRMFrameDescriptor
        if (frame->format != AV_PIX_FMT_DRM_PRIME) {
            fprintf(stderr, "[rtsp] unexpected pix fmt %d (want drm_prime)\n", frame->format);
            av_frame_unref(frame); continue;
        }
        AVDRMFrameDescriptor* desc = (AVDRMFrameDescriptor*)frame->data[0];
        if (!desc || desc->nb_objects < 1) { av_frame_unref(frame); continue; }
        int fd = desc->objects[0].fd;
        int fw = frame->width, fh = frame->height;

        // RGA: NV12 dma-buf -> 640x640 BGR letterbox
        bool ok = rga_dma_to_bgr_letterbox(fd, fw, fh, out640);
        av_frame_unref(frame);
        if (ok) return true;
        // RGA 失败则丢这帧继续
    }
}

void RtspSource::close() {
    if (frame_) { av_frame_free((AVFrame**)&frame_); frame_ = nullptr; }
    if (pkt_) { av_packet_free((AVPacket**)&pkt_); pkt_ = nullptr; }
    if (dec_ctx_) { avcodec_free_context((AVCodecContext**)&dec_ctx_); dec_ctx_ = nullptr; }
    if (hw_dev_drm_) { av_buffer_unref((AVBufferRef**)&hw_dev_drm_); hw_dev_drm_ = nullptr; }
    if (hw_dev_rkmpp_) { av_buffer_unref((AVBufferRef**)&hw_dev_rkmpp_); hw_dev_rkmpp_ = nullptr; }
    if (fmt_ctx_) { avformat_close_input((AVFormatContext**)&fmt_ctx_); fmt_ctx_ = nullptr; }
    opened_ = false; vstream_ = -1; w_ = h_ = 0;
}

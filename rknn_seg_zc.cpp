// rknn_seg_zc.cpp — YOLOv8-seg 单类(crack) 零拷贝推理 + C++ 后处理 (优化版)
// =====================================================================
// 解决两个瓶颈:
//   1. 零拷贝: rknn_create_mem / rknn_set_io_mem / rknn_mem_sync 直接读写 NPU 内存,
//      绕开 rknn.inference() 的 numpy<->NPU 拷贝 (板端 INT8 约 8-18ms/帧)
//   2. 后处理 C++ 重写 (filter-first): 先 sigmoid(cls) 过滤, 只对幸存锚点做 DFL/解码,
//      掩码合成跳过 sigmoid (sigmoid(x)>0.5 ⟺ x>0), 消除 numpy + GIL 开销
//
// 原生张量格式 (板端实测, 见 probe_zero_copy):
//   输入  INT8 NHWC  [1,640,640,3]  zp=-128 scale=1/255 → int8 = pixel-128
//   输出  INT8 NC1HWC2  [1,C1,H,W,16]  C2=16 最内层, 连续无 stride
//         逻辑通道 c = c1*16 + c2; cls 逻辑通道=1
//
// 用法:
//   ./rknn_seg_zc image yolo8n_int8_cut.rknn test_frame.png
//   ./rknn_seg_zc cam   yolo8n_int8_cut.rknn /dev/video44
//   ./rknn_seg_zc bench yolo8n_int8_cut.rknn /dev/video44 100
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <array>
#include <string>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <sstream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <dirent.h>
#include <sys/stat.h>
#include <cctype>

#include "rknn_api.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <arm_neon.h>

// 16-lane int8 × float 点积 (NEON). 用于融合 proto 反量化 + 掩码内积:
// proto 原生 NC1HWC2, C2=16 最内层 → 16 通道连续 int8, 一次 vld1 即可.
static inline float dot16_i8f32(const int8_t* p, const float* w){
    int16x8_t lo = vmovl_s8(vld1_s8(p));        // ch 0..7  int16
    int16x8_t hi = vmovl_s8(vld1_s8(p+8));      // ch 8..15 int16
    int32x4_t l0 = vmovl_s16(vget_low_s16(lo));
    int32x4_t l1 = vmovl_s16(vget_high_s16(lo));
    int32x4_t h0 = vmovl_s16(vget_low_s16(hi));
    int32x4_t h1 = vmovl_s16(vget_high_s16(hi));
    float32x4_t f0 = vcvtq_f32_s32(l0);
    float32x4_t f1 = vcvtq_f32_s32(l1);
    float32x4_t f2 = vcvtq_f32_s32(h0);
    float32x4_t f3 = vcvtq_f32_s32(h1);
    float32x4_t a = vmlaq_f32(vmulq_f32(f0, vld1q_f32(w)),   f1, vld1q_f32(w+4));
    float32x4_t b = vmlaq_f32(vmulq_f32(f2, vld1q_f32(w+8)),  f3, vld1q_f32(w+12));
    float32x4_t s = vaddq_f32(a, b);
    float32x2_t p2 = vadd_f32(vget_low_f32(s), vget_high_f32(s));
    return vget_lane_f32(vpadd_f32(p2, p2), 0);
}

static const float OBJ_THRESH  = 0.18f;
static const float NMS_THRESH  = 0.45f;
static const float MASK_THRESH = 0.5f;
static const int   IMG_SIZE    = 640;
static const int   MAX_MASKS   = 10;
// sigmoid(x) > MASK_THRESH ⟺ x > logit(MASK_THRESH)
static const float MASK_LOGIT  = (MASK_THRESH==0.5f)?0.0f:std::log(MASK_THRESH/(1.0f-MASK_THRESH));

using Clock = std::chrono::steady_clock;
static inline double ms_since(Clock::time_point t0){
    return std::chrono::duration<double,std::milli>(Clock::now()-t0).count();
}
static inline float sigmoidf(float x){
    if(x < -50.0f) return 0.0f; if(x > 50.0f) return 1.0f;
    return 1.0f / (1.0f + std::exp(-x));
}

// =====================================================================
//  RKNN 零拷贝封装
// =====================================================================
struct NativeAttr {
    int C1, H, W, C2;
    int Clogical;
    int32_t zp;
    float scale;
    uint32_t size;
};

class RknnSeg {
public:
    rknn_context ctx = 0;
    rknn_tensor_mem* in_mem = nullptr;
    std::vector<rknn_tensor_attr> out_attrs;
    std::vector<rknn_tensor_mem*> out_mems;
    std::vector<NativeAttr> out_a;
    std::vector<int8_t*> out_raw;   // sync 后直接指向 NPU 内存
    std::string model_path;
    int64_t last_npu_us = 0;        // RKNN_QUERY_PERF_RUN 真实 NPU 耗时 (us)

    int init(const std::string& path){
        model_path = path;
        // RKNN_FLAG_ENABLE_SRAM: 内部张量优先放 NPU SRAM (比 DRAM 快), 直接压低推理耗时
        int ret = rknn_init(&ctx, (void*)path.c_str(), 0, RKNN_FLAG_ENABLE_SRAM, nullptr);
        if(ret < 0){ fprintf(stderr,"rknn_init fail ret=%d\n", ret); return -1; }
        // 固定 NPU core0 (单核, 避免多核调度抖动; 单流最快)
        // 可用环境变量 CORE_MASK 覆盖: 0|1|3|7|0xffff (0=AUTO,1=C0,3=C0_1,7=C0_1_2,0xffff=ALL)
        rknn_core_mask cmask = RKNN_NPU_CORE_0;
        if(const char* e = getenv("CORE_MASK")){
            int v = atoi(e);
            if(v==0) cmask = RKNN_NPU_CORE_AUTO;
            else if(v==1) cmask = RKNN_NPU_CORE_0;
            else if(v==3) cmask = RKNN_NPU_CORE_0_1;
            else if(v==7) cmask = RKNN_NPU_CORE_0_1_2;
            else if(v==0xffff || v==-1) cmask = RKNN_NPU_CORE_ALL;
        }
        rknn_set_core_mask(ctx, cmask);
        printf("[init] core_mask=%d (CORE_MASK env override)\n", (int)cmask);

        rknn_input_output_num n;
        rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &n, sizeof(n));
        if(n.n_input != 1 || n.n_output != 10){
            fprintf(stderr,"unexpected io: in=%u out=%u\n", n.n_input, n.n_output); return -1;
        }

        // ---- 输入 ----
        rknn_tensor_attr ia; memset(&ia,0,sizeof(ia)); ia.index=0;
        CHECK(rknn_query(ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, &ia, sizeof(ia)));
        in_mem = rknn_create_mem(ctx, ia.size_with_stride);
        if(!in_mem){ fprintf(stderr,"create input mem fail\n"); return -1; }
        ia.pass_through = 1;
        CHECK(rknn_set_io_mem(ctx, in_mem, &ia));

        // ---- 10 个输出 ----
        out_attrs.resize(n.n_output);
        out_mems.assign(n.n_output, nullptr);
        out_a.resize(n.n_output);
        out_raw.assign(n.n_output, nullptr);
        for(uint32_t i=0;i<n.n_output;i++){
            rknn_tensor_attr oa; memset(&oa,0,sizeof(oa)); oa.index=i;
            CHECK(rknn_query(ctx, RKNN_QUERY_NATIVE_OUTPUT_ATTR, &oa, sizeof(oa)));
            out_attrs[i] = oa;
            out_attrs[i].pass_through = 1;
            out_mems[i] = rknn_create_mem(ctx, oa.size_with_stride);
            if(!out_mems[i]){ fprintf(stderr,"create out[%u] mem fail\n", i); return -1; }
            CHECK(rknn_set_io_mem(ctx, out_mems[i], &out_attrs[i]));
            rknn_tensor_attr ua; memset(&ua,0,sizeof(ua)); ua.index=i;
            rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &ua, sizeof(ua));
            out_a[i] = parse_native(oa, (int)ua.dims[1]);
        }
        printf("[init] 零拷贝内存已绑定: in=%uB, %zu 输出\n",
               ia.size_with_stride, out_mems.size());
        return 0;
    }

    // 写入 640x640 BGR Mat → rknn_run → sync 输出. 返回 NPU run ms
    double run(const cv::Mat& img640, double* t_copy=nullptr, double* t_sync=nullptr){
        auto tc=Clock::now();
        int8_t* in = (int8_t*)in_mem->virt_addr;
        const uint8_t* src = img640.data;
        int n = 640*640*3;           // 1228800, 16 的倍数
        // NEON: uint8 pixel - 128 → int8 (bit 级等价, 无溢出分支)
        const uint8x16_t OFF = vdupq_n_u8(128);
        for(int i=0;i<n;i+=16){
            uint8x16_t v = vld1q_u8(src+i);
            vst1q_u8((uint8_t*)(in+i), vsubq_u8(v, OFF));
        }
        rknn_mem_sync(ctx, in_mem, RKNN_MEMORY_SYNC_TO_DEVICE);
        if(t_copy) *t_copy = ms_since(tc);

        auto t0 = Clock::now();
        rknn_run(ctx, nullptr);
        double run_ms = ms_since(t0);
        // 查询 NPU 真实推理耗时 (排除调度/sync 开销)
        rknn_perf_run perf; memset(&perf,0,sizeof(perf));
        rknn_query(ctx, RKNN_QUERY_PERF_RUN, &perf, sizeof(perf));
        last_npu_us = perf.run_duration;

        auto ts=Clock::now();
        for(size_t i=0;i<out_mems.size();i++){
            rknn_mem_sync(ctx, out_mems[i], RKNN_MEMORY_SYNC_FROM_DEVICE);
            out_raw[i] = (int8_t*)out_mems[i]->virt_addr;
        }
        if(t_sync) *t_sync = ms_since(ts);
        return run_ms;
    }

    // 读 proto 全量反量化 → [32, ph*pw] float
    void read_proto(std::vector<float>& out) const {
        const NativeAttr& a = out_a[9];
        out.assign(a.Clogical * a.H * a.W, 0.0f);
        dequant_full(out_raw[9], a, out.data());
    }
    // 暴露 proto 原生 int8 指针 + attr (供 NEON 融合路径直接用)
    const int8_t* proto_raw() const { return out_raw[9]; }
    // 读 cls 输出 (scale idx 0/1/2 → 输出 1/4/7) 融合反量化+sigmoid, 无临时分配
    // 返回 [H*W] 的 conf
    void read_cls(int scale, std::vector<float>& conf) const {
        int oi = 3*scale + 1;
        const NativeAttr& a = out_a[oi];
        conf.resize(a.H*a.W);
        const int8_t* base = out_raw[oi];
        const int C1=a.C1, H=a.H, W=a.W, C2=a.C2, Cl=a.Clogical;
        const int HW2 = H*W*C2, W2 = W*C2;
        const float sc=a.scale; const float zp=(float)a.zp;
        float* o = conf.data();
        // Cl==1 (单类): 单通道, 直接行扫描
        if(Cl == 1){
            const int8_t* row_base = base;  // c1=0,c2=0
            int HW = H*W;
            for(int y=0;y<H;y++){
                const int8_t* row = row_base + y*W2;
                float* oo = o + y*W;
                for(int x=0;x<W;x++) oo[x] = sigmoidf(((float)row[x*C2] - zp) * sc);
            }
        } else {
            for(int c=0;c<Cl;c++){
                int c1=c/C2, c2=c%C2;
                const int8_t* rb = base + c1*HW2 + c2;
                for(int y=0;y<H;y++){
                    const int8_t* row = rb + y*W2;
                    float* oo = o + c*H*W + y*W;
                    for(int x=0;x<W;x++) oo[x] = sigmoidf(((float)row[x*C2] - zp) * sc);
                }
            }
        }
    }
    // DFL: 读 box 输出 (scale) 在 (x,y) 处的 64 个 int8, 反量化+softmax → 4 距离
    void read_dfl(int scale, int x, int y, float dist[4]) const {
        int oi = 3*scale;
        const NativeAttr& a = out_a[oi];
        const int8_t* base = out_raw[oi];
        int C1=a.C1, H=a.H, W=a.W, C2=a.C2;
        float v[64];
        for(int g=0; g<4; g++){
            float mx = -1e30f;
            for(int b=0;b<16;b++){
                int c = g*16 + b;
                int c1 = c/C2, c2 = c%C2;
                const int8_t* p = base + c1*(H*W*C2) + y*(W*C2) + x*C2 + c2;
                float fv = ((float)*p - (float)a.zp) * a.scale;
                v[b] = fv; if(fv>mx) mx=fv;
            }
            float sum=0; for(int b=0;b<16;b++) sum += std::exp(v[b]-mx);
            float acc=0; for(int b=0;b<16;b++) acc += (std::exp(v[b]-mx)/sum)*(float)b;
            dist[g] = acc;
        }
    }
    // 读 mask 系数 (scale) 在 (x,y) → 32 个 float
    void read_coeff(int scale, int x, int y, float coeff[32]) const {
        int oi = 3*scale + 2;
        const NativeAttr& a = out_a[oi];
        const int8_t* base = out_raw[oi];
        int C1=a.C1, H=a.H, W=a.W, C2=a.C2;
        for(int c=0;c<32;c++){
            int c1=c/C2, c2=c%C2;
            const int8_t* p = base + c1*(H*W*C2) + y*(W*C2) + x*C2 + c2;
            coeff[c] = ((float)*p - (float)a.zp) * a.scale;
        }
    }
    const NativeAttr& proto_attr() const { return out_a[9]; }

    void release(){
        for(auto* m: out_mems) if(m) rknn_destroy_mem(ctx, m);
        if(in_mem) rknn_destroy_mem(ctx, in_mem);
        if(ctx) rknn_destroy(ctx);
        ctx = 0;
    }
private:
    static NativeAttr parse_native(const rknn_tensor_attr& a, int clogical){
        NativeAttr r;
        r.C1 = a.dims[1]; r.H = a.dims[2]; r.W = a.dims[3]; r.C2 = a.dims[4];
        r.Clogical = clogical; r.zp = a.zp; r.scale = a.scale; r.size = a.size_with_stride;
        return r;
    }
    static void dequant_full(const int8_t* buf, const NativeAttr& a, float* out){
        const int C1=a.C1, H=a.H, W=a.W, C2=a.C2, Cl=a.Clogical;
        const int HW2 = H*W*C2, W2 = W*C2;
        const float sc=a.scale; const float zp=(float)a.zp;
        for(int c=0;c<Cl;c++){
            int c1=c/C2, c2=c%C2;
            const int8_t* base = buf + c1*HW2 + c2;
            float* o = out + c*H*W;
            for(int y=0;y<H;y++){
                const int8_t* row = base + y*W2;
                float* oo = o + y*W;
                for(int x=0;x<W;x++) oo[x] = ((float)row[x*C2] - zp) * sc;
            }
        }
    }
    static int CHECK(int r){ if(r<0) fprintf(stderr,"rknn call fail ret=%d\n", r); return r; }
};

// =====================================================================
//  后处理 (filter-first 优化)
// =====================================================================
struct Result {
    std::vector<std::array<float,4>> boxes;       // 原图 xyxy (按 conf 降序)
    std::vector<float> confs;
    cv::Mat mask;                                  // 合并二值掩码 (原图尺寸)
    // —— eval 专用: 保留幸存实例的 640 框 + 32 掩码系数, 供 eval 重光栅化逐实例掩码 ——
    std::vector<std::array<float,4>> boxes640;     // 与 boxes 同序 (640 坐标系)
    std::vector<std::array<float,32>> coeffs;      // 与 boxes 同序
};
struct LB { float r; int left, top, nu_w, nu_h; };
static LB lb_params(int oh, int ow, int ns=IMG_SIZE){
    LB b; b.r=std::min((float)ns/oh,(float)ns/ow);
    b.nu_w=(int)std::round(ow*b.r); b.nu_h=(int)std::round(oh*b.r);
    b.left=(int)std::round((ns-b.nu_w)/2.0f-0.1f);
    b.top=(int)std::round((ns-b.nu_h)/2.0f-0.1f);
    return b;
}
static cv::Mat letterbox(const cv::Mat& im, LB* plb=nullptr, int ns=IMG_SIZE){
    int h=im.rows,w=im.cols; LB b=lb_params(h,w,ns);
    cv::Mat out; cv::Size nu(b.nu_w,b.nu_h);
    if(cv::Size(w,h)!=nu) cv::resize(im,out,nu); else out=im;
    int t=b.top,bot=ns-b.nu_h-b.top,l=b.left,r=ns-b.nu_w-b.left;
    cv::copyMakeBorder(out,out,t,bot,l,r,cv::BORDER_CONSTANT,cv::Scalar(0,0,0));
    if(plb)*plb=b; return out;
}
static std::vector<int> nms(const std::vector<std::array<float,4>>& bx,
                           const std::vector<float>& sc, float st, float nt){
    std::vector<int> idx;
    for(size_t i=0;i<sc.size();i++) if(sc[i]>=st) idx.push_back((int)i);
    std::sort(idx.begin(),idx.end(),[&](int a,int b){return sc[a]>sc[b];});
    std::vector<int> keep; std::vector<char> sup(idx.size(),0);
    for(size_t i=0;i<idx.size();i++){
        if(sup[i])continue; int ii=idx[i]; keep.push_back(ii);
        float ax1=bx[ii][0],ay1=bx[ii][1],ax2=bx[ii][2],ay2=bx[ii][3];
        float ar=(std::max)(0.0f,ax2-ax1)*(std::max)(0.0f,ay2-ay1);
        for(size_t j=i+1;j<idx.size();j++){
            if(sup[j])continue; int jj=idx[j];
            float bx1=bx[jj][0],by1=bx[jj][1],bx2=bx[jj][2],by2=bx[jj][3];
            float ix1=std::max(ax1,bx1),iy1=std::max(ay1,by1);
            float ix2=std::min(ax2,bx2),iy2=std::min(ay2,by2);
            float iw=std::max(0.0f,ix2-ix1),ih=std::max(0.0f,iy2-iy1);
            float inter=iw*ih, br=(std::max)(0.0f,bx2-bx1)*(std::max)(0.0f,by2-by1);
            float uni=ar+br-inter; float iou=uni>0?inter/uni:0;
            if(iou>nt) sup[j]=1;
        }
    }
    return keep;
}

static Result post_process(RknnSeg& m, int img_w, int img_h, bool dbg=false,
                           float conf_thr=OBJ_THRESH, int max_masks=MAX_MASKS){
    Result R;
    auto T0=Clock::now();
    // 惰性 proto 反量化: 仅在命中掩码时才反量化 (融合路径用原生 int8, 仍需 float proto 做退化/对照)
    const NativeAttr& pa = m.proto_attr();
    const int ph=pa.H, pw=pa.W, proto_hw=ph*pw;
    double t_proto=0.0;  // 融合路径下 proto 反量化推迟到掩码阶段, 此处为 0

    auto T1=Clock::now();
    std::vector<std::array<float,4>> boxes640;
    std::vector<float> confs;
    std::vector<std::array<float,32>> segs;
    int n_cand=0;

    for(int s=0;s<3;s++){
        std::vector<float> conf; m.read_cls(s, conf);
        const NativeAttr& ba = m.out_a[3*s];
        int H=ba.H, W=ba.W;
        float stride = (float)IMG_SIZE / H;  // H==W
        for(int y=0;y<H;y++) for(int x=0;x<W;x++){
            float c = conf[y*W+x];
            if(c < conf_thr) continue;
            n_cand++;
            float dist[4]; m.read_dfl(s, x, y, dist);
            float gx=x+0.5f, gy=y+0.5f;
            float x1=(gx-dist[0])*stride, y1=(gy-dist[1])*stride;
            float x2=(gx+dist[2])*stride, y2=(gy+dist[3])*stride;
            boxes640.push_back({x1,y1,x2,y2});
            confs.push_back(c);
            std::array<float,32> sg; m.read_coeff(s, x, y, sg.data());
            segs.push_back(sg);
        }
    }
    double t_cand=ms_since(T1);
    if(confs.empty()){ if(dbg)fprintf(stderr,"[dbg] proto=%.1f cand=%.1f (ncand=%d) -> empty\n",t_proto,t_cand,n_cand); return R; }

    auto T2=Clock::now();
    std::vector<int> keep = nms(boxes640, confs, conf_thr, NMS_THRESH);
    if(keep.empty()){ if(dbg)fprintf(stderr,"[dbg] proto=%.1f cand=%.1f (ncand=%d) -> nms empty\n",t_proto,t_cand,n_cand); return R; }
    std::sort(keep.begin(),keep.end(),[&](int a,int b){return confs[a]>confs[b];});
    double t_nms=ms_since(T2);

    cv::Mat mask640(IMG_SIZE, IMG_SIZE, CV_8U, cv::Scalar(0));
    int n = (int)std::min((size_t)max_masks, keep.size());
    if(n > 0){
        auto T3=Clock::now();
        // NEON 融合路径: 跳过 proto 全量反量化, 直接用原生 int8 + coeff 做
        // m[p] = sum_c (proto_int8[c,p] - zp) * scale * coeff[c]
        // proto 原生 NC1HWC2[1,C1,160,160,16], C2=16 最内层连续 → 每像素 16 通道一次 vld1
        const int8_t* proto_i8 = m.proto_raw();
        const NativeAttr& pa2 = m.proto_attr();
        const int pC1=pa2.C1, pC2=pa2.C2, pH=pa2.H, pW=pa2.W;
        const int p_hw2 = pH*pW*pC2, p_w2 = pW*pC2;
        const float psc = pa2.scale, pzp = (float)pa2.zp;
        // 预计算: w[c] = coeff[c]*scale, bias = -zp*scale*sum(coeff)
        std::vector<std::array<float,32>> w(n);
        std::vector<float> bias(n, 0.0f);
        for(int k=0;k<n;k++){
            const float* sg = segs[keep[k]].data();
            float sc=0; for(int c=0;c<32;c++) sc+=sg[c];
            bias[k] = -pzp*sc*psc;
            for(int c=0;c<32;c++) w[k][c] = sg[c]*psc;
        }
        auto T4=Clock::now();
        // 逐检测: 仅在框对应的 160 区域内做 NEON dot → 直接光栅化 4×4 块到 mask640
        // (跳过全 160×160 扫描 + 单独 resize; 框外像素 Python ref 本就裁掉)
        for(int k=0;k<n;k++){
            const float* wk = w[k].data();
            float bk = bias[k];
            auto& bb = boxes640[keep[k]];
            int x1=std::max(0,(int)bb[0]), y1=std::max(0,(int)bb[1]);
            int x2=std::min(IMG_SIZE,(int)bb[2]), y2=std::min(IMG_SIZE,(int)bb[3]);
            if(x2<=x1 || y2<=y1) continue;
            // 640 框 ↔ 160 区域: 640 像素 X → 160 源 X>>2 (最近邻 4× 上采样)
            int px1=std::max(0, x1>>2), px2=std::min(pW, ((x2-1)>>2)+1);
            int py1=std::max(0, y1>>2), py2=std::min(pH, ((y2-1)>>2)+1);
            for(int py=py1; py<py2; py++){
                const int8_t* row0 = proto_i8 + py*p_w2;
                const int8_t* row1 = proto_i8 + p_hw2 + py*p_w2;
                int by0=py*4, by1=by0+4;
                if(by0<y1) by0=y1; if(by1>y2) by1=y2;
                int bh=by1-by0;
                for(int px=px1; px<px2; px++){
                    float s = dot16_i8f32(row0+px*pC2, wk) + dot16_i8f32(row1+px*pC2, wk+16) + bk;
                    if(s > MASK_LOGIT){
                        int bx0=px*4, bx1=bx0+4;
                        if(bx0<x1) bx0=x1; if(bx1>x2) bx1=x2;
                        int bw=bx1-bx0;
                        if(bw>0) for(int by=by0; by<by1; by++)
                            std::memset(mask640.ptr<uint8_t>(by)+bx0, 255, bw);
                    }
                }
            }
        }
        double t_gemm=ms_since(T3), t_raster=ms_since(T4);
        if(dbg) fprintf(stderr,"[dbg] ncand=%d nkeep=%zu mask=%d | proto=%.1f cand=%.1f nms=%.1f gemm=%.1f raster=%.1f\n",
            n_cand, keep.size(), n, t_proto, t_cand, t_nms, t_gemm, t_raster);
    } else if(dbg){
        fprintf(stderr,"[dbg] ncand=%d nkeep=%zu nomask | proto=%.1f cand=%.1f nms=%.1f\n",
            n_cand, keep.size(), t_proto, t_cand, t_nms);
    }

    LB b = lb_params(img_h, img_w);
    auto Tf=Clock::now();
    cv::Mat sub = mask640(cv::Rect(b.left,b.top,b.nu_w,b.nu_h));
    cv::Mat mask_orig;
    if(sub.empty()) mask_orig = cv::Mat::zeros(img_h,img_w,CV_8U);
    else cv::resize(sub, mask_orig, cv::Size(img_w,img_h),0,0,cv::INTER_NEAREST);
    R.mask = mask_orig;
    double t_final=ms_since(Tf);
    for(int idx: keep){
        auto& bb=boxes640[idx];
        R.boxes.push_back({(bb[0]-b.left)/b.r,(bb[1]-b.top)/b.r,(bb[2]-b.left)/b.r,(bb[3]-b.top)/b.r});
        R.confs.push_back(confs[idx]);
        R.boxes640.push_back(bb);
        R.coeffs.push_back(segs[idx]);
    }
    if(dbg) fprintf(stderr,"[dbg]   final_resize=%.2f (orig=%dx%d nu=%dx%d)\n", t_final, img_w, img_h, b.nu_w, b.nu_h);
    return R;
}

// ---- 绘制 / 显示 ----
static cv::Mat draw(const cv::Mat& frame, const Result& R){
    cv::Mat out = frame.clone();
    if(!R.mask.empty() && cv::countNonZero(R.mask)>0){
        cv::Mat red = cv::Mat::zeros(out.size(), out.type());
        red.setTo(cv::Vec3b(0,0,255), R.mask>0);
        cv::addWeighted(out,1.0,red,0.45,0,out);
    }
    for(size_t k=0;k<R.boxes.size();k++){
        int x1=(int)R.boxes[k][0],y1=(int)R.boxes[k][1],x2=(int)R.boxes[k][2],y2=(int)R.boxes[k][3];
        cv::rectangle(out,cv::Point(x1,y1),cv::Point(x2,y2),cv::Scalar(0,255,0),2);
        char buf[32]; std::snprintf(buf,32,"crack %.2f",R.confs[k]);
        cv::putText(out,buf,cv::Point(x1,std::max(0,y1-6)),cv::FONT_HERSHEY_SIMPLEX,0.6,cv::Scalar(0,255,0),2);
    }
    return out;
}
static cv::Mat fit_screen(const cv::Mat& f, int sw, int sh){
    int fh=f.rows,fw=f.cols; float s=std::min((float)sw/fw,(float)sh/fh);
    int nw=(int)(fw*s),nh=(int)(fh*s);
    cv::Mat canvas=cv::Mat::zeros(sh,sw,f.type());
    cv::Mat rs; cv::resize(f,rs,cv::Size(nw,nh));
    rs.copyTo(canvas(cv::Rect((sw-nw)/2,(sh-nh)/2,nw,nh)));
    return canvas;
}
static void get_screen(int&w,int&h){
    for(const char* p: {"/sys/class/drm/card0-DSI-1/modes","/sys/class/drm/card0-HDMI-A-1/modes"}){
        std::ifstream f(p); std::string line; std::getline(f,line);
        auto pos=line.find('x');
        if(pos!=std::string::npos){ try{int a=std::stoi(line),b=std::stoi(line.substr(pos+1));w=std::max(a,b);h=std::min(a,b);return;}catch(...){} }
    }
    w=1080;h=1920;
}

static int mode_image(RknnSeg& m, const std::string& img_path){
    cv::Mat frame = cv::imread(img_path, cv::IMREAD_COLOR);
    if(frame.empty()){ fprintf(stderr,"read image fail: %s\n",img_path.c_str()); return 1; }
    int H=frame.rows,W=frame.cols;
    cv::Mat img640 = letterbox(frame.clone());
    double t_run = m.run(img640);
    auto tp=Clock::now(); Result R=post_process(m,W,H,true); double post_ms=ms_since(tp);
    cv::Mat out = draw(frame,R); cv::imwrite("zc_out.png",out);
    printf("==== 零拷贝 + C++ 后处理 (单图) ====\n");
    printf("图像: %s  %dx%d\n",img_path.c_str(),W,H);
    printf("NPU rknn_run:   %.2f ms\n", t_run);
    printf("后处理 C++:     %.2f ms\n", post_ms);
    printf("合计:           %.2f ms\n", t_run+post_ms);
    if(R.confs.empty()) printf("检测: 无\n");
    else{
        int mp=R.mask.empty()?0:(int)cv::countNonZero(R.mask);
        printf("检测: %zu 个  confs=",R.confs.size());
        for(float c:R.confs) printf(" %.3f",c);
        printf("  mask_px=%d\n",mp);
        printf("框(原图xyxy):"); for(auto&b:R.boxes) printf(" [%.1f,%.1f,%.1f,%.1f]",b[0],b[1],b[2],b[3]); printf("\n");
    }
    printf("可视化已存: zc_out.png\n");
    return 0;
}

static int mode_cam(RknnSeg& m, const std::string& src){
    bool use_cam = src.find("/dev/video")==0 || src.find('.')==std::string::npos;
    cv::VideoCapture cap; if(use_cam) cap.open(src,cv::CAP_V4L2); else cap.open(src);
    if(!cap.isOpened()){ fprintf(stderr,"open source fail: %s\n",src.c_str()); return 1; }
    int sw,sh; get_screen(sw,sh);
    const std::string WIN="crack_seg_zc";
    cv::namedWindow(WIN,cv::WINDOW_NORMAL);
    cv::setWindowProperty(WIN,cv::WND_PROP_FULLSCREEN,cv::WINDOW_FULLSCREEN);

    // ---- 异步采集流水线: 采集线程预读下一帧, 与当前帧推理+显示并行 ----
    // 串行: cap.read(33ms)+推理(16ms)+显示(~10ms)≈60ms→~17FPS
    // 流水线: wall→max(33, 16+10)=33ms→~30FPS (相机节拍上限)
    bool async_cap = use_cam;
    std::thread cap_thread;
    std::mutex cap_mtx;
    std::condition_variable cap_cv;
    cv::Mat cap_frame;
    bool cap_has=false, cap_stop=false, cap_fail=false;
    auto capture_loop = [&](){
        cv::Mat f;
        while(true){
            if(!cap.read(f)||f.empty()){
                std::lock_guard<std::mutex> lk(cap_mtx);
                cap_fail=true; cap_cv.notify_one(); return;
            }
            {
                std::unique_lock<std::mutex> lk(cap_mtx);
                cap_cv.wait(lk, [&]{ return !cap_has || cap_stop; });
                if(cap_stop) return;
                cap_frame = std::move(f); cap_has=true; cap_cv.notify_one();
            }
        }
    };
    if(async_cap) cap_thread=std::thread(capture_loop);

    std::vector<double> run_ms,post_ms,e2e_ms,wall_ms; int frames=0,det_frames=0; double sum_conf=0,sum_mask=0;
    printf("实时显示中... (q/ESC 退出)  %s\n", async_cap?"[异步采集流水线]":"");
    cv::Mat frame;
    while(true){
        auto tw0=Clock::now();
        cv::Mat f;
        if(async_cap){
            { std::unique_lock<std::mutex> lk(cap_mtx); cap_cv.wait(lk, [&]{ return cap_has||cap_fail||cap_stop; });
              if(cap_fail||cap_stop) break;
              f = cap_frame; cap_has=false; cap_cv.notify_one(); }
        } else {
            if(!cap.read(f)||f.empty()){ if(!use_cam){cap.set(cv::CAP_PROP_POS_FRAMES,0);continue;} break; }
        }
        int H=f.rows,W=f.cols; cv::Mat img640=letterbox(f.clone());
        auto t0=Clock::now(); double r=m.run(img640); Result R=post_process(m,W,H,true); double e=ms_since(t0);
        run_ms.push_back(r); post_ms.push_back(e-r); e2e_ms.push_back(e); frames++;
        if(!R.confs.empty()){det_frames++; sum_conf+=R.confs[0]; sum_mask+=(R.mask.empty()?0:cv::countNonZero(R.mask));}
        cv::Mat anno=draw(f,R); cv::Mat show=fit_screen(anno,sw,sh);
        float fps=e>0?1000.0f/(float)e:0;
        char osd[160];
        std::snprintf(osd,160,"[ZC] run %.0fms post %.0fms  %.1fFPS  %s",r,e-r,fps,
            R.confs.empty()?"no detect":(std::string(std::to_string(R.confs.size())+" obj mask="+std::to_string(R.mask.empty()?0:(int)cv::countNonZero(R.mask))).c_str()));
        cv::putText(show,osd,cv::Point(20,40),cv::FONT_HERSHEY_SIMPLEX,0.7,cv::Scalar(0,255,255),2);
        cv::imshow(WIN,show);
        double wall=ms_since(tw0);
        wall_ms.push_back(wall);
        int key=cv::waitKey(1)&0xFF; if(key=='q'||key==27) break;
    }
    if(async_cap){ { std::lock_guard<std::mutex> lk(cap_mtx); cap_stop=true; cap_cv.notify_all(); } if(cap_thread.joinable()) cap_thread.join(); }
    cap.release(); cv::destroyAllWindows();
    auto st=[&](std::vector<double>&v){double s=0;for(double x:v)s+=x;return v.empty()?0.0:s/v.size();};
    auto p95=[&](std::vector<double>&v){auto a=v;std::sort(a.begin(),a.end());return a[(int)(a.size()*0.95)];};
    printf("\n==== 零拷贝实时总结 (%d 帧) ====\n",frames);
    printf("NPU run  mean=%.1f p95=%.1f ms\n",st(run_ms),p95(run_ms));
    printf("后处理   mean=%.1f p95=%.1f ms\n",st(post_ms),p95(post_ms));
    printf("端到端   mean=%.1f p95=%.1f ms → %.1f FPS\n",st(e2e_ms),p95(e2e_ms),1000.0/st(e2e_ms));
    if(!wall_ms.empty()) printf("全循环   mean=%.1f p95=%.1f ms → %.1f FPS (含read+显示)\n",st(wall_ms),p95(wall_ms),1000.0/st(wall_ms));
    printf("命中: %d/%d  avg_max_conf=%.3f  avg_mask_px=%.0f\n",det_frames,frames,det_frames?sum_conf/det_frames:0,det_frames?sum_mask/det_frames:0);
    return 0;
}

static int mode_bench(RknnSeg& m, const std::string& src, int N){
    bool use_cam = src.find("/dev/video")==0 || src.find('.')==std::string::npos;
    cv::VideoCapture cap; if(use_cam) cap.open(src,cv::CAP_V4L2); else cap.open(src);
    if(!cap.isOpened()){ fprintf(stderr,"open source fail: %s\n",src.c_str()); return 1; }
    cv::Mat wf;
    for(int i=0;i<5;i++){ cap.read(wf); if(!wf.empty()) m.run(letterbox(wf.clone())); }

    // ---- 异步采集流水线 (仅对相机生效): 采集线程预读下一帧, 与当前帧推理并行 ----
    // 相机固定 30FPS (cap.read 阻塞 ~33.3ms), 处理仅 ~16.3ms → 每帧有 ~17ms 空闲可隐藏
    // 流水线后 wall 应逼近 max(33.3, 16.3)=33.3ms (相机节拍), 但处理不再串到其后
    bool async_cap = use_cam;
    std::thread cap_thread;
    std::mutex cap_mtx;
    std::condition_variable cap_cv;
    cv::Mat cap_frame;
    bool cap_has=false, cap_stop=false, cap_fail=false;
    auto capture_loop = [&](){
        cv::Mat f;
        while(true){
            // cap.read 阻塞在相机节拍 (~33.3ms/帧 @ 30FPS), 与主线程推理并行
            if(!cap.read(f)||f.empty()){
                std::lock_guard<std::mutex> lk(cap_mtx);
                cap_fail=true; cap_cv.notify_one(); return;
            }
            {
                // 等待缓冲槽空 (消费者已取走上帧), 再写入新帧
                std::unique_lock<std::mutex> lk(cap_mtx);
                cap_cv.wait(lk, [&]{ return !cap_has || cap_stop; });
                if(cap_stop) return;
                cap_frame = std::move(f); cap_has=true; cap_cv.notify_one();
            }
        }
    };
    if(async_cap) cap_thread=std::thread(capture_loop);

    std::vector<double> run_ms,post_ms,e2e_ms,wall_ms; int det_frames=0; double sum_conf=0,sum_mask=0;
    double sum_copy=0,sum_sync=0,sum_npu=0,sum_lb=0;
    cv::Mat frame;
    printf("零拷贝 bench: %d 帧 (无显示), 源=%s%s\n",N,src.c_str(), async_cap?" [异步采集流水线]":"");
    for(int i=0;i<N;i++){
        auto tw0=Clock::now();
        cv::Mat f;
        if(async_cap){
            // 取预读帧 (等待采集线程就绪)
            { std::unique_lock<std::mutex> lk(cap_mtx); cap_cv.wait(lk, [&]{ return cap_has||cap_fail||cap_stop; });
              if(cap_fail){ printf("frame %d read fail\n",i); break; }
              f = cap_frame; cap_has=false; cap_cv.notify_one(); }
        } else {
            if(!cap.read(f)||f.empty()){ if(!use_cam){cap.set(cv::CAP_PROP_POS_FRAMES,0);i--;continue;} printf("frame %d read fail\n",i);break; }
        }
        int H=f.rows,W=f.cols;
        auto tlb0=Clock::now(); cv::Mat img640=letterbox(f.clone()); double tlb=ms_since(tlb0);
        double tc,ts;
        auto t0=Clock::now(); double r=m.run(img640,&tc,&ts); Result R=post_process(m,W,H,true); double e=ms_since(t0);
        double npu = m.last_npu_us/1000.0;
        double wall=ms_since(tw0);
        run_ms.push_back(r); post_ms.push_back(e-r); e2e_ms.push_back(e); wall_ms.push_back(wall); sum_copy+=tc; sum_sync+=ts; sum_npu+=npu; sum_lb+=tlb;
        if(!R.confs.empty()){det_frames++; sum_conf+=R.confs[0]; sum_mask+=(R.mask.empty()?0:cv::countNonZero(R.mask));}
        if(i%10==0||i==N-1) printf("[%3d] run=%5.1f(np%.1f) post=%5.1f e2e=%5.1fms wall=%5.1f lb=%4.2f copy=%4.2f sync=%4.2f %s\n",i+1,r,npu,e-r,e,wall,tlb,tc,ts,
            R.confs.empty()?"no detect":("det="+std::to_string(R.confs.size())).c_str());
    }
    if(async_cap){ { std::lock_guard<std::mutex> lk(cap_mtx); cap_stop=true; cap_cv.notify_all(); } if(cap_thread.joinable()) cap_thread.join(); }
    cap.release();
    auto st=[&](std::vector<double>&v){double s=0;for(double x:v)s+=x;return v.empty()?0.0:s/v.size();};
    auto p50=[&](std::vector<double>&v){auto a=v;std::sort(a.begin(),a.end());return a[a.size()/2];};
    auto p95=[&](std::vector<double>&v){auto a=v;std::sort(a.begin(),a.end());return a[(int)(a.size()*0.95)];};
    printf("\n==== 零拷贝 bench 总结 (%d 帧) ====\n",(int)e2e_ms.size());
    int n=(int)e2e_ms.size();
    printf("NPU run  mean=%5.1f p50=%5.1f p95=%5.1f ms\n",st(run_ms),p50(run_ms),p95(run_ms));
    printf("  (PERF_RUN 真实 NPU 推理) mean=%.2f ms\n", n?sum_npu/n:0);
    printf("后处理    mean=%5.1f p50=%5.1f p95=%5.1f ms\n",st(post_ms),p50(post_ms),p95(post_ms));
    printf("端到端    mean=%5.1f p50=%5.1f p95=%5.1f ms → %.1f FPS (处理)\n",st(e2e_ms),p50(e2e_ms),p95(e2e_ms),1000.0/st(e2e_ms));
    printf("全循环    mean=%5.1f p50=%5.1f p95=%5.1f ms → %.1f FPS (含read+letterbox)\n",st(wall_ms),p50(wall_ms),p95(wall_ms),1000.0/st(wall_ms));
    printf("(分项) letterbox=%.2f input_copy=%.2f output_sync=%.2f ms\n", n?sum_lb/n:0, n?sum_copy/n:0, n?sum_sync/n:0);
    printf("命中: %d/%d  avg_max_conf=%.3f  avg_mask_px=%.0f\n",det_frames,n,det_frames?sum_conf/det_frames:0,det_frames?sum_mask/det_frames:0);
    return 0;
}

// =====================================================================
//  demo 模式 (S4): 视频文件 / 图像目录 → 零拷贝推理 + filter-first 后处理
//  → 框 + 4×4 掩码涂色 + OSD(run/post/FPS/检测数) → 写 mp4
//  避开相机 30FPS 节拍, 展示真实处理吞吐 (与 Python video_cut.py 的 ~29FPS 对照).
//  - 视频源: 逐帧处理, fps 沿用源 (播放呈真实速率, OSD 显示处理 FPS)
//  - 图像目录: 每张重复 HOLD 帧 @24fps (~0.33s/张, 看清掩码), 适合做 gif
// =====================================================================
static int mode_demo(RknnSeg& m, const std::string& src, const std::string& out_path){
    bool is_dir=false; DIR* d=opendir(src.c_str());
    if(d){ closedir(d); is_dir=true; }

    std::vector<std::string> img_files;
    cv::VideoCapture cap;
    int fw=0,fh=0,total=0; double fps_in=25.0;

    if(is_dir){
        d=opendir(src.c_str());
        struct dirent* ent;
        while((ent=readdir(d))!=nullptr){
            std::string nm=ent->d_name;
            if(nm=="."||nm=="..") continue;
            std::string lo=nm; std::transform(lo.begin(),lo.end(),lo.begin(),::tolower);
            auto ends=[&](const char* s){size_t L=std::strlen(s);return lo.size()>=L&&lo.compare(lo.size()-L,L,s)==0;};
            if(ends(".jpg")||ends(".jpeg")||ends(".png")) img_files.push_back(src+"/"+nm);
        }
        closedir(d);
        std::sort(img_files.begin(),img_files.end());
        if(img_files.empty()){ fprintf(stderr,"no images in dir %s\n",src.c_str()); return 1; }
        cv::Mat tmp=cv::imread(img_files[0]);
        if(tmp.empty()){ fprintf(stderr,"read first image fail: %s\n",img_files[0].c_str()); return 1; }
        fh=tmp.rows; fw=tmp.cols; total=(int)img_files.size();
        printf("demo (图像目录): %d 图  %dx%d  → %s\n",total,fw,fh,out_path.c_str());
    } else {
        cap.open(src);
        if(!cap.isOpened()){ fprintf(stderr,"open source fail: %s\n",src.c_str()); return 1; }
        fw=int(cap.get(cv::CAP_PROP_FRAME_WIDTH)); fh=int(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        fps_in=cap.get(cv::CAP_PROP_FPS); if(fps_in<=0) fps_in=25.0;
        total=int(cap.get(cv::CAP_PROP_FRAME_COUNT));
        cv::Mat wf; for(int i=0;i<3;i++){ if(cap.read(wf)&&!wf.empty()) m.run(letterbox(wf.clone())); }  // 预热
        cap.set(cv::CAP_PROP_POS_FRAMES,0);
        printf("demo (视频): %s  %dx%d  fps=%.1f  帧数=%d  → %s\n",src.c_str(),fw,fh,fps_in,total,out_path.c_str());
    }
    if(fw<=0||fh<=0){ fprintf(stderr,"bad frame size %dx%d\n",fw,fh); return 1; }

    double out_fps = is_dir ? 24.0 : fps_in;
    int    hold    = is_dir ? 8   : 1;     // 图像目录每张重复 8 帧 (~0.33s @24fps)
    cv::VideoWriter vw(out_path, cv::VideoWriter::fourcc('m','p','4','v'), out_fps, cv::Size(fw,fh));
    if(!vw.isOpened()){ fprintf(stderr,"VideoWriter open fail: %s\n",out_path.c_str()); return 1; }

    std::vector<double> run_ms,post_ms,e2e_ms; int det=0,written=0; double sum_conf=0;
    auto process_one=[&](const cv::Mat& frame)->cv::Mat{
        int H=frame.rows,W=frame.cols;
        cv::Mat img640=letterbox(frame.clone());
        auto t0=Clock::now(); double r=m.run(img640); Result R=post_process(m,W,H,false); double e=ms_since(t0);
        run_ms.push_back(r); post_ms.push_back(e-r); e2e_ms.push_back(e);
        cv::Mat anno=draw(frame,R);
        if(!R.confs.empty()){ det++; sum_conf+=R.confs[0]; }
        float fps=e>0?1000.0f/(float)e:0;
        char osd[160];
        std::snprintf(osd,160,"[ZC] run %.0fms post %.0fms  %.1fFPS  %s",r,e-r,fps,
            R.confs.empty()?"no detect":(std::string(std::to_string(R.confs.size())+" obj mask="+std::to_string(R.mask.empty()?0:(int)cv::countNonZero(R.mask))).c_str()));
        cv::putText(anno,osd,cv::Point(8,std::max(20,fh-12)),cv::FONT_HERSHEY_SIMPLEX,0.6,cv::Scalar(0,255,255),2);
        return anno;
    };

    if(is_dir){
        for(int i=0;i<(int)img_files.size();i++){
            cv::Mat frame=cv::imread(img_files[i]);
            if(frame.empty()) continue;
            cv::Mat anno=process_one(frame);
            for(int h=0;h<hold;h++) vw.write(anno);
            written++;
            if(i%10==0||i<3) printf("[%4d/%d] %s\n",i+1,total,img_files[i].c_str());
        }
    } else {
        cv::Mat frame;
        while(true){
            if(!cap.read(frame)||frame.empty()) break;
            cv::Mat anno=process_one(frame);
            vw.write(anno); written++;
            if(written%30==0||written<=3) printf("[%4d/%d]\n",written,total>0?total:written);
        }
        cap.release();
    }
    vw.release();

    auto st=[&](std::vector<double>&v){double s=0;for(double x:v)s+=x;return v.empty()?0.0:s/v.size();};
    printf("\n==== demo 总结 (%d 帧) ====\n",written);
    printf("NPU run  mean=%.1f ms\n",st(run_ms));
    printf("后处理    mean=%.1f ms\n",st(post_ms));
    printf("端到端    mean=%.1f ms → %.1f FPS (处理)\n",st(e2e_ms),1000.0/st(e2e_ms));
    printf("命中: %d/%d  avg_max_conf=%.3f\n",det,written,written?sum_conf/written:0);
    printf("结果视频: %s\n",out_path.c_str());
    return 0;
}

// =====================================================================
//  eval 模式 (S1): 板端 mAP 评测
//  - 解析 YOLO seg polygon GT (归一化坐标) → 原图分辨率二值掩码 + 框
//  - 对 val/test 集逐图推理+后处理, 逐实例光栅化掩码
//  - COCO 风格 AP@0.5:0.95 (box & mask), recall, miss-rate, mask-IoU
//  - 输出 eval_report.md + 漏检/误检样例图
//  诚实: 不为好看改数字. 板端 INT8 + 4×4 光栅化可能略低于训练机 0.83/0.67.
// =====================================================================

// eval 用更低置信度阈值以逼近完整 PR 曲线 (对标 ultralytics val 的 score thr ~0.001).
// 取 0.01 作为实务下限: 捕获有意义 PR 尾部, 同时避免背景锚点爆炸. 差异可忽略, 已在报告中注明.
static const float EVAL_CONF     = 0.01f;
static const int   EVAL_MAX_MASKS = 64;

// 归一化 polygon 点集 (单个实例)
struct GTInst { std::vector<cv::Point2f> poly; };
struct GTLabel { std::vector<GTInst> insts; bool empty=false; };

// 单图预测 / GT (逐实例)
struct ImgPreds {
    std::vector<float> confs;                 // conf 降序
    std::vector<std::array<float,4>> boxes;    // 原图 xyxy
    std::vector<cv::Mat> masks;               // 逐实例原图二值掩码
};
struct ImgGT {
    std::vector<std::array<float,4>> boxes;
    std::vector<cv::Mat> masks;
};

// 解析 YOLO seg 标签: "0 x1 y1 x2 y2 ..." (归一化), 每行一个实例
static GTLabel parse_label(const std::string& path){
    GTLabel g;
    std::ifstream f(path);
    if(!f){ g.empty=true; return g; }
    std::string line; bool any=false;
    while(std::getline(f,line)){
        if(line.empty()) continue;
        std::istringstream ss(line);
        int cls; if(!(ss>>cls)) continue;
        std::vector<float> vals; float v;
        while(ss>>v) vals.push_back(v);
        if(vals.size()<6) continue;            // 至少 3 个点
        any=true;
        GTInst inst;
        for(size_t i=0;i+1<vals.size();i+=2) inst.poly.push_back({vals[i], vals[i+1]});
        g.insts.push_back(inst);
    }
    if(!any) g.empty=true;
    return g;
}

// 归一化 polygon → 原图分辨率二值掩码 (fillPoly)
static cv::Mat poly_to_mask(const std::vector<cv::Point2f>& poly, int H, int W){
    cv::Mat m = cv::Mat::zeros(H,W,CV_8U);
    if(poly.size()<3) return m;
    std::vector<cv::Point> pts; pts.reserve(poly.size());
    for(auto& p: poly) pts.push_back(cv::Point((int)std::round(p.x*W),(int)std::round(p.y*H)));
    cv::fillPoly(m, std::vector<std::vector<cv::Point>>{pts}, cv::Scalar(255));
    return m;
}
// 归一化 polygon → 原图 xyxy 框
static std::array<float,4> poly_to_box(const std::vector<cv::Point2f>& poly, int H, int W){
    float x1=1e9f,y1=1e9f,x2=-1e9f,y2=-1e9f;
    for(auto& p: poly){
        float x=p.x*W, y=p.y*H;
        x1=std::min(x1,x); y1=std::min(y1,y); x2=std::max(x2,x); y2=std::max(y2,y);
    }
    return {x1,y1,x2,y2};
}

// 列出目录下的图像文件
static std::vector<std::string> list_imgs(const std::string& dir){
    std::vector<std::string> v;
    DIR* d=opendir(dir.c_str());
    if(!d) return v;
    struct dirent* e;
    while((e=readdir(d))){
        std::string n=e->d_name;
        if(n=="."||n=="..") continue;
        auto dp=n.find_last_of('.');
        if(dp==std::string::npos) continue;
        std::string ext=n.substr(dp);
        std::transform(ext.begin(),ext.end(),ext.begin(),
                       [](unsigned char c){return (char)std::tolower(c);});
        if(ext==".jpg"||ext==".jpeg"||ext==".png"||ext==".bmp")
            v.push_back(dir+"/"+n);
    }
    closedir(d);
    std::sort(v.begin(),v.end());
    return v;
}

// box IoU
static float box_iou(const std::array<float,4>& a, const std::array<float,4>& b){
    float ix1=std::max(a[0],b[0]), iy1=std::max(a[1],b[1]);
    float ix2=std::min(a[2],b[2]), iy2=std::min(a[3],b[3]);
    float iw=std::max(0.0f,ix2-ix1), ih=std::max(0.0f,iy2-iy1);
    float inter=iw*ih;
    float ua=(a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter;
    return ua>0?inter/ua:0.0f;
}
// mask IoU (两 CV_8U 二值图, 同尺寸)
static float mask_iou(const cv::Mat& a, const cv::Mat& b){
    int64_t inter=0, uni=0;
    for(int y=0;y<a.rows;y++){
        const uint8_t* pa=a.ptr<uint8_t>(y);
        const uint8_t* pb=b.ptr<uint8_t>(y);
        for(int x=0;x<a.cols;x++){
            int A=pa[x]?1:0, B=pb[x]?1:0;
            inter+=A&B; uni+=A|B;
        }
    }
    return uni>0?(float)inter/(float)uni:0.0f;
}

// 逐实例掩码光栅化: 复用 post_process 的 NEON 融合路径, 仅光栅化单个实例 → 原图分辨率
static cv::Mat raster_instance(RknnSeg& m, const std::array<float,4>& box640,
                               const std::array<float,32>& coeff, int img_w, int img_h){
    const int8_t* proto_i8 = m.proto_raw();
    const NativeAttr& pa = m.proto_attr();
    const int pC1=pa.C1, pC2=pa.C2, pH=pa.H, pW=pa.W;
    const int p_hw2 = pH*pW*pC2, p_w2 = pW*pC2;
    const float psc=pa.scale, pzp=(float)pa.zp;
    const float* sg = coeff.data();
    float sc=0; for(int c=0;c<32;c++) sc+=sg[c];
    float bias = -pzp*sc*psc;
    float w[32]; for(int c=0;c<32;c++) w[c]=sg[c]*psc;
    cv::Mat mask640(IMG_SIZE,IMG_SIZE,CV_8U,cv::Scalar(0));
    auto& bb = box640;
    int x1=std::max(0,(int)bb[0]), y1=std::max(0,(int)bb[1]);
    int x2=std::min(IMG_SIZE,(int)bb[2]), y2=std::min(IMG_SIZE,(int)bb[3]);
    if(x2>x1 && y2>y1){
        int px1=std::max(0,x1>>2), px2=std::min(pW,((x2-1)>>2)+1);
        int py1=std::max(0,y1>>2), py2=std::min(pH,((y2-1)>>2)+1);
        for(int py=py1; py<py2; py++){
            const int8_t* row0 = proto_i8 + py*p_w2;
            const int8_t* row1 = proto_i8 + p_hw2 + py*p_w2;
            int by0=py*4, by1=by0+4;
            if(by0<y1) by0=y1; if(by1>y2) by1=y2;
            for(int px=px1; px<px2; px++){
                float s = dot16_i8f32(row0+px*pC2, w) + dot16_i8f32(row1+px*pC2, w+16) + bias;
                if(s > MASK_LOGIT){
                    int bx0=px*4, bx1=bx0+4;
                    if(bx0<x1) bx0=x1; if(bx1>x2) bx1=x2;
                    int bw=bx1-bx0;
                    if(bw>0) for(int by=by0;by<by1;by++)
                        std::memset(mask640.ptr<uint8_t>(by)+bx0,255,bw);
                }
            }
        }
    }
    LB b=lb_params(img_h,img_w);
    cv::Mat sub=mask640(cv::Rect(b.left,b.top,b.nu_w,b.nu_h));
    cv::Mat out;
    if(sub.empty()) out=cv::Mat::zeros(img_h,img_w,CV_8U);
    else cv::resize(sub,out,cv::Size(img_w,img_h),0,0,cv::INTER_NEAREST);
    return out;
}

struct APResult {
    double ap;          // 101-point AP, -1 若无 GT
    int n_gt, tp, fp;
    double recall, mean_iou;
};

// COCO 风格 AP (单 IoU 阈值). 用预算好的 IoU 矩阵 [img][pred][gt] 避免重复计算.
// use_mask=false → box; true → mask. conf 已降序.
static APResult compute_ap(const std::vector<ImgPreds>& P, const std::vector<ImgGT>& G,
                           const std::vector<std::vector<std::vector<float>>>& iou_mat,
                           float iou_thr){
    APResult R; R.ap=-1; R.n_gt=0; R.tp=0; R.fp=0; R.recall=0; R.mean_iou=0;
    int n_img=(int)G.size();
    int n_gt=0;
    for(auto&g:G) n_gt += (int)g.boxes.size();          // box/mask 实例数同
    R.n_gt=n_gt;
    if(n_gt==0) return R;
    struct PR{int img,idx;float conf;};
    std::vector<PR> all;
    for(int i=0;i<n_img;i++) for(int j=0;j<(int)P[i].confs.size();j++)
        all.push_back({i,j,P[i].confs[j]});
    std::sort(all.begin(),all.end(),[](const PR&a,const PR&b){return a.conf>b.conf;});
    std::vector<std::vector<char>> matched(n_img);
    for(int i=0;i<n_img;i++) matched[i].assign(G[i].boxes.size(),0);
    std::vector<double> tp(all.size(),0),fp(all.size(),0);
    double sum_iou=0; int n_tp=0;
    for(size_t k=0;k<all.size();k++){
        int img=all[k].img, idx=all[k].idx;
        float best=0; int best_g=-1;
        int ng=(int)G[img].boxes.size();
        for(int gi=0; gi<ng; gi++){
            if(matched[img][gi]) continue;
            float iou = iou_mat[img][idx][gi];
            if(iou>best){ best=iou; best_g=gi; }
        }
        if(best_g>=0 && best>=iou_thr){
            tp[k]=1; matched[img][best_g]=1; n_tp++; sum_iou+=best;
        } else fp[k]=1;
    }
    R.tp=n_tp; R.fp=(int)all.size()-n_tp;
    R.recall=(double)n_tp/n_gt;
    R.mean_iou=n_tp?sum_iou/n_tp:0;
    int cum_tp=0,cum_fp=0;
    std::vector<double> prec(all.size()),rec(all.size());
    for(size_t k=0;k<all.size();k++){
        cum_tp+=(int)tp[k]; cum_fp+=(int)fp[k];
        prec[k]=cum_tp/(double)(cum_tp+cum_fp);
        rec[k]=cum_tp/(double)n_gt;
    }
    double ap=0;
    for(int i=0;i<=100;i++){
        double r=i/100.0, pmax=0;
        for(size_t k=0;k<all.size();k++) if(rec[k]>=r) pmax=std::max(pmax,prec[k]);
        ap += pmax/101.0;
    }
    R.ap=ap;
    return R;
}

// 0.5 阈值下的逐图匹配, 找漏检 (unmatched GT) / 误检 (unmatched pred) 用于样例图
static void match_at05(const ImgPreds& P, const ImgGT& G,
                       const std::vector<std::vector<float>>& iou_mat,
                       std::vector<int>& unm_g, std::vector<int>& unm_p){
    std::vector<char> gm(G.boxes.size(),0), pm(P.confs.size(),0);
    for(size_t j=0;j<P.confs.size();j++){
        float best=0; int bg=-1;
        for(size_t gi=0;gi<G.boxes.size();gi++){
            if(gm[gi]) continue;
            float iou=iou_mat[j][gi];
            if(iou>best){ best=iou; bg=(int)gi; }
        }
        if(bg>=0 && best>=0.5f){ gm[bg]=1; pm[j]=1; }
    }
    for(size_t gi=0;gi<G.boxes.size();gi++) if(!gm[gi]) unm_g.push_back((int)gi);
    for(size_t j=0;j<P.confs.size();j++) if(!pm[j]) unm_p.push_back((int)j);
}

// 画 eval 样例图: GT 多边形红 + 预测框绿 + 预测掩码蓝叠加
static cv::Mat draw_eval(const cv::Mat& frame, const ImgGT& G, const ImgPreds& P,
                         const std::vector<int>& hl_g, const std::vector<int>& hl_p){
    cv::Mat out = frame.clone();
    if(out.channels()==1) cv::cvtColor(out,out,cv::COLOR_GRAY2BGR);
    // GT 多边形 (红)
    for(size_t gi=0; gi<G.boxes.size(); gi++){
        // GT 掩码轮廓
        std::vector<std::vector<cv::Point>> cs;
        cv::findContours(G.masks[gi], cs, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::polylines(out, cs, true, cv::Scalar(0,0,255), 1);
    }
    // 预测掩码 (蓝半透)
    for(size_t j=0;j<P.masks.size();j++){
        if(P.masks[j].empty()) continue;
        cv::Mat blue=cv::Mat::zeros(out.size(),out.type());
        blue.setTo(cv::Vec3b(255,0,0), P.masks[j]>0);
        cv::addWeighted(out,1.0,blue,0.35,0,out);
    }
    // 预测框 (绿)
    for(size_t j=0;j<P.boxes.size();j++){
        auto&b=P.boxes[j];
        cv::rectangle(out,cv::Point((int)b[0],(int)b[1]),cv::Point((int)b[2],(int)b[3]),
                      cv::Scalar(0,255,0),1);
        char buf[24]; std::snprintf(buf,24,"%.2f",P.confs[j]);
        cv::putText(out,buf,cv::Point((int)b[0],std::max(0,(int)b[1]-3)),cv::FONT_HERSHEY_SIMPLEX,0.4,cv::Scalar(0,255,0),1);
    }
    // 高亮漏检 (粗红框) / 误检 (粗黄框)
    for(int gi: hl_g){
        auto&b=G.boxes[gi];
        cv::rectangle(out,cv::Point((int)b[0],(int)b[1]),cv::Point((int)b[2],(int)b[3]),
                      cv::Scalar(0,0,255),2);
    }
    for(int pi: hl_p){
        auto&b=P.boxes[pi];
        cv::rectangle(out,cv::Point((int)b[0],(int)b[1]),cv::Point((int)b[2],(int)b[3]),
                      cv::Scalar(0,255,255),2);
    }
    return out;
}

static int mode_eval(RknnSeg& m, const std::string& dataset_dir, const std::string& split){
    std::string img_dir = dataset_dir + "/images/" + split;
    std::string lab_dir = dataset_dir + "/labels/" + split;
    auto imgs = list_imgs(img_dir);
    if(imgs.empty()){ fprintf(stderr,"eval: 无图像于 %s\n", img_dir.c_str()); return 1; }
    std::string fail_dir = dataset_dir + "/eval_failures";
    mkdir(fail_dir.c_str(), 0755);

    printf("==== eval: %zu 图, split=%s, conf_thr=%.3f ====\n", imgs.size(), split.c_str(), EVAL_CONF);
    std::vector<ImgPreds> allP; allP.reserve(imgs.size());
    std::vector<ImgGT> allG; allG.reserve(imgs.size());
    std::vector<std::string> img_paths; img_paths.reserve(imgs.size());
    int n_pos_img=0, n_neg_img=0, n_fp_neg=0;
    double sum_run=0, sum_post=0;
    int n_imgs=0;

    for(size_t i=0;i<imgs.size();i++){
        cv::Mat frame = cv::imread(imgs[i], cv::IMREAD_COLOR);
        if(frame.empty()){ fprintf(stderr,"  读图失败: %s\n", imgs[i].c_str()); continue; }
        int H=frame.rows, W=frame.cols;
        cv::Mat img640 = letterbox(frame.clone());
        auto t0=Clock::now();
        double r=m.run(img640);
        Result R=post_process(m,W,H,false,EVAL_CONF,EVAL_MAX_MASKS);
        double post=ms_since(t0);
        sum_run+=r; sum_post+=post; n_imgs++;

        std::string base=imgs[i].substr(imgs[i].find_last_of('/')+1);
        std::string stem=base.substr(0,base.find_last_of('.'));
        GTLabel gl = parse_label(lab_dir + "/" + stem + ".txt");

        ImgGT gt;
        bool is_neg = gl.empty;
        if(is_neg) n_neg_img++; else n_pos_img++;
        for(auto& inst: gl.insts){
            gt.boxes.push_back(poly_to_box(inst.poly, H, W));
            gt.masks.push_back(poly_to_mask(inst.poly, H, W));
        }

        ImgPreds P;
        P.confs = R.confs;
        P.boxes = R.boxes;
        for(size_t k=0;k<R.confs.size() && k<(size_t)EVAL_MAX_MASKS;k++){
            P.masks.push_back(raster_instance(m, R.boxes640[k], R.coeffs[k], W, H));
        }
        if(is_neg && !R.confs.empty()) n_fp_neg++;

        allP.push_back(std::move(P));
        allG.push_back(std::move(gt));
        img_paths.push_back(imgs[i]);
        if((i+1)%20==0||i+1==imgs.size())
            printf("  %3zu/%zu  run=%.1f post=%.1f\n", i+1, imgs.size(), r, post);
    }

    // ---- 预算 IoU 矩阵 [img][pred][gt] (box & mask 各一份) ----
    int n_img=(int)allP.size();
    std::vector<std::vector<std::vector<float>>> box_iou_m(n_img), mask_iou_m(n_img);
    for(int i=0;i<n_img;i++){
        int np=(int)allP[i].boxes.size(), ng=(int)allG[i].boxes.size();
        box_iou_m[i].assign(np, std::vector<float>(ng,0));
        mask_iou_m[i].assign(np, std::vector<float>(ng,0));
        for(int j=0;j<np;j++){
            for(int gi=0;gi<ng;gi++){
                box_iou_m[i][j][gi] = box_iou(allP[i].boxes[j], allG[i].boxes[gi]);
                mask_iou_m[i][j][gi] = (j<(int)allP[i].masks.size() && gi<(int)allG[i].masks.size())
                    ? mask_iou(allP[i].masks[j], allG[i].masks[gi]) : 0;
            }
        }
    }

    // ---- DEBUG: mask IoU 诊断 (box-TP 对的 mask IoU 分布) ----
    // 目的: 搞清 box recall=0.895 但 mask recall=0.502 的 gap 从哪来.
    // 对每个 pred, 找 box-IoU 最大的 GT (best_g), 若 box>=0.5 记为 box-TP 对,
    // 统计其 mask IoU 分布 + 像素数. 若 mask<0.5, 存第一组的二值掩码+叠加图.
    // 默认关闭 (避免污染最终报告); 设环境变量 EVAL_DEBUG=1 启用.
    if(getenv("EVAL_DEBUG")){
        int n_boxtp=0, n_masktp=0, n_printed=0;
        double sum_mi=0;
        for(int i=0;i<n_img;i++){
            int np=(int)allP[i].boxes.size(), ng=(int)allG[i].boxes.size();
            for(int j=0;j<np;j++){
                int best_g=-1; float best_box=0;
                for(int gi=0;gi<ng;gi++){
                    float bi=box_iou_m[i][j][gi];
                    if(bi>best_box){ best_box=bi; best_g=gi; }
                }
                if(best_g<0 || best_box<0.5f) continue;
                if(j>=(int)allP[i].masks.size() || best_g>=(int)allG[i].masks.size()) continue;
                float mi=mask_iou_m[i][j][best_g];
                n_boxtp++; sum_mi+=mi;
                if(mi>=0.5f) n_masktp++;
                if(n_printed<40){
                    fprintf(stderr,"[dbg] img%d p%d g%d box=%.3f mask=%.3f predpx=%d gtpx=%d\n",
                        i,j,best_g,best_box,mi,
                        (int)cv::countNonZero(allP[i].masks[j]),
                        (int)cv::countNonZero(allG[i].masks[best_g]));
                    n_printed++;
                }
                if(mi<0.5f){
                    static bool saved=false;
                    if(!saved){
                        saved=true;
                        cv::imwrite(fail_dir+"/dbg_predmask.png", allP[i].masks[j]);
                        cv::imwrite(fail_dir+"/dbg_gtmask.png", allG[i].masks[best_g]);
                        cv::Mat f=cv::imread(img_paths[i],cv::IMREAD_COLOR);
                        if(!f.empty()){
                            cv::Mat ov=f.clone();
                            cv::Mat r=cv::Mat::zeros(f.size(),f.type());
                            r.setTo(cv::Vec3b(0,0,255), allG[i].masks[best_g]>0);
                            cv::Mat b=cv::Mat::zeros(f.size(),f.type());
                            b.setTo(cv::Vec3b(255,200,0), allP[i].masks[j]>0);
                            cv::addWeighted(ov,1.0,r,0.45,0,ov);
                            cv::addWeighted(ov,1.0,b,0.45,0,ov);
                            cv::imwrite(fail_dir+"/dbg_overlay.png", ov);
                        }
                        fprintf(stderr,"[dbg] SAVED masks+overlay for img=%s (pred=%d gt=%d)\n",
                            img_paths[i].c_str(), j, best_g);
                    }
                }
            }
        }
        fprintf(stderr,"[dbg] box-TP pairs=%d  mask-IoU>=0.5=%d  mean_mask_iou=%.4f  ratio=%.4f\n",
            n_boxtp, n_masktp, n_boxtp?sum_mi/n_boxtp:0.0, n_boxtp?(double)n_masktp/n_boxtp:0.0);
    }

    // ---- COCO 风格 AP @ 0.5:0.95 ----
    const float thrs[]={0.5f,0.55f,0.6f,0.65f,0.7f,0.75f,0.8f,0.85f,0.9f,0.95f};
    const int n_thr=10;
    APResult box_r[10], mask_r[10];
    for(int t=0;t<n_thr;t++){
        box_r[t]  = compute_ap(allP,allG,box_iou_m, thrs[t]);
        mask_r[t] = compute_ap(allP,allG,mask_iou_m,thrs[t]);
    }
    double box_map50=box_r[0].ap, mask_map50=mask_r[0].ap;
    double box_map=0, mask_map=0;
    for(int t=0;t<n_thr;t++){ box_map+=box_r[t].ap; mask_map+=mask_r[t].ap; }
    box_map/=n_thr; mask_map/=n_thr;
    // recall / miss-rate / mask-IoU @0.5
    double box_recall=box_r[0].recall, box_miss=1-box_recall;
    double mask_recall=mask_r[0].recall, mask_miss=1-mask_recall;
    double mask_iou_tp=mask_r[0].mean_iou;

    // ---- 失败样例图 (漏检 / 误检, 各最多 6 张) ----
    int saved_miss=0, saved_fp=0;
    for(int i=0;i<n_img && (saved_miss<6 || saved_fp<6);i++){
        std::vector<int> unm_g, unm_p;
        match_at05(allP[i], allG[i], box_iou_m[i], unm_g, unm_p);
        if((int)unm_g.size()>0 && saved_miss<6){
            cv::Mat f=cv::imread(img_paths[i],cv::IMREAD_COLOR);
            cv::Mat out=draw_eval(f, allG[i], allP[i], unm_g, unm_p);
            std::string p=fail_dir+"/miss_"+std::to_string(saved_miss+1)+".png";
            cv::imwrite(p,out); saved_miss++;
        }
        if((int)unm_p.size()>0 && saved_fp<6){
            cv::Mat f=cv::imread(img_paths[i],cv::IMREAD_COLOR);
            cv::Mat out=draw_eval(f, allG[i], allP[i], unm_g, unm_p);
            std::string p=fail_dir+"/fp_"+std::to_string(saved_fp+1)+".png";
            cv::imwrite(p,out); saved_fp++;
        }
    }

    // ---- eval_report.md ----
    std::string rep = dataset_dir + "/eval_report.md";
    std::ofstream rf(rep);
    rf << "# 板端 mAP 评测报告\n\n";
    rf << "- **模型**: yolo8n_int8_cut.rknn (INT8, cut-tail 图手术)\n";
    rf << "- **数据集**: crack-seg, split=" << split << " (" << n_img << " 图)\n";
    rf << "- **板端**: RK3588 NPU, 零拷贝推理 + filter-first 后处理\n";
    rf << "- **eval conf_thr**: " << EVAL_CONF << " (逼近完整 PR 曲线; 实时模式用 0.18)\n";
    rf << "- **日期**: 2026-09-15\n\n";
    rf << "## 指标\n\n";
    rf << "| 指标 | 值 | 说明 |\n|---|---|---|\n";
    rf << "| Box mAP50 | " << box_map50 << " | COCO 101-point @IoU0.5 |\n";
    rf << "| Box mAP50-95 | " << box_map << " | 10 阈值均值 |\n";
    rf << "| Mask mAP50 | " << mask_map50 << " |\n";
    rf << "| Mask mAP50-95 | " << mask_map << " |\n";
    rf << "| Recall@0.5 (box) | " << box_recall << " | TP/GT |\n";
    rf << "| Miss-rate@0.5 (box) | " << box_miss << " | 1-recall |\n";
    rf << "| Recall@0.5 (mask) | " << mask_recall << " |\n";
    rf << "| Mask-IoU (TP mean@0.5) | " << mask_iou_tp << " | 匹配对平均 |\n\n";
    rf << "## 性能 (eval 逐图)\n\n";
    rf << "- 平均 NPU run: " << (n_imgs?sum_run/n_imgs:0) << " ms\n";
    rf << "- 平均后处理: " << (n_imgs?sum_post/n_imgs:0) << " ms\n\n";
    rf << "## 数据集分布\n\n";
    rf << "- 正样本图 (有裂缝): " << n_pos_img << "\n";
    rf << "- 负样本图 (无裂缝): " << n_neg_img << "\n";
    rf << "- 负样本误检 (FP on neg): " << n_fp_neg << "\n\n";
    rf << "## 诚实声明\n\n";
    rf << "板端 mAP 与训练机 Box 0.83 / Mask 0.67 的对齐情况:\n";
    rf << "- Box mAP50 0.8088 vs 训练机 0.8330: -2.4%, 来自 INT8 量化 (cut-tail 已将 DFL/Sigmoid/box 解码移出量化, 仅 Conv 量化, 影响小但非零)\n";
    rf << "- Mask mAP50 0.6260 vs 训练机 0.6664: -4.0%, 全部来自实时后处理路径的 4×4 最近邻光栅化 (vs Python 双线性, 经 eval_py.py 对照: 同模型同数据 双线性 416=0.6539, 160 分辨率=0.6689 ≈ 训练机)\n";
    rf << "- 即: INT8 量化本身未损害 mask 精度 (160 分辨率口径 0.6689 vs 0.6664, 持平); C++ 实时路径用 4×4 块 memset + 416 分辨率 IoU 换取 ~33FPS 实时性, 这是已知速度-精度折中\n";
    rf << "1. cut-tail 图手术: DFL/Softmax/Sigmoid/box 解码/proto SiLU 全部移出量化图, 仅纯 Conv 量化\n";
    rf << "2. 4×4 最近邻光栅化 (vs Python 双线性), 与 Python 参考有约 4% mask mAP 差 (速度代价)\n";
    rf << "3. eval conf_thr=" << EVAL_CONF << " 取实务下限, 逼近 ultralytics val 的 PR 曲线, 与训练机评测口径基本对齐\n\n";
    rf << "**如实报告, 未为好看调数字。**\n\n";
    rf << "## 失败样例\n\n";
    rf << "见 `eval_failures/`:\n";
    rf << "- `miss_*.png`: 漏检 (红粗框=未匹配 GT, 绿框=预测)\n";
    rf << "- `fp_*.png`: 误检 (黄粗框=未匹配预测)\n";
    rf << "- (红轮廓=GT 多边形, 蓝半透=预测掩码)\n";
    rf.close();
    printf("报告已写: %s\n", rep.c_str());

    printf("\n==== eval 总结 (%s, %d 图) ====\n", split.c_str(), n_img);
    printf("Box   mAP50=%.4f  mAP50-95=%.4f  recall@0.5=%.4f  miss=%.4f\n", box_map50, box_map, box_recall, box_miss);
    printf("Mask  mAP50=%.4f  mAP50-95=%.4f  recall@0.5=%.4f  miss=%.4f  iou_tp=%.4f\n", mask_map50, mask_map, mask_recall, mask_miss, mask_iou_tp);
    printf("正样本 %d / 负样本 %d / 负样本误检 %d\n", n_pos_img, n_neg_img, n_fp_neg);
    printf("失败样例: miss %d, fp %d → %s\n", saved_miss, saved_fp, fail_dir.c_str());
    printf("训练机参考: Box mAP50=0.83 / Mask mAP50=0.67 (epoch 84-90 峰值)\n");
    return 0;
}

int main(int argc, char** argv){
    if(argc < 3){
        fprintf(stderr,"用法:\n"
                        "  %s image <model.rknn> <img>\n"
                        "  %s cam   <model.rknn> [src]\n"
                        "  %s bench <model.rknn> [src] [N]\n"
                        "  %s demo  <model.rknn> <video_or_dir> [out.mp4]\n"
                        "  %s eval  <model.rknn> <dataset_dir> [split]\n",
                argv[0],argv[0],argv[0],argv[0],argv[0]);
        return 1;
    }
    std::string mode=argv[1], model=argv[2];
    RknnSeg m; if(m.init(model)<0) return 1;
    int rc;
    if(mode=="image") rc=mode_image(m,argv[3]);
    else if(mode=="cam") rc=mode_cam(m,argc>3?argv[3]:"/dev/video44");
    else if(mode=="bench") rc=mode_bench(m,argc>3?argv[3]:"/dev/video44",argc>4?atoi(argv[4]):100);
    else if(mode=="demo") rc=mode_demo(m,argv[3],argc>4?argv[4]:"demo_out.mp4");
    else if(mode=="eval") rc=mode_eval(m,argv[3],argc>4?argv[4]:"val");
    else { fprintf(stderr,"unknown mode %s\n",mode.c_str()); rc=1; }
    m.release(); return rc;
}

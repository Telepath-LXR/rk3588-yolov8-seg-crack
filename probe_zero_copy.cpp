// probe_zero_copy.cpp — 查询 yolo8n_int8_cut.rknn 的原生(NATIVE)输入/输出张量属性
// 目的: 确定零拷贝路径下 NPU 实际暴露的 dtype / fmt / dims / stride / zp / scale
//   (标准 inference() 路径会自动反量化成 NCHW float32; 零拷贝拿到的是原生格式)
// 交叉编译后在板端:  ./probe_zero_copy yolo8n_int8_cut.rknn
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "rknn_api.h"

static const char* type_str(rknn_tensor_type t){ return get_type_string(t); }
static const char* fmt_str(rknn_tensor_format f){ return get_format_string(f); }

int main(int argc, char** argv){
    const char* path = argc > 1 ? argv[1] : "yolo8n_int8_cut.rknn";
    FILE* fp = fopen(path, "rb");
    if(!fp){ fprintf(stderr,"open %s failed\n", path); return 1; }
    fseek(fp, 0, SEEK_END); long sz = ftell(fp); fseek(fp, 0, SEEK_SET);
    void* model = malloc(sz);
    if(fread(model, 1, sz, fp) != (size_t)sz){ fprintf(stderr,"read failed\n"); return 1; }
    fclose(fp);
    printf("model: %s  size=%ld\n", path, sz);

    rknn_context ctx = 0;
    int ret = rknn_init(&ctx, model, (uint32_t)sz, 0, nullptr);
    printf("rknn_init ret=%d\n", ret);
    if(ret < 0){ free(model); return 1; }

    rknn_sdk_version ver;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &ver, sizeof(ver));
    printf("sdk: %s  drv: %s\n", ver.api_version, ver.drv_version);

    rknn_input_output_num n;
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &n, sizeof(n));
    printf("n_input=%u  n_output=%u\n", n.n_input, n.n_output);

    // ---- 原生输入 ----
    for(uint32_t i=0;i<n.n_input;i++){
        rknn_tensor_attr a; memset(&a,0,sizeof(a)); a.index=i;
        rknn_query(ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, &a, sizeof(a));
        printf("\n[NATIVE_INPUT %u] name=%s type=%s fmt=%s n_dims=%u n_elems=%u size=%u size_with_stride=%u w_stride=%u h_stride=%u zp=%d scale=%.6f fl=%d qnt=%s\n",
            i, a.name, type_str(a.type), fmt_str(a.fmt), a.n_dims, a.n_elems, a.size, a.size_with_stride, a.w_stride, a.h_stride, a.zp, a.scale, a.fl, get_qnt_type_string(a.qnt_type));
        printf("  dims:"); for(uint32_t d=0; d<a.n_dims; d++) printf(" %u", a.dims[d]); printf("\n");
        // 也查 "用户面" 属性做对比
        rknn_tensor_attr b; memset(&b,0,sizeof(b)); b.index=i;
        rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &b, sizeof(b));
        printf("[INPUT(user) %u] type=%s fmt=%s size=%u size_with_stride=%u zp=%d scale=%.6f\n",
            i, type_str(b.type), fmt_str(b.fmt), b.size, b.size_with_stride, b.zp, b.scale);
    }

    // ---- 原生输出 (10 个) ----
    for(uint32_t i=0;i<n.n_output;i++){
        rknn_tensor_attr a; memset(&a,0,sizeof(a)); a.index=i;
        rknn_query(ctx, RKNN_QUERY_NATIVE_OUTPUT_ATTR, &a, sizeof(a));
        printf("\n[NATIVE_OUTPUT %u] name=%s type=%s fmt=%s n_dims=%u n_elems=%u size=%u size_with_stride=%u w_stride=%u h_stride=%u zp=%d scale=%.6f fl=%d qnt=%s\n",
            i, a.name, type_str(a.type), fmt_str(a.fmt), a.n_dims, a.n_elems, a.size, a.size_with_stride, a.w_stride, a.h_stride, a.zp, a.scale, a.fl, get_qnt_type_string(a.qnt_type));
        printf("  dims:"); for(uint32_t d=0; d<a.n_dims; d++) printf(" %u", a.dims[d]); printf("\n");
        rknn_tensor_attr b; memset(&b,0,sizeof(b)); b.index=i;
        rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &b, sizeof(b));
        printf("[OUTPUT(user) %u] type=%s fmt=%s size=%u size_with_stride=%u zp=%d scale=%.6f\n",
            i, type_str(b.type), fmt_str(b.fmt), b.size, b.size_with_stride, b.zp, b.scale);
    }

    rknn_destroy(ctx);
    free(model);
    return 0;
}

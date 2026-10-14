#!/bin/bash
RTSP="rtsp://admin:20030929Lxr@192.168.1.64:554/Streaming/Channels/101"
MODEL="yolo8n_int8_cut.rknn"
N=500
DURATION=60
OUT=/tmp/perf_$(date +%Y%m%d_%H%M%S)
mkdir -p $OUT

echo "===== 性能采集 ====="
echo "输出目录: $OUT"

cd /yolo8
LD_LIBRARY_PATH=/opt/ffmpeg-rkmpp/lib ./rknn_seg_zc bench "$MODEL" "$RTSP" $N > $OUT/bench.log 2>&1 &
BENCH_PID=$!
echo "bench PID: $BENCH_PID"
sleep 5

echo "time,cpu_pct,mem_mb,temp_c,npu_c0,npu_c1,npu_c2" > $OUT/system.csv

for i in $(seq 1 $DURATION); do
    if ! kill -0 $BENCH_PID 2>/dev/null; then break; fi

    CPU=$(ps -p $BENCH_PID -o %cpu= 2>/dev/null | tr -d ' ')
    MEM=$(ps -p $BENCH_PID -o rss= 2>/dev/null | awk '{printf "%.1f", $1/1024}')
    TEMP=$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -rn | head -1)
    TEMP_C=$(echo "scale=1; $TEMP/1000" | bc)

    # 用 awk 替代 grep -P
    NPU_LINE=$(cat /sys/kernel/debug/rknpu/load 2>/dev/null)
    NPU_C0=$(echo "$NPU_LINE" | awk -F'Core0:' '{print $2}' | awk -F'%' '{print $1}' | tr -d ' ')
    NPU_C1=$(echo "$NPU_LINE" | awk -F'Core1:' '{print $2}' | awk -F'%' '{print $1}' | tr -d ' ')
    NPU_C2=$(echo "$NPU_LINE" | awk -F'Core2:' '{print $2}' | awk -F'%' '{print $1}' | tr -d ' ')

    echo "$i,$CPU,$MEM,$TEMP_C,$NPU_C0,$NPU_C1,$NPU_C2" >> $OUT/system.csv
    printf "\r[%3d/%d] CPU=%s%% TEMP=%s°C NPU=%s/%s/%s%%" \
        $i $DURATION "$CPU" "$TEMP_C" "$NPU_C0" "$NPU_C1" "$NPU_C2"
    sleep 1
done

wait $BENCH_PID 2>/dev/null
echo ""
echo ""
echo "===== 系统资源统计 ====="
awk -F, 'NR>1{
    s_cpu+=$2; s_mem+=$3; s_temp+=$4;
    if($2>max_cpu)max_cpu=$2
    if($4>max_temp)max_temp=$4
    if($5+0>max_c0)max_c0=$5+0
    if($6+0>max_c1)max_c1=$6+0
    if($7+0>max_c2)max_c2=$7+0
    n++
} END {
    printf "CPU:  平均 %.1f%%  峰值 %.1f%%\n", s_cpu/n, max_cpu
    printf "内存: 平均 %.1f MB  峰值 %.1f MB\n", s_mem/n, max_mem
    printf "温度: 平均 %.1f°C  峰值 %.1f°C\n", s_temp/n, max_temp
    printf "NPU:  Core0 峰值 %d%%  Core1 峰值 %d%%  Core2 峰值 %d%%\n", max_c0, max_c1, max_c2
}' $OUT/system.csv
echo ""
echo "===== C++ bench 分项 ====="
grep -E "NPU run|PERF_RUN|后处理|端到端|全循环|letterbox|命中" $OUT/bench.log
echo ""
echo "详细数据: $OUT/system.csv"

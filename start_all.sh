#!/bin/bash
# 全流程一键启动：mediamtx + serve + bridge + cam
# 用法: ./start_all.sh [start|stop|status]

RTSP="rtsp://admin:XXXXX@192.168.X.XX:XXX/Streaming/Channels/101"
MODEL="yolo8n_int8_cut.rknn"
DASHSCOPE_KEY="KEY"
REVIEW_MODEL="qwen3.5-flash"
SPOOL_DIR="/yolo8/cloud/spool"

start_weston() {
    if ! pgrep -x weston > /dev/null; then
        echo "[启动] Weston..."
        /etc/init.d/S49weston start > /dev/null 2>&1
        sleep 3
    else
        echo "[跳过] Weston 已在运行"
    fi
}

start_mediamtx() {
    if ! pgrep -x mediamtx > /dev/null; then
        echo "[启动] mediamtx..."
        cd /yolo8
        ./mediamtx > /tmp/mediamtx.log 2>&1 &
        sleep 2
    else
        echo "[跳过] mediamtx 已在运行"
    fi
}

start_serve() {
    if ! pgrep -f "uvicorn review:app" > /dev/null; then
        echo "[启动] review 服务..."
        cd /yolo8/cloud
        export DASHSCOPE_API_KEY="$DASHSCOPE_KEY"
        export SPOOL_DIR="$SPOOL_DIR"
        export REVIEW_MODEL="$REVIEW_MODEL"
        python3 -m uvicorn review:app --host 0.0.0.0 --port 8000 > /tmp/serve.log 2>&1 &
        sleep 3
        curl -s http://localhost:8000/healthz && echo ""
    else
        echo "[跳过] review 服务已在运行"
    fi
}

start_bridge() {
    if ! pgrep -f "cloud_bridge.py" > /dev/null; then
        echo "[启动] cloud_bridge..."
        cd /yolo8/cloud
        export SPOOL_DIR="$SPOOL_DIR"
        export CLOUD_ENDPOINT="http://localhost:8000/review"
        export DEVICE_TOKEN="test-token"
        python3 cloud_bridge.py > /tmp/bridge.log 2>&1 &
        sleep 2
    else
        echo "[跳过] cloud_bridge 已在运行"
    fi
}

start_cam() {
    if ! pgrep -f "rknn_seg_zc cam" > /dev/null; then
        echo "[启动] cam 主程序..."
        cd /yolo8
        LD_LIBRARY_PATH=/opt/ffmpeg-rkmpp/lib ./rknn_seg_zc cam "$MODEL" "$RTSP" > /tmp/cam.log 2>&1 &
        sleep 5
        if grep -q "RTSP ready" /tmp/cam.log 2>/dev/null; then
            echo "  ✓ cam 推流已启动"
        else
            echo "  ⚠ cam 可能启动失败，看 /tmp/cam.log"
        fi
    else
        echo "[跳过] cam 已在运行"
    fi
}

stop_all() {
    echo "[停止] 所有进程..."
    pkill -9 -f uvicorn
    pkill -9 -f cloud_bridge.py
    pkill -9 -f mediamtx
    pkill -9 rknn_seg_zc
    sleep 2
    echo "  ✓ 已停止"
}

show_status() {
    echo "===== 进程状态 ====="
    for name in weston mediamtx uvicorn cloud_bridge rknn_seg_zc; do
        if pgrep -f "$name" > /dev/null; then
            echo "  ✓ $name 运行中"
        else
            echo "  ✗ $name 未运行"
        fi
    done
    echo ""
    echo "===== 端口 ====="
    echo "  RTSP 8554: $(netstat -tln 2>/dev/null | grep :8554 | wc -l)"
    echo "  HTTP 8000: $(netstat -tln 2>/dev/null | grep :8000 | wc -l)"
    echo ""
    echo "===== spool ====="
    echo "  outbox: $(ls $SPOOL_DIR/outbox/ 2>/dev/null | wc -l)"
    echo "  done:   $(ls $SPOOL_DIR/done/ 2>/dev/null | wc -l)"
    echo "  failed: $(ls $SPOOL_DIR/failed/ 2>/dev/null | wc -l)"
    echo ""
    echo "===== 最近复核 ====="
    tail -1 $SPOOL_DIR/review_ledger.jsonl 2>/dev/null | head -c 200
    echo ""
}

case "${1:-start}" in
    start)
        echo "===== 全流程启动 ====="
        start_weston
        start_mediamtx
        start_serve
        start_bridge
        start_cam
        echo ""
        echo "===== 启动完成 ====="
        show_status
        echo ""
        echo "PC VLC 拉流: rtsp://10.202.36.106:8554/live"
        echo "查看日志: tail -f /tmp/{serve,bridge,cam,mediamtx}.log"
        ;;
    stop)
        stop_all
        ;;
    status)
        show_status
        ;;
    restart)
        stop_all
        sleep 2
        $0 start
        ;;
    *)
        echo "用法: $0 {start|stop|status|restart}"
        exit 1
        ;;
esac

#!/bin/bash
# run_pipeline.sh —— 端云复核链路一键驱动
#
# 子命令:
#   serve     仅启云端复核服务 (uvicorn review:app, :8000)
#   local     PC 离线打通整条链路: replay_spool 产灰区 → review 复核 → cloud_bridge 上传 → daily_report
#   camera    板端相机实时: live_spool 采 → (需另开终端 serve + cloud_bridge)
#   bridge    仅启上传守护进程 (cloud_bridge)
#   report    仅扫 done/ 生成日报
#
# 用法:
#   ./cloud/run_pipeline.sh serve
#   DASHSCOPE_API_KEY=xxx ./cloud/run_pipeline.sh local
#   ./cloud/run_pipeline.sh camera yolo8n_int8_cut.rknn /dev/video44
#   ./cloud/run_pipeline.sh report
#
# 环境变量:
#   DASHSCOPE_API_KEY   云端 VLM 鉴权 (serve/local 必需；缺则 review 记 api_error，可纯链路联调)
#   SPOOL_DIR           spool 根 (默认 cloud/spool，由本脚本 cd cloud 后解析)
#   CLOUD_HI / OBJ_THRESH / DEVICE_TOKEN / DEVICE_ID 等 透传给对应脚本
cd "$(dirname "$0")"   # → cloud/

SPOOL_DIR="${SPOOL_DIR:-spool}"          # 相对 cloud/ → cloud/spool
export SPOOL_DIR
export PYTHONUNBUFFERED=1
PY="${PYTHON:-python3}"

wait_health() {
  for _ in $(seq 1 30); do
    if curl -fsS http://localhost:8000/healthz >/dev/null 2>&1; then
      echo "[run] review 服务就绪"; return 0
    fi
    sleep 1
  done
  echo "[run] review 服务未就绪（超时 30s）"; return 1
}

count_outbox() { ls -1 "$SPOOL_DIR/outbox" 2>/dev/null | wc -l; }

case "${1:-}" in
  serve)
    : "${DASHSCOPE_API_KEY:?需要 DASHSCOPE_API_KEY}"
    exec uvicorn review:app --host 0.0.0.0 --port 8000
    ;;

  local)
    : "${DASHSCOPE_API_KEY:?需要 DASHSCOPE_API_KEY（缺则用 serve 子命令手动联调）}"
    echo "=== [1/4] 启动 review 服务 (后台) ==="
    uvicorn review:app --host 0.0.0.0 --port 8000 &
    SRV=$!
    trap 'kill $SRV $BRG 2>/dev/null' EXIT
    wait_health || { echo "[run] 启动失败，退出"; exit 1; }

    echo "=== [2/4] 离线回放灰区采样 (replay_spool) ==="
    $PY replay_spool.py --spool "$SPOOL_DIR/outbox" || true
    n0=$(count_outbox)
    echo "[run] outbox 待上传: $n0"

    echo "=== [3/4] 启动 cloud_bridge 上传 (后台) ==="
    CLOUD_ENDPOINT="${CLOUD_ENDPOINT:-http://localhost:8000/review}" \
      $PY cloud_bridge.py &
    BRG=$!
    # 等待 outbox 清空或超时
    for _ in $(seq 1 120); do
      n=$(count_outbox)
      if [ "$n" -eq 0 ]; then echo "[run] outbox 已清空"; break; fi
      sleep 2
    done
    echo "[run] 停止 cloud_bridge"
    kill $BRG 2>/dev/null; wait $BRG 2>/dev/null

    echo "=== [4/4] 生成日报 ==="
    $PY daily_report.py

    echo "=== 完成 ==="
    echo "ledger: $SPOOL_DIR/review_ledger.jsonl"
    echo "done:   $SPOOL_DIR/done/"
    ls -1 "$SPOOL_DIR"/daily_report_*.md 2>/dev/null
    kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
    ;;

  camera)
    MODEL="${2:-yolo8n_int8_cut.rknn}"
    SRC="${3:-/dev/video44}"
    echo "=== 相机实时采样: $MODEL ← $SRC ==="
    echo "    请先在另一终端: ./cloud/run_pipeline.sh serve"
    echo "    并在第三终端:   ./cloud/run_pipeline.sh bridge"
    echo "    outbox=$SPOOL_DIR/outbox"
    exec $PY live_spool.py "$MODEL" "$SRC"
    ;;

  bridge)
    CLOUD_ENDPOINT="${CLOUD_ENDPOINT:-http://localhost:8000/review}" \
      exec $PY cloud_bridge.py
    ;;

  report)
    exec $PY daily_report.py "${@:2}"
    ;;

  *)
    echo "用法: $0 {serve|local|camera [model] [src]|bridge|report}"
    echo "  serve   启云端复核服务 (需 DASHSCOPE_API_KEY)"
    echo "  local   PC 离线打通整条链路 (需 DASHSCOPE_API_KEY)"
    echo "  camera  板端相机实时采样 (需另开 serve+bridge 终端)"
    echo "  bridge  仅启上传守护进程"
    echo "  report 仅生成日报 (--all 看历史全部)"
    exit 1
    ;;
esac

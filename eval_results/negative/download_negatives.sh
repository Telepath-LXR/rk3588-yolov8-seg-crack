#!/usr/bin/env bash
# 下载负样本（无裂缝表面）用于误检率评测。
# 来源：GitHub rkyadav2710/Neural_Net-for-Concrete-Crack-Detection 的 SDNET2018 Wall_crack_dataset/W/UW（Uncracked Wall）
# 协议：SDNET2018（Utah State Univ. Maguire et al. 2018）研究用途；GitHub 镜像仓为学习项目。报告需注明来源。
# 说明：git trees API 有未认证限流，故改为按已知命名规律 7069-<i>.jpg 探测下载。
# 用法: ./download_negatives.sh [数量]   (默认 90)
set -e
N="${1:-90}"
REPO="rkyadav2710/Neural_Net-for-Concrete-Crack-Detection"
DEST="/home/alientek/yolo8-seg/eval_results/negative/images"
mkdir -p "$DEST"

echo "下载最多 $N 张 UW 负样本 → $DEST"
ok=0; i=0; miss=0
# 命名规律 7069-<n>.jpg, n 从 1 起；连续 miss 超阈值则停
while [ "$ok" -lt "$N" ] && [ "$i" -lt 2000 ] && [ "$miss" -lt 40 ]; do
  i=$((i+1))
  url="https://raw.githubusercontent.com/$REPO/master/Wall_crack_dataset/W/UW/7069-$i.jpg"
  fn="$DEST/neg_$(printf '%03d' $((ok+1))).jpg"
  if curl -sfL --max-time 20 -o "$fn" "$url"; then
    # 校验是有效 JPEG 且非 0 字节
    if [ -s "$fn" ] && file "$fn" | grep -qi "image"; then
      ok=$((ok+1)); miss=0
    else
      rm -f "$fn"; miss=$((miss+1))
    fi
  else
    rm -f "$fn"; miss=$((miss+1))
  fi
done
echo "完成: $ok 张负样本（探测 $i 个序号）"
ls "$DEST" | wc -l

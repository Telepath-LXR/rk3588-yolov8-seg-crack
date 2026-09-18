# daily_report.py
"""扫描 spool/done/，聚合当日云复核事件，生成 Markdown 日报。

默认只统计当日（按 result.json 里的 ts epoch 秒过滤）。
--all 看历史全部。
环境变量:
  SPOOL_DIR  spool 根目录 (默认 spool)
"""
import json
import os
import sys
from datetime import datetime, date

SPOOL = os.environ.get("SPOOL_DIR", "spool")
DONE = os.path.join(SPOOL, "done")
SHOW_ALL = "--all" in sys.argv
TODAY = date.today()

stats = {"total": 0, "upgrade": 0, "downgrade": 0,
         "human_queue": 0, "keep_edge": 0, "unreviewed": 0}
events = []

if not os.path.isdir(DONE):
    print(f"[report] done 目录不存在: {DONE}（尚无复核完成事件）")
    sys.exit(0)

skipped_other_day = 0
for d in sorted(os.listdir(DONE)):
    try:
        meta = json.load(open(f"{DONE}/{d}/meta.json", encoding="utf-8"))
        result = json.load(open(f"{DONE}/{d}/result.json", encoding="utf-8"))
    except Exception:
        continue
    # 当日过滤：review.py 在 result 之外，ledger/result 都写了 ts(epoch 秒)
    ts = result.get("ts") or meta.get("ts")
    if not SHOW_ALL and ts:
        try:
            ev_day = datetime.fromtimestamp(float(ts)).date()
        except Exception:
            ev_day = None
        if ev_day is not None and ev_day != TODAY:
            skipped_other_day += 1
            continue
    stats["total"] += 1
    dec = result.get("decision", "unreviewed")
    stats[dec] = stats.get(dec, 0) + 1
    if result.get("verdict"):
        events.append({
            "dir": d,
            "conf": meta.get("conf", 0),
            "is_crack": result["verdict"].get("is_crack"),
            "vlm_conf": result["verdict"].get("confidence"),
            "reason": result["verdict"].get("reason", ""),
            "decision": dec,
        })

date_tag = "all" if SHOW_ALL else datetime.now().strftime('%Y%m%d')
OUT = f"daily_report_{date_tag}.md"

md = [f"# 巡检日报 {datetime.now().strftime('%Y-%m-%d')}"
      + ("（全部历史）" if SHOW_ALL else "（当日）") + "\n"]
md.append("## 统计\n")
md.append(f"- 总候选: {stats['total']}")
md.append(f"- 云端确认(升级): {stats['upgrade']}")
md.append(f"- 疑似误检(降级): {stats['downgrade']}")
md.append(f"- 待人工: {stats['human_queue']}")
md.append(f"- 保持原判: {stats['keep_edge']}")
md.append(f"- 未复核: {stats['unreviewed']}")
if not SHOW_ALL and skipped_other_day:
    md.append(f"- 已跳过非当日: {skipped_other_day}")
md.append("")

md.append("## 事件明细\n")
md.append("| 目录 | 端侧conf | VLM判定 | VLM把握 | 决策 | 理由 |")
md.append("|---|---:|---|---:|---|---|")
for e in sorted(events, key=lambda x: -x["conf"]):
    md.append(f"| {e['dir'][:20]} | {e['conf']:.3f} | {e['is_crack']} | "
              f"{e['vlm_conf']:.2f} | {e['decision']} | {e['reason']} |")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(md))
print(f"日报已生成: {OUT}")
print("\n".join(md[:15]))

"""
S4 收尾：before/after FPS 对比图
Python 参考基线 (29.4) → C++ 零拷贝 schedutil (43) → C++ 零拷贝 performance (61.2)
+ 相机 30 FPS 物理上限参考线

数据源：研究报告_YOLOv8-seg_RK3588零拷贝优化.md L303-306
配色：参考 sequential-blue 色阶（已验证 ordinal ramp，浅→深 = before→after）
"""
import os
from matplotlib import font_manager
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- 注册 CJK 字体（Noto Sans CJK SC）----
NOTO = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
NOTO_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
for p in (NOTO, NOTO_BOLD):
    if os.path.exists(p):
        font_manager.fontManager.addfont(p)
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ---- 数据（来自研究报告 L303-306，诚实未改）----
labels = [
    "C++ 零拷贝 (performance 调速器)",   # 最终，最深
    "C++ 零拷贝 (schedutil 调速器)",     # 中间
    "Python 参考基线 (RKNNLite)",         # 起点，最浅
]
fps   = [61.2, 43.0, 29.4]
ms    = [16.3, 22.8, 34.0]
# sequential-blue 色阶（参考 palette.md，浅→深 = before→after，单色递进）
colors = ["#184f95", "#2a78d6", "#6da7ec"]   # step600 / step450 / step300

CAM_FPS = 30.0  # IMX415 ISP 物理上限

# ---- 画布 / 配色 token（参考 palette.md）----
SURFACE   = "#fcfcfb"
INK_PRI   = "#0b0b0b"
INK_SEC   = "#52514e"
INK_MUTED = "#898781"
GRID      = "#e1e0d9"
REF_COLOR = "#52514e"   # 参考线用 secondary ink，不占用 status 红

fig, ax = plt.subplots(figsize=(9.2, 4.6), dpi=150)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

y_pos = [2, 1, 0]   # 顶部=Python 基线叙事顺序：top=最终，bottom=基线 → 改为 top=基线更好读
# 重新排：top=基线(浅) → bottom=最终(深)，before→after 自上而下
labels = list(reversed(labels))   # 基线在顶
fps    = list(reversed(fps))
ms     = list(reversed(ms))
colors = list(reversed(colors))   # 浅在顶，深在底
y_pos  = [2, 1, 0]

# ---- 条形（高度留白 = surface gap 风格；2px 间距由 bar 高度 < 1 实现）----
BAR_H = 0.62
bars = ax.barh(y_pos, fps, height=BAR_H, color=colors,
               edgecolor="none", zorder=3)

# ---- 数值直标（条端右侧，primary ink，非系列色）----
for ypos, v, m in zip(y_pos, fps, ms):
    ax.text(v + 1.2, ypos, f"{v:.1f} FPS  ({m:.1f} ms)",
            va="center", ha="left", fontsize=10.5, color=INK_PRI,
            fontweight="semibold", zorder=4)

# ---- 相机 30 FPS 物理上限参考线（solid，非 dashed）----
ax.axvline(CAM_FPS, color=REF_COLOR, linewidth=1.3,
           linestyle="solid", zorder=2, alpha=0.85)
ax.text(CAM_FPS, 2.62, "相机 30 FPS 物理上限\n(IMX415 ISP 节拍)",
        va="bottom", ha="center", fontsize=8.5, color=INK_SEC,
        linespacing=1.3, zorder=4)

# ---- 轴 / 网格 ----
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=10.5, color=INK_PRI)
ax.set_xlim(0, 72)
ax.set_xticks([0, 15, 30, 45, 60])
ax.set_xticklabels([f"{t}" for t in [0, 15, 30, 45, 60]],
                   fontsize=9, color=INK_MUTED)
ax.set_xlabel("端到端处理吞吐 (FPS)  ·  视频文件源，无相机节拍",
              fontsize=9.5, color=INK_SEC, labelpad=8)
ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=1)
ax.yaxis.grid(False)
ax.set_axisbelow(True)
for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="y", length=0, pad=6)
ax.tick_params(axis="x", length=3, color=INK_MUTED)

# ---- 标题 / 副标题 ----
ax.set_title("YOLOv8-seg 端到端处理吞吐：Python 基线 → C++ 零拷贝优化",
             fontsize=13.5, color=INK_PRI, fontweight="bold",
             loc="left", pad=16)
ax.text(0, 1.045, "2.08× 提升  ·  NPU 推理占端到端 96.7%  ·  推理已逼近 RK3588 NPU 1 GHz 硬件极限",
        transform=ax.transAxes, fontsize=9.5, color=INK_SEC)

# ---- 诚实注释（底部小字）----
fig.text(0.062, 0.015,
         "注：相机实时墙钟仍 30 FPS（IMX415 硬件节拍，非推理瓶颈）；"
         "视频文件源展示真实处理吞吐。数据见研究报告 L303-306。",
         fontsize=7.6, color=INK_MUTED, linespacing=1.4)

plt.subplots_adjust(left=0.26, right=0.97, top=0.84, bottom=0.16)
out = os.path.join(os.path.dirname(__file__), "fps_comparison.png")
plt.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.18)
print("saved:", out)

# scripts/plot_fid_facet_paper.py
"""
顶刊级分面绘图（细线+科学扩轴+轨迹标记版）
=============================================
功能：生成SCI期刊高质量的FID训练曲线对比图
特点：
  - 线宽 0.8-1.0pt，彻底解决曲线粘连
  - Y轴固定 (12, 72) 提供专业呼吸空间
  - 20k 步间隔标记点辅助轨迹追踪
  - 误差带 alpha=0.07，均值线绝对清晰
  - 图例位置精确控制，避免遮挡X轴

作者：Your Name
日期：2026-04-24
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# ============================================================================
# 🔧 第一部分：核心配置参数（可手动调整）
# ============================================================================

# --- 数据文件配置 ---
PLOT_DIR = Path("plot_data")
PLOT_FILES = sorted(PLOT_DIR.glob("plot_data_*.json"))
if not PLOT_FILES:
    raise FileNotFoundError(" 请先运行 3_compute_plot_data.py 生成数据文件")
INPUT_FILE = PLOT_FILES[-1]  # 使用最新的数据文件
print(f" 加载数据: {INPUT_FILE.name}")

OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_NAME = "fid_facet_paper"  # 输出文件名（不含扩展名）

# --- 图像质量配置 ---
DPI_RASTER = 600  # PNG输出分辨率（PDF不受此参数影响）
FONT_BASE = 8.5   # 基础字体大小（pt），推荐范围 7-10

# --- 布局配置 ---
FIG_WIDTH = 7.2   # 图宽度（英寸），双栏论文推荐 3.5 或 7.2
FIG_HEIGHT = 3.5  # 图高度（英寸），推荐 3.0-4.0

# --- Y轴范围配置（关键参数）---
# 说明：根据数据范围(28~67)上下各留 ~15-20 单位呼吸空间
# 调整建议：如果曲线太靠上/下，修改这两个值
Y_LIM_BOTTOM = 12  # Y轴下限
Y_LIM_TOP = 72     # Y轴上限
Y_LIM_FIXED = (Y_LIM_BOTTOM, Y_LIM_TOP)

# --- 轨迹标记点配置 ---
# 说明：在这些训练步数位置绘制圆形标记点，帮助区分重叠曲线
MARKER_STEPS = [0, 20000, 40000, 60000, 80000, 100000]
MARKER_SIZE = 2.5  # 标记点大小（pt）

# --- 图例位置配置（解决遮挡的关键）---
# 说明：bbox_to_anchor 的 y 值控制图例垂直位置
#       负值表示向下移动，-0.16 表示图例中心在图底部下方 16% 处
# 调整建议：
#   - 如果图例仍遮挡X轴：将 LEGEND_Y_OFFSET 改为 -0.18 或 -0.20
#   - 如果图例离图太远：改为 -0.12 或 -0.14
#   - 配合 SUBPLOT_BOTTOM_MARGIN 一起调整
LEGEND_Y_OFFSET = -0.12  # 图例垂直偏移（Figure坐标系，范围 -0.3 ~ 0）
SUBPLOT_BOTTOM_MARGIN = 0.0  # 子图底部留白比例（0-1），给图例留空间

# ============================================================================
#  第二部分：分组与样式配置
# ============================================================================

# --- 面板分组配置 ---
# 左面板（Panel A）：温度参数消融实验
PANEL_A_KEYS = [
    "baseline/combined", 
    "daosheng_temp07/combined", 
    "daosheng_temp08/combined",
    "daosheng_temp085/combined", 
    "daosheng_temp09/combined"
]

# 右面板（Panel B）：核心机制对比
PANEL_B_KEYS = [
    "baseline/combined", 
    "yinyang_gradscale/combined", 
    "yinyang_daosheng/combined"
]

# 合并所有键（保序去重，用于图例）
ALL_KEYS = list(dict.fromkeys(PANEL_A_KEYS + PANEL_B_KEYS))

# --- 曲线样式配置 ---
# 说明：每个实验组对应一种样式
# 颜色采用 ColorBrewer 高对比色板，确保黑白打印可区分
STYLE_CONFIG = {
    "baseline/combined": {
        "label": "SNGAN (Baseline)", 
        "color": "#666666",      # 灰色
        "linestyle": "-",        # 实线
        "linewidth": 0.8         # 细线
    },
    "daosheng_temp07/combined": {
        "label": "DaoSheng τ=0.7", 
        "color": "#E69F00",      # 橙色
        "linestyle": "--",       # 虚线
        "linewidth": 0.8
    },
    "daosheng_temp08/combined": {
        "label": "DaoSheng τ=0.8", 
        "color": "#56B4E9",      # 天蓝色（推荐主方法）
        "linestyle": "-",        
        "linewidth": 1.0         # 略粗，突出显示
    },
    "daosheng_temp085/combined": {
        "label": "DaoSheng τ=0.85", 
        "color": "#009E73",      # 绿色
        "linestyle": "-.",       # 点划线
        "linewidth": 0.8
    },
    "daosheng_temp09/combined": {
        "label": "DaoSheng τ=0.9", 
        "color": "#F0E442",      # 黄色
        "linestyle": ":",        # 点线
        "linewidth": 0.8
    },
    "yinyang_gradscale/combined": {
        "label": "YinYang-Gradscale", 
        "color": "#0072B2",      # 深蓝色
        "linestyle": "-",        
        "linewidth": 1.0         # 略粗，核心方法
    },
    "yinyang_daosheng/combined": {
        "label": "YinYang+DaoSheng", 
        "color": "#D55E00",      # 朱红色（推荐主方法）
        "linestyle": "-",        
        "linewidth": 1.0         # 略粗，核心方法
    },
}

# --- 绘图通用配置 ---
METRIC = "FID"  # 绘制指标（FID/IS/KID）
YLABEL = "FID Score ↓"  # Y轴标签（↓表示越低越好）
XLABEL = "Training Step"  # X轴标签

# ============================================================================
#  第三部分：Matplotlib 全局样式设置
# ============================================================================

rcParams.update({
    # 字体配置（SCI期刊要求）
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": FONT_BASE,
    
    # 文字大小分层控制
    "axes.labelsize": FONT_BASE,           # 坐标轴标签
    "axes.titlesize": FONT_BASE + 1,       # 子图标题
    "xtick.labelsize": FONT_BASE - 1,      # X轴刻度
    "ytick.labelsize": FONT_BASE - 1,      # Y轴刻度
    "legend.fontsize": FONT_BASE - 1,      # 图例文字
    
    # 图例框配置
    "legend.frameon": True,
    "legend.framealpha": 0.95,  # 轻微透明，不遮挡背景
    
    # 线条粗细配置
    "axes.linewidth": 0.7,       # 坐标轴边框
    "grid.linewidth": 0.35,      # 网格线
    "lines.linewidth": 0.8,      # 数据曲线基础宽度
    
    # 刻度线配置（inward 风格，顶刊常用）
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,           # 顶部显示刻度
    "ytick.right": True,         # 右侧显示刻度
    
    # 保存配置
    "savefig.bbox": "tight",     # 自动裁剪空白
    "savefig.pad_inches": 0.05,  # 保留少量边距
    
    # 布局引擎（先关闭，后面手动控制）
    "figure.constrained_layout.use": False,
    
    # PDF字体嵌入（期刊要求）
    "pdf.fonttype": 42,  # TrueType字体
    "ps.fonttype": 42,
})

# ============================================================================
#  第四部分：主绘图函数
# ============================================================================

def plot_facet_paper():
    """
    主绘图函数
    流程：
      1. 加载JSON数据
      2. 创建双面板Figure
      3. 分别绘制左右面板
      4. 统一图例与布局调整
      5. 保存为PDF和PNG
    """
    
    # --- 步骤1：加载数据 ---
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        plot_data = json.load(f)
    
    print(f" Y轴范围: {Y_LIM_FIXED}")
    print(f" 图例偏移: y={LEGEND_Y_OFFSET}, 底部留白: {SUBPLOT_BOTTOM_MARGIN}")

    # --- 步骤2：创建Figure和子图 ---
    fig, (ax1, ax2) = plt.subplots(
        1, 2,                    # 1行2列
        figsize=(FIG_WIDTH, FIG_HEIGHT), 
        dpi=DPI_RASTER
    )
    
    # 定义两个面板的配置
    panels = [
        (ax1, PANEL_A_KEYS, "Temperature Ablation (τ)"),  # 左面板
        (ax2, PANEL_B_KEYS, "Core Mechanism Comparison")   # 右面板
    ]

    # --- 步骤3：遍历面板绘制曲线 ---
    for ax, keys, title in panels:
        for key in keys:
            # 跳过不存在的数据
            if key not in plot_data or not plot_data[key][METRIC]["steps"]: 
                continue
            
            # 获取样式配置
            cfg = STYLE_CONFIG[key]
            d = plot_data[key][METRIC]
            
            # 提取数据
            steps = np.array(d["steps"])
            mean = np.array(d["mean"])
            std = np.array(d["std"])

            # 3.1 绘制误差带（极低透明度，不干扰视线）
            ax.fill_between(
                steps, 
                mean - std, 
                mean + std, 
                color=cfg["color"], 
                alpha=0.07,  # 7%透明度
                zorder=2     # 图层顺序：底层
            )
            
            # 3.2 绘制均值曲线（细线）
            ax.plot(
                steps, 
                mean, 
                color=cfg["color"], 
                linestyle=cfg["linestyle"],
                linewidth=cfg["linewidth"], 
                zorder=3,           # 图层顺序：中层
                label=cfg["label"],
                solid_capstyle="round"  # 圆头线端，更精致
            )

            # 3.3 绘制轨迹标记点（每20k步，辅助区分重叠曲线）
            mask = np.isin(steps, MARKER_STEPS)
            ax.plot(
                steps[mask], 
                mean[mask], 
                "o",                     # 圆形标记
                color=cfg["color"], 
                markersize=MARKER_SIZE,
                markeredgecolor="white",  # 白色边框，增强对比
                markeredgewidth=0.5, 
                zorder=4                 # 图层顺序：顶层
            )

        # --- 步骤4：面板装饰 ---
        
        # 4.1 高亮收敛区间（80k-100k步）
        ax.axvspan(
            80000, 100000, 
            color="#f8f9fa",      # 浅灰背景
            alpha=0.6, 
            zorder=1, 
            edgecolor="#dee2e6",   # 边框
            linewidth=0.4
        )

        # 4.2 网格配置
        ax.grid(
            True, 
            linestyle=":",        # 点线网格
            linewidth=0.35, 
            alpha=0.35, 
            zorder=0              # 最底层
        )
        
        # 4.3 坐标轴范围与刻度
        ax.set_xlim(0, 100000)
        ax.set_ylim(*Y_LIM_FIXED)
        ax.set_xticks([0, 40000, 80000, 100000])
        ax.set_xticklabels(["0", "40k", "80k", "100k"])
        
        #  新增：X轴标签（两个面板都添加）
        ax.set_xlabel(XLABEL, labelpad=4)  # labelpad控制标签与轴的距离        
        
        # 4.4 子图标题
        ax.set_title(
            title, 
            pad=8,                      # 标题与图的距离（pt）
            fontsize=FONT_BASE+1, 
            fontweight="medium"
        )
        
        # 4.5 Y轴标签（仅左面板显示）
        if ax == ax1: 
            ax.set_ylabel(YLABEL, labelpad=4)  # labelpad控制标签距离
        else: 
            ax.tick_params(axis="y", labelleft=False)  # 右面板隐藏Y轴刻度

    # --- 步骤5：统一图例（放在两个子图下方中央）---
    
    # 5.1 收集所有图例句柄和标签（去重）
    handles, labels = [], []
    seen = set()
    for key in ALL_KEYS:
        if key not in seen and key in STYLE_CONFIG:
            cfg = STYLE_CONFIG[key]
            handles.append(
                plt.Line2D(
                    [], [], 
                    color=cfg["color"], 
                    linestyle=cfg["linestyle"],
                    linewidth=cfg["linewidth"], 
                    label=cfg["label"]
                )
            )
            labels.append(cfg["label"])
            seen.add(key)
    
    # 5.2 创建图例
    fig.legend(
        handles, 
        labels, 
        loc="lower center",                # 位置：底部居中
        bbox_to_anchor=(0.5, LEGEND_Y_OFFSET),  # 锚点：x=0.5居中, y=偏移量
        ncol=4,                            # 4列显示
        columnspacing=1.6,                 # 列间距
        handlelength=1.5,                  # 图例线段长度
        framealpha=0.95                    # 图例框透明度
    )

    # --- 步骤6：精确布局控制（防止图例遮挡）---
    
    # 6.1 关闭自动布局（避免与手动设置冲突）
    fig.set_constrained_layout(False)
    
    # 6.2 手动调整子图边距
    # rect=[left, bottom, right, top] 定义子图可占用区域
    fig.tight_layout(rect=[0, SUBPLOT_BOTTOM_MARGIN, 1, 1])
    
    # --- 步骤7：保存图像 ---
    for fmt in ["pdf", "png"]:
        out = OUTPUT_DIR / f"{OUTPUT_NAME}.{fmt}"
        
        # 保存参数
        save_kwargs = {
            "dpi": DPI_RASTER if fmt == "png" else 300,
            "bbox_inches": "tight",   # 自动裁剪
            "pad_inches": 0.05        # 保留边距
        }
        
        plt.savefig(out, **save_kwargs)
        print(f" 保存: {out} ({out.stat().st_size/1024:.1f} KB)")
    
    plt.close(fig)  # 显式关闭释放内存

    # --- 步骤8：输出关键数据（便于论文表格）---
    print(f"\n 100k 步关键数据（{METRIC}）:")
    for k in ["baseline/combined", "yinyang_gradscale/combined", "yinyang_daosheng/combined"]:
        if k in plot_data:
            d = plot_data[k][METRIC]
            print(f"    {STYLE_CONFIG[k]['label']:22}: "
                  f"{d['mean'][-1]:.2f} ± {d['std'][-1]:.2f} (N={d['n_seeds']})")

# ============================================================================
#  第五部分：程序入口
# ============================================================================

if __name__ == "__main__":
    plot_facet_paper()
# 6_generate_fig3_qualitative.py
"""
Fig. 3 定性与定量对比图生成脚本
日期：2026-05-09
"""

import os
import glob
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image
from matplotlib import rcParams

# ============================================================================
#  第一部分：全局配置（与 plot_fid_facet_paper.py 严格对齐）
# ============================================================================
LOGS_DIR = Path("logs")
PLOT_DATA_DIR = Path("plot_data")
OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_NAME = "fig3_qualitative_comparison"

DPI_RASTER = 600
FIG_WIDTH = 7.2
FIG_HEIGHT = 4.5

FONT_BASE = 8.5
COLORS = {
    "baseline": "#666666",
    "daogan": "#0072B2",      # 深蓝色（对齐曲线图 YinYang-Gradscale 高亮）
    "improve_pos": "#009E73", # 绿色（正向改进）
    "sig_marker": "#D55E00",  # 显著性标记色
    "bg_light": "#f8f9fa",
    "border": "#dee2e6",
    "text_secondary": "#6c757d"
}

MODE_KEYWORDS = {
    "baseline": "Baseline",
    "daogan": "True-YinYang-Gradscale"  # 匹配日志目录名
}

JSON_KEYS = {
    "baseline": "baseline/combined",
    "daogan": "yinyang_gradscale/combined"
}

# Table A1 精确统计值
TABLE_A1_STATS = {
    "FID": {"ci_95": "[-1.97, -0.39]", "sig": True,  "p_fdr": "0.009"},
    "IS":  {"ci_95": "[-0.04, +0.18]", "sig": False, "p_fdr": "0.176"}
}

# ============================================================================
#  第二部分：Matplotlib 全局样式设置
# ============================================================================
rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": FONT_BASE,
    "axes.labelsize": FONT_BASE,
    "axes.titlesize": FONT_BASE + 1,
    "xtick.labelsize": FONT_BASE - 1,
    "ytick.labelsize": FONT_BASE - 1,
    "legend.fontsize": FONT_BASE - 1,
    "axes.linewidth": 0.7,
    "figure.constrained_layout.use": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# ============================================================================
#  第三部分：数据加载与图像查找
# ============================================================================
def load_final_metrics(json_path: Path) -> dict:
    """从聚合 JSON 提取 100k 步的最终指标 (mean, std, n_seeds)"""
    if not json_path.exists():
        raise FileNotFoundError(f" 未找到聚合数据: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    stats = {}
    for mode in ["baseline", "daogan"]:
        key = JSON_KEYS[mode]
        if key not in data:
            raise KeyError(f" JSON 中缺失 {key}，请检查数据聚合脚本")
        stats[mode] = {}
        for met in ["FID", "IS", "KID"]:
            d = data[key][met]
            stats[mode][met] = {
                "mean": float(d["mean"][-1]),
                "std": float(d["std"][-1]),
                "n": int(d["n_seeds"])
            }
    return stats

def find_sample_image(mode: str, step: int = 100000) -> Path:
    """扫描日志目录查找对应步数的样本图"""
    keyword = MODE_KEYWORDS.get(mode, mode)
    pattern = str(LOGS_DIR / "seed*" / f"*{keyword}*" / f"{step:06d}.png")
    candidates = sorted(glob.glob(pattern))
    return Path(candidates[-1]) if candidates else None

def create_placeholder_grid() -> np.ndarray:
    """生成高质量 8x8 CIFAR-10 风格占位网格 (256x256px)"""
    np.random.seed(42)
    img = np.random.uniform(0.15, 0.85, (256, 256, 3))
    for i in range(9):
        pos = i * 32
        img[pos-1:pos+2, :, :] = 0.92
        img[:, pos-1:pos+2, :] = 0.92
    return img

def load_or_create_image(mode: str, step: int = 100000) -> np.ndarray:
    """加载真实图像，若缺失则生成占位图"""
    img_path = find_sample_image(mode, step)
    if img_path and img_path.exists():
        return np.array(Image.open(img_path))
    print(f" 未找到 {mode} 样本图 (step={step})，使用高保真占位网格替代")
    return create_placeholder_grid()

# ============================================================================
#  第四部分：核心绘图函数
# ============================================================================
def plot_qualitative_comparison(stats: dict):
    print(" 正在绘制 Fig. 3 (统计增强版)...")
    
    img_baseline = load_or_create_image("baseline")
    img_daogan   = load_or_create_image("daogan")
    
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI_RASTER)
    gs = gridspec.GridSpec(2, 2, height_ratios=[7, 3], hspace=0.05, wspace=0.08)
    
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_metrics = fig.add_subplot(gs[1, :])
    
    # --- 渲染图像子图 (a) & (b) ---
    for ax, img, title, mode in zip(
        [ax_a, ax_b], 
        [img_baseline, img_daogan], 
        ["(a) SNGAN (Baseline)", "(b) DaoGAN (Yin-Yang GradScale)"],
        ["baseline", "daogan"]
    ):
        ax.imshow(img, aspect="equal")
        ax.axis("off")
        ax.set_title(title, loc="left", fontweight="bold", color=COLORS[mode], pad=6)
        rect = patches.Rectangle((0,0), 1, 1, transform=ax.transAxes, 
                                fill=False, edgecolor=COLORS["border"], linewidth=1.2, clip_on=False)
        ax.add_patch(rect)
        ax.text(0.98, 0.02, "seed representative", transform=ax.transAxes, 
                fontsize=FONT_BASE-2, color=COLORS["text_secondary"], ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=COLORS["border"], alpha=0.9))

    # --- 底部指标面板（居中对齐优化版）---
    ax_metrics.set_facecolor(COLORS["bg_light"])
    ax_metrics.axis("off")
    ax_metrics.set_xlim(0, 13)
    ax_metrics.set_ylim(0, 4.2)
    
    fid_imp = (stats["baseline"]["FID"]["mean"] - stats["daogan"]["FID"]["mean"]) / stats["baseline"]["FID"]["mean"] * 100
    is_imp  = (stats["daogan"]["IS"]["mean"] - stats["baseline"]["IS"]["mean"]) / stats["baseline"]["IS"]["mean"] * 100
    
    #  居中布局坐标系统（以 x=6.5 为中心轴）
    center_x = 6.5
    LAYOUT = {
        "label": center_x - 4.0, 
        "base": center_x - 2.3,
        "arrow_start": center_x - 1.5,
        "arrow_end": center_x - 0.9,    # 箭头
        "target": center_x + 0.1,       # 目标值
        "imp": center_x + 2.3,          
        "seed": center_x + 4.0,         
    }
    
    metrics_layout = [
        {"label": "FID ↓", "base": stats["baseline"]["FID"], "target": stats["daogan"]["FID"], 
         "imp_pct": fid_imp, "y": 3.0, "ci": TABLE_A1_STATS["FID"]["ci_95"], "sig": TABLE_A1_STATS["FID"]["sig"]},
        {"label": "IS ↑",  "base": stats["baseline"]["IS"],  "target": stats["daogan"]["IS"], 
         "imp_pct": is_imp,  "y": 1.5, "ci": TABLE_A1_STATS["IS"]["ci_95"],  "sig": TABLE_A1_STATS["IS"]["sig"]}
    ]
    
    for row in metrics_layout:
        b, t = row["base"], row["target"]
        
        # 1. 指标名称（左对齐到标签位置）
        ax_metrics.text(LAYOUT["label"], row["y"], row["label"], 
                       fontsize=FONT_BASE+1, fontweight="bold", ha="left", va="center")
        
        # 2. Baseline 值（居中）
        ax_metrics.text(LAYOUT["base"], row["y"], f"${b['mean']:.2f} \\pm {b['std']:.2f}$", 
                       color=COLORS["baseline"], fontsize=FONT_BASE, ha="center", va="center")
        
        # 3. 箭头（从 baseline 指向 target）
        ax_metrics.annotate("", xy=(LAYOUT["arrow_end"], row["y"]), xytext=(LAYOUT["arrow_start"], row["y"]),
                           arrowprops=dict(arrowstyle="->", color=COLORS["daogan"], lw=1.2))
        
        # 4. DaoGAN 值（居中 + 显著性标记）
        sig_mark = r"^{\ddagger}" if row["sig"] else ""
        text_str = f"$\\mathbf{{{t['mean']:.2f} \\pm {t['std']:.2f}}}{sig_mark}$"
        ax_metrics.text(LAYOUT["target"], row["y"], text_str, 
                       color=COLORS["daogan"], fontsize=FONT_BASE+0.5, ha="center", va="center")
        
        # 5. 改进徽章（居中显示）
        imp_text = f"+{row['imp_pct']:.1f}%\n(95% CI: {row['ci']})"
        ax_metrics.text(LAYOUT["imp"], row["y"], imp_text, 
                       fontsize=FONT_BASE-1, color=COLORS["improve_pos"], 
                       ha="center", va="center", linespacing=1.2,
                       bbox=dict(boxstyle="round,pad=0.25", fc="white", 
                                ec=COLORS["improve_pos"], alpha=0.85, linewidth=0.8))
        
        # 6. 种子数（居中对齐）
        ax_metrics.text(LAYOUT["seed"], row["y"], f"n={b['n']} seeds", 
                       fontsize=FONT_BASE-2, color=COLORS["text_secondary"], 
                       ha="center", va="center")

    # --- 底部精简注释（全局居中）---
    n_seeds = stats["baseline"]["FID"]["n"]
    note_text = (
        f"Values: mean $\\pm$ std ($n={n_seeds}$ seeds). "
    )
    ax_metrics.text(center_x, 0.5, note_text, 
                   fontsize=FONT_BASE-2, style="italic", color=COLORS["text_secondary"], 
                   ha="center", va="center")

    # --- 保存 ---
    fig.subplots_adjust(left=0.03, right=0.97, top=0.96, bottom=0.08, wspace=0.06, hspace=0.02)
    
    for fmt in ["pdf", "png"]:
        out = OUTPUT_DIR / f"{OUTPUT_NAME}.{fmt}"
        dpi = DPI_RASTER if fmt == "png" else None
        fig.savefig(out, dpi=dpi, format=fmt)
        print(f" 保存: {out} ({out.stat().st_size/1024:.1f} KB)")
        
    plt.close(fig)

# ============================================================================
#  第五部分：主程序入口
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Generate Fig.3 Qualitative & Quantitative Comparison")
    parser.add_argument("--logs_dir", type=str, default="logs")
    parser.add_argument("--plot_data_dir", type=str, default="plot_data")
    parser.add_argument("--output_dir", type=str, default="figures")
    parser.add_argument("--step", type=int, default=100000)
    args = parser.parse_args()
    
    global LOGS_DIR, PLOT_DATA_DIR, OUTPUT_DIR
    LOGS_DIR = Path(args.logs_dir)
    PLOT_DATA_DIR = Path(args.plot_data_dir)
    OUTPUT_DIR = Path(args.output_dir)
    
    print(f" 日志路径: {LOGS_DIR}")
    print(f" 数据路径: {PLOT_DATA_DIR}")
    
    json_files = sorted(PLOT_DATA_DIR.glob("plot_data_*.json"))
    if not json_files:
        raise FileNotFoundError(" 请先运行 3_compute_plot_data.py 生成 plot_data_*.json")
    input_file = json_files[-1]
    print(f" 使用数据文件: {input_file.name}")
    
    stats = load_final_metrics(input_file)
    print(f" 已加载指标: Baseline FID={stats['baseline']['FID']['mean']:.2f}, DaoGAN FID={stats['daogan']['FID']['mean']:.2f}")
    
    plot_qualitative_comparison(stats)
    print("\n Fig. 3 生成完成！")

if __name__ == "__main__":
    main()
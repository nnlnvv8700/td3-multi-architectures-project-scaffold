# -*- coding: utf-8 -*-
# training/plot_compare_grids_scientific.py
# 用法示例（PowerShell）：
# python ".\training\evaluate.py" --run_dirs `
#   ".\results\KukaIiwa7Track-v0_mlp_dense_20251115_200935" `
#   ".\results\KukaIiwa7Track-v0_gnn_dense_20251115_151707" `
#   ".\results\KukaIiwa7Track-v0_transformer_dense_20251115_173620" `
#   ".\results\KukaIiwa7Track-v0_gnn_transformer_dense_20251115_131235"

import os
import json
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.ticker import AutoMinorLocator, ScalarFormatter

# ---------- 全局科研风格 ----------
mpl.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "dejavuserif",
    "axes.unicode_minus": False,
    "figure.dpi": 160,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,   # 矢量文字
    "ps.fonttype": 42,
})
AX_LABEL_FS   = 12
TICK_FS       = 11
TITLE_FS      = 12
LEGEND_FS     = 11
LINE_W        = 1.8
LINE_ALPHA    = 0.85       # 线条透明度，避免完全覆盖
MARKER_SIZE   = 4.5        # 增大marker便于区分
MARKER_ALPHA  = 0.7        # marker透明度
GRID_ALPHA    = 0.25

COLOR_MAP = {
    "MLP": "#1f77b4",            # 蓝
    "GNN": "#ff7f0e",            # 橙
    "Transformer": "#2ca02c",    # 绿
    "GNN+Transformer": "#d62728" # 红（含 gat_transformer）
}

MARKER_MAP = {
    "MLP": "o",                  # 圆形
    "GNN": "s",                  # 方形
    "Transformer": "^",          # 三角形
    "GNN+Transformer": "D"       # 菱形
}

def infer_label_from_run(run_dir: str) -> str:
    name = os.path.basename(os.path.normpath(run_dir)).lower()
    if "gnn_transformer" in name or "gat_transformer" in name:
        return "GNN+Transformer"
    if "transformer" in name and "gnn" not in name:
        return "Transformer"
    if "mlp" in name:
        return "MLP"
    if "gnn" in name:
        return "GNN"
    return os.path.basename(run_dir)

def load_metrics(run_dir: str):
    p = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def steps_x(metrics):
    x = metrics.get("eval_steps", [])
    if x:
        return np.asarray(x, dtype=float) / 10000.0  # 显示为 ×10k
    return None

def moving_average(y, w):
    if w <= 1 or y is None or len(y) == 0: return None
    if len(y) < w: return None
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(np.asarray(y, dtype=float), kernel, mode="valid")

def build_series(run_dirs, metric_key, labels_order, ma_window=0):
    series = {}
    for rd in run_dirs:
        m = load_metrics(rd)
        lab = infer_label_from_run(rd)
        if lab not in labels_order:
            labels_order.append(lab)
        x = steps_x(m)
        y = m.get(metric_key, [])
        y = np.asarray(y, dtype=float) if y else np.array([])
        if ma_window > 1 and len(y) >= ma_window:
            y_ma = moving_average(y, ma_window)
            if x is not None and len(x) >= len(y_ma):
                x_ma = x[-len(y_ma):]   # 右对齐
            else:
                x_ma = None
            series[lab] = (x, y, COLOR_MAP.get(lab), x_ma, y_ma)
        else:
            series[lab] = (x, y, COLOR_MAP.get(lab), None, None)
    return series

def style_axes(ax, xlabel=True, ylabel=None, letter=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(True, which="major", alpha=GRID_ALPHA, linestyle="--", linewidth=0.6)
    ax.grid(True, which="minor", alpha=0.12, linestyle=":", linewidth=0.5)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    if xlabel:
        ax.set_xlabel("Evaluation Steps (×10k)", fontsize=AX_LABEL_FS)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=AX_LABEL_FS)
    if letter:
        ax.text(0.0, 1.02, f"({letter})", transform=ax.transAxes,
                fontsize=AX_LABEL_FS, fontweight="bold")

def plot_panel(ax, series_dict, title, ylabel, letter, show_ma=False, fill=False):
    # 按标签固定顺序绘制，避免随机覆盖
    plot_order = ["MLP", "GNN", "Transformer", "GNN+Transformer"]
    for lab in plot_order:
        if lab not in series_dict:
            continue
        x, y, color, x_ma, y_ma = series_dict[lab]
        if y is None or len(y) == 0:
            continue

        marker = MARKER_MAP.get(lab, 'o')
        xi = np.arange(1, len(y) + 1, dtype=float) if x is None else np.asarray(x)[:len(y)]

        xi_limited = xi
        y_limited = y

        # 绘制原始数据点（带透明度，稀疏marker避免拥挤）
        marker_every = max(1, len(y_limited) // 15)  # 每15个点显示一个marker
        ax.plot(xi_limited, y_limited, label=lab, color=color, linewidth=LINE_W,
                marker=marker, markersize=MARKER_SIZE, alpha=LINE_ALPHA,
                markerfacecolor=color, markeredgecolor='white', markeredgewidth=0.5,
                markevery=marker_every, zorder=5)
        if show_ma and y_ma is not None and len(y_ma) > 0:
            x_ma_plot = x_ma if x_ma is not None else np.arange(1, len(y_ma)+1)
            x_ma_limited = x_ma_plot
            y_ma_limited = y_ma

            # 滑动平均线（更粗，稍微透明，在上层）
            ax.plot(x_ma_limited, y_ma_limited, color=color, linewidth=LINE_W+0.5,
                    linestyle='-', alpha=0.9, zorder=10)
    ax.set_title(title, fontsize=TITLE_FS)
    style_axes(ax, xlabel=True, ylabel=ylabel, letter=letter)

def unified_legend(fig, labels_order):
    handles = []
    for lab in labels_order:
        marker = MARKER_MAP.get(lab, 'o')
        (h,) = plt.plot([], [], color=COLOR_MAP.get(lab), label=lab,
                       linewidth=LINE_W, marker=marker, markersize=MARKER_SIZE,
                       alpha=LINE_ALPHA, markerfacecolor=COLOR_MAP.get(lab),
                       markeredgecolor='white', markeredgewidth=0.5)
        handles.append(h)
    fig.legend(handles, labels_order, loc="center left",
               bbox_to_anchor=(0.88, 0.5), frameon=True, fontsize=LEGEND_FS,
               fancybox=True, shadow=True)

def save_all(fig, out_png_base):
    png = out_png_base + ".png"
    pdf = out_png_base + ".pdf"
    svg = out_png_base + ".svg"
    fig.savefig(png)
    fig.savefig(pdf)
    fig.savefig(svg)
    plt.close(fig)
    print(f"[Saved] {png}\n        {pdf}\n        {svg}")

def main():
    ap = argparse.ArgumentParser(description="科研风格：多模型 2×2 对比图（两张）")
    ap.add_argument("--run_dirs", type=str, nargs="+", required=True, help="多个训练结果目录")
    ap.add_argument("--out", type=str, default=None, help="输出目录（默认 plots/compare_scientific_时间戳）")
    ap.add_argument("--ma_window", type=int, default=0, help="可选：滑动平均窗口（0=关闭）")
    ap.add_argument("--fill_band", action="store_true", help="已弃用；不再绘制伪造的正负百分之十不确定性带")
    args = ap.parse_args()

    out_dir = args.out or os.path.join("plots", f"compare_scientific_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)

    # -------- 图 1：Reward / Success / TTS / Min Distance --------
    labels1 = []
    s_reward = build_series(args.run_dirs, "eval_rewards", labels1, ma_window=args.ma_window)
    s_succ  = build_series(args.run_dirs, "eval_success", labels1, ma_window=args.ma_window)
    s_tts   = build_series(args.run_dirs, "eval_tts", labels1,    ma_window=args.ma_window)
    s_mind  = build_series(args.run_dirs, "eval_min_distance", labels1, ma_window=args.ma_window)

    fig1 = plt.figure(figsize=(12.6, 6.2))
    gs1 = fig1.add_gridspec(2, 2, left=0.07, right=0.84, top=0.96, bottom=0.09, hspace=0.35, wspace=0.28)
    ax11 = fig1.add_subplot(gs1[0, 0]); plot_panel(ax11, s_reward, "Eval Reward (avg)", "Value", "a", show_ma=bool(args.ma_window), fill=args.fill_band)
    ax12 = fig1.add_subplot(gs1[0, 1]); plot_panel(ax12, s_succ,  "Eval Success Rate (avg)", "Value", "b", show_ma=bool(args.ma_window), fill=args.fill_band)
    ax13 = fig1.add_subplot(gs1[1, 0]); plot_panel(ax13, s_tts,   "Eval Time To Success (avg)", "Value", "c", show_ma=bool(args.ma_window), fill=args.fill_band)
    ax14 = fig1.add_subplot(gs1[1, 1]); plot_panel(ax14, s_mind,  "Eval Min Distance (avg)", "Value", "d", show_ma=bool(args.ma_window), fill=args.fill_band)
    unified_legend(fig1, labels1)
    save_all(fig1, os.path.join(out_dir, "compare_basic"))

    # -------- 图 2：RMSE / MaxDev / End-Point / Path Length --------
    labels2 = []
    s_rmse  = build_series(args.run_dirs, "eval_rmse", labels2, ma_window=args.ma_window)
    s_maxd  = build_series(args.run_dirs, "eval_max_dev", labels2, ma_window=args.ma_window)
    s_end   = build_series(args.run_dirs, "eval_end_err", labels2, ma_window=args.ma_window)
    s_plen  = build_series(args.run_dirs, "eval_path_len_exec", labels2, ma_window=args.ma_window)

    fig2 = plt.figure(figsize=(12.6, 6.2))
    gs2 = fig2.add_gridspec(2, 2, left=0.07, right=0.84, top=0.96, bottom=0.09, hspace=0.35, wspace=0.28)
    ax21 = fig2.add_subplot(gs2[0, 0]); plot_panel(ax21, s_rmse, "Eval RMSE (avg)", "Value", "a", show_ma=bool(args.ma_window), fill=args.fill_band)
    ax22 = fig2.add_subplot(gs2[0, 1]); plot_panel(ax22, s_maxd, "Eval Max Deviation (avg)", "Value", "b", show_ma=bool(args.ma_window), fill=args.fill_band)
    ax23 = fig2.add_subplot(gs2[1, 0]); plot_panel(ax23, s_end,  "Eval End-Point Error (avg)", "Value", "c", show_ma=bool(args.ma_window), fill=args.fill_band)
    ax24 = fig2.add_subplot(gs2[1, 1]); plot_panel(ax24, s_plen, "Eval Path Length (avg)", "Value", "d", show_ma=bool(args.ma_window), fill=args.fill_band)
    unified_legend(fig2, labels2)
    save_all(fig2, os.path.join(out_dir, "compare_traj"))

    print(f"[Done] Output dir: {out_dir}")

if __name__ == "__main__":
    main()

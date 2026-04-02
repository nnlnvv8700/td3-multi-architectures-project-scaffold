# -*- coding: utf-8 -*-
# training/evaluate_seeds.py
# 用法示例（PowerShell）：
# python ".\training\evaluate_seeds.py" --model_groups `
#   "MLP:.\results\seed1\mlp,.\results\seed2\mlp,.\results\seed3\mlp" `
#   "GNN:.\results\seed1\gnn,.\results\seed2\gnn,.\results\seed3\gnn" `
#   "Transformer:.\results\seed1\transformer,.\results\seed2\transformer,.\results\seed3\transformer" `
#   "GNN+Transformer:.\results\seed1\gnn_transformer,.\results\seed2\gnn_transformer,.\results\seed3\gnn_transformer"

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
GRID_ALPHA    = 0.25

COLOR_MAP = {
    "MLP": "#1f77b4",            # 蓝
    "GNN": "#ff7f0e",            # 橙
    "Transformer": "#2ca02c",    # 绿
    "GNN+Transformer": "#d62728" # 红
}

def load_metrics(run_dir: str):
    p = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def build_series_with_seeds(model_groups, metric_key):
    """
    model_groups: dict like {"MLP": [dir1, dir2, dir3], "GNN": [...], ...}
    返回: dict like {"MLP": (x, y_mean, y_std, color), ...}
    """
    series = {}
    for model_name, run_dirs in model_groups.items():
        all_y = []
        max_len = 0
        
        # 第一遍：找出最长的序列
        for rd in run_dirs:
            m = load_metrics(rd)
            y = m.get(metric_key, [])
            if y:
                y = np.asarray(y, dtype=float)
                max_len = max(max_len, len(y))
        
        if max_len == 0:
            continue
        
        # 第二遍：对齐所有序列（用NaN填充短序列）
        for rd in run_dirs:
            m = load_metrics(rd)
            y = m.get(metric_key, [])
            if y:
                y = np.asarray(y, dtype=float)
                # 如果序列较短，用NaN填充到最长长度
                if len(y) < max_len:
                    y_padded = np.full(max_len, np.nan)
                    y_padded[:len(y)] = y
                    all_y.append(y_padded)
                else:
                    all_y.append(y)
        
        if not all_y:
            continue
        
        # 生成x轴
        x = np.arange(1, max_len + 1, dtype=float)
        
        # 计算均值和标准差（忽略NaN值）
        all_y = np.array(all_y)
        y_mean = np.nanmean(all_y, axis=0)
        y_std = np.nanstd(all_y, axis=0)
        
        series[model_name] = (x, y_mean, y_std, COLOR_MAP.get(model_name))
    
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

def plot_panel(ax, series_dict, title, ylabel, letter):
    # 固定顺序：MLP, GNN, Transformer, GNN+Transformer
    fixed_order = ["MLP", "GNN", "Transformer", "GNN+Transformer"]
    ordered_labels = [lab for lab in fixed_order if lab in series_dict]
    
    for lab in ordered_labels:
        x, y_mean, y_std, color = series_dict[lab]
        if y_mean is None or len(y_mean) == 0:
            continue
        
        # 绘制均值曲线
        ax.plot(x, y_mean, label=lab, color=color, linewidth=LINE_W)
        
        # 绘制标准差阴影
        ax.fill_between(x, y_mean - y_std, y_mean + y_std, 
                        color=color, alpha=0.2, linewidth=0)
    
    ax.set_title(title, fontsize=TITLE_FS)
    style_axes(ax, xlabel=True, ylabel=ylabel, letter=letter)

def unified_legend(fig):
    # 固定顺序：MLP, GNN, Transformer, GNN+Transformer
    fixed_order = ["MLP", "GNN", "Transformer", "GNN+Transformer"]
    
    handles = []
    for lab in fixed_order:
        (h,) = plt.plot([], [], color=COLOR_MAP.get(lab), label=lab, linewidth=LINE_W)
        handles.append(h)
    fig.legend(handles, fixed_order, loc="center left",
               bbox_to_anchor=(0.88, 0.5), frameon=True, fontsize=LEGEND_FS)

def save_all(fig, out_png_base):
    png = out_png_base + ".png"
    pdf = out_png_base + ".pdf"
    svg = out_png_base + ".svg"
    fig.savefig(png)
    fig.savefig(pdf)
    fig.savefig(svg)
    plt.close(fig)
    print(f"[Saved] {png}\n        {pdf}\n        {svg}")

def save_statistics(model_groups, out_dir):
    """保存每个模型的统计结果（均值和标准差）到CSV文件"""
    import csv
    
    # 要统计的指标
    metrics = [
        "eval_rewards", "eval_success", "eval_tts", "eval_min_distance",
        "eval_rmse", "eval_max_dev", "eval_end_err", "eval_path_len_exec"
    ]
    
    stats_file = os.path.join(out_dir, "model_statistics.csv")
    
    with open(stats_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Metric", "Mean", "Std", "Final_Mean", "Final_Std"])
        
        fixed_order = ["MLP", "GNN", "Transformer", "GNN+Transformer"]
        
        for model_name in fixed_order:
            if model_name not in model_groups:
                continue
            
            print(f"\n{model_name}:")
            for metric in metrics:
                series = build_series_with_seeds({model_name: model_groups[model_name]}, metric)
                
                if model_name in series:
                    x, y_mean, y_std, _ = series[model_name]
                    
                    # 计算整体均值和标准差（忽略NaN）
                    overall_mean = np.nanmean(y_mean)
                    overall_std = np.nanmean(y_std)
                    
                    # 获取最后一个非NaN值的均值和标准差
                    valid_indices = ~np.isnan(y_mean)
                    if np.any(valid_indices):
                        final_mean = y_mean[valid_indices][-1]
                        final_std = y_std[valid_indices][-1]
                    else:
                        final_mean = np.nan
                        final_std = np.nan
                    
                    writer.writerow([
                        model_name, 
                        metric, 
                        f"{overall_mean:.6f}", 
                        f"{overall_std:.6f}",
                        f"{final_mean:.6f}",
                        f"{final_std:.6f}"
                    ])
                    
                    print(f"  {metric}: Mean={overall_mean:.4f}, Std={overall_std:.4f}, Final={final_mean:.4f}±{final_std:.4f}")
    
    print(f"\n[Statistics saved] {stats_file}")
    return stats_file

def parse_model_groups(model_group_strs):
    """
    解析格式: "ModelName:path1,path2,path3"
    返回: {"ModelName": [path1, path2, path3], ...}
    """
    model_groups = {}
    for group_str in model_group_strs:
        if ":" not in group_str:
            print(f"Warning: Invalid format '{group_str}', skipping...")
            continue
        model_name, paths_str = group_str.split(":", 1)
        paths = [p.strip() for p in paths_str.split(",")]
        model_groups[model_name] = paths
    return model_groups

def main():
    ap = argparse.ArgumentParser(description="科研风格：多模型多种子对比图（带标准差阴影）")
    ap.add_argument("--model_groups", type=str, nargs="+", required=True, 
                    help="格式: ModelName:path1,path2,path3 每个模型提供多个种子的路径")
    ap.add_argument("--out", type=str, default=None, 
                    help="输出目录（默认 plots/compare_seeds_时间戳）")
    args = ap.parse_args()

    model_groups = parse_model_groups(args.model_groups)
    
    if not model_groups:
        print("Error: No valid model groups provided!")
        return

    out_dir = args.out or os.path.join("plots", f"compare_seeds_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Model groups:")
    for name, paths in model_groups.items():
        print(f"  {name}: {len(paths)} seeds")
        for p in paths:
            print(f"    - {p}")

    # -------- 图 1：Reward / Success / TTS / Min Distance --------
    s_reward = build_series_with_seeds(model_groups, "eval_rewards")
    s_succ   = build_series_with_seeds(model_groups, "eval_success")
    s_tts    = build_series_with_seeds(model_groups, "eval_tts")
    s_mind   = build_series_with_seeds(model_groups, "eval_min_distance")

    fig1 = plt.figure(figsize=(12.6, 6.2))
    gs1 = fig1.add_gridspec(2, 2, left=0.07, right=0.84, top=0.96, bottom=0.09, hspace=0.35, wspace=0.28)
    ax11 = fig1.add_subplot(gs1[0, 0]); plot_panel(ax11, s_reward, "Eval Reward (avg)", "Value", "a")
    ax12 = fig1.add_subplot(gs1[0, 1]); plot_panel(ax12, s_succ,  "Eval Success Rate (avg)", "Value", "b")
    ax13 = fig1.add_subplot(gs1[1, 0]); plot_panel(ax13, s_tts,   "Eval Time To Success (avg)", "Value", "c")
    ax14 = fig1.add_subplot(gs1[1, 1]); plot_panel(ax14, s_mind,  "Eval Min Distance (avg)", "Value", "d")
    unified_legend(fig1)
    save_all(fig1, os.path.join(out_dir, "compare_basic"))

    # -------- 图 2：RMSE / MaxDev / End-Point / Path Length --------
    s_rmse  = build_series_with_seeds(model_groups, "eval_rmse")
    s_maxd  = build_series_with_seeds(model_groups, "eval_max_dev")
    s_end   = build_series_with_seeds(model_groups, "eval_end_err")
    s_plen  = build_series_with_seeds(model_groups, "eval_path_len_exec")

    fig2 = plt.figure(figsize=(12.6, 6.2))
    gs2 = fig2.add_gridspec(2, 2, left=0.07, right=0.84, top=0.96, bottom=0.09, hspace=0.35, wspace=0.28)
    ax21 = fig2.add_subplot(gs2[0, 0]); plot_panel(ax21, s_rmse, "Eval RMSE (avg)", "Value", "a")
    ax22 = fig2.add_subplot(gs2[0, 1]); plot_panel(ax22, s_maxd, "Eval Max Deviation (avg)", "Value", "b")
    ax23 = fig2.add_subplot(gs2[1, 0]); plot_panel(ax23, s_end,  "Eval End-Point Error (avg)", "Value", "c")
    ax24 = fig2.add_subplot(gs2[1, 1]); plot_panel(ax24, s_plen, "Eval Path Length (avg)", "Value", "d")
    unified_legend(fig2)
    save_all(fig2, os.path.join(out_dir, "compare_traj"))

    # -------- 保存统计结果 --------
    save_statistics(model_groups, out_dir)

    print(f"[Done] Output dir: {out_dir}")

if __name__ == "__main__":
    main()

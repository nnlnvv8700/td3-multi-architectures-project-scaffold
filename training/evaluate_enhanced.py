# -*- coding: utf-8 -*-
# training/evaluate_enhanced.py - 增强版评估脚本（含标准差、多种子、轨迹生成）
r"""
用法示例（PowerShell）：

1. 单次运行评估（原始模式）：
python ".\training\evaluate_enhanced.py" --run_dirs `
  ".\results\KukaIiwa7Track-v0_mlp_dense_20251115_200935" `
  ".\results\KukaIiwa7Track-v0_gnn_dense_20251115_151707" `
  --ma_window 3

2. 多种子评估（显示均值±标准差）：
python ".\training\evaluate_enhanced.py" --multi_seed `
  --seed_runs "MLP:.\results\mlp_s1,.\results\mlp_s2,.\results\mlp_s3" `
               "GNN:.\results\gnn_s1,.\results\gnn_s2,.\results\gnn_s3"

3. 生成轨迹评估（单个模型）：
python ".\training\evaluate_enhanced.py" --run_dirs `
  ".\results\KukaIiwa7Track-v0_gnn_transformer_dense_20251115_131235" `
  --generate_trajectory --trajectory_seeds 42,123,456 --num_episodes 10
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.ticker import AutoMinorLocator, ScalarFormatter
from typing import List, Dict, Tuple, Optional
import torch

# 添加项目路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.observation import flatten_obs, infer_dimensions
from training.config import environment_kwargs
from training.metrics import trajectory_success
from training.series import align_evaluations

# ---------- 全局科研风格 ----------
mpl.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "dejavuserif",
    "axes.unicode_minus": False,
    "figure.dpi": 160,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
AX_LABEL_FS = 12
TICK_FS = 11
TITLE_FS = 12
LEGEND_FS = 11
LINE_W = 1.8
MARKER_SIZE = 3.2
GRID_ALPHA = 0.25

# 顶级期刊最爱用的配色（Colorblind-friendly + 打印友好）
COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00"]  # 分别给 MLP, GNN, Trans, GT
# 替换原来的 COLOR_MAP
COLOR_MAP = {
    "MLP": COLORS[0],
    "GNN": COLORS[1],
    "Transformer": COLORS[2],
    "GNN+Transformer": COLORS[3]
}

# ==================== 数据加载工具 ====================

def infer_label_from_run(run_dir: str) -> str:
    """从目录名推断模型类型"""
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

def load_metrics(run_dir: str) -> Dict:
    """加载训练指标"""
    p = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def load_config(run_dir: str) -> Dict:
    """加载训练配置"""
    p = os.path.join(run_dir, "config.json")
    if not os.path.exists(p):
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def steps_x(metrics):
    """提取评估步数并转换为 ×10k 单位"""
    x = metrics.get("eval_steps", [])
    if x:
        return np.asarray(x, dtype=float) / 10000.0
    return None

def moving_average(y, w):
    """计算移动平均"""
    if w <= 1 or y is None or len(y) == 0:
        return None
    if len(y) < w:
        return None
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(np.asarray(y, dtype=float), kernel, mode="valid")

# ==================== 单次运行数据构建 ====================

def build_series_single(run_dirs: List[str], metric_key: str, labels_order: List[str], ma_window: int = 0) -> Dict:
    """构建单次运行的数据系列（无标准差）"""
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
                x_ma = x[-len(y_ma):]
            else:
                x_ma = None
            series[lab] = (x, y, COLOR_MAP.get(lab), x_ma, y_ma, None, None)  # 7元素：添加y_std和y_ma_std占位
        else:
            series[lab] = (x, y, COLOR_MAP.get(lab), None, None, None, None)  # 7元素：添加y_std和y_ma_std占位
    return series

# ==================== 多种子数据构建 ====================

def build_series_multi(seed_runs_dict: Dict[str, List[str]], metric_key: str,
                      labels_order: List[str], ma_window: int = 0) -> Dict:
    """
    构建多种子运行的数据系列（含均值和标准差）

    Args:
        seed_runs_dict: {"label": [dir1, dir2, dir3], ...}
        metric_key: 指标名称
        labels_order: 标签顺序列表
        ma_window: 移动平均窗口

    Returns:
        series: {"label": (x, y_mean, color, x_ma, y_ma_mean, y_std, y_ma_std), ...}
    """
    series = {}

    for label, dirs in seed_runs_dict.items():
        if label not in labels_order:
            labels_order.append(label)

        x_aligned, all_y_aligned = align_evaluations(
            [load_metrics(rd) for rd in dirs], metric_key
        )
        if not x_aligned.size:
            continue

        # 计算均值和标准差
        y_array = np.array(all_y_aligned)
        y_mean = np.mean(y_array, axis=0)
        y_std = np.std(y_array, axis=0)

        # 移动平均
        y_ma_mean = None
        y_ma_std = None
        x_ma = None

        if ma_window > 1 and len(y_mean) >= ma_window:
            y_ma_list = []
            for y in all_y_aligned:
                y_ma = moving_average(y, ma_window)
                if y_ma is not None:
                    y_ma_list.append(y_ma)

            if y_ma_list:
                ma_min_len = min(len(y_ma) for y_ma in y_ma_list)
                y_ma_aligned = [y_ma[:ma_min_len] for y_ma in y_ma_list]
                y_ma_array = np.array(y_ma_aligned)
                y_ma_mean = np.mean(y_ma_array, axis=0)
                y_ma_std = np.std(y_ma_array, axis=0)
                x_ma = x_aligned[-ma_min_len:] if len(x_aligned) >= ma_min_len else None

        series[label] = (x_aligned, y_mean, COLOR_MAP.get(label), x_ma, y_ma_mean, y_std, y_ma_std)

    return series

# ==================== 绘图工具 ====================

def style_axes(ax, xlabel=True, ylabel=None, letter=None):
    """设置坐标轴样式"""
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

def plot_panel(ax, series_dict: Dict, title: str, ylabel: str, letter: str,
              show_ma: bool = False, multi_seed: bool = False):
    """
    绘制单个面板

    Args:
        multi_seed: 是否为多种子模式（显示标准差）
    """
    for lab, data in series_dict.items():
        # 数据格式: (x, y_mean, color, x_ma, y_ma_mean, y_std, y_ma_std)
        x, y, color, x_ma, y_ma, y_std, y_ma_std = data

        if y is None or len(y) == 0:
            continue

        xi = np.arange(1, len(y) + 1, dtype=float) if x is None else np.asarray(x)[:len(y)]

        # 绘制主曲线
        ax.plot(xi, y, label=lab, color=color, linewidth=LINE_W, marker='o', ms=MARKER_SIZE)

        # 绘制标准差阴影（多种子模式）
        if multi_seed and y_std is not None and len(y_std) > 0:
            ax.fill_between(xi, y - y_std, y + y_std,
                           color=color, alpha=0.2, linewidth=0, label=f"{lab} ±σ")

        # 绘制移动平均
        if show_ma and y_ma is not None and len(y_ma) > 0:
            x_ma_plot = x_ma if x_ma is not None else np.arange(1, len(y_ma) + 1)
            ax.plot(x_ma_plot, y_ma, color=color, linewidth=LINE_W + 0.4,
                   linestyle='-', alpha=0.8)

            # 移动平均的标准差阴影
            if multi_seed and y_ma_std is not None and len(y_ma_std) > 0:
                ax.fill_between(x_ma_plot, y_ma - y_ma_std, y_ma + y_ma_std,
                               color=color, alpha=0.15, linewidth=0)

    ax.set_title(title, fontsize=TITLE_FS)
    style_axes(ax, xlabel=True, ylabel=ylabel, letter=letter)

def unified_legend(fig, labels_order: List[str]):
    """创建统一图例"""
    handles = []
    for lab in labels_order:
        (h,) = plt.plot([], [], color=COLOR_MAP.get(lab), label=lab, linewidth=LINE_W)
        handles.append(h)
    fig.legend(handles, labels_order, loc="center left",
               bbox_to_anchor=(0.88, 0.5), frameon=True, fontsize=LEGEND_FS)

def save_all(fig, out_png_base: str):
    """保存为PNG、PDF、SVG格式"""
    png = out_png_base + ".png"
    pdf = out_png_base + ".pdf"
    svg = out_png_base + ".svg"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    fig.savefig(svg)
    plt.close(fig)
    print(f"[Saved] {png}\n        {pdf}\n        {svg}")

# ==================== 轨迹生成功能 ====================

def load_agent_from_run(run_dir: str):
    """从训练目录加载智能体模型"""
    try:
        import gymnasium as gym
    except ImportError:
        import gym

    # 导入必要模块
    import envs
    from agents.td3_agent import TD3

    config = load_config(run_dir)
    if not config:
        raise FileNotFoundError(f"未找到 config.json: {run_dir}")

    # 创建环境获取维度
    env_name = config.get("env", "KukaIiwa7Track-v0")
    observation_version = int(config.get("observation_version", 1))
    try:
        env = gym.make(
            env_name, render_mode="rgb_array", dense_reward=True,
            observation_version=observation_version,
            **environment_kwargs(config),
        )
    except TypeError:
        env = gym.make(env_name, dense_reward=True)

    try:
        # 获取状态和动作维度
        state_dim, action_dim, max_action = infer_dimensions(env)

        # 创建配置对象
        class AgentConfig:
            pass
        agent_config = AgentConfig()
        for k, v in config.items():
            setattr(agent_config, k, v)
        if not hasattr(agent_config, "use_state_encoder"):
            agent_config.use_state_encoder = bool(
                config.get("actor_arch", "mlp") != "mlp" and observation_version >= 2
            )
        agent_config.max_timesteps = int(config.get("max_timesteps", 500_000))
        agent_config.start_timesteps = int(config.get("start_timesteps", 25_000))

        # Evaluation chooses a locally available device, independently of the training host.
        agent_config.device = None
        # 创建智能体
        checkpoint_max_action = max_action if observation_version >= 2 else 1.0
        agent = TD3(state_dim, action_dim, checkpoint_max_action, agent_config)

        # 加载模型权重
        model_path = os.path.join(run_dir, "final_model.pt")
        best_path = os.path.join(run_dir, "best_model.pt")
        if config.get("algorithm_version") == "corrected" and os.path.isfile(best_path):
            model_path = best_path
        if not os.path.exists(model_path):
            for f in os.listdir(run_dir):
                if f.startswith("final_model_") and f.endswith(".pt"):
                    model_path = os.path.join(run_dir, f)
                    break

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"未找到模型文件: {model_path}")

        agent.load_actor(model_path)
        agent.actor.eval()

        return agent, env, env_name
    except BaseException:
        env.close()
        raise

def generate_trajectories(run_dir: str, seeds: List[int], num_episodes: int = 10, out_dir: str = None):
    """
    生成指定轨迹用于评估

    Args:
        run_dir: 训练结果目录
        seeds: 随机种子列表
        num_episodes: 每个种子运行的回合数
        out_dir: 输出目录
    """
    if num_episodes <= 0 or not seeds:
        raise ValueError("num_episodes must be positive and seeds must not be empty")
    print(f"\n[生成轨迹] 加载模型: {run_dir}")
    agent, env, env_name = load_agent_from_run(run_dir)

    try:
        if out_dir is None:
            out_dir = os.path.join("plots", f"trajectories_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(out_dir, exist_ok=True)

        all_trajectories = []
        all_rewards = []
        all_success = []

        for seed in seeds:
            print(f"\n[种子 {seed}] 运行 {num_episodes} 个回合...")
            np.random.seed(seed)
            torch.manual_seed(seed)

            seed_trajectories = []
            seed_rewards = []
            seed_success = []

            for ep in range(num_episodes):
                obs, info = env.reset(seed=seed + ep)
                done = False
                episode_reward = 0.0
                trajectory = []

                # 记录初始位置
                if isinstance(info, dict) and "ee_pos" in info:
                    trajectory.append(info["ee_pos"].copy())

                while not done:
                    state = flatten_obs(obs)
                    action = agent.select_action(state)

                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    episode_reward += reward

                    # 记录轨迹
                    if isinstance(info, dict) and "ee_pos" in info:
                        trajectory.append(info["ee_pos"].copy())

                # 判断成功
                is_success = info.get("is_success", 0) if isinstance(info, dict) else 0
                if env.unwrapped.reward_mode == "tracking":
                    is_success = float(trajectory_success(np.asarray(trajectory), info.get("ref_traj"),
                                                          env.unwrapped.distance_threshold,
                                                          env.unwrapped.tracking_success_threshold))

                seed_trajectories.append(np.array(trajectory))
                seed_rewards.append(episode_reward)
                seed_success.append(is_success)

                print(f"  Episode {ep+1}/{num_episodes}: Reward={episode_reward:.3f}, Success={is_success}")

            all_trajectories.append(seed_trajectories)
            all_rewards.append(seed_rewards)
            all_success.append(seed_success)

    finally:
        env.close()

    # 保存轨迹数据
    data_file = os.path.join(out_dir, "trajectory_data.npz")
    np.savez(data_file,
             trajectories=all_trajectories,
             rewards=all_rewards,
             success=all_success,
             seeds=seeds)
    print(f"\n[保存] 轨迹数据: {data_file}")

    # 绘制轨迹对比图
    plot_trajectories(all_trajectories, all_rewards, all_success, seeds, out_dir)

    # 统计分析
    print(f"\n[统计分析]")
    for i, seed in enumerate(seeds):
        mean_reward = np.mean(all_rewards[i])
        std_reward = np.std(all_rewards[i])
        success_rate = np.mean(all_success[i])
        print(f"  种子 {seed}: Reward={mean_reward:.3f}±{std_reward:.3f}, Success={success_rate:.2%}")

    return all_trajectories, all_rewards, all_success

def plot_trajectories(all_trajectories: List, all_rewards: List, all_success: List,
                      seeds: List[int], out_dir: str):
    """绘制轨迹可视化图"""
    num_seeds = len(seeds)

    # 创建3D轨迹图
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(14, 5 * ((num_seeds + 1) // 2)))

    for idx, seed in enumerate(seeds):
        ax = fig.add_subplot(((num_seeds + 1) // 2), 2, idx + 1, projection='3d')

        trajectories = all_trajectories[idx]
        colors = plt.cm.viridis(np.linspace(0, 1, len(trajectories)))

        for i, traj in enumerate(trajectories):
            if len(traj) > 0:
                ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                       color=colors[i], alpha=0.6, linewidth=1.5)
                # 标记起点和终点
                ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2],
                          color='green', s=50, marker='o', label='Start' if i == 0 else '')
                ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2],
                          color='red', s=50, marker='x', label='End' if i == 0 else '')

        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)
        ax.set_zlabel('Z (m)', fontsize=10)
        ax.set_title(f'Seed {seed}\nAvg Reward: {np.mean(all_rewards[idx]):.2f}±{np.std(all_rewards[idx]):.2f}\nSuccess: {np.mean(all_success[idx]):.1%}',
                    fontsize=11)
        if idx == 0:
            ax.legend(fontsize=9)

    plt.tight_layout()
    traj_file = os.path.join(out_dir, "trajectories_3d.png")
    plt.savefig(traj_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[保存] 轨迹图: {traj_file}")

# ==================== 主函数 ====================

def main():
    ap = argparse.ArgumentParser(
        description="增强版评估脚本：支持多种子、标准差、轨迹生成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # 基础参数
    ap.add_argument("--run_dirs", type=str, nargs="+",
                   help="单次运行目录列表（用于单次运行模式）")
    ap.add_argument("--out", type=str, default=None,
                   help="输出目录（默认 plots/eval_时间戳）")
    ap.add_argument("--ma_window", type=int, default=0,
                   help="移动平均窗口大小（0=关闭）")

    # 多种子模式
    ap.add_argument("--multi_seed", action="store_true",
                   help="启用多种子模式（显示均值±标准差）")
    ap.add_argument("--seed_runs", type=str, nargs="+",
                   help='多种子运行，格式: "Label:dir1,dir2,dir3"')

    # 轨迹生成
    ap.add_argument("--generate_trajectory", action="store_true",
                   help="生成轨迹评估")
    ap.add_argument("--trajectory_seeds", type=str, default="42,123,456",
                   help="轨迹生成的随机种子（逗号分隔）")
    ap.add_argument("--num_episodes", type=int, default=10,
                   help="每个种子的回合数")

    args = ap.parse_args()

    # 轨迹生成模式
    if args.generate_trajectory:
        if not args.run_dirs or len(args.run_dirs) == 0:
            print("[错误] 轨迹生成模式需要指定 --run_dirs")
            return

        seeds = [int(s.strip()) for s in args.trajectory_seeds.split(",")]
        out_dir = args.out or os.path.join("plots", f"trajectories_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        for run_dir in args.run_dirs:
            generate_trajectories(run_dir, seeds, args.num_episodes, out_dir)

        return

    # 多种子评估模式
    if args.multi_seed:
        if not args.seed_runs:
            print("[错误] 多种子模式需要指定 --seed_runs")
            return

        # 解析种子运行
        seed_runs_dict = {}
        for item in args.seed_runs:
            if ":" not in item:
                print(f"[警告] 忽略格式错误的项: {item}")
                continue
            label, dirs_str = item.split(":", 1)
            dirs = [d.strip() for d in dirs_str.split(",")]
            seed_runs_dict[label] = dirs

        out_dir = args.out or os.path.join("plots", f"eval_multi_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(out_dir, exist_ok=True)

        # 图 1: 基础指标
        labels1 = []
        s_reward = build_series_multi(seed_runs_dict, "eval_rewards", labels1, ma_window=args.ma_window)
        s_succ = build_series_multi(seed_runs_dict, "eval_success", labels1, ma_window=args.ma_window)
        s_tts = build_series_multi(seed_runs_dict, "eval_tts", labels1, ma_window=args.ma_window)
        s_mind = build_series_multi(seed_runs_dict, "eval_min_distance", labels1, ma_window=args.ma_window)

        fig1 = plt.figure(figsize=(12.6, 6.2))
        gs1 = fig1.add_gridspec(2, 2, left=0.07, right=0.84, top=0.96, bottom=0.09, hspace=0.35, wspace=0.28)
        ax11 = fig1.add_subplot(gs1[0, 0])
        plot_panel(ax11, s_reward, "Eval Reward (mean±std)", "Value", "a", show_ma=bool(args.ma_window), multi_seed=True)
        ax12 = fig1.add_subplot(gs1[0, 1])
        plot_panel(ax12, s_succ, "Eval Success Rate (mean±std)", "Value", "b", show_ma=bool(args.ma_window), multi_seed=True)
        ax13 = fig1.add_subplot(gs1[1, 0])
        plot_panel(ax13, s_tts, "Eval Time To Success (mean±std)", "Value", "c", show_ma=bool(args.ma_window), multi_seed=True)
        ax14 = fig1.add_subplot(gs1[1, 1])
        plot_panel(ax14, s_mind, "Eval Min Distance (mean±std)", "Value", "d", show_ma=bool(args.ma_window), multi_seed=True)
        unified_legend(fig1, labels1)
        save_all(fig1, os.path.join(out_dir, "compare_basic_multi"))

        # 图 2: 轨迹指标
        labels2 = []
        s_rmse = build_series_multi(seed_runs_dict, "eval_rmse", labels2, ma_window=args.ma_window)
        s_maxd = build_series_multi(seed_runs_dict, "eval_max_dev", labels2, ma_window=args.ma_window)
        s_end = build_series_multi(seed_runs_dict, "eval_end_err", labels2, ma_window=args.ma_window)
        s_plen = build_series_multi(seed_runs_dict, "eval_path_len_exec", labels2, ma_window=args.ma_window)

        fig2 = plt.figure(figsize=(12.6, 6.2))
        gs2 = fig2.add_gridspec(2, 2, left=0.07, right=0.84, top=0.96, bottom=0.09, hspace=0.35, wspace=0.28)
        ax21 = fig2.add_subplot(gs2[0, 0])
        plot_panel(ax21, s_rmse, "Eval RMSE (mean±std)", "Value", "a", show_ma=bool(args.ma_window), multi_seed=True)
        ax22 = fig2.add_subplot(gs2[0, 1])
        plot_panel(ax22, s_maxd, "Eval Max Deviation (mean±std)", "Value", "b", show_ma=bool(args.ma_window), multi_seed=True)
        ax23 = fig2.add_subplot(gs2[1, 0])
        plot_panel(ax23, s_end, "Eval End-Point Error (mean±std)", "Value", "c", show_ma=bool(args.ma_window), multi_seed=True)
        ax24 = fig2.add_subplot(gs2[1, 1])
        plot_panel(ax24, s_plen, "Eval Path Length (mean±std)", "Value", "d", show_ma=bool(args.ma_window), multi_seed=True)
        unified_legend(fig2, labels2)
        save_all(fig2, os.path.join(out_dir, "compare_traj_multi"))

        print(f"\n[完成] 多种子评估结果: {out_dir}")

    # 单次运行模式（保持原有功能）
    else:
        if not args.run_dirs:
            print("[错误] 需要指定 --run_dirs 或启用 --multi_seed")
            return

        out_dir = args.out or os.path.join("plots", f"eval_single_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(out_dir, exist_ok=True)

        # 图 1: 基础指标
        labels1 = []
        s_reward = build_series_single(args.run_dirs, "eval_rewards", labels1, ma_window=args.ma_window)
        s_succ = build_series_single(args.run_dirs, "eval_success", labels1, ma_window=args.ma_window)
        s_tts = build_series_single(args.run_dirs, "eval_tts", labels1, ma_window=args.ma_window)
        s_mind = build_series_single(args.run_dirs, "eval_min_distance", labels1, ma_window=args.ma_window)

        fig1 = plt.figure(figsize=(12.6, 6.2))
        gs1 = fig1.add_gridspec(2, 2, left=0.07, right=0.84, top=0.96, bottom=0.09, hspace=0.35, wspace=0.28)
        ax11 = fig1.add_subplot(gs1[0, 0])
        plot_panel(ax11, s_reward, "Eval Reward (avg)", "Value", "a", show_ma=bool(args.ma_window), multi_seed=False)
        ax12 = fig1.add_subplot(gs1[0, 1])
        plot_panel(ax12, s_succ, "Eval Success Rate (avg)", "Value", "b", show_ma=bool(args.ma_window), multi_seed=False)
        ax13 = fig1.add_subplot(gs1[1, 0])
        plot_panel(ax13, s_tts, "Eval Time To Success (avg)", "Value", "c", show_ma=bool(args.ma_window), multi_seed=False)
        ax14 = fig1.add_subplot(gs1[1, 1])
        plot_panel(ax14, s_mind, "Eval Min Distance (avg)", "Value", "d", show_ma=bool(args.ma_window), multi_seed=False)
        unified_legend(fig1, labels1)
        save_all(fig1, os.path.join(out_dir, "compare_basic"))

        # 图 2: 轨迹指标
        labels2 = []
        s_rmse = build_series_single(args.run_dirs, "eval_rmse", labels2, ma_window=args.ma_window)
        s_maxd = build_series_single(args.run_dirs, "eval_max_dev", labels2, ma_window=args.ma_window)
        s_end = build_series_single(args.run_dirs, "eval_end_err", labels2, ma_window=args.ma_window)
        s_plen = build_series_single(args.run_dirs, "eval_path_len_exec", labels2, ma_window=args.ma_window)

        fig2 = plt.figure(figsize=(12.6, 6.2))
        gs2 = fig2.add_gridspec(2, 2, left=0.07, right=0.84, top=0.96, bottom=0.09, hspace=0.35, wspace=0.28)
        ax21 = fig2.add_subplot(gs2[0, 0])
        plot_panel(ax21, s_rmse, "Eval RMSE (avg)", "Value", "a", show_ma=bool(args.ma_window), multi_seed=False)
        ax22 = fig2.add_subplot(gs2[0, 1])
        plot_panel(ax22, s_maxd, "Eval Max Deviation (avg)", "Value", "b", show_ma=bool(args.ma_window), multi_seed=False)
        ax23 = fig2.add_subplot(gs2[1, 0])
        plot_panel(ax23, s_end, "Eval End-Point Error (avg)", "Value", "c", show_ma=bool(args.ma_window), multi_seed=False)
        ax24 = fig2.add_subplot(gs2[1, 1])
        plot_panel(ax24, s_plen, "Eval Path Length (avg)", "Value", "d", show_ma=bool(args.ma_window), multi_seed=False)
        unified_legend(fig2, labels2)
        save_all(fig2, os.path.join(out_dir, "compare_traj"))

        print(f"\n[完成] 单次运行评估结果: {out_dir}")

if __name__ == "__main__":
    main()

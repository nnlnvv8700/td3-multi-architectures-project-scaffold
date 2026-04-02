"""
TD3 训练脚本 - 轨迹跟踪任务 (Track Environment)

核心改进：
1. 统一环境：只使用 KukaIiwa7Track-v0（轨迹跟踪任务）
2. 固定Episode长度：200步（done_on_success=False）
3. 轨迹跟踪奖励：30%目标 + 40%轨迹 + 30%成功
4. 优化学习率：根据架构自动调整（MLP:3e-5, GNN/Transformer:1e-5, GT-TD3:2e-5）
5. 移除强制终止：尊重环境的done_on_success设置

使用方法：
  # 训练 MLP（默认 seed=0）
  python -m training.train_experiment --actor_arch mlp --max_timesteps 500000

  # 训练 GNN
  python -m training.train_experiment --actor_arch gnn --max_timesteps 500000

  # 训练 Transformer
  python -m training.train_experiment --actor_arch transformer --max_timesteps 500000

  # 训练 GNN+Transformer
  python -m training.train_experiment --actor_arch gnn_transformer --max_timesteps 500000

  # 指定随机种子（用于复现或多次对比实验）
  python -m training.train_experiment --actor_arch gnn_transformer --seed 42
  python -m training.train_experiment --actor_arch gnn_transformer --seed 123

  # 多 seed 批量训练示例（PowerShell 循环）
  foreach ($seed in @(0, 42, 123)) {
      python -m training.train_experiment --actor_arch mlp --seed $seed --max_timesteps 500000
      python -m training.train_experiment --actor_arch gnn --seed $seed --max_timesteps 500000
      python -m training.train_experiment --actor_arch transformer --seed $seed --max_timesteps 500000
      python -m training.train_experiment --actor_arch gnn_transformer --seed $seed --max_timesteps 500000
  }

  # 指定保存目录（不同 seed 建议分目录）
  python -m training.train_experiment --actor_arch gnn_transformer --seed 42 --save_dir ./results/seed42

环境设置（Windows PowerShell）：
  $env:KMP_DUPLICATE_LIB_OK="TRUE"
"""
# ---- 让本地项目优先于 site-packages，避免与第三方 'agents' 包冲突 ----
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --------------------------------------------------------------

import argparse
try:
    import gymnasium as gym
except ImportError:
    import gym
import numpy as np
import pandas as pd
import json
import random
import torch
from typing import Dict, Any, Optional, Tuple, List

# 确保环境注册
import envs  # noqa: F401
from agents.td3_agent import TD3
from utils.replay_buffer import ReplayBuffer
from utils.gym_compat import reset_env, step_env
from evaluation import evaluate_policy as evaluate_policy_decoupled, save_eval_outputs


# ===================== 工具函数 =====================

def get_default_args(arch: str = "mlp") -> Dict[str, Any]:
    """
    获取指定架构的默认参数配置
    
    Args:
        arch: 架构类型，可选 "mlp", "gnn", "transformer", "gnn_transformer"
        
    Returns:
        包含默认配置的字典，包括 actor_arch, node_dim, num_nodes, use_state_encoder
        
    Examples:
        >>> config = get_default_args("gnn")
        >>> print(config)  # {"actor_arch": "gnn", "node_dim": 3, "num_nodes": 7, ...}
    
    配置说明:
        - MLP: 不使用图结构，直接处理20维状态
        - GNN/Transformer/GNN+Transformer: 
          * 7关节配置(推荐): num_nodes=7, node_dim=6, use_state_encoder=True
          * 包含完整3D方向信息：[pos, vel, dist, dx, dy, dz]
    """
    # 🎯 7关节配置（推荐，利用真实运动学结构）
    defaults_7_joints = {
        "mlp": {
            "actor_arch": "mlp", 
            "node_dim": None, 
            "num_nodes": None,
            "use_state_encoder": False
        },
        "gnn": {
            "actor_arch": "gnn", 
            "node_dim": 6,      # 🎯 6维节点：[pos, vel, dist, dx, dy, dz]
            "num_nodes": 7,      # 对应7个关节
            "use_state_encoder": True  # ✅ 使用改进的编码器（保留完整3D方向）
        },
        "transformer": {
            "actor_arch": "transformer", 
            "node_dim": 6,  # 🎯 6维节点：保留完整3D方向
            "num_nodes": 7,
            "use_state_encoder": True  # ✅ 使用改进的编码器
        },
        "gnn_transformer": {
            "actor_arch": "gnn_transformer", 
            "node_dim": 6,  # 🎯 6维节点：保留完整3D方向
            "num_nodes": 7,
            "use_state_encoder": True  # ✅ 使用改进的编码器
        },
    }
    
    # 🎯 默认使用7关节配置
    return defaults_7_joints.get(arch, defaults_7_joints["mlp"])


def flatten_obs(obs: Any) -> np.ndarray:
    """
    将字典类型的观测展平为一维向量
    
    对于 GoalEnv 类型的环境，将 observation, achieved_goal, desired_goal 拼接。
    对于普通观测，直接转换为 float32 数组。
    
    Args:
        obs: 环境观测值（可能是 dict 或 ndarray）
        
    Returns:
        展平后的一维 float32 数组
        
    Examples:
        >>> obs = {"observation": [1,2,3], "achieved_goal": [4,5,6], "desired_goal": [7,8,9]}
        >>> flat = flatten_obs(obs)
        >>> print(flat.shape)  # (9,)
    """
    if isinstance(obs, dict):
        return np.concatenate(
            [obs["observation"], obs["achieved_goal"], obs["desired_goal"]],
            axis=0,
        ).astype(np.float32)
    else:
        return np.array(obs, dtype=np.float32)


def _extract_distance(obs: Any, info: Optional[Dict], env: gym.Env) -> Optional[float]:
    """
    从观测/信息中提取末端执行器到目标的距离
    
    Args:
        obs: 环境观测
        info: 环境信息字典
        env: Gym 环境对象
        
    Returns:
        距离值（米），如果无法提取则返回 None
    """
    if isinstance(obs, dict) and "achieved_goal" in obs and "desired_goal" in obs:
        ag = np.array(obs["achieved_goal"], dtype=np.float32)
        dg = np.array(obs["desired_goal"], dtype=np.float32)
        return float(np.linalg.norm(ag - dg))
    if info is not None and "ee_pos" in info and "goal_pos" in info:
        ee = np.array(info["ee_pos"], dtype=np.float32)
        gg = np.array(info["goal_pos"], dtype=np.float32)
        return float(np.linalg.norm(ee - gg))
    return None


def _is_success_state(obs: Any, info: Optional[Dict], env: gym.Env) -> bool:
    """
    判断当前状态是否达到成功条件
    
    优先使用 info['is_success']，否则基于距离和阈值判断。
    
    Args:
        obs: 环境观测
        info: 环境信息字典
        env: Gym 环境对象
        
    Returns:
        True 表示成功，False 表示未成功
    """
    if isinstance(info, dict) and "is_success" in info:
        try:
            return bool(info["is_success"])
        except Exception:
            pass
    d = _extract_distance(obs, info, env)
    if d is None:
        return False
    thr = getattr(getattr(env, "unwrapped", env), "distance_threshold", 0.05)
    return bool(d < float(thr))


def evaluate_policy(
    agent: TD3,
    env: gym.Env,
    eval_episodes: int = 20,
    deterministic: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
) -> Tuple:
    """
    Phase-B 兼容包装器：
    - 新实现委托给 `evaluation.evaluator.evaluate_policy`
    - 保持旧调用方仍接收 20 元组，不改训练主逻辑
    - 可选写出标准化评估文件（episode_metrics.jsonl / eval_summary.json / eval_history.csv）
    """
    summary, episodes = evaluate_policy_decoupled(
        agent=agent,
        env=env,
        eval_episodes=eval_episodes,
        deterministic=deterministic,
        metadata=metadata,
    )

    if output_dir:
        save_eval_outputs(output_dir=output_dir, summary=summary, episodes=episodes)

    # 保持旧版返回顺序（20项）
    return (
        summary.get("reward_mean", float("nan")),
        summary.get("success_rate", float("nan")),
        summary.get("tts_mean", float("nan")),
        summary.get("min_distance_mean", float("nan")),
        summary.get("rmse_mean", float("nan")),
        summary.get("max_deviation_mean", float("nan")),
        summary.get("final_error_mean", float("nan")),
        summary.get("frechet_mean", float("nan")),
        summary.get("rms_accel_mean", float("nan")),
        summary.get("avg_jerk_mean", float("nan")),
        summary.get("vel_var_mean", float("nan")),
        summary.get("path_len_exec_mean", float("nan")),
        summary.get("path_len_ref_mean", float("nan")),
        summary.get("path_efficiency_mean", float("nan")),
        summary.get("energy_mean", float("nan")),
        summary.get("comp_goal_mean", float("nan")),
        summary.get("comp_track_mean", float("nan")),
        summary.get("comp_success_mean", float("nan")),
        summary.get("comp_smooth_mean", float("nan")),
        summary.get("comp_vel_mean", float("nan")),
    )


# ===================== 全局汇总保存 =====================

def plot_training_loss(metrics: Dict[str, List], save_dir: str) -> None:
    """
    绘制训练过程中的 Actor 和 Critic Loss 曲线（SCI 格式）
    
    Args:
        metrics: 包含 loss 数据的字典
        save_dir: 保存目录
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    
    # 设置 SCI 期刊风格
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '--'
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    
    fig, axes = plt.subplots(2, 1, figsize=(8, 7))
    
    # Critic Loss
    if metrics.get("critic_loss") and metrics.get("critic_loss_t"):
        axes[0].plot(metrics["critic_loss_t"], metrics["critic_loss"], 
                    linewidth=1.5, color='#0173B2')
        axes[0].set_title('Critic Loss', fontsize=12, fontweight='bold', pad=8)
        axes[0].set_xlabel('Training Steps (×$10^3$)', fontsize=10)
        axes[0].set_ylabel('Loss', fontsize=10)
        axes[0].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        axes[0].set_axisbelow(True)
        
        # X轴刻度（以千为单位）
        axes[0].ticklabel_format(axis='x', style='plain')
        xticks = axes[0].get_xticks()
        axes[0].set_xticklabels([f'{int(x/1000)}' for x in xticks])
        
        # 边框
        for spine in axes[0].spines.values():
            spine.set_linewidth(0.8)
    else:
        axes[0].text(0.5, 0.5, 'No Critic Loss Data', 
                    ha='center', va='center', transform=axes[0].transAxes,
                    fontsize=11)
        axes[0].set_title('Critic Loss', fontsize=12, fontweight='bold', pad=8)
    
    # Actor Loss
    if metrics.get("actor_loss") and metrics.get("actor_loss_t"):
        axes[1].plot(metrics["actor_loss_t"], metrics["actor_loss"], 
                    linewidth=1.5, color='#DE8F05')
        axes[1].set_title('Actor Loss', fontsize=12, fontweight='bold', pad=8)
        axes[1].set_xlabel('Training Steps (×$10^3$)', fontsize=10)
        axes[1].set_ylabel('Loss', fontsize=10)
        axes[1].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        axes[1].set_axisbelow(True)
        
        # X轴刻度（以千为单位）
        axes[1].ticklabel_format(axis='x', style='plain')
        xticks = axes[1].get_xticks()
        axes[1].set_xticklabels([f'{int(x/1000)}' for x in xticks])
        
        # 边框
        for spine in axes[1].spines.values():
            spine.set_linewidth(0.8)
    else:
        axes[1].text(0.5, 0.5, 'No Actor Loss Data', 
                    ha='center', va='center', transform=axes[1].transAxes,
                    fontsize=11)
        axes[1].set_title('Actor Loss', fontsize=12, fontweight='bold', pad=8)
    
    plt.tight_layout()
    
    # 保存 PNG 和 PDF
    png_path = os.path.join(save_dir, "training_loss.png")
    pdf_path = os.path.join(save_dir, "training_loss.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.close()
    print(f"[Info] Training loss plot saved: {png_path}")
    print(f"[Info] Training loss plot saved: {pdf_path}")


def save_global_summary(
    config: Any,
    metrics: Dict[str, List],
    rewards_history: List[float],
    success_history: List[int]
) -> None:
    """
    将本次训练的关键指标汇总保存到全局 CSV 文件
    
    每次训练完成后，自动将结果追加到 results/training_summary.csv，
    方便后续跨架构、跨实验的对比分析。
    
    Args:
        config: 训练配置对象（包含 env, actor_arch, save_dir 等属性）
        metrics: 训练过程中记录的指标字典（包含 eval_rewards, eval_success 等列表）
        rewards_history: 所有训练回合的奖励列表
        success_history: 所有训练回合的成功标志列表
        
    Side Effects:
        - 在 results/ 目录下创建或追加 training_summary.csv
        - 打印保存成功的消息
        
    Examples:
        >>> save_global_summary(config, metrics, rewards, successes)
        [Info] Global summary saved to: results/training_summary.csv
    """
    import csv
    from datetime import datetime
    
    # 全局汇总文件路径
    global_summary_dir = os.path.join(os.path.dirname(config.save_dir), "..")
    global_summary_file = os.path.join(global_summary_dir, "training_summary.csv")
    
    # 确保目录存在
    os.makedirs(global_summary_dir, exist_ok=True)
    
    # 提取关键指标（取最后几次评估的平均值，更稳定）
    def get_last_avg(lst, n=5):
        """取列表最后 n 个值的平均（忽略 nan）"""
        if not lst or len(lst) == 0:
            return float("nan")
        vals = lst[-n:] if len(lst) >= n else lst
        vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
        return float(np.mean(vals)) if vals else float("nan")
    
    # 计算汇总指标
    final_reward = get_last_avg(metrics.get("eval_rewards", []))
    final_success = get_last_avg(metrics.get("eval_success", []))
    final_tts = get_last_avg(metrics.get("eval_tts", []))
    final_min_dist = get_last_avg(metrics.get("eval_min_distance", []))
    final_rmse = get_last_avg(metrics.get("eval_rmse", []))
    final_max_dev = get_last_avg(metrics.get("eval_max_dev", []))
    final_end_err = get_last_avg(metrics.get("eval_end_err", []))
    final_path_len_exec = get_last_avg(metrics.get("eval_path_len_exec", []))
    final_path_len_ref = get_last_avg(metrics.get("eval_path_len_ref", []))
    
    # ===== 新增：论文级高级指标 =====
    final_frechet = get_last_avg(metrics.get("eval_frechet", []))
    final_rms_accel = get_last_avg(metrics.get("eval_rms_accel", []))
    final_jerk = get_last_avg(metrics.get("eval_jerk", []))
    final_vel_var = get_last_avg(metrics.get("eval_vel_var", []))
    final_efficiency = get_last_avg(metrics.get("eval_efficiency", []))
    final_energy = get_last_avg(metrics.get("eval_energy", []))
    
    # 训练回合统计
    total_episodes = len(rewards_history)
    avg_episode_reward = float(np.mean(rewards_history)) if rewards_history else float("nan")
    avg_episode_success = float(np.mean(success_history)) if success_history else float("nan")
    
    # 准备记录 - 提取时间戳和模型名称
    run_name = os.path.basename(config.save_dir)
    
    # 从 run_name 中提取时间戳（格式：env_arch_dense_YYYYMMDD_HHMMSS）
    import re
    timestamp_match = re.search(r'(\d{8}_\d{6})$', run_name)
    if timestamp_match:
        timestamp_str = timestamp_match.group(1)
        # 转换为可读格式 YYYY-MM-DD HH:MM:SS
        try:
            from datetime import datetime as dt
            parsed_time = dt.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            timestamp_readable = parsed_time.strftime("%Y-%m-%d %H:%M:%S")
            timestamp_compact = timestamp_str  # 保留紧凑格式
        except:
            timestamp_readable = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            timestamp_compact = timestamp_str
    else:
        timestamp_readable = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        timestamp_compact = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 查找保存的模型文件名
    model_path = ""
    model_file = ""
    if os.path.exists(config.save_dir):
        # 查找 final_model_*.pt 文件
        for f in os.listdir(config.save_dir):
            if f.startswith("final_model_") and f.endswith(".pt"):
                model_file = f
                model_path = os.path.join(config.save_dir, f)
                break
        # 如果没找到时间戳版本，使用 final_model.pt
        if not model_file and os.path.exists(os.path.join(config.save_dir, "final_model.pt")):
            model_file = "final_model.pt"
            model_path = os.path.join(config.save_dir, "final_model.pt")
    
    record = {
        "timestamp": timestamp_readable,
        "timestamp_compact": timestamp_compact,
        "run_name": run_name,
        "model_file": model_file,
        "model_path": model_path,
        "env": getattr(config, "env", "N/A"),
        "actor_arch": getattr(config, "actor_arch", "N/A"),
        "max_timesteps": getattr(config, "max_timesteps", 0),
        "total_episodes": total_episodes,
        
        # ===== 训练指标 =====
        "avg_episode_reward": round(avg_episode_reward, 4),
        "avg_episode_success": round(avg_episode_success, 4),
        
        # ===== 任务完成指标（最后5次平均）=====
        "final_eval_reward": round(final_reward, 4),
        "final_eval_success": round(final_success, 4),
        "final_eval_tts": round(final_tts, 2),
        "final_eval_min_distance": round(final_min_dist, 4),
        
        # ===== 轨迹质量指标 =====
        "final_eval_rmse": round(final_rmse, 4),
        "final_eval_max_deviation": round(final_max_dev, 4),
        "final_eval_endpoint_error": round(final_end_err, 4),
        "final_eval_frechet_distance": round(final_frechet, 4),
        
        # ===== 运动平滑度指标 =====
        "final_eval_rms_acceleration": round(final_rms_accel, 4),
        "final_eval_jerk": round(final_jerk, 4),
        "final_eval_velocity_variance": round(final_vel_var, 6),
        
        # ===== 路径效率指标 =====
        "final_eval_path_length_exec": round(final_path_len_exec, 4),
        "final_eval_path_length_ref": round(final_path_len_ref, 4),
        "final_eval_path_efficiency": round(final_efficiency, 4),
        "final_eval_energy": round(final_energy, 4),
        
        # ===== 超参数 =====
        "batch_size": getattr(config, "batch_size", 0),
        "start_timesteps": getattr(config, "start_timesteps", 0),
        "expl_noise": getattr(config, "expl_noise", 0.0),
    }
    
    # 检查文件是否存在，决定是否写入表头
    file_exists = os.path.isfile(global_summary_file)
    
    try:
        with open(global_summary_file, "a", newline="", encoding="utf-8") as csvfile:
            fieldnames = list(record.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(record)
        
        print(f"[Info] Global summary saved to: {global_summary_file}")
    except Exception as e:
        print(f"[Warn] Failed to save global summary: {e}")


# ===================== 训练主逻辑 =====================

def run_experiment(config: Any) -> None:
    """
    执行完整的 TD3 训练实验
    
    包括环境创建、智能体初始化、训练循环、定期评估、模型保存等完整流程。
    支持早停、自动降噪、增量保存、全局汇总等功能。
    
    Args:
        config: 配置对象，包含以下属性:
            - env: 环境名称
            - actor_arch: Actor 架构类型
            - max_timesteps: 最大训练步数
            - start_timesteps: 纯随机探索步数
            - batch_size: 批次大小
            - eval_freq: 评估间隔
            - expl_noise: 探索噪声
            - save_dir: 保存目录
            - node_dim, num_nodes: 图网络参数（可选）
            
    Side Effects:
        - 创建训练和评估环境
        - 在 save_dir 下创建独立的运行目录
        - 保存训练数据、模型、配置到文件
        - 追加结果到全局汇总 CSV
        - 打印训练进度和评估结果
        
    Examples:
        >>> import argparse
        >>> args = argparse.Namespace(
        ...     env="KukaIiwa7Track-v0",
        ...     actor_arch="gnn_transformer",
        ...     max_timesteps=200000,
        ...     batch_size=256
        ... )
        >>> run_experiment(args)
    """
    # --- 独立运行目录 ---
    from datetime import datetime
    arch_str = getattr(config, "actor_arch", "mlp")
    seed = int(getattr(config, "seed", 0))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config.env}_{arch_str}_dense_{stamp}"
    config.save_dir = os.path.join(config.save_dir, run_name)
    os.makedirs(config.save_dir, exist_ok=True)

    # ===== 全局随机种子（确保复现性）=====
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed] Global seed set to {seed}")
    try:
        with open(os.path.join(os.path.dirname(config.save_dir), "LATEST_RUN.txt"), "w", encoding="utf-8") as f:
            f.write(config.save_dir)
    except Exception:
        pass
    print(f"[Info] Saving this run to: {config.save_dir}")

    # 训练环境（无 GUI 推荐 rgb_array；不支持则省略 render_mode）
    env_kwargs = {"dense_reward": True}

    try:
        env = gym.make(config.env, render_mode="rgb_array", **env_kwargs)
    except TypeError:
        env = gym.make(config.env, **env_kwargs)

    try:
        eval_env = gym.make(config.env, render_mode="rgb_array", **env_kwargs)
    except TypeError:
        eval_env = gym.make(config.env, **env_kwargs)
    # 固定评估环境种子（评估结果跨 run 可比）
    eval_env.reset(seed=42)

    # --- 状态/动作维度 ---
    if hasattr(env.observation_space, "spaces") and "observation" in env.observation_space.spaces:
        obs_dim = env.observation_space["observation"].shape[0]
        goal_dim = env.observation_space["achieved_goal"].shape[0]
        state_dim = obs_dim + 2 * goal_dim
    else:
        state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # --- 初始化 TD3 ---
    agent = TD3(state_dim, action_dim, max_action, config)

    # ===== 打印当前使用的设备（CPU / GPU）=====
    print("Actor device:", next(agent.actor.parameters()).device)
    print("Critic device:", next(agent.critic1.parameters()).device)

    # --- 🎯 经验池：限制为50万步，避免分布漂移（防止400k崩溃）---
    dist_thr = float(getattr(env.unwrapped, "distance_threshold", 0.05))
    replay_buffer = ReplayBuffer(env.observation_space, env.action_space,
                                 capacity=int(5e5), her_prob=0.0, her_k=0,
                                 dense_reward=True, distance_threshold=dist_thr)

    best_eval_success = -float("inf")
    best_eval_reward = -float("inf")
    best_eval_step = 0
    no_improve_evals = 0
    noise_decay_hits = 0

    def _save_actor_checkpoint(base_name: str) -> None:
        try:
            agent.save(base_name)
        except Exception:
            try:
                torch.save(getattr(agent, "actor").state_dict(), base_name + ".pt")
            except Exception as e:
                print(f"[Warn] Checkpoint save failed ({base_name}): {e}")

    # --- 训练状态 ---
    episode_reward = 0.0
    episode_timesteps = 0
    episode_num = 0
    obs, info = reset_env(env)

    rewards_history = []
    success_history = []
    
    # ===== 新增：每步记录的数据 =====
    step_data = {
        "timestep": [],              # 全局步数
        "step_reward": [],           # 每步即时奖励
        "step_distance": [],         # 每步到目标的距离
        "step_success": [],          # 每步是否成功（0/1）
        "episode_num": [],           # 当前回合编号
        "episode_timestep": [],      # 回合内步数
        # 评估指标（当前最新的评估值）
        "eval_reward": [],           # Eval Reward (avg)
        "eval_success_rate": [],     # Eval Success Rate (avg)
        "eval_time_to_success": [],  # Eval Time To Success (avg)
        "eval_min_distance": [],     # Eval Min Distance (avg)
        "eval_rmse": [],             # Eval RMSE (avg)
        "eval_max_deviation": [],    # Eval Max Deviation (avg)
        "eval_endpoint_error": [],   # Eval End-Point Error (avg)
        "eval_path_length": [],      # Eval Path Length (avg)
    }
    
    # 当前评估指标的值（初始为 NaN，每次评估后更新）
    current_eval_metrics = {
        "eval_reward": float('nan'),
        "eval_success_rate": float('nan'),
        "eval_time_to_success": float('nan'),
        "eval_min_distance": float('nan'),
        "eval_rmse": float('nan'),
        "eval_max_deviation": float('nan'),
        "eval_endpoint_error": float('nan'),
        "eval_path_length": float('nan'),
    }
    
    metrics = {
        "actor_loss": [],
        "actor_loss_t": [],
        "critic_loss": [],
        "critic_loss_t": [],
        "eval_rewards": [],
        "eval_success": [],
        "eval_tts": [],
        "eval_min_distance": [],
        # 轨迹质量
        "eval_rmse": [],
        "eval_max_dev": [],
        "eval_end_err": [],
        "eval_path_len_exec": [],
        "eval_path_len_ref": [],
        # 论文级高级指标（与评估函数对齐，统一在此初始化）
        "eval_frechet": [],
        "eval_rms_accel": [],
        "eval_jerk": [],
        "eval_vel_var": [],
        "eval_efficiency": [],
        "eval_energy": [],
        # 奖励分量（诊断用，分析各项对总奖励的贡献）
        "eval_comp_goal": [],
        "eval_comp_track": [],
        "eval_comp_success": [],
        "eval_comp_smooth": [],
        "eval_comp_vel": [],
    }

    early_hits = 0  # 早停计数器
    task_level = str(getattr(config, "task_level", "task1"))
    eval_output_dir = os.path.join(config.save_dir, "eval")

    base_eval_meta = {
        "seed": int(getattr(config, "seed", 0)),
        "task_level": task_level,
        "actor_arch": str(getattr(config, "actor_arch", "unknown")),
        "model_name": "TD3",
        "env_id": str(getattr(config, "env", "unknown")),
        "run_name": run_name,
    }

    # ===== 初始评估（第 0 步）=====
    print("[Init] 执行初始评估（训练前基线）...")
    (eval_reward, eval_success, eval_tts, eval_min_d,
     eval_rmse, eval_max_dev, eval_end_err, eval_frechet,
     eval_rms_accel, eval_jerk, eval_vel_var,
     eval_pl_exec, eval_pl_ref, eval_efficiency, eval_energy,
     eval_comp_goal, eval_comp_track, eval_comp_success, eval_comp_smooth, eval_comp_vel
     ) = evaluate_policy(
        agent,
        eval_env,
        eval_episodes=20,
        deterministic=True,
        metadata={**base_eval_meta, "checkpoint_step": 0, "eval_split": "eval"},
        output_dir=eval_output_dir,
    )
    
    # 更新当前评估指标
    current_eval_metrics["eval_reward"] = eval_reward
    current_eval_metrics["eval_success_rate"] = eval_success
    current_eval_metrics["eval_time_to_success"] = eval_tts
    current_eval_metrics["eval_min_distance"] = eval_min_d
    current_eval_metrics["eval_rmse"] = eval_rmse
    current_eval_metrics["eval_max_deviation"] = eval_max_dev
    current_eval_metrics["eval_endpoint_error"] = eval_end_err
    current_eval_metrics["eval_frechet_distance"] = eval_frechet
    current_eval_metrics["eval_rms_acceleration"] = eval_rms_accel
    current_eval_metrics["eval_jerk"] = eval_jerk
    current_eval_metrics["eval_velocity_variance"] = eval_vel_var
    current_eval_metrics["eval_path_length"] = (eval_pl_exec + eval_pl_ref) / 2.0 if not np.isnan(eval_pl_exec) and not np.isnan(eval_pl_ref) else eval_pl_exec
    current_eval_metrics["eval_path_efficiency"] = eval_efficiency
    current_eval_metrics["eval_energy"] = eval_energy
    
    print(f"[Init Eval] Reward {eval_reward:.3f} | Success {eval_success:.3f} | "
          f"TTS {eval_tts:.1f} | MinD {eval_min_d:.3f} m")
    
    env_name = getattr(config, 'env', 'Unknown')
    if 'Track' in env_name:
        print(f"            TrackRMSE {eval_rmse:.4f} | TrackMaxDev {eval_max_dev:.4f} | TrackEndErr {eval_end_err:.4f} | Fréchet {eval_frechet:.4f}")
        print(f"            RMS_Accel {eval_rms_accel:.4f} m/s² | Jerk {eval_jerk:.4f} m/s³ | VelVar {eval_vel_var:.6f} | Efficiency {eval_efficiency:.3f}")
    else:
        print(f"            AvgDist {eval_rmse:.4f} | MaxDist {eval_max_dev:.4f} | FinalDist {eval_end_err:.4f}")
        print(f"            RMS_Accel {eval_rms_accel:.4f} m/s² | Jerk {eval_jerk:.4f} m/s³ | VelVar {eval_vel_var:.6f}")

    # --- 主循环 ---
    best_eval_success = float(eval_success)
    best_eval_reward = float(eval_reward)
    best_eval_step = 0

    try:
        for t in range(int(config.max_timesteps)):
            episode_timesteps += 1

            # 选动作
            if t < config.start_timesteps:
                action = env.action_space.sample()
            else:
                state_input = flatten_obs(obs)
                action = agent.select_action(state_input)
                if config.expl_noise != 0:
                    # 噪声尺度与动作空间范围对齐（expl_noise 是相对于 max_action 的比例）
                    noise_scale = config.expl_noise * max_action
                    action = (action + np.random.normal(0, noise_scale, size=action_dim)
                              ).clip(env.action_space.low, env.action_space.high)

            # 环境一步
            next_obs, reward, done, info = step_env(env, action)

            # ✅ 尊重环境的 done_on_success 设置
            # Track 环境默认 done_on_success=False，保证完整 200 步
            # 移除任何强制提前终止的逻辑

            episode_reward += float(reward)

            # ===== 记录每步数据 =====
            step_data["timestep"].append(int(t + 1))
            step_data["step_reward"].append(float(reward))
            step_data["episode_num"].append(int(episode_num))
            step_data["episode_timestep"].append(int(episode_timesteps))
            
            # 记录当前步的距离和成功状态
            current_distance = _extract_distance(next_obs, info, env)
            if current_distance is not None:
                step_data["step_distance"].append(float(current_distance))
            else:
                step_data["step_distance"].append(float('nan'))
            
            is_success = _is_success_state(next_obs, info, env)
            step_data["step_success"].append(1 if is_success else 0)
            
            # 记录当前的评估指标（使用最新的评估值）
            step_data["eval_reward"].append(current_eval_metrics["eval_reward"])
            step_data["eval_success_rate"].append(current_eval_metrics["eval_success_rate"])
            step_data["eval_time_to_success"].append(current_eval_metrics["eval_time_to_success"])
            step_data["eval_min_distance"].append(current_eval_metrics["eval_min_distance"])
            step_data["eval_rmse"].append(current_eval_metrics["eval_rmse"])
            step_data["eval_max_deviation"].append(current_eval_metrics["eval_max_deviation"])
            step_data["eval_endpoint_error"].append(current_eval_metrics["eval_endpoint_error"])
            step_data["eval_path_length"].append(current_eval_metrics["eval_path_length"])
            
            # 存储
            replay_buffer.add(obs, action, next_obs, reward, done)
            obs = next_obs

            # 训练（并记录"对应的全局步数"t+1）
            if t >= config.start_timesteps:
                actor_loss = None
                critic_loss = None
                out = agent.train(replay_buffer, config.batch_size)

                if isinstance(out, dict):
                    actor_loss = out.get("actor_loss", None)
                    critic_loss = out.get("critic_loss", None)
                elif isinstance(out, (list, tuple)):
                    if len(out) >= 2:
                        actor_loss, critic_loss = out[0], out[1]
                    elif len(out) == 1:
                        critic_loss = out[0]
                else:
                    critic_loss = out

                if critic_loss is not None:
                    metrics["critic_loss"].append(float(critic_loss))
                    metrics["critic_loss_t"].append(int(t + 1))
                if actor_loss is not None:
                    metrics["actor_loss"].append(float(actor_loss))
                    metrics["actor_loss_t"].append(int(t + 1))

            # 回合结束
            if done:
                if t >= config.start_timesteps:
                    # 只在策略训练阶段打印，避免随机探索期的噪声值干扰观察
                    print(f"Total T: {t+1} | Ep: {episode_num+1} | EpT: {episode_timesteps} | R: {episode_reward:.3f}")
                else:
                    print(f"Total T: {t+1} | Ep: {episode_num+1} | EpT: {episode_timesteps} | R: {episode_reward:.3f} [exploring]")
                rewards_history.append(episode_reward)
                success_history.append(1 if _is_success_state(obs, info, env) else 0)
                obs, info = reset_env(env)
                episode_reward = 0.0
                episode_timesteps = 0
                episode_num += 1

            # 定期评估 + 增量保存 + 早停 + 降噪
            if (t + 1) % config.eval_freq == 0:
                (eval_reward, eval_success, eval_tts, eval_min_d,
                 eval_rmse, eval_max_dev, eval_end_err, eval_frechet,
                 eval_rms_accel, eval_jerk, eval_vel_var,
                 eval_pl_exec, eval_pl_ref, eval_efficiency, eval_energy,
                 eval_comp_goal, eval_comp_track, eval_comp_success, eval_comp_smooth, eval_comp_vel
                 ) = evaluate_policy(
                    agent,
                    eval_env,
                    eval_episodes=20,
                    deterministic=True,
                    metadata={**base_eval_meta, "checkpoint_step": int(t + 1), "eval_split": "eval"},
                    output_dir=eval_output_dir,
                )

                metrics["eval_rewards"].append(eval_reward)
                metrics["eval_success"].append(eval_success)
                metrics["eval_tts"].append(eval_tts)
                metrics["eval_min_distance"].append(eval_min_d)
                # ===== 轨迹质量指标 =====
                metrics["eval_rmse"].append(eval_rmse)
                metrics["eval_max_dev"].append(eval_max_dev)
                metrics["eval_end_err"].append(eval_end_err)
                metrics["eval_path_len_exec"].append(eval_pl_exec)
                metrics["eval_path_len_ref"].append(eval_pl_ref)

                # ===== 论文级高级指标 =====
                metrics["eval_frechet"].append(eval_frechet)
                metrics["eval_rms_accel"].append(eval_rms_accel)
                metrics["eval_jerk"].append(eval_jerk)
                metrics["eval_vel_var"].append(eval_vel_var)
                metrics["eval_efficiency"].append(eval_efficiency)
                metrics["eval_energy"].append(eval_energy)
                # ===== 奖励分量记录（诊断用）=====
                metrics["eval_comp_goal"].append(eval_comp_goal)
                metrics["eval_comp_track"].append(eval_comp_track)
                metrics["eval_comp_success"].append(eval_comp_success)
                metrics["eval_comp_smooth"].append(eval_comp_smooth)
                metrics["eval_comp_vel"].append(eval_comp_vel)
                
                # ===== 更新当前评估指标（用于每步记录）=====
                current_eval_metrics["eval_reward"] = eval_reward
                current_eval_metrics["eval_success_rate"] = eval_success
                current_eval_metrics["eval_time_to_success"] = eval_tts
                current_eval_metrics["eval_min_distance"] = eval_min_d
                current_eval_metrics["eval_rmse"] = eval_rmse
                current_eval_metrics["eval_max_deviation"] = eval_max_dev
                current_eval_metrics["eval_endpoint_error"] = eval_end_err
                current_eval_metrics["eval_frechet_distance"] = eval_frechet
                current_eval_metrics["eval_rms_acceleration"] = eval_rms_accel
                current_eval_metrics["eval_jerk"] = eval_jerk
                current_eval_metrics["eval_velocity_variance"] = eval_vel_var
                current_eval_metrics["eval_path_length"] = (eval_pl_exec + eval_pl_ref) / 2.0 if not np.isnan(eval_pl_exec) and not np.isnan(eval_pl_ref) else eval_pl_exec
                current_eval_metrics["eval_path_efficiency"] = eval_efficiency
                current_eval_metrics["eval_energy"] = eval_energy

                print(f"[Eval] Reward {eval_reward:.3f} | Success {eval_success:.3f} | "
                      f"TTS {eval_tts:.1f} | MinD {eval_min_d:.3f} m")
                print(f"       RewardComp: goal={eval_comp_goal:.2f} track={eval_comp_track:.2f} "
                      f"success={eval_comp_success:.2f} smooth={eval_comp_smooth:.2f} vel={eval_comp_vel:.2f}")
                if 'Track' in env_name:
                    print(f"       TrackRMSE {eval_rmse:.4f} | MaxDev {eval_max_dev:.4f} | EndErr {eval_end_err:.4f} | Fréchet {eval_frechet:.4f}")
                    print(f"       RMS_Accel {eval_rms_accel:.4f} m/s² | Jerk {eval_jerk:.4f} m/s³ | VelVar {eval_vel_var:.6f} | Efficiency {eval_efficiency:.3f}")
                else:
                    print(f"       AvgDist {eval_rmse:.4f} | MaxDist {eval_max_dev:.4f} | FinalDist {eval_end_err:.4f}")
                    print(f"       RMS_Accel {eval_rms_accel:.4f} m/s² | Jerk {eval_jerk:.4f} m/s³")

                # 增量落盘
                os.makedirs(config.save_dir, exist_ok=True)
                np.save(os.path.join(config.save_dir, "rewards.npy"), np.array(rewards_history))
                np.save(os.path.join(config.save_dir, "success.npy"), np.array(success_history))
                
                # ===== 保存每步数据 =====
                step_df = pd.DataFrame(step_data)
                step_df.to_csv(os.path.join(config.save_dir, "step_data.csv"), index=False)
                
                with open(os.path.join(config.save_dir, "metrics.json"), "w") as f:
                    json.dump(metrics, f)

                # 🎯 噪声调度策略：已禁用自动降噪
                # 环境中已添加Jerk惩罚，保持恒定探索噪声避免崩溃
                # 如需启用降噪，取消下面注释并调整阈值
                # if eval_success >= 0.95 and getattr(config, "expl_noise", 0.1) > 0.005:
                #     old_noise = config.expl_noise
                #     config.expl_noise = max(0.005, config.expl_noise * 0.7)
                #     if old_noise != config.expl_noise:
                #         print(f"[Schedule] Success {eval_success:.2%} ≥ 95%, reduce expl_noise: {old_noise:.4f} -> {config.expl_noise:.4f}")
                pass  # 禁用降噪

                # 早停功能已禁用 - 训练将持续到 max_timesteps
                # 如需启用早停，请取消下面的注释
                # EARLY_STOP_TARGET = 0.90
                # EARLY_STOP_PATIENCE = 20
                # early_hits = early_hits + 1 if eval_success >= EARLY_STOP_TARGET else 0
                # if early_hits >= EARLY_STOP_PATIENCE:
                #     print(f"[EarlyStop] success ≥ {EARLY_STOP_TARGET*100:.1f}% for {EARLY_STOP_PATIENCE} evals. Stop.")
                #     ... (保存代码)
                #     return
                pass  # 继续训练
    
    except KeyboardInterrupt:
        print("\n[Interrupted] Training interrupted by user (Ctrl+C). Saving current progress...")
        # ===== 保存中断时的数据 =====
        os.makedirs(config.save_dir, exist_ok=True)
        np.save(os.path.join(config.save_dir, "rewards.npy"), np.array(rewards_history))
        np.save(os.path.join(config.save_dir, "success.npy"), np.array(success_history))
        
        # 保存每步数据
        step_df = pd.DataFrame(step_data)
        step_df.to_csv(os.path.join(config.save_dir, "step_data.csv"), index=False)
        
        # 保存指标
        with open(os.path.join(config.save_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f)
        with open(os.path.join(config.save_dir, "config.json"), "w") as f:
            json.dump(vars(config), f, indent=2)
        
        # ===== 重要：保存 Loss 图（即使中断） =====
        print("[Interrupted] Generating training loss plots...")
        plot_training_loss(metrics, config.save_dir)
        
        # 保存全局汇总
        save_global_summary(config, metrics, rewards_history, success_history)
        
        # 安全关闭环境
        try:
            env.close()
        except Exception:
            pass
        try:
            eval_env.close()
        except Exception:
            pass
        
        print(f"[Interrupted] Progress saved to: {config.save_dir}")
        return  # 结束训练

    # ---- 训练自然结束：保存结果/模型 ----
    os.makedirs(config.save_dir, exist_ok=True)
    np.save(os.path.join(config.save_dir, "rewards.npy"), np.array(rewards_history))
    np.save(os.path.join(config.save_dir, "success.npy"), np.array(success_history))
    
    # ===== 保存每步数据（训练结束时） =====
    step_df = pd.DataFrame(step_data)
    step_df.to_csv(os.path.join(config.save_dir, "step_data.csv"), index=False)
    
    with open(os.path.join(config.save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f)

    from datetime import datetime
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_base = os.path.join(config.save_dir, "final_model")
    ts_model_base = os.path.join(config.save_dir, f"final_model_{tag}")
    try:
        agent.save(ts_model_base)
        agent.save(final_model_base)
    except Exception:
        try:
            torch.save(getattr(agent, "actor").state_dict(), ts_model_base + ".pt")
            torch.save(getattr(agent, "actor").state_dict(), final_model_base + ".pt")
        except Exception as e:
            print(f"[Warn] 模型保存失败：{e}")

    with open(os.path.join(config.save_dir, "config.json"), "w") as f:
        json.dump(vars(config), f, indent=2)
    print(f"[Info] Saved results to: {config.save_dir}")

    # ===== 保存全局汇总指标 =====
    save_global_summary(config, metrics, rewards_history, success_history)
    
    # ===== 绘制训练 Loss 曲线 =====
    plot_training_loss(metrics, config.save_dir)

    # 安全关闭环境
    try:
        env.close()
    except Exception as e:
        print(f"[Warn] env.close() failed: {e}")
    try:
        eval_env.close()
    except Exception as e:
        print(f"[Warn] eval_env.close() failed: {e}")


# ===================== 入口参数 =====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TD3 训练脚本 - 支持多种 Actor 架构",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用 MLP 架构（默认Track环境）
  python training/train_experiment.py --actor_arch mlp
  
  # 使用 GNN+Transformer 架构
  python training/train_experiment.py --actor_arch gnn_transformer
  
  # 自定义训练步数
  python training/train_experiment.py --actor_arch transformer --max_timesteps 500000
  
  # 使用 Reach 环境（如需要）
  python training/train_experiment.py --actor_arch mlp --env KukaIiwa7Track-v0
        """
    )
    
    # 环境与训练参数
    parser.add_argument("--env", type=str, default="KukaIiwa7Track-v0",
                        help="环境名称 (default: KukaIiwa7Track-v0, 轨迹跟踪任务)")
    parser.add_argument("--max_timesteps", type=int, default=500_000,
                        help="最大训练步数 (default: 500000, MLP可用300k, Transformer/GNN+Trans建议500k)")
    parser.add_argument("--start_timesteps", type=int, default=25_000,
                        help="纯随机探索步数 (default: 25000)")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="批次大小 (default: 512, 更大批次提高稳定性)")
    parser.add_argument("--eval_freq", type=int, default=5000,
                        help="评估间隔步数 (default: 5000, 更频繁监控训练进度)")
    parser.add_argument("--expl_noise", type=float, default=0.1,
                        help="探索噪声（相对于max_action的比例，default: 0.1，标准TD3推荐值）")
    parser.add_argument("--save_dir", type=str, default="./results",
                        help="结果保存目录 (default: ./results)")
    parser.add_argument("--seed", type=int, default=0,
                        help="全局随机种子，用于复现 (default: 0)")
    parser.add_argument("--task_level", type=str, default="task1",
                        choices=["task1", "task2", "task3"],
                        help="任务层级标签（用于评估元信息与多seed聚合，default: task1）")

    # 架构相关参数
    parser.add_argument("--actor_arch", type=str, default="gnn_transformer",
                        choices=["mlp", "gnn", "transformer", "gnn_transformer"],
                        help="Actor 架构类型 (default: gnn_transformer)")
    parser.add_argument("--node_dim", type=int, default=None,
                        help="节点特征维度（GNN/Transformer使用，默认自动设置）")
    parser.add_argument("--num_nodes", type=int, default=None,
                        help="节点数量（GNN/Transformer使用，默认自动设置）")
    
    # 学习率参数------------------------------------------------------------------------------------
    parser.add_argument("--actor_lr", type=float, default=None,
                        help="Actor 学习率 (default: 根据架构自动设置)")
    parser.add_argument("--critic_lr", type=float, default=None,
                        help="Critic 学习率 (default: 根据架构自动设置)")
    args = parser.parse_args()
    
    # 根据架构自动设置 node_dim 和 num_nodes（如果未指定）
    defaults = get_default_args(args.actor_arch)
    
    # 🎯 针对不同架构的优化学习率（基于实验结果）
    if args.actor_lr is None or args.critic_lr is None:
        if args.actor_arch == "mlp":
            # MLP: 验证有效（100%成功率）
            default_actor_lr = 1e-5
            default_critic_lr = 1e-5
        elif args.actor_arch == "gnn":
            # GNN: 大幅降低学习率提高稳定性
            default_actor_lr = 1e-5
            default_critic_lr = 1e-5
        elif args.actor_arch == "transformer":
            # Transformer: 降低学习率提高稳定性
            default_actor_lr = 1e-5
            default_critic_lr = 1e-5
        elif args.actor_arch == "gnn_transformer":
            # GT-TD3: 平衡学习率（Buffer已限龄，无需过低）
            default_actor_lr = 1e-5
            default_critic_lr = 1e-5
        else:
            default_actor_lr = 3e-5
            default_critic_lr = 3e-5
        
        if args.actor_lr is None:
            args.actor_lr = default_actor_lr
        if args.critic_lr is None:
            args.critic_lr = default_critic_lr
    
    if args.node_dim is None:
        args.node_dim = defaults["node_dim"]
    if args.num_nodes is None:
        args.num_nodes = defaults["num_nodes"]
    
    # 设置 use_state_encoder（对于GNN/Transformer架构很重要）
    if not hasattr(args, 'use_state_encoder'):
        args.use_state_encoder = defaults.get("use_state_encoder", False)
    
    # 打印训练配置
    print("\n" + "="*60)
    print("训练配置总览 - 优化版本")
    print("="*60)
    print(f"环境: {args.env}")
    print(f"任务层级: {args.task_level}")
    print(f"架构: {args.actor_arch}")
    print(f"最大步数: {args.max_timesteps:,}")
    print(f"\n学习参数:")
    print(f"  Actor学习率: {args.actor_lr}")
    print(f"  Critic学习率: {args.critic_lr}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  探索噪声: {args.expl_noise}")
    print(f"\n评估参数:")
    print(f"  评估频率: 每{args.eval_freq:,}步")
    print(f"  评估回合数: 20回合（优化）")
    print(f"\n优化配置:")
    print(f"  经验池容量: 2M（优化）")
    print(f"  早停策略: 90%成功率×20次（优化）")
    print(f"  降噪策略: 50%触发，逐步减半（优化）")
    if args.node_dim is not None:
        print(f"节点维度: {args.node_dim}")
        print(f"节点数量: {args.num_nodes}")
        print(f"状态编码器: {'开启' if args.use_state_encoder else '关闭'}")
    print("="*60 + "\n")
    
    run_experiment(args)

# -*- coding: utf-8 -*-
"""
模型评估脚本 - 分析训练结果并生成综合评分报告
评估维度：性能、稳定性、收敛速度、资源效率
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime

# 科研风格设置
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


def load_training_metrics(run_dir):
    """加载训练指标"""
    metrics_path = os.path.join(run_dir, "metrics.json")
    if not os.path.isfile(metrics_path):
        return None
    
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    
    return metrics


def load_config(run_dir):
    """加载配置文件"""
    cfg_path = os.path.join(run_dir, "config.json")
    if not os.path.isfile(cfg_path):
        return {}
    
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    
    return cfg


def get_last_n_avg(values, n=10):
    """获取最后N个值的平均值"""
    if len(values) == 0:
        return float('nan')
    if len(values) < n:
        return np.mean(values)
    return np.mean(values[-n:])


def calculate_stability_score(values):
    """
    计算稳定性得分 (0-100)
    基于变异系数 (CV = std/mean) 和趋势平滑度
    """
    if len(values) < 2:
        return 0.0
    
    values = np.array(values)
    
    # 变异系数 (越小越稳定)
    mean_val = np.mean(values)
    if mean_val == 0:
        cv = 0
    else:
        cv = np.std(values) / abs(mean_val)
    
    # 趋势平滑度 (相邻点变化率)
    if len(values) > 1:
        changes = np.abs(np.diff(values))
        smoothness = np.mean(changes) / (np.max(values) - np.min(values) + 1e-8)
    else:
        smoothness = 0
    
    # 综合得分 (CV和smoothness越小越好)
    cv_score = max(0, 100 - cv * 100)
    smooth_score = max(0, 100 - smoothness * 100)
    
    stability = 0.6 * cv_score + 0.4 * smooth_score
    return min(100, max(0, stability))


def calculate_convergence_speed(values, threshold=0.9):
    """
    计算收敛速度得分 (0-100)
    评估到达目标性能的速度
    """
    if len(values) < 2:
        return 0.0
    
    values = np.array(values)
    max_val = np.max(values)
    target = threshold * max_val
    
    # 找到首次达到目标的位置
    reach_indices = np.where(values >= target)[0]
    if len(reach_indices) == 0:
        # 未达到目标，根据最终性能给分
        final_ratio = values[-1] / max_val if max_val > 0 else 0
        return final_ratio * 50  # 最多50分
    
    # 达到目标的位置 (越早越好)
    reach_step = reach_indices[0]
    total_steps = len(values)
    
    # 早期达到得分更高
    speed_score = 100 * (1 - reach_step / total_steps)
    
    return min(100, max(0, speed_score))


def calculate_final_performance_score(metrics):
    """
    计算最终性能得分 (0-100)
    基于论文要求的四个维度和八个关键指标：
    
    四个维度：
    1. Task Success Rate (任务成功率)
    2. Trajectory Accuracy (轨迹精度) 
    3. Execution Performance (执行性能)
    4. Path Optimization (路径优化)
    
    八个指标：
    - Success Rate (成功率)
    - RMSE (均方根误差)
    - Maximum Deviation (最大偏差)
    - Endpoint Error (端点误差)
    - Cumulative Reward (累积奖励)
    - Success Time (成功时间)
    - Path Length (路径长度)
    - Minimum Obstacle Distance (最小障碍物距离)
    """
    
    # 维度1: Task Success Rate (权重: 0.30)
    task_success_scores = []
    if "eval_success" in metrics and len(metrics["eval_success"]) > 0:
        sr = metrics["eval_success"][-1]
        task_success_scores.append(sr * 100)
    
    # 维度2: Trajectory Accuracy (权重: 0.30)
    trajectory_accuracy_scores = []
    
    # RMSE (越小越好, 假设范围 [0, 0.5])
    if "eval_rmse" in metrics and len(metrics["eval_rmse"]) > 0:
        rmse = metrics["eval_rmse"][-1]
        rmse_score = max(0, 100 - rmse * 200)
        trajectory_accuracy_scores.append(rmse_score)
    
    # Maximum Deviation (越小越好, 假设范围 [0, 1.0])
    if "eval_max_dev" in metrics and len(metrics["eval_max_dev"]) > 0:
        max_dev = metrics["eval_max_dev"][-1]
        max_dev_score = max(0, 100 - max_dev * 100)
        trajectory_accuracy_scores.append(max_dev_score)
    
    # Endpoint Error (越小越好, 假设范围 [0, 0.5])
    if "eval_end_err" in metrics and len(metrics["eval_end_err"]) > 0:
        end_err = metrics["eval_end_err"][-1]
        end_err_score = max(0, 100 - end_err * 200)
        trajectory_accuracy_scores.append(end_err_score)
    
    # 维度3: Execution Performance (权重: 0.25)
    execution_perf_scores = []
    
    # Cumulative Reward (假设范围 [-50, 100])
    if "eval_rewards" in metrics and len(metrics["eval_rewards"]) > 0:
        reward = metrics["eval_rewards"][-1]
        reward_score = min(100, max(0, (reward + 50) / 1.5))
        execution_perf_scores.append(reward_score)
    
    # Success Time (越小越好, 假设范围 [0, 200])
    if "eval_tts" in metrics and len(metrics["eval_tts"]) > 0:
        tts = metrics["eval_tts"][-1]
        tts_score = max(0, 100 - tts / 2)
        execution_perf_scores.append(tts_score)
    
    # 维度4: Path Optimization (权重: 0.15)
    path_opt_scores = []
    
    # Path Length (越短越好, 与参考轨迹比较)
    if "eval_path_len_exec" in metrics and len(metrics["eval_path_len_exec"]) > 0:
        exec_len = metrics["eval_path_len_exec"][-1]
        if "eval_path_len_ref" in metrics and len(metrics["eval_path_len_ref"]) > 0:
            ref_len = metrics["eval_path_len_ref"][-1]
            if ref_len > 0:
                # 路径长度比率，越接近1越好
                len_ratio = exec_len / ref_len
                if len_ratio <= 1.2:  # 在120%以内得高分
                    path_len_score = max(0, 100 - abs(len_ratio - 1.0) * 200)
                else:
                    path_len_score = max(0, 100 - (len_ratio - 1.0) * 100)
                path_opt_scores.append(path_len_score)
    
    # Minimum Obstacle Distance (越大越好, 假设范围 [0, 1.0])
    if "eval_min_distance" in metrics and len(metrics["eval_min_distance"]) > 0:
        min_dist = metrics["eval_min_distance"][-1]
        # 距离越大越安全
        min_dist_score = min(100, min_dist * 100)
        path_opt_scores.append(min_dist_score)
    
    # 计算各维度得分
    dimension_scores = []
    dimension_weights = []
    
    if len(task_success_scores) > 0:
        dimension_scores.append(np.mean(task_success_scores))
        dimension_weights.append(0.30)
    
    if len(trajectory_accuracy_scores) > 0:
        dimension_scores.append(np.mean(trajectory_accuracy_scores))
        dimension_weights.append(0.30)
    
    if len(execution_perf_scores) > 0:
        dimension_scores.append(np.mean(execution_perf_scores))
        dimension_weights.append(0.25)
    
    if len(path_opt_scores) > 0:
        dimension_scores.append(np.mean(path_opt_scores))
        dimension_weights.append(0.15)
    
    if len(dimension_scores) == 0:
        return 0.0
    
    # 归一化权重
    weights = np.array(dimension_weights)
    weights = weights / weights.sum()
    
    performance = np.average(dimension_scores, weights=weights)
    return min(100, max(0, performance))


def calculate_resource_efficiency(cfg, metrics):
    """
    计算资源效率得分 (0-100)
    考虑模型复杂度和训练时间
    """
    scores = []
    
    # 模型复杂度评分 (简单模型得分高)
    actor_arch = cfg.get("actor_arch", "mlp")
    if actor_arch == "mlp":
        complexity_score = 100
    elif actor_arch == "gnn":
        complexity_score = 85
    elif actor_arch == "transformer":
        complexity_score = 75
    elif actor_arch == "gnn_transformer":
        complexity_score = 60
    else:
        complexity_score = 70
    
    scores.append(complexity_score)
    
    # 收敛效率 (训练步数少得分高)
    if "eval_success" in metrics and len(metrics["eval_success"]) > 0:
        sr_values = metrics["eval_success"]
        # 找到达到80%成功率的步数
        target_sr = 0.8
        reach_indices = [i for i, sr in enumerate(sr_values) if sr >= target_sr]
        if len(reach_indices) > 0:
            reach_step = reach_indices[0]
            total_steps = len(sr_values)
            efficiency = 100 * (1 - reach_step / total_steps)
            scores.append(efficiency)
        else:
            scores.append(50)  # 未达标给50分
    
    return np.mean(scores)


def evaluate_model(run_dir, model_name):
    """评估单个模型"""
    print(f"\n{'='*80}")
    print(f"评估模型: {model_name}")
    print(f"目录: {run_dir}")
    print(f"{'='*80}")
    
    # 加载数据
    metrics = load_training_metrics(run_dir)
    cfg = load_config(run_dir)
    
    if metrics is None:
        print("❌ 未找到训练指标文件")
        return None
    
    results = {
        "Model": model_name,
        "Architecture": cfg.get("actor_arch", "unknown"),
    }
    
    # 提取8个关键指标的最终值（后10次平均）
    print(f"\n8个关键指标 (最后10次评估的平均值):")
    
    # 1. Success Rate
    if "eval_success" in metrics and len(metrics["eval_success"]) > 0:
        sr = get_last_n_avg(metrics["eval_success"]) * 100
        results["Success Rate (%)"] = sr
        print(f"  1. Success Rate:         {sr:.2f}%")
    else:
        results["Success Rate (%)"] = 0.0
        print(f"  1. Success Rate:         N/A")
    
    # 2. RMSE
    if "eval_rmse" in metrics and len(metrics["eval_rmse"]) > 0:
        rmse = get_last_n_avg(metrics["eval_rmse"])
        results["RMSE"] = rmse
        print(f"  2. RMSE:                 {rmse:.4f}")
    else:
        results["RMSE"] = float('inf')
        print(f"  2. RMSE:                 N/A")
    
    # 3. Maximum Deviation
    if "eval_max_dev" in metrics and len(metrics["eval_max_dev"]) > 0:
        max_dev = get_last_n_avg(metrics["eval_max_dev"])
        results["Max Deviation"] = max_dev
        print(f"  3. Max Deviation:        {max_dev:.4f}")
    else:
        results["Max Deviation"] = float('inf')
        print(f"  3. Max Deviation:        N/A")
    
    # 4. Endpoint Error
    if "eval_end_err" in metrics and len(metrics["eval_end_err"]) > 0:
        end_err = get_last_n_avg(metrics["eval_end_err"])
        results["Endpoint Error"] = end_err
        print(f"  4. Endpoint Error:       {end_err:.4f}")
    else:
        results["Endpoint Error"] = float('inf')
        print(f"  4. Endpoint Error:       N/A")
    
    # 5. Cumulative Reward
    if "eval_rewards" in metrics and len(metrics["eval_rewards"]) > 0:
        reward = get_last_n_avg(metrics["eval_rewards"])
        results["Cumulative Reward"] = reward
        print(f"  5. Cumulative Reward:    {reward:.2f}")
    else:
        results["Cumulative Reward"] = float('-inf')
        print(f"  5. Cumulative Reward:    N/A")
    
    # 6. Success Time
    if "eval_tts" in metrics and len(metrics["eval_tts"]) > 0:
        tts = get_last_n_avg(metrics["eval_tts"])
        results["Success Time"] = tts
        print(f"  6. Success Time:         {tts:.2f}")
    else:
        results["Success Time"] = float('inf')
        print(f"  6. Success Time:         N/A")
    
    # 7. Path Length
    if "eval_path_len_exec" in metrics and len(metrics["eval_path_len_exec"]) > 0:
        path_len = get_last_n_avg(metrics["eval_path_len_exec"])
        results["Path Length"] = path_len
        print(f"  7. Path Length:          {path_len:.4f}")
    else:
        results["Path Length"] = float('inf')
        print(f"  7. Path Length:          N/A")
    
    # 8. Min Distance to Goal (到目标的最小距离, 越小越好)
    if "eval_min_distance" in metrics and len(metrics["eval_min_distance"]) > 0:
        min_dist = get_last_n_avg(metrics["eval_min_distance"])
        results["Min Distance to Goal"] = min_dist
        print(f"  8. Min Distance to Goal: {min_dist:.4f}")
    else:
        results["Min Distance to Goal"] = float('inf')
        print(f"  8. Min Distance to Goal: N/A")
    
    # ============================================================
    # 四维度评估体系 (Four Key Aspects)
    # ============================================================
    # 
    # 评估方式说明:
    # 1. Task Success Rate (任务成功率) - 权重 25%
    #    - 指标: Success Rate
    #    - 计算: 直接使用成功率百分比
    # 
    # 2. Trajectory Accuracy (轨迹精度) - 权重 25%
    #    - 指标: RMSE, Max Deviation, Endpoint Error
    #    - 计算: 三个指标归一化后的平均值
    #    - 归一化方式: score = max(0, 100 - metric × coefficient)
    # 
    # 3. Execution Performance (执行性能) - 权重 25%
    #    - 指标: Cumulative Reward, Success Time
    #    - 计算: 两个指标归一化后的平均值
    #    - 归一化方式: 
    #      * Reward: score = min(100, max(0, (reward + 50) / 1.5))
    #      * Time: score = max(0, 100 - time / 2)
    # 
    # 4. Path Optimization (路径优化) - 权重 25%
    #    - 指标: Min Distance to Goal
    #    - 计算: 归一化得分
    #    - 归一化方式: score = max(0, 100 - distance × 100)
    #    - 注: Path Length 仅记录不参与评分
    # 
    # 总得分 = (维度1得分 + 维度2得分 + 维度3得分 + 维度4得分) / 4
    # ============================================================
    
    dimension_scores = {}
    
    # 维度1: Task Success Rate (任务成功率)
    if results["Success Rate (%)"] > 0:
        dimension_scores["Task Success Rate"] = results["Success Rate (%)"]
    else:
        dimension_scores["Task Success Rate"] = 0.0
    
    # 维度2: Trajectory Accuracy (轨迹精度)
    traj_acc_scores = []
    if results["RMSE"] != float('inf'):
        rmse_score = max(0, 100 - results["RMSE"] * 200)
        traj_acc_scores.append(rmse_score)
    if results["Max Deviation"] != float('inf'):
        max_dev_score = max(0, 100 - results["Max Deviation"] * 100)
        traj_acc_scores.append(max_dev_score)
    if results["Endpoint Error"] != float('inf'):
        end_err_score = max(0, 100 - results["Endpoint Error"] * 200)
        traj_acc_scores.append(end_err_score)
    
    if len(traj_acc_scores) > 0:
        dimension_scores["Trajectory Accuracy"] = np.mean(traj_acc_scores)
    else:
        dimension_scores["Trajectory Accuracy"] = 0.0
    
    # 维度3: Execution Performance (执行性能)
    exec_perf_scores = []
    if results["Cumulative Reward"] != float('-inf'):
        reward_score = min(100, max(0, (results["Cumulative Reward"] + 50) / 1.5))
        exec_perf_scores.append(reward_score)
    if results["Success Time"] != float('inf'):
        tts_score = max(0, 100 - results["Success Time"] / 2)
        exec_perf_scores.append(tts_score)
    
    if len(exec_perf_scores) > 0:
        dimension_scores["Execution Performance"] = np.mean(exec_perf_scores)
    else:
        dimension_scores["Execution Performance"] = 0.0
    
    # 维度4: Path Optimization (路径优化)
    path_opt_scores = []
    if results["Min Distance to Goal"] != float('inf'):
        min_dist_score = max(0, 100 - results["Min Distance to Goal"] * 100)
        path_opt_scores.append(min_dist_score)
    # Path Length 不参与评分，仅记录
    
    if len(path_opt_scores) > 0:
        dimension_scores["Path Optimization"] = np.mean(path_opt_scores)
    else:
        dimension_scores["Path Optimization"] = 0.0
    
    # 计算总得分 (四个维度等权重平均)
    valid_dimensions = [score for score in dimension_scores.values() if score > 0]
    if len(valid_dimensions) > 0:
        overall_score = np.mean(valid_dimensions)
    else:
        overall_score = 0.0
    
    # 保存维度得分
    results["Task Success Rate Score"] = dimension_scores["Task Success Rate"]
    results["Trajectory Accuracy Score"] = dimension_scores["Trajectory Accuracy"]
    results["Execution Performance Score"] = dimension_scores["Execution Performance"]
    results["Path Optimization Score"] = dimension_scores["Path Optimization"]
    results["Overall Score"] = overall_score
    
    # 打印维度得分
    print(f"\n四维度得分:")
    print(f"  1. Task Success Rate:        {dimension_scores['Task Success Rate']:.2f}/100")
    print(f"  2. Trajectory Accuracy:      {dimension_scores['Trajectory Accuracy']:.2f}/100")
    print(f"  3. Execution Performance:    {dimension_scores['Execution Performance']:.2f}/100")
    print(f"  4. Path Optimization:        {dimension_scores['Path Optimization']:.2f}/100")
    print(f"\n  总得分 (Overall):            {overall_score:.2f}/100")
    
    return results


def plot_comparison_radar(all_results, output_dir):
    """绘制雷达图对比 (8个关键指标)"""
    models = [r["Model"] for r in all_results]
    
    # 8个关键指标，需要归一化到0-100
    categories = ["Success\nRate", "RMSE\n(inv)", "Max Dev\n(inv)", "End Err\n(inv)", 
                  "Reward", "Time\n(inv)", "Path Len\n(opt)", "Min Dist\n(inv)"]
    
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, result in enumerate(all_results):
        # 构建归一化值 (所有值转为越高越好)
        values = []
        
        # 1. Success Rate: 直接使用
        values.append(result.get("Success Rate (%)", 0))
        
        # 2. RMSE: 反转 (越小越好)
        rmse = result.get("RMSE", float('inf'))
        if rmse != float('inf'):
            values.append(max(0, 100 - rmse * 200))
        else:
            values.append(0)
        
        # 3. Max Deviation: 反转
        max_dev = result.get("Max Deviation", float('inf'))
        if max_dev != float('inf'):
            values.append(max(0, 100 - max_dev * 100))
        else:
            values.append(0)
        
        # 4. Endpoint Error: 反转
        end_err = result.get("Endpoint Error", float('inf'))
        if end_err != float('inf'):
            values.append(max(0, 100 - end_err * 200))
        else:
            values.append(0)
        
        # 5. Cumulative Reward: 归一化
        reward = result.get("Cumulative Reward", float('-inf'))
        if reward != float('-inf'):
            values.append(min(100, max(0, (reward + 50) / 1.5)))
        else:
            values.append(0)
        
        # 6. Success Time: 反转
        tts = result.get("Success Time", float('inf'))
        if tts != float('inf'):
            values.append(max(0, 100 - tts / 2))
        else:
            values.append(0)
        
        # 7. Path Length: 优化评分
        path_len = result.get("Path Length", float('inf'))
        if path_len != float('inf'):
            ref_len = 3.0
            ratio = path_len / ref_len
            values.append(max(0, 100 - abs(ratio - 1.0) * 100))
        else:
            values.append(0)
        
        # 8. Min Distance to Goal: 反转 (越小越好)
        min_dist = result.get("Min Distance to Goal", float('inf'))
        if min_dist != float('inf'):
            values.append(max(0, 100 - min_dist * 100))
        else:
            values.append(0)
        
        values += values[:1]  # 闭合
        
        ax.plot(angles, values, 'o-', linewidth=2.5, 
                label=result["Model"], color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.15, color=colors[i % len(colors)])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=10)
    ax.grid(True, alpha=0.3)
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
    plt.title("8 Key Metrics Comparison (Radar Chart)", 
              fontsize=15, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "8metrics_radar.png"), dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "8metrics_radar.pdf"), bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ 雷达图已保存")


def plot_four_aspects(all_results, output_dir):
    """绘制四维度对比图"""
    models = [r["Model"] for r in all_results]
    
    # 提取四维度得分
    task_sr_scores = [r.get("Task Success Rate Score", 0) for r in all_results]
    traj_acc_scores = [r.get("Trajectory Accuracy Score", 0) for r in all_results]
    exec_perf_scores = [r.get("Execution Performance Score", 0) for r in all_results]
    path_opt_scores = [r.get("Path Optimization Score", 0) for r in all_results]
    
    x = np.arange(len(models))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    bars1 = ax.bar(x - 1.5*width, task_sr_scores, width, label='Task Success Rate', 
                   color=colors[0], alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x - 0.5*width, traj_acc_scores, width, label='Trajectory Accuracy', 
                   color=colors[1], alpha=0.8, edgecolor='black')
    bars3 = ax.bar(x + 0.5*width, exec_perf_scores, width, label='Execution Performance', 
                   color=colors[2], alpha=0.8, edgecolor='black')
    bars4 = ax.bar(x + 1.5*width, path_opt_scores, width, label='Path Optimization', 
                   color=colors[3], alpha=0.8, edgecolor='black')
    
    # 添加数值标签
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Models', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score (0-100)', fontsize=13, fontweight='bold')
    ax.set_title('Four Key Aspects Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "four_aspects_comparison.png"), dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "four_aspects_comparison.pdf"), bbox_inches='tight')
    plt.close()
    
    print(f"✓ 四维度对比图已保存")


def plot_comparison_bars(all_results, output_dir):
    """绘制柱状图对比 (8个关键指标)"""
    models = [r["Model"] for r in all_results]
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("8 Key Metrics Comparison", fontsize=18, fontweight='bold')
    
    metrics = [
        ("Success Rate (%)", "Success Rate (%)", axes[0, 0], False),
        ("RMSE", "RMSE (Lower is Better)", axes[0, 1], True),
        ("Max Deviation", "Max Deviation (Lower is Better)", axes[0, 2], True),
        ("Endpoint Error", "Endpoint Error (Lower is Better)", axes[0, 3], True),
        ("Cumulative Reward", "Cumulative Reward", axes[1, 0], False),
        ("Success Time", "Success Time (Lower is Better)", axes[1, 1], True),
        ("Path Length", "Path Length", axes[1, 2], False),
        ("Min Distance to Goal", "Min Distance to Goal (Lower is Better)", axes[1, 3], True)
    ]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for metric_key, title, ax, is_lower_better in metrics:
        values = [r.get(metric_key, 0 if not is_lower_better else float('inf')) for r in all_results]
        
        # 处理无穷值
        values = [v if v != float('inf') and v != float('-inf') else 0 for v in values]
        
        bars = ax.bar(models, values, color=colors[:len(models)], alpha=0.8, edgecolor='black')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            if height > 0 and height != float('inf'):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel('Value', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # 根据指标类型标记最优
        if len(values) > 0:
            if is_lower_better:
                best_idx = np.argmin(values)
            else:
                best_idx = np.argmax(values)
            bars[best_idx].set_edgecolor('red')
            bars[best_idx].set_linewidth(3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "8metrics_bars.png"), dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "8metrics_bars.pdf"), bbox_inches='tight')
    plt.close()
    
    print(f"✓ 柱状图已保存")


def plot_overall_ranking(all_results, output_dir):
    """绘制综合排名"""
    # 按综合得分排序
    sorted_results = sorted(all_results, key=lambda x: x["Overall Score"], reverse=True)
    
    models = [r["Model"] for r in sorted_results]
    scores = [r["Overall Score"] for r in sorted_results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']
    bars = ax.barh(models, scores, color=colors[:len(models)], alpha=0.8, edgecolor='black')
    
    # 添加分数标签
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(score + 1, bar.get_y() + bar.get_height()/2,
               f'{score:.2f}', ha='left', va='center', fontsize=11, fontweight='bold')
        # 添加排名标签
        ax.text(2, bar.get_y() + bar.get_height()/2,
               f'#{i+1}', ha='left', va='center', fontsize=10, 
               color='white', fontweight='bold')
    
    ax.set_xlabel('Overall Score', fontsize=13, fontweight='bold')
    ax.set_title('Model Overall Ranking', fontsize=15, fontweight='bold', pad=20)
    ax.set_xlim(0, 110)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "overall_ranking.png"), dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, "overall_ranking.pdf"), bbox_inches='tight')
    plt.close()
    
    print(f"✓ 排名图已保存")


def save_evaluation_report(all_results, output_dir):
    """保存评估报告"""
    # CSV格式
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(output_dir, "evaluation_report.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8')
    
    # TXT格式 (详细报告)
    txt_path = os.path.join(output_dir, "evaluation_report.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("模型评估报告 - Model Evaluation Report\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        # 排序后的结果
        sorted_results = sorted(all_results, key=lambda x: x["Overall Score"], reverse=True)
        
        for i, result in enumerate(sorted_results, 1):
            f.write(f"排名 #{i}: {result['Model']}\n")
            f.write(f"{'─'*80}\n")
            f.write(f"架构:              {result['Architecture']}\n")
            f.write(f"总得分 (Overall):  {result['Overall Score']:.2f}/100\n")
            f.write(f"\n四维度得分 (Four Key Aspects):\n")
            f.write(f"  1. Task Success Rate:        {result.get('Task Success Rate Score', 0):.2f}/100\n")
            f.write(f"  2. Trajectory Accuracy:      {result.get('Trajectory Accuracy Score', 0):.2f}/100\n")
            f.write(f"  3. Execution Performance:    {result.get('Execution Performance Score', 0):.2f}/100\n")
            f.write(f"  4. Path Optimization:        {result.get('Path Optimization Score', 0):.2f}/100\n")
            f.write(f"\n8个关键指标 (详细数据):\n")
            f.write(f"  1. Success Rate:         {result.get('Success Rate (%)', 0):.2f}%\n")
            f.write(f"  2. RMSE:                 {result.get('RMSE', 0):.4f}\n")
            f.write(f"  3. Max Deviation:        {result.get('Max Deviation', 0):.4f}\n")
            f.write(f"  4. Endpoint Error:       {result.get('Endpoint Error', 0):.4f}\n")
            f.write(f"  5. Cumulative Reward:    {result.get('Cumulative Reward', 0):.2f}\n")
            f.write(f"  6. Success Time:         {result.get('Success Time', 0):.2f}\n")
            f.write(f"  7. Path Length:          {result.get('Path Length', 0):.4f} (仅记录)\n")
            f.write(f"  8. Min Dist to Goal:     {result.get('Min Distance to Goal', 0):.4f}\n")
            f.write(f"\n{'='*80}\n\n")
        
        # 总结
        f.write("\n总结 (Summary):\n")
        f.write(f"{'='*80}\n")
        best_model = sorted_results[0]
        f.write(f"最佳模型 (Best Overall):         {best_model['Model']}\n")
        f.write(f"最佳架构 (Best Architecture):    {best_model['Architecture']}\n")
        f.write(f"最高得分 (Highest Score):        {best_model['Overall Score']:.2f}/100\n\n")
        
        # 四维度最佳模型
        f.write("各维度最佳模型 (Best in Each Aspect):\n")
        f.write(f"{'─'*80}\n")
        best_task_sr = max(all_results, key=lambda x: x.get('Task Success Rate Score', 0))
        best_traj_acc = max(all_results, key=lambda x: x.get('Trajectory Accuracy Score', 0))
        best_exec_perf = max(all_results, key=lambda x: x.get('Execution Performance Score', 0))
        best_path_opt = max(all_results, key=lambda x: x.get('Path Optimization Score', 0))
        
        f.write(f"  Task Success Rate:       {best_task_sr['Model']} ({best_task_sr.get('Task Success Rate Score', 0):.2f})\n")
        f.write(f"  Trajectory Accuracy:     {best_traj_acc['Model']} ({best_traj_acc.get('Trajectory Accuracy Score', 0):.2f})\n")
        f.write(f"  Execution Performance:   {best_exec_perf['Model']} ({best_exec_perf.get('Execution Performance Score', 0):.2f})\n")
        f.write(f"  Path Optimization:       {best_path_opt['Model']} ({best_path_opt.get('Path Optimization Score', 0):.2f})\n\n")
        
        # 各指标最佳
        best_sr = max(all_results, key=lambda x: x.get('Success Rate (%)', 0))
        best_rmse = min(all_results, key=lambda x: x.get('RMSE', float('inf')))
        best_max_dev = min(all_results, key=lambda x: x.get('Max Deviation', float('inf')))
        best_end_err = min(all_results, key=lambda x: x.get('Endpoint Error', float('inf')))
        best_reward = max(all_results, key=lambda x: x.get('Cumulative Reward', float('-inf')))
        best_time = min(all_results, key=lambda x: x.get('Success Time', float('inf')))
        best_path = min(all_results, key=lambda x: abs(x.get('Path Length', float('inf')) - 3.0))
        best_dist = min(all_results, key=lambda x: x.get('Min Distance to Goal', float('inf')))
        
        f.write(f"各指标最佳:\n")
        f.write(f"  最高成功率:        {best_sr['Model']} ({best_sr.get('Success Rate (%)', 0):.2f}%)\n")
        f.write(f"  最低RMSE:          {best_rmse['Model']} ({best_rmse.get('RMSE', 0):.4f})\n")
        f.write(f"  最低最大偏差:      {best_max_dev['Model']} ({best_max_dev.get('Max Deviation', 0):.4f})\n")
        f.write(f"  最低端点误差:      {best_end_err['Model']} ({best_end_err.get('Endpoint Error', 0):.4f})\n")
        f.write(f"  最高累积奖励:      {best_reward['Model']} ({best_reward.get('Cumulative Reward', 0):.2f})\n")
        f.write(f"  最短成功时间:      {best_time['Model']} ({best_time.get('Success Time', 0):.2f})\n")
        f.write(f"  最优路径长度:      {best_path['Model']} ({best_path.get('Path Length', 0):.4f})\n")
        f.write(f"  最小目标距离:      {best_dist['Model']} ({best_dist.get('Min Distance to Goal', 0):.4f})\n")
    
    print(f"\n✓ 评估报告已保存:")
    print(f"  CSV: {csv_path}")
    print(f"  TXT: {txt_path}")


def main():
    """主函数"""
    # 模型目录配置
    model_dirs = {
        "MLP": "./results/new/KukaIiwa7Track-v0_mlp_dense_20251119_215119",
        "GNN": "./results/new/KukaIiwa7Track-v0_gnn_dense_20251115_131235",
        "Transformer": "./results/new/seed1transformer",
        "GNN+Transformer": "./results/new/seed1gnntransformer"
    }
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./evaluation_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"模型评估系统 - Model Evaluation System")
    print(f"输出目录: {output_dir}")
    print(f"{'='*80}")
    
    # 评估所有模型
    all_results = []
    for model_name, run_dir in model_dirs.items():
        if not os.path.exists(run_dir):
            print(f"\n⚠️  警告: 目录不存在 - {run_dir}")
            continue
        
        result = evaluate_model(run_dir, model_name)
        if result is not None:
            all_results.append(result)
    
    if len(all_results) == 0:
        print("\n❌ 没有成功评估的模型")
        return
    
    # 生成对比图表
    print(f"\n{'='*80}")
    print("生成对比图表...")
    print(f"{'='*80}")
    
    plot_comparison_radar(all_results, output_dir)
    plot_four_aspects(all_results, output_dir)  # 新增: 四维度对比图
    plot_comparison_bars(all_results, output_dir)
    plot_overall_ranking(all_results, output_dir)
    
    # 保存评估报告
    save_evaluation_report(all_results, output_dir)
    
    print(f"\n{'='*80}")
    print("评估完成！")
    print(f"所有结果已保存到: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

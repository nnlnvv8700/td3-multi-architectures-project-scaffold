# -*- coding: utf-8 -*-
# quick_test_evaluate.py - 快速测试增强版评估脚本

"""
快速测试新的评估功能（无需实际训练）

运行方式：
python quick_test_evaluate.py
"""

import os
import sys
import subprocess
import numpy as np
import json

# 添加项目路径
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def create_mock_training_run(base_dir, arch_name, seed=None, num_evals=50):
    """创建模拟的训练结果目录"""
    from datetime import datetime

    # 创建目录名
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_suffix = f"_seed{seed}" if seed is not None else ""
    run_name = f"KukaIiwa7Track-v0_{arch_name}_dense_{stamp}{seed_suffix}"
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # 生成模拟指标数据
    # 不同架构有不同的性能特征
    base_performance = {
        "mlp": {"success": 0.85, "reward": 120, "rmse": 0.045, "jerk": 2.3},
        "gnn": {"success": 0.92, "reward": 145, "rmse": 0.032, "jerk": 1.9},
        "transformer": {"success": 0.94, "reward": 155, "rmse": 0.028, "jerk": 1.8},
        "gnn_transformer": {"success": 0.97, "reward": 165, "rmse": 0.021, "jerk": 1.5}
    }

    perf = base_performance.get(arch_name, base_performance["mlp"])

    # 添加种子相关的随机性
    if seed is not None:
        np.random.seed(seed)

    # 生成评估指标（模拟训练过程）
    eval_steps = list(range(5000, 5000 * (num_evals + 1), 5000))

    # 模拟训练曲线（逐渐改善）
    eval_rewards = []
    eval_success = []
    eval_rmse = []
    eval_jerk = []

    for i in range(num_evals):
        progress = (i + 1) / num_evals
        noise = np.random.randn() * 0.05

        # 奖励逐渐增加
        reward = perf["reward"] * progress + noise * 10
        eval_rewards.append(max(0, reward))

        # 成功率逐渐增加
        success = min(1.0, perf["success"] * progress + noise * 0.05)
        eval_success.append(max(0, success))

        # RMSE逐渐减小
        rmse = perf["rmse"] * (1 - 0.8 * progress) + abs(noise) * 0.005
        eval_rmse.append(max(0.001, rmse))

        # Jerk逐渐减小
        jerk = perf["jerk"] * (1 - 0.6 * progress) + abs(noise) * 0.1
        eval_jerk.append(max(0.1, jerk))

    metrics = {
        "eval_steps": eval_steps,
        "eval_rewards": eval_rewards,
        "eval_success": eval_success,
        "eval_tts": [150 - i * 2 for i in range(num_evals)],
        "eval_min_distance": [0.05 - i * 0.0008 for i in range(num_evals)],
        "eval_rmse": eval_rmse,
        "eval_max_dev": [r * 1.5 for r in eval_rmse],
        "eval_end_err": [r * 0.8 for r in eval_rmse],
        "eval_path_len_exec": [1.2 - i * 0.01 for i in range(num_evals)],
        "eval_path_len_ref": [1.0] * num_evals,
        "eval_jerk": eval_jerk,
    }

    # 保存 metrics.json
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # 保存 config.json
    config = {
        "env": "KukaIiwa7Track-v0",
        "actor_arch": arch_name,
        "max_timesteps": 500000,
        "batch_size": 512,
        "seed": seed
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"✓ 创建模拟训练: {run_dir}")
    return run_dir

def main():
    print("=" * 70)
    print("快速测试增强版评估脚本")
    print("=" * 70)

    # 创建测试目录
    test_base = os.path.join(PROJECT_ROOT, "test_results")
    os.makedirs(test_base, exist_ok=True)

    print("\n[步骤 1] 创建模拟训练数据...")

    # 为每个架构创建3个随机种子的训练结果
    architectures = ["mlp", "gnn", "transformer", "gnn_transformer"]
    seeds = [42, 123, 456]

    run_dirs = {}
    for arch in architectures:
        run_dirs[arch] = []
        for seed in seeds:
            run_dir = create_mock_training_run(test_base, arch, seed=seed, num_evals=50)
            run_dirs[arch].append(run_dir)

    print(f"\n✓ 成功创建 {len(architectures) * len(seeds)} 个模拟训练结果")

    # 测试1: 单次运行评估
    print("\n" + "=" * 70)
    print("[测试 1] 单次运行评估（每个架构取第一个种子）")
    print("=" * 70)

    single_dirs = [run_dirs[arch][0] for arch in architectures]
    cmd1 = [sys.executable, os.path.join(PROJECT_ROOT, "training", "evaluate_enhanced.py"), "--run_dirs", *single_dirs, "--ma_window", "3", "--out", os.path.join(PROJECT_ROOT, "test_plots", "single_run_test")]

    print("\n命令:")
    print(cmd1)
    print("\n执行中...")
    subprocess.run(cmd1, check=True)

    # 测试2: 多种子评估
    print("\n" + "=" * 70)
    print("[测试 2] 多种子评估（显示均值±标准差）")
    print("=" * 70)

    seed_runs_args = []
    for arch in architectures:
        label = arch.replace("_", "+").upper()
        if label == "GNN+TRANSFORMER":
            label = "GNN+Transformer"
        dirs_str = ",".join(run_dirs[arch])
        seed_runs_args.append(f"{label}:{dirs_str}")

    cmd2 = [sys.executable, os.path.join(PROJECT_ROOT, "training", "evaluate_enhanced.py"), "--multi_seed", "--seed_runs", *seed_runs_args, "--ma_window", "3", "--out", os.path.join(PROJECT_ROOT, "test_plots", "multi_seed_test")]

    print("\n命令:")
    print(cmd2)
    print("\n执行中...")
    subprocess.run(cmd2, check=True)

    print("\n" + "=" * 70)
    print("[测试完成]")
    print("=" * 70)
    print(f"\n模拟数据位置: {test_base}")
    print(f"测试输出位置: {os.path.join(PROJECT_ROOT, 'test_plots')}")
    print("\n提示:")
    print("  - 查看 test_plots/single_run_test/ 查看单次运行结果")
    print("  - 查看 test_plots/multi_seed_test/ 查看多种子结果（含标准差）")
    print("  - 模拟数据可安全删除（test_results/ 目录）")
    print("\n要测试轨迹生成功能，需要实际训练的模型文件。")

if __name__ == "__main__":
    main()

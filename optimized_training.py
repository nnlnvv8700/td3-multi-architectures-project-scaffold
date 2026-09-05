
"""
优化的训练脚本 - 基于三架构对比分析结果

根据分析结果自动调整训练配置：
- MLP: 300k步，学习率3e-5（已验证100%成功率）
- GNN: 500k步，学习率1e-5（降低不稳定性）
- Transformer: 500k步，学习率1e-5（提高收敛性）
- GNN+Transformer: 500k步，学习率2e-5（论文主角）

使用方法：
    # 训练单个架构
    python optimized_training.py --arch mlp
    python optimized_training.py --arch gnn
    python optimized_training.py --arch transformer
    python optimized_training.py --arch gnn_transformer

    # 训练所有架构（用于论文对比）
    python optimized_training.py --all
"""

import subprocess
from pathlib import Path
import sys
import os
from datetime import datetime

# 架构配置（基于分析结果优化）
ARCHITECTURE_CONFIGS = {
    "mlp": {
        "max_timesteps": 300_000,  # MLP 收敛快，300k足够
        "actor_lr": 3e-5,
        "critic_lr": 3e-5,
        "expl_noise": 0.02,
        "description": "MLP (Baseline) - 已验证100%成功率",
        "expected_performance": "100% 成功率"
    },
    "gnn": {
        "max_timesteps": 500_000,  # GNN 需要更多时间
        "actor_lr": 1e-5,          # 降低学习率提高稳定性
        "critic_lr": 1e-5,
        "expl_noise": 0.01,        # 降低探索噪声
        "description": "GNN (空间建模) - 优化配置",
        "expected_performance": "目标：> 50% 成功率"
    },
    "transformer": {
        "max_timesteps": 500_000,  # Transformer 需要更多数据
        "actor_lr": 1e-5,          # 更保守的学习率
        "critic_lr": 1e-5,
        "expl_noise": 0.02,
        "description": "Transformer (时序建模) - 优化配置",
        "expected_performance": "目标：> 70% 成功率"
    },
    "gnn_transformer": {
        "max_timesteps": 500_000,  # GT-TD3 主角
        "actor_lr": 2e-5,          # 平衡学习率
        "critic_lr": 2e-5,
        "expl_noise": 0.02,
        "description": "GNN+Transformer (论文主角) - GT-TD3",
        "expected_performance": "目标：> 90% 成功率（超越MLP）"
    }
}

def train_architecture(arch_name, config, env="KukaIiwa7Track-v0"):
    """训练单个架构"""
    print("\n" + "="*80)
    print(f"开始训练: {config['description']}")
    print(f"预期性能: {config['expected_performance']}")
    print("="*80)

    # 构建训练命令
    cmd = [
        sys.executable, "-m", "training.train_experiment",
        "--actor_arch", arch_name,
        "--env", env,
        "--max_timesteps", str(config["max_timesteps"]),
        "--actor_lr", str(config["actor_lr"]),
        "--critic_lr", str(config["critic_lr"]),
        "--expl_noise", str(config["expl_noise"])
    ]

    print(f"\n执行命令: {' '.join(cmd)}\n")

    # 设置环境变量（避免OpenMP冲突）
    env_vars = os.environ.copy()
    env_vars["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # 执行训练
    try:
        result = subprocess.run(cmd, env=env_vars, cwd=Path(__file__).resolve().parent)

        if result.returncode == 0:
            print(f"\n✓ {arch_name} 训练完成！")
            return True
        else:
            print(f"\n✗ {arch_name} 训练失败（退出码: {result.returncode}）")
            return False

    except KeyboardInterrupt:
        print(f"\n⚠️ {arch_name} 训练被用户中断")
        return False
    except Exception as e:
        print(f"\n✗ {arch_name} 训练出错: {e}")
        return False

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="优化的架构训练脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 训练MLP（快速验证）
    python optimized_training.py --arch mlp

    # 训练GNN+Transformer（论文主角）
    python optimized_training.py --arch gnn_transformer

    # 训练所有架构（论文对比）
    python optimized_training.py --all

    # 使用Reach环境
    python optimized_training.py --arch mlp --env KukaIiwa7Track-v0
        """
    )

    parser.add_argument("--arch", type=str,
                       choices=list(ARCHITECTURE_CONFIGS.keys()),
                       help="要训练的架构")
    parser.add_argument("--all", action="store_true",
                       help="训练所有架构（按推荐顺序）")
    parser.add_argument("--env", type=str, default="KukaIiwa7Track-v0",
                       help="环境名称 (default: KukaIiwa7Track-v0)")

    args = parser.parse_args()

    # 显示配置信息
    print("\n" + "="*80)
    print("优化训练配置（基于分析结果）")
    print("="*80)
    print("\n当前配置:")
    for arch, config in ARCHITECTURE_CONFIGS.items():
        print(f"\n{arch}:")
        print(f"  描述: {config['description']}")
        print(f"  训练步数: {config['max_timesteps']:,}")
        print(f"  学习率: Actor={config['actor_lr']}, Critic={config['critic_lr']}")
        print(f"  探索噪声: {config['expl_noise']}")
        print(f"  预期: {config['expected_performance']}")

    print("\n" + "="*80)

    # 训练逻辑
    if args.all:
        # 训练所有架构（推荐顺序）
        print("\n开始训练所有架构（推荐顺序）:")
        print("1. MLP (基线，快速)")
        print("2. Transformer (时序)")
        print("3. GNN (空间)")
        print("4. GNN+Transformer (主角)")

        input("\n按 Enter 继续...")

        results = {}

        # 推荐训练顺序
        training_order = ["mlp", "transformer", "gnn", "gnn_transformer"]

        for arch in training_order:
            config = ARCHITECTURE_CONFIGS[arch]
            success = train_architecture(arch, config, args.env)
            results[arch] = success

            if not success:
                print(f"\n⚠️ {arch} 训练失败，是否继续训练其他架构？")
                choice = input("继续 (y/n): ").lower()
                if choice != 'y':
                    break

        # 显示汇总
        print("\n" + "="*80)
        print("训练汇总")
        print("="*80)
        for arch, success in results.items():
            status = "✓ 成功" if success else "✗ 失败"
            print(f"{arch:20s} {status}")

        print("\n可使用以下命令查看结果对比:")
        print("python analyze_three_architectures.py")

    elif args.arch:
        # 训练单个架构
        config = ARCHITECTURE_CONFIGS[args.arch]
        train_architecture(args.arch, config, args.env)

    else:
        parser.print_help()
        print("\n请指定 --arch 或 --all")

if __name__ == "__main__":
    main()

"""
GAT (Graph Attention Network) 训练脚本
算法大师级优化版本

架构升级：
1. 静态图卷积 (GCN) -> 动态图注意力 (GATv2)
2. 隐藏层维度 64 -> 256 (提升4倍容量)
3. 引入 Multi-Head Attention (4 heads)
4. 添加 LayerNorm + Residual + Dropout (Transformer级稳定性)

训练参数：
- Batch Size: 1024 (大批次稳定梯度)
- Learning Rate: 1e-5 (标准Transformer学习率)
- Noise: 0.005 (低噪声精细微调)
"""

import sys
import subprocess
from pathlib import Path

def main():
    # 🎯 大师级优化配置
    optimized_params = {
        # 架构 (代码已升级为GAT)
        "--actor_arch": "gnn",

        # 训练参数
        "--max_timesteps": "500000",
        "--start_timesteps": "25000",
        "--batch_size": "1024",      # 大批次
        "--actor_lr": "1e-5",        # 恢复到1e-5 (GAT容量大，可以承受稍大学习率)
        "--critic_lr": "1e-5",
        "--expl_noise": "0.005",     # 低噪声
        "--eval_freq": "5000",
        "--save_dir": "./results",
    }

    cmd = [sys.executable, "-m", "training.train_experiment"]
    for key, value in optimized_params.items():
        cmd.extend([key, value])

    # 允许用户覆盖参数 (如 --seed)
    cmd.extend(sys.argv[1:])

    print("="*60)
    print("🧠 GAT (Graph Attention) 大师级训练")
    print("="*60)
    print("\n🚀 架构升级:")
    print("  • Core: Multi-Head Graph Attention (4 Heads)")
    print("  • Capacity: 256 hidden units (was 64)")
    print("  • Stability: LayerNorm + Residuals + Dropout")
    print("\n⚙️ 训练配置:")
    print(f"  • Batch Size: {optimized_params['--batch_size']}")
    print(f"  • Learning Rate: {optimized_params['--actor_lr']}")
    print(f"  • Noise: {optimized_params['--expl_noise']}")
    print("\n📌 实验目标:")
    print("  • 检验运动学链图先验是否改善轨迹控制")
    print("  • 使用多随机种子报告成功率与轨迹质量")
    print("="*60 + "\n")

    subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], check=True)

if __name__ == "__main__":
    main()

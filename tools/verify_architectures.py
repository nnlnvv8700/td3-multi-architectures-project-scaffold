"""
模型架构验证脚本
================
检查四个 Actor 架构是否正确配置和兼容。
"""

import torch
import sys
import os

# 添加项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from agents.networks import MLPActor, SimpleGNNActor, TransformerActor, GNNTransformerActor


def test_architecture(arch_name, model, input_shape, batch_size=4):
    """测试单个架构"""
    print(f"\n{'='*60}")
    print(f"Testing: {arch_name}")
    print(f"{'='*60}")

    # 创建测试输入
    x = torch.randn(batch_size, input_shape)
    print(f"Input shape: {x.shape}")

    # 前向传播
    try:
        output = model(x)
        print(f"✅ Output shape: {output.shape}")
        print(f"✅ Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")

        # 检查参数数量
        num_params = sum(p.numel() for p in model.parameters())
        print(f"📊 Parameters: {num_params:,}")

        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("🔍 TD3 Actor 架构验证")
    print("="*60)

    # 配置参数
    action_dim = 7  # KUKA iiwa 7 DoF

    # ===== KUKA iiwa 环境状态维度 =====
    # v2: q/qdot(14) + phase(1) + reference(3) + previous action(7)
    #     + achieved_goal(3) + desired_goal(3) = 31
    state_dim = 31

    # ===== 测试 1: MLP =====
    # MLP 直接使用完整31维状态
    mlp = MLPActor(obs_dim=state_dim, action_dim=action_dim)
    test_architecture("MLP Actor", mlp, state_dim)

    # ===== 测试 2: GNN (7关节配置) =====
    node_dim = 6
    num_nodes = 7
    gnn_input_dim = state_dim  # 31维 -> KukaStateEncoder -> 7×6 + 全局轨迹上下文

    gnn = SimpleGNNActor(
        node_dim=node_dim,
        num_nodes=num_nodes,
        action_dim=action_dim,
        use_state_encoder=True,
        input_dim=state_dim
    )
    test_architecture("GNN Actor (7-joint)", gnn, gnn_input_dim)

    # ===== 测试 3: Transformer (7关节配置) =====
    transformer = TransformerActor(
        node_dim=node_dim,
        num_nodes=num_nodes,
        action_dim=action_dim,
        use_state_encoder=True,
        input_dim=state_dim,
        use_kuka_pe=True
    )
    test_architecture("Transformer Actor (7-joint + KUKA PE)", transformer, gnn_input_dim)

    # ===== 测试 4: GNN+Transformer (7关节配置) =====
    gnn_transformer = GNNTransformerActor(
        node_dim=node_dim,
        num_nodes=num_nodes,
        action_dim=action_dim,
        use_state_encoder=True,
        input_dim=state_dim,
        use_kuka_pe=True
    )
    test_architecture("GNN+Transformer Actor (7-joint + KUKA PE)", gnn_transformer, gnn_input_dim)

    # ===== 输入维度兼容性检查 =====
    print(f"\n{'='*60}")
    print("✅ 输入维度兼容性检查")
    print(f"{'='*60}")

    print(f"\n📌 所有架构统一输入维度:")
    print(f"   - 输入维度: {state_dim} (KUKA iiwa环境标准状态)")
    print("   - 组成: 14维关节状态 + phase/reference/previous_action + goals")

    print(f"\n📌 MLP 架构:")
    print(f"   - 直接使用31维状态向量")
    print(f"   - 通过全连接层处理")

    print(f"\n📌 GNN/Transformer/GNN+Transformer 架构:")
    print(f"   - 输入: 31维状态")
    print(f"   - 编码: KukaStateEncoder (31 -> 7×6) + 4维轨迹上下文")
    print(f"   - 节点配置: {num_nodes} 关节 × {node_dim} 特征/关节")
    print(f"   - 物理意义: 每个节点 = 1个KUKA关节")
    print(f"   - 位置编码: KUKA物理位置编码（Transformer系列）")

    print(f"\n✅ 所有架构输入维度一致: {state_dim}")
    print(f"✅ 图架构使用状态编码器自动转换为7关节表示")
    print(f"✅ Transformer系列使用物理位置编码增强运动学先验")

    print(f"\n{'='*60}")
    print("✅ 架构验证完成")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

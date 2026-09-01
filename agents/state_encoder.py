"""
状态编码器 - 将环境状态编码为图节点表示
===========================================

针对KUKA iiwa 7-DoF机械臂的特征工程。

作者: Auto-generated
日期: 2025-11-10
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class KukaStateEncoder(nn.Module):
    """
    KUKA iiwa状态编码器

    将 GoalEnv 展平状态编码为7个关节的节点表示，兼容旧20维与新31维观测。

    输入状态结构:
        - joint_positions (7,): q₁, q₂, ..., q₇
        - joint_velocities (7,): q̇₁, q̇₂, ..., q̇₇
        - achieved_goal (3,): 末端执行器位置 [x, y, z]
        - desired_goal (3,): 目标位置 [x, y, z]
        v1总计20维；v2额外包含轨迹阶段、当前参考点和前一动作，总计31维。

    输出节点表示:
        - 7个节点，每个节点 node_dim 维特征
        - 每个节点代表一个关节
    """

    def __init__(self, node_dim: int = 3, encoding_mode: str = "simple"):
        """
        初始化编码器

        Args:
            node_dim: 每个节点的特征维度（3或4推荐）
            encoding_mode: 编码模式
                - "simple": 简单编码（位置、速度、目标信息）
                - "rich": 丰富编码（包含更多特征）
        """
        super().__init__()
        self.node_dim = node_dim
        self.encoding_mode = encoding_mode

        if node_dim not in [3, 4, 5, 6]:
            print(f"Warning: node_dim={node_dim} 不是常用配置，推荐使用6维完整目标方向编码")

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        编码状态为节点表示

        Args:
            state: (B, 20) 展平的状态向量

        Returns:
            nodes: (B, 7 * node_dim) 7个关节节点的展平表示
        """
        batch_size = state.shape[0]

        # 解析公共关节状态
        joint_pos = state[:, 0:7]      # (B, 7) 关节位置
        joint_vel = state[:, 7:14]     # (B, 7) 关节速度

        # v1 flattened state (20): q, qdot, achieved, desired
        # v2 flattened state (31): q, qdot, phase, current_ref, prev_action,
        #                          achieved, desired
        if state.shape[1] >= 31:
            prev_action = state[:, 18:25]
            achieved = state[:, 25:28]
            desired = state[:, 28:31]
        else:
            prev_action = torch.zeros_like(joint_pos)
            achieved = state[:, 14:17]
            desired = state[:, 17:20]

        if self.node_dim == 3:
            # 模式1: [关节位置, 关节速度, 归一化目标距离]
            goal_vector = desired - achieved  # (B, 3)
            goal_dist = torch.norm(goal_vector, dim=1, keepdim=True)  # (B, 1)
            goal_dist_normalized = torch.tanh(goal_dist)  # 归一化到[-1,1]

            # 广播到每个关节: (B, 1) -> (B, 7)
            goal_feature = goal_dist_normalized.expand(-1, 7)  # (B, 7)

            # 组合: (B, 7, 3)
            nodes = torch.stack([joint_pos, joint_vel, goal_feature], dim=2)

        elif self.node_dim == 4:
            # 模式2: [关节位置, 关节速度, 目标距离, 目标方向X]
            goal_vector = desired - achieved  # (B, 3)
            goal_dist = torch.norm(goal_vector, dim=1, keepdim=True)  # (B, 1)
            goal_dist_normalized = torch.tanh(goal_dist)
            goal_dir_x = goal_vector[:, 0:1]  # 目标方向的X分量

            # 广播: (B, 1) -> (B, 7)
            goal_dist_feat = goal_dist_normalized.expand(-1, 7)  # (B, 7)
            goal_dir_feat = goal_dir_x.expand(-1, 7)  # (B, 7)

            # 组合: (B, 7, 4)
            nodes = torch.stack([joint_pos, joint_vel, goal_dist_feat, goal_dir_feat], dim=2)

        elif self.node_dim == 5:
            # 模式3: [关节位置, 关节速度, 目标距离, 目标方向X, 目标方向Y]
            goal_vector = desired - achieved  # (B, 3)
            goal_dist = torch.norm(goal_vector, dim=1, keepdim=True)
            goal_dist_normalized = torch.tanh(goal_dist)
            goal_dir_x = goal_vector[:, 0:1]
            goal_dir_y = goal_vector[:, 1:2]

            # 广播: (B, 1) -> (B, 7)
            goal_dist_feat = goal_dist_normalized.expand(-1, 7)
            goal_dir_x_feat = goal_dir_x.expand(-1, 7)
            goal_dir_y_feat = goal_dir_y.expand(-1, 7)

            # 组合: (B, 7, 5)
            nodes = torch.stack([
                joint_pos, joint_vel,
                goal_dist_feat, goal_dir_x_feat, goal_dir_y_feat
            ], dim=2)

        elif self.node_dim == 6:
            # v2: [关节位置, 关节速度, 上一步动作, 目标方向X/Y/Z]
            # 每个关节节点都保留完整三维目标方向，同时携带自身动作历史。
            goal_vector = desired - achieved  # (B, 3)
            goal_dir_x = goal_vector[:, 0:1]
            goal_dir_y = goal_vector[:, 1:2]
            goal_dir_z = goal_vector[:, 2:3]
            goal_dir_x_feat = goal_dir_x.expand(-1, 7)
            goal_dir_y_feat = goal_dir_y.expand(-1, 7)
            goal_dir_z_feat = goal_dir_z.expand(-1, 7)

            if state.shape[1] >= 31:
                nodes = torch.stack([
                    joint_pos, joint_vel, prev_action,
                    goal_dir_x_feat, goal_dir_y_feat, goal_dir_z_feat,
                ], dim=2)
            else:
                # 旧 20 维 checkpoint 的兼容语义。
                goal_dist = torch.tanh(torch.norm(goal_vector, dim=1, keepdim=True))
                nodes = torch.stack([
                    joint_pos, joint_vel, goal_dist.expand(-1, 7),
                    goal_dir_x_feat, goal_dir_y_feat, goal_dir_z_feat,
                ], dim=2)

        else:
            # 通用模式：将所有信息均匀分配（简单padding/重复）
            # (B, 7, 2) -> 位置和速度
            base_features = torch.stack([joint_pos, joint_vel], dim=2)

            # 目标特征
            goal_vector = desired - achieved
            goal_dist = torch.norm(goal_vector, dim=1, keepdim=True)
            goal_features = torch.cat([goal_dist, goal_vector], dim=1)  # (B, 4)

            # 广播并padding
            goal_feat_per_node = goal_features.unsqueeze(1).expand(-1, 7, -1)[:, :, :self.node_dim-2]
            nodes = torch.cat([base_features, goal_feat_per_node], dim=2)

        # 展平: (B, 7, node_dim) -> (B, 7*node_dim)
        return nodes.reshape(batch_size, -1)


class StateEncoderWithPadding(nn.Module):
    """
    带Padding的状态编码器

    当 7*node_dim ≠ 20 时，使用可学习的线性层进行映射。
    """

    def __init__(self, input_dim: int = 20, num_nodes: int = 7, node_dim: int = 3):
        """
        Args:
            input_dim: 输入状态维度（20）
            num_nodes: 节点数量（7）
            node_dim: 每个节点的特征维度
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        output_dim = num_nodes * node_dim

        if input_dim != output_dim:
            # 使用线性层映射
            self.projection = nn.Linear(input_dim, output_dim)
        else:
            # 如果维度相同，使用恒等映射
            self.projection = nn.Identity()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (B, input_dim)

        Returns:
            encoded: (B, num_nodes * node_dim)
        """
        return self.projection(state)


def create_kuka_adjacency_matrix(num_nodes: int = 7, add_self_loops: bool = False) -> torch.Tensor:
    """
    创建KUKA iiwa的运动学链邻接矩阵

    Args:
        num_nodes: 关节数量（默认7）
        add_self_loops: 是否添加自环

    Returns:
        adjacency: (num_nodes, num_nodes) 邻接矩阵
    """
    A = torch.zeros(num_nodes, num_nodes)

    # 串联链式连接
    for i in range(num_nodes - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0

    # 可选自环
    if add_self_loops:
        for i in range(num_nodes):
            A[i, i] = 1.0

    return A


def extract_kuka_global_context(state: torch.Tensor) -> torch.Tensor:
    """提取图节点之外的轨迹上下文：[phase, current_ref - achieved]。"""
    if state.shape[1] < 31:
        return state.new_zeros((state.shape[0], 0))
    phase = state[:, 14:15]
    current_ref = state[:, 15:18]
    achieved = state[:, 25:28]
    return torch.cat([phase, current_ref - achieved], dim=1)


# ===== 便捷函数 =====

def get_recommended_config(num_nodes: int = 7) -> dict:
    """
    获取推荐的配置

    Args:
        num_nodes: 节点数量

    Returns:
        配置字典
    """
    if num_nodes == 7:
        return {
            "num_nodes": 7,
            "node_dim": 6,
            "encoding": "simple",
            "description": "7关节配置 - 完整目标方向 + 动作历史"
        }
    elif num_nodes == 4:
        return {
            "num_nodes": 4,
            "node_dim": 5,  # 4*5=20，完全匹配
            "encoding": None,  # 不需要编码器
            "description": "4节点配置 - 抽象表示（原配置）"
        }
    else:
        # 自动计算最接近20的node_dim
        node_dim = max(1, round(20 / num_nodes))
        return {
            "num_nodes": num_nodes,
            "node_dim": node_dim,
            "encoding": "padding",
            "description": f"{num_nodes}节点配置 - 需要padding"
        }


if __name__ == "__main__":
    # 测试编码器
    print("="*60)
    print("测试 KUKA 状态编码器")
    print("="*60)

    batch_size = 4
    state = torch.randn(batch_size, 20)

    for node_dim in [3, 4, 5]:
        encoder = KukaStateEncoder(node_dim=node_dim)
        encoded = encoder(state)
        print(f"\nnode_dim={node_dim}:")
        print(f"  输入: {state.shape}")
        print(f"  输出: {encoded.shape} (期望: ({batch_size}, {7*node_dim}))")
        print(f"  ✅" if encoded.shape == (batch_size, 7*node_dim) else "  ❌")

    # 测试邻接矩阵
    print(f"\n{'='*60}")
    print("KUKA 运动学链邻接矩阵 (7×7):")
    print("="*60)
    adj = create_kuka_adjacency_matrix(7, add_self_loops=False)
    print(adj)

    print(f"\n{'='*60}")
    print("推荐配置:")
    print("="*60)
    for n in [4, 7]:
        cfg = get_recommended_config(n)
        print(f"\n{cfg['description']}")
        print(f"  num_nodes: {cfg['num_nodes']}")
        print(f"  node_dim: {cfg['node_dim']}")
        print(f"  总维度: {cfg['num_nodes'] * cfg['node_dim']}")

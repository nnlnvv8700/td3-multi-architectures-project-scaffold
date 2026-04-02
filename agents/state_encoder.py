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
    KUKA iiwa状态编码器（增强版）

    将20维展平状态编码为7个关节的节点表示。

    输入状态结构:
        - joint_positions (7,): q₁, q₂, ..., q₇
        - joint_velocities (7,): q̇₁, q̇₂, ..., q̇₇
        - achieved_goal (3,): 末端执行器位置 [x, y, z]
        - desired_goal (3,): 目标位置 [x, y, z]
        总计: 20维

    输出维度: (B, 7 * (node_dim + 1))
        每个节点特征 = [手工特征(node_dim维)] + [可学习关节标识嵌入(1维)]
        注意：实际输出维度是 7×(node_dim+1)，而非 7×node_dim

    改进：
        1. 可学习关节索引嵌入（Joint Index Embedding）：每个关节一个可学习标量，
           以 concat 方式追加到节点特征末尾，使所有特征维度均可感知关节身份。
           （旧版为 add 到 dim-0，其余维度无法区分关节）
        2. 累积旋转角特征：从基座到每个关节的累积转动量，捕捉"臂形"信息
        3. 目标方向归一化为单位向量（完整3D方向，不受距离尺度影响）
        4. 速度归一化（避免大速度主导特征空间）
    """

    def __init__(self, node_dim: int = 3, encoding_mode: str = "simple"):
        super().__init__()
        self.node_dim = node_dim
        self.encoding_mode = encoding_mode

        # 可学习关节索引嵌入（1维标量/关节）：concat 到节点特征末尾
        # 初始化为均匀间隔 [-1, 1]，赋予有序先验；输出维度实际为 node_dim+1
        self.joint_embed = nn.Parameter(
            torch.linspace(-1.0, 1.0, 7).unsqueeze(1)  # (7, 1)
        )

        if node_dim not in [3, 4, 5, 6]:
            print(f"Warning: node_dim={node_dim} 可能不是最优选择，推荐3、4、5或6")
        # 实际输出每节点特征维度 = node_dim + 1（含可学习嵌入）
        self.output_node_dim = node_dim + 1

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (B, 20) 展平的状态向量

        Returns:
            nodes: (B, 7 * (node_dim+1)) 7个关节节点的展平表示
        """
        batch_size = state.shape[0]

        # 解析状态
        joint_pos = state[:, 0:7]      # (B, 7) 关节位置（弧度）
        joint_vel = state[:, 7:14]     # (B, 7) 关节速度
        achieved = state[:, 14:17]     # (B, 3) 末端位置
        desired = state[:, 17:20]      # (B, 3) 目标位置

        # 目标向量（目标-当前）
        goal_vector = desired - achieved           # (B, 3)
        goal_dist = torch.norm(goal_vector, dim=1, keepdim=True).clamp_min(1e-6)  # (B, 1)

        # 归一化目标方向为单位向量（消除距离尺度影响）
        goal_dir = goal_vector / goal_dist                                         # (B, 3) 单位向量

        # 归一化距离（工作空间直径约1.2m）
        goal_dist_norm = goal_dist / 1.2                                           # (B, 1)

        # 速度归一化（关节速度范围约 ±1.5 rad/s）
        joint_vel_norm = joint_vel / 1.5                                           # (B, 7)

        # 累积旋转角：从基座到关节 i 的累加，捕捉"臂形"状态
        # cumsum: q_cum[i] = q1 + q2 + ... + q(i+1)，归一化到 [-1, 1]
        # KUKA iiwa 关节范围约 ±170° ≈ ±3.0 rad；7关节累积最大约 21 rad
        joint_pos_cum = torch.cumsum(joint_pos, dim=1) / (7 * 3.0)                # (B, 7)

        if self.node_dim == 3:
            # [位置, 速度归一化, 距离] - 紧凑版
            goal_dist_feat = goal_dist_norm.expand(-1, 7)                          # (B, 7)
            nodes = torch.stack([joint_pos, joint_vel_norm, goal_dist_feat], dim=2)

        elif self.node_dim == 4:
            # [位置, 速度, 距离, 累积旋转] - 加入臂形信息
            goal_dist_feat = goal_dist_norm.expand(-1, 7)
            nodes = torch.stack([joint_pos, joint_vel_norm, goal_dist_feat, joint_pos_cum], dim=2)

        elif self.node_dim == 5:
            # [位置, 速度, 累积旋转, 目标距离, 目标方向X] - 平衡版
            goal_dist_feat = goal_dist_norm.expand(-1, 7)
            goal_dir_x = goal_dir[:, 0:1].expand(-1, 7)
            nodes = torch.stack([
                joint_pos, joint_vel_norm, joint_pos_cum,
                goal_dist_feat, goal_dir_x
            ], dim=2)

        elif self.node_dim == 6:
            # [位置, 速度, 累积旋转, 目标距离, 目标方向X, Y]
            # 目标方向Z = sqrt(1 - X² - Y²) 可由网络推断，6维避免冗余
            goal_dist_feat = goal_dist_norm.expand(-1, 7)
            goal_dir_x = goal_dir[:, 0:1].expand(-1, 7)
            goal_dir_y = goal_dir[:, 1:2].expand(-1, 7)
            nodes = torch.stack([
                joint_pos, joint_vel_norm, joint_pos_cum,
                goal_dist_feat, goal_dir_x, goal_dir_y
            ], dim=2)

        else:
            # 通用模式：位置 + 速度 + 目标信息均匀分配
            goal_dist_feat = goal_dist_norm.expand(-1, 7)
            goal_dir_x = goal_dir[:, 0:1].expand(-1, 7)
            goal_dir_y = goal_dir[:, 1:2].expand(-1, 7)
            goal_dir_z = goal_dir[:, 2:3].expand(-1, 7)
            base = torch.stack([joint_pos, joint_vel_norm, joint_pos_cum,
                                 goal_dist_feat, goal_dir_x, goal_dir_y, goal_dir_z], dim=2)
            nodes = base[:, :, :self.node_dim]

        # 拼接可学习关节索引嵌入（concat 而非 add）
        # joint_embed: (7, 1) → 拼接到每个节点特征末尾
        # 优势：所有特征维度都能通过后续线性层感知关节身份，
        #       而加法只修改 dim-0，其余特征无法区分关节
        # 代价：输出维度变为 node_dim+1（调用方需知晓）
        embed = self.joint_embed.unsqueeze(0).expand(batch_size, -1, -1)  # (B, 7, 1)
        nodes = torch.cat([nodes, embed], dim=2)   # (B, 7, node_dim+1)

        # 展平: (B, 7, node_dim+1) -> (B, 7*(node_dim+1))
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
            "node_dim": 3,  # 7*3=21，接近20
            "encoding": "simple",
            "description": "7关节配置 - 真实运动学结构"
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

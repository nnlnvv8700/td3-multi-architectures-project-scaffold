import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 导入状态编码器（可选）
try:
    from .state_encoder import (
        KukaStateEncoder,
        StateEncoderWithPadding,
        extract_kuka_global_context,
        create_kuka_adjacency_matrix,
    )
    ENCODER_AVAILABLE = True
except ImportError:
    ENCODER_AVAILABLE = False
    print("[Warning] state_encoder not available, using default encoding")


# -----------------------------
# KUKA iiwa 物理位置编码
# -----------------------------
class KukaPositionalEncoding(nn.Module):
    """
    基于KUKA iiwa 7-DoF串联运动学链的位置编码

    特点:
        1. 正弦位置编码（原始Transformer）- 捕获序列顺序
        2. 距离感知偏置 - 反映关节间物理距离
        3. 层次化权重 - 基座附近关节影响更大

    Args:
        d_model: Transformer隐藏层维度
        num_joints: 关节数量（KUKA iiwa为7）
        max_len: 最大序列长度（默认等于关节数）
        use_distance_bias: 是否启用距离偏置

    Example:
        >>> pe = KukaPositionalEncoding(d_model=128, num_joints=7)
        >>> x = torch.randn(32, 7, 128)  # (batch, joints, d_model)
        >>> x_pe = pe(x)  # 添加位置信息
    """
    def __init__(
        self,
        d_model: int,
        num_joints: int = 7,
        max_len: int = None,
        use_distance_bias: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.num_joints = num_joints
        max_len = max_len or num_joints

        # 1. 正弦位置编码（标准Transformer做法）
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            -(math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

        # 2. 距离感知偏置（反映运动学链结构）
        if use_distance_bias:
            # 距离矩阵：关节i到关节j的跳数
            dist_matrix = torch.zeros(num_joints, num_joints)
            for i in range(num_joints):
                for j in range(num_joints):
                    dist_matrix[i, j] = abs(i - j)

            # 可学习的距离嵌入（将距离转换为特征）
            self.distance_embed = nn.Embedding(
                num_embeddings=num_joints,  # 最大距离=6（关节0到关节6）
                embedding_dim=d_model
            )
            self.register_buffer('dist_matrix', dist_matrix.long())
        else:
            self.distance_embed = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_joints, d_model)

        Returns:
            x + positional_encoding: (batch_size, num_joints, d_model)
        """
        batch_size, seq_len, d_model = x.shape

        # 添加正弦位置编码
        pe = self.pe[:seq_len, :].unsqueeze(0)  # (1, seq_len, d_model)
        x = x + pe

        # 添加距离感知偏置（可选）
        if self.distance_embed is not None:
            # 对每个关节，计算其与所有其他关节的距离嵌入的平均
            dist_emb = self.distance_embed(self.dist_matrix)  # (num_joints, num_joints, d_model)
            dist_bias = dist_emb.mean(dim=1)  # (num_joints, d_model) - 平均距离特征
            x = x + dist_bias.unsqueeze(0)[:, :seq_len, :]

        return x


# -----------------------------
# MLP (baseline)
# -----------------------------
class MLPActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=(256, 256), max_action=1.0):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, action_dim))
        self.net = nn.Sequential(*layers)
        self.max_action = max_action

    def forward(self, x):
        return self.max_action * torch.tanh(self.net(x))


class MLPCritic(nn.Module):
    """🔧 优化版Critic：更深更宽的网络 + LayerNorm提升稳定性"""
    def __init__(self, obs_dim, action_dim, hidden=(512, 512, 256)):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], hidden[2])  # 🎯 新增第3层
        self.out = nn.Linear(hidden[2], 1)

        # 🎯 添加 LayerNorm 提升训练稳定性
        self.norm1 = nn.LayerNorm(hidden[0])
        self.norm2 = nn.LayerNorm(hidden[1])
        self.norm3 = nn.LayerNorm(hidden[2])

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        x = self.norm1(F.relu(self.fc1(x)))
        x = self.norm2(F.relu(self.fc2(x)))
        x = self.norm3(F.relu(self.fc3(x)))
        return self.out(x)


# -----------------------------
# Simple GNN (Upgraded to GAT)
# -----------------------------
class GATLayer(nn.Module):
    """
    Graph Attention Layer (GATv2 style)
    支持动态注意力权重的图卷积层
    """
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        assert self.head_dim * num_heads == out_dim, "out_dim must be divisible by num_heads"

        self.scale = self.head_dim ** -0.5

        # Q, K, V projections
        self.q_proj = nn.Linear(in_dim, out_dim)
        self.k_proj = nn.Linear(in_dim, out_dim)
        self.v_proj = nn.Linear(in_dim, out_dim)

        # Output projection
        self.out_proj = nn.Linear(out_dim, out_dim)

        # Normalization & Regularization
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 4, out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, adj=None):
        # x: (B, N, D)
        B, N, D = x.shape
        residual = x

        # 1. Multi-Head Attention
        # (B, N, D) -> (B, N, H, d) -> (B, H, N, d)
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention Scores: (B, H, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply Mask (if adj provided)
        if adj is not None:
            # adj: (N, N) -> (1, 1, N, N)
            # 假设 adj=1 表示连接，adj=0 表示断开
            # 我们希望对断开的连接施加 -inf
            mask = (adj == 0).float() * -1e9
            attn = attn + mask.view(1, 1, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Aggregate: (B, H, N, N) @ (B, H, N, d) -> (B, H, N, d)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.out_proj(out)
        out = self.dropout(out)

        # Add & Norm 1
        x = self.norm1(residual + out)

        # 2. Feed Forward
        residual = x
        out = self.ffn(x)

        # Add & Norm 2
        x = self.norm2(residual + out)

        return x


class SimpleGNNActor(nn.Module):
    """
    GNN Actor (Upgraded to GAT)

    改进点:
    1. 引入 GAT (Graph Attention) 机制，替代静态邻接矩阵
    2. 增加网络容量 (hidden_dim 64 -> 256)
    3. 使用 LayerNorm 和 Residual 连接
    4. 保持 GNN 的接口兼容性
    """
    def __init__(
        self,
        node_dim,
        num_nodes,
        action_dim,
        adjacency=None,
        node_hidden=256,  # 🎯 增大容量: 64 -> 256
        max_action=1.0,
        use_state_encoder=False,
        input_dim=20
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.use_state_encoder = use_state_encoder
        self.context_dim = 4 if use_state_encoder and input_dim >= 31 else 0

        # 状态编码器
        if use_state_encoder and ENCODER_AVAILABLE:
            if input_dim == num_nodes * node_dim:
                self.state_encoder = nn.Identity()
            elif num_nodes == 7 and ENCODER_AVAILABLE:
                self.state_encoder = KukaStateEncoder(node_dim=node_dim)
                print(f"[GNN] Using KukaStateEncoder: {input_dim} -> {num_nodes}×{node_dim}")
            else:
                self.state_encoder = StateEncoderWithPadding(input_dim, num_nodes, node_dim)
                print(f"[GNN] Using StateEncoderWithPadding")
        else:
            self.state_encoder = nn.Identity()

        # 输入投影: node_dim -> node_hidden
        self.input_proj = nn.Sequential(
            nn.Linear(node_dim, node_hidden),
            nn.LayerNorm(node_hidden),
            nn.ReLU()
        )

        # GAT Layers (2层)
        # 使用全连接图注意力，允许任意关节间通信
        self.gat1 = GATLayer(node_hidden, node_hidden, num_heads=4, dropout=0.05)
        self.gat2 = GATLayer(node_hidden, node_hidden, num_heads=4, dropout=0.05)

        # 图结构 (作为Mask使用)
        if adjacency is not None:
            adj = torch.as_tensor(adjacency, dtype=torch.float32)
        else:
            adj = (
                create_kuka_adjacency_matrix(num_nodes, add_self_loops=True)
                if num_nodes == 7 else torch.ones(num_nodes, num_nodes)
            )
        self.register_buffer("adjacency", adj)

        # 输出层
        self.readout = nn.Sequential(
            nn.Linear(node_hidden * num_nodes + self.context_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(256, action_dim),
        )
        self.max_action = max_action

    def forward(self, x):
        b = x.shape[0]

        # 1. 状态编码
        raw_state = x
        context = extract_kuka_global_context(raw_state) if self.context_dim else None
        x = self.state_encoder(x)
        nodes = x.view(b, self.num_nodes, self.node_dim)

        # 2. 输入投影
        h = self.input_proj(nodes)  # (B, N, H)

        # 3. GAT Layers (Message Passing)
        h = self.gat1(h, self.adjacency)
        h = self.gat2(h, self.adjacency)

        # 4. Readout
        flat = h.reshape(b, -1)
        if context is not None:
            flat = torch.cat([flat, context], dim=1)
        return self.max_action * torch.tanh(self.readout(flat))


# -----------------------------
# Transformer (node sequence)
# -----------------------------
class TransformerActor(nn.Module):
    """
    Input: (B, num_nodes * node_dim)

    新增支持:
    - use_state_encoder=True: 自动将展平状态编码为节点表示
    - use_kuka_pe=True: 使用KUKA物理位置编码（仅当num_nodes=7时）
    """
    def __init__(
        self,
        node_dim,
        num_nodes,
        action_dim,
        d_model=128,
        nhead=4,
        num_layers=2,
        max_action=1.0,
        use_state_encoder=False,
        input_dim=20,
        use_kuka_pe=True  # 🎯 新增：是否使用KUKA物理位置编码
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.use_state_encoder = use_state_encoder
        self.use_kuka_pe = use_kuka_pe
        self.context_dim = 4 if use_state_encoder and input_dim >= 31 else 0

        # 状态编码器
        if use_state_encoder and ENCODER_AVAILABLE:
            if input_dim == num_nodes * node_dim:
                self.state_encoder = nn.Identity()
            elif num_nodes == 7 and ENCODER_AVAILABLE:
                self.state_encoder = KukaStateEncoder(node_dim=node_dim)
                print(f"[Transformer] Using KukaStateEncoder: {input_dim} -> {num_nodes}×{node_dim}")
            else:
                self.state_encoder = StateEncoderWithPadding(
                    input_dim=input_dim,
                    num_nodes=num_nodes,
                    node_dim=node_dim
                )
                print(f"[Transformer] Using StateEncoderWithPadding: {input_dim} -> {num_nodes}×{node_dim}")
        else:
            self.state_encoder = nn.Identity()

        self.input_proj = nn.Linear(node_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 🎯 位置编码选择：KUKA物理位置编码 vs 可学习位置编码
        if use_kuka_pe and num_nodes == 7:
            self.pos_encoding = KukaPositionalEncoding(
                d_model=d_model,
                num_joints=num_nodes,
                use_distance_bias=True
            )
            print(f"[Transformer] Using KukaPositionalEncoding with distance bias")
        else:
            # 回退到可学习位置编码
            self.pos_encoding = None
            self.pos_embed = nn.Parameter(torch.zeros(num_nodes, d_model))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            if use_kuka_pe and num_nodes != 7:
                print(f"[Transformer] Warning: use_kuka_pe=True but num_nodes={num_nodes} (expect 7), using learnable PE")

        self.readout = nn.Sequential(
            nn.Linear(num_nodes * d_model + self.context_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )
        self.max_action = max_action

    def forward(self, x):
        b = x.shape[0]

        # 状态编码
        raw_state = x
        context = extract_kuka_global_context(raw_state) if self.context_dim else None
        x = self.state_encoder(x)

        seq = x.view(b, self.num_nodes, self.node_dim)      # (B, N, D)
        t = self.input_proj(seq)                             # (B, N, d_model)

        # 🎯 应用位置编码
        if self.pos_encoding is not None:
            # KUKA物理位置编码
            t = self.pos_encoding(t)
        else:
            # 可学习位置编码
            t = t + self.pos_embed.unsqueeze(0)

        t = self.transformer(t)                              # (B, N, d_model)
        flat = t.reshape(b, -1)
        if context is not None:
            flat = torch.cat([flat, context], dim=1)
        return self.max_action * torch.tanh(self.readout(flat))


# -----------------------------
# GNN + Transformer (final)
# -----------------------------
class GNNTransformerActor(nn.Module):
    """
    Pipeline (Upgraded to GAT + Transformer):
      1) GAT Layers (Graph Attention) -> 提取局部拓扑特征
      2) Linear proj to d_model + KUKA物理位置编码
      3) Transformer encoder -> 提取全局序列特征
      4) Readout MLP -> action

    Expected input shape: (B, num_nodes * node_dim)

    新增支持:
    - use_state_encoder=True: 自动将展平状态编码为节点表示
    - use_kuka_pe=True: 使用KUKA物理位置编码（仅当num_nodes=7时）
    """
    def __init__(
        self,
        node_dim,
        num_nodes,
        action_dim,
        adjacency=None,
        node_hidden=256,  # 🎯 增大容量: 64 -> 256
        d_model=128,
        nhead=4,
        num_layers=2,
        max_action=1.0,
        use_state_encoder=False,
        input_dim=20,
        use_kuka_pe=True  # 🎯 新增：是否使用KUKA物理位置编码
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.use_state_encoder = use_state_encoder
        self.use_kuka_pe = use_kuka_pe
        self.context_dim = 4 if use_state_encoder and input_dim >= 31 else 0

        # 状态编码器
        if use_state_encoder and ENCODER_AVAILABLE:
            if input_dim == num_nodes * node_dim:
                self.state_encoder = nn.Identity()
            elif num_nodes == 7 and ENCODER_AVAILABLE:
                self.state_encoder = KukaStateEncoder(node_dim=node_dim)
                print(f"[GNN+Transformer] Using KukaStateEncoder: {input_dim} -> {num_nodes}×{node_dim}")
            else:
                self.state_encoder = StateEncoderWithPadding(
                    input_dim=input_dim,
                    num_nodes=num_nodes,
                    node_dim=node_dim
                )
                print(f"[GNN+Transformer] Using StateEncoderWithPadding: {input_dim} -> {num_nodes}×{node_dim}")
        else:
            self.state_encoder = nn.Identity()

        # 1) GAT Feature Extractor (替代原来的简单MLP+聚合)
        # 先将输入投影到 hidden dim
        self.input_proj_gat = nn.Sequential(
            nn.Linear(node_dim, node_hidden),
            nn.LayerNorm(node_hidden),
            nn.ReLU()
        )

        # GAT Layer (提取图特征)
        self.gat_layer = GATLayer(node_hidden, node_hidden, num_heads=4, dropout=0.05)

        # 图结构 (作为Mask使用)
        if adjacency is not None:
            adj = torch.as_tensor(adjacency, dtype=torch.float32)
        else:
            adj = (
                create_kuka_adjacency_matrix(num_nodes, add_self_loops=True)
                if num_nodes == 7 else torch.ones(num_nodes, num_nodes)
            )
        self.register_buffer("adjacency", adj)

        # 2) project to transformer d_model + 位置编码
        self.bridge_proj = nn.Linear(node_hidden, d_model)

        # 🎯 位置编码选择：KUKA物理位置编码 vs 可学习位置编码
        if use_kuka_pe and num_nodes == 7:
            self.pos_encoding = KukaPositionalEncoding(
                d_model=d_model,
                num_joints=num_nodes,
                use_distance_bias=True
            )
            print(f"[GNN+Transformer] Using KukaPositionalEncoding with distance bias")
        else:
            # 回退到可学习位置编码
            self.pos_encoding = None
            self.pos_embed = nn.Parameter(torch.zeros(num_nodes, d_model))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            if use_kuka_pe and num_nodes != 7:
                print(f"[GNN+Transformer] Warning: use_kuka_pe=True but num_nodes={num_nodes} (expect 7), using learnable PE")

        # 3) transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4) readout to action
        self.readout = nn.Sequential(
            nn.Linear(num_nodes * d_model + self.context_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )
        self.max_action = max_action

    def forward(self, x):
        b = x.shape[0]

        # 状态编码
        raw_state = x
        context = extract_kuka_global_context(raw_state) if self.context_dim else None
        x = self.state_encoder(x)

        nodes = x.view(b, self.num_nodes, self.node_dim)           # (B, N, D)

        # 1) GAT Feature Extraction
        h = self.input_proj_gat(nodes)                              # (B, N, H)
        h = self.gat_layer(h, self.adjacency)                       # (B, N, H)

        # 2) Bridge to Transformer
        t = self.bridge_proj(h)                                     # (B, N, d_model)

        # 🎯 应用位置编码
        if self.pos_encoding is not None:
            # KUKA物理位置编码
            t = self.pos_encoding(t)
        else:
            # 可学习位置编码
            t = t + self.pos_embed.unsqueeze(0)

        # 3) Transformer Global Modeling
        t = self.transformer(t)                                     # (B, N, d_model)

        # 4) Readout
        flat = t.reshape(b, -1)
        if context is not None:
            flat = torch.cat([flat, context], dim=1)
        return self.max_action * torch.tanh(self.readout(flat))

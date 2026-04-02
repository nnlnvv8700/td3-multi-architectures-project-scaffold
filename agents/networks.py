import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

# 导入状态编码器(可选)
try:
    from .state_encoder import KukaStateEncoder, StateEncoderWithPadding
    ENCODER_AVAILABLE = True
except ImportError:
    ENCODER_AVAILABLE = False
    print("[Warning] state_encoder not available, using default encoding")


# -----------------------------------------------------------------------
# PAPE: Physics-Aware Positional Encoding（KUKA iiwa 7 URDF真实参数版）
#
# 从 PyBullet pybullet_data/kuka_iiwa/model.urdf 直接读取的真实参数：
#
#   关节轴方向：所有7个关节在各自局部坐标系中均为 +Z 轴旋转
#   （注意：这是局部坐标系表示，不是绝对空间方向）
#
#   关节origin距离（相对父关节坐标系的欧氏偏移，单位mm）：
#     J0: 132.9  J1: 87.8  J2: 151.4  J3: 90.6
#     J4: 122.3  J5: 141.1  J6: 80.4
#
#   关节运动范围（度）：
#     J0:±170  J1:±120  J2:±170  J3:±120
#     J4:±170  J5:±120  J6:±175
#
# 物理先验设计策略（因所有轴相同，axis_bonus无意义，改为以下三项）：
#
#   A. 累积连杆距离惩罚：关节间实际空间距离越大，直接耦合越弱
#      → 用 origin 欧氏距离累加，归一化为 [0,1]
#
#   B. 运动范围耦合奖励：两关节的运动范围乘积越大，协同空间越大
#      → range_i × range_j 归一化，近端大范围关节（J0/J2/J4）协同更强
#
#   C. 关节对称性结构：KUKA iiwa 7 呈 3+1+3 对称结构（奇偶关节交替负载）
#      → J0,J2,J4 为奇数索引（170°），J1,J3,J5 为偶数（120°），J6 特殊（175°）
#      → 同类型关节（同为大范围或小范围）之间有更强的结构耦合
#
#   P(i,j) = range_coupling(i,j) - link_dist(i,j) + symmetry_bonus(i,j)
# -----------------------------------------------------------------------

class PAPEBias(nn.Module):
    """
    Physics-Aware Positional Encoding Bias for KUKA iiwa 7.

    使用从 URDF 直接读取的真实参数构建物理先验矩阵：
      - 关节 origin 欧氏距离（mm）：[132.9, 87.8, 151.4, 90.6, 122.3, 141.1, 80.4]
      - 关节运动范围（度）：[170, 120, 170, 120, 170, 120, 175]
      - 所有关节轴均为局部 +Z（轴方向一致，不再用于区分耦合类型）

    注意力 logit 偏置 = scale_h * P(i,j) + tanh(learned_bias_h)
      P[i,j] > 0 → 耦合强（范围大+距离近） → 注意力高
      P[i,j] < 0 → 耦合弱（距离远）        → 注意力低

    hierarchy_weights: 由关节运动范围归一化到 [0.6, 1.2]
    """
    # ── KUKA iiwa 7 URDF 真实参数（从 pybullet_data 实测）────────────────
    # 每个关节相对于父坐标系的 origin 欧氏距离 (mm)
    _LINK_ORIGIN_DIST = [132.9, 87.8, 151.4, 90.6, 122.3, 141.1, 80.4]
    # 关节运动范围 (度，单侧最大值)
    _JOINT_RANGES = [170.0, 120.0, 170.0, 120.0, 170.0, 120.0, 175.0]
    # 所有关节轴在局部坐标系下均为 +Z（从 URDF getJointInfo axis 实测）
    # 不再使用 _JOINT_AXES 做正交判断

    def __init__(self, num_joints: int = 7, nhead: int = 4):
        super().__init__()
        self.num_joints = num_joints
        self.nhead = nhead

        # ── 构建固定物理先验矩阵 P(N, N) ──────────────────────────────────
        P = self._build_physics_prior(num_joints)
        self.register_buffer('physics_prior', P)  # (N, N) 固定，不参与梯度

        # ── 每 head 可学习 scale（初值+1.0，保持P的物理语义方向）──────────
        self.prior_scale = nn.Parameter(torch.ones(nhead))  # (H,)

        # ── 可学习残差偏置（初值0，tanh限幅）──────────────────────────────
        self.learned_bias = nn.Parameter(torch.zeros(nhead, num_joints, num_joints))

        # ── 层次化输出权重：由关节运动范围归一化到 [0.6, 1.2] ──────────────
        ranges = torch.tensor(self._JOINT_RANGES[:num_joints], dtype=torch.float32)
        r_min, r_max = ranges.min(), ranges.max()
        hw = 0.6 + 0.6 * (ranges - r_min) / (r_max - r_min + 1e-6)
        self.register_buffer('hierarchy_weights', hw)  # (N,)

    @classmethod
    def _build_physics_prior(cls, num_joints: int) -> torch.Tensor:
        """
        构建 (N, N) 物理先验矩阵，基于 URDF 真实参数：

        P[i,j] = range_coupling[i,j] - link_dist[i,j] + symmetry_bonus[i,j]

        range_coupling[i,j]:
            两关节运动范围乘积归一化，范围越大协同空间越大，耦合越强。
            J0/J2/J4/J6（170/175°）之间耦合最强，J1/J3/J5（120°）最弱。

        link_dist[i,j]:
            i→j 之间累积的 origin 欧氏距离（mm），归一化到 [0,1]。
            距离越大，直接空间影响越弱，作为衰减惩罚项。

        symmetry_bonus[i,j]:
            KUKA iiwa 7 的奇偶关节对称结构奖励：
            同类型关节（同为大范围170°/175° 或 同为小范围120°）
            在结构上承担相似负载角色，有额外的结构耦合加成(+0.1)。
        """
        dists  = cls._LINK_ORIGIN_DIST[:num_joints]  # mm
        ranges = cls._JOINT_RANGES[:num_joints]       # 度

        # ── A. 累积连杆距离惩罚 ───────────────────────────────────────────
        link_dist = torch.zeros(num_joints, num_joints)
        for i in range(num_joints):
            for j in range(num_joints):
                if i == j:
                    continue
                lo, hi = min(i, j), max(i, j)
                # origin[k] 是关节 k 相对父关节的偏移，累加得到路径长度
                link_dist[i, j] = sum(dists[lo:hi])  # mm

        max_dist = link_dist.max().clamp_min(1e-6)
        link_dist = link_dist / max_dist  # 归一化 [0,1]

        # ── B. 运动范围耦合奖励 ───────────────────────────────────────────
        range_t = torch.tensor(ranges, dtype=torch.float32)   # (N,)
        # 外积：range_i * range_j，归一化到 [0,1]
        range_coupling = torch.outer(range_t, range_t)
        max_coup = range_coupling.max().clamp_min(1e-6)
        range_coupling = range_coupling / max_coup  # [0,1]
        # 对角线（自身）置0，后续 attention mask 不依赖对角线
        range_coupling.fill_diagonal_(0.0)

        # ── C. 同类型结构对称性奖励 ───────────────────────────────────────
        # 大范围（>=160°）: J0,J2,J4,J6；小范围（<160°）: J1,J3,J5
        symmetry_bonus = torch.zeros(num_joints, num_joints)
        for i in range(num_joints):
            for j in range(num_joints):
                if i == j:
                    continue
                same_type = (ranges[i] >= 160.0) == (ranges[j] >= 160.0)
                if same_type:
                    symmetry_bonus[i, j] = 0.1

        # ── 合并：P = 范围耦合 - 连杆距离 + 对称奖励 ──────────────────────
        P = range_coupling - link_dist + symmetry_bonus  # (N, N)
        return P

    def get_attn_bias(self) -> torch.Tensor:
        """Returns (nhead, N, N) attention logit bias tensor."""
        # P[i,j] > 0 → 强耦合（范围大+近距离） → scale*P>0 → softmax 更高注意力
        # P[i,j] < 0 → 弱耦合（远距离）         → scale*P<0 → softmax 更低注意力
        phys_bias = self.prior_scale.view(self.nhead, 1, 1) * self.physics_prior  # (H,N,N)
        # 残差：tanh 限幅防训练初期梯度爆炸，初值0不破坏物理先验
        return phys_bias + torch.tanh(self.learned_bias)


class PhysicsAwareTransformerLayer(nn.Module):
    """
    自定义 Transformer Encoder 层:
      1. 将 PAPEBias 的 (nhead, N, N) 偏置注入 attention logit
         （偏置由 KUKA iiwa 7 真实物理参数构建：轴方向正交耦合 + 连杆距离衰减）
      2. 层次化权重（由关节运动范围归一化）作用于 attn_out（softmax后），不破坏概率分布
      3. Pre-LayerNorm（更稳定的训练）
      4. GELU FFN
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        pape_bias: Optional[PAPEBias] = None,
    ):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.attn_drop = nn.Dropout(dropout)  # 独立 attention dropout
        self.pape_bias = pape_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, d_model)"""
        B, N, D = x.shape
        H, Dh = self.nhead, self.head_dim

        # Pre-LayerNorm Self-Attention
        residual = x
        x_ln = self.norm1(x)

        Q = self.q_proj(x_ln).view(B, N, H, Dh).transpose(1, 2)  # (B, H, N, Dh)
        K = self.k_proj(x_ln).view(B, N, H, Dh).transpose(1, 2)
        V = self.v_proj(x_ln).view(B, N, H, Dh).transpose(1, 2)

        # attention logits + PAPE 距离偏置（核心物理先验注入）
        scale = math.sqrt(Dh)
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / scale   # (B, H, N, N)

        if self.pape_bias is not None:
            attn_bias = self.pape_bias.get_attn_bias()                # (H, N, N)
            attn_logits = attn_logits + attn_bias.unsqueeze(0)        # → (B, H, N, N)

        attn_w = self.attn_drop(torch.softmax(attn_logits, dim=-1))
        attn_out = torch.matmul(attn_w, V)                            # (B, H, N, Dh)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, D)
        attn_out = self.out_proj(attn_out)                            # (B, N, D)

        # 层次化权重作用于 out_proj 之后（节点级缩放），语义清晰：
        # 关节运动范围大 → 该节点的注意力聚合输出贡献更大 → 残差中权重更高
        if self.pape_bias is not None:
            hw = self.pape_bias.hierarchy_weights[:N]                 # (N,)
            attn_out = attn_out * hw.view(1, N, 1)                    # (B, N, D)

        x = residual + self.drop(attn_out)

        # Pre-LayerNorm FFN
        x = x + self.ff(self.norm2(x))
        return x


class PAPETransformerEncoder(nn.Module):
    """
    Transformer encoder composed of PhysicsAwareTransformerLayer stacks.
    All layers share the same PAPEBias instance (shared params, more stable, fewer params).
    Sinusoidal PE is added to tokens once, before the first layer.
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        num_joints: int = 7,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.pape = PAPEBias(num_joints=num_joints, nhead=nhead)

        # 正弦位置编码(仅捕获序列顺序,与注意力偏置分离)
        position = torch.arange(num_joints).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe = torch.zeros(num_joints, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('sinusoidal_pe', pe)    # (N, d_model)

        # FFN 扩展比使用传入值（外部应传 d_model*4 获得标准Transformer配置）
        self.layers = nn.ModuleList([
            PhysicsAwareTransformerLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                pape_bias=self.pape,    # 所有层共享同一组物理先验参数
            )
            for _ in range(num_layers)
        ])
        # Pre-LN 架构在输出端需要额外 LN，稳定后续模块的输入分布
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, d_model)"""
        x = x + self.sinusoidal_pe[:x.size(1)].unsqueeze(0)
        for layer in self.layers:
            x = layer(x)
        return self.output_norm(x)


# 保留旧名以防外部代码引用(重定向到新实现)
KukaPositionalEncoding = PAPETransformerEncoder


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
    """
    升级版 Critic：
      · 三层隐藏层（256-256-128），末层缩维有助于 Q 值收缩
      · 第2、3层前加 LayerNorm：防止 Q 值量级失控，稳定 Bellman 残差
      · ELU 激活：负半轴梯度非零，缓解死神经元，Q 值估计更平滑
    """
    def __init__(self, obs_dim, action_dim, hidden=(256, 256, 128)):
        super().__init__()
        dims = [obs_dim + action_dim] + list(hidden)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i > 0:   # 第一层保留原始尺度信息，不加 LN
                layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.ELU())
        self.trunk = nn.Sequential(*layers)
        self.out = nn.Linear(hidden[-1], 1)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.out(self.trunk(x))


# -----------------------------
# Simple GNN (lightweight)
# -----------------------------
class SimpleGNNActor(nn.Module):
    """
    Input state is flattened: shape (B, num_nodes * node_dim)
    Adjacency optional. If provided, shape (N, N).
    
    For KUKA iiwa 7-DoF:
    - Can use num_nodes=7 to represent 7 joints (kinematic chain)
    - Or use default num_nodes=4 for abstract node representation
    
    - use_state_encoder=True: auto-encode 20-dim state to 7 joint nodes
    - use_state_encoder=False: assume input already has correct dims (default)
    """
    def __init__(
        self, 
        node_dim, 
        num_nodes, 
        action_dim, 
        adjacency=None, 
        node_hidden=64, 
        max_action=1.0,
        use_state_encoder=False,  # 新增参数
        input_dim=20  # 输入状态维度
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.use_state_encoder = use_state_encoder
        
        # 状态编码器(如果需要)
        if use_state_encoder and ENCODER_AVAILABLE:
            if input_dim == num_nodes * node_dim:
                # 维度匹配,不需要编码
                self.state_encoder = nn.Identity()
            elif num_nodes == 7 and ENCODER_AVAILABLE:
                # 使用KUKA专用编码器
                self.state_encoder = KukaStateEncoder(node_dim=node_dim)
                node_dim = self.state_encoder.output_node_dim  # 实际每节点维度 = node_dim+1
                print(f"[GNN] Using KukaStateEncoder: {input_dim} -> {num_nodes}x{node_dim}")
            else:
                # 使用通用padding编码器
                self.state_encoder = StateEncoderWithPadding(
                    input_dim=input_dim, 
                    num_nodes=num_nodes, 
                    node_dim=node_dim
                )
                print(f"[GNN] Using StateEncoderWithPadding: {input_dim} -> {num_nodes}x{node_dim}")
        else:
            self.state_encoder = nn.Identity()
        self.node_dim = node_dim  # 更新为实际使用的维度
        
        # per-node encoder:2层(与论文一致)
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim, node_hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(node_hidden, node_hidden),
            nn.LeakyReLU(0.1),
        )

        # 链式拓扑邻接矩阵(带自环,运动学结构归纳偏置)
        if adjacency is not None:
            adj = torch.as_tensor(adjacency, dtype=torch.float32)
        else:
            adj = self._create_kinematic_chain_adj(num_nodes, add_self_loops=True)
        self.register_buffer("adjacency", self._row_normalize(adj))

        # 2层 GNN：门控更新机制（GRU 风格）
        # update_gate 决定邻居聚合信息融入比例；aggr_proj 投影邻居消息
        self.gnn_norm = nn.ModuleList([nn.LayerNorm(node_hidden) for _ in range(2)])
        self.gnn_aggr_proj = nn.ModuleList([
            nn.Linear(node_hidden, node_hidden, bias=False) for _ in range(2)
        ])
        # 门控：sigmoid(W_z·concat[h_self, h_aggr])，决定保留自身特征的比例
        self.gnn_gate = nn.ModuleList([
            nn.Linear(node_hidden * 2, node_hidden) for _ in range(2)
        ])

        # readout: mean-pool + max-pool 双路聚合（比 flatten 关节等变、参数更少）
        self.readout = nn.Sequential(
            nn.Linear(node_hidden * 2, 256),  # concat(mean, max)
            nn.ELU(),
            nn.Linear(256, action_dim),
        )
        self.max_action = max_action

    @staticmethod
    def _create_kinematic_chain_adj(n, add_self_loops=True):
        """Kinematic chain adjacency matrix with optional self-loops."""
        A = torch.zeros(n, n)
        for i in range(n - 1):
            A[i, i + 1] = 1.0
            A[i + 1, i] = 1.0
        if add_self_loops:
            for i in range(n):
                A[i, i] = 1.0
        return A

    @staticmethod
    def _row_normalize(A):
        rowsum = A.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return A / rowsum

    def forward(self, x):
        b = x.shape[0]

        # 状态编码
        x = self.state_encoder(x)                          # (B, N*D)

        nodes = x.view(b, self.num_nodes, self.node_dim)   # (B, N, node_dim)
        h = self.node_mlp(nodes)                           # (B, N, node_hidden)

        # 2层 门控GNN（链式拓扑）
        adj = self.adjacency.expand(b, -1, -1)             # (B, N, N)
        for i in range(2):
            agg = self.gnn_aggr_proj[i](torch.bmm(adj, h))  # (B, N, node_hidden)
            # 门控：融合比例由 [自身, 邻居] 联合决定，自适应保留有用信息
            gate = torch.sigmoid(self.gnn_gate[i](torch.cat([h, agg], dim=-1)))
            h = self.gnn_norm[i](gate * h + (1.0 - gate) * agg)

        # readout: mean-pool || max-pool 双路聚合
        h_mean = h.mean(dim=1)                             # (B, node_hidden)
        h_max  = h.max(dim=1).values                       # (B, node_hidden)
        feat = torch.cat([h_mean, h_max], dim=-1)          # (B, node_hidden*2)
        return self.max_action * torch.tanh(self.readout(feat))


# -----------------------------
# Transformer (node sequence)
# -----------------------------
class TransformerActor(nn.Module):
    """
    Input: (B, num_nodes * node_dim)
    Pipeline:
      state_encoder -> input_proj -> PAPETransformerEncoder (PAPE bias injected into attn logits) -> readout -> tanh
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
        use_kuka_pe=True,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_dim = node_dim

        # 状态编码器
        if use_state_encoder and ENCODER_AVAILABLE:
            if input_dim == num_nodes * node_dim:
                self.state_encoder = nn.Identity()
            elif num_nodes == 7:
                self.state_encoder = KukaStateEncoder(node_dim=node_dim)
                node_dim = self.state_encoder.output_node_dim  # 实际每节点维度 = node_dim+1
                print(f"[Transformer] KukaStateEncoder: {input_dim} -> {num_nodes}x{node_dim}")
            else:
                self.state_encoder = StateEncoderWithPadding(input_dim, num_nodes, node_dim)
        else:
            self.state_encoder = nn.Identity()
        self.node_dim = node_dim  # 更新为实际使用的维度

        self.input_proj = nn.Linear(node_dim, d_model)

        # PAPE Transformer（FFN=4x 标准配置，输出附 LN）
        self._use_pape = use_kuka_pe and (num_nodes == 7)
        if self._use_pape:
            self.transformer = PAPETransformerEncoder(
                d_model=d_model, nhead=nhead, num_layers=num_layers,
                num_joints=num_nodes, dim_feedforward=d_model * 4,  # 标准4x扩展
            )
            print(f"[Transformer] PAPETransformerEncoder (attn-bias, {num_nodes} joints, FFN=4x)")
        else:
            self.pos_embed = nn.Parameter(torch.zeros(num_nodes, d_model))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, batch_first=True,
                dim_feedforward=d_model * 4, dropout=0.0, norm_first=True,  # Pre-LN
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers,
                norm=nn.LayerNorm(d_model),  # 输出 LN
            )

        # readout: mean-pool + max-pool 双路聚合（比 flatten 泛化更好，参数更少）
        self.readout = nn.Sequential(
            nn.Linear(d_model * 2, 256),   # concat(mean, max)
            nn.ELU(),
            nn.Linear(256, action_dim),
        )
        self.max_action = max_action

    def forward(self, x):
        b = x.shape[0]
        x = self.state_encoder(x)
        t = self.input_proj(x.view(b, self.num_nodes, self.node_dim))  # (B, N, d_model)
        if not self._use_pape:
            t = t + self.pos_embed.unsqueeze(0)
        t = self.transformer(t)                                         # (B, N, d_model)
        # mean-pool + max-pool 双路聚合
        feat = torch.cat([t.mean(dim=1), t.max(dim=1).values], dim=-1)  # (B, d_model*2)
        return self.max_action * torch.tanh(self.readout(feat))


# -----------------------------
# GNN + Transformer (GT-TD3)
# -----------------------------
class GNNTransformerActor(nn.Module):
    """
    GT-TD3 核心 Actor 流水线:
      状态编码 → per-node MLP
      → 2层链式GNN(局部耦合,残差+LayerNorm)
      → input_proj
      → PAPETransformerEncoder(全局依赖 + 物理先验注意力偏置)
      → readout → tanh
    """
    def __init__(
        self,
        node_dim,
        num_nodes,
        action_dim,
        adjacency=None,
        node_hidden=64,
        d_model=128,
        nhead=4,
        num_layers=2,
        max_action=1.0,
        use_state_encoder=False,
        input_dim=20,
        use_kuka_pe=True,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_dim = node_dim

        # 状态编码器
        if use_state_encoder and ENCODER_AVAILABLE:
            if input_dim == num_nodes * node_dim:
                self.state_encoder = nn.Identity()
            elif num_nodes == 7:
                self.state_encoder = KukaStateEncoder(node_dim=node_dim)
                node_dim = self.state_encoder.output_node_dim  # 实际每节点维度 = node_dim+1
                print(f"[GNN+Trans] KukaStateEncoder: {input_dim} -> {num_nodes}x{node_dim}")
            else:
                self.state_encoder = StateEncoderWithPadding(input_dim, num_nodes, node_dim)
        else:
            self.state_encoder = nn.Identity()
        self.node_dim = node_dim  # 更新为实际使用的维度

        # 1) per-node encoder(2层,LeakyReLU)
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim, node_hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(node_hidden, node_hidden),
            nn.LeakyReLU(0.1),
        )

        # 2) 链式拓扑邻接矩阵(带自环)
        if adjacency is not None:
            adj = torch.as_tensor(adjacency, dtype=torch.float32)
        else:
            adj = self._chain_adj(num_nodes, self_loops=True)
        self.register_buffer("adjacency", self._row_norm(adj))

        # 2层 门控GNN：GRU 风格自适应更新
        self.gnn_norm = nn.ModuleList([nn.LayerNorm(node_hidden) for _ in range(2)])
        self.gnn_aggr_proj = nn.ModuleList([
            nn.Linear(node_hidden, node_hidden, bias=False) for _ in range(2)
        ])
        self.gnn_gate = nn.ModuleList([
            nn.Linear(node_hidden * 2, node_hidden) for _ in range(2)
        ])

        # 3) GNN→Transformer 投影
        self.input_proj = nn.Linear(node_hidden, d_model)

        # 跨层门控融合：GNN 局部特征 ⊕ Transformer 全局特征，自适应权重
        # 让网络自主决定：是否在某些关节上更信任局部运动学约束 vs 全局注意力
        self.cross_gate = nn.Linear(d_model * 2, d_model)
        self.cross_norm = nn.LayerNorm(d_model)

        # 4) PAPE Transformer（FFN=4x 标准配置）
        self._use_pape = use_kuka_pe and (num_nodes == 7)
        if self._use_pape:
            self.transformer = PAPETransformerEncoder(
                d_model=d_model, nhead=nhead, num_layers=num_layers,
                num_joints=num_nodes, dim_feedforward=d_model * 4,  # 标准4x
            )
            print(f"[GNN+Trans] Gated-GNN(2L) + PAPETransformerEncoder(FFN=4x) + CrossGate")
        else:
            self.pos_embed = nn.Parameter(torch.zeros(num_nodes, d_model))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, batch_first=True,
                dim_feedforward=d_model * 4, dropout=0.0, norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers,
                norm=nn.LayerNorm(d_model),
            )

        # 5) readout: mean-pool + max-pool 双路聚合
        self.readout = nn.Sequential(
            nn.Linear(d_model * 2, 256),
            nn.ELU(),
            nn.Linear(256, action_dim),
        )
        self.max_action = max_action

    @staticmethod
    def _chain_adj(n, self_loops=True):
        A = torch.zeros(n, n)
        for i in range(n - 1):
            A[i, i + 1] = 1.0
            A[i + 1, i] = 1.0
        if self_loops:
            A.fill_diagonal_(1.0)
        return A

    @staticmethod
    def _row_norm(A):
        return A / A.sum(dim=1, keepdim=True).clamp_min(1e-6)

    def forward(self, x):
        b = x.shape[0]

        x = self.state_encoder(x)
        nodes = x.view(b, self.num_nodes, self.node_dim)   # (B, N, node_dim)

        # 1) per-node MLP
        h = self.node_mlp(nodes)                           # (B, N, node_hidden)

        # 2) 2层 门控GNN（链式，GRU风格自适应融合）
        adj = self.adjacency.expand(b, -1, -1)
        for i in range(2):
            agg = self.gnn_aggr_proj[i](torch.bmm(adj, h))  # (B, N, node_hidden)
            gate = torch.sigmoid(self.gnn_gate[i](torch.cat([h, agg], dim=-1)))
            h = self.gnn_norm[i](gate * h + (1.0 - gate) * agg)

        # 3) 投影 + PAPE Transformer
        gnn_feat = self.input_proj(h)                      # (B, N, d_model) 局部特征
        t = gnn_feat
        if not self._use_pape:
            t = t + self.pos_embed.unsqueeze(0)
        t = self.transformer(t)                            # (B, N, d_model) 全局特征

        # 跨层门控融合：每个关节节点自适应平衡局部运动学约束与全局依赖
        fusion_gate = torch.sigmoid(self.cross_gate(torch.cat([gnn_feat, t], dim=-1)))
        t = self.cross_norm(fusion_gate * gnn_feat + (1.0 - fusion_gate) * t)

        # 4) mean-pool + max-pool 双路聚合 → readout
        feat = torch.cat([t.mean(dim=1), t.max(dim=1).values], dim=-1)  # (B, d_model*2)
        return self.max_action * torch.tanh(self.readout(feat))

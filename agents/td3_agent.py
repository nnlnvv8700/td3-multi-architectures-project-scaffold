import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional

# 相对导入同目录下的网络实现
from .networks import (
    MLPActor, MLPCritic,
    SimpleGNNActor, TransformerActor, GNNTransformerActor
)


def build_actor(state_dim: int, action_dim: int, cfg, max_action: float = 1.0) -> nn.Module:
    """
    根据配置构建 Actor 网络
    
    Args:
        state_dim: 状态维度
        action_dim: 动作维度
        cfg: 配置对象，包含 actor_arch, node_dim, num_nodes, use_state_encoder 等
        max_action: 动作空间最大值（从 env.action_space.high 获取）
        
    Returns:
        构建好的 Actor 网络模块
    """
    arch = getattr(cfg, "actor_arch", "mlp")
    use_encoder = getattr(cfg, "use_state_encoder", False)
    
    if arch == "mlp":
        return MLPActor(state_dim, action_dim, max_action=max_action)
    
    elif arch == "gnn":
        node_dim = int(getattr(cfg, "node_dim", 5))
        num_nodes = int(getattr(cfg, "num_nodes", 4))
        return SimpleGNNActor(
            node_dim=node_dim, 
            num_nodes=num_nodes, 
            action_dim=action_dim,
            use_state_encoder=use_encoder,
            input_dim=state_dim,
            max_action=max_action
        )
    
    elif arch == "transformer":
        node_dim = int(getattr(cfg, "node_dim", 5))
        num_nodes = int(getattr(cfg, "num_nodes", 4))
        use_kuka_pe = getattr(cfg, "use_kuka_pe", True)
        return TransformerActor(
            node_dim=node_dim, 
            num_nodes=num_nodes, 
            action_dim=action_dim,
            use_state_encoder=use_encoder,
            input_dim=state_dim,
            use_kuka_pe=use_kuka_pe,
            max_action=max_action
        )
    
    elif arch == "gnn_transformer":
        node_dim = int(getattr(cfg, "node_dim", 5))
        num_nodes = int(getattr(cfg, "num_nodes", 4))
        use_kuka_pe = getattr(cfg, "use_kuka_pe", True)
        return GNNTransformerActor(
            node_dim=node_dim, 
            num_nodes=num_nodes, 
            action_dim=action_dim,
            use_state_encoder=use_encoder,
            input_dim=state_dim,
            use_kuka_pe=use_kuka_pe,
            max_action=max_action
        )
    
    else:
        raise ValueError(f"Unknown actor_arch: {arch}")


class TD3:
    """
    与训练脚本对齐的 TD3 实现：
      构造: TD3(state_dim, action_dim, max_action, cfg)
      关键属性: self.gamma, self.tau, self.policy_noise, self.noise_clip, self.policy_delay, self.device
      方法: select_action(...), train(...), save(base_path)
    """

    def __init__(self, state_dim: int, action_dim: int, max_action: float, cfg):
        # 设备
        device_str = getattr(cfg, "device", None)
        if device_str is not None:
            self.device = torch.device(device_str)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 架构与网络
        self.actor = build_actor(state_dim, action_dim, cfg, max_action=max_action).to(self.device)
        self.actor_target = build_actor(state_dim, action_dim, cfg, max_action=max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic1 = MLPCritic(state_dim, action_dim).to(self.device)
        self.critic2 = MLPCritic(state_dim, action_dim).to(self.device)
        self.critic1_target = MLPCritic(state_dim, action_dim).to(self.device)
        self.critic2_target = MLPCritic(state_dim, action_dim).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # 优化器
        actor_lr = float(getattr(cfg, "actor_lr", 1e-3))
        critic_lr = float(getattr(cfg, "critic_lr", 1e-3))
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        # 两个 Critic 独立优化器：梯度分离，避免交叉干扰
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=critic_lr)

        # 超参数（确保都作为成员存在）
        self.gamma = float(getattr(cfg, "gamma", 0.99))
        self.tau = float(getattr(cfg, "tau", 0.003))  # 🎯 中间值：0.001→0.003，平衡稳定性和收敛速度
        self.policy_noise = float(getattr(cfg, "policy_noise", 0.2))  # 标准 TD3 推荐值
        self.noise_clip = float(getattr(cfg, "noise_clip", 0.5))  # 标准 TD3 推荐值
        self.policy_delay = int(getattr(cfg, "policy_delay", 2))
        # 动作幅度正则：防止 actor 输出饱和（tanh接近±1时梯度消失）
        self.action_reg_coef = float(getattr(cfg, "action_reg_coef", 1e-3))

        self.total_it = 0
        self.max_action = float(max_action)

    @torch.no_grad()
    def select_action(self, state, deterministic: bool = True):
        """
        state: np.ndarray or list, shape (state_dim,)
        返回动作 numpy 数组，范围由 actor 的 tanh 决定（通常是 [-1,1]）
        """
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        action = self.actor(state)
        return action.squeeze(0).cpu().numpy()

    def train(self, replay_buffer, batch_size: int = 256):
        import numpy as np  # 仅用于类型检查
        self.total_it += 1

        # 1) 取 batch（兼容 .sample / .sample_batch）
        try:
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        except AttributeError:
            b = replay_buffer.sample_batch(batch_size)
            state, action, next_state = b["obs"], b["acts"], b["obs2"]
            reward = b["rews"].reshape(-1, 1)
            not_done = 1.0 - b["done"].reshape(-1, 1)

        # 2) 统一转 torch 并放 device
        state      = torch.as_tensor(state,      dtype=torch.float32, device=self.device)
        action     = torch.as_tensor(action,     dtype=torch.float32, device=self.device)
        next_state = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
        reward     = torch.as_tensor(reward,     dtype=torch.float32, device=self.device)
        not_done   = torch.as_tensor(not_done,   dtype=torch.float32, device=self.device)

        # 3) 目标 Q
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            target_q1 = self.critic1_target(next_state, next_action)
            target_q2 = self.critic2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + not_done * self.gamma * target_q
            # 注意：不再硬编码 clamp Q 值范围。数值稳定性由 SmoothL1Loss + grad_clip 保证。
            # 如需额外安全保障，可启用 reward normalization（推荐）。

        # 4) 更新 Critic（两个独立优化器，梯度不交叉）
        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)
        # Huber Loss 对离群 TD-error 更鲁棒
        critic1_loss = nn.SmoothL1Loss()(current_q1, target_q)
        critic2_loss = nn.SmoothL1Loss()(current_q2, target_q)

        self.critic1_opt.zero_grad(set_to_none=True)
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        self.critic1_opt.step()

        self.critic2_opt.zero_grad(set_to_none=True)
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic2_opt.step()

        critic_loss = (critic1_loss + critic2_loss) / 2.0

        # Critic target 每步软更新（标准TD3：Critic target不与policy_delay绑定）
        with torch.no_grad():
            for p, tp in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
            for p, tp in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

        actor_loss: Optional[torch.Tensor] = None

        # 5) 延迟更新 Actor + Actor target 软更新
        if self.total_it % self.policy_delay == 0:
            pi = self.actor(state)
            # 动作幅度正则：惩罚 tanh 饱和（|a|→1 时梯度消失），鼓励动作保持在线性区
            action_reg = (pi ** 2).mean()
            actor_loss = -self.critic1(state, pi).mean() + self.action_reg_coef * action_reg
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_opt.step()

            with torch.no_grad():
                for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                    tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

        # 6) 返回 loss（便于记录）
        if actor_loss is None:
            return float(critic_loss.item())
        else:
            return {"actor_loss": float(actor_loss.item()), "critic_loss": float(critic_loss.item())}

    def save(self, base_path: str):
        """保存 actor 权重（与 test/evaluate 脚本对齐）"""
        torch.save(self.actor.state_dict(), base_path + ".pt")

    def save_checkpoint(self, path: str):
        """
        保存完整训练检查点（可断点续训）。
        保存内容：所有网络权重 + 三个优化器状态 + 训练步数。
        用法：agent.save_checkpoint("results/run1/ckpt_step500000")
        """
        torch.save({
            "total_it": self.total_it,
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "critic1_target": self.critic1_target.state_dict(),
            "critic2_target": self.critic2_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic1_opt": self.critic1_opt.state_dict(),
            "critic2_opt": self.critic2_opt.state_dict(),
        }, path + ".ckpt")

    def load_checkpoint(self, path: str):
        """
        加载完整训练检查点，恢复所有网络和优化器状态。
        用法：agent.load_checkpoint("results/run1/ckpt_step500000")
        """
        ckpt = torch.load(path + ".ckpt", map_location=self.device, weights_only=True)
        self.total_it = int(ckpt["total_it"])
        self.actor.load_state_dict(ckpt["actor"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic1.load_state_dict(ckpt["critic1"])
        self.critic2.load_state_dict(ckpt["critic2"])
        self.critic1_target.load_state_dict(ckpt["critic1_target"])
        self.critic2_target.load_state_dict(ckpt["critic2_target"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic1_opt.load_state_dict(ckpt["critic1_opt"])
        self.critic2_opt.load_state_dict(ckpt["critic2_opt"])

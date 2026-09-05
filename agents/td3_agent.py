import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional
from pathlib import Path
import json

from utils.persistence import atomic_output

# 相对导入同目录下的网络实现
from .networks import (
    MLPActor, MLPCritic,
    SimpleGNNActor, TransformerActor, GNNTransformerActor
)


def resolve_device(device=None) -> torch.device:
    """Resolve a requested CPU/CUDA device before allocating models or simulators."""
    resolved = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if resolved.type not in ("cpu", "cuda"):
        raise ValueError(f"Unsupported device: {resolved}; use cpu or cuda[:index]")
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(f"CUDA requested ({resolved}), but CUDA is unavailable")
        if resolved.index is not None and resolved.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index out of range: {resolved.index}")
    return resolved


def build_actor(state_dim: int, action_dim: int, cfg, max_action: float = 1.0) -> nn.Module:
    """
    根据配置构建 Actor 网络

    Args:
        state_dim: 状态维度
        action_dim: 动作维度
        cfg: 配置对象，包含 actor_arch, node_dim, num_nodes, use_state_encoder 等

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
            max_action=max_action,
        )

    elif arch == "transformer":
        node_dim = int(getattr(cfg, "node_dim", 5))
        num_nodes = int(getattr(cfg, "num_nodes", 4))
        use_kuka_pe = getattr(cfg, "use_kuka_pe", True)  # 🎯 默认启用KUKA位置编码
        return TransformerActor(
            node_dim=node_dim,
            num_nodes=num_nodes,
            action_dim=action_dim,
            use_state_encoder=use_encoder,
            input_dim=state_dim,
            use_kuka_pe=use_kuka_pe,
            max_action=max_action,
        )

    elif arch == "gnn_transformer":
        # ✅ 修复：使用真正的 GNN+Transformer 融合架构
        node_dim = int(getattr(cfg, "node_dim", 5))
        num_nodes = int(getattr(cfg, "num_nodes", 4))
        use_kuka_pe = getattr(cfg, "use_kuka_pe", True)  # 🎯 默认启用KUKA位置编码
        return GNNTransformerActor(
            node_dim=node_dim,
            num_nodes=num_nodes,
            action_dim=action_dim,
            use_state_encoder=use_encoder,
            input_dim=state_dim,
            use_kuka_pe=use_kuka_pe,
            max_action=max_action,
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
        self.device = resolve_device(device_str)
        self.state_dim = state_dim
        self.corrected = getattr(cfg, "algorithm_version", "legacy") == "corrected"
        self.control_mode = getattr(cfg, "control_mode", "direct")
        self.critic_loss_kind = getattr(cfg, "critic_loss", "huber")
        if int(getattr(cfg, "policy_delay", 2)) <= 0:
            raise ValueError("policy_delay must be positive")

        # 架构与网络
        self.actor = build_actor(state_dim, action_dim, cfg, max_action).to(self.device)
        self.actor_target = build_actor(state_dim, action_dim, cfg, max_action).to(self.device)
        if getattr(cfg, "zero_init_actor", False):
            output = [module for module in self.actor.modules() if isinstance(module, nn.Linear)][-1]
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.eval()
        if self.corrected:
            # TD3 is a deterministic actor method; exploration is explicit action noise.
            for network in (self.actor, self.actor_target):
                for module in network.modules():
                    if isinstance(module, nn.Dropout):
                        module.p = 0.0
                    elif isinstance(module, nn.MultiheadAttention):
                        module.dropout = 0.0

        self.critic1 = MLPCritic(state_dim, action_dim).to(self.device)
        self.critic2 = MLPCritic(state_dim, action_dim).to(self.device)
        self.critic1_target = MLPCritic(state_dim, action_dim).to(self.device)
        self.critic2_target = MLPCritic(state_dim, action_dim).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic1_target.eval()
        self.critic2_target.eval()

        # 优化器
        actor_lr = float(getattr(cfg, "actor_lr", 1e-3))
        critic_lr = float(getattr(cfg, "critic_lr", 1e-3))
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=critic_lr)

        # 🎯 添加学习率调度器（余弦退火）
        max_steps = int(getattr(cfg, "max_timesteps", 500000))
        start_steps = int(getattr(cfg, "start_timesteps", 0))
        policy_delay = int(getattr(cfg, "policy_delay", 2))
        first_update = max(start_steps, int(getattr(cfg, "batch_size", 1)) - 1) if self.corrected else start_steps
        critic_updates = max(1, max_steps - first_update)
        actor_updates = max(1, critic_updates // policy_delay)
        self.actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.actor_opt,
            T_max=actor_updates,
            eta_min=actor_lr * 0.1  # 最低为初始学习率的10%
        )
        self.critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.critic_opt,
            T_max=critic_updates,
            eta_min=critic_lr * 0.1
        )

        # 超参数（确保都作为成员存在）
        self.gamma = float(getattr(cfg, "gamma", 0.99))
        self.tau = float(getattr(cfg, "tau", 0.003))  # 🎯 中间值：0.001→0.003，平衡稳定性和收敛速度
        self.policy_noise = float(getattr(cfg, "policy_noise", 0.05))  # 🎯 低noise
        self.noise_clip = float(getattr(cfg, "noise_clip", 0.2))  # 🎯 低clip
        self.policy_delay = policy_delay

        self.total_it = 0
        self.max_action = float(max_action)
        self.last_diagnostics = {}

    def _state_tensor(self, state):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if self.corrected:
            if state.shape[-1] not in (20, 31):
                raise ValueError("Corrected KUKA TD3 expects 20-D or 31-D observations")
            state = state.clone()
            state[..., :7] /= torch.pi
            state[..., 7:14] /= self.max_action
            if state.shape[-1] == 31:
                state[..., 18:25] /= self.max_action
        return state

    def _critic_action(self, action):
        return action / self.max_action if self.corrected else action

    @torch.no_grad()
    def select_action(self, state, deterministic: bool = True):
        """
        state: np.ndarray or list, shape (state_dim,)
        Return actions in [-max_action, max_action], preserving the actor's mode.
        """
        state = self._state_tensor(state)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if state.ndim != 2 or state.shape[1] != self.state_dim or state.shape[0] == 0:
            raise ValueError(f"Expected state shape ({self.state_dim},) or (B, {self.state_dim}); got {tuple(state.shape)}")
        if not torch.isfinite(state).all():
            raise ValueError("State must contain only finite values")
        restore_training = self.actor.training
        if deterministic:
            self.actor.eval()
        try:
            action = self.actor(state)
        finally:
            if deterministic:
                self.actor.train(restore_training)
        return action.squeeze(0).cpu().numpy()

    def train(self, replay_buffer, batch_size: int = 256):
        self.total_it += 1

        # 1) 取 batch（支持优先级回放和普通回放）
        use_per = bool(getattr(replay_buffer, "use_per", False))

        if use_per:
            # 优先级经验回放
            batch, indices, weights = replay_buffer.sample(batch_size)
            state, action, next_state, reward, not_done = batch
            weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
        else:
            # 普通回放
            try:
                state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
            except AttributeError:
                b = replay_buffer.sample_batch(batch_size)
                state, action, next_state = b["obs"], b["acts"], b["obs2"]
                reward = b["rews"].reshape(-1, 1)
                not_done = 1.0 - b["done"].reshape(-1, 1)
            weights = None

        # 2) 统一转 torch 并放 device
        state      = self._state_tensor(state)
        action     = torch.as_tensor(action,     dtype=torch.float32, device=self.device)
        next_state = self._state_tensor(next_state)
        reward     = torch.as_tensor(reward,     dtype=torch.float32, device=self.device)
        not_done   = torch.as_tensor(not_done,   dtype=torch.float32, device=self.device)

        # 3) 目标 Q
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )

            target_q1 = self.critic1_target(next_state, self._critic_action(next_action))
            target_q2 = self.critic2_target(next_state, self._critic_action(next_action))
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + not_done * self.gamma * target_q

            # 🔧 裁剪目标 Q 值，防止数值爆炸（基于奖励范围动态调整）
            if not self.corrected:
                target_q = target_q.clamp(-200, 200)

        # 4) 更新 Critic（使用Huber Loss提高稳定性，支持重要性采样权重）
        current_q1 = self.critic1(state, self._critic_action(action))
        current_q2 = self.critic2(state, self._critic_action(action))

        if self.corrected:
            loss_function = nn.SmoothL1Loss(reduction="none") if self.critic_loss_kind == "huber" else nn.MSELoss(reduction="none")
            point_loss = loss_function(current_q1, target_q) + loss_function(current_q2, target_q)
            critic_loss = (point_loss * weights.reshape(-1, 1)).mean() if weights is not None else point_loss.mean()
            if use_per:
                errors = torch.maximum((target_q - current_q1).abs(), (target_q - current_q2).abs())
                replay_buffer.update_priorities(indices, errors.detach().cpu().numpy().ravel())
        elif weights is not None:
            # 🎯 使用重要性采样权重
            td_error1 = target_q - current_q1
            td_error2 = target_q - current_q2
            critic_loss = (weights.unsqueeze(1) * (td_error1 ** 2)).mean() + \
                         (weights.unsqueeze(1) * (td_error2 ** 2)).mean()

            # 更新优先级
            if use_per:
                td_errors = td_error1.detach().cpu().numpy().flatten()
                replay_buffer.update_priorities(indices, td_errors)
        else:
            # 普通Loss
            critic_loss = nn.SmoothL1Loss()(current_q1, target_q) + nn.SmoothL1Loss()(current_q2, target_q)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        # 🔧 梯度裁剪，防止梯度爆炸（更深网络需要更严格的裁剪）
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 0.5)
        self.critic_opt.step()
        # 🎯 更新Critic学习率
        self.critic_scheduler.step()

        actor_loss: Optional[torch.Tensor] = None

        # 5) 延迟更新 Actor + 软更新 target
        if self.total_it % self.policy_delay == 0:
            actor_state = state
            if self.corrected and hasattr(replay_buffer, "sample_uniform_states"):
                actor_state = self._state_tensor(replay_buffer.sample_uniform_states(batch_size))
            critic_parameters = list(self.critic1.parameters())
            previous_flags = [parameter.requires_grad for parameter in critic_parameters]
            if self.corrected:
                for parameter in critic_parameters:
                    parameter.requires_grad_(False)
            try:
                actor_loss = -self.critic1(actor_state, self._critic_action(self.actor(actor_state))).mean()
                self.actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
            finally:
                for parameter, flag in zip(critic_parameters, previous_flags):
                    parameter.requires_grad_(flag)
            # 🔧 Actor 梯度裁剪（更严格）
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_opt.step()
            # 🎯 更新Actor学习率
            self.actor_scheduler.step()

            with torch.no_grad():
                for p, tp in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                    tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
                for p, tp in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                    tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
                for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                    tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

        self.last_diagnostics = {
            "q_mean": float(current_q1.detach().mean()),
            "target_q_mean": float(target_q.mean()),
            "td_abs_mean": float((target_q - current_q1).detach().abs().mean()),
            "actor_lr": self.actor_opt.param_groups[0]["lr"],
            "critic_lr": self.critic_opt.param_groups[0]["lr"],
        }
        # 6) 返回 loss（便于记录）
        if actor_loss is None:
            return float(critic_loss.item())
        else:
            return {"actor_loss": float(actor_loss.item()), "critic_loss": float(critic_loss.item())}

    def save(self, base_path: str):
        """仅保存 actor 权重（与 test 脚本对齐）"""
        with atomic_output(str(base_path) + ".pt") as stream:
            torch.save(self.actor.state_dict(), stream)
        metadata = {"algorithm_version": "corrected" if self.corrected else "legacy",
                    "state_dim": self.state_dim, "max_action": self.max_action,
                    "control_mode": self.control_mode}
        with atomic_output(str(base_path) + ".meta.json") as stream:
            stream.write(json.dumps(metadata).encode("utf-8"))

    def load_actor(self, path: str) -> None:
        """Load an actor-only checkpoint for inference; this does not resume training."""
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"Actor checkpoint does not exist: {source}")
        metadata_path = source.with_suffix(".meta.json")
        if self.corrected and not metadata_path.is_file():
            raise ValueError("Corrected policy requires its .meta.json sidecar to verify input/action semantics")
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            expected_version = "corrected" if self.corrected else "legacy"
            if (metadata["algorithm_version"] != expected_version or
                    metadata["control_mode"] != self.control_mode or metadata["max_action"] != self.max_action):
                raise ValueError("Checkpoint policy semantics differ; load its saved run configuration")
        state_dict = torch.load(source, map_location=self.device, weights_only=True)
        if not isinstance(state_dict, dict):
            raise ValueError(f"Expected an actor state_dict in {source}")
        expected = self.actor.state_dict()
        if set(expected) != set(state_dict) or any(
            not isinstance(state_dict[key], torch.Tensor)
            or state_dict[key].shape != value.shape for key, value in expected.items()
        ):
            raise ValueError(
                "Checkpoint does not match the actor architecture/observation version. "
                "Use its saved config and the matching historical network implementation."
            )
        self.actor.load_state_dict(state_dict)

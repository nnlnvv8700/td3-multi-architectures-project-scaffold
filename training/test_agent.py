# ---- 让本地项目优先于 site-packages，避免与第三方 'agents' 包冲突 ----
import os, sys, json, argparse, csv
import numpy as np
import torch
import torch.nn as nn
try:
    import gymnasium as gym
except ImportError:
    import gym

# Matplotlib 出图（新罗马 + 3D 轨迹）
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  激活 3D
from datetime import datetime

# 全局字体 Times New Roman
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["mathtext.fontset"] = "dejavuserif"
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["figure.dpi"] = 140

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 确保环境注册（到点+轨迹两种）
import envs.kuka_iiwa_env  # noqa: F401
try:
    import envs  # noqa: F401
except Exception:
    pass

from agents.td3_agent import TD3
from utils.gym_compat import reset_env, step_env


def flatten_obs(obs):
    if isinstance(obs, dict):
        return np.concatenate(
            [obs["observation"], obs["achieved_goal"], obs["desired_goal"]],
            axis=0,
        ).astype(np.float32)
    else:
        return np.array(obs, dtype=np.float32)


class LegacyKukaStateEncoder(nn.Module):
    """Compatibility encoder for old checkpoints that expected 7xnode_dim without joint-id embedding."""

    def __init__(self, node_dim=6):
        super().__init__()
        self.node_dim = int(node_dim)

    def forward(self, state):
        batch_size = state.shape[0]
        joint_pos = state[:, 0:7]
        joint_vel = state[:, 7:14]
        achieved = state[:, 14:17]
        desired = state[:, 17:20]

        goal_vector = desired - achieved
        goal_dist = torch.norm(goal_vector, dim=1, keepdim=True).clamp_min(1e-6)
        goal_dir = goal_vector / goal_dist
        goal_dist_norm = goal_dist / 1.2
        joint_vel_norm = joint_vel / 1.5
        joint_pos_cum = torch.cumsum(joint_pos, dim=1) / (7 * 3.0)

        if self.node_dim == 6:
            goal_dist_feat = goal_dist_norm.expand(-1, 7)
            goal_dir_x = goal_dir[:, 0:1].expand(-1, 7)
            goal_dir_y = goal_dir[:, 1:2].expand(-1, 7)
            nodes = torch.stack([
                joint_pos, joint_vel_norm, joint_pos_cum,
                goal_dist_feat, goal_dir_x, goal_dir_y
            ], dim=2)
        elif self.node_dim == 5:
            goal_dist_feat = goal_dist_norm.expand(-1, 7)
            goal_dir_x = goal_dir[:, 0:1].expand(-1, 7)
            nodes = torch.stack([
                joint_pos, joint_vel_norm, joint_pos_cum,
                goal_dist_feat, goal_dir_x
            ], dim=2)
        elif self.node_dim == 4:
            goal_dist_feat = goal_dist_norm.expand(-1, 7)
            nodes = torch.stack([joint_pos, joint_vel_norm, goal_dist_feat, joint_pos_cum], dim=2)
        else:
            goal_dist_feat = goal_dist_norm.expand(-1, 7)
            nodes = torch.stack([joint_pos, joint_vel_norm, goal_dist_feat], dim=2)
        return nodes.reshape(batch_size, -1)


class LegacyKukaPositionalEncoding(nn.Module):
    """Compatibility module that matches old checkpoint keys."""

    def __init__(self, num_nodes=7, d_model=128):
        super().__init__()
        pe = torch.zeros(num_nodes, d_model)
        dist = torch.zeros(num_nodes, num_nodes, dtype=torch.long)
        for i in range(num_nodes):
            for j in range(num_nodes):
                dist[i, j] = abs(i - j)
        hw = torch.linspace(0.8, 1.2, num_nodes)
        self.register_buffer("pe", pe)
        self.register_buffer("dist_matrix", dist)
        self.register_buffer("hierarchy_weights", hw)
        self.distance_embed = nn.Embedding(num_nodes, d_model)

    def forward(self, x):
        dist_emb = self.distance_embed(self.dist_matrix).mean(dim=1)
        return x + self.pe.unsqueeze(0) + 0.10 * dist_emb.unsqueeze(0)


class LegacyGNNTransformerActor(nn.Module):
    """Compatibility actor for older gnn_transformer checkpoints."""

    def __init__(self, node_dim, num_nodes, action_dim, max_action=1.0):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.node_dim = int(node_dim)
        self.max_action = float(max_action)
        self.state_encoder = LegacyKukaStateEncoder(node_dim=self.node_dim)
        self.node_mlp = nn.Sequential(
            nn.Linear(self.node_dim, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.1),
        )
        adj = torch.zeros(self.num_nodes, self.num_nodes, dtype=torch.float32)
        for i in range(self.num_nodes - 1):
            adj[i, i + 1] = 1.0
            adj[i + 1, i] = 1.0
        self.register_buffer("adjacency", adj)
        self.input_proj = nn.Linear(64, 128)
        self.pos_encoding = LegacyKukaPositionalEncoding(num_nodes=self.num_nodes, d_model=128)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=2048,
            dropout=0.0,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.readout = nn.Sequential(
            nn.Linear(self.num_nodes * 128, 256),
            nn.ELU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x):
        b = x.shape[0]
        nodes = self.state_encoder(x).view(b, self.num_nodes, self.node_dim)
        h = self.node_mlp(nodes)
        adj = self.adjacency.unsqueeze(0).expand(b, -1, -1)
        deg = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
        h = 0.5 * h + 0.5 * torch.bmm(adj / deg, h)
        t = self.input_proj(h)
        t = self.pos_encoding(t)
        t = self.transformer(t)
        feat = t.reshape(b, -1)
        return self.max_action * torch.tanh(self.readout(feat))


def read_latest_run(results_root="./results"):
    hint = os.path.join(results_root, "LATEST_RUN.txt")
    if os.path.exists(hint):
        try:
            with open(hint, "r", encoding="utf-8") as f:
                p = f.read().strip()
            if p and os.path.isdir(p):
                return p
        except Exception:
            pass
    return None


def find_weight_file(run_dir):
    if not os.path.isdir(run_dir):
        return None
    preferred = []
    # 优先 final_model*.pt/.pth
    for name in ("final_model.pt", "final_model.pth"):
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            preferred.append(p)
    for fn in os.listdir(run_dir):
        if fn.startswith("final_model_") and (fn.endswith(".pt") or fn.endswith(".pth")):
            preferred.append(os.path.join(run_dir, fn))
    if preferred:
        preferred.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return preferred[0]
    # 任意 .pt/.pth
    any_ckpt = [os.path.join(run_dir, fn) for fn in os.listdir(run_dir)
                if fn.endswith(".pt") or fn.endswith(".pth")]
    if any_ckpt:
        any_ckpt.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return any_ckpt[0]
    return None


def read_run_config(run_dir):
    cfg_path = os.path.join(run_dir, "config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[警告] 读取 {cfg_path} 失败：{e}")
    return {}


def resolve_arch_defaults(actor_arch):
    """与训练时约定一致：mlp -> None/None；其余 -> 5/4"""
    if actor_arch == "mlp":
        return None, None
    return 5, 4


def resolve_inputs(args):
    """
    解析 run_dir / model_path / run_cfg：
    - 优先使用 --run_dir
    - 否则使用 --model（文件或目录）
    - 都没给就从 results/LATEST_RUN.txt 读取最近一次
    """
    run_dir, model_path = None, None
    run_cfg = {}

    if args.run_dir:
        run_dir = args.run_dir
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f"--run_dir 不存在或不是目录：{run_dir}")
        model_path = find_weight_file(run_dir)
        run_cfg = read_run_config(run_dir)
    elif args.model:
        if os.path.isdir(args.model):
            run_dir = args.model
            model_path = find_weight_file(run_dir)
            run_cfg = read_run_config(run_dir)
        elif os.path.isfile(args.model):
            model_path = args.model
            run_dir = os.path.dirname(args.model)
            run_cfg = read_run_config(run_dir)
        else:
            raise FileNotFoundError(f"--model 路径不存在：{args.model}")
    else:
        run_dir = read_latest_run(args.results_dir)
        if not run_dir:
            raise FileNotFoundError("未提供 --run_dir / --model，且 results/LATEST_RUN.txt 不存在或无效。")
        model_path = find_weight_file(run_dir)
        run_cfg = read_run_config(run_dir)

    if model_path is None:
        raise FileNotFoundError(
            "未找到模型权重：请用 --model 指定 .pt/.pth 或包含权重的目录，"
            "或确保 results/LATEST_RUN.txt 指向的训练目录下存在权重文件。"
        )

    return run_dir, model_path, run_cfg


def build_env(env_id, dense_reward_flag=True, render_mode="human"):
    try:
        env = gym.make(env_id, render_mode=render_mode, dense_reward=bool(dense_reward_flag))
    except TypeError:
        env = gym.make(env_id, render_mode=render_mode)
    return env


def _select_action_eval(agent, state_vec):
    """兼容 select_action 的不同签名"""
    try:
        return agent.select_action(state_vec, deterministic=True)
    except TypeError:
        return agent.select_action(state_vec)


def _select_action_eval_actor(actor, state_vec):
    state_t = torch.as_tensor(state_vec, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        action = actor(state_t).cpu().numpy()[0]
    return action


def _extract_positions(obs, info, env):
    """
    返回 (ee_pos, goal_pos)，均为 (3,) ndarray。
    优先从 info 取；否则从 obs['achieved_goal']/['desired_goal'] 推断。
    若都不可得，返回 (None, None)。
    """
    ee, gg = None, None
    try:
        if isinstance(info, dict):
            if "ee_pos" in info:
                ee = np.array(info["ee_pos"], dtype=np.float32)
            if "goal_pos" in info:
                gg = np.array(info["goal_pos"], dtype=np.float32)
    except Exception:
        pass

    if isinstance(obs, dict):
        if ee is None and "achieved_goal" in obs:
            ee = np.array(obs["achieved_goal"], dtype=np.float32)
        if gg is None and "desired_goal" in obs:
            gg = np.array(obs["desired_goal"], dtype=np.float32)

    return ee, gg


def _distance(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b))) if (a is not None and b is not None) else np.nan


def _rmse_path(exec_path, ref_path):
    """对齐长度后计算 RMSE（若 ref 缺失返回 np.nan）"""
    if exec_path is None or ref_path is None:
        return np.nan
    L = min(len(exec_path), len(ref_path))
    if L <= 1:
        return np.nan
    d = exec_path[:L] - ref_path[:L]
    return float(np.sqrt(np.mean(np.sum(d*d, axis=1))))


def _path_length(points):
    if points is None or len(points) < 2:
        return np.nan
    diffs = np.diff(points, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)


def safe_savefig(fig, out_dir, filename_base):
    """
    防覆盖保存：若 out_dir/filename_base.png 已存在，则自动追加 _1/_2 …
    返回最终路径。
    """
    safe_mkdir(out_dir)
    base = os.path.join(out_dir, filename_base)
    path = base + ".png"
    if not os.path.exists(path):
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig); return path
    i = 1
    while True:
        cand = f"{base}_{i}.png"
        if not os.path.exists(cand):
            fig.savefig(cand, bbox_inches="tight")
            plt.close(fig); return cand
        i += 1


def safe_save_image(image, out_dir, filename_base):
    safe_mkdir(out_dir)
    base = os.path.join(out_dir, filename_base)
    path = base + ".png"
    image_uint8 = np.asarray(image, dtype=np.uint8)
    if not os.path.exists(path):
        plt.imsave(path, image_uint8)
        return path
    i = 1
    while True:
        cand = f"{base}_{i}.png"
        if not os.path.exists(cand):
            plt.imsave(cand, image_uint8)
            return cand
        i += 1


def stylize_pybullet_frame(image):
    img = np.asarray(image, dtype=np.float32).copy()
    h, w, _ = img.shape

    y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    sky_top = np.array([171.0, 204.0, 236.0], dtype=np.float32).reshape(1, 1, 3)
    sky_bottom = np.array([225.0, 236.0, 247.0], dtype=np.float32).reshape(1, 1, 3)
    sky = sky_top * (1.0 - y) + sky_bottom * y

    white_mask = (
        (img.mean(axis=2, keepdims=True) > 205.0) &
        ((img.max(axis=2, keepdims=True) - img.min(axis=2, keepdims=True)) < 45.0)
    ).astype(np.float32)

    upper_weight = np.clip(1.0 - (np.arange(h, dtype=np.float32)[:, None, None] / (0.78 * h)), 0.0, 1.0)
    mask = white_mask * upper_weight
    img = img * (1.0 - mask) + sky * mask
    return np.clip(img, 0.0, 255.0).astype(np.uint8)


def plot_metrics_basic(out_dir, rewards, success, tts, min_d, run_basename):
    ep = np.arange(1, len(rewards) + 1)

    # 1) 每回合曲线（4 合 1）
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(ep, rewards, linewidth=1.2)
    ax1.set_title("Episode Reward"); ax1.set_xlabel("Episode"); ax1.set_ylabel("Reward"); ax1.grid(alpha=0.3, ls="--")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(ep, success, linewidth=1.2, marker='o', ms=2)
    ax2.set_title("Success (0/1)"); ax2.set_xlabel("Episode"); ax2.set_ylabel("Success"); ax2.grid(alpha=0.3, ls="--")

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(ep, tts, linewidth=1.2)
    ax3.set_title("Time To Success (steps)"); ax3.set_xlabel("Episode"); ax3.set_ylabel("Steps"); ax3.grid(alpha=0.3, ls="--")

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(ep, min_d, linewidth=1.2)
    ax4.set_title("Min Distance (m)"); ax4.set_xlabel("Episode"); ax4.set_ylabel("Meters"); ax4.grid(alpha=0.3, ls="--")

    fig.suptitle(f"Evaluation (Random {len(rewards)} Episodes) - {run_basename}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p1 = safe_savefig(fig, out_dir, f"{run_basename}_eval_curves")

    # 2) 直方图（TTS/MinD）
    fig2 = plt.figure(figsize=(10, 4.2))
    ax21 = fig2.add_subplot(1, 2, 1)
    ax22 = fig2.add_subplot(1, 2, 2)

    ax21.hist(tts, bins=10, edgecolor='black'); ax21.set_title("TTS Histogram"); ax21.set_xlabel("Steps"); ax21.set_ylabel("Count"); ax21.grid(alpha=0.3, ls="--")
    valid_min_d = [d for d in min_d if np.isfinite(d)]
    if len(valid_min_d) == 0: valid_min_d = [np.nan]
    ax22.hist(valid_min_d, bins=10, edgecolor='black'); ax22.set_title("Min Distance Histogram"); ax22.set_xlabel("Meters"); ax22.set_ylabel("Count"); ax22.grid(alpha=0.3, ls="--")

    fig2.tight_layout()
    p2 = safe_savefig(fig2, out_dir, f"{run_basename}_hist")

    return [p1, p2]


def plot_metrics_trajectory(out_dir, rmses, max_devs, end_errs, path_len_exec, path_len_ref, run_basename):
    """仅在有参考轨迹时调用"""
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(np.arange(len(rmses)), rmses); ax1.set_title("RMSE (m)"); ax1.set_xlabel("Episode"); ax1.set_ylabel("Meters"); ax1.grid(alpha=0.3, ls="--")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(np.arange(len(max_devs)), max_devs); ax2.set_title("Max Deviation (m)"); ax2.set_xlabel("Episode"); ax2.set_ylabel("Meters"); ax2.grid(alpha=0.3, ls="--")

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.bar(np.arange(len(end_errs)), end_errs); ax3.set_title("End-Point Error (m)"); ax3.set_xlabel("Episode"); ax3.set_ylabel("Meters"); ax3.grid(alpha=0.3, ls="--")

    ax4 = fig.add_subplot(gs[1, 1])
    x = np.arange(len(path_len_exec))
    ax4.plot(x, path_len_exec, label="Exec Path Len")
    if any(np.isfinite(path_len_ref)):
        ax4.plot(x, path_len_ref, label="Ref Path Len")
    ax4.set_title("Path Length"); ax4.set_xlabel("Episode"); ax4.set_ylabel("Meters"); ax4.legend(frameon=False); ax4.grid(alpha=0.3, ls="--")

    fig.suptitle(f"Trajectory Metrics - {run_basename}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = safe_savefig(fig, out_dir, f"{run_basename}_trajectory_metrics")
    return [p]


def plot_best_trajectory(out_dir, best_path, best_goal, run_basename, best_idx, best_label="min_distance", ref_traj=None):
    """
    画 3D 轨迹：末端路径 + 目标点 (+ 参考轨迹)
    """
    if best_path is None or len(best_path) == 0 or best_goal is None:
        return None

    path = np.array(best_path)  # (T,3)
    g = np.array(best_goal).reshape(-1)  # (3,)

    fig = plt.figure(figsize=(6.5, 6.2))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(path[:, 0], path[:, 1], path[:, 2], linewidth=1.6, label="EE Path")
    if ref_traj is not None:
        ax.plot(ref_traj[:, 0], ref_traj[:, 1], ref_traj[:, 2], linewidth=1.0, linestyle="--", label="Ref Traj")
    ax.scatter(path[0, 0], path[0, 1], path[0, 2], marker='o', s=40, label="Start")
    ax.scatter(path[-1, 0], path[-1, 1], path[-1, 2], marker='^', s=50, label="End")
    ax.scatter(g[0], g[1], g[2], marker='*', s=120, label="Goal")

    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title(f"最佳轨迹（按 {best_label}）- 第{best_idx+1}回合")
    ax.legend(frameon=False)

    # 尝试设定近似等轴比
    try:
        ranges = np.array([path[:, 0].ptp(), path[:, 1].ptp(), path[:, 2].ptp()])
        max_range = ranges.max() if np.isfinite(ranges).all() and ranges.max() > 0 else 0.1
        mid = path.mean(axis=0)
        ax.set_xlim(mid[0] - max_range/2, mid[0] + max_range/2)
        ax.set_ylim(mid[1] - max_range/2, mid[1] + max_range/2)
        ax.set_zlim(mid[2] - max_range/2, mid[2] + max_range/2)
    except Exception:
        pass

    return safe_savefig(fig, out_dir, f"{run_basename}_best_trajectory")


def save_csv(out_dir, run_basename, rows, header):
    safe_mkdir(out_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"{run_basename}_metrics_{ts}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"[信息] 指标 CSV 已保存：{path}")
    return path


def run_test(args):
    # 解析路径与配置
    run_dir, model_path, run_cfg = resolve_inputs(args)

    # 解析 actor 架构与默认 node_dim/num_nodes（避免 int(None) 报错）
    actor_arch = run_cfg.get("actor_arch", args.actor_arch)
    def_node_dim, def_num_nodes = resolve_arch_defaults(actor_arch)
    node_dim_cfg = run_cfg.get("node_dim", None)
    num_nodes_cfg = run_cfg.get("num_nodes", None)
    node_dim = node_dim_cfg if node_dim_cfg is not None else (args.node_dim if args.node_dim is not None else def_node_dim)
    num_nodes = num_nodes_cfg if num_nodes_cfg is not None else (args.num_nodes if args.num_nodes is not None else def_num_nodes)

    # 稠密/稀疏（没有就默认稠密）
    dense_reward_flag = bool(run_cfg.get("dense_reward", True))

    # 输出目录：项目根目录 / plots_result（自动创建）
    out_dir = os.path.join(PROJECT_ROOT, "plots_result")
    os.makedirs(out_dir, exist_ok=True)
    run_basename = os.path.basename(os.path.normpath(run_dir))

    print("[信息] 测试参数：")
    print(f"  环境ID       : {args.env}")
    print(f"  架构         : {actor_arch}")
    print(f"  node_dim     : {node_dim}")
    print(f"  num_nodes    : {num_nodes}")
    print(f"  稠密奖励     : {dense_reward_flag}")
    print(f"  运行目录     : {run_dir}")
    print(f"  模型权重     : {model_path}")
    print(f"  输出目录     : {out_dir}")
    print(f"  测试回合数   : {args.episodes}（随机目标/起点，每回合不同）")

    # 创建环境（GUI）
    env = build_env(args.env, dense_reward_flag, render_mode=args.render_mode)

    # 维度
    if hasattr(env.observation_space, "spaces") and "observation" in env.observation_space.spaces:
        obs_dim = env.observation_space["observation"].shape[0]
        goal_dim = env.observation_space["achieved_goal"].shape[0]
        state_dim = obs_dim + 2 * goal_dim
    else:
        state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # 构造“伪 cfg”传入 TD3
    class _Cfg:
        pass
    cfg = _Cfg()
    cfg.actor_arch = actor_arch
    cfg.node_dim = node_dim
    cfg.num_nodes = num_nodes
    cfg.batch_size = getattr(args, "batch_size", 256)
    cfg.expl_noise = 0.0  # 测试不用外部噪声

    # 初始化 TD3 并加载权重
    agent = TD3(state_dim, action_dim, max_action, cfg)
    sd = torch.load(model_path, map_location="cpu")
    policy_actor = agent.actor
    use_legacy_actor = False
    try:
        agent.actor.load_state_dict(sd)
    except RuntimeError:
        is_legacy_gnn_transformer = (
            actor_arch == "gnn_transformer"
            and "pos_encoding.pe" in sd
            and "readout.0.weight" in sd
            and tuple(sd["readout.0.weight"].shape) == (256, 896)
        )
        if not is_legacy_gnn_transformer:
            raise
        print("[信息] 检测到旧版 gnn_transformer 权重，切换兼容评估模式")
        policy_actor = LegacyGNNTransformerActor(
            node_dim=node_dim,
            num_nodes=num_nodes,
            action_dim=action_dim,
            max_action=max_action,
        )
        policy_actor.load_state_dict(sd, strict=True)
        use_legacy_actor = True
    policy_actor.eval()

    # 评估并记录若干随机点（默认 30）
    ep_rewards, ep_success, ep_tts, ep_min_d = [], [], [], []
    ep_paths, ep_goals, ep_refs = [], [], []
    rmses, max_devs, end_errs, path_len_exec, path_len_ref = [], [], [], [], []
    best_idx_by_min_d = None
    best_min_d_val = float("inf")
    screenshot_path = None

    for ep in range(args.episodes):
        obs, info = reset_env(env)  # 不设种子 => 随机点
        if args.save_pybullet_screenshot and screenshot_path is None:
            frame = env.unwrapped.render(width=args.screenshot_width, height=args.screenshot_height)
            if frame is not None:
                frame = stylize_pybullet_frame(frame)
                screenshot_path = safe_save_image(
                    frame,
                    out_dir,
                    f"{run_basename}_pybullet_task_scene_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                )
        done = False
        total_reward = 0.0
        tts = None
        min_d = float("inf")
        step_idx = 0
        path_xyz = []

        # 初始位姿/目标
        ee0, g0 = _extract_positions(obs, info, env)
        if ee0 is not None:
            path_xyz.append(ee0.copy())

        # 参考轨迹（如果是 Track 环境，info 内应携带）
        ref_traj = None
        if isinstance(info, dict) and "ref_traj" in info:
            ref_traj = np.array(info["ref_traj"], dtype=np.float32)

        while not done:
            state = flatten_obs(obs)
            with torch.no_grad():
                if use_legacy_actor:
                    action = _select_action_eval_actor(policy_actor, state)
                else:
                    action = _select_action_eval(agent, state)
            obs, reward, done, info = step_env(env, action)
            total_reward += float(reward)

            ee, g = _extract_positions(obs, info, env)
            if ee is not None:
                path_xyz.append(ee.copy())
            if ee is not None and g is not None:
                d = _distance(ee, g)
                if d < min_d:
                    min_d = d
                thr = float(getattr(env.unwrapped, "distance_threshold", 0.05))
                if tts is None and d < thr:
                    tts = step_idx + 1

            # 动态更新参考轨迹（Track 环境在 step 中也会刷新）
            if ref_traj is None and isinstance(info, dict) and "ref_traj" in info:
                ref_traj = np.array(info["ref_traj"], dtype=np.float32)

            step_idx += 1

        succ = 1.0 if (tts is not None) else 0.0
        ep_rewards.append(total_reward)
        ep_success.append(succ)
        ep_tts.append(tts if tts is not None else step_idx)
        ep_min_d.append(min_d if np.isfinite(min_d) else np.nan)
        exec_path = np.array(path_xyz) if len(path_xyz) > 0 else None
        ep_paths.append(exec_path)
        ep_goals.append(g if g is not None else g0)
        ep_refs.append(ref_traj)
        if False and args.save_pybullet_screenshot and screenshot_path is None and exec_path is not None:
            env.unwrapped.add_trajectory_overlay(exec_path, ref_path=ref_traj)
            frame = env.unwrapped.render(width=args.screenshot_width, height=args.screenshot_height)
            if frame is not None:
                frame = stylize_pybullet_frame(frame)
                screenshot_path = safe_save_image(
                    frame,
                    out_dir,
                    f"{run_basename}_pybullet_traj_scene_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                )

        # 轨迹指标（如果没有 ref_traj 则为 NaN）
        if exec_path is not None:
            rmses.append(_rmse_path(exec_path, ref_traj))
            # 最大偏差（相对参考）
            if ref_traj is not None and len(exec_path) > 1 and len(ref_traj) > 1:
                L = min(len(exec_path), len(ref_traj))
                max_devs.append(float(np.max(np.linalg.norm(exec_path[:L] - ref_traj[:L], axis=1))))
                end_errs.append(float(np.linalg.norm(exec_path[-1] - ref_traj[-1])))
                path_len_ref.append(_path_length(ref_traj))
            else:
                max_devs.append(np.nan)
                end_errs.append(np.nan)
                path_len_ref.append(np.nan)
            path_len_exec.append(_path_length(exec_path))
        else:
            rmses.append(np.nan); max_devs.append(np.nan); end_errs.append(np.nan)
            path_len_exec.append(np.nan); path_len_ref.append(np.nan)

        # 用“最小距离”选最优回合（如相等可再比较奖励）
        if np.isfinite(ep_min_d[-1]) and ep_min_d[-1] < best_min_d_val:
            best_min_d_val = ep_min_d[-1]
            best_idx_by_min_d = ep

        print(f"[回合 {ep+1}/{args.episodes}] 奖励 {total_reward:.3f} | 成功 {succ:.0f} | "
              f"TTS {ep_tts[-1]} | MinD {ep_min_d[-1]:.3f} m")

    # 汇总
    print("\n[汇总]")
    print(f"  平均奖励       : {np.mean(ep_rewards):.3f}")
    print(f"  成功率         : {np.mean(ep_success):.3f}")
    print(f"  平均成功步数   : {np.mean(ep_tts):.1f}")
    print(f"  平均最小距离(m): {np.nanmean(ep_min_d):.3f}")
    if any(np.isfinite(rmses)):
        print(f"  平均RMSE(m)    : {np.nanmean(rmses):.3f}")
        print(f"  平均最大偏差(m): {np.nanmean(max_devs):.3f}")
        print(f"  平均终点误差(m): {np.nanmean(end_errs):.3f}")
        print(f"  平均执行路径长(m): {np.nanmean(path_len_exec):.3f}")
        if any(np.isfinite(path_len_ref)):
            print(f"  平均参考路径长(m): {np.nanmean(path_len_ref):.3f}")

    # 保存图片（带时间戳，防覆盖）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_basename_ts = f"{run_basename}_{timestamp}"
    saved_paths = []
    if screenshot_path:
        saved_paths.append(screenshot_path)

    # 基础评估图
    saved_paths += plot_metrics_basic(out_dir, ep_rewards, ep_success, ep_tts, ep_min_d, run_basename_ts)

    # 轨迹指标图（仅当存在参考轨迹时）
    if any(ref is not None for ref in ep_refs):
        saved_paths += plot_metrics_trajectory(out_dir, rmses, max_devs, end_errs, path_len_exec, path_len_ref, run_basename_ts)

    # 最优轨迹（按 Min Distance）
    ref_for_best = None
    if best_idx_by_min_d is not None and ep_refs[best_idx_by_min_d] is not None:
        ref_for_best = ep_refs[best_idx_by_min_d]
    if best_idx_by_min_d is not None:
        p_traj = plot_best_trajectory(
            out_dir,
            best_path=ep_paths[best_idx_by_min_d],
            best_goal=ep_goals[best_idx_by_min_d],
            run_basename=run_basename_ts,
            best_idx=best_idx_by_min_d,
            best_label="最小距离",
            ref_traj=ref_for_best
        )
        if p_traj:
            saved_paths.append(p_traj)

    print("\n[已保存图像]")
    for p in saved_paths:
        if p:
            print("  ->", p)

    # 保存 CSV（每回合一行）
    rows = []
    header = [
        "episode", "reward", "success", "tts", "min_distance",
        "rmse", "max_deviation", "end_point_error",
        "path_length_exec", "path_length_ref"
    ]
    for i in range(args.episodes):
        rows.append([
            i+1,
            float(ep_rewards[i]),
            float(ep_success[i]),
            float(ep_tts[i]),
            float(ep_min_d[i]),
            float(rmses[i]) if np.isfinite(rmses[i]) else "",
            float(max_devs[i]) if np.isfinite(max_devs[i]) else "",
            float(end_errs[i]) if np.isfinite(end_errs[i]) else "",
            float(path_len_exec[i]) if np.isfinite(path_len_exec[i]) else "",
            float(path_len_ref[i]) if np.isfinite(path_len_ref[i]) else "",
        ])
    save_csv(out_dir, run_basename_ts, rows, header)

    # 关闭环境
    try:
        env.close()
    except Exception as e:
        print(f"[警告] env.close() 失败：{e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="在随机目标上评估 TD3 并生成图片与CSV")
    parser.add_argument("--env", type=str, default="KukaIiwa7Track-v0",
                        help="轨迹跟踪任务 (default: KukaIiwa7Track-v0)")
    parser.add_argument("--results_dir", type=str, default="./results",
                        help="用于自动读取最近一次 run 的根目录")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="指定某次训练目录（包含 config.json / 权重文件）")
    parser.add_argument("--model", type=str, default=None,
                        help="也可直接指定 .pt/.pth 文件，或包含权重的目录")
    parser.add_argument("--episodes", type=int, default=30,
                        help="评估回合数（默认 30，随机起点/目标）")

    # 当 config.json 缺失/字段为 None 时的回退参数
    parser.add_argument("--render_mode", type=str, default="human",
                        choices=["human", "rgb_array"],
                        help="PyBullet rendering mode used during evaluation")
    parser.add_argument("--save_pybullet_screenshot", action="store_true",
                        help="save one PyBullet screenshot to plots_result for paper figures")
    parser.add_argument("--screenshot_width", type=int, default=1280,
                        help="saved PyBullet screenshot width")
    parser.add_argument("--screenshot_height", type=int, default=720,
                        help="saved PyBullet screenshot height")
    parser.add_argument("--actor_arch", type=str, default="gnn_transformer",
                        choices=["mlp", "gnn", "transformer", "gnn_transformer"])
    parser.add_argument("--node_dim", type=int, default=None)
    parser.add_argument("--num_nodes", type=int, default=None)

    args = parser.parse_args()
    run_test(args)

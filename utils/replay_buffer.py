import numpy as np

class ReplayBuffer:
    """
    🎯 增强版Replay Buffer: 支持HER + 优先级采样 (PER)

    - Hindsight Experience Replay: 提升稀疏奖励任务样本效率
    - Prioritized Experience Replay: 基于TD-error优先采样重要样本
    Trajectory tracking must disable HER because its reward uses reference/action history.

    参考:
    - Andrychowicz et al., "Hindsight Experience Replay", NeurIPS 2017
    - Schaul et al., "Prioritized Experience Replay", ICLR 2016
    """

    def __init__(self, observation_space, action_space,
                 capacity=int(1e6),
                 her_prob=0.95, her_k=8,
                 dense_reward=False,
                 distance_threshold=0.05,
                 use_per=True, alpha=0.6, beta_start=0.4, beta_frames=100000,
                 per_mode="legacy"):
        if per_mode not in ("legacy", "proportional"):
            raise ValueError("per_mode must be legacy or proportional")
        self.per_mode = per_mode
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if not 0 <= her_prob <= 1 or her_k < 0:
            raise ValueError("her_prob must be in [0, 1] and her_k must be non-negative")
        if not np.isfinite(alpha) or alpha < 0 or not 0 <= beta_start <= 1 or beta_frames <= 0:
            raise ValueError("Invalid PER alpha, beta_start or beta_frames")
        self.ptr = 0
        self.size = 0

        # 🎯 优先级经验回放 (PER) 配置
        self.use_per = use_per
        if self.use_per:
            self.priorities = np.ones((self.capacity,), dtype=np.float32)
            self.alpha = alpha  # 优先级指数
            self.beta_start = beta_start  # 重要性采样起始值
            self.beta_frames = beta_frames  # 退火帧数
            self.frame = 0
            self.max_priority = 1.0

        # spaces (assume GoalEnv dict space, but fallbacks supported)
        self.has_goal = hasattr(observation_space, "spaces") and "observation" in observation_space.spaces
        if self.has_goal:
            self.obs_dim = int(np.prod(observation_space["observation"].shape))
            self.g_dim = int(np.prod(observation_space["achieved_goal"].shape))
        else:
            self.obs_dim = int(np.prod(observation_space.shape))
            self.g_dim = 0

        self.act_dim = int(np.prod(action_space.shape))

        # main arrays
        self.obs  = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.obs2 = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)

        self.ag   = np.zeros((self.capacity, self.g_dim), dtype=np.float32) if self.g_dim > 0 else None
        self.ag2  = np.zeros((self.capacity, self.g_dim), dtype=np.float32) if self.g_dim > 0 else None
        self.dg   = np.zeros((self.capacity, self.g_dim), dtype=np.float32) if self.g_dim > 0 else None

        self.acts = np.zeros((self.capacity, self.act_dim), dtype=np.float32)
        self.rews = np.zeros((self.capacity,), dtype=np.float32)
        self.done = np.zeros((self.capacity,), dtype=np.float32)

        # episode bookkeeping: for each index i, ep_end[i] is the (inclusive) end index of the episode
        self.ep_end = -np.ones((self.capacity,), dtype=np.int64)
        self._current_ep_start = 0  # where the ongoing episode started (in buffer index)

        # HER config
        self.her_prob = float(her_prob)
        self.her_k = int(her_k)
        self.dense_reward = bool(dense_reward)
        self.distance_threshold = float(distance_threshold)

    # ---------- helpers ----------
    def _flat_obs(self, o):
        return np.asarray(o, dtype=np.float32).reshape(-1)

    def _compute_distance(self, a, b):
        return float(np.linalg.norm(a - b))

    def _sparse_reward(self, ag2, goal):
        # 0 if achieved within threshold else -1
        d = self._compute_distance(ag2, goal)
        return 0.0 if d < self.distance_threshold else -1.0

    def _set_episode_end(self, start_idx, end_idx):
        """Set ep_end for indices [start_idx ... end_idx] on the ring."""
        if start_idx <= end_idx:
            self.ep_end[start_idx:end_idx+1] = end_idx
        else:
            # wrapped
            self.ep_end[start_idx:self.capacity] = end_idx
            self.ep_end[0:end_idx+1] = end_idx

    # ---------- API ----------
    def add(self, obs, act, next_obs, rew, done, *, terminal=None):
        """obs/next_obs can be dict (GoalEnv) or flat arrays."""
        i = self.ptr

        # unpack obs
        if isinstance(obs, dict) and self.has_goal:
            o  = self._flat_obs(obs["observation"])
            ag = self._flat_obs(obs["achieved_goal"])
            dg = self._flat_obs(obs["desired_goal"])
        else:
            o = self._flat_obs(obs)
            ag = None
            dg = None

        if isinstance(next_obs, dict) and self.has_goal:
            o2  = self._flat_obs(next_obs["observation"])
            ag2 = self._flat_obs(next_obs["achieved_goal"])
        else:
            o2 = self._flat_obs(next_obs)
            ag2 = None

        action = np.asarray(act, dtype=np.float32).reshape(-1)
        fields = [("observation", o, self.obs_dim), ("next observation", o2, self.obs_dim),
                  ("action", action, self.act_dim)]
        if self.has_goal:
            fields.extend([("achieved_goal", ag, self.g_dim), ("desired_goal", dg, self.g_dim),
                           ("next achieved_goal", ag2, self.g_dim)])
        for name, value, dimension in fields:
            if value is None or value.shape != (dimension,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must contain {dimension} finite values")
        terminal = done if terminal is None else terminal
        if not np.isfinite(rew) or float(done) not in (0.0, 1.0) or float(terminal) not in (0.0, 1.0):
            raise ValueError("reward must be finite and done must be boolean")

        # Validate the complete transition before mutating any stored field.
        self.obs[i]  = o
        self.obs2[i] = o2
        if self.g_dim > 0:
            self.ag[i]  = ag
            self.dg[i]  = dg
            self.ag2[i] = ag2 if ag2 is not None else 0.0

        self.acts[i] = action
        self.rews[i] = float(rew)
        self.done[i] = float(terminal)
        self.ep_end[i] = -1  # will be filled when episode ends

        # 🎯 新样本赋予最高优先级
        if self.use_per:
            self.priorities[i] = self.max_priority

        # move pointer
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

        # if episode ended at this transition, back-fill ep_end for the whole episode
        if done:
            self._set_episode_end(self._current_ep_start, i)
            self._current_ep_start = self.ptr  # next episode starts at next write position

    def sample_batch(self, batch_size=256):
        """Return dict with flattened s, s2, a, r, d; with HER relabeling if enabled and goals exist."""
        if self.size == 0:
            raise ValueError("ReplayBuffer is empty")
        if not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if self.use_per and self.per_mode == "legacy" and batch_size > self.size:
            raise ValueError("PER samples without replacement: batch_size cannot exceed buffer size")

        # 🎯 优先级采样
        if self.use_per:
            # 计算当前beta值（线性退火）
            beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)
            self.frame += 1

            # 基于优先级采样
            priorities = self.priorities[:self.size]
            if self.per_mode == "proportional":
                priorities = priorities.astype(np.float64)
            priorities = priorities ** self.alpha
            probs = priorities / priorities.sum()

            idxs = np.random.choice(self.size, size=batch_size, p=probs,
                                    replace=self.per_mode == "proportional")

            # 计算重要性采样权重
            weights = (self.size * probs[idxs]) ** (-beta)
            normalizer = (self.size * probs.min()) ** (-beta) if self.per_mode == "proportional" else weights.max()
            weights = weights / normalizer
        else:
            # 均匀采样
            idxs = np.random.randint(0, self.size, size=batch_size)
            weights = None

        if self.her_prob == 0 or self.g_dim == 0:
            # Batch the canonical no-HER path. Preserve the old RNG consumption
            # for completed GoalEnv episodes, so subsequent sampling is unchanged.
            if self.g_dim > 0:
                np.random.rand(np.count_nonzero(self.ep_end[idxs] >= 0))
                states = np.concatenate([self.obs[idxs], self.ag[idxs], self.dg[idxs]], axis=1)
                next_states = np.concatenate([self.obs2[idxs], self.ag2[idxs], self.dg[idxs]], axis=1)
            else:
                states, next_states = self.obs[idxs], self.obs2[idxs]
            batch = dict(obs=states, obs2=next_states, acts=self.acts[idxs],
                         rews=self.rews[idxs, None], done=self.done[idxs, None])
            if self.use_per:
                batch.update(weights=weights, indices=idxs)
            return batch

        s_list, s2_list, a_list, r_list, d_list = [], [], [], [], []

        for i in idxs:
            a  = self.acts[i]
            relabeled = False

            if self.g_dim == 0:
                # no goal: vanilla TD3
                s  = self.obs[i]
                s2 = self.obs2[i]
                r  = self.rews[i]
                d  = self.done[i]
            else:
                goal = self.dg[i].copy()

                # HER relabeling if episode end known and coin flip succeeds
                if self.ep_end[i] >= 0 and np.random.rand() < self.her_prob:
                    end = int(self.ep_end[i])
                    if end >= i:
                        j = np.random.randint(i, end + 1)
                    else:
                        # wrap-around case
                        choices = np.concatenate([np.arange(i, self.size), np.arange(0, end + 1)])
                        j = int(np.random.choice(choices))
                    goal = self.ag[j].copy()
                    relabeled = True

                s  = np.concatenate([self.obs[i],  self.ag[i],  goal], axis=0)
                s2 = np.concatenate([self.obs2[i], self.ag2[i], goal], axis=0)

                if not relabeled:
                    # 未执行 HER 时必须保留环境真正返回的奖励和终止标记。
                    # 轨迹跟踪奖励不仅依赖目标距离，还依赖参考点和动作平滑度，
                    # ReplayBuffer 无法仅凭 achieved_goal 正确重建。
                    r = self.rews[i]
                    d = self.done[i]
                elif self.dense_reward:
                    # 兼容旧的 dense+HER 用法。新的轨迹训练默认 her_prob=0，
                    # 因而不会进入该近似分支。
                    r = -self._compute_distance(self.ag2[i], goal)
                    d = 0.0
                else:
                    r = self._sparse_reward(self.ag2[i], goal)
                    d = 1.0 if r == 0.0 else 0.0  # 以 relabeled 成功为终止

            s_list.append(s)
            s2_list.append(s2)
            a_list.append(a)
            r_list.append(r)
            d_list.append(d)

        batch = dict(
            obs=np.asarray(s_list, dtype=np.float32),
            obs2=np.asarray(s2_list, dtype=np.float32),
            acts=np.asarray(a_list, dtype=np.float32),
            rews=np.asarray(r_list, dtype=np.float32).reshape(-1, 1),
            done=np.asarray(d_list, dtype=np.float32).reshape(-1, 1),
        )

        # 🎯 返回权重和索引（用于优先级更新）
        if self.use_per:
            batch["weights"] = weights
            batch["indices"] = idxs

        return batch

    def update_priorities(self, indices, td_errors):
        """🎯 根据TD-error更新优先级（PER核心）"""
        if not self.use_per:
            return

        indices = np.asarray(indices)
        td_errors = np.asarray(td_errors, dtype=np.float32).reshape(-1)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("Priority indices must be a one-dimensional integer array")
        if len(indices) != len(td_errors) or not np.isfinite(td_errors).all():
            raise ValueError("TD errors must be finite and match the priority indices")
        if np.any(indices < 0) or np.any(indices >= self.size):
            raise ValueError("Priority index outside populated buffer")
        if not len(indices):
            return
        priorities = np.abs(td_errors) + 1e-6  # 避免零优先级
        if self.per_mode == "proportional":
            self.priorities[np.unique(indices)] = 0.0
            np.maximum.at(self.priorities, indices, priorities)
        else:
            self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max())

    def sample_uniform_states(self, batch_size):
        """Uniform policy-update states, independent of prioritized critic sampling."""
        if self.size == 0 or batch_size <= 0:
            raise ValueError("Cannot sample an empty buffer or non-positive batch")
        indices = np.random.randint(self.size, size=batch_size)
        if self.g_dim:
            return np.concatenate([self.obs[indices], self.ag[indices], self.dg[indices]], axis=1)
        return self.obs[indices].copy()

    # ===== 新增：兼容你的 TD3.train() 调用 =====
    def sample(self, batch_size=256):
        """
        兼容接口：返回 (state, action, next_state, reward, not_done) [+ weights, indices]
        形状分别为：
          state      : (B, state_dim)
          action     : (B, act_dim)
          next_state : (B, state_dim)
          reward     : (B, 1)
          not_done   : (B, 1) = 1 - done

        🎯 如果启用PER，额外返回 (weights, indices)
        """
        b = self.sample_batch(batch_size)
        state      = b["obs"]
        action     = b["acts"]
        next_state = b["obs2"]
        reward     = b["rews"]  # 已是 (B,1)
        not_done   = 1.0 - b["done"]  # (B,1)

        if self.use_per:
            weights = b["weights"]
            indices = b["indices"]
            return (state, action, next_state, reward, not_done), indices, weights
        else:
            return state, action, next_state, reward, not_done

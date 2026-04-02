import numpy as np

class ReplayBuffer:
    """
    HER-capable replay buffer for GoalEnv-like observations.
    - Stores per-episode boundaries to implement 'future' relabeling.
    - Works as circular buffer; when an episode ends, we back-fill its end index.
    """

    def __init__(self, observation_space, action_space,
                 capacity=int(1e6),
                 her_prob=0.95, her_k=8,
                 dense_reward=False,
                 distance_threshold=0.05):
        self.capacity = int(capacity)
        self.ptr = 0
        self.size = 0

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
    def add(self, obs, act, next_obs, rew, done):
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

        # write
        self.obs[i]  = o
        self.obs2[i] = o2
        if self.g_dim > 0:
            self.ag[i]  = ag
            self.dg[i]  = dg
            self.ag2[i] = ag2 if ag2 is not None else 0.0

        self.acts[i] = np.asarray(act, dtype=np.float32).reshape(-1)
        self.rews[i] = float(rew)
        self.done[i] = float(done)
        self.ep_end[i] = -1  # will be filled when episode ends

        # move pointer
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

        # if episode ended at this transition, back-fill ep_end for the whole episode
        if done:
            self._set_episode_end(self._current_ep_start, i)
            self._current_ep_start = self.ptr  # next episode starts at next write position

    def sample_batch(self, batch_size=256):
        """Return dict with flattened s, s2, a, r, d; with HER relabeling if enabled and goals exist."""
        assert self.size > 0, "ReplayBuffer is empty."

        idxs = np.random.randint(0, self.size, size=batch_size)
        s_list, s2_list, a_list, r_list, d_list = [], [], [], [], []

        for i in idxs:
            a  = self.acts[i]

            if self.g_dim == 0:
                # no goal: vanilla TD3
                s  = self.obs[i]
                s2 = self.obs2[i]
                r  = self.rews[i]
                d  = self.done[i]
            else:
                goal = self.dg[i].copy()
                relabeled = False

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

                if relabeled:
                    # HER relabeled: 必须重新计算 reward（goal 已改变）
                    if self.dense_reward:
                        r = -self._compute_distance(self.ag2[i], goal)
                        d = 0.0
                    else:
                        r = self._sparse_reward(self.ag2[i], goal)
                        d = 1.0 if r == 0.0 else 0.0
                else:
                    # 未 relabel: 直接使用环境存储的原始 reward（保留完整的多分量 reward）
                    r = self.rews[i]
                    d = self.done[i]

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
        return batch

    # ===== 新增：兼容你的 TD3.train() 调用 =====
    def sample(self, batch_size=256):
        """
        兼容接口：返回 (state, action, next_state, reward, not_done)
        形状分别为：
          state      : (B, state_dim)
          action     : (B, act_dim)
          next_state : (B, state_dim)
          reward     : (B, 1)
          not_done   : (B, 1) = 1 - done
        """
        b = self.sample_batch(batch_size)
        state      = b["obs"]
        action     = b["acts"]
        next_state = b["obs2"]
        reward     = b["rews"]  # 已是 (B,1)
        not_done   = 1.0 - b["done"]  # (B,1)
        return state, action, next_state, reward, not_done

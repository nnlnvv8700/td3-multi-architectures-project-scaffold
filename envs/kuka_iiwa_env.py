import os
try:
    import gymnasium as gym
    from gymnasium import spaces
    from gymnasium.envs.registration import register
except ImportError:
    import gym
    from gym import spaces
    from gym.envs.registration import register
import numpy as np
import pybullet as p
import pybullet_data
import time


class KukaIiwa7TrackEnv(gym.Env):
    """
    KUKA LBR iiwa 7-DoF 轨迹跟踪环境 (PyBullet)

    核心功能：
    - 生成从起点到目标的直线参考轨迹 (200步)
    - 在每一步提供轨迹跟踪反馈和奖励
    - 适用于轨迹规划研究

    Observation (Dict):
      - observation: joint positions & velocities (14,)
      - achieved_goal: end-effector XYZ position (3,)
      - desired_goal: target XYZ position (3,)

    Action: joint velocity commands (7,)

    Reward (Dense) - 轨迹规划 + 平滑度优化设计：
      - Goal: -tanh(dist) + exp(-10*dist) (权重 27%) - 终点吸引
      - Tracking: exp(-5*tracking_error) (权重 36%) - 轨迹跟踪精度（最重要）
      - Success: {0, 1} (权重 27%) - 任务完成信号
      - 🎯 Smoothness: -jerk_magnitude * 0.05 (权重 10%) - 运动平滑度惩罚（新增）
      - Episode Range: -22 (初始) → +18 (学习中) → +120 (成功+平滑)

    Reward (Sparse):
      - 0 if dist < threshold else -1

    Reference: OpenAI Robotics (NeurIPS 2018) + Trajectory Tracking + Smoothness Penalty
    默认不提前终止 (done_on_success=False)，保证完整200步轨迹评估。
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        render_mode="human",
        dense_reward=True,
        distance_threshold=0.10,  # 🔧 放宽到 10cm（原 5cm 太严格）
        max_steps=200,
        sim_steps_per_action=10,
        joint_vel_limit=1.5,
        seed=None,
        done_on_success=False,  # 🔧 轨迹跟踪默认不提前终止
        observation_version=2,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.dense_reward = bool(dense_reward)
        self.distance_threshold = float(distance_threshold)
        self.max_steps = int(max_steps)
        self.sim_steps_per_action = int(sim_steps_per_action)
        self.joint_vel_limit = float(joint_vel_limit)
        self.done_on_success = bool(done_on_success)
        self.observation_version = int(observation_version)

        # 轨迹跟踪专用属性
        self._ref_traj = None  # 参考轨迹 (T, 3)
        self._current_step = 0  # 当前步数

        # 🎯 动作平滑度：跟踪上一步动作（替代Jerk惩罚）
        self._prev_action = np.zeros(7, dtype=np.float32)  # 上一步动作
        self._has_prev_action = False
        self._action_smooth_weight = 0.05  # 动作平滑度权重

        self.n_joints = 7
        self.dt = 1.0 / 240.0
        self.step_counter = 0
        self.rng = np.random.default_rng(seed)

        # Observation and Action spaces
        # v1: q(7) + qdot(7)
        # v2: q(7) + qdot(7) + phase(1) + current_ref(3) + prev_action(7)
        observation_dim = 14 if self.observation_version == 1 else 25
        obs_low = -np.inf * np.ones(observation_dim, dtype=np.float32)
        obs_high = np.inf * np.ones(observation_dim, dtype=np.float32)
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=obs_low, high=obs_high, dtype=np.float32),
            "achieved_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "desired_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        })
        self.action_space = spaces.Box(
            low=-self.joint_vel_limit, high=self.joint_vel_limit,
            shape=(self.n_joints,), dtype=np.float32
        )

        # PyBullet connection/client
        self._p_client = None       # connection id
        self.robot_id = None
        self.ee_link = None
        self.goal = None
        self._goal_body_id = None   # for removing old goal viz

    # ---------- Bullet session ----------
    def _connect(self):
        if self._p_client is None:
            if self.render_mode == "human":
                self._p_client = p.connect(p.GUI)
                # 关闭 Bullet 自带 GUI 面板
                p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self._p_client)
            else:
                self._p_client = p.connect(p.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._p_client)
            p.setTimeStep(self.dt, physicsClientId=self._p_client)
            p.setGravity(0, 0, -9.81, physicsClientId=self._p_client)

    def _disconnect(self):
        """内部安全断开：幂等 & 容错。"""
        cid = self._p_client
        self._p_client = None
        try:
            if cid is not None:
                try:
                    p.disconnect(cid)
                except Exception:
                    try:
                        p.disconnect()
                    except Exception:
                        pass
        except Exception:
            pass

    def _reset_sim(self):
        p.resetSimulation(physicsClientId=self._p_client)
        p.setTimeStep(self.dt, physicsClientId=self._p_client)
        p.setGravity(0, 0, -9.81, physicsClientId=self._p_client)
        p.loadURDF("plane.urdf", physicsClientId=self._p_client)

        # Load KUKA iiwa model
        kuka_urdf = os.path.join(pybullet_data.getDataPath(), "kuka_iiwa/model.urdf")
        self.robot_id = p.loadURDF(
            kuka_urdf, basePosition=[0, 0, 0], useFixedBase=True,
            physicsClientId=self._p_client,
        )

        # End-effector link (URDF EEF is link index 6)
        self.ee_link = 6

        # Reset joints & disable default motors (velocity controlled by us)
        for j in range(self.n_joints):
            p.resetJointState(
                self.robot_id, j, targetValue=0.0, targetVelocity=0.0,
                physicsClientId=self._p_client,
            )
            p.setJointMotorControl2(
                self.robot_id, j, p.VELOCITY_CONTROL, targetVelocity=0.0,
                force=0.0, physicsClientId=self._p_client,
            )

        # ⚠️ 注意：resetSimulation 已清空所有 body，这里不要再 removeBody 旧球，直接清引用即可
        self._goal_body_id = None

        # New random target
        self.goal = self._sample_goal()
        self._add_goal_viz()

    def _add_goal_viz(self):
        """visualize target as a small sphere; keep body id for removal on next reset."""
        radius = 0.02
        col = p.createVisualShape(
            p.GEOM_SPHERE, radius=radius, rgbaColor=[1, 0, 0, 0.85],
            physicsClientId=self._p_client,
        )
        self._goal_body_id = p.createMultiBody(
            baseMass=0, baseVisualShapeIndex=col, basePosition=self.goal.tolist(),
            physicsClientId=self._p_client,
        )

    def _sample_goal(self):
        """sample a reachable goal in front working space."""
        x = self.rng.uniform(0.4, 0.8)
        y = self.rng.uniform(-0.25, 0.25)
        z = self.rng.uniform(0.2, 0.6)
        return np.array([x, y, z], dtype=np.float32)

    # ---------- Gym API ----------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._connect()
        self._reset_sim()
        self.step_counter = 0
        self._current_step = 0  # 重置轨迹跟踪步数
        self._prev_action = np.zeros(self.n_joints, dtype=np.float32)
        self._has_prev_action = False

        # 参考轨迹包含初始状态和每次动作后的状态，共 max_steps + 1 个点。
        ee_pos = self._eef_pos()
        goal_pos = self.goal.copy()
        self._ref_traj = self._build_ref_traj(ee_pos, goal_pos, self.max_steps + 1)
        obs = self._get_obs()

        info = {
            "is_success": self._is_success(obs["achieved_goal"], self.goal),
            "ref_traj": self._ref_traj,  # 提供完整参考轨迹
            "ee_pos": ee_pos,
            "goal_pos": goal_pos,
            "reference_point": self._current_reference_point(),
            "phase": self._phase(),
            "prev_action": self._prev_action.copy(),
        }
        return obs, info

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        previous_action = self._prev_action.copy()
        had_previous_action = self._has_prev_action

        # velocity control
        p.setJointMotorControlArray(
            self.robot_id,
            jointIndices=list(range(self.n_joints)),
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=action.tolist(),
            forces=[150] * self.n_joints,
            physicsClientId=self._p_client,
        )

        for _ in range(self.sim_steps_per_action):
            p.stepSimulation(physicsClientId=self._p_client)
            if self.render_mode == "human":
                time.sleep(self.dt)

        self.step_counter += 1
        self._current_step = min(self.step_counter, self.max_steps)
        self._prev_action = action.copy()
        self._has_prev_action = True

        obs = self._get_obs()
        ee_pos = obs["achieved_goal"]
        goal_pos = obs["desired_goal"]
        dist = np.linalg.norm(ee_pos - goal_pos)

        if self.dense_reward:
            # 🎯 轨迹跟踪专用奖励设计
            # 参考：OpenAI Robotics (2018) + 轨迹跟踪扩展 + 平滑度优化

            # 1. 目标距离奖励（归一化 + 精确度）
            goal_reward = -np.tanh(dist) + np.exp(-10.0 * dist)

            # 2. 轨迹跟踪奖励（最重要）
            tracking_reward = 0.0
            if self._ref_traj is not None and self._current_step < len(self._ref_traj):
                ref_point = self._current_reference_point()
                tracking_error = float(np.linalg.norm(ee_pos - ref_point))
                tracking_reward = np.exp(-5.0 * tracking_error)  # 0.1m→0.61, 0.2m→0.37

            # 3. 成功信号
            success_reward = 1.0 if dist < self.distance_threshold else 0.0

            # 4. 🎯 动作平滑度惩罚（替代Jerk，梯度稳定）
            action_smooth_penalty = 0.0
            if had_previous_action:
                # 一阶差分：|| a_t - a_{t-1} ||^2
                action_diff = np.linalg.norm(action - previous_action)
                action_smooth_penalty = -self._action_smooth_weight * (action_diff ** 2)

            # 5. 组合（28% 目标 + 37% 轨迹 + 28% 成功 + 7% 平滑度）
            # 轨迹跟踪任务：强调轨迹精度 + 适度平滑度约束
            reward = float(
                0.28 * goal_reward +         # 基础：终点吸引
                0.37 * tracking_reward +     # 主要：轨迹跟踪精度
                0.28 * success_reward +      # 强调：任务完成信号
                0.07 * action_smooth_penalty  # 🎯 新增：动作平滑度（一阶差分，稳定）
            )

            # Episode 累积范围（200 步）：
            # - 初始随机：0.3*(-0.5)*200 + 0.4*(0.1)*200 ≈ -22
            # - 学习中期：0.3*(-0.1)*200 + 0.4*(0.3)*200 ≈ +18
            # - 成功稳定：0.3*(0.6)*200 + 0.4*(0.8)*200 + 0.3*1*100 ≈ +134
        else:
            # 稀疏奖励模式
            reward = 0.0 if dist < self.distance_threshold else -1.0

        terminated = False
        if self._is_success(obs["achieved_goal"], obs["desired_goal"]) and self.done_on_success:
            terminated = True
        truncated = (not terminated) and (self.step_counter >= self.max_steps)

        info = {
            "is_success": self._is_success(obs["achieved_goal"], obs["desired_goal"]),
            "ref_traj": self._ref_traj,  # 持续提供参考轨迹
            "ee_pos": ee_pos,
            "goal_pos": goal_pos,
            "reference_point": self._current_reference_point(),
            "phase": self._phase(),
            "prev_action": self._prev_action.copy(),
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        # 使用 GUI 模式时，PyBullet 自带窗口就是渲染
        pass

    # ---------- Close helpers ----------
    def _safe_is_connected(self):
        try:
            if self._p_client is not None:
                return bool(p.isConnected(self._p_client))
            return bool(p.isConnected())
        except Exception:
            return False

    def _body_exists(self, body_uid):
        try:
            p.getBodyInfo(body_uid, physicsClientId=self._p_client)
            return True
        except Exception:
            return False

    def close(self):
        """Idempotent close: allow multiple calls without raising and avoid C++ warnings."""
        try:
            if self._safe_is_connected():
                if self._goal_body_id is not None and self._body_exists(self._goal_body_id):
                    try:
                        p.removeBody(self._goal_body_id, physicsClientId=self._p_client)
                    except Exception:
                        pass
            self._goal_body_id = None
        except Exception:
            pass
        self._disconnect()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ---------- Helpers ----------
    def _get_q_qdot(self):
        q = []
        qd = []
        for j in range(self.n_joints):
            js = p.getJointState(self.robot_id, j, physicsClientId=self._p_client)
            q.append(js[0])
            qd.append(js[1])
        return np.array(q, dtype=np.float32), np.array(qd, dtype=np.float32)

    def _eef_pos(self):
        ls = p.getLinkState(
            self.robot_id, self.ee_link, computeForwardKinematics=True,
            physicsClientId=self._p_client,
        )
        pos = np.array(ls[4], dtype=np.float32)  # worldLinkFramePosition
        return pos

    def _get_obs(self):
        q, qd = self._get_q_qdot()
        achieved = self._eef_pos()
        observation = np.concatenate([q, qd], axis=0)
        if self.observation_version >= 2:
            observation = np.concatenate([
                observation,
                np.array([self._phase()], dtype=np.float32),
                self._current_reference_point(),
                self._prev_action,
            ], axis=0)
        obs = {
            "observation": observation.astype(np.float32),
            "achieved_goal": achieved,
            "desired_goal": self.goal.copy(),
        }
        return obs

    def _phase(self) -> float:
        """当前轨迹阶段，归一化到 [0, 1]。"""
        return float(self._current_step) / float(max(1, self.max_steps))

    def _current_reference_point(self) -> np.ndarray:
        if self._ref_traj is None:
            return self._eef_pos()
        index = min(self._current_step, len(self._ref_traj) - 1)
        return np.asarray(self._ref_traj[index], dtype=np.float32).copy()

    def _is_success(self, achieved, desired):
        return float(np.linalg.norm(achieved - desired) < self.distance_threshold)

    def _build_ref_traj(self, p0: np.ndarray, p1: np.ndarray, T: int) -> np.ndarray:
        """生成从 p0 到 p1 的直线参考轨迹 (T, 3)"""
        alphas = np.linspace(0.0, 1.0, num=T, dtype=np.float32)
        return (1.0 - alphas[:, None]) * p0[None, :] + alphas[:, None] * p1[None, :]


# 注册环境到 Gymnasium/Gym（重复注册时静默忽略）
try:
    register(
        id="KukaIiwa7Track-v0",
        entry_point="envs.kuka_iiwa_env:KukaIiwa7TrackEnv",
        max_episode_steps=200,
    )
except Exception:
    pass

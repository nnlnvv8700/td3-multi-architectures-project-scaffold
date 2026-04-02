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


# ===== Huber 损失函数（模块级，避免重复创建）=====
def _huber(x: float, delta: float = 0.5) -> float:
    """Huber 损失：小偏差二次，大偏差线性（无界，梯度不消失）
    
    Args:
        x: 输入值（差分大小）
        delta: 阈值，x<=delta 使用二次，x>delta 使用线性
        
    Returns:
        Huber 损失值
    """
    return x * x if x <= delta else 2.0 * delta * x - delta * delta


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

    Reward (Dense) - 轨迹规划 + Huber 平滑度优化设计：
      - Goal: exp(-3*dist) (权重 22%) - 终点吸引
      - Tracking: exp(-5*tracking_error) (权重 32%) - 轨迹跟踪精度（最重要）
      - Success: exp(-10*dist) (权重 22%) - 成功信号（5cm阈值+5步连续）
      - 🎯 Smooth: -Huber(action_diff, δ=0.5) × 0.08 - 动作平滑（无界惩罚）
      - 🎯 Jerk: -Huber(jerk, δ=1.0) × 0.05 - 加加速度惩罚（无界惩罚）
      - Velocity: -Huber(vel_violation, δ=0.5) × 0.03 - 速度限制
      - Episode Range: 好策略 +10~+40 | 坏策略 -200~-500（Huber无界）
      
    Reward (Sparse):
      - 0 if dist < threshold else -1

    Reference: OpenAI Robotics (NeurIPS 2018) + Trajectory Tracking + Huber Penalty
    默认不提前终止 (done_on_success=False)，保证完整200步轨迹评估。
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        render_mode="human",
        dense_reward=True,
        distance_threshold=0.05,  # 5cm 严格标准（与 success_reward 对齐）
        success_hold_steps=5,      # 连续保持 N 步才算真正成功
        max_steps=200,
        sim_steps_per_action=10,
        joint_vel_limit=1.5,
        seed=None,
        done_on_success=False,  # 轨迹跟踪默认不提前终止
    ):
        super().__init__()
        self.render_mode = render_mode
        self.dense_reward = bool(dense_reward)
        self.distance_threshold = float(distance_threshold)
        self.success_hold_steps = int(success_hold_steps)  # 连续达标步数阈值
        self.max_steps = int(max_steps)
        self.sim_steps_per_action = int(sim_steps_per_action)
        self.joint_vel_limit = float(joint_vel_limit)
        self.done_on_success = bool(done_on_success)
        
        # 轨迹跟踪专用属性
        self._ref_traj = None  # 参考轨迹 (T, 3)
        self._current_step = 0  # 当前步数
        self._success_streak = 0  # 当前连续达标步数
        
        # 🎯 动作平滑度：跟踪上一步动作（用于 Huber 惩罚）
        self._prev_action = None       # 上一步动作
        self._prev_prev_action = None  # 上上步动作（用于 jerk 二阶差分）

        self.n_joints = 7
        self.dt = 1.0 / 240.0
        self.step_counter = 0
        self.rng = np.random.default_rng(seed)

        # Observation and Action spaces
        obs_low = -np.inf * np.ones(14, dtype=np.float32)
        obs_high = np.inf * np.ones(14, dtype=np.float32)
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
        self._scene_body_ids = []
        self._debug_item_ids = []
        self._trajectory_body_ids = []
        self._camera_target = [0.50, 0.0, 0.18]
        self._camera_distance = 1.72
        self._camera_yaw = 50.0
        self._camera_pitch = -18.0
        self._camera_roll = 0.0
        self._camera_fov = 72.0
        self._camera_near = 0.1
        self._camera_far = 3.0

    # ---------- Bullet session ----------
    def _connect(self):
        if self._p_client is None:
            bg_options = (
                "--background_color_red=0.90 "
                "--background_color_green=0.95 "
                "--background_color_blue=1.00"
            )
            if self.render_mode == "human":
                self._p_client = p.connect(p.GUI, options=bg_options)
                # 关闭 Bullet 自带 GUI 面板
                p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
                p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
            else:
                self._p_client = p.connect(p.DIRECT, options=bg_options)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setTimeStep(self.dt)
            p.setGravity(0, 0, -9.81)

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
        p.resetSimulation()
        p.setTimeStep(self.dt)
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        self._scene_body_ids = []
        self._debug_item_ids = []
        self._trajectory_body_ids = []
        self._add_scene_decor()

        # Load KUKA iiwa model
        kuka_urdf = os.path.join(pybullet_data.getDataPath(), "kuka_iiwa/model.urdf")
        self.robot_id = p.loadURDF(kuka_urdf, basePosition=[0, 0, 0], useFixedBase=True)

        # End-effector link (URDF EEF is link index 6)
        self.ee_link = 6

        # Reset joints & disable default motors (velocity controlled by us)
        for j in range(self.n_joints):
            p.resetJointState(self.robot_id, j, targetValue=0.0, targetVelocity=0.0)
            p.setJointMotorControl2(self.robot_id, j, p.VELOCITY_CONTROL, targetVelocity=0.0, force=0.0)

        # ⚠️ 注意：resetSimulation 已清空所有 body，这里不要再 removeBody 旧球，直接清引用即可
        self._goal_body_id = None

        # New random target
        self.goal = self._sample_goal()
        self._add_goal_viz()
        if self.render_mode == "human":
            p.resetDebugVisualizerCamera(
                cameraDistance=self._camera_distance,
                cameraYaw=self._camera_yaw,
                cameraPitch=self._camera_pitch,
                cameraTargetPosition=self._camera_target,
            )

    def _add_scene_box(self, half_extents, position, rgba):
        visual_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=rgba,
        )
        body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual_id,
            basePosition=position,
        )
        self._scene_body_ids.append(body_id)
        return body_id

    def _add_scene_decor(self):
        # A neutral pedestal keeps the scene grounded without blocking the camera.
        self._add_scene_box(
            half_extents=[0.36, 0.36, 0.015],
            position=[0.0, 0.0, -0.015],
            rgba=[0.72, 0.74, 0.78, 1.0],
        )
        # A thin workspace mat adds context in the foreground for paper figures.
        self._add_scene_box(
            half_extents=[0.42, 0.32, 0.002],
            position=[0.55, 0.0, 0.001],
            rgba=[0.88, 0.93, 0.99, 0.95],
        )

    def clear_debug_overlays(self):
        while self._debug_item_ids:
            item_id = self._debug_item_ids.pop()
            try:
                p.removeUserDebugItem(item_id)
            except Exception:
                pass

    def clear_trajectory_overlay(self):
        while self._trajectory_body_ids:
            body_id = self._trajectory_body_ids.pop()
            try:
                if self._body_exists(body_id):
                    p.removeBody(body_id)
            except Exception:
                pass

    def add_trajectory_overlay(self, exec_path, ref_path=None):
        self.clear_debug_overlays()
        self.clear_trajectory_overlay()
        self._draw_polyline(exec_path, color=[0.10, 0.42, 0.95], width=4.5)
        if ref_path is not None:
            self._draw_polyline(ref_path, color=[0.05, 0.70, 0.30], width=2.2)
        self._add_path_markers(exec_path)

    def _draw_polyline(self, points, color, width):
        if points is None:
            return
        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim != 2 or len(pts) < 2:
            return
        thickness = 0.015 if width >= 4.0 else 0.008
        alpha = 0.95 if width >= 4.0 else 0.68
        stride = 1 if width >= 4.0 else 6
        self._add_path_segments(
            pts,
            thickness=thickness,
            rgba=[float(color[0]), float(color[1]), float(color[2]), alpha],
            stride=stride,
        )

    def _add_path_segments(self, points, thickness, rgba, stride):
        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim != 2 or len(pts) < 2:
            return
        step = max(1, int(stride))
        lift = np.array([0.0, 0.0, thickness * 0.55], dtype=np.float32)
        for idx in range(0, len(pts) - 1, step):
            p0 = pts[idx] + lift
            p1 = pts[min(idx + step, len(pts) - 1)] + lift
            seg = p1 - p0
            seg_len = float(np.linalg.norm(seg))
            if seg_len < 1e-6:
                continue
            visual_id = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[seg_len * 0.5, thickness * 0.5, thickness * 0.5],
                rgbaColor=rgba,
            )
            body_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=visual_id,
                basePosition=((p0 + p1) * 0.5).tolist(),
                baseOrientation=self._quat_from_x_axis(seg),
            )
            self._trajectory_body_ids.append(body_id)

    def _quat_from_x_axis(self, vec):
        direction = np.asarray(vec, dtype=np.float32)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-8:
            return [0.0, 0.0, 0.0, 1.0]
        direction /= norm
        base = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        dot = float(np.clip(np.dot(base, direction), -1.0, 1.0))
        if dot > 1.0 - 1e-7:
            return [0.0, 0.0, 0.0, 1.0]
        if dot < -1.0 + 1e-7:
            return [0.0, 0.0, 1.0, 0.0]
        axis = np.cross(base, direction)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-8:
            return [0.0, 0.0, 0.0, 1.0]
        axis /= axis_norm
        angle = float(np.arccos(dot))
        half = angle * 0.5
        sin_half = float(np.sin(half))
        return [
            float(axis[0] * sin_half),
            float(axis[1] * sin_half),
            float(axis[2] * sin_half),
            float(np.cos(half)),
        ]

    def _add_path_markers(self, points):
        if points is None:
            return
        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim != 2 or len(pts) == 0:
            return
        self._add_marker_sphere(pts[0] + np.array([0.0, 0.0, 0.012], dtype=np.float32), 0.016, [0.95, 0.75, 0.15, 1.0])
        self._add_marker_sphere(pts[-1] + np.array([0.0, 0.0, 0.012], dtype=np.float32), 0.016, [0.05, 0.05, 0.05, 1.0])
        start_id = p.addUserDebugText(
            text="Start",
            textPosition=(pts[0] + np.array([0.0, 0.0, 0.05], dtype=np.float32)).tolist(),
            textColorRGB=[0.10, 0.42, 0.95],
            textSize=1.1,
            lifeTime=0,
        )
        end_id = p.addUserDebugText(
            text="End",
            textPosition=(pts[-1] + np.array([0.0, 0.0, 0.05], dtype=np.float32)).tolist(),
            textColorRGB=[0.02, 0.02, 0.02],
            textSize=1.1,
            lifeTime=0,
        )
        self._debug_item_ids.extend([start_id, end_id])

    def _add_marker_sphere(self, position, radius, rgba):
        visual_id = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=rgba,
        )
        body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual_id,
            basePosition=np.asarray(position, dtype=np.float32).tolist(),
        )
        self._trajectory_body_ids.append(body_id)

    def _add_goal_viz(self):
        """visualize target as a small sphere; keep body id for removal on next reset."""
        radius = 0.02
        col = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=[1, 0, 0, 0.85])
        self._goal_body_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=col, basePosition=self.goal.tolist())

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
        self._success_streak = 0       # 重置连续成功计数
        self._prev_action = None       # 重置动作历史
        self._prev_prev_action = None  # 重置二阶历史
        
        obs = self._get_obs()
        
        # 生成参考轨迹：从当前EE位置到目标的直线轨迹
        ee_pos = obs["achieved_goal"]
        goal_pos = obs["desired_goal"]
        self._ref_traj = self._build_ref_traj(ee_pos, goal_pos, self.max_steps)
        
        info = {
            "is_success": self._is_success(obs["achieved_goal"], self.goal),
            "ref_traj": self._ref_traj,  # 提供完整参考轨迹
            "ee_pos": ee_pos,
            "goal_pos": goal_pos,
        }
        return obs, info

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # velocity control
        p.setJointMotorControlArray(
            self.robot_id,
            jointIndices=list(range(self.n_joints)),
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=action.tolist(),
            forces=[150] * self.n_joints,
        )

        for _ in range(self.sim_steps_per_action):
            p.stepSimulation()
            if self.render_mode == "human":
                time.sleep(self.dt)

        obs = self._get_obs()
        ee_pos = obs["achieved_goal"]
        goal_pos = obs["desired_goal"]
        dist = np.linalg.norm(ee_pos - goal_pos)

        if self.dense_reward:
            # ─── 轨迹跟踪专用奖励设计 ────────────────────────────────────────
            # 设计原则：所有正向分量值域 ∈ [0, +1]，惩罚分量值域 ∈ [-1, 0]
            # 初始(远)→低奖励，收敛(近)→高奖励，曲线自然上升

            # 1. 目标距离奖励 → 值域 [0, 1]
            #    dist=0→1.0, dist=0.1→0.74, dist=0.3→0.41, dist=0.5→0.22
            goal_reward = np.exp(-3.0 * dist)

            # 2. 轨迹跟踪奖励 → 值域 [0, 1]
            tracking_reward = 0.0
            tracking_error = 0.0
            if self._ref_traj is not None and self._current_step < len(self._ref_traj):
                ref_point = self._ref_traj[self._current_step]
                tracking_error = float(np.linalg.norm(ee_pos - ref_point))
                tracking_reward = np.exp(-5.0 * tracking_error)

            # 3. 成功信号 → 值域 [0, 1]
            #    exp(-10*0.05)=0.607，exp(-10*0.10)=0.368
            #    阈值 5cm 时奖励 0.607，与 distance_threshold=0.05 对齐
            success_reward = float(np.exp(-10.0 * dist))

            # 4. 动作平滑度惩罚 —— Huber 式惩罚（无界，梯度不消失）
            #    huber(x, d) = x²        if x <= d   (小偏差：平方，梯度线性增长)
            #                  2*d*x-d²  if x >  d   (大偏差：线性，梯度恒定=2d)
            #    优于 tanh：tanh 在 diff>1 梯度→0，策略发现"抖多抖少惩罚一样"→失效
            #    delta=0.5：diff=0.5 以内二次，超出线性；典型好策略 diff<0.3 惩罚<0.09
            action_smooth_penalty = 0.0
            jerk_penalty = 0.0
            if self._prev_action is not None:
                action_diff = float(np.linalg.norm(action - self._prev_action))
                action_smooth_penalty = -_huber(action_diff, delta=0.5)
                # jerk 惩罚（二阶差分），delta 放宽到 1.0（jerk 天然比 diff 大）
                if self._prev_prev_action is not None:
                    jerk_val = float(np.linalg.norm(
                        action - 2 * self._prev_action + self._prev_prev_action))
                    jerk_penalty = -_huber(jerk_val, delta=1.0)

            # 5. 关节速度惩罚（超限才惩罚，Huber 式）
            vel_limit_frac = 0.9
            vel_violation = np.maximum(0.0, np.abs(action) - vel_limit_frac * self.joint_vel_limit)
            vel_raw = float(np.sum(vel_violation))
            vel_penalty = -_huber(vel_raw, delta=0.5)

            # 6. 组合奖励
            #    正向 76%: 22% 目标 + 32% 轨迹 + 22% 成功
            #    惩罚权重小（Huber 无界，自然随行为变差而增大）:
            #      smooth: diff=0.3(好) → -0.04/step；diff=2(差) → -0.55/step
            #      jerk:   jerk=0.5(好) → -0.025/step；jerk=5(差) → -0.225/step
            reward = float(
                0.22 * goal_reward +
                0.32 * tracking_reward +
                0.22 * success_reward +
                0.08 * action_smooth_penalty +
                0.03 * vel_penalty +
                0.05 * jerk_penalty
            )
            # 各分量存入 _last_reward_components，供外部评估使用
            self._last_reward_components = {
                "goal": float(0.22 * goal_reward),
                "tracking": float(0.32 * tracking_reward),
                "success": float(0.22 * success_reward),
                "smooth": float(0.08 * action_smooth_penalty),
                "vel": float(0.03 * vel_penalty),
                "jerk": float(0.05 * jerk_penalty),
                "dist": float(dist),
                "tracking_error": float(tracking_error) if self._ref_traj is not None and self._current_step <= len(self._ref_traj) else float("nan"),
            }
        else:
            # 稀疏奖励模式
            reward = 0.0 if dist < self.distance_threshold else -1.0

        self.step_counter += 1
        self._current_step += 1  # 更新轨迹跟踪步数
        self._prev_prev_action = self._prev_action  # 保存上一步为上上步
        self._prev_action = action.copy()  # 保存当前动作

        # 更新连续成功计数（dist 在上方 step() 内已计算）
        if dist < self.distance_threshold:
            self._success_streak += 1
        else:
            self._success_streak = 0  # 一旦离开阈值，重置计数
        is_success_now = float(self._success_streak >= self.success_hold_steps)

        terminated = False
        if is_success_now and self.done_on_success:
            terminated = True
        truncated = (not terminated) and (self.step_counter >= self.max_steps)

        info = {
            "is_success": is_success_now,
            "ref_traj": self._ref_traj,  # 持续提供参考轨迹
            "ee_pos": ee_pos,
            "goal_pos": goal_pos,
            "reward_components": getattr(self, "_last_reward_components", {}),
        }
        return obs, reward, terminated, truncated, info

    def render(self, width=1280, height=720):
        if not self._safe_is_connected() or self.robot_id is None:
            return None

        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self._camera_target,
            distance=self._camera_distance,
            yaw=self._camera_yaw,
            pitch=self._camera_pitch,
            roll=self._camera_roll,
            upAxisIndex=2,
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=self._camera_fov,
            aspect=float(width) / float(height),
            nearVal=self._camera_near,
            farVal=self._camera_far,
        )
        renderer = p.ER_BULLET_HARDWARE_OPENGL if self.render_mode == "human" else p.ER_TINY_RENDERER
        _, _, rgba, _, _ = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=renderer,
        )
        rgba = np.reshape(rgba, (height, width, 4))
        return np.ascontiguousarray(rgba[:, :, :3], dtype=np.uint8)

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
            p.getBodyInfo(body_uid)
            return True
        except Exception:
            return False

    def close(self):
        """Idempotent close: allow multiple calls without raising and avoid C++ warnings."""
        try:
            self.clear_debug_overlays()
            self.clear_trajectory_overlay()
            if self._safe_is_connected():
                if self._goal_body_id is not None and self._body_exists(self._goal_body_id):
                    try:
                        p.removeBody(self._goal_body_id)
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
            js = p.getJointState(self.robot_id, j)
            q.append(js[0])
            qd.append(js[1])
        return np.array(q, dtype=np.float32), np.array(qd, dtype=np.float32)

    def _eef_pos(self):
        ls = p.getLinkState(self.robot_id, self.ee_link, computeForwardKinematics=True)
        pos = np.array(ls[4], dtype=np.float32)  # worldLinkFramePosition
        return pos

    def _get_obs(self):
        q, qd = self._get_q_qdot()
        achieved = self._eef_pos()
        obs = {
            "observation": np.concatenate([q, qd], axis=0),
            "achieved_goal": achieved,
            "desired_goal": self.goal.copy(),
        }
        return obs

    def _is_success(self, achieved, desired):
        """连续保持 success_hold_steps 步在阈值内才算成功（step() 中维护计数器）。
        此方法仅查询当前计数状态，不用于 reset() 的初始 is_success（初始必为 0）。
        """
        return float(self._success_streak >= self.success_hold_steps)
    
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

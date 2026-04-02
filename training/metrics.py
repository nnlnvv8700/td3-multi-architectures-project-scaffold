"""
训练指标计算模块
===============
提供训练和评估过程中所需的各类指标计算函数。

包括：
- 轨迹指标（RMSE、最大偏差、终点误差、路径长度）
- 成功判定
- 距离计算
- 统计辅助函数

作者: Auto-generated
日期: 2025-11-10
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
import gymnasium as gym


def extract_distance(
    obs: Any,
    info: Optional[Dict],
    env: gym.Env
) -> Optional[float]:
    """
    从观测和信息中提取末端执行器到目标的距离
    
    Args:
        obs: 环境观测值（可能是 dict 或 array）
        info: 环境返回的信息字典
        env: Gym 环境对象
        
    Returns:
        距离值（米），如果无法提取则返回 None
        
    Examples:
        >>> distance = extract_distance(obs, info, env)
        >>> if distance is not None:
        ...     print(f"Distance to goal: {distance:.3f}m")
    """
    # 尝试从 dict 观测中提取
    if isinstance(obs, dict) and "achieved_goal" in obs and "desired_goal" in obs:
        ag = np.array(obs["achieved_goal"], dtype=np.float32)
        dg = np.array(obs["desired_goal"], dtype=np.float32)
        return float(np.linalg.norm(ag - dg))
    
    # 尝试从 info 中提取
    if info is not None and "ee_pos" in info and "goal_pos" in info:
        ee = np.array(info["ee_pos"], dtype=np.float32)
        gg = np.array(info["goal_pos"], dtype=np.float32)
        return float(np.linalg.norm(ee - gg))
    
    return None


def is_success_state(
    obs: Any,
    info: Optional[Dict],
    env: gym.Env
) -> bool:
    """
    判断当前状态是否达到成功条件
    
    优先使用 info['is_success']，否则基于距离和阈值判断。
    
    Args:
        obs: 环境观测值
        info: 环境信息字典
        env: Gym 环境对象
        
    Returns:
        True 表示成功，False 表示失败
        
    Examples:
        >>> if is_success_state(obs, info, env):
        ...     print("Task completed successfully!")
    """
    # 优先使用环境提供的成功标志
    if isinstance(info, dict) and "is_success" in info:
        try:
            return bool(info["is_success"])
        except Exception:
            pass
    
    # 基于距离判断
    distance = extract_distance(obs, info, env)
    if distance is None:
        return False
    
    threshold = float(getattr(getattr(env, "unwrapped", env), "distance_threshold", 0.05))
    return bool(distance < threshold)


def extract_ee_and_goal(
    obs: Any,
    info: Optional[Dict]
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    从观测和信息中提取末端执行器位置和目标位置
    
    Args:
        obs: 环境观测值
        info: 环境信息字典
        
    Returns:
        (ee_pos, goal_pos) 元组，每个都是 shape (3,) 的数组，无法提取则为 None
        
    Examples:
        >>> ee, goal = extract_ee_and_goal(obs, info)
        >>> if ee is not None and goal is not None:
        ...     distance = np.linalg.norm(ee - goal)
    """
    ee, goal = None, None
    
    # 尝试从 info 提取
    if isinstance(info, dict):
        if "ee_pos" in info:
            ee = np.array(info["ee_pos"], dtype=np.float32)
        if "goal_pos" in info:
            goal = np.array(info["goal_pos"], dtype=np.float32)
    
    # 尝试从 obs 提取（补充或覆盖）
    if isinstance(obs, dict):
        if ee is None and "achieved_goal" in obs:
            ee = np.array(obs["achieved_goal"], dtype=np.float32)
        if goal is None and "desired_goal" in obs:
            goal = np.array(obs["desired_goal"], dtype=np.float32)
    
    return ee, goal


def compute_rmse_path(
    executed_path: Optional[np.ndarray],
    reference_path: Optional[np.ndarray]
) -> float:
    """
    计算执行路径与参考路径之间的均方根误差（RMSE）
    
    两条路径会对齐到较短的长度进行比较。
    
    Args:
        executed_path: 执行路径，shape (N, 3)
        reference_path: 参考路径，shape (M, 3)
        
    Returns:
        RMSE 值（米），如果无法计算则返回 nan
        
    Examples:
        >>> rmse = compute_rmse_path(exec_path, ref_path)
        >>> print(f"Trajectory RMSE: {rmse:.4f}m")
    """
    if executed_path is None or reference_path is None:
        return np.nan
    
    min_length = min(len(executed_path), len(reference_path))
    if min_length <= 1:
        return np.nan
    
    diff = executed_path[:min_length] - reference_path[:min_length]
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def compute_path_length(points: Optional[np.ndarray]) -> float:
    """
    计算路径的总长度（连续点之间的欧氏距离之和）
    
    Args:
        points: 路径点序列，shape (N, 3)
        
    Returns:
        路径长度（米），如果无法计算则返回 nan
        
    Examples:
        >>> length = compute_path_length(trajectory_points)
        >>> print(f"Path length: {length:.3f}m")
    """
    if points is None or len(points) < 2:
        return np.nan
    
    diffs = np.diff(points, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    return float(np.sum(distances))


def compute_max_deviation(
    executed_path: Optional[np.ndarray],
    reference_path: Optional[np.ndarray]
) -> float:
    """
    计算执行路径相对参考路径的最大偏差
    
    Args:
        executed_path: 执行路径，shape (N, 3)
        reference_path: 参考路径，shape (M, 3)
        
    Returns:
        最大偏差值（米），如果无法计算则返回 nan
        
    Examples:
        >>> max_dev = compute_max_deviation(exec_path, ref_path)
        >>> print(f"Maximum deviation: {max_dev:.4f}m")
    """
    if executed_path is None or reference_path is None:
        return np.nan
    
    if len(executed_path) < 1 or len(reference_path) < 1:
        return np.nan
    
    min_length = min(len(executed_path), len(reference_path))
    deviations = np.linalg.norm(
        executed_path[:min_length] - reference_path[:min_length],
        axis=1
    )
    return float(np.max(deviations))


def compute_endpoint_error(
    executed_path: Optional[np.ndarray],
    reference_path: Optional[np.ndarray]
) -> float:
    """
    计算执行路径终点与参考路径终点之间的误差
    
    Args:
        executed_path: 执行路径，shape (N, 3)
        reference_path: 参考路径，shape (M, 3)
        
    Returns:
        终点误差值（米），如果无法计算则返回 nan
        
    Examples:
        >>> end_error = compute_endpoint_error(exec_path, ref_path)
        >>> print(f"Endpoint error: {end_error:.4f}m")
    """
    if executed_path is None or reference_path is None:
        return np.nan
    
    if len(executed_path) < 1 or len(reference_path) < 1:
        return np.nan
    
    return float(np.linalg.norm(executed_path[-1] - reference_path[-1]))


def nanmean(values: np.ndarray) -> float:
    """
    计算平均值，忽略 NaN 值
    
    Args:
        values: 数值数组
        
    Returns:
        平均值，如果全是 NaN 或为空则返回 nan
        
    Examples:
        >>> avg = nanmean(np.array([1.0, np.nan, 3.0, 4.0]))
        >>> print(f"Average: {avg:.2f}")  # 2.67
    """
    values = np.array(values, dtype=np.float64)
    return float(np.nanmean(values)) if values.size > 0 else float("nan")


def moving_average(data: np.ndarray, window_size: int = 50) -> np.ndarray:
    """
    计算移动平均（用于平滑曲线）
    
    Args:
        data: 原始数据
        window_size: 窗口大小
        
    Returns:
        平滑后的数据
        
    Examples:
        >>> smoothed = moving_average(noisy_data, window_size=100)
    """
    if data is None or len(data) == 0:
        return np.array([])
    
    window_size = max(1, int(window_size))
    if len(data) < window_size:
        return np.array([])
    
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


class MetricsAccumulator:
    """
    指标累积器 - 用于在训练/评估过程中收集和计算指标
    
    Examples:
        >>> accumulator = MetricsAccumulator()
        >>> for episode in episodes:
        ...     accumulator.add_episode(reward=10.5, success=True, tts=45)
        >>> stats = accumulator.get_statistics()
        >>> print(f"Average reward: {stats['mean_reward']:.2f}")
    """
    
    def __init__(self):
        """初始化累积器"""
        self.reset()
    
    def reset(self) -> None:
        """重置所有统计数据"""
        self.rewards = []
        self.successes = []
        self.tts_values = []
        self.min_distances = []
        self.rmse_values = []
        self.max_deviations = []
        self.endpoint_errors = []
        self.path_lengths_exec = []
        self.path_lengths_ref = []
    
    def add_episode(
        self,
        reward: float,
        success: bool,
        tts: Optional[int] = None,
        min_distance: Optional[float] = None,
        rmse: Optional[float] = None,
        max_deviation: Optional[float] = None,
        endpoint_error: Optional[float] = None,
        path_length_exec: Optional[float] = None,
        path_length_ref: Optional[float] = None
    ) -> None:
        """
        添加一个回合的指标
        
        Args:
            reward: 回合总奖励
            success: 是否成功
            tts: Time to Success（成功所需步数）
            min_distance: 最小距离
            rmse: 轨迹 RMSE
            max_deviation: 最大偏差
            endpoint_error: 终点误差
            path_length_exec: 执行路径长度
            path_length_ref: 参考路径长度
        """
        self.rewards.append(reward)
        self.successes.append(1.0 if success else 0.0)
        
        if tts is not None:
            self.tts_values.append(tts)
        if min_distance is not None:
            self.min_distances.append(min_distance)
        if rmse is not None:
            self.rmse_values.append(rmse)
        if max_deviation is not None:
            self.max_deviations.append(max_deviation)
        if endpoint_error is not None:
            self.endpoint_errors.append(endpoint_error)
        if path_length_exec is not None:
            self.path_lengths_exec.append(path_length_exec)
        if path_length_ref is not None:
            self.path_lengths_ref.append(path_length_ref)
    
    def get_statistics(self) -> Dict[str, float]:
        """
        获取统计数据
        
        Returns:
            包含各项统计指标的字典
        """
        return {
            "mean_reward": float(np.mean(self.rewards)) if self.rewards else 0.0,
            "std_reward": float(np.std(self.rewards)) if self.rewards else 0.0,
            "mean_success": float(np.mean(self.successes)) if self.successes else 0.0,
            "mean_tts": nanmean(self.tts_values),
            "mean_min_distance": nanmean(self.min_distances),
            "mean_rmse": nanmean(self.rmse_values),
            "mean_max_deviation": nanmean(self.max_deviations),
            "mean_endpoint_error": nanmean(self.endpoint_errors),
            "mean_path_length_exec": nanmean(self.path_lengths_exec),
            "mean_path_length_ref": nanmean(self.path_lengths_ref),
            "num_episodes": len(self.rewards)
        }

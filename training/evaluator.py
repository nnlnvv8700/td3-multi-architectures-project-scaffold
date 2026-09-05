"""
评估器模块
=========
提供智能体评估的封装类和函数。

包括：
- 标准策略评估
- 轨迹跟踪评估
- 批量评估
- 评估结果保存

作者: Auto-generated
日期: 2025-11-10
"""

import numpy as np
import gymnasium as gym
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
from training.observation import flatten_obs
from utils.gym_compat import reset_env, step_env

from training.metrics import (
    extract_ee_and_goal,
    compute_rmse_path,
    compute_path_length,
    compute_max_deviation,
    compute_endpoint_error,
    is_success_state,
    extract_distance,
    MetricsAccumulator
)
from training.metrics import trajectory_success


@dataclass
class EvaluationResult:
    """
    评估结果数据类
    
    Attributes:
        mean_reward: 平均回合奖励
        success_rate: 成功率
        mean_tts: 平均 Time to Success
        mean_min_distance: 平均最小距离
        mean_rmse: 平均轨迹 RMSE
        mean_max_deviation: 平均最大偏差
        mean_endpoint_error: 平均终点误差
        mean_path_length_exec: 平均执行路径长度
        mean_path_length_ref: 平均参考路径长度
        num_episodes: 评估回合数
    """
    mean_reward: float
    success_rate: float
    mean_tts: float
    mean_min_distance: float
    mean_rmse: float
    mean_max_deviation: float
    mean_endpoint_error: float
    mean_path_length_exec: float
    mean_path_length_ref: float
    num_episodes: int
    endpoint_success_rate: float = float("nan")
    tracking_success_rate: float = float("nan")
    mean_action_rate: float = float("nan")
    mean_jerk: float = float("nan")
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            "mean_reward": self.mean_reward,
            "success_rate": self.success_rate,
            "mean_tts": self.mean_tts,
            "mean_min_distance": self.mean_min_distance,
            "mean_rmse": self.mean_rmse,
            "mean_max_deviation": self.mean_max_deviation,
            "mean_endpoint_error": self.mean_endpoint_error,
            "mean_path_length_exec": self.mean_path_length_exec,
            "mean_path_length_ref": self.mean_path_length_ref,
            "num_episodes": self.num_episodes,
            "endpoint_success_rate": self.endpoint_success_rate,
            "tracking_success_rate": self.tracking_success_rate,
            "mean_action_rate": self.mean_action_rate,
            "mean_jerk": self.mean_jerk,
        }
    
    def __str__(self) -> str:
        """格式化输出"""
        return (
            f"Evaluation Results ({self.num_episodes} episodes):\n"
            f"  Reward: {self.mean_reward:.2f}\n"
            f"  Success Rate: {self.success_rate:.2%}\n"
            f"  Time to Success: {self.mean_tts:.1f}\n"
            f"  Min Distance: {self.mean_min_distance:.4f}m\n"
            f"  RMSE: {self.mean_rmse:.4f}m\n"
            f"  Max Deviation: {self.mean_max_deviation:.4f}m\n"
            f"  Endpoint Error: {self.mean_endpoint_error:.4f}m"
        )


class PolicyEvaluator:
    """
    策略评估器
    
    用于评估训练好的智能体在环境中的表现。
    
    Examples:
        >>> evaluator = PolicyEvaluator(env, agent)
        >>> result = evaluator.evaluate(num_episodes=10)
        >>> print(result)
    """
    
    def __init__(
        self,
        env: gym.Env,
        agent: Any,
        distance_threshold: float = 0.05,
        seed: Optional[int] = 10000,
    ):
        """
        初始化评估器
        
        Args:
            env: Gym 环境
            agent: 智能体对象（需要有 select_action 方法）
            distance_threshold: 成功判定的距离阈值（米）
        """
        self.env = env
        self.agent = agent
        self.distance_threshold = distance_threshold
        self.seed = seed
    
    def _select_action_deterministic(self, state: np.ndarray) -> np.ndarray:
        """
        选择确定性动作（兼容不同的 select_action 签名）
        
        Args:
            state: 状态向量
            
        Returns:
            动作向量
        """
        try:
            return self.agent.select_action(state, deterministic=True)
        except TypeError:
            return self.agent.select_action(state)
    
    def _flatten_observation(self, obs: Any) -> np.ndarray:
        """
        将观测展平为向量（处理 dict 类型观测）
        
        Args:
            obs: 原始观测
            
        Returns:
            展平后的状态向量
        """
        return flatten_obs(obs)

    def _reset_env(self, env: gym.Env, seed: Optional[int] = None) -> Tuple[Any, Dict]:
        return reset_env(env, **({"seed": seed} if seed is not None else {}))

    def _step_env(self, env: gym.Env, action: np.ndarray) -> Tuple[Any, float, bool, Dict]:
        return step_env(env, action)

    def evaluate_episode(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        评估单个回合
        
        Returns:
            包含回合指标的字典
        """
        obs, info = self._reset_env(self.env, seed=seed)
        done = False
        total_reward = 0.0
        step_count = 0
        tts = None
        min_distance = float("inf")
        applied_actions = [np.zeros(self.env.action_space.shape)]
        
        # 记录执行轨迹
        executed_path = []
        ee0, _ = extract_ee_and_goal(obs, info)
        if ee0 is not None:
            executed_path.append(ee0.copy())
        
        # 获取参考轨迹（如果有）
        reference_path = None
        if isinstance(info, dict) and "ref_traj" in info:
            try:
                reference_path = np.array(info["ref_traj"], dtype=np.float32)
            except Exception:
                pass
        
        # 执行回合
        while not done:
            action = self._select_action_deterministic(self._flatten_observation(obs))
            obs, reward, done, info = self._step_env(self.env, action)
            applied_actions.append(np.asarray(info.get("applied_action", action)).copy())
            total_reward += float(reward)
            
            # 记录末端位置
            ee, goal = extract_ee_and_goal(obs, info)
            if ee is not None:
                executed_path.append(ee.copy())
            
            # 计算距离
            if ee is not None and goal is not None:
                distance = float(np.linalg.norm(ee - goal))
                if distance < min_distance:
                    min_distance = distance
                
                # 记录首次成功时间
                if tts is None and distance < self.distance_threshold:
                    tts = step_count + 1
            
            # 更新参考轨迹（Track环境可能在step中更新）
            if reference_path is None and isinstance(info, dict) and "ref_traj" in info:
                try:
                    reference_path = np.array(info["ref_traj"], dtype=np.float32)
                except Exception:
                    pass
            
            step_count += 1
        
        # 计算轨迹指标
        exec_path_array = np.array(executed_path) if len(executed_path) > 0 else None
        rmse = compute_rmse_path(exec_path_array, reference_path)
        endpoint_error = compute_endpoint_error(exec_path_array, reference_path)
        endpoint_success = bool(endpoint_error < self.distance_threshold)
        tracking_success = trajectory_success(exec_path_array, reference_path, self.distance_threshold,
                                               getattr(self.env.unwrapped, "tracking_success_threshold", 0.05))
        period = getattr(self.env.unwrapped, "dt", 1.0 / 240) * getattr(self.env.unwrapped, "sim_steps_per_action", 10)
        action_rate = float(np.sqrt(np.mean((np.diff(applied_actions, axis=0) / period) ** 2)))
        jerk = float("nan")
        if exec_path_array is not None and len(exec_path_array) >= 4:
            jerk = float(np.mean(np.linalg.norm(np.diff(exec_path_array, n=3, axis=0) / period**3, axis=1)))
        
        return {
            "reward": total_reward,
            "success": tracking_success if getattr(self.env.unwrapped, "reward_mode", "legacy") == "tracking" else tts is not None,
            "tts": tts if tts is not None else step_count,
            "min_distance": min_distance if np.isfinite(min_distance) else np.nan,
            "rmse": rmse,
            "max_deviation": compute_max_deviation(exec_path_array, reference_path),
            "endpoint_error": endpoint_error,
            "endpoint_success": endpoint_success,
            "tracking_success": tracking_success,
            "action_rate": action_rate,
            "jerk": jerk,
            "path_length_exec": compute_path_length(exec_path_array),
            "path_length_ref": compute_path_length(reference_path),
            "steps": step_count
        }
    
    def evaluate(
        self,
        num_episodes: int = 10,
        verbose: bool = False
    ) -> EvaluationResult:
        """
        评估多个回合并返回统计结果
        
        Args:
            num_episodes: 评估回合数
            verbose: 是否打印详细信息
            
        Returns:
            评估结果对象
            
        Examples:
            >>> result = evaluator.evaluate(num_episodes=10, verbose=True)
            >>> print(f"Success rate: {result.success_rate:.2%}")
        """
        if num_episodes <= 0:
            raise ValueError("num_episodes must be positive")
        accumulator = MetricsAccumulator()
        episode_records = []
        
        for episode_idx in range(num_episodes):
            episode_seed = None if self.seed is None else self.seed + episode_idx
            episode_result = self.evaluate_episode(seed=episode_seed)
            episode_records.append(episode_result)
            
            accumulator.add_episode(
                reward=episode_result["reward"],
                success=episode_result["success"],
                tts=episode_result["tts"],
                min_distance=episode_result["min_distance"],
                rmse=episode_result["rmse"],
                max_deviation=episode_result["max_deviation"],
                endpoint_error=episode_result["endpoint_error"],
                path_length_exec=episode_result["path_length_exec"],
                path_length_ref=episode_result["path_length_ref"]
            )
            
            if verbose:
                success_mark = "✓" if episode_result["success"] else "✗"
                print(f"  Episode {episode_idx+1}/{num_episodes}: "
                      f"{success_mark} R={episode_result['reward']:.2f}, "
                      f"Steps={episode_result['steps']}")
        
        stats = accumulator.get_statistics()
        
        return EvaluationResult(
            mean_reward=stats["mean_reward"],
            success_rate=stats["mean_success"],
            mean_tts=stats["mean_tts"],
            mean_min_distance=stats["mean_min_distance"],
            mean_rmse=stats["mean_rmse"],
            mean_max_deviation=stats["mean_max_deviation"],
            mean_endpoint_error=stats["mean_endpoint_error"],
            mean_path_length_exec=stats["mean_path_length_exec"],
            mean_path_length_ref=stats["mean_path_length_ref"],
            num_episodes=stats["num_episodes"],
            endpoint_success_rate=float(np.mean([record["endpoint_success"] for record in episode_records])),
            tracking_success_rate=float(np.mean([record["tracking_success"] for record in episode_records])),
            mean_action_rate=float(np.mean([record["action_rate"] for record in episode_records])),
            mean_jerk=float(np.mean([record["jerk"] for record in episode_records])),
        )


def evaluate_policy(
    agent: Any,
    env: gym.Env,
    eval_episodes: int = 10,
    seed: Optional[int] = 10000,
) -> Tuple[float, float, float, float, float, float, float, float, float]:
    """
    评估策略的便捷函数（保持向后兼容）
    
    Args:
        agent: 智能体对象
        env: Gym 环境
        eval_episodes: 评估回合数
        
    Returns:
        (平均奖励, 成功率, 平均TTS, 平均最小距离, 平均RMSE, 
         平均最大偏差, 平均终点误差, 平均执行路径长度, 平均参考路径长度)
        
    Examples:
        >>> metrics = evaluate_policy(agent, env, eval_episodes=10)
        >>> avg_reward, success_rate, tts, min_d, rmse, max_dev, end_err, pl_exec, pl_ref = metrics
    """
    distance_threshold = float(getattr(env.unwrapped, "distance_threshold", 0.05))
    evaluator = PolicyEvaluator(env, agent, distance_threshold, seed=seed)
    result = evaluator.evaluate(num_episodes=eval_episodes, verbose=False)
    
    return (
        result.mean_reward,
        result.success_rate,
        result.mean_tts,
        result.mean_min_distance,
        result.mean_rmse,
        result.mean_max_deviation,
        result.mean_endpoint_error,
        result.mean_path_length_exec,
        result.mean_path_length_ref
    )

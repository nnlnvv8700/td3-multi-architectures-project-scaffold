from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def _nanmean(values: List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return float("nan")
    return float(np.mean(valid))


def _nanstd(values: List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return float("nan")
    return float(np.std(valid))


def rmse_path(exec_path: Optional[np.ndarray], ref_path: Optional[np.ndarray]) -> float:
    if exec_path is None or ref_path is None:
        return float("nan")
    length = min(len(exec_path), len(ref_path))
    if length <= 1:
        return float("nan")
    diff = exec_path[:length] - ref_path[:length]
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def path_length(points: Optional[np.ndarray]) -> float:
    if points is None or len(points) < 2:
        return float("nan")
    diffs = np.diff(points, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def smoothness_metrics(points: Optional[np.ndarray], dt: float) -> Tuple[float, float, float]:
    if points is None or len(points) < 3:
        return float("nan"), float("nan"), float("nan")

    velocities = np.diff(points, axis=0) / dt
    if len(velocities) < 2:
        return float("nan"), float("nan"), float("nan")

    accelerations = np.diff(velocities, axis=0) / dt
    accel_mag = np.linalg.norm(accelerations, axis=1)
    rms_accel = float(np.sqrt(np.mean(accel_mag ** 2)))

    if len(accelerations) >= 2:
        jerks = np.diff(accelerations, axis=0) / dt
        jerk_mag = np.linalg.norm(jerks, axis=1)
        avg_jerk = float(np.mean(jerk_mag))
    else:
        avg_jerk = float("nan")

    vel_mag = np.linalg.norm(velocities, axis=1)
    vel_var = float(np.var(vel_mag))
    return rms_accel, avg_jerk, vel_var


def discrete_frechet_distance(path1: np.ndarray, path2: np.ndarray) -> float:
    n, m = len(path1), len(path2)
    dist_matrix = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            dist_matrix[i, j] = np.linalg.norm(path1[i] - path2[j])

    ca = np.full((n, m), np.inf, dtype=np.float64)
    ca[0, 0] = dist_matrix[0, 0]

    for i in range(1, n):
        ca[i, 0] = max(ca[i - 1, 0], dist_matrix[i, 0])
    for j in range(1, m):
        ca[0, j] = max(ca[0, j - 1], dist_matrix[0, j])

    for i in range(1, n):
        for j in range(1, m):
            ca[i, j] = max(min(ca[i - 1, j], ca[i, j - 1], ca[i - 1, j - 1]), dist_matrix[i, j])

    return float(ca[n - 1, m - 1])


def path_efficiency(exec_length: float, ref_length: float) -> float:
    if np.isnan(exec_length) or np.isnan(ref_length) or ref_length == 0:
        return float("nan")
    if exec_length <= 0:
        return 0.0
    return float(min(ref_length / exec_length, 1.0))


def compute_episode_metrics(
    *,
    episode_idx: int,
    reward_sum: float,
    episode_len: int,
    success: float,
    tts: Optional[int],
    min_distance: float,
    exec_path: Optional[np.ndarray],
    ref_path: Optional[np.ndarray],
    final_goal: Optional[np.ndarray],
    reward_components: Dict[str, float],
    meta: Dict[str, Any],
    dt: float = 0.0417,
    collision_count: int = 0,
    min_clearance: Optional[float] = None,
) -> Dict[str, Any]:
    rmse = float("nan")
    max_dev = float("nan")
    end_err = float("nan")
    frechet = float("nan")

    if exec_path is not None and ref_path is not None and len(exec_path) > 1 and len(ref_path) > 1:
        length = min(len(exec_path), len(ref_path))
        rmse = rmse_path(exec_path, ref_path)
        deviations = np.linalg.norm(exec_path[:length] - ref_path[:length], axis=1)
        max_dev = float(np.max(deviations))
        end_err = float(np.linalg.norm(exec_path[-1] - ref_path[-1]))
        frechet = discrete_frechet_distance(exec_path[:length], ref_path[:length])
    elif exec_path is not None and len(exec_path) > 1 and final_goal is not None:
        distances = np.linalg.norm(exec_path - final_goal, axis=1)
        rmse = float(np.mean(distances))
        max_dev = float(np.max(distances))
        end_err = float(distances[-1])

    exec_len = path_length(exec_path)
    ref_len = path_length(ref_path)
    if np.isnan(ref_len) and exec_path is not None and len(exec_path) > 1 and final_goal is not None:
        ref_len = float(np.linalg.norm(exec_path[0] - final_goal))

    efficiency = path_efficiency(exec_len, ref_len)
    rms_accel, avg_jerk, vel_var = smoothness_metrics(exec_path, dt=dt)
    # Path length squared has no units of energy; torque/velocity power is not recorded.
    energy = float("nan")

    record: Dict[str, Any] = {
        "episode_idx": int(episode_idx),
        "reward_sum": float(reward_sum),
        "success": float(success),
        "episode_len": int(episode_len),
        "tts": int(tts) if tts is not None else int(episode_len),
        "min_distance": float(min_distance) if np.isfinite(min_distance) else float("nan"),
        "rmse": rmse,
        "max_deviation": max_dev,
        "final_error": end_err,
        "frechet": frechet,
        "path_len_exec": exec_len,
        "path_len_ref": ref_len,
        "path_efficiency": efficiency,
        "rms_accel": rms_accel,
        "avg_jerk": avg_jerk,
        "vel_var": vel_var,
        "energy": energy,
        "path_length_squared_proxy": float(exec_len ** 2),
        "comp_goal": float(reward_components.get("goal", 0.0)),
        "comp_track": float(reward_components.get("tracking", 0.0)),
        "comp_success": float(reward_components.get("success", 0.0)),
        "comp_smooth": float(reward_components.get("smoothness", reward_components.get("smooth", 0.0))),
        "comp_terminal": float(reward_components.get("terminal", 0.0)),
        "comp_vel": float(reward_components.get("vel", 0.0)),
        "comp_jerk": float(reward_components.get("jerk", 0.0)),
        "collision_count": int(collision_count),
        "collision": float(1.0 if collision_count > 0 else 0.0),
        "min_clearance": float(min_clearance) if min_clearance is not None else float("nan"),
        "detour_ratio": float("nan"),
        "jerk_proxy": avg_jerk,
    }

    record.update(meta)
    return record


def aggregate_episode_metrics(episode_metrics: List[Dict[str, Any]], meta: Dict[str, Any]) -> Dict[str, Any]:
    if not episode_metrics:
        out = {
            "n_episodes": 0,
            "success_rate": float("nan"),
        }
        out.update(meta)
        return out

    def values(key: str) -> List[float]:
        return [float(row.get(key, float("nan"))) for row in episode_metrics]

    summary: Dict[str, Any] = {
        "n_episodes": len(episode_metrics),
        "success_rate": _nanmean(values("success")),
        "reward_mean": _nanmean(values("reward_sum")),
        "reward_std": _nanstd(values("reward_sum")),
        "episode_len_mean": _nanmean(values("episode_len")),
        "episode_len_std": _nanstd(values("episode_len")),
        "tts_mean": _nanmean(values("tts")),
        "tts_std": _nanstd(values("tts")),
        "min_distance_mean": _nanmean(values("min_distance")),
        "min_distance_std": _nanstd(values("min_distance")),
        "rmse_mean": _nanmean(values("rmse")),
        "rmse_std": _nanstd(values("rmse")),
        "max_deviation_mean": _nanmean(values("max_deviation")),
        "max_deviation_std": _nanstd(values("max_deviation")),
        "final_error_mean": _nanmean(values("final_error")),
        "final_error_std": _nanstd(values("final_error")),
        "frechet_mean": _nanmean(values("frechet")),
        "frechet_std": _nanstd(values("frechet")),
        "path_len_exec_mean": _nanmean(values("path_len_exec")),
        "path_len_exec_std": _nanstd(values("path_len_exec")),
        "path_len_ref_mean": _nanmean(values("path_len_ref")),
        "path_len_ref_std": _nanstd(values("path_len_ref")),
        "path_efficiency_mean": _nanmean(values("path_efficiency")),
        "path_efficiency_std": _nanstd(values("path_efficiency")),
        "rms_accel_mean": _nanmean(values("rms_accel")),
        "rms_accel_std": _nanstd(values("rms_accel")),
        "avg_jerk_mean": _nanmean(values("avg_jerk")),
        "avg_jerk_std": _nanstd(values("avg_jerk")),
        "vel_var_mean": _nanmean(values("vel_var")),
        "vel_var_std": _nanstd(values("vel_var")),
        "energy_mean": _nanmean(values("energy")),
        "energy_std": _nanstd(values("energy")),
        "comp_goal_mean": _nanmean(values("comp_goal")),
        "comp_track_mean": _nanmean(values("comp_track")),
        "comp_success_mean": _nanmean(values("comp_success")),
        "comp_smooth_mean": _nanmean(values("comp_smooth")),
        "comp_terminal_mean": _nanmean(values("comp_terminal")),
        "comp_vel_mean": _nanmean(values("comp_vel")),
        "comp_jerk_mean": _nanmean(values("comp_jerk")),
        "collision_rate": _nanmean(values("collision")),
        "collision_count_mean": _nanmean(values("collision_count")),
        "min_clearance_mean": _nanmean(values("min_clearance")),
        "detour_ratio_mean": _nanmean(values("detour_ratio")),
        "jerk_proxy_mean": _nanmean(values("jerk_proxy")),
    }
    summary.update(meta)
    return summary

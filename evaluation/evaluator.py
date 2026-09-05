from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import csv
import json
import numpy as np

from utils.gym_compat import reset_env, step_env
from .metrics import compute_episode_metrics, aggregate_episode_metrics


from training.observation import flatten_obs as _flatten_obs
from training.metrics import trajectory_success


def _select_action(agent, state_vec: np.ndarray, deterministic: bool) -> np.ndarray:
    try:
        return agent.select_action(state_vec, deterministic=deterministic)
    except TypeError:
        return agent.select_action(state_vec)


def _extract_ee_and_goal(obs: Any, info: Optional[Dict]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    ee, goal = None, None
    if isinstance(info, dict):
        if "ee_pos" in info:
            ee = np.asarray(info["ee_pos"], dtype=np.float32)
        if "goal_pos" in info:
            goal = np.asarray(info["goal_pos"], dtype=np.float32)
    if isinstance(obs, dict):
        if ee is None and "achieved_goal" in obs:
            ee = np.asarray(obs["achieved_goal"], dtype=np.float32)
        if goal is None and "desired_goal" in obs:
            goal = np.asarray(obs["desired_goal"], dtype=np.float32)
    return ee, goal


def evaluate_n_episodes(
    agent,
    env,
    n_episodes: int = 20,
    *,
    deterministic: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    metadata = dict(metadata or {})
    metadata.setdefault("deterministic", bool(deterministic))
    if n_episodes <= 0:
        raise ValueError("n_episodes must be positive")
    tracking_mode = getattr(env.unwrapped, "reward_mode", "legacy") == "tracking"
    metadata["evaluation_protocol"] = "full_trajectory_rmse_and_endpoint_v2" if tracking_mode else "legacy_hold_success"

    threshold = float(getattr(env.unwrapped, "distance_threshold", 0.05))
    hold_steps = int(getattr(env.unwrapped, "success_hold_steps", 5))
    if tracking_mode:
        hold_steps = 1

    episodes: List[Dict[str, Any]] = []

    for episode_idx in range(n_episodes):
        episode_seed = None if seed is None else seed + episode_idx
        obs, info = reset_env(env, **({"seed": episode_seed} if episode_seed is not None else {}))
        done = False

        step_idx = 0
        success_streak = 0
        tts: Optional[int] = None
        min_distance = float("inf")
        reward_sum = 0.0
        final_goal = None

        exec_path: List[np.ndarray] = []
        ref_path = None
        reward_comp = {"goal": 0.0, "tracking": 0.0, "success": 0.0, "smooth": 0.0, "vel": 0.0, "jerk": 0.0}

        min_clearance: Optional[float] = None
        collision_count = 0

        ee0, _ = _extract_ee_and_goal(obs, info)
        if ee0 is not None:
            exec_path.append(ee0.copy())

        if isinstance(info, dict) and "ref_traj" in info:
            try:
                ref_path = np.asarray(info["ref_traj"], dtype=np.float32)
            except Exception:
                ref_path = None

        while not done:
            action = _select_action(agent, _flatten_obs(obs), deterministic=deterministic)
            obs, reward, done, info = step_env(env, action)
            reward_sum += float(reward)

            if isinstance(info, dict) and "reward_components" in info:
                rc = info["reward_components"]
                for key, value in rc.items():
                    reward_comp[key] = reward_comp.get(key, 0.0) + float(value)

            if isinstance(info, dict):
                if bool(info.get("collision", False)):
                    collision_count += 1
                if "clearance" in info:
                    clr = float(info["clearance"])
                    min_clearance = clr if min_clearance is None else min(min_clearance, clr)

            ee, goal = _extract_ee_and_goal(obs, info)
            if ee is not None:
                exec_path.append(ee.copy())
            if goal is not None:
                final_goal = goal

            if ee is not None and goal is not None:
                dist = float(np.linalg.norm(ee - goal))
                min_distance = min(min_distance, dist)
                ok = bool(info.get("is_success", False)) if isinstance(info, dict) else False
                if ok or dist < threshold:
                    success_streak += 1
                else:
                    success_streak = 0
                if tts is None and success_streak >= hold_steps:
                    tts = step_idx + 1

            if ref_path is None and isinstance(info, dict) and "ref_traj" in info:
                try:
                    ref_path = np.asarray(info["ref_traj"], dtype=np.float32)
                except Exception:
                    ref_path = None

            step_idx += 1

        episode_meta = dict(metadata)
        episode_meta["episode_seed"] = episode_seed
        successful = tts is not None
        if tracking_mode:
            successful = trajectory_success(np.asarray(exec_path), ref_path, threshold,
                                             env.unwrapped.tracking_success_threshold)

        record = compute_episode_metrics(
            episode_idx=episode_idx,
            reward_sum=reward_sum,
            episode_len=step_idx,
            success=float(successful),
            tts=tts,
            min_distance=min_distance,
            exec_path=np.asarray(exec_path, dtype=np.float32) if exec_path else None,
            ref_path=ref_path,
            final_goal=final_goal,
            reward_components=reward_comp,
            meta=episode_meta,
            collision_count=collision_count,
            min_clearance=min_clearance,
            dt=getattr(env.unwrapped, "dt", 1 / 240) * getattr(env.unwrapped, "sim_steps_per_action", 10),
        )
        episodes.append(record)

    summary = aggregate_episode_metrics(episodes, meta=metadata)
    return summary, episodes


def evaluate_policy(
    agent,
    env,
    eval_episodes: int = 20,
    *,
    deterministic: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    return evaluate_n_episodes(
        agent=agent,
        env=env,
        n_episodes=eval_episodes,
        deterministic=deterministic,
        metadata=metadata,
        seed=seed,
    )


def evaluate_one_checkpoint(
    agent,
    env,
    checkpoint_path: str,
    *,
    eval_episodes: int = 20,
    deterministic: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    import torch

    state_dict = torch.load(checkpoint_path, map_location=getattr(agent, "device", "cpu"), weights_only=True)
    if isinstance(state_dict, dict) and "actor" in state_dict:
        agent.actor.load_state_dict(state_dict["actor"])
    elif isinstance(state_dict, dict):
        agent.actor.load_state_dict(state_dict)
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    return evaluate_policy(
        agent=agent,
        env=env,
        eval_episodes=eval_episodes,
        deterministic=deterministic,
        metadata=metadata,
    )


def save_eval_outputs(
    output_dir: str,
    summary: Dict[str, Any],
    episodes: List[Dict[str, Any]],
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    episode_jsonl = out_dir / "episode_metrics.jsonl"
    with episode_jsonl.open("a", encoding="utf-8") as fp:
        for row in episodes:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_json = out_dir / "eval_summary.json"
    with summary_json.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    history_csv = out_dir / "eval_history.csv"
    write_header = not history_csv.exists()
    with history_csv.open("a", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(summary.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(summary)

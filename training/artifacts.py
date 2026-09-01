"""Experiment directory, checkpoint and summary persistence."""

import csv
import json
import os
from datetime import datetime
from typing import Any, Dict, Iterable, List

import numpy as np


def create_run_dir(config: Any) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_suffix = "" if config.seed is None else f"_seed{config.seed}"
    name = f"{config.env}_{config.actor_arch}_dense_{stamp}{seed_suffix}"
    run_dir = os.path.abspath(os.path.join(config.save_dir, name))
    os.makedirs(run_dir, exist_ok=False)
    config.save_dir = run_dir
    with open(os.path.join(os.path.dirname(run_dir), "LATEST_RUN.txt"), "w", encoding="utf-8") as fh:
        fh.write(run_dir)
    save_config(config)
    return run_dir


def save_config(config: Any) -> None:
    with open(os.path.join(config.save_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(vars(config), fh, indent=2, ensure_ascii=False)


def _write_episode_csv(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_progress(
    config: Any,
    agent: Any,
    metrics: Dict[str, List],
    rewards: List[float],
    successes: List[int],
    episodes: List[Dict[str, Any]],
    save_model: bool = False,
) -> None:
    os.makedirs(config.save_dir, exist_ok=True)
    np.save(os.path.join(config.save_dir, "rewards.npy"), np.asarray(rewards, dtype=np.float32))
    np.save(os.path.join(config.save_dir, "success.npy"), np.asarray(successes, dtype=np.int8))
    with open(os.path.join(config.save_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False)
    _write_episode_csv(os.path.join(config.save_dir, "episodes.csv"), episodes)
    save_config(config)
    if save_model:
        agent.save(os.path.join(config.save_dir, "final_model"))


def append_summary(config: Any, metrics: Dict[str, List], rewards: List[float], successes: List[int]) -> str:
    """Write a stable v2 schema instead of appending to the malformed legacy CSV."""
    path = os.path.join(os.path.dirname(config.save_dir), "training_summary_v2.csv")
    last = lambda key: metrics.get(key, [float("nan")])[-1] if metrics.get(key) else float("nan")
    record = {
        "run_name": os.path.basename(config.save_dir),
        "actor_arch": config.actor_arch,
        "seed": config.seed,
        "timesteps": config.max_timesteps,
        "episodes": len(rewards),
        "train_reward_mean": float(np.mean(rewards[-20:])) if rewards else float("nan"),
        "train_success_mean": float(np.mean(successes[-20:])) if successes else float("nan"),
        "eval_reward": last("eval_rewards"),
        "eval_success": last("eval_success"),
        "eval_tts_steps": last("eval_tts"),
        "eval_min_distance": last("eval_min_distance"),
        "eval_rmse": last("eval_rmse"),
        "eval_endpoint_error": last("eval_end_err"),
    }
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(record))
        if not exists:
            writer.writeheader()
        writer.writerow(record)
    return path

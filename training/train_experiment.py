"""Canonical TD3 training entry point.

All experiment configuration, observation handling, evaluation and persistence live
in dedicated modules. Run with:

    python -m training.train_experiment --actor_arch gnn_transformer
"""

import os
import random
import sys
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - legacy fallback
    import gym
import numpy as np
import torch

import envs  # noqa: F401 - register custom environment
from agents.td3_agent import TD3
from training.artifacts import append_summary, create_run_dir, save_progress
from training.config import get_default_args, parse_args
from training.evaluator import evaluate_policy
from training.metrics import extract_distance, is_success_state
from training.observation import flatten_obs, infer_dimensions
from utils.gym_compat import reset_env, step_env
from utils.replay_buffer import ReplayBuffer


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_env(config: Any, evaluation: bool = False):
    kwargs = {
        "render_mode": "rgb_array",
        "dense_reward": True,
        "observation_version": config.observation_version,
    }
    env = gym.make(config.env, **kwargs)
    seed = config.eval_seed if evaluation else config.seed
    if seed is not None:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env


def new_metrics() -> Dict[str, List]:
    return {
        "actor_loss": [],
        "actor_loss_t": [],
        "critic_loss": [],
        "critic_loss_t": [],
        "eval_steps": [],
        "eval_rewards": [],
        "eval_success": [],
        "eval_tts": [],
        "eval_min_distance": [],
        "eval_rmse": [],
        "eval_max_dev": [],
        "eval_end_err": [],
        "eval_path_len_exec": [],
        "eval_path_len_ref": [],
    }


def run_evaluation(agent: TD3, env, config: Any, step: int, metrics: Dict[str, List]) -> Tuple:
    result = evaluate_policy(
        agent,
        env,
        eval_episodes=config.eval_episodes,
        seed=config.eval_seed,
    )
    keys = (
        "eval_rewards", "eval_success", "eval_tts", "eval_min_distance",
        "eval_rmse", "eval_max_dev", "eval_end_err",
        "eval_path_len_exec", "eval_path_len_ref",
    )
    metrics["eval_steps"].append(int(step))
    for key, value in zip(keys, result):
        metrics[key].append(float(value))
    print(
        f"[eval step={step:,}] reward={result[0]:.3f} "
        f"success={result[1]:.1%} tts={result[2]:.1f} steps "
        f"min_distance={result[3]:.4f}m rmse={result[4]:.4f}m"
    )
    return result


def run_experiment(config: Any) -> str:
    if config.seed is not None:
        set_global_seed(config.seed)
    run_dir = create_run_dir(config)
    print(f"[run] {run_dir}")

    env = make_env(config, evaluation=False)
    eval_env = make_env(config, evaluation=True)
    state_dim, action_dim, max_action = infer_dimensions(env)
    print(
        f"[config] arch={config.actor_arch} state={state_dim} action={action_dim} "
        f"node_dim={config.node_dim} seed={config.seed}"
    )

    agent = TD3(state_dim, action_dim, max_action, config)
    replay_buffer = ReplayBuffer(
        env.observation_space,
        env.action_space,
        capacity=config.buffer_size,
        her_prob=0.0,
        her_k=0,
        dense_reward=True,
        distance_threshold=float(env.unwrapped.distance_threshold),
        use_per=config.use_per,
        beta_frames=max(1, config.max_timesteps - config.start_timesteps),
    )

    metrics = new_metrics()
    rewards: List[float] = []
    successes: List[int] = []
    episodes: List[Dict[str, Any]] = []
    obs, _ = reset_env(env, seed=config.seed)
    episode_reward = 0.0
    episode_steps = 0
    episode_success = False
    episode_index = 0
    completed_steps = 0
    interrupted = False

    try:
        run_evaluation(agent, eval_env, config, 0, metrics)
        for t in range(config.max_timesteps):
            completed_steps = t + 1
            episode_steps += 1

            if t < config.start_timesteps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(flatten_obs(obs), deterministic=True)
                if config.expl_noise > 0:
                    action = action + np.random.normal(0.0, config.expl_noise, size=action_dim)
                action = np.clip(action, env.action_space.low, env.action_space.high)

            next_obs, reward, done, info = step_env(env, action)
            replay_buffer.add(obs, action, next_obs, reward, done)
            obs = next_obs
            episode_reward += float(reward)
            episode_success = episode_success or is_success_state(obs, info, env)

            if t >= config.start_timesteps and replay_buffer.size >= config.batch_size:
                losses = agent.train(replay_buffer, config.batch_size)
                if isinstance(losses, dict):
                    metrics["actor_loss"].append(float(losses["actor_loss"]))
                    metrics["actor_loss_t"].append(completed_steps)
                    metrics["critic_loss"].append(float(losses["critic_loss"]))
                    metrics["critic_loss_t"].append(completed_steps)
                else:
                    metrics["critic_loss"].append(float(losses))
                    metrics["critic_loss_t"].append(completed_steps)

            if done:
                distance = extract_distance(obs, info, env)
                rewards.append(episode_reward)
                successes.append(int(episode_success))
                episodes.append({
                    "episode": episode_index,
                    "global_step": completed_steps,
                    "steps": episode_steps,
                    "reward": episode_reward,
                    "success": int(episode_success),
                    "final_distance": distance,
                })
                episode_index += 1
                obs, _ = reset_env(env)
                episode_reward = 0.0
                episode_steps = 0
                episode_success = False

            if completed_steps % config.eval_freq == 0:
                run_evaluation(agent, eval_env, config, completed_steps, metrics)
                save_progress(config, agent, metrics, rewards, successes, episodes)
                agent.save(os.path.join(config.save_dir, "checkpoint_latest"))

    except KeyboardInterrupt:
        interrupted = True
        print("[interrupt] saving current progress")
    finally:
        config.completed_timesteps = completed_steps
        config.interrupted = interrupted
        save_progress(config, agent, metrics, rewards, successes, episodes, save_model=True)
        append_summary(config, metrics, rewards, successes)
        env.close()
        eval_env.close()

    print(f"[done] steps={completed_steps:,} episodes={len(rewards)} run={run_dir}")
    return run_dir


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()

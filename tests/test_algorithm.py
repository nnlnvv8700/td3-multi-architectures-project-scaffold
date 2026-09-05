"""Algorithm-level invariants and paired control tests, independent of convergence."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pybullet as p
import torch

from agents.td3_agent import TD3
from envs.kuka_iiwa_env import KukaIiwa7TrackEnv
from envs.rewards import TrackingReward
from training.config import parse_args
from training.evaluator import PolicyEvaluator
from training.train_experiment import run_experiment
from utils.control import damped_least_squares
from utils.gym_compat import step_env
from utils.replay_buffer import ReplayBuffer

ROOT = Path(__file__).resolve().parents[1]


def corrected_config(output, arch="mlp"):
    return parse_args(
        [
            "--config",
            str(ROOT / "configs/algorithm_v2.yaml"),
            "--actor_arch",
            arch,
            "--device",
            "cpu",
            "--torch_threads",
            "2",
            "--max_timesteps",
            "6",
            "--start_timesteps",
            "0",
            "--batch_size",
            "2",
            "--buffer_size",
            "16",
            "--env_max_steps",
            "3",
            "--eval_episodes",
            "1",
            "--eval_freq",
            "6",
            "--save_dir",
            str(output),
        ]
    )


class AlgorithmTests(unittest.TestCase):
    def test_corrected_loss_is_same_with_unit_per_weights_and_unclipped_targets(self):
        class FixedReplay:
            def __init__(self, use_per):
                self.use_per = use_per
                self.errors = None

            def sample(self, batch_size):
                batch = (
                    np.zeros((2, 31)),
                    np.zeros((2, 7)),
                    np.zeros((2, 31)),
                    np.full((2, 1), 1000.0),
                    np.zeros((2, 1)),
                )
                return (batch, np.array([0, 1]), np.ones(2)) if self.use_per else batch

            def update_priorities(self, indices, errors):
                self.errors = errors

        for use_per in (False, True):
            agent = TD3(31, 7, 1.5, corrected_config("unused"))
            with torch.no_grad():
                for critic in (agent.critic1, agent.critic2):
                    for parameter in critic.parameters():
                        parameter.zero_()
            buffer = FixedReplay(use_per)
            loss = agent.train(buffer, 2)
            self.assertAlmostEqual(loss, 1999.0, places=4)
            self.assertEqual(agent.last_diagnostics["target_q_mean"], 1000.0)
            if use_per:
                np.testing.assert_array_equal(buffer.errors, [1000.0, 1000.0])

    def test_tracking_evaluators_agree_on_success_rmse_and_control_period(self):
        from evaluation.evaluator import evaluate_n_episodes

        class Zero:
            def select_action(self, state, deterministic=True):
                return np.zeros(7)

        env = KukaIiwa7TrackEnv(
            render_mode="rgb_array",
            reward_mode="tracking",
            control_mode="residual",
            max_steps=30,
            sim_steps_per_action=5,
        )
        try:
            core = PolicyEvaluator(env, Zero(), distance_threshold=0.1).evaluate_episode(seed=123)
            _, extended = evaluate_n_episodes(Zero(), env, n_episodes=1, seed=123)
            self.assertEqual(bool(extended[0]["success"]), core["success"])
            self.assertAlmostEqual(extended[0]["rmse"], core["rmse"], places=5)
            self.assertAlmostEqual(extended[0]["avg_jerk"], core["jerk"], delta=0.001)
            self.assertEqual(extended[0]["episode_seed"], 123)
            self.assertTrue(np.isnan(extended[0]["energy"]))
        finally:
            env.close()

    def test_reward_prefers_on_time_tracking_to_early_arrival(self):
        reward = TrackingReward()
        reference, goal = np.zeros(3), np.array([0.5, 0, 0])
        on_time, terms = reward(reference, reference, goal, np.zeros(7), np.zeros(7), 1.5)
        early, _ = reward(goal, reference, goal, np.zeros(7), np.zeros(7), 1.5)
        self.assertGreater(on_time, early)
        self.assertEqual(terms["terminal"], 0)
        self.assertAlmostEqual(on_time, sum(terms.values()))
        late, terms = reward(
            reference, reference, goal, np.zeros(7), np.zeros(7), 1.5, terminal=True
        )
        self.assertLess(late, on_time)
        self.assertLess(terms["terminal"], 0)

    def test_smoothness_cost_is_invariant_to_joint_count_and_action_scale(self):
        reward = TrackingReward()
        results = []
        for joints, limit in [(7, 1.5), (14, 3.0)]:
            _, components = reward(
                np.zeros(3),
                np.zeros(3),
                np.zeros(3),
                np.full(joints, 0.1 * limit),
                np.zeros(joints),
                limit,
            )
            results.append(components["smoothness"])
        self.assertAlmostEqual(*results)

    def test_finite_horizon_and_external_truncation_masks(self):
        env = KukaIiwa7TrackEnv(render_mode="rgb_array", max_steps=1, reward_mode="tracking")
        try:
            obs, _ = env.reset(seed=42)
            obs2, reward, done, info = step_env(env, np.zeros(7))
            self.assertTrue(done and info["terminated"])
            self.assertFalse(info["truncated"])
            buffer = ReplayBuffer(
                env.observation_space, env.action_space, capacity=8, her_prob=0, use_per=False
            )
            buffer.add(obs, np.zeros(7), obs2, reward, done, terminal=False)
            self.assertEqual(buffer.sample(1)[4][0, 0], 1)  # external cutoff bootstraps
            buffer.add(obs, np.zeros(7), obs2, reward, done, terminal=info["terminated"])
            self.assertEqual(buffer.done[1], 1)  # true trajectory endpoint does not
            self.assertEqual(buffer.ep_end[0], 0)  # both cases end replay episodes
        finally:
            env.close()

    def test_proportional_per_weights_and_duplicate_updates(self):
        env = KukaIiwa7TrackEnv(render_mode="rgb_array")
        try:
            obs, _ = env.reset(seed=42)
            buffer = ReplayBuffer(
                env.observation_space,
                env.action_space,
                capacity=8,
                her_prob=0,
                per_mode="proportional",
                alpha=1,
                beta_start=1,
            )
            for _ in range(2):
                buffer.add(obs, np.zeros(7), obs, 0, False)
            buffer.update_priorities([0, 1], [1, 4])
            np.random.seed(7)
            batch = buffer.sample_batch(10000)
            self.assertAlmostEqual(np.mean(batch["indices"] == 1), 0.8, delta=0.02)
            expected = buffer.priorities.min() / buffer.priorities[batch["indices"]]
            np.testing.assert_allclose(batch["weights"], expected, rtol=1e-6)
            buffer.update_priorities([1, 1], [5, 2])
            self.assertAlmostEqual(buffer.priorities[1], 5 + 1e-6, places=5)
        finally:
            env.close()

    def test_dls_is_finite_at_singular_configuration(self):
        command = damped_least_squares(np.zeros((3, 7)), np.ones(3))
        np.testing.assert_array_equal(command, np.zeros(7))

    def test_bullet_jacobian_matches_observed_endpoint(self):
        env = KukaIiwa7TrackEnv(render_mode="rgb_array")
        try:
            env.reset(seed=7)
            q = np.array([0.1, 0.4, 0.2, -0.5, 0.1, 0.3, 0.0])

            def position(values):
                for j, value in enumerate(values):
                    p.resetJointState(env.robot_id, j, float(value), physicsClientId=env._p_client)
                return env._eef_pos().astype(float)

            position(q)
            analytic, _ = p.calculateJacobian(
                env.robot_id,
                env.ee_link,
                [0, 0, 0],
                q.tolist(),
                [0] * 7,
                [0] * 7,
                physicsClientId=env._p_client,
            )
            numerical = np.zeros((3, 7))
            for j in range(7):
                shift = np.eye(7)[j] * 0.001
                numerical[:, j] = (position(q + shift) - position(q - shift)) / 0.002
            np.testing.assert_allclose(analytic, numerical, atol=1e-4)
        finally:
            env.close()

    def test_zero_residual_equals_dls_and_keeps_applied_action_in_state(self):
        env = KukaIiwa7TrackEnv(
            render_mode="rgb_array", reward_mode="tracking", control_mode="residual"
        )
        try:
            env.reset(seed=42)
            expected = env.baseline_action()
            obs, _, _, _, info = env.step(np.zeros(7))
            np.testing.assert_allclose(info["applied_action"], expected)
            np.testing.assert_allclose(obs["observation"][18:25], expected, rtol=1e-6)
            np.testing.assert_array_equal(info["policy_action"], np.zeros(7))
        finally:
            env.close()

    def test_all_corrected_architectures_train_and_restore_semantics(self):
        with tempfile.TemporaryDirectory() as output:
            for arch in ("mlp", "gnn", "transformer", "gnn_transformer"):
                with self.subTest(architecture=arch):
                    config = corrected_config(output, arch)
                    run = Path(run_experiment(config))
                    self.assertTrue((run / "best_model.pt").is_file())
                    self.assertTrue((run / "final_model.meta.json").is_file())
                    metrics = json.loads((run / "metrics.json").read_text())
                    self.assertTrue(np.isfinite(metrics["critic_loss"]).all())
                    agent = TD3(31, 7, 1.5, config)
                    agent.load_actor(run / "final_model.pt")
                    self.assertTrue(np.isfinite(agent.select_action(np.zeros(31))).all())
                    # Check the actual state normalization used for training and inference.
                    scaled = agent._state_tensor(np.ones(31))
                    self.assertAlmostEqual(float(scaled[0]), 1 / np.pi)
                    self.assertAlmostEqual(float(scaled[18]), 1 / 1.5)
                    wrong = corrected_config(output, arch)
                    wrong.algorithm_version = "legacy"
                    with self.assertRaisesRegex(ValueError, "semantics"):
                        TD3(31, 7, 1.5, wrong).load_actor(run / "final_model.pt")
                    if arch == "mlp":
                        from training.evaluate_enhanced import load_agent_from_run
                        from training.test_agent import find_weight_file

                        self.assertEqual(Path(find_weight_file(run)), run / "best_model.pt")
                        agent.load_actor(run / "best_model.pt")
                        loaded, env, _ = load_agent_from_run(str(run))
                        try:
                            self.assertEqual(env.unwrapped.control_mode, "residual")
                            np.testing.assert_array_equal(
                                loaded.select_action(np.ones(31)), agent.select_action(np.ones(31))
                            )
                        finally:
                            env.close()

    def test_corrected_dropout_does_not_change_policy_between_modes(self):
        config = corrected_config("unused", "gnn_transformer")
        config.zero_init_actor = False
        agent = TD3(31, 7, 1.5, config)
        state = np.ones(31, dtype=np.float32)
        np.testing.assert_allclose(
            agent.select_action(state, False), agent.select_action(state, True), atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()

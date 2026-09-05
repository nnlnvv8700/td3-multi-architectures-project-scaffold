import unittest

import numpy as np

from envs.kuka_iiwa_env import KukaIiwa7TrackEnv
from utils.replay_buffer import ReplayBuffer


class ReplayBufferTests(unittest.TestCase):
    def test_invalid_inputs_do_not_corrupt_buffer(self):
        env = KukaIiwa7TrackEnv(render_mode="rgb_array")
        try:
            obs, _ = env.reset(seed=7)
            buffer = ReplayBuffer(env.observation_space, env.action_space,
                                  capacity=8, her_prob=0, use_per=True)
            with self.assertRaisesRegex(ValueError, "empty"):
                buffer.sample_batch(1)
            with self.assertRaises(ValueError):
                buffer.add(obs, np.zeros(1), obs, 1.0, False)
            self.assertEqual(buffer.size, 0)
            buffer.add(obs, np.zeros(7), obs, 1.0, False)
            with self.assertRaisesRegex(ValueError, "without replacement"):
                buffer.sample_batch(2)
            with self.assertRaisesRegex(ValueError, "finite"):
                buffer.update_priorities([0], [np.nan])
            self.assertEqual(buffer.priorities[0], 1.0)
        finally:
            env.close()

    def test_dense_reward_and_done_are_preserved_without_her(self):
        env = KukaIiwa7TrackEnv(render_mode="rgb_array", max_steps=1, dense_reward=True)
        try:
            obs, _ = env.reset(seed=7)
            action = np.zeros(7, dtype=np.float32)
            obs2, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer = ReplayBuffer(
                env.observation_space,
                env.action_space,
                capacity=8,
                her_prob=0.0,
                dense_reward=True,
                use_per=False,
            )
            buffer.add(obs, action, obs2, reward, done)
            batch = buffer.sample_batch(1)
            self.assertAlmostEqual(float(batch["rews"][0, 0]), float(reward), places=6)
            self.assertEqual(float(batch["done"][0, 0]), 1.0)
            self.assertEqual(batch["obs"].shape[1], 31)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()

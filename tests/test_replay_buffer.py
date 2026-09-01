import unittest

import numpy as np

from envs.kuka_iiwa_env import KukaIiwa7TrackEnv
from utils.replay_buffer import ReplayBuffer


class ReplayBufferTests(unittest.TestCase):
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

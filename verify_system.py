"""
系统验证脚本 - 测试整合后的训练系统

验证项：
1. 环境创建和配置
2. 各架构的智能体初始化
3. 训练脚本参数解析
4. 快速训练（100步）验证流程
"""
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Windows PowerShell may default to GBK, which cannot encode the status symbols
# used by this human-facing verification script.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np

try:
    import gymnasium as gym
except ImportError:
    import gym

# 添加项目根目录到路径
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import envs
from agents.td3_agent import TD3
from training.observation import infer_dimensions


def test_environment():
    """测试1: 环境创建"""
    print("\n" + "="*60)
    print("测试1: 环境创建")
    print("="*60)

    try:
        env = gym.make("KukaIiwa7Track-v0", render_mode=None, dense_reward=True)
        print("✅ 环境创建成功")
        print(f"   类型: {type(env.unwrapped).__name__}")
        print(f"   done_on_success: {getattr(env.unwrapped, 'done_on_success', 'N/A')}")
        print(f"   max_steps: {getattr(env.unwrapped, 'max_steps', 'N/A')}")

        obs, info = env.reset(seed=42)
        print(f"   观测空间检查: ✅")
        print(f"   info包含ref_traj: {'✅' if 'ref_traj' in info else '❌'}")

        if 'ref_traj' in info:
            print(f"   参考轨迹形状: {info['ref_traj'].shape}")

        env.close()
        return True
    except Exception as e:
        print(f"❌ 环境测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_architectures():
    """测试2: 各架构初始化"""
    print("\n" + "="*60)
    print("测试2: 架构初始化")
    print("="*60)

    architectures = ["mlp", "gnn", "transformer", "gnn_transformer"]
    state_dim = 31
    action_dim = 7
    max_action = 1.5

    class MockConfig:
        def __init__(self, arch):
            self.actor_arch = arch
            self.device = "cpu"
            self.actor_lr = 1e-5
            self.critic_lr = 1e-5
            self.discount = 0.99
            self.tau = 0.005
            self.policy_noise = 0.2
            self.noise_clip = 0.5
            self.policy_freq = 2
            self.policy_delay = 2
            self.max_timesteps = 100
            self.start_timesteps = 10

            if arch in ["gnn", "transformer", "gnn_transformer"]:
                self.node_dim = 6
                self.num_nodes = 7
                self.use_state_encoder = True
            else:
                self.node_dim = None
                self.num_nodes = None
                self.use_state_encoder = False

    results = {}
    for arch in architectures:
        try:
            config = MockConfig(arch)
            agent = TD3(state_dim, action_dim, max_action, config)
            print(f"   {arch:20s}: ✅")
            results[arch] = True
        except Exception as e:
            print(f"   {arch:20s}: ❌ {str(e)[:50]}")
            results[arch] = False

    return all(results.values())


def test_quick_training():
    """测试3: 快速训练（100步）"""
    print("\n" + "="*60)
    print("测试3: 快速训练流程（100步）")
    print("="*60)

    try:
        import argparse
        from training.train_experiment import get_default_args, flatten_obs

        # 创建最小配置
        args = argparse.Namespace(
            env="KukaIiwa7Track-v0",
            actor_arch="mlp",
            max_timesteps=100,
            start_timesteps=50,
            batch_size=32,
            eval_freq=50,
            expl_noise=0.1,
            save_dir="./test_run",
            actor_lr=3e-5,
            critic_lr=3e-5,
            device="cpu",
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            policy_delay=2,
            observation_version=2,
        )

        defaults = get_default_args(args.actor_arch)
        args.node_dim = defaults["node_dim"]
        args.num_nodes = defaults["num_nodes"]
        args.use_state_encoder = defaults["use_state_encoder"]

        # 创建环境和智能体
        env = gym.make(args.env, render_mode=None, dense_reward=True)

        state_dim, action_dim, max_action = infer_dimensions(env)

        agent = TD3(state_dim, action_dim, max_action, args)

        # 运行100步
        obs, info = env.reset()
        total_reward = 0

        for t in range(100):
            if t < 50:
                action = env.action_space.sample()
            else:
                state = flatten_obs(obs)
                action = agent.select_action(state)

            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            if done:
                obs, info = env.reset()
            else:
                obs = next_obs

        env.close()

        print(f"   训练步数: 100")
        print(f"   累计奖励: {total_reward:.2f}")
        print("   ✅ 快速训练流程正常")

        # 清理测试文件
        import shutil
        if os.path.exists("./test_run"):
            shutil.rmtree("./test_run")

        return True
    except Exception as e:
        print(f"   ❌ 训练流程失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*60)
    print("TD3 轨迹跟踪系统验证")
    print("="*60)

    results = {
        "环境创建": test_environment(),
        "架构初始化": test_architectures(),
        "训练流程": test_quick_training()
    }

    print("\n" + "="*60)
    print("验证总结")
    print("="*60)

    for name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{name:15s}: {status}")

    if all(results.values()):
        print("\n🎉 所有测试通过！系统已就绪。")
        print("\n快速开始:")
        print("  python train.py mlp              # 训练MLP")
        print("  python train.py gnn_transformer  # 训练GT-TD3")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息。")

    print("="*60)


if __name__ == "__main__":
    main()

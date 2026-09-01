# GT-TD3：KUKA iiwa 多架构轨迹跟踪

> A reproducible TD3 research scaffold for KUKA LBR iiwa trajectory tracking in PyBullet.

本项目在 PyBullet 中构建 KUKA LBR iiwa 七自由度机械臂轨迹跟踪任务，从头实现
Twin Delayed Deep Deterministic Policy Gradient（TD3），并在统一 Critic、训练流程和
评估协议下比较 MLP、GNN、Transformer 与 GNN+Transformer 四种 Actor。

项目重点不是宣称某个架构达到 SOTA，而是提供一套结构清晰、可复现、适合继续做消融
实验的机器人强化学习框架。

## 项目亮点

- 自定义 Gymnasium/PyBullet 七自由度连续控制环境。
- 完整 TD3：双 Critic、Clipped Double-Q、目标策略平滑、延迟策略更新和软目标更新。
- 四种可替换 Actor，共用相同 Critic 和训练入口。
- KUKA 七关节链式图拓扑与关节位置编码。
- 轨迹阶段、当前参考点和前一动作显式进入状态，避免隐藏状态破坏 Markov 性。
- 固定评估目标集、完整随机种子控制、PER、模型与实验配置自动归档。
- 成功率、TTS、最小距离、轨迹 RMSE、最大偏差、终点误差和路径长度等指标。
- PyBullet 多 client 隔离及 9 项单元/端到端回归测试。

## 系统结构

```text
KukaIiwa7TrackEnv
  │  31-D state / 7-D joint velocity action
  ▼
Actor π(s) ───────────────┐
  ├─ MLP                  │ action
  ├─ GNN                  ▼
  ├─ Transformer       PyBullet
  └─ GNN+Transformer      │ transition
                          ▼
                    ReplayBuffer (PER)
                          │ batch
                          ▼
                Twin Critics Q1 / Q2
                          │
          clipped target + delayed actor update
```

训练调用链：

```text
training.config
      ↓
training.train_experiment
      ├── envs.kuka_iiwa_env
      ├── agents.td3_agent → agents.networks
      ├── utils.replay_buffer
      ├── training.evaluator
      └── training.artifacts → results/<run>/
```

## 任务定义

| 项目 | 设置 |
|---|---|
| 环境 ID | `KukaIiwa7Track-v0` |
| 机器人 | KUKA LBR iiwa，7 DoF |
| 动作 | 7 维关节速度，范围 `[-1.5, 1.5]` |
| 仿真频率 | 240 Hz |
| 控制周期 | 每次动作执行 10 个仿真步，约 24 Hz |
| 回合长度 | 200 个控制周期，约 8.33 秒 |
| 参考轨迹 | 初始末端位置到目标位置的笛卡尔直线，共 201 个对齐点 |
| 成功阈值 | 末端与目标距离 `< 0.10 m` |
| 终止策略 | 默认成功后不提前结束，以评估完整轨迹 |

### v2 观测

GoalEnv 字典观测展平后共 31 维：

```text
observation (25)
  = joint_position (7)
  + joint_velocity (7)
  + trajectory_phase (1)
  + current_reference_point (3)
  + previous_action (7)

achieved_goal (3) + desired_goal (3)
```

图模型将状态编码为 7 个关节节点，每个节点 6 维：

```text
[q_i, qdot_i, previous_action_i, goal_dx, goal_dy, goal_dz]
```

轨迹阶段与 `current_reference - achieved_goal` 作为 4 维全局上下文接入 Actor readout，
因此图模型不会丢失当前参考轨迹信息。

## 四种 Actor

| Actor | 状态处理 | 结构先验 |
|---|---|---|
| MLP | 直接输入 31 维状态 | 无显式机器人拓扑 |
| GNN | 7 个关节节点上的多头图注意力 | KUKA 链式邻接矩阵 |
| Transformer | 关节 token + Transformer Encoder | 关节序号/距离位置编码 |
| GNN+Transformer | 图注意力后接 Transformer | 局部拓扑与全局依赖融合 |

当前四种 Actor 参数量并不完全一致，因此实验结果应解释为“不同表示方案的系统比较”，
不能仅凭单次实验将差异完全归因于架构归纳偏置。严谨消融应进一步做参数量匹配。

## 安装

推荐 Python 3.10+。建议使用独立虚拟环境：

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux/macOS：

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

若需要 CUDA，请根据本机 CUDA 版本先从 PyTorch 官方渠道安装对应 PyTorch，再安装其余依赖。

## 快速开始

### 运行回归测试

```bash
python -m unittest discover -s tests -v
python verify_system.py
```

### 训练

唯一规范训练入口：

```bash
python -m training.train_experiment --actor_arch mlp
python -m training.train_experiment --actor_arch gnn
python -m training.train_experiment --actor_arch transformer
python -m training.train_experiment --actor_arch gnn_transformer
```

推荐显式设置训练种子和固定评估种子：

```bash
python -m training.train_experiment \
  --actor_arch gnn_transformer \
  --seed 42 \
  --eval_seed 10000 \
  --max_timesteps 500000
```

Windows PowerShell 可写成单行：

```powershell
python -m training.train_experiment --actor_arch gnn_transformer --seed 42 --eval_seed 10000 --max_timesteps 500000
```

查看全部参数：

```bash
python -m training.train_experiment --help
```

`train.py` 只是兼容包装器，实际训练逻辑仅在
`training.train_experiment` 中维护。

### 测试模型

```bash
python -m training.test_agent \
  --run_dir results/<run_directory> \
  --episodes 30 \
  --seed 10000
```

默认使用无 GUI 的 `rgb_array` 模式；观察仿真时添加：

```bash
--render_mode human
```

### 绘制多架构对比图

```bash
python -m training.evaluate --run_dirs \
  results/<mlp_run> \
  results/<gnn_run> \
  results/<transformer_run> \
  results/<fusion_run>
```

训练记录显式保存 `eval_steps`。绘图不会再用评估序号冒充训练步数，也不会在没有多种子
统计的情况下构造伪不确定性区间。

## 实验输出

每次训练自动创建独立目录：

```text
results/<env>_<arch>_dense_<timestamp>_seed<seed>/
├── config.json
├── final_model.pt
├── checkpoint_latest.pt
├── metrics.json
├── episodes.csv
├── rewards.npy
└── success.npy
```

跨实验稳定表头汇总写入：

```text
results/training_summary_v2.csv
```

历史 `training_summary.csv` 存在字段演化导致的列错位，仅供追溯，不应继续用于论文统计。

## 目录结构

```text
.
├── agents/
│   ├── td3_agent.py          # TD3 更新逻辑与模型保存
│   ├── networks.py           # 四种 Actor 与共享 Critic
│   └── state_encoder.py      # 关节节点编码和图结构
├── envs/
│   └── kuka_iiwa_env.py      # PyBullet 轨迹跟踪环境
├── training/
│   ├── train_experiment.py   # 唯一规范训练循环
│   ├── config.py             # 参数与架构默认值
│   ├── observation.py        # 统一观测处理
│   ├── evaluator.py          # 固定种子策略评估
│   ├── artifacts.py          # checkpoint、指标和汇总
│   ├── test_agent.py         # 模型测试与轨迹导出
│   └── evaluate.py           # 多架构结果绘图
├── utils/
│   ├── replay_buffer.py      # PER 与兼容 HER 的经验池
│   └── gym_compat.py         # Gym/Gymnasium API 兼容层
├── tests/                    # 单元和端到端回归测试
├── requirements.txt
└── README.md
```

## 正确性修复

当前版本相对历史代码重点修复了：

1. 所有 PyBullet 调用显式传递 `physicsClientId`，训练和评估仿真完全隔离。
2. 未做 HER 重标记时，ReplayBuffer 保留环境真实组合奖励与终止标记。
3. 将轨迹阶段、当前参考点和前一动作加入状态，补足奖励所依赖的状态变量。
4. 图节点保留完整三维目标方向，并使用 KUKA 链式邻接矩阵。
5. Actor 和目标动作使用环境真实动作上限。
6. 确定性评估关闭 Dropout，目标 Actor 固定为 evaluation mode。
7. 环境、动作空间、PyTorch、NumPy 与固定评估目标统一种子管理。
8. 修复余弦学习率调度越过最低点后重新升高的问题。
9. 将原 1502 行训练文件拆分为单一职责模块，并加入端到端测试。

## 复现实验建议

用于论文或面试展示时，建议遵循：

- 每种架构使用完全相同的训练种子，例如 `42/123/456/789/2025`。
- 使用相同 `eval_seed`，保证不同模型面对同一批测试目标。
- 至少运行 3–5 个随机种子，报告均值与标准差。
- 同时报告成功率、轨迹误差和平滑度，不只比较奖励。
- 明确区分探索噪声、网络 Dropout 和环境随机性。
- 若要证明图先验有效，增加链式邻接、全连接邻接和无图结构的消融实验。
- 若要比较架构优劣，增加参数量匹配版本与 SAC/PPO 等基线。

## 已知限制

- 当前参考轨迹是笛卡尔直线，不包含障碍物与复杂轨迹规划。
- 当前使用关节速度控制，尚未建模真实驱动器、时延、摩擦误差和传感器噪声。
- 尚未进行 domain randomization 或 sim-to-real 验证。
- 历史 GNN checkpoint 可能来自已经替换的旧网络类，参数结构不兼容时需使用历史代码或重新训练。
- 当前仓库中的历史实验结果不能与修复后的 v2 状态和奖励链路直接比较。

## 引用

核心算法参考：

```bibtex
@inproceedings{fujimoto2018addressing,
  title={Addressing Function Approximation Error in Actor-Critic Methods},
  author={Fujimoto, Scott and van Hoof, Herke and Meger, David},
  booktitle={Proceedings of the 35th International Conference on Machine Learning},
  year={2018}
}
```

如果本项目用于课程、科研或二次开发，请同时注明 PyTorch、Gymnasium 与 PyBullet。

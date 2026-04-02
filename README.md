# TD3 Multi‑Architecture for KUKA LBR iiwa (PyBullet)

> 一套开箱即用的 TD3 强化学习项目，支持 **MLP / GNN / Transformer / GNN+Transformer** 四种 Actor 架构；内含 **KUKA LBR iiwa 七自由度**到球目标的到达任务（PyBullet），完整的训练、测试与可视化评估管线。当前默认 **全程稠密奖励**，并提供批量随机点测试与轨迹出图。

---

## 目录结构（关键文件）

```
Td3 Multi Architectures Project Scaffold/
├── agents/
│   ├── td3_agent.py           # TD3 实现（设备/超参/训练循环/保存）
│   └── networks.py            # MLP / GNN / Transformer / GNN+Transformer
├── envs/
│   └── kuka_iiwa_env.py       # KUKA iiwa Reach 任务（PyBullet）
├── utils/
│   ├── replay_buffer.py       # 经验池（支持HER的实现可选；当前训练默认关闭HER）
│   └── gym_compat.py          # gym/gymnasium 兼容封装
├── training/
│   ├── train_experiment.py    # 训练脚本（默认全程稠密奖励；自动分run目录存档）
│   ├── evaluate.py            # 评估脚本（生成奖励/成功率/损失/平均损失/评估指标图）
│   └── test_agent.py          # 随机点批量测试 + 最优轨迹（3D）出图
├── results/                   # 每次训练的结果与权重（脚本自动创建）
│   └── LATEST_RUN.txt         # 指向最近一次 run 目录的指针（自动维护）
├── plots/                     # evaluate.py 输出的图像（plots/<run名称>/...）
└── plots_result/              # test_agent.py 输出的图像（随机评估/最优轨迹）
```

> **提示**：`training/train_experiment.py` 顶部已将本地项目根目录加入 `sys.path`，避免 Windows 下与第三方 `agents` 包发生命名冲突。

---

## 环境依赖

- Python 3.8+（建议 3.10）
- PyTorch >= 1.12
- gymnasium >= 0.28（已做 gym 兼容封装；**不要**再安装旧 gym）
- pybullet
- numpy, matplotlib, pyyaml, tensorboard（可选）

示例 `pip`：
```bash
pip install torch numpy matplotlib pyyaml gymnasium pybullet tensorboard
```

> 若你看到 “Gym has been unmaintained…” 的警告，说明环境中仍有旧 `gym`，**不必理会**（项目实际使用的是 `gymnasium` + 我们的 `utils/gym_compat.py`）。

---

## 任务说明（KUKA iiwa Reach）

- **目标**：七自由度机械臂末端（EEF）到达随机球目标点。
- **观测**：关节/末端/目标等特征（Dict 或向量）；脚本已支持将 Dict 展平。
- **奖励**：当前默认 **全程稠密奖励**（越近越好、到阈值结束）。
- **成功判据**：EEF 与目标距离 < `distance_threshold`（环境内可配置，默认 0.05 m）。

---

## 训练（Train）

**一条命令开始训练：**
```bash
python training/train_experiment.py --env KukaIiwa7Reach-v0
```

- 脚本会在 `results/` 里自动新建独立 run 目录并存储：
  - `rewards.npy`、`success.npy`、`metrics.json`
  - `final_model.pt` 与带时间戳的备份 `final_model_YYYYMMDD_HHMMSS.pt`
  - `config.json`（保存本次训练配置）
  - `results/LATEST_RUN.txt` 指向最近 run，便于 evaluate/test 自动读取

**切换模型架构**  
`training/train_experiment.py` 最下方有一处：
```python
arch = "gnn_transformer"  # "mlp" / "gnn" / "transformer" / "gnn_transformer"
```
将其改为你想要的架构后保存，再运行训练即可。

**常用参数（在脚本内默认即可）：**
- `--max_timesteps`（总步数，默认 300k）
- `--start_timesteps`（纯随机探索步数，默认 25k）
- `--eval_freq`（评估间隔步数，默认 10k）
- `--batch_size`（默认 256）
- `--expl_noise`（训练时动作高斯噪声，成功率↑后脚本会自动降噪）

> 训练中记录的 **loss 横轴** 是“全局步数”，而不是训练迭代次数；这是为了与 TD3 的延迟 Actor 更新语义一致。

---

## 评估（Evaluate → 输出到 `./plots/<run名称>/`）

```bash
# 默认读取最近一次 run（results/LATEST_RUN.txt），图片输出到 ./plots/<run名称>/
python training/evaluate.py

# 或指定某次 run：
python training/evaluate.py --run_dir results/<你的run目录>
```

**生成的图像（Times New Roman 字体，自动防覆盖）：**
- `rewards.png`：训练每回合奖励
- `success.png`：训练每回合成功标记 + 移动平均
- `train_losses.png`：Actor/Critic Loss（横轴为全局步数）
- `loss_average.png`：Actor/Critic 平均损失柱状图
- `eval_metrics.png`：评估 Reward / Success / TTS（成功所用步数） / Min Distance（最小距离）

> 可选：`--ma_window` 调整成功率平滑窗口（默认 20）。

---

## 测试（Test → 随机 30 点 + 最优轨迹 → 输出到 `./plots_result/`）

```bash
# 默认读取最近一次 run；评估 30 个随机起点/目标；GUI 可视化；输出图片在 ./plots_result/
python training/test_agent.py

# 指定 run 或评估回合数：
python training/test_agent.py --run_dir results/<你的run目录> --episodes 50
```

**输出图像：**
- `*_eval_curves.png`：四合一曲线（回合奖励、成功标记、TTS、最小距离）
- `*_hist.png`：TTS 与最小距离的直方图
- `*_best_trajectory.png`：**最优回合（按最小距离）末端 3D 轨迹**（带起点/终点/目标）

> 如需固定测试点以复现实验，可在 `test_agent.py` 中增加 `--seed` 和 `--same_every_episode`（README 末尾“可选增强”有示例）。

---

## 重要实现说明

### 1) 设备与数据类型健壮性
- `td3_agent.py` 的 `train()` 对 ReplayBuffer 的输出统一 `torch.as_tensor(..., device=self.device)`，兼容 **numpy / torch** 返回，避免 `.to(...)` 报错。  
- 训练脚本会记录 `actor/critic loss` 与对应全局步数，评估脚本按“全局步数”作图，便于横向对齐。

### 2) 稠密奖励与早停策略
- 全程稠密奖励（训练/评估），到达阈值即提前终止回合，提升样本效率。
- 评估成功率 ≥ 0.95 连续三次时触发**早停**并保存最终模型（同时保留时间戳副本）。

### 3) 结果与路径组织
- 每次训练各自在 `results/<env>_<arch>_dense_<timestamp>/` 中存档。
- `evaluate.py` 输出到 `plots/<run名称>/`；`test_agent.py` 输出到 `plots_result/`（带时间戳防覆盖）。

---

## 常见问题（FAQ）

**Q1：报 “Gym has been unmaintained since 2022…”**  
A：无影响。本项目用的是 `gymnasium`，并通过 `utils/gym_compat.py` 适配了新老接口。

**Q2：Windows 上 `agents` 名称冲突**  
A：`training/*.py` 顶部已将项目根目录提前加入 `sys.path`，优先使用你项目内的 `agents/`。

**Q3：训练太慢？**  
- 关闭 GUI：训练脚本用的是 `render_mode="direct"`（无 GUI）。
- 降低 `eval_freq`、减小 `max_timesteps` 或 `batch_size`（会影响效果）。
- 确认 PyTorch 在 GPU 上运行（`torch.cuda.is_available()`）。

**Q4：成功率高但奖励仍为负？**  
Reach 任务常见：稠密奖励是“负距离”一类形态，收敛到 **小负数**是合理的；更关心 **成功率 / 最小距离 / TTS**。

---

## 可选增强（按需启用）

- **固定测试集以复现实验**：给 `test_agent.py` 增加参数 `--seed`、`--same_every_episode`，用同一套随机点对比不同模型（代码片段可向我索取）。
- **导出论文风格 PDF 图**：在 `evaluate.py / test_agent.py` 的保存函数中并行保存 `.pdf`，打印更清晰。
- **HER + 稀疏奖励**：若将来要切到稀疏，建议同步打开 HER 并重建经验池，以免旧稠密样本干扰策略。

---

## 许可与引用

- 本项目基于标准 PyTorch / Gymnasium / PyBullet 生态实现。若用于论文或公开项目，请注明：
  - “TD3 Multi‑Architecture for KUKA LBR iiwa (PyBullet), 2025-09-04”
  - 并引用原始 TD3 论文：Fujimoto et al., “Addressing Function Approximation Error in Actor-Critic Methods”, ICML 2018.

---

**最后更新：** 2025-09-04 03:04:17  
如需我帮你把训练/测试脚本再加上更多指标或导出表格（CSV/Excel），直接告诉我即可。

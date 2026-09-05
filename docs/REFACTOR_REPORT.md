# GT-TD3 工程审查与重构报告

> 本文件记录先前保持算法逻辑的工程重构阶段。后续已按用户要求新增可选算法版本；
> 最新奖励、TD3/PER、终止与评估协议修改及验证结果见 [算法优化报告](ALGORITHM_OPTIMIZATION.md)。
> 下文测试数量和“未修改算法”等描述均指工程阶段当时的状态。

日期：2026-09-05。范围：当前工作区中的训练、仿真、网络、经验回放、评估和历史启动工具。

本轮按照“理解调用关系 → 基线测试 → 局部修复 → 回归验证”执行。没有删除历史实验、已有结果、权重或用户未纳入 Git 的文件，没有提交 Git commit。原有 9 项测试在修改前全部通过，但没有覆盖下面列出的故障路径。

## 1. 原有结构与运行流程

项目是 KUKA LBR iiwa 七自由度机械臂的 PyBullet 轨迹跟踪强化学习实验框架。核心环境从初始末端位置到随机目标生成直线参考轨迹，TD3 使用四种 Actor 表示进行比较。

```text
train.py <arch> / 各历史训练预设
  → training.train_experiment
    → training.config：命令行参数与架构默认值
    → envs.kuka_iiwa_env：7 维关节速度控制、参考轨迹、奖励
    → training.observation：GoalEnv 字典 → 31 维 v2 / 20 维 v1 状态
    → agents.td3_agent → agents.networks → agents.state_encoder
    → utils.replay_buffer：采样、PER 权重、TD error 优先级更新
    → training.evaluator → training.metrics：固定种子评估
    → training.artifacts：配置、权重、指标、回合记录与汇总
```

- Actor：MLP、图注意力 GNN、Transformer、GNN+Transformer。Critic、训练器和环境共享。
- 默认 31 维观测包括关节状态、轨迹阶段、当前参考点、前一动作、当前末端位置和目标。
- 默认 PER 开启、HER 关闭；PER 分支使用加权平方误差，普通回放使用 SmoothL1Loss。两者并非同一种损失。
- `training/test_agent.py` 加载权重并运行独立评估；`training/evaluate*.py` 处理训练结果和轨迹展示。
- `evaluation/` 是独立的论文扩展评估体系，带保持成功、平滑度等指标，不与训练评估等价。
- 数据主要来自在线仿真和 ReplayBuffer，不存在主训练流程必需的外部数据集。
- `results/<run>/` 保存运行产物；`assets/paper/` 是论文图片；根目录还有历史分析、验证、训练预设及说明文件。

完整静态文件、导入及顶层符号清单见 [source_inventory.md](source_inventory.md)。该清单不把“无静态引用”误判为“可以删除”。

## 2. 发现的问题与已完成修复

| 优先级 / 类别 | 问题位置 → 原因与后果 | 修改方案 |
|---|---|---|
| P0 严重 Bug | `quick_test_evaluate.py` → 嵌套 f-string 转义错误，脚本无法解析；路径二次加引号 | 改为 `subprocess` 参数列表，正确传递带空格路径，检查退出码 |
| P0 严重 Bug | `training/evaluate_enhanced.py` → `flatten_obs` 只在另一个函数内导入，轨迹生成时 NameError | 统一模块导入；新增真实权重加载与轨迹生成回归测试 |
| P0 潜在 Bug | `agents/td3_agent.py` → Tensor 输入绕过 float32/device 转换 | ndarray、list、Tensor 统一转换；增加维度与有限值检查 |
| P1 稳定性 | `training/train_experiment.py` → 第二环境、模型初始化或最终保存异常时，旧 finally 未覆盖或提前退出 | 用 ExitStack 管理全部已创建环境；保存异常不覆盖原始训练异常 |
| P1 稳定性 | `envs/kuka_iiwa_env.py` → 断开失败后调用无 client ID 的全局 disconnect，可能影响另一环境 | 只操作自己持有的 client；未连接时不探测默认连接 |
| P1 潜在 Bug | `TD3.select_action` → forward 异常后 Actor 留在 eval 模式 | 使用 try/finally 恢复调用前模式 |
| P1 配置 | `training/config.py` → delay=0、非法 gamma/tau、负噪声、无效节点维度、错误设备等延迟失败 | 文件与 CLI 参数校验；程序化调用同样检查；GPU 可用性在分配前检查 |
| P1 输入 | 环境 / ReplayBuffer → 错误动作形状可被广播，NaN 可污染仿真或经验 | 仿真推进和写入前验证形状、有限值；优先级错误不再污染采样概率 |
| P1 结果记录 | `training/artifacts.py` → 汇总记录计划步数而非实际完成步数 | 保持 CSV 列名，`timesteps` 记录 `completed_timesteps` |
| P1 结果记录 | 训练终点不是 eval_freq 的整数倍时，保存的末次评估落后于最终模型 | 正常完成后补一次最终步数评估；指标数组明确记录真实步数 |
| P1 数据完整性 | 权重、JSON、NPY、CSV 直接覆盖，写入失败损坏旧文件 | 同目录临时文件 + flush/fsync + 原子替换；兼容原权重文件格式 |
| P1 并发 | 同秒同架构运行目录冲突；并行 CSV 首次写入竞争 | 目录时间戳增加微秒；共享汇总使用本地进程间文件锁 |
| P1 并发 | 批量任务预分配 GPU 后进入共享队列，快任务可能与慢任务争用同一 GPU | 每张 GPU 使用独立串行队列；队列之间并发；超时增加 CLI 配置；失败返回非零退出码 |
| P1 结果展示 | `evaluate_seeds.py` 用序号冒充训练步数，增强评估按数组索引平均不同 checkpoint | 提取共享对齐模块，只统计共同的真实评估步；缺失或错误步数明确报错 |
| P1 结果展示 | 单次评估设置 ma_window=1 后对 None 求长度，直接崩溃 | 窗口为 1 时保留原始数据，只有窗口大于 1 时计算平滑曲线 |
| P1 评估配置 | 扩展模拟参数新增后，独立评估若仍用固定默认值会测试错误环境 | 训练和评估共享环境参数映射，恢复保存的时间上限、控制步数、速度限制和阈值 |
| P2 可维护性 | 多套 observation 展平和 Gym 适配代码重复 | 训练及扩展评估复用规范函数；保留原有函数入口 |
| P2 checkpoint | 加载逻辑重复且维度不匹配错误发生较晚 | 新增 `TD3.load_actor`，检查文件、类型、键及 shape 后加载；明确只用于 Actor 推理 |
| P2 路径 / 环境 | 历史预设写死 `python`、依赖当前工作目录、失败后继续报成功 | 使用 `sys.executable`、明确项目目录、传播子进程错误；根目录 train.py 直接调用统一入口 |
| P2 规范 | 缺少统一静态检查、开发依赖；pytest 容易误收集历史长实验脚本 | 新增 pyproject.toml、requirements-dev.txt，测试范围限定 tests/ |
| P3 文档 / 日志 | 网络构造大量 print，部分注释与真实奖励不一致；旧启动横幅声称未实现的改进 | 网络说明改 debug 日志；训练使用 logging 并保存 run.log；更新奖励说明和旧 GNN 启动器描述 |
| P3 边界 | 位置编码假设隐藏维数为偶数，奇数宽度时赋值 shape 不一致 | 余弦分支切到实际列数；当前默认偶数宽度的计算不变 |
| P4 性能 | 默认关闭 HER 仍逐样本拼接状态 | 无 HER 分支改为批量索引和拼接，保留旧随机数消耗和所有返回字段 |

## 3. 配置与新结构

现有包边界已经适合该项目，未为了套模板搬进 `src/`，从而避免破坏脚本路径和 checkpoint 所依赖的类定义。

```text
project/
├── train.py                         # 保留原架构位置参数
├── configs/smoke.yaml               # 小规模 CPU 验证配置
├── pyproject.toml                   # pytest + Ruff 正确性规则
├── requirements.txt
├── requirements-dev.txt
├── agents/                         # 原网络结构；新增可靠的推理和加载检查
├── envs/                           # 原观测/动作/奖励；输入及 client 管理修复
├── training/
│   ├── config.py                   # 默认值、JSON/YAML、CLI 校验、环境参数映射
│   ├── train_experiment.py         # 统一训练生命周期与日志
│   ├── artifacts.py                # 产物、runtime.json、实际步数汇总
│   ├── observation.py
│   ├── series.py                   # 多种子评估步数对齐
│   ├── evaluator.py / metrics.py
│   └── test_agent.py / evaluate*.py / 历史预设
├── evaluation/                     # 保留独立论文扩展评估协议
├── utils/
│   ├── persistence.py              # 原子文件替换、共享汇总锁
│   ├── replay_buffer.py
│   ├── gym_compat.py
│   └── trajectory_planner.py       # 未确认废弃，保留
├── tests/                          # 原测试 + 工程故障与完整链路回归
├── tools/audit_project.py           # 全仓语法检查和静态调用清单
├── docs/REFACTOR_REPORT.md
├── docs/source_inventory.md
└── results/<run>/                   # 原产物 + run.log、runtime.json
```

默认参数仍以 `training/config.py` 为准。`--config` 可读 JSON/YAML，显式 CLI 参数覆盖文件值。配置文件使用与 CLI 一致的平铺字段，未知字段拒绝，避免拼写错误悄悄失效。相对配置路径和 save_dir 均按调用工作目录解释；历史子进程预设的工作目录固定为项目根目录。

根目录原有 `experiment_configs.yaml` 使用 Pendulum/旧字段，是历史配置，当前主入口原本就没有读取它；保留原文件，不将它冒充为新系统的可运行默认配置。旧 run 的 `config.json` 含运行时元数据，继续用于评估，不直接作为 `--config` 的训练输入。

## 4. 兼容性及算法边界

- 原启动方式继续可用：`python train.py mlp`、`python -m training.train_experiment`、`python training/train_experiment.py`。新增 `--config` 和环境/日志参数均为可选项。
- 默认 max_steps=200、sim_steps_per_action=10、joint_vel_limit=1.5、distance_threshold=0.10 保持不变；Gym 外层 TimeLimit 与配置中的回合长度同步。
- `final_model.pt` 和 `checkpoint_latest.pt` 仍是纯 Actor state_dict；没有改成复合 checkpoint，没有变更网络参数名称或默认 shape。新增 runtime.json 只记录 Python、平台和主要依赖版本。
- v1/v2 现有网络实现均做了保存/重新加载测试；无法保证更早、架构不同的历史权重可载入，载入失败会明确提示。仅有 Actor 权重仍不能精确恢复训练。
- 奖励、观测排列、动作范围默认值、TD3 target、loss、optimizer、learning rate、梯度裁剪、网络拓扑、终止逻辑均保持当前主实现。没有新增所谓性能更优的算法预设。
- **结果记录修正**：中断训练实际步数、最终评估补齐、多种子共同 checkpoint 对齐会改变输出记录或曲线。这些不会改变训练中已有动作和梯度更新，但旧汇总图应重新生成。
- 参数校验有意收紧：非有限数值、不能实现的编码维度、缺失/错配多种子评估步数现在直接报错，避免输出误导性结果。
- 原子性以单个文件为边界，整个运行目录并不是跨文件事务；共享锁面向本地文件系统。

## 5. 验证方法与结果

环境：Windows，Python 3.13.5，PyTorch 2.12.0，NumPy 2.5.2，Gymnasium 1.2.3，PyBullet 3.2.7；当前 PyTorch 检测 CUDA 不可用。

1. 修改前：`python -m unittest discover -s tests -v`，9 项通过。
2. 修改后：`python -m pytest -q`，覆盖原 9 项与新增回归；具体最终数量见本报告末尾验证记录。
3. 四种 Actor × 普通/PER 回放，共 8 组真实 PyBullet 小训练；每组完成 Critic 更新、延迟 Actor 更新、周期/最终评估、文件保存及 Actor 重载；loss 有限。
4. v1 的四种架构保存/重载、v2 推理、错误维度权重拒绝且不改动当前 Actor、Tensor float64 推理、forward 异常后模式恢复通过。
5. 初始化/评估/保存故障注入、原子写入中断、空回放/错误动作/NaN 优先级、非法配置/GPU、多种子步数对齐、GPU 队列隔离均纳入回归。
6. 真实 CLI：`python -m training.train_experiment --config configs/smoke.yaml` 成功，运行 4 个控制步并写出完整产物。
7. 在项目外的临时工作目录启动 train.py 并完成小训练；原直接训练入口及 4 个历史预设的 `--help` 返回 0；独立 test_agent 加载权重评估 1 回合，生成 CSV 和 4 张图，返回 0。
8. 增强评估实际加载模型和保存的环境参数，生成两步回合的 3 个轨迹点；绘图函数在该自动化回归中替换为 mock，重点验证加载与轨迹数据流程。
9. Ruff 全仓检查覆盖语法、未定义名称等正确性规则；`tools/audit_project.py` 对 64 个 Python 文件进行编译及 Python 3.10 语法解析。此项不等于在 Python 3.10 解释器上执行过运行测试。
10. 经验回放与 `git show HEAD:utils/replay_buffer.py` 的修改前版本对照：普通/PER 各 5 个种子、batch=512，所有 batch 字段和采样后 NumPy 随机状态完全一致。

局部采样基准（CPU，2048 条数据、batch=512、100 次）：普通回放约 0.160→0.006 秒；PER 约 0.169→0.018 秒。该结果仅度量采样，不能外推为整段训练的提速或算法收敛提升。

## 6. 尚未解决或尚未验证的问题

以下内容需要单独实验或明确协议，不能通过工程整理宣称解决：

1. **这是算法层面的修改，而不是工程重构：终止/截断处理。** 原实现合并 terminated 和 truncated，回放在二者发生时均停止 bootstrap。如果时间上限代表连续任务的外部截断，应区分两者；但当前状态含任务阶段，也可能希望将末段作为有限时域终点。建议先明确任务定义，再比较两种 target 处理。修改会影响价值估计和学习结果，本轮未改变。
2. **HER 的轨迹一致性与环形覆盖。** 当前默认 HER=0；历史 HER 只重标记目标并近似重算距离奖励，没有同步轨迹参考/动作历史，`her_k` 也没有实际控制采样。长回合与环形覆盖下的未来目标索引需要独立验证。建议轨迹训练继续关闭 HER；不能把该历史分支当作已验证的轨迹 HER。
3. **PER 与非 PER 的损失差异。** 原实现分别使用加权 MSE 和 Huber。若统一损失以做公平消融，属于算法实验，必须重新跑种子，本轮未修改。
4. **评估协议仍有历史差异。** 训练评估采用首次进入阈值，`evaluation/` 默认要求连续保持成功，增强轨迹报告还记录最终一步成功。论文扩展模块的平滑度默认 dt=0.0417；它尚未随新控制参数自动更新。不同协议的结果不能直接混用，主训练未接入该扩展模块。
5. **不能精确断点续训。** 仍只保存 Actor，不含 Critic、optimizer、scheduler、ReplayBuffer、RNG 与仿真状态。本轮没有用“checkpoint”文件名冒充完整训练恢复。
6. **没有进行完整训练或性能结论验证。** 小训练只证明链路与回归正确，不能证明收敛、成功率、稳定性或论文数值改进。
7. **真实 CUDA、多 GPU、Linux 和其他 Python 版本没有运行验证。** GPU 队列为无 GPU 的调度回归；设备不可用检查已测。跨平台分支需要对应机器验证。批量运行仍有默认 7200 秒超时，可显式配置；长任务的取消/子进程组管理尚可进一步完善。
8. **历史工具并未全部执行真实实验。** 已全仓编译/静态检查，主要工具链运行验证；根目录的长实验、论文抓取、交互 GUI、全部绘图组合及所有现存权重没有逐一执行。旧优化说明中的提升比例和部分架构描述不能视为当前实验事实。

## 7. 后续建议

**高优先级**：固定一套任务终止和评估协议；评估完整 checkpoint 恢复需求；在 GPU 环境验证四架构，再运行多种子正式训练。若使用扩展平滑度指标，先传递实际控制周期。

**中优先级**：建立训练与评估配置 schema 版本；明确历史脚本维护名单；将不同预设整理成可追踪配置；完善批量任务取消、长时运行和跨平台 CI。

**低优先级**：在确认外部调用后逐步归档旧脚本和说明；再按维护需求拆分大型绘图文件；对大容量 PER 的 O(buffer_size) 概率计算做专门性能分析，避免未经测量替换采样算法。

## 8. 最终验证记录

- `python -m pytest -q`：27 项通过（含四种架构 × 两种回放的子案例）。
- `ruff check .`：通过；核心新增/修改模块额外检查无用 import、局部无用变量，通过。
- `python tools/audit_project.py`：64 个 Python 文件编译及 Python 3.10 语法解析通过。
- `git diff --check`：通过。
- 共享汇总写入补充实测：3 个独立进程各写 10 次，30 条完整记录且只有一个表头。
- 测试日志保留于 `tmp/refactor-pytest.log`，CLI 运行日志保留于 `tmp/refactor-cli.log`；这些本地验证文件已加入忽略规则。

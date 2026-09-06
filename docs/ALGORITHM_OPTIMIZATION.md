# GT-TD3 算法优化与验证记录

日期：2026-09-05。适用入口：`python -m training.train_experiment`。

本轮在工程重构之后，对奖励、控制方式、TD3 更新、经验回放和评估协议进行了修改。新版通过 `configs/algorithm_v2.yaml` 显式启用；不传新版配置时，保留 legacy 算法、原奖励和直接速度控制。四种 Actor 的网络层结构与参数形状没有替换。

**当前证据支持实现正确性和运动学基线的有效性，不支持“学习到的残差优于控制器”或“某种网络架构更优”的结论。** 本次短训练中，残差策略的验证集最佳模型均来自训练前的零残差初始化。

## 1. 问题与处理

| 位置 | 原问题及影响 | 本轮处理 |
|---|---|---|
| `envs/kuka_iiwa_env.py` | 每步终点奖励与按时间跟踪参考点竞争，可能鼓励提前到终点等待 | 新增时间对齐的 tracking 奖励，终点项只在轨迹末步生效 |
| 环境、`utils/control.py` | 七关节速度从随机探索开始，忽略已有运动学信息 | 可选阻尼最小二乘控制器加学习残差，加入预测关节限位裁剪 |
| `agents/td3_agent.py` | 关节角、速度、历史动作尺度不同；Dropout 引入额外策略随机性 | corrected 分支统一物理尺度，关闭 Actor 和目标 Actor 的 Dropout |
| TD3 价值更新 | 固定 Q 目标裁剪可能扭曲新奖励下的价值；PER 与均匀回放使用不同损失 | corrected 去掉固定 Q 裁剪，统一 Huber/MSE，再应用重要性权重 |
| `utils/replay_buffer.py` | 不放回采样与常用 PER 权重公式不一致；重复索引更新存在歧义 | corrected 使用有放回比例采样、全局权重归一化、重复索引取最大优先级 |
| TD3 策略更新 | Actor 沿用优先采样状态，改变策略梯度的状态分布 | corrected Actor 独立均匀抽样状态，更新时冻结 Critic 参数 |
| 环境、训练器 | 真正任务结束和外部时间截断没有分别处理价值自举 | tracking 轨迹结束为 terminated；外部 truncated 保留自举 |
| 训练与评估 | “曾进入终点范围”不能表示完整轨迹跟踪成功；最后模型可能退化 | 完整轨迹 RMSE 与终点误差联合判定；独立验证集选择 best_model |
| 扩展评估 | 路径长度平方被称为能耗；差分指标可能使用错误时间步 | 无真实能量数据时 energy 为 NaN；保留明确命名的代理指标，使用环境实际控制周期 |

## 2. 奖励定义

设执行一步后的末端位置为 \(p_t\)，同一时刻参考位置为 \(p_t^{ref}\)，实际施加的关节速度为 \(u_t\)，速度上限为 \(u_{max}\)。定义

\[
\rho(x)=\sqrt{1+x^2}-1,
\]

\[
r_t=-\rho\!\left(\frac{\|p_t-p_t^{ref}\|}{0.10}\right)
-0.02\frac{1}{7}\sum_{j=1}^{7}\left(\frac{u_{t,j}-u_{t-1,j}}{u_{max,j}}\right)^2
-2\mathbf{1}_{t=T}\rho\!\left(\frac{\|p_t-p_{goal}\|}{0.05}\right).
\]

距离单位为米。尺度和权重均可配置，以上为候选配置，并非经过搜索得到的最优值。Pseudo-Huber 代价在小误差附近近似二次、在大误差处保持线性增长，避免原指数奖励在远离参考时趋于平坦。

- 取消每步终点吸引和提前抵达奖励，完整执行规定时长的轨迹。
- 平滑项使用实际控制命令，与上一时刻的实际命令比较；第一步与零命令比较。该项是速度命令变化惩罚，不是真实力矩、能耗或末端 jerk 惩罚。
- 使用关节均值和速度上限归一化，避免关节数与动作量纲直接改变惩罚尺度。该定义针对固定控制周期；改变周期需要重新校准权重。
- 每步及每回合记录 tracking、smoothness、terminal 的加权贡献，可检查哪一项主导训练。
- tracking 必须搭配 31 维 v2 观测、dense 奖励和完整回合；配置校验拒绝提前成功终止。

## 3. 运动学残差控制

基线使用末端位置 Jacobian 与参考速度：

\[
v_t^{cmd}=\frac{p_{t+1}^{ref}-p_t^{ref}}{\Delta t}+K(p_t^{ref}-p_t),\qquad
u_t^{base}=J_t^\top(J_tJ_t^\top+\lambda^2 I)^{-1}v_t^{cmd}.
\]

先裁剪基线速度，再组合 Actor 输出的策略命令 \(a_t\)：

\[
u_t=\operatorname{clip}\left(u_t^{base}+0.2a_t\right).
\]

实现通过线性方程求解而非显式求逆计算 DLS，默认 \(K=4\)、\(\lambda=0.05\)。可选限位逻辑进一步限制预测位置 \(q+u\Delta t\) 在 URDF 关节限位内，新配置留出 0.05 rad 裕量。这是速度命令层面的约束，不能保证动力学仿真或实体机器人绝不越界。

Actor 最后线性层可零初始化，使训练前策略等于纯运动学控制器。其后网络学习有界速度修正。回放存储 **策略命令 a**，Critic 对该策略命令估值；环境状态中的前一动作和平滑惩罚使用 **实际命令 u**。两者不能互换。探索噪声和 TD3 目标平滑噪声加在策略命令空间，环境随后执行残差缩放。

这里仅使用平移 Jacobian；没有新增末端姿态目标、避障、自碰撞约束或零空间姿态优化。当前任务仍是固定初始姿态到随机目标的直线位置轨迹。运动学残差的设计动机可参考 [Residual Reinforcement Learning for Robot Control](https://arxiv.org/abs/1812.03201)，本仓库的实现和性能应以本地实验为准。

## 4. corrected TD3 与 PER

- 状态输入在训练、目标估计与推理中共同转换：关节角除以 π，关节速度和上一实际命令除以速度上限；位置仍以米表示，阶段仍为 [0,1]。回放保留原始状态，转换集中在 Agent。
- Critic 输入的策略命令除以速度上限；Actor 对外仍输出原动作空间中的数值。
- corrected Actor 与目标 Actor 的 Dropout 设为 0，保留层和权重形状。探索由显式噪声控制，避免训练/推理策略因 Dropout 不一致。
- 保留双 Critic、目标策略平滑、延迟策略更新及软目标更新；移除任意的 [-200,200] 目标值裁剪。目标为 `r + gamma * (1 - terminal) * min(Q1_target, Q2_target)`。
- 真实有限轨迹结束不自举；仅外部截断仍自举。回合结束标志与 TD 自举掩码分别保存。
- Critic 支持 Huber 和 MSE；PER 对逐样本损失加权，与不开启 PER 时保持相同的基础损失定义。
- 比例 PER 使用 `P(i) ∝ priority(i)^alpha` 的有放回采样。重要性权重按整个缓冲区的最大权重归一化，beta 随实际训练更新退火。双 Critic 中较大的绝对 TD error 用作优先级；一次采样中重复索引取最大值更新。
- Actor 使用独立均匀状态样本。Critic 参数在 Actor 更新时不求梯度，但仍允许梯度通过 Critic 的动作输入传回 Actor。
- 新配置采用分离学习率、探索噪声退火，记录 Q、目标 Q、绝对 TD error 与学习率诊断。固定状态缩放、关闭 Dropout 和均匀 Actor 抽样是本项目的设计选择，仍需通过长训练消融检验效果。

TD3 的标准更新机制参考 [OpenAI Spinning Up：TD3](https://spinningup.openai.com/en/latest/algorithms/td3.html)；比例优先回放与重要性修正参考 [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)。HER 在主训练流程中继续关闭，本轮没有为新的时间相关奖励实现 HER 重标记。

## 5. 模型与评估协议

tracking 模式的成功条件为：执行完整参考轨迹，所有对应点的 RMSE < 0.05 m，且末端终点误差 < 0.10 m。初始位置也纳入 RMSE。两个阈值可配置。仅抵达终点的 endpoint_success_rate 单独报告；TTS 是首次进入终点范围的时间步，不能代表完整轨迹成功时间。

核心评估、独立测试和扩展论文评估共享完整轨迹成功判定。legacy 主评估仍保留旧的“曾进入终点范围”口径，比较不同奖励时应使用明确的 tracking_success_rate，而非直接混用 success_rate。

corrected 从第 0 步起进行验证，按“跟踪成功率优先，其次 RMSE，最后终点误差”保存 `best_model.pt`。验证目标种子默认从 10000 开始；本轮留出测试从 20000 开始。独立测试入口优先加载 corrected 的 best_model，仍保留 final_model 供显式检查。选出第 0 步是合法结果，意味着后续训练尚未改进初始策略。

权重仍是 Actor 的 state_dict，同时新增同名 `.meta.json`，用于验证 algorithm_version、动作边界及 direct/residual 语义。**复制新版模型时，应携带元数据和同目录 config.json，不能仅复制 .pt。** corrected 拒绝没有元数据的权重；历史 legacy 无侧文件权重仍允许加载。旧权重不能直接按 corrected 缩放和残差语义解释。

模型文件仍不是完整训练恢复快照，未保存恢复所需的全部优化器、回放和随机状态。不要把重新加载 Actor 称作无缝续训。

扩展指标的 energy 现在为 NaN，因为没有采集关节力矩和速度积分；旧值保留为 path_length_squared_proxy。末端 jerk 用实际环境控制周期计算，单位 m/s³；动作变化率单位 rad/s²。新旧时间步或能耗口径的结果不可直接混合比较。

## 6. 已执行验证

算法修改完成时的检查为 38 项测试通过；仓库清理后仍以 `python -m pytest -q`、Ruff 和全源文件编译作为最终门禁。新版 GNN+Transformer 的独立测试命令已实际加载 `best_model.pt`，完成一个留出目标回合并输出指标和图片。

自动回归覆盖奖励偏好、末步奖励、平滑项尺度、终止/截断自举、PER 分布及权重、重复优先级、奇异 Jacobian 求解、Bullet Jacobian 数值差分、零残差控制、四种 Actor 的训练/重载、确定性、未裁剪 Q 目标和评估协议一致性。

另执行 GNN+Transformer 真实 PyBullet 200 步训练：100 步预热、batch=16、两个验证目标，训练和权重保存正常结束。初始验证 RMSE 约 0.0111 m、成功率 100%；第 200 步约 0.0504 m、50%，因此 best_model 正确保留第 0 步。该检查验证运行链路，不是网络优越性实验。

### 短训练对比

使用 MLP、训练种子 42 和 123，每个变体每种子 600 环境步（约 3 回合），预热 150 步、batch=64。每次训练在验证集上选模型，最终统一评估同一组 5 个留出目标。表中学习策略为两个训练种子的均值；两个无训练基线各评估一次相同目标集。

| 方法 / 检查点 | RMSE (m) | 终点误差 (m) | 完整跟踪成功率 | 末端 jerk (m/s³) |
|---|---:|---:|---:|---:|
| 零速度基线 | 0.607260 | 1.050474 | 0% | 0.000746 |
| 纯 DLS 控制器 | 0.011249 | 0.000922 | 100% | 1.813390 |
| legacy / final | 0.536849 | 0.845101 | 0% | 0.381849 |
| 仅替换奖励 / final | 0.619071 | 0.983007 | 0% | 1.129075 |
| corrected 直接控制 / final | 0.667591 | 0.692642 | 0% | 4.860197 |
| corrected 直接控制 / validation_best | 0.607260 | 1.050474 | 0% | 0.000746 |
| corrected 残差控制 / final | 0.027065 | 0.023317 | 90% | 1.624457 |
| corrected 残差控制 / validation_best | 0.011249 | 0.000922 | 100% | 1.813390 |

原始逐运行记录见 [algorithm_benchmark_20260905.json](algorithm_benchmark_20260905.json)，本地完整输出位于 `results/algorithm_benchmark_20260905/`。不同奖励的总回报没有共同尺度，因此不用于横向排序。

**两个残差运行的最佳检查点都是第 0 步，即纯 DLS，100% 不是学习增益。** 仅改奖励和 corrected 直接控制在这次短训练中均未超过 legacy 的 RMSE；不能据此断言长训练优劣。零速度策略 jerk 极小但完全没有完成任务，说明平滑指标必须和跟踪精度共同解释。

这是训练流程诊断：corrected 配置同时调整优化器、初始化和探索等因素，并非每行只改变一个变量的严格单因素消融。样本量很小，没有计算可靠置信区间，也没有完成 50 万步训练、多架构排序、GPU 长时间稳定性测试或实机验证。

## 7. 使用方式

在项目根目录执行新版候选配置：

```bash
python -m training.train_experiment --config configs/algorithm_v2.yaml
```

保持 corrected 训练与奖励、切换回直接速度控制：

```bash
python -m training.train_experiment --config configs/algorithm_v2.yaml --control_mode direct
```

只启用新的奖励（其余采用 legacy 默认配置）：

```bash
python -m training.train_experiment --reward_mode tracking
```

快速检查 GNN+Transformer 链路：

```bash
python -m training.train_experiment --config configs/algorithm_v2.yaml --device cpu --torch_threads 2 --max_timesteps 200 --start_timesteps 100 --batch_size 16 --buffer_size 512 --eval_freq 200 --eval_episodes 2 --save_dir results/algorithm_v2_smoke
```

独立测试（将路径换成实际生成的运行目录；20000 与默认验证种子分离）：

```bash
python -m training.test_agent --run_dir "results/algorithm_v2/<运行目录>" --episodes 20 --seed 20000
python -m training.test_agent --model "results/algorithm_v2/<运行目录>/final_model.pt" --episodes 20 --seed 20000
```

复现本轮短训练对比与检查：

```bash
python -m training.benchmark_algorithms --steps 600 --seeds 42 123 --episodes 5 --threads 2 --output results/algorithm_benchmark
python -m pytest -q
python tools/audit_project.py
ruff check .
```

后续科研实验应先固定任务分布和总交互预算，分别比较纯 DLS、直接 TD3 与残差 TD3，再在相同训练设置下比较四种网络。必须保留独立测试目标、多训练种子和纯控制器基线；只有残差策略稳定超过该基线，才能报告学习带来的提升。若要证明泛化，还需要额外的起始姿态、曲线轨迹、负载和模型误差测试，本轮没有生成这些结论。

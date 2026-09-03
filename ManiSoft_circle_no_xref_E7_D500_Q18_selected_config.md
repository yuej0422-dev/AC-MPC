# ManiSoft circle：选定 E7 D5.0/Q1.8 offline 配置记录

## 1. 选定结果

本文件冻结当前选定的 no-xref implicit-reference AWAC-KMPC 离线配置。模型选择的 primary metric 是 **10-episode deterministic return**，而不是 tip RMSE。

| 项目 | 数值 |
|---|---:|
| 组别 | **E7_D500_Q18** |
| Offline update | **12,500** |
| Deterministic return | **595.207677** |
| Sparse / dense episode return | 252.000000 / 343.207677 |
| Tip XYZ RMSE / P95 | 10.868068 / 24.153310 mm |
| Tip success@2.5 mm | 30.1% |
| 三节点 joint RMSE / P95 | 9.104323 / 19.589708 mm |
| 三节点 joint success@2.5 mm | 50.4% |
| Physical action saturation | 0% |

选定权重：

```text
/root/autodl-tmp/AC-MPC/runs/o2o/diagnostics/no_xref_authority_refinement_20260902/runs/E7_D500_Q18/best_return.pt
```

```text
SHA256: b2eca8b15f95a3a3c834b9c36055ecb9f9d49a280285e1542fc7f7921b376eac
size:   20,264,724 bytes
```

注意：`best_return.pt` 是按照本轮协议重新明确保存的 return-best checkpoint。不要用 `offline_020000.pt` 代替；20k 的 return 只有 533.582786。

## 2. 实验身份与随机种子

| 项目 | 配置 |
|---|---|
| Method | AWAC-KMPC |
| Task | `manisoft_circle` |
| Training seed | **20260851** |
| Offline updates | 20,000 |
| Online steps | 0 |
| Device | CUDA |
| GPU | NVIDIA A100-PCIE-40GB，40,960 MiB |
| CPU affinity | cores 70–79 |
| CPU math threads | OMP/MKL/OpenBLAS 各 10 |
| Python | 3.12.3 |
| PyTorch / CUDA | 2.5.1+cu124 / CUDA 12.4 |
| NumPy | 1.26.4 |
| Config fingerprint | `54fe85ac614676a159545ebfc65babec20d065a038274d4e7b0432901eea72c2` |

随机流在 learner 内部分为独立的 actor initialization、critic initialization 和 training-sampling substream；actor 结构改变不会因消耗全局 RNG 而间接改变 critic 初始化。

## 3. 数据、模型与任务资产

### Offline dataset

```text
path:   /root/autodl-tmp/AC-MPC/runs/o2o/diagnostics/dataset_rebuild/manisoft_circle_curated_200k_v1.npz
SHA256: e374d27d28e2755d415c0a316a180569f3ece5b99904dc19fcdf45654cc748a4
```

数据集有 200,000 条 transition：observation/next_observation 为 46 维，action 为 18 维；另保存 reward、discount、episode ID/step、MC return、quality tier、三节点误差及逐时刻 target positions。

### Koopman model

```text
path:   /root/autodl-tmp/AC-MPC/work_dirs/manisoft_abs_u06_1132ep_h0_formal_walker_512/koopman_h0_formal_walker_loss/koopman_history/best_validation.pt
SHA256: d97234dd406d48d39dc01e5a2ffa86edc105313285c7405f43f0581e0f019b9d
```

- H0，无 history context。
- Physical state 45 维，absolute action 18 维。
- Encoder：45 → 512 → 512 → 32，SiLU。
- Koopman lifted body state 是 normalized physical state 与 learned lift 的组合，共 77 维；再拼接标量时钟 `tau=t/(T-1)`，actor/critic state input 共 **78 维**。
- Koopman `A/B/C` 和 encoder 在 RL 中冻结。

### Feedforward、reference 与 scenario

```text
u_ff:
  /root/autodl-tmp/AC-MPC/runs/o2o/probes/manisoft_circle_phase15_degraded/policy.npz
  SHA256 515adce9b5e6b189179e87b236d183eb301230e4156d0e42095e533d2cf43ffe

reference:
  /root/autodl-tmp/AC-MPC/work_dirs/manisoft_circle_r010_benchmark_klqr/trajectory.npz
  SHA256 b297e0a8bff3fd38de5701c7c068739bb8c2456708903eac1d7b80eff615dfb5

scenario:
  /root/autodl-tmp/ManiSoft/configs/demo_elastica_fast.yaml
  SHA256 f7cd98f2826843bcad473fcb305860db994e7c97abb476ae72363e9c5623fa75
```

`u_ff(t)` 由 KMPC 内部按时钟查表；不进入 actor observation。Reference 仅用于环境 reward/evaluation；**actor 和 KMPC cost 均不显式使用 xref**。

## 4. Observation、动力学和动作语义

Actor 输入：

```text
o_actor = [frozen Koopman lifted body state, tau]  # 78-D
```

Actor 不输入：

```text
xref, target positions, u_ff, future reference
```

KMPC 使用 absolute-action Koopman dynamics：

```text
z[k+1] = A z[k] + B (u_ff[k] + Delta_u[k])
```

- Actor/KMPC 最终输出为 18 维 physical residual `Delta_u`。
- 环境执行 `u = clip(u_ff(t) + Delta_u, -0.5, 0.5)`。
- Replay action 坐标也是 physical residual action。
- Episode 1,000 control steps；control dt=0.02 s，physics dt=0.0002 s，100 substeps/control step，总时长 20 s。

## 5. Actor 网络与 Q,p-KMPC 参数化

### 网络

```text
78 -> Linear(256) -> GELU
   -> Linear(256) -> GELU
   -> Linear(180)
```

- 两个 hidden layers，width=256。
- 最终 180 维 head：
  - `delta_q_xyz`: 5 stages × 9 coordinates = 45；
  - `raw_D_xyz`: 5 × 9 = 45；
  - `raw_d_action`: 5 × 18 = 90。
- Output head 的 weight/bias 均 zero initialization，因此 update 0 时 `Delta_u=0`，精确复现 pure feedforward。
- Actor checkpoint state 中含冻结 Koopman encoder；actor state-dict 共 512,981 parameters，其中可学习 controller 约 132k parameters。
- Exploration log standard deviation：init `-3.5`，upper clamp `-3.0`。

### Horizon 与求解器

| 项目 | 配置 |
|---|---:|
| Horizon | 5 |
| Stage sharing | 否；5 stages 独立输出 |
| Projected-gradient solver iterations | 5 |
| Terminal cost | 无 |
| Explicit xref | 无 |
| State-cost gate | 无，恒为 1 |

### Implicit XYZ state cost

仅三个观测节点的 XYZ 坐标具有 adaptive state cost：physical indices

```text
[0,1,2, 15,16,17, 30,31,32]
```

每个 horizon stage：

```text
D_xyz = D_nom_xyz(z_t, u_ff) + Delta_D_xyz
Delta_D_xyz = D_scale * tanh(raw_D_xyz)

Q_xyz = Qbase_xyz * exp(Delta_q_xyz)
Qbase_xyz = 1
Delta_q_xyz in [-1.5, 1.8]

p_xyz = -Q_xyz * D_xyz
```

其中正侧 q bound 使用保持 update-0 局部斜率为 1.5 的平滑非对称参数化；`Q/Qbase` 范围约为 `[exp(-1.5), exp(1.8)] = [0.2231, 6.04965]`。

E7 使用 `D_scale = 5 × D_scale_base`。九个 physical bounds（m）为：

```text
[0.04964376, 0.03723996, 0.02500000,
 0.14999999, 0.14999999, 0.02500000,
 0.14999999, 0.14999999, 0.02500000]
```

`D_nom` 是从当前 lifted state 出发，使用冻结 Koopman model 和 horizon `u_ff`、令 `Delta_u=0` 得到的 nominal rollout；并非显式轨迹 xref。

### Fixed velocity regularization

Linear/angular velocity indices：

```text
[9..14, 24..29, 39..44]
```

固定 `Q_velocity=0.05`，cost center 是 nominal Koopman rollout；velocity D/Q 不学习。其他未列出的 physical state coordinates 的 state cost 为 0。

### Action-side affine cost

```text
R = 10000                     # fixed
d_action = 0.01*tanh(raw_d_action)
p_action = -R*d_action
```

因此每个 stage 的 residual action cost 等价于：

```text
0.5 * (Delta_u - d_action)^T R (Delta_u - d_action) + constant
```

最终传入 KMPC 的变量仍是 diagonal `Q` 和 affine `p`。

## 6. Critic 网络

RLPD-style vectorized ensemble：

- 10 个 Q heads。
- 每个 head 的输入为 `[lifted_state_78, residual_action_18]`，共 96 维。
- 每个 head：`96 -> 256 -> LayerNorm -> ReLU -> 256 -> LayerNorm -> ReLU -> 1`。
- 两个 hidden layers，width=256；LayerNorm 参数逐 head 独立。
- Critic 共 919,050 parameters；另有同结构 target critic。
- Actor 计算 advantage 时对 10 个 heads 取 **ensemble mean**。
- Bellman target 每次随机抽 2/10 target heads，再取两者 minimum（REDQ target）。
- Target Polyak coefficient `tau=0.005`，discount `gamma=0.99`。
- Critic-head loss reduction 为 mean。

## 7. AWAC 更新与 optimizer

### Effective offline update

```text
data_Q   = mean_10_heads Q(s, a_dataset)
policy_Q = mean_10_heads Q(s, a_policy_sample)
advantage = data_Q - policy_Q
weight = min(exp(advantage / beta), 100)
beta = 1.0

actor_loss = -mean(weight * log pi(a_dataset | s))
```

- AWAC selectivity mode：`all`，所有 batch rows 均用于 actor loss。
- Actor update interval：每 2 个 critic updates 更新一次。
- Offline critic UTD：实际为 1/update；配置中的 `online_utd=20` 只适用于 online，本次没有 online。
- CQL：关闭。
- Critic entropy backup：关闭。
- Q-anchor、p-anchor、offline BC auxiliary weight：全部为 0。
- Batch size：256，完全来自 offline dataset；`offline_replay_ratio=0.5` 是 online mixed replay 配置，本次 offline-only 阶段不生效。

Optimizer 均为 PyTorch Adam（默认 betas/eps），并做 global gradient-norm clipping：

| 参数组 | Learning rate |
|---|---:|
| Actor | 3e-5 |
| Critic | 3e-5 |
| Temperature config field | 3e-5 |
| Gradient clip norm | 10.0 |

AWAC actor 路径不使用 SAC entropy actor objective，也不更新 temperature；因此 `initial_temperature=1`、`target_entropy=-9` 和 temperature LR 被保存在通用配置中，但不是这次 AWAC offline actor update 的有效优化项。

实现复现注意：当前 `experiments/dmc/o2o/learner.py` 的 AWAC actor update 在 gradient clipping 前连续调用了两次 `actor_loss.backward()`。这意味着实际训练忠实包含“双 backward 累积梯度”行为；由于之后执行 norm clipping，其影响取决于当步是否触发 clip。复现该 checkpoint 时不能在不声明的情况下悄然改掉这一点。

## 8. Reward 与评估协议

三节点 XYZ tracking error：

```text
e_joint = sqrt(0.2*e_node0^2 + 0.2*e_node1^2 + 0.6*e_tip^2)
```

单步 hybrid reward：

```text
r_sparse = 0.5 * I(e_joint <= 0.0025 m)
r_dense  = 0.5 * exp(-e_joint / 0.01 m)
r = r_sparse + r_dense
```

因此理论单步最大 reward 为 1。Tip RMSE/P95 只计算末端节点 XYZ 欧氏距离；return 使用三个节点的联合误差。

| 项目 | 配置 |
|---|---:|
| Deterministic evaluation episodes | 10 |
| Offline evaluation interval | 2,500 updates |
| Milestone/checkpoint cadence | 2,500 updates |
| Metrics log interval | 500 updates |
| Best selection | 最大 deterministic return |

固定确定性任务使 10 episodes 的结果完全一致，因此该 checkpoint 的 `return_std=0`。

## 9. 选定 checkpoint 的 authority/数值诊断

| 指标 | 数值 |
|---|---:|
| `abs(Delta_D)/D_scale` P50/P95/P99 | 0.9378 / 1.0000 / 1.0000 |
| D utilization fraction >0.99 | 38.55% |
| Physical `abs(Delta_D)` P50/P95/max | 37.22 / 150.00 / 150.00 mm |
| Q/Qbase P50/P95/max | 6.0467 / 6.04965 / 6.04965 |
| Q within top 1% of upper range | 77.25% |
| `abs(d_action)` P50/P95/max | 0.009178 / 0.010000 / 0.010000 |
| `abs(d_action)>0.0099` | 33.83% |
| Applied residual P95/max | 0.011383 / 0.014984 |
| Physical saturation | 0% |
| Solver projected-gradient relative mean/P95 | 0.004819 / 0.006624 |

该结果虽然按 return 胜出，但 D、Q 和 action cost center 均有显著贴边；它是选定配置，不代表有证据继续扩大 authority。

## 10. Provenance 与复现入口

原始命令：

```text
/root/autodl-tmp/AC-MPC/runs/o2o/diagnostics/no_xref_authority_refinement_20260902/runs/E7_D500_Q18/command.txt
SHA256 fec7dab6b64fdf2ef1111c12a41b2cc0ed8bb0df8243ba28f57f242340fba8b8
```

关键源文件 SHA256：

```text
beb89b5fe65e5cb2e9698d021e82db85b02bf84d648c4b98c1c2866998c2faa0  manisoft_port/antmaze_ac/rl/time_implicit_xyz_kmpc_actor.py
92f46fc8133e699d674cfe287cf38978df861507dd92a1c58aa1d03f9d554f3c  manisoft_port/scripts/train_manisoft_circle_time_awac_kmpc.py
b8439d46ef49603da27a82f585268b967ccb413d9eb8940ec9b9e5feab6fe856  experiments/dmc/o2o/learner.py
8baa593493f16c455212780c7fafe60cb2ff63f2ef9d8c3d2409c69e38815c3b  experiments/dmc/o2o/networks.py
2a49ecba186fb0259776224cbd27fdf3a1f5c954e11c3e3364800d14513c53d9  manisoft_port/antmaze_ac/envs/circle_reward.py
```

同目录保留：

```text
best.pt             # 通用 trainer 历史 best 规则产物
best_return.pt      # 本轮正式选定权重
latest.pt           # 20k recovery learner/replay state
offline_020000.pt   # 20k milestone
```

后续部署、render 或 online bootstrap 应明确使用 `best_return.pt`，并验证 online/deployment @0 能复现本文件记录的 deterministic evaluation 后再继续。

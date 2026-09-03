# ManiSoft circle：AWAC-KMPC 最佳 offline → online 结果整理

## 一句话结论

当前 AWAC-KMPC 画圆的最佳链路是：

```text
同一组 offline 训练计划（总计跑到 20k）
→ 选择 offline@10k checkpoint，而不是退化后的 offline@20k
→ A3 online：actor replay = 0% offline + 100% online
→ online@9k 得到最佳 tip RMSE = 6.831 mm
```

因此需要区分：

- “对应的 offline 20k”是该组离线训练的完整 0–20k 曲线；
- A3 实际初始化权重是其中的 `offline_010000.pt`；
- `offline_020000.pt` 已退化到 9.356 mm，没有用于 A3。

## 1. Offline AWAC-KMPC：完整 20k 曲线

Run：

```text
runs/o2o/diagnostics/dmax01_awac_kmpc_off20k_on10k_seed20260851
```

| Offline update | Tip RMSE | Tip P95 | Return | 说明 |
|---:|---:|---:|---:|---|
| 0 | 13.747 mm | 23.323 mm | 206.34 | 初始控制器 |
| 5k | 7.756 mm | **11.997 mm** | **450.22** | 最佳 P95 / return |
| 10k | **6.982 mm** | 13.292 mm | 445.95 | 最佳 RMSE；选作 online source |
| 15k | 7.685 mm | 16.064 mm | 429.26 | 开始明显退化 |
| 20k | 9.356 mm | 16.582 mm | 333.88 | offline final，未用于最佳 online 链路 |

Offline 阶段的核心现象：前 5k–10k 学到有效改善，继续优化到 15k–20k 后明显退化。以 tip RMSE 为 checkpoint 选择标准，最佳离线权重为：

```text
runs/o2o/diagnostics/dmax01_awac_kmpc_off20k_on10k_seed20260851/offline_010000.pt
SHA256: 6b4faa560a2bef513c3845f747f467a9965039e6da856100e5d4fc604f5b4138
```

## 2. A3 online：actor 100% online replay

Run：

```text
runs/o2o/diagnostics/actor_replay_composition_screen_rerun_20260902/
  a3_actor00_normal_warmup
```

配置要点：

- Source actor：上述 `offline_010000.pt`，online@0 精确复现 6.982 mm；
- 0–5k：actor 冻结，采集 source-policy online buffer并更新 critic；
- 5k 后：actor 解冻；
- Actor batch：0% offline + 100% online；
- Critic batch：50% offline + 50% online；
- Critic：10-head LayerNorm ensemble；
- UTD=20，actor LR=`1e-6`，critic LR=`5e-5`，actor update interval=4；
- `backup_entropy=false`，actor entropy 关闭，CQL 关闭；
- `d_action_max=0.01`，H=5，solver iterations=5；
- 评估为 10-episode deterministic evaluation。

### 完整 online 曲线

| Online step | Tip RMSE | Tip P95 | Return | 阶段 |
|---:|---:|---:|---:|---|
| 0 | 6.982 mm | 13.292 mm | 445.95 | offline@10k source |
| 1k | 6.982 mm | 13.292 mm | 445.95 | actor frozen |
| 2k | 6.982 mm | 13.292 mm | 445.95 | actor frozen |
| 3k | 6.982 mm | 13.292 mm | 445.95 | actor frozen |
| 4k | 6.982 mm | 13.292 mm | 445.95 | actor frozen |
| 5k | 6.982 mm | 13.292 mm | 445.95 | warmup结束 |
| 6k | 6.904 mm | 12.640 mm | 448.18 | actor active 1k |
| 7k | 6.859 mm | 12.528 mm | **449.77** | 最佳 return |
| 8k | 6.846 mm | 12.385 mm | 449.22 | 稳定改善 |
| 9k | **6.831 mm** | 12.305 mm | 447.11 | 最佳 RMSE |
| 10k | 6.843 mm | **12.252 mm** | 442.28 | online final / 最佳 P95 |

最佳 RMSE checkpoint：

```text
runs/o2o/diagnostics/actor_replay_composition_screen_rerun_20260902/
  a3_actor00_normal_warmup/online_009000.pt
```

## 3. Offline source 与最佳 online 的直接比较

| 节点 | Tip RMSE | Tip P95 | Return |
|---|---:|---:|---:|
| Offline@10k / Online@0 | 6.982 mm | 13.292 mm | 445.95 |
| A3 Online@9k | **6.831 mm** | **12.305 mm** | 447.11 |
| A3 Online@10k | 6.843 mm | **12.252 mm** | 442.28 |

A3@9k 相对 offline@10k：

- Tip RMSE 降低 `0.151 mm`，约 `2.16%`；
- Tip P95 降低 `0.987 mm`，约 `7.43%`；
- Return 增加约 `1.16`。

需要注意，三个指标的最佳节点不完全一致：

- 最佳 tip RMSE：online@9k，6.831 mm；
- 最佳 tip P95：online@10k，12.252 mm；
- 最佳 return：online@7k，449.77。

如果后续将“主要方法最佳 checkpoint”按 tip RMSE 定义，应统一使用 A3 `online_009000.pt`；如果更强调尾部误差，则可单独报告 online@10k，但不要把它误写为 RMSE 最佳节点。

## 4. 当前可采用的正式表述

> AWAC-KMPC 在 20k 离线训练计划中于 10k update 达到最佳离线 tip RMSE 6.982 mm，之后继续离线更新发生退化。以该 offline@10k checkpoint 初始化 online learning，并将 AWAC actor replay 调整为 100% online、critic 保持 50/50 offline-online replay 后，策略在 online@9k 达到最佳 tip RMSE 6.831 mm 和 tip P95 12.305 mm；online@10k 的 tip RMSE 为 6.843 mm，tip P95 进一步降低至 12.252 mm。

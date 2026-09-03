# ManiSoft circle：E7 Frozen Base + Full-Capacity Online Residual Actor

## 1. Source 与方法

- source checkpoint：`/root/autodl-tmp/AC-MPC/runs/o2o/diagnostics/no_xref_authority_refinement_20260902/runs/E7_D500_Q18/best_return.pt`
- source checkpoint SHA256：`b2eca8b15f95a3a3c834b9c36055ecb9f9d49a280285e1542fc7f7921b376eac`
- source deterministic return：`595.207677 = 252 sparse + 343.207677 dense`
- base actor 全程冻结；online actor 为 `78->256(GELU)->256(GELU)->180`
- online hidden trunk 从 E7 复制，最终 head 精确 zero-init，log_std 冻结
- residual 在 decoded cost-map coordinate 叠加：D 20%、Q log ±0.20、action centre ±0.001
- no xref，H=5，solver=5，Koopman/u_ff/reward/beta/replay/UTD 均保持协议不变

## 2. Online@0 equality

32-state probe 上 raw residual、decoded residual、最终 Q/p 与 first MPC action 对 E7 的差异均为 0。单-seed deterministic return 为 594.702476；与该评估 seed 下 source 完全相同。base actor SHA 在所有 evaluation 中保持不变。

## 3. Shared 0~5k critic warmup

| online step | return | sparse | dense | joint RMSE (mm) | tip RMSE (mm) | tip P95 (mm) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 594.702 | 251.500 | 343.202 | 9.103 | 10.867 | 24.147 |
| 1000 | 594.702 | 251.500 | 343.202 | 9.103 | 10.867 | 24.147 |
| 2000 | 594.702 | 251.500 | 343.202 | 9.103 | 10.867 | 24.147 |
| 3000 | 594.702 | 251.500 | 343.202 | 9.103 | 10.867 | 24.147 |
| 4000 | 594.702 | 251.500 | 343.202 | 9.103 | 10.867 | 24.147 |
| 5000 | 594.702 | 251.500 | 343.202 | 9.103 | 10.867 | 24.147 |

## 4. R0-R3 screen best

| group | best step | return | sparse | dense | tip RMSE (mm) | tip P95 (mm) |
|---|---:|---:|---:|---:|---:|---:|
| R0 | 5000 | 594.702 | 251.500 | 343.202 | 10.867 | 24.147 |
| R1 | 7500 | 603.877 | 257.000 | 346.877 | 11.003 | 25.385 |
| R2 | 7500 | 608.069 | 261.000 | 347.069 | 11.075 | 25.651 |
| R3 | 7500 | 605.328 | 259.000 | 346.328 | 11.120 | 26.043 |

按协议选择的 winner：**R2**。

## 5. Winner clean formal 5k~10k

| online step | return | sparse | dense | joint RMSE (mm) | tip RMSE (mm) | tip P95 (mm) |
|---:|---:|---:|---:|---:|---:|---:|
| 5000 | 594.702 | 251.500 | 343.202 | 9.103 | 10.867 | 24.147 |
| 5500 | 596.928 | 253.000 | 343.928 | 9.129 | 10.897 | 24.413 |
| 6000 | 600.216 | 255.000 | 345.216 | 9.173 | 10.950 | 24.835 |
| 6500 | 602.826 | 257.000 | 345.826 | 9.210 | 10.994 | 25.128 |
| 7000 | 605.063 | 258.500 | 346.563 | 9.247 | 11.038 | 25.410 |
| 7500 | 608.069 | 261.000 | 347.069 | 9.278 | 11.075 | 25.651 |
| 8000 | 609.872 | 262.500 | 347.372 | 9.301 | 11.102 | 25.810 |
| 8500 | 612.399 | 265.000 | 347.399 | 9.320 | 11.126 | 25.926 |
| 9000 | 613.280 | 266.000 | 347.280 | 9.328 | 11.136 | 25.973 |
| 9500 | 623.828 | 276.500 | 347.328 | 9.348 | 11.159 | 26.101 |
| 10000 | 625.613 | 278.500 | 347.113 | 9.368 | 11.183 | 26.234 |

formal best：online@10000，return=625.613，sparse=278.500，dense=347.113，tip RMSE/P95=11.183/26.234 mm。

相对附件 source return 595.207677 的变化：**+30.405**。

## 6. Residual authority、policy drift 与控制健康

| metric | formal best |
|---|---:|
| `online_residual_D_utilization_p50` | 0.19347813 |
| `online_residual_D_utilization_p95` | 0.5966278 |
| `online_residual_D_utilization_max` | 0.7299239 |
| `online_residual_delta_q_abs_p50` | 0.016658532 |
| `online_residual_delta_q_abs_p95` | 0.085065782 |
| `online_residual_delta_q_abs_max` | 0.1501026 |
| `online_residual_delta_d_abs_p50` | 0 |
| `online_residual_delta_d_abs_p95` | 0 |
| `online_residual_delta_d_abs_max` | 0 |
| `implicit_policy_kl_from_shared_mean` | 0.0015165849 |
| `online_residual_trunk_parameter_update_norm` | 0.51827417 |
| `online_residual_D_head_parameter_update_norm` | 0.1349598 |
| `online_residual_Q_head_parameter_update_norm` | 0.092818432 |
| `online_residual_action_head_parameter_update_norm` | 0 |
| `base_actor_parameter_change` | 0 |
| `action_saturation_fraction` | 0 |
| `kmpc_projected_gradient_relative_mean` | 0.0047387537 |

base actor hash：`beaa45d0ee2a9c73a399fb1fca040e5eb4c23178d6a5c0029883d1a1519d4b44`；parameter change=`0.0`。

## 7. Critic/AWAC/gradient 时间序列

| online step | critic_loss | critic_q_mean | target_q_mean | advantage_mean | advantage_std | advantage_weight_mean | advantage_weight_max | actor_grad_norm | online_residual_shared_trunk_grad_norm | online_residual_D_head_grad_norm | online_residual_Q_head_grad_norm | online_residual_action_head_grad_norm |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5500 | 0.2803 | 0 | 13.222 | 0.006931 | 0.1161 | 1.0138 | 1.4781 | 0.056122 | 0.00094501 | 0.039926 | 0.03943 | 0 |
| 6000 | 0.21969 | 0 | 12.239 | 0.0073782 | 0.14286 | 1.0178 | 1.6152 | 0.11405 | 0.0011408 | 0.07637 | 0.084691 | 0 |
| 6500 | 0.25397 | 0 | 12.303 | -0.0070271 | 0.16534 | 1.0067 | 1.8019 | 0.11805 | 0.0031683 | 0.087783 | 0.078872 | 0 |
| 7000 | 0.21729 | 0 | 12.681 | 0.0026819 | 0.14967 | 1.0139 | 1.4826 | 0.096266 | 0.0036871 | 0.070228 | 0.065739 | 0 |
| 7500 | 0.24144 | 0 | 12.593 | -0.0095613 | 0.15937 | 1.0035 | 1.8749 | 0.064627 | 0.0038083 | 0.044906 | 0.04632 | 0 |
| 8000 | 0.21657 | 0 | 12.405 | 0.013741 | 0.16513 | 1.0279 | 1.7801 | 0.053761 | 0.0049915 | 0.031714 | 0.043122 | 0 |
| 8500 | 0.18071 | 0 | 12.205 | -0.0068382 | 0.17474 | 1.0084 | 1.7475 | 0.048894 | 0.003794 | 0.036257 | 0.032583 | 0 |
| 9000 | 0.19427 | 0 | 11.896 | 0.0090733 | 0.20841 | 1.0334 | 2.8413 | 0.16404 | 0.004294 | 0.11944 | 0.11237 | 0 |
| 9500 | 0.19279 | 0 | 11.586 | -0.011535 | 0.16415 | 1.002 | 1.6137 | 0.058616 | 0.0061751 | 0.036031 | 0.045819 | 0 |
| 10000 | 0.20705 | 0 | 11.401 | 0.0067708 | 0.19042 | 1.0258 | 2.242 | 0.070407 | 0.0064553 | 0.047755 | 0.051333 | 0 |

## 8. Checkpoints

- return-best：`/root/autodl-tmp/AC-MPC/runs/o2o/diagnostics/E7_full_capacity_online_residual_20260903/E7_fullres_formal_R2/best_online_return.pt`
- final：`/root/autodl-tmp/AC-MPC/runs/o2o/diagnostics/E7_full_capacity_online_residual_20260903/E7_fullres_formal_R2/online_010000.pt`

## 9. 结论

- winner：**R2**
- 推荐 online residual composition：**保留 full-capacity D+Q residual**
- 是否超过 595.21：**是**
- 是否超过 600/610/620/630：**True/True/True/False**
- sparse 是否超过 252：**True**

本轮按要求到此停止，不自动启动其它 sweep。

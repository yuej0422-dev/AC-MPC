# ManiSoft Circle 正式 Offline/Online 结果与复现手册

## 1. 正式实验定义

本目录是 ManiSoft Circle 最终正式归档，固定为：

- 训练 seed：`20260851`、`20260852`、`20260853`、`20260854`、`20260855`
- Offline：`10,000` updates
- Online：`15,000` environment steps
- 评估与 checkpoint 间隔：`2,500`
- 方法：`AWAC`、`Cal-QL`、`IQL`、`AWAC-lift`、`AWAC-raw`、`RLPD`、`AWAC-KMPC`
- 每个固定节点使用 1 个确定性 1000-step episode 评估
- 表中 `std` 是 5 个训练 seed 间的样本标准差

归档结构：

```text
manisoft_circle_5seed7_offline10k_online15k_20260905/
├── REPRODUCTION.md
├── results.csv
├── offline/seed<seed>/<method>/
└── online/seed<seed>/<method>/
```

`RLPD` 按标准 online-only 协议运行，没有 offline 预训练结果。因此 offline
为 `5 seed × 6 method = 30` 组，online 为 `5 seed × 7 method = 35` 组。

正式比较只能读取：

- Offline：`offline_010000.pt` 与 `evaluation_offline_010000.json`
- Online：`online_015000.pt` 与 `evaluation_online_015000.json`

不得用 `best.pt`、`latest.pt`、`offline_020000.pt` 或中间 online checkpoint
替代固定节点结果。每个方法目录内的 `command.txt`、`run.json`、
`metrics.jsonl`、checkpoint 和 evaluation JSON 是该次运行的原始记录。

## 2. 正式结果

### Offline @ 10k

| 方法 | Return | Tip RMSE | Tip P95 | Success@2.5mm |
|---|---:|---:|---:|---:|
| AWAC-KMPC | **569.90 ± 10.27** | **10.69 ± 0.17 mm** | 23.99 ± 0.65 mm | 35.20 ± 2.69% |
| IQL | 480.18 ± 107.51 | 13.11 ± 1.65 mm | **23.75 ± 4.77 mm** | 28.06 ± 11.42% |
| AWAC-lift | 316.76 ± 210.46 | 37.95 ± 32.00 mm | 75.33 ± 56.54 mm | 17.66 ± 14.44% |
| AWAC | 294.67 ± 150.43 | 35.69 ± 39.92 mm | 66.11 ± 67.54 mm | 16.64 ± 12.60% |
| AWAC-raw | 268.92 ± 136.23 | 36.99 ± 38.81 mm | 69.55 ± 67.08 mm | 14.72 ± 10.19% |
| Cal-QL | 0.49 ± 0.19 | 183.76 ± 13.47 mm | 254.64 ± 17.64 mm | 0% |
| RLPD | N/A | N/A | N/A | N/A |

### Online @ 15k

| 方法 | Return | Tip RMSE | Tip P95 | Success@2.5mm |
|---|---:|---:|---:|---:|
| AWAC-KMPC (C3) | **607.18 ± 18.67** | **10.77 ± 0.20 mm** | **25.56 ± 0.43 mm** | **42.34 ± 5.81%** |
| IQL | 519.27 ± 147.91 | 14.27 ± 3.05 mm | 31.39 ± 5.88 mm | 33.96 ± 18.40% |
| AWAC | 298.34 ± 168.73 | 36.13 ± 40.37 mm | 66.36 ± 67.49 mm | 17.78 ± 16.09% |
| RLPD | 236.64 ± 31.21 | 14.99 ± 1.08 mm | 28.43 ± 1.82 mm | 3.40 ± 3.15% |
| AWAC-lift | 228.85 ± 143.38 | 44.56 ± 32.32 mm | 95.16 ± 64.16 mm | 7.10 ± 7.40% |
| AWAC-raw | 176.15 ± 82.22 | 38.97 ± 40.34 mm | 72.83 ± 67.08 mm | 3.74 ± 2.31% |
| Cal-QL | 11.68 ± 9.93 | 104.98 ± 15.92 mm | 158.99 ± 14.32 mm | 0.06 ± 0.13% |

所有逐 seed 原始数值、evaluation 路径和 checkpoint 路径见根目录
`results.csv`。

## 3. 固定任务与依赖

仓库与环境：

```bash
export REPO=/root/autodl-tmp/AC-MPC
export MANISOFT_ROOT=/root/autodl-tmp/ManiSoft
export PYTHON=$REPO/.venv/bin/python
export TRAINER=$REPO/manisoft_port/scripts/train_manisoft_circle_time_awac_kmpc.py
export BUNDLE=$REPO/runs/o2o/formal/manisoft_circle_5seed7_offline10k_online15k_20260905
export DATASET=$REPO/runs/o2o/diagnostics/dataset_rebuild/manisoft_circle_curated_200k_E7_canonical.npz
export FF=$REPO/runs/o2o/probes/manisoft_circle_phase15_degraded/policy.npz
export KOOPMAN=$REPO/work_dirs/manisoft_abs_u06_1132ep_h0_formal_walker_512/koopman_h0_formal_walker_loss/koopman_history/best_validation.pt
export REFERENCE=$REPO/work_dirs/manisoft_circle_r010_benchmark_klqr/trajectory.npz
export SCENARIO=$MANISOFT_ROOT/configs/demo_elastica_fast.yaml
```

固定输入：

| 输入 | 路径 | SHA256 |
|---|---|---|
| Offline buffer | `runs/o2o/diagnostics/dataset_rebuild/manisoft_circle_curated_200k_E7_canonical.npz` | 当前 canonical copy：`b3f68047...b7379` |
| Feedforward | `runs/o2o/probes/manisoft_circle_phase15_degraded/policy.npz` | `515adce9...3ffe` |
| Koopman | `work_dirs/manisoft_abs_u06_1132ep_h0_formal_walker_512/koopman_h0_formal_walker_loss/koopman_history/best_validation.pt` | `d97234dd...19b9d` |
| Reference | `work_dirs/manisoft_circle_r010_benchmark_klqr/trajectory.npz` | `b297e0a8...dfb5` |
| Scenario | `$MANISOFT_ROOT/configs/demo_elastica_fast.yaml` | `f7cd98f2...fa75` |

历史运行的 `run.json` 对同一路径 buffer 记录了旧的字节级 SHA
`e374d27d...48a4`；当前设备以 `CIRCLE_RL_CANONICAL_BUFFER.md` 固定的
canonical copy 为准，不另外切换数据集。

代码版本：

- AC-MPC base commit：`d6852120c898cfb65c19468a94b123e44261562d`
- ManiSoft commit：`4e02bb87962604c6ab6abf06f3f273a1c49c1270`
- PyElastica commit：`d55e86dc0bd1b46b3b8ca88cd460b71392f7c655`
- LieGroups commit：`2d20ac60419fa7316f18906defd39d3623f4e5e7`
- Python `3.12.3`，PyTorch `2.5.1+cu124`，CUDA runtime `12.4`
- 参考设备：NVIDIA A100-PCIE-40GB，driver `550.90.07`

任务语义：

- observation：45D physical state + 1D normalized task time `tau`
- policy observation 不含 `xref` 和 `u_ff`
- action：18D residual，实际作用为
  `physical_u = clip(u_ff(t) + residual_u, -0.5, 0.5)`
- episode：1000 steps；control dt `0.02 s`；physics dt `0.0002 s`
- reward：`0.5 × sparse + 0.5 × dense`
- dense scale：`0.01 m`；三节点 XYZ 权重 `[0.2, 0.2, 0.6]`
- success radius：`2.5 mm`

## 4. Offline @ 10k 协议

公共参数：

- batch size `256`，replay capacity `10,000`
- actor/critic/temperature LR 均为 `3e-5`
- discount `0.99`，gradient clip norm `10`
- evaluation/checkpoint interval `2,500`
- backup entropy 关闭
- KMPC horizon `5`，solver iterations `5`
- `kmpc_log_std_init=-3.5`，`kmpc_log_std_max=-3.0`

方法差异：

| 目录名 | CLI method | 表征/Actor | Critic profile | UTD | Actor interval | 其他 |
|---|---|---|---|---:|---:|---|
| AWAC-KMPC | `AWAC-KMPC` | frozen Koopman lift + implicit KMPC | 10-head LayerNorm | 20 | 2 | no-xref，D ratio 5，Q log upper 1.8，velocity cost 0.05，`d_action_max=0.01` |
| AWAC | `AWAC` | raw + MLP | 2 critics/plain | 1 | 1 | AWAC temperature 1，weight cap 100 |
| Cal_QL | `Cal-QL` | raw + MLP | ExORL CQL, width 1024 | 1 | 1 | 3 CQL actions，CQL weight 0.01，target tau 0.01 |
| IQL | `IQL` | raw + MLP | 2 critics/plain | 1 | 1 | expectile 0.7，temperature 3，weight cap 100 |
| AWAC-lift | `AWAC-lift` | frozen Koopman lift + MLP | 10-head LayerNorm | 20 | 1 | 无 KMPC |
| AWAC-raw | `AWAC-raw` | raw + MLP | 10-head LayerNorm | 20 | 1 | 无 lift、无 KMPC |

复现时对每个 `SEED × LABEL` 运行一次，输出到新目录，不覆盖正式结果：

```bash
SEED=20260851                 # 20260851 ... 20260855
LABEL=AWAC-KMPC               # AWAC-KMPC/AWAC/Cal_QL/IQL/AWAC_lift/AWAC_raw
OUT=$REPO/runs/o2o/reproductions/circle/offline/seed$SEED/$LABEL

DATASET=$REPO/runs/o2o/diagnostics/dataset_rebuild/manisoft_circle_curated_200k_E7_canonical.npz
FF=$REPO/runs/o2o/probes/manisoft_circle_phase15_degraded/policy.npz
KOOPMAN=$REPO/work_dirs/manisoft_abs_u06_1132ep_h0_formal_walker_512/koopman_h0_formal_walker_loss/koopman_history/best_validation.pt
REFERENCE=$REPO/work_dirs/manisoft_circle_r010_benchmark_klqr/trajectory.npz
SCENARIO=$MANISOFT_ROOT/configs/demo_elastica_fast.yaml

METHOD=$LABEL; UTD=1; INTERVAL=1; NUM_ENVS=5; WORKERS=5; EXTRA=()
case "$LABEL" in
  AWAC-KMPC)
    UTD=20; INTERVAL=2
    EXTRA=(--implicit-xyz-no-xref --implicit-xyz-velocity-cost-scale 0.05
      --implicit-xyz-d-scale-ratio 5.0 --implicit-xyz-q-log-upper 1.8
      --action-cost-center-limit 0.01)
    ;;
  AWAC) METHOD=AWAC ;;
  Cal_QL) METHOD=Cal-QL; NUM_ENVS=1; WORKERS=1 ;;
  IQL) METHOD=IQL ;;
  AWAC_lift) METHOD=AWAC-lift; UTD=20 ;;
  AWAC_raw) METHOD=AWAC-raw; UTD=20 ;;
esac

test ! -e "$OUT" || { echo "refusing to overwrite: $OUT" >&2; exit 1; }
mkdir -p "$OUT"
CMD=(env
  PYTHONPATH=$REPO/manisoft_port:$REPO:$MANISOFT_ROOT:$MANISOFT_ROOT/third_party/pyelastica:$MANISOFT_ROOT/third_party/liegroups
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5
  "$PYTHON" "$TRAINER" "${EXTRA[@]}"
  --method "$METHOD" --feedforward "$FF" --dataset "$DATASET"
  --koopman "$KOOPMAN" --scenario "$SCENARIO" --reference "$REFERENCE"
  --output "$OUT" --offline-updates 10000 --online-steps 0
  --offline-eval-interval 2500 --online-eval-interval 2500
  --checkpoint-save-interval 2500 --log-interval-updates 500
  --eval-episodes 1 --batch-size 256 --replay-capacity 10000
  --actor-learning-rate 3e-5 --critic-learning-rate 3e-5
  --temperature-learning-rate 3e-5 --offline-replay-ratio 0.5
  --online-warmup-steps 0 --kmpc-horizon 5 --kmpc-solver-iterations 5
  --kmpc-log-std-init -3.5 --kmpc-log-std-max -3.0
  --reward-mode hybrid --sparse-reward-weight 0.5 --dense-reward-weight 0.5
  --dense-reward-scale-m 0.01 --actor-update-interval "$INTERVAL"
  --online-utd "$UTD" --num-envs "$NUM_ENVS" --env-workers "$WORKERS"
  --online-cql-mode off --disable-backup-entropy --seed "$SEED" --device cuda)
printf '%q ' "${CMD[@]}" > "$OUT/command.txt"
tmux new-session -d -s "circle_off10k_${LABEL}_${SEED}" -c "$REPO" \
  "$(printf '%q ' "${CMD[@]}") >$(printf '%q' "$OUT/tmux.log") 2>&1"
```

## 5. Online @ 15k：六个基线方法

共同配置：

- actor LR `2e-6`，critic LR `5e-5`，temperature LR `1e-6`
- online budget `15k`，评估/checkpoint 每 `2.5k`
- AWAC、Cal-QL、IQL、AWAC-lift、AWAC-raw 从相同 seed 的 offline@10k 启动
- RLPD 无 offline bootstrap，随机初始化后先收集 `5k` online warmup
- AWAC/Cal-QL/IQL：UTD `1`，actor interval `1`
- AWAC-lift/AWAC-raw：UTD `20`，actor interval `2`，actor replay `100% online`
- RLPD：UTD `20`，actor interval `1`
- Cal-QL 使用 `online_cql_mode=all_valid_mc`，单环境串行评估
- 其他方法 online CQL 关闭；backup entropy 关闭

复现模板：

```bash
SEED=20260851
LABEL=AWAC                    # AWAC/Cal_QL/IQL/AWAC_lift/AWAC_raw/RLPD
OUT=$REPO/runs/o2o/reproductions/circle/online/seed$SEED/$LABEL

METHOD=$LABEL; OFFLINE_UPDATES=10000; UTD=1; INTERVAL=1; WARMUP=0
NUM_ENVS=5; WORKERS=5; CQL_MODE=off; EXTRA=(--disable-backup-entropy)
BOOTSTRAP=$BUNDLE/offline/seed$SEED/$LABEL/offline_010000.pt
case "$LABEL" in
  AWAC) METHOD=AWAC ;;
  Cal_QL) METHOD=Cal-QL; NUM_ENVS=1; WORKERS=1; CQL_MODE=all_valid_mc ;;
  IQL) METHOD=IQL ;;
  AWAC_lift)
    METHOD=AWAC-lift; UTD=20; INTERVAL=2
    EXTRA+=(--actor-offline-replay-ratio 0.0)
    ;;
  AWAC_raw)
    METHOD=AWAC-raw; UTD=20; INTERVAL=2
    EXTRA+=(--actor-offline-replay-ratio 0.0)
    ;;
  RLPD)
    METHOD=RLPD; OFFLINE_UPDATES=0; UTD=20; WARMUP=5000; BOOTSTRAP=
    ;;
esac

test ! -e "$OUT" || { echo "refusing to overwrite: $OUT" >&2; exit 1; }
mkdir -p "$OUT"
CMD=(env
  PYTHONPATH=$REPO/manisoft_port:$REPO:$MANISOFT_ROOT:$MANISOFT_ROOT/third_party/pyelastica:$MANISOFT_ROOT/third_party/liegroups
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5
  "$PYTHON" "$TRAINER" --method "$METHOD"
  --feedforward "$FF" --dataset "$DATASET" --koopman "$KOOPMAN"
  --scenario "$SCENARIO" --reference "$REFERENCE" --output "$OUT"
  --offline-updates "$OFFLINE_UPDATES" --online-steps 15000
  --offline-eval-interval 2500 --online-eval-interval 2500
  --checkpoint-save-interval 2500 --log-interval-updates 500
  --eval-episodes 1 --batch-size 256 --replay-capacity 15000
  --actor-learning-rate 2e-6 --critic-learning-rate 5e-5
  --temperature-learning-rate 1e-6 --actor-update-interval "$INTERVAL"
  --online-utd "$UTD" --offline-replay-ratio 0.5
  --online-warmup-steps "$WARMUP" --kmpc-horizon 5 --kmpc-solver-iterations 5
  --kmpc-log-std-init -3.5 --kmpc-log-std-max -3.0
  --reward-mode hybrid --sparse-reward-weight 0.5 --dense-reward-weight 0.5
  --dense-reward-scale-m 0.01 --num-envs "$NUM_ENVS" --env-workers "$WORKERS"
  --online-cql-mode "$CQL_MODE" --seed "$SEED" --device cuda "${EXTRA[@]}")
if [[ -n "$BOOTSTRAP" ]]; then
  CMD+=(--bootstrap-checkpoint "$BOOTSTRAP" --bootstrap-allow-schedule-change)
fi
printf '%q ' "${CMD[@]}" > "$OUT/command.txt"
tmux new-session -d -s "circle_on15k_${LABEL}_${SEED}" -c "$REPO" \
  "$(printf '%q ' "${CMD[@]}") >$(printf '%q' "$OUT/tmux.log") 2>&1"
```

## 6. Online @ 15k：AWAC-KMPC C3

C3 是最终采用的 KMPC online 协议：

- 从同 seed 的 AWAC-KMPC offline@10k 仅加载 actor
- 新建 10-head LayerNorm critic
- 前 `5k` online steps 只训练 critic
- actor LR `2e-6`，critic LR `5e-5`，critic UTD `20`
- actor interval `2`
- critic replay offline/online `50/50`；actor replay `100% online`
- full-capacity implicit-xyz `DQ` residual
- horizon `5`，solver iterations `5`，`d_action_max=0.01`
- backup/actor entropy 与 online CQL 均关闭

复现模板：

```bash
SEED=20260851
LABEL=AWAC-KMPC
OUT=$REPO/runs/o2o/reproductions/circle/online/seed$SEED/$LABEL
BOOTSTRAP=$BUNDLE/offline/seed$SEED/AWAC-KMPC/offline_010000.pt

test ! -e "$OUT" || { echo "refusing to overwrite: $OUT" >&2; exit 1; }
mkdir -p "$OUT"
CMD=(env
  PYTHONPATH=$REPO/manisoft_port:$REPO:$MANISOFT_ROOT:$MANISOFT_ROOT/third_party/pyelastica:$MANISOFT_ROOT/third_party/liegroups
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10
  "$PYTHON" "$TRAINER"
  --implicit-xyz-no-xref --full-capacity-online-residual --full-residual-channels DQ
  --implicit-xyz-velocity-cost-scale 0.05 --implicit-xyz-d-scale-ratio 5.0
  --implicit-xyz-q-log-upper 1.8 --action-cost-center-limit 0.01
  --method AWAC-KMPC --feedforward "$FF" --dataset "$DATASET"
  --koopman "$KOOPMAN" --scenario "$SCENARIO" --reference "$REFERENCE"
  --output "$OUT" --offline-updates 0 --online-steps 15000
  --offline-eval-interval 2500 --online-eval-interval 2500
  --checkpoint-save-interval 2500 --log-interval-updates 500
  --eval-episodes 1 --batch-size 256 --replay-capacity 10000
  --actor-learning-rate 2e-6 --critic-learning-rate 5e-5
  --temperature-learning-rate 1e-6 --online-utd 20 --online-warmup-steps 0
  --online-critic-only-steps 5000 --actor-update-interval 2
  --offline-replay-ratio 0.5 --actor-offline-replay-ratio 0.0
  --kmpc-horizon 5 --kmpc-solver-iterations 5
  --kmpc-log-std-init -3.5 --kmpc-log-std-max -3.0
  --reward-mode hybrid --sparse-reward-weight 0.5 --dense-reward-weight 0.5
  --dense-reward-scale-m 0.01 --online-cql-mode off
  --disable-backup-entropy --disable-actor-entropy --num-envs 5 --env-workers 5
  --seed "$SEED" --device cuda --bootstrap-checkpoint "$BOOTSTRAP"
  --bootstrap-actor-only --bootstrap-allow-dataset-mismatch
  --bootstrap-allow-schedule-change)
printf '%q ' "${CMD[@]}" > "$OUT/command.txt"
tmux new-session -d -s "circle_C3_on15k_${SEED}" -c "$REPO" \
  "$(printf '%q ' "${CMD[@]}") >$(printf '%q' "$OUT/tmux.log") 2>&1"
```

## 7. 复现检查

启动前：

```bash
test -x "$PYTHON" && test -f "$TRAINER"
test -f "$DATASET" && test -f "$FF" && test -f "$KOOPMAN"
test -f "$REFERENCE" && test -f "$SCENARIO"
nvidia-smi
```

运行中：

```bash
tmux ls
tail -f "$OUT/tmux.log"
```

完成条件：

- Offline 必须存在 `offline_010000.pt` 和 `evaluation_offline_010000.json`
- Online 必须存在 `online_015000.pt` 和 `evaluation_online_015000.json`
- 不允许提前停止，必须跑满固定预算
- 统计时从上述固定 evaluation JSON 读取 `return_mean`、`tip_rmse_m`、
  `tip_p95_m`、`tip_success_rate_2p5mm`
- 五 seed 聚合使用算术平均与样本标准差（`ddof=1`）

旧 Circle 测试的低频文本备份位于：

```text
runs/o2o/archive/manisoft_circle_preformal_records_20260905.tar.gz
```

## 8. 已保存的 AWAC-KMPC best rollout

按照正式固定 checkpoint 的 return 在五个 seed 中选择，offline 与 online
最优项均为 seed `20260855`，各保存一个确定性 1000-step rollout：

| 阶段 | 源 checkpoint | Return | Tip RMSE | Tip P95 | Success@2.5mm |
|---|---|---:|---:|---:|---:|
| Offline | `offline/seed20260855/AWAC-KMPC/offline_010000.pt` | 582.8521 | 10.4530 mm | 23.4766 mm | 37.7% |
| Online | `online/seed20260855/AWAC-KMPC/online_015000.pt` | 634.0179 | 10.4859 mm | 25.6546 mm | 49.0% |

轨迹文件：

```text
rollouts/AWAC-KMPC/offline_best/trajectory.npz
rollouts/AWAC-KMPC/offline_best/summary.json
rollouts/AWAC-KMPC/online_best/trajectory.npz
rollouts/AWAC-KMPC/online_best/summary.json
```

`summary.json` 记录源 checkpoint、评估 seed、汇总指标，以及 NPZ 内全部
数组的 shape 和 dtype。轨迹包含 observation/physical state、reward、目标
位置、节点误差、tip/joint error、残差与实际动作、feedforward、KMPC cost
和 implicit residual 诊断量。

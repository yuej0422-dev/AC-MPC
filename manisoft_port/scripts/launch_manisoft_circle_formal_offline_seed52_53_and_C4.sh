#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/AC-MPC
entry=$repo/manisoft_port/scripts/train_manisoft_circle_time_awac_kmpc.py
dataset=$repo/runs/o2o/diagnostics/dataset_rebuild/manisoft_circle_curated_200k_E7_canonical.npz
koopman=$repo/work_dirs/manisoft_abs_u06_1132ep_h0_formal_walker_512/koopman_h0_formal_walker_loss/koopman_history/best_validation.pt
scenario=/root/autodl-tmp/ManiSoft/configs/demo_elastica_fast.yaml
reference=$repo/work_dirs/manisoft_circle_r010_benchmark_klqr/trajectory.npz
feedforward=$repo/runs/o2o/probes/manisoft_circle_phase15_degraded/policy.npz
formal_base=$repo/runs/o2o/formal/manisoft_circle_5seed7_offline_20260903
diagnostic_base=$repo/runs/o2o/diagnostics/E7_R2_online_optimization_20260903
python=$repo/.venv/bin/python
pythonpath=$repo/manisoft_port:$repo:/root/autodl-tmp/ManiSoft:/root/autodl-tmp/ManiSoft/third_party/pyelastica:/root/autodl-tmp/ManiSoft/third_party/liegroups

common_offline=(
  --feedforward "$feedforward"
  --dataset "$dataset"
  --koopman "$koopman"
  --scenario "$scenario"
  --reference "$reference"
  --offline-updates 20000
  --online-steps 0
  --offline-eval-interval 2500
  --online-eval-interval 2500
  --checkpoint-save-interval 2500
  --log-interval-updates 500
  --eval-episodes 1
  --batch-size 256
  --replay-capacity 10000
  --actor-learning-rate 3e-5
  --critic-learning-rate 3e-5
  --temperature-learning-rate 3e-5
  --offline-replay-ratio 0.5
  --online-warmup-steps 0
  --kmpc-horizon 5
  --kmpc-solver-iterations 5
  --kmpc-log-std-init -3.5
  --kmpc-log-std-max -3.0
  --reward-mode hybrid
  --sparse-reward-weight 0.5
  --dense-reward-weight 0.5
  --dense-reward-scale-m 0.01
  --device cuda
)

launch_offline() {
  local seed=$1 method=$2 tag=$3 cpus=$4 utd=$5 actor_interval=$6
  local num_envs=5 env_workers=5
  if [[ "$method" == "Cal-QL" ]]; then
    num_envs=1
    env_workers=1
  fi
  local out=$formal_base/seed${seed}/$tag
  local session=ms_circle_formal_s${seed: -2}_${tag}
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "session exists: $session"
    return
  fi
  if [[ -e "$out/run.json" || -e "$out/latest.pt" ]]; then
    echo "existing run skipped: $out"
    return
  fi
  mkdir -p "$out"
  local method_extra=()
  if [[ "$method" == "AWAC-KMPC" ]]; then
    method_extra=(
      --implicit-xyz-no-xref
      --implicit-xyz-velocity-cost-scale 0.05
      --implicit-xyz-d-scale-ratio 5.0
      --implicit-xyz-q-log-upper 1.8
      --action-cost-center-limit 0.01
      --online-cql-mode off
      --disable-backup-entropy
    )
  fi
  printf '%q ' env PYTHONPATH="$pythonpath" CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5 taskset -c "$cpus" "$python" "$entry" --method "$method" "${method_extra[@]}" "${common_offline[@]}" --actor-update-interval "$actor_interval" --online-utd "$utd" --num-envs "$num_envs" --env-workers "$env_workers" --seed "$seed" --output "$out" > "$out/command.txt"
  printf '\n' >> "$out/command.txt"
  local cmd
  cmd=$(<"$out/command.txt")
  tmux new-session -d -s "$session" -n train -c "$repo" "$cmd >'$out/tmux.log' 2>&1"
  echo "$session -> $out"
}

launch_seed() {
  local seed=$1 offset=$2
  launch_offline "$seed" AWAC-KMPC E7_AWAC_KMPC "$((offset + 0))-$((offset + 4))" 20 2
  launch_offline "$seed" AWAC AWAC "$((offset + 10))-$((offset + 14))" 1 1
  launch_offline "$seed" Cal-QL Cal_QL "$((offset + 20))-$((offset + 24))" 1 1
  launch_offline "$seed" IQL IQL "$((offset + 30))-$((offset + 34))" 1 1
  launch_offline "$seed" AWAC-raw AWAC_raw "$((offset + 40))-$((offset + 44))" 20 1
  launch_offline "$seed" AWAC-lift AWAC_lift "$((offset + 50))-$((offset + 54))" 20 1
}

launch_c4() {
  local out=$diagnostic_base/E7_C4_online0_lr5e5_ratio1to10
  local session=ms_circle_E7_R2_C4_10K
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "session exists: $session"
    return
  fi
  if [[ -e "$out/run.json" || -e "$out/latest.pt" ]]; then
    echo "existing run skipped: $out"
    return
  fi
  mkdir -p "$out"
  local cmd=(
    env PYTHONPATH="$pythonpath" CUDA_VISIBLE_DEVICES=0
    OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10
    taskset -c 60-69 "$python" "$entry"
    --implicit-xyz-no-xref
    --full-capacity-online-residual
    --full-residual-channels DQ
    --implicit-xyz-velocity-cost-scale 0.05
    --implicit-xyz-d-scale-ratio 5.0
    --implicit-xyz-q-log-upper 1.8
    --feedforward "$feedforward"
    --action-cost-center-limit 0.01
    --method AWAC-KMPC
    --dataset "$dataset"
    --koopman "$koopman"
    --scenario "$scenario"
    --reference "$reference"
    --offline-updates 0
    --online-steps 10000
    --offline-eval-interval 5000
    --online-eval-interval 500
    --checkpoint-save-interval 500
    --log-interval-updates 500
    --eval-episodes 1
    --batch-size 256
    --replay-capacity 10000
    --actor-learning-rate 5e-5
    --critic-learning-rate 5e-5
    --temperature-learning-rate 1e-6
    --online-utd 10
    --online-warmup-steps 0
    --online-critic-only-steps 5000
    --actor-update-interval 1
    --kmpc-horizon 5
    --kmpc-solver-iterations 5
    --kmpc-log-std-init -3.5
    --kmpc-log-std-max -3.0
    --reward-mode hybrid
    --sparse-reward-weight 0.5
    --dense-reward-weight 0.5
    --dense-reward-scale-m 0.01
    --offline-replay-ratio 0.5
    --actor-offline-replay-ratio 0.0
    --online-cql-mode off
    --disable-backup-entropy
    --disable-actor-entropy
    --num-envs 5
    --env-workers 5
    --seed 20260851
    --device cuda
    --bootstrap-checkpoint "$repo/runs/o2o/diagnostics/no_xref_authority_refinement_20260902/runs/E7_D500_Q18/best_return.pt"
    --bootstrap-actor-only
    --bootstrap-allow-dataset-mismatch
    --output "$out"
  )
  printf '%q ' "${cmd[@]}" > "$out/command.txt"
  printf '\n' >> "$out/command.txt"
  local shell_cmd
  shell_cmd=$(<"$out/command.txt")
  tmux new-session -d -s "$session" -n train -c "$repo" "$shell_cmd >'$out/tmux.log' 2>&1"
  echo "$session -> $out"
}

mkdir -p "$formal_base" "$diagnostic_base"
launch_seed 20260852 0
launch_seed 20260853 5
launch_c4

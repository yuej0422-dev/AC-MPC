#!/usr/bin/env bash
set -euo pipefail

repo=/root/autodl-tmp/AC-MPC
entry=$repo/manisoft_port/scripts/train_manisoft_circle_time_awac_kmpc.py
dataset=$repo/runs/o2o/diagnostics/dataset_rebuild/manisoft_circle_curated_200k_E7_canonical.npz
koopman=$repo/work_dirs/manisoft_abs_u06_1132ep_h0_formal_walker_512/koopman_h0_formal_walker_loss/koopman_history/best_validation.pt
scenario=/root/autodl-tmp/ManiSoft/configs/demo_elastica_fast.yaml
reference=$repo/work_dirs/manisoft_circle_r010_benchmark_klqr/trajectory.npz
feedforward=$repo/runs/o2o/probes/manisoft_circle_phase15_degraded/policy.npz
base=$repo/runs/o2o/diagnostics/E7_formal_offline_baselines_20260903_seed20260851
common=(
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
  --actor-update-interval 1
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
  --seed 20260851
  --device cuda
)

launch() {
  local method=$1 tag=$2 cpus=$3 utd=$4
  local num_envs=5 env_workers=5
  if [[ "$method" == "Cal-QL" ]]; then
    # Exact online Monte-Carlo return bookkeeping requires one environment,
    # including during an offline-only run's evaluation setup.
    num_envs=1
    env_workers=1
  fi
  local out=$base/$tag session=ms_circle_E7_off_${tag}
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "session exists: $session"; return
  fi
  if [[ -e "$out/latest.pt" || -e "$out/run.json" ]]; then
    echo "existing run skipped: $out"; return
  fi
  mkdir -p "$out"
  printf '%q ' env PYTHONPATH="$repo/manisoft_port:$repo:/root/autodl-tmp/ManiSoft:/root/autodl-tmp/ManiSoft/third_party/pyelastica:/root/autodl-tmp/ManiSoft/third_party/liegroups" CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 taskset -c "$cpus" "$repo/.venv/bin/python" "$entry" --method "$method" "${common[@]}" --num-envs "$num_envs" --env-workers "$env_workers" --online-utd "$utd" --output "$out" > "$out/command.txt"
  printf '\n' >> "$out/command.txt"
  local cmd; cmd=$(cat "$out/command.txt")
  tmux new-session -d -s "$session" -n train -c "$repo" "$cmd >'$out/tmux.log' 2>&1"
  echo "$session -> $out"
}

mkdir -p "$base"
launch AWAC AWAC 0-9 1
launch Cal-QL Cal_QL 40-49 1
launch IQL IQL 50-59 1
launch AWAC-raw AWAC_raw 60-69 20
launch AWAC-lift AWAC_lift 70-79 20

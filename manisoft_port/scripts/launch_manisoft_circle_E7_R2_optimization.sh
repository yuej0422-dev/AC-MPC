#!/usr/bin/env bash
# R2@10k online actor-strength continuation screen and clean winner extension.
set -euo pipefail

repo=/root/autodl-tmp/AC-MPC
python=$repo/.venv/bin/python
entry=$repo/manisoft_port/scripts/train_manisoft_circle_time_awac_kmpc.py
base=$repo/runs/o2o/diagnostics/E7_R2_online_optimization_20260903
source=$base/source/R2_online_010000_full.pt
bootstrap=$base/source/R2_online_010000_actor_bootstrap_offline.pt
e7_offline=$repo/runs/o2o/diagnostics/no_xref_authority_refinement_20260902/runs/E7_D500_Q18/best_return.pt
dataset=$repo/runs/o2o/diagnostics/dataset_rebuild/manisoft_circle_curated_200k_E7_canonical.npz
koopman=$repo/work_dirs/manisoft_abs_u06_1132ep_h0_formal_walker_512/koopman_h0_formal_walker_loss/koopman_history/best_validation.pt
scenario=/root/autodl-tmp/ManiSoft/configs/demo_elastica_fast.yaml
reference=$repo/work_dirs/manisoft_circle_r010_benchmark_klqr/trajectory.npz
feedforward=$repo/runs/o2o/probes/manisoft_circle_phase15_degraded/policy.npz

export PYTHONPATH=$repo/manisoft_port:$repo:/root/autodl-tmp/ManiSoft:/root/autodl-tmp/ManiSoft/third_party/pyelastica:/root/autodl-tmp/ManiSoft/third_party/liegroups

common=(
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
  --offline-eval-interval 5000
  --online-eval-interval 500
  --checkpoint-save-interval 500
  --log-interval-updates 500
  --eval-episodes 1
  --batch-size 256
  --replay-capacity 10000
  --critic-learning-rate 5e-5
  --temperature-learning-rate 1e-6
  --online-utd 20
  --online-warmup-steps 0
  --online-critic-only-steps 5000
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
)

launch_tmux() {
  local session=$1 output=$2 cpu_set=$3
  shift 3
  [[ -f "$source" ]] || { echo "missing full source: $source" >&2; return 1; }
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "session already exists: $session" >&2
    return 1
  fi
  if [[ -e "$output/run.json" || -e "$output/latest.pt" ]]; then
    echo "refusing to overwrite existing run: $output" >&2
    return 1
  fi
  mkdir -p "$output"
  printf '%q ' env "PYTHONPATH=$PYTHONPATH" CUDA_VISIBLE_DEVICES=0 \
    OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 \
    taskset -c "$cpu_set" "$python" "$entry" "${common[@]}" "$@" \
    >"$output/command.txt"
  printf '\n' >>"$output/command.txt"
  local command
  command=$(cat "$output/command.txt")
  tmux new-session -d -s "$session" -n train -c "$repo" \
    "$command >'$output/tmux.log' 2>&1"
  echo "$session -> $output"
}

case "${1:-}" in
  screens)
    launch_tmux ms_circle_E7_R2_C0 "$base/C0_R2_lr1e6_int4" 0-9 \
      --output "$base/C0_R2_lr1e6_int4" \
      --continue-online-checkpoint "$source" --online-steps 12500 \
      --actor-learning-rate 1e-6 --actor-update-interval 4
    launch_tmux ms_circle_E7_R2_C1 "$base/C1_R2_lr2e6_int4" 10-19 \
      --output "$base/C1_R2_lr2e6_int4" \
      --continue-online-checkpoint "$source" --online-steps 12500 \
      --actor-learning-rate 2e-6 --actor-update-interval 4
    launch_tmux ms_circle_E7_R2_C2 "$base/C2_R2_lr1e6_int2" 20-29 \
      --output "$base/C2_R2_lr1e6_int2" \
      --continue-online-checkpoint "$source" --online-steps 12500 \
      --actor-learning-rate 1e-6 --actor-update-interval 2
    launch_tmux ms_circle_E7_R2_C3 "$base/C3_R2_lr2e6_int2" 30-39 \
      --output "$base/C3_R2_lr2e6_int2" \
      --continue-online-checkpoint "$source" --online-steps 12500 \
      --actor-learning-rate 2e-6 --actor-update-interval 2
    ;;
  c3_replacement)
    launch_tmux ms_circle_E7_R2_C3 "$base/C3_R2_lr2e6_int2" 30-39 \
      --output "$base/C3_R2_lr2e6_int2" \
      --continue-online-checkpoint "$source" --online-steps 12500 \
      --actor-learning-rate 2e-6 --actor-update-interval 2
    ;;
  restart_c1c3)
    [[ -f "$e7_offline" ]] || { echo "missing E7 offline source: $e7_offline" >&2; exit 1; }
    launch_tmux ms_circle_E7_R2_C1_10K "$base/E7_C1_online0_lr2e6_int4" 10-19 \
      --output "$base/E7_C1_online0_lr2e6_int4" \
      --bootstrap-checkpoint "$e7_offline" --bootstrap-actor-only --bootstrap-allow-dataset-mismatch \
      --online-steps 10000 --actor-learning-rate 2e-6 --actor-update-interval 4
    launch_tmux ms_circle_E7_R2_C2_10K "$base/E7_C2_online0_lr1e6_int2" 20-29 \
      --output "$base/E7_C2_online0_lr1e6_int2" \
      --bootstrap-checkpoint "$e7_offline" --bootstrap-actor-only --bootstrap-allow-dataset-mismatch \
      --online-steps 10000 --actor-learning-rate 1e-6 --actor-update-interval 2
    launch_tmux ms_circle_E7_R2_C3_10K "$base/E7_C3_online0_lr2e6_int2" 30-39 \
      --output "$base/E7_C3_online0_lr2e6_int2" \
      --bootstrap-checkpoint "$e7_offline" --bootstrap-actor-only --bootstrap-allow-dataset-mismatch \
      --online-steps 10000 --actor-learning-rate 2e-6 --actor-update-interval 2
    ;;
  formal)
    winner=${2:?formal requires C0, C1, C2, or C3}
    case "$winner" in
      C0) lr=1e-6; interval=4 ;;
      C1) lr=2e-6; interval=4 ;;
      C2) lr=1e-6; interval=2 ;;
      C3) lr=2e-6; interval=2 ;;
      *) echo "invalid winner: $winner" >&2; exit 2 ;;
    esac
    out=$base/E7_R2_online_extended_${winner}
    launch_tmux ms_circle_E7_R2_formal "$out" 0-9 \
      --output "$out" --continue-online-checkpoint "$source" \
      --online-steps 20000 --actor-learning-rate "$lr" \
      --actor-update-interval "$interval"
    ;;
  *)
    echo "usage: $0 {screens|c3_replacement|restart_c1c3|formal C0|C1|C2|C3}" >&2
    exit 2
    ;;
esac

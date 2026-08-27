# Hopper Hop native-1000 PPO 50M campaign

This branch isolates the environment-discretization hypothesis from the formal
TD-MPC2 action-repeat-2 Hopper experiments.

## Phase 1: reference PPO

- Task: `dm_control.suite.hopper:hop`.
- Protocol: native `control_timestep=0.02`, action repeat 1, 20 seconds and
  exactly 1,000 policy decisions per complete episode.
- Policy initialization: from scratch; no TD-MPC2, ExORL or previous offline
  dataset is read.
- Training seed: `20260861` for the first run.
- Budget: 49,999,872 environment transitions.  This is the nearest PPO
  rollout-batch-aligned value below 50,000,000 (`256 envs x 8 steps`).
- Runtime allocation: 256 logical environments distributed over the available
  CPU workers, with policy/value optimization on CUDA.
- Durable state: `latest.pt` is overwritten every 100 PPO updates and `best.pt`
  tracks the best rolling stochastic training return.
- Historical snapshots: `diagnostics/step_NNNNNNN.pt` every 500,000 requested
  environment transitions.  The payload records the exact aligned step.
- Phase 1 deliberately uses `--no-collect`: PPO rollout data are not treated as
  the later offline dataset and no existing dataset is consumed.

Configuration: `campaigns/hopper_hop_native1000_ppo50m.yaml`.

## Phase 2: checkpoint-conditioned collection

After PPO finishes, select checkpoints spanning early, middle, late and final
training.  Each checkpoint must collect into a disjoint directory using the
checkpoint's saved observation-normalization statistics and the same native
protocol.  Collection will retain complete 1,000-step episodes, policy
checkpoint SHA256, deterministic/stochastic policy mode, optional exploration
noise, environment seeds, requested/applied actions, rewards, discounts and
terminal semantics.

The final offline dataset will be built only from these self-collected chunks.
Its checkpoint mixture and quality selection will be declared after inspecting
the checkpoint return distribution; no legacy Hopper archive is an eligible
source.

# ManiSoft fixed-circle O2O dataset

The fixed task is a 1,000-step episode: 100 minimum-jerk entry steps followed
by one counter-clockwise lap of the 10 cm circle.  The final reference uses a
perfect node-20 circle and periodic Fourier fits of the physically feasible
node-6/node-14 trajectories.

The offline archive follows the canonical Formal Walker transition layout:

- `observation`, `next_observation`: float32 `[N,46]`, containing the unchanged
  45-D ManiSoft physical state and one normalized clock `t/1000`.  Target XYZ
  is not present in the observation.
- `action`: float32 `[N,18]`, the final absolute activation applied to the
  simulator.  PPO residual actions and delta actions are not stored.
- `reward`: float32 `[N]` with values in `{0,1}`.  It is one when the joint
  9-D XYZ error of nodes 6, 14 and 20 at the next time index is at most 2.5 mm.
- `discount`, `episode_id`, `episode_step`, `terminated`, `truncated`, and
  `mc_return` have the same alignment and timeout semantics as Formal Walker.
- `quality_tier`, error diagnostics, the fixed target table, and
  `metadata_json` provide audit information but are not policy inputs.

The 100 episodes comprise 40 expert, 30 successful, and 30 medium/near-expert
rollouts.  Every episode is accepted by measured RMSE and binary-reward-rate
gates; labels are never assigned solely from the requested noise level.

Rebuild with `scripts/build_manisoft_circle_o2o_dataset.py`, using the selected
600k PPO checkpoint, its matching VecNormalize state, and the integral-KLQR
benchmark reference.  The builder refuses overwrite, validates transition
continuity and Monte-Carlo returns, writes atomically, and records SHA-256
identities for every source artifact.

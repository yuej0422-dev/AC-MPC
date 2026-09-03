# ManiSoft circle mainline

This branch records the reproducible Circle tracking line, separate from the
ManiSoft obstacle-push/avoidance experiments.

## Canonical control path

```text
(lifted body state, task time) -> actor -> Q,p cost map -> KMPC -> absolute action
```

The Circle environment adds the phase-indexed `u_ff` at the KMPC boundary. The
Koopman dynamics therefore use absolute actions, while the learned controller
outputs a residual action/cost map. The canonical offline dataset is the
curated 200k feed-forward/residual buffer used by E7. The E7 AWAC-KMPC offline
run and E7 R2/C-series online continuation are the reference experiments.

## Mainline experiments

- E7 AWAC-KMPC: implicit time-conditioned cost map, full residual channels,
  `d_action_max=0.01`, H=5, solver=5.
- C0/C1/C2/C3/C4: online continuation screens from the E7 offline actor;
  replay, critic UTD, actor interval and learning-rate changes are recorded in
  each run directory.
- Formal offline baselines: official AWAC, Cal-QL, IQL, AWAC-raw and
  AWAC-lift, with `u_ff` and one-episode deterministic evaluations.

The shared O2O learner changes in this branch provide explicit actor-update
intervals, separate actor/critic replay batches, physical action scaling and
strict JSON-safe diagnostics. They are task-agnostic infrastructure used by
the Circle line as well as the existing DMC tests.

## Deliberately excluded

Obstacle-push/avoidance launchers, environments, teacher policies, collision
buffers, wall-route experiments and their checkpoints are not part of this
mainline commit. They remain in the working tree for their independent
experiments and must not be imported by Circle training.

Generated datasets, `work_dirs/`, `.venv/`, tmux logs, checkpoints and rendered
figures are runtime artifacts and are not versioned here.

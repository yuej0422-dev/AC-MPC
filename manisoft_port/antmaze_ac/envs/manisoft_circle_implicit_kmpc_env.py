"""History-aware implicit-phase adapter for the KMPC state-only ablation."""

from __future__ import annotations

import numpy as np

from .circle_task_state import PHASE_DIM, phase_features
from .manisoft_circle_o2o_env import (
    HISTORY_CONTEXT_DIM,
    PHYSICAL_DIM,
    TASK_NAME,
    ManiSoftCircleO2OAdapter,
)


POLICY_OBSERVATION_DIM = PHYSICAL_DIM + PHASE_DIM
COLLECTOR_OBSERVATION_DIM = POLICY_OBSERVATION_DIM + HISTORY_CONTEXT_DIM


class ManiSoftCircleImplicitKmpcAdapter(ManiSoftCircleO2OAdapter):
    """Replace the scalar clock with a periodic ``sin/cos`` task state."""

    @property
    def obs_dim(self) -> int:
        return COLLECTOR_OBSERVATION_DIM

    def _observation(self) -> np.ndarray:
        if self._state is None:
            raise RuntimeError("Environment must be reset first")
        phase = phase_features(self._step_count, self.episode_steps)
        result = np.concatenate(
            (self._state, phase, self._history_context()),
        )
        if result.shape != (COLLECTOR_OBSERVATION_DIM,):
            raise ValueError("Implicit KMPC observation has the wrong shape")
        if not np.isfinite(result).all():
            raise FloatingPointError("Implicit KMPC observation is non-finite")
        return result.astype(np.float32, copy=False)

    def protocol_metadata(self) -> dict[str, object]:
        base = super().protocol_metadata()
        base.update(
            protocol_name="manisoft_fixed_circle_implicit_phase_kmpc_v2",
            protocol_schema_version=2,
            policy_observation_dim=POLICY_OBSERVATION_DIM,
            policy_observation="physical_state_45 + phase_sin_cos_2",
            target_in_observation=False,
            collector_observation_dim=COLLECTOR_OBSERVATION_DIM,
            phase_dynamics="exact one-step rotation inside KMPC rollout",
        )
        return base


def make_manisoft_circle_implicit_kmpc_adapter(
    task_name: str,
    seed: int = 0,
    control_timestep: float | None = None,
    time_limit: float | None = None,
) -> ManiSoftCircleImplicitKmpcAdapter:
    if task_name != TASK_NAME:
        raise ValueError(f"Expected task {TASK_NAME!r}, got {task_name!r}")
    return ManiSoftCircleImplicitKmpcAdapter(
        task_name,
        seed=seed,
        control_timestep=control_timestep,
        time_limit=time_limit,
    )

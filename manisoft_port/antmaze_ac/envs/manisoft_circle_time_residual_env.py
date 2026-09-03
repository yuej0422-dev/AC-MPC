"""Clock-only ManiSoft circle observations with residual actions around u_ff(t)."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from .circle_phase_feedforward import FrozenCirclePhaseFeedforward
from . import manisoft_circle_o2o_env as _circle_o2o
from .manisoft_circle_o2o_env import TASK_NAME, ManiSoftCircleO2OAdapter


FEEDFORWARD_ENV = "ACMPC_MANISOFT_CIRCLE_FEEDFORWARD"
RESIDUAL_LIMIT_ENV = "ACMPC_MANISOFT_CIRCLE_RESIDUAL_LIMIT"


class ManiSoftCircleTimeResidualAdapter(ManiSoftCircleO2OAdapter):
    """Expose ``physical_state_45 + tau_1`` and accept residual control."""

    def __init__(self, task_name: str, **kwargs: Any) -> None:
        feedforward_path = os.environ.get(FEEDFORWARD_ENV)
        if not feedforward_path:
            raise RuntimeError(f"{FEEDFORWARD_ENV} must identify the frozen policy")
        self.feedforward = FrozenCirclePhaseFeedforward(feedforward_path)
        self.residual_limit = float(os.environ.get(RESIDUAL_LIMIT_ENV, "0.3"))
        if not np.isfinite(self.residual_limit) or not 0 < self.residual_limit <= 0.5:
            raise ValueError("Residual action limit must lie in (0, 0.5]")
        # The residual-control protocol uses a physical actuator box of
        # +/-0.5.  The parent adapter reads this module-level value when it
        # constructs the underlying environment, so set it before delegating.
        _circle_o2o.ABSOLUTE_ACTION_LIMIT = 0.5
        super().__init__(task_name, **kwargs)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        requested_residual = np.asarray(action, dtype=np.float32).reshape(-1)
        if requested_residual.shape != (18,) or not np.isfinite(requested_residual).all():
            raise ValueError("Residual action must be finite with shape (18,)")
        bounded_residual = np.clip(
            requested_residual, -self.residual_limit, self.residual_limit
        ).astype(np.float32)
        feedforward_action = self.feedforward.action(self._step_count, self.episode_steps)
        physical_requested = feedforward_action + bounded_residual
        observation, reward, done, base_info = super().step(physical_requested)
        physical_applied = np.asarray(base_info["applied_action"], dtype=np.float32)
        applied_residual = physical_applied - feedforward_action
        info = dict(base_info)
        info.update(
            applied_action=applied_residual.astype(np.float32),
            requested_action=requested_residual.copy(),
            feedforward_action=feedforward_action.copy(),
            physical_requested_action=physical_requested.astype(np.float32),
            physical_applied_action=physical_applied.copy(),
            action_semantics="residual_u",
        )
        return observation, reward, done, info

    def protocol_metadata(self) -> dict[str, Any]:
        base = super().protocol_metadata()
        base.update(
            protocol_name="manisoft_circle_time_residual_control_structure_v1",
            policy_observation="physical_state_45 + normalized_task_time_tau_1",
            learner_observation="Koopman_lifted_body_state + tau_1",
            target_in_observation=False,
            xref_in_observation=False,
            feedforward_in_observation=False,
            action_semantics="residual_u",
            physical_action_semantics="u_ff(t) + residual_u",
            residual_limit=self.residual_limit,
            feedforward=self.feedforward.identity(),
        )
        return base


def make_manisoft_circle_time_residual_adapter(
    task_name: str,
    seed: int = 0,
    control_timestep: float | None = None,
    time_limit: float | None = None,
) -> ManiSoftCircleTimeResidualAdapter:
    return ManiSoftCircleTimeResidualAdapter(
        task_name,
        seed=seed,
        control_timestep=control_timestep,
        time_limit=time_limit,
    )

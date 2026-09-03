"""Implicit-phase circle environment with a frozen coarse feedforward action."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from .circle_phase_feedforward import FrozenCirclePhaseFeedforward
from .manisoft_circle_implicit_kmpc_env import (
    TASK_NAME,
    ManiSoftCircleImplicitKmpcAdapter,
)


FEEDFORWARD_ENV = "ACMPC_MANISOFT_CIRCLE_FEEDFORWARD"
RESIDUAL_LIMIT_ENV = "ACMPC_MANISOFT_CIRCLE_RESIDUAL_LIMIT"


class ManiSoftCircleResidualAdapter(ManiSoftCircleImplicitKmpcAdapter):
    """Interpret learner actions as feedback residuals around u_ff(phase)."""

    def __init__(self, task_name: str, **kwargs: Any) -> None:
        feedforward_path = os.environ.get(FEEDFORWARD_ENV)
        if not feedforward_path:
            raise RuntimeError(f"{FEEDFORWARD_ENV} must identify the frozen policy")
        self.feedforward = FrozenCirclePhaseFeedforward(feedforward_path)
        self.residual_limit = float(os.environ.get(RESIDUAL_LIMIT_ENV, "0.1"))
        if not np.isfinite(self.residual_limit) or not 0 < self.residual_limit <= 0.3:
            raise ValueError("Residual action limit must lie in (0, 0.3]")
        super().__init__(task_name, **kwargs)

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        requested_residual = np.asarray(action, dtype=np.float32).reshape(-1)
        if requested_residual.shape != (18,) or not np.isfinite(
            requested_residual
        ).all():
            raise ValueError("Residual action must be finite with shape (18,)")
        bounded_residual = np.clip(
            requested_residual,
            -self.residual_limit,
            self.residual_limit,
        ).astype(np.float32)
        feedforward_action = self.feedforward.action(
            self._step_count,
            self.episode_steps,
        )
        physical_requested = feedforward_action + bounded_residual
        observation, reward, done, base_info = super().step(physical_requested)
        physical_applied = np.asarray(
            base_info["applied_action"], dtype=np.float32
        )
        applied_residual = physical_applied - feedforward_action
        info = dict(base_info)
        info.update(
            applied_action=applied_residual.astype(np.float32),
            requested_action=requested_residual.copy(),
            feedforward_action=feedforward_action.copy(),
            physical_requested_action=physical_requested.astype(np.float32),
            physical_applied_action=physical_applied.copy(),
            residual_clip_fraction=float(
                np.mean(np.abs(requested_residual) >= self.residual_limit)
            ),
            action_semantics="residual_u",
        )
        return observation, reward, done, info

    def protocol_metadata(self) -> dict[str, Any]:
        base = super().protocol_metadata()
        base.update(
            protocol_name="manisoft_fixed_circle_residual_feedback_v1",
            protocol_schema_version=1,
            learner_observation="physical Koopman lift + phase_sin_cos_2",
            learner_lifted_observation_dim=79,
            target_in_observation=False,
            action_semantics="residual_u",
            residual_limit=self.residual_limit,
            physical_action_semantics="u_ff(phase) + residual_u",
            feedforward=self.feedforward.identity(),
            feedforward_design="coarse phase-only nominal controller",
            step_limit=1000,
        )
        return base


def make_manisoft_circle_residual_adapter(
    task_name: str,
    seed: int = 0,
    control_timestep: float | None = None,
    time_limit: float | None = None,
) -> ManiSoftCircleResidualAdapter:
    if task_name != TASK_NAME:
        raise ValueError(f"Expected task {TASK_NAME!r}, got {task_name!r}")
    return ManiSoftCircleResidualAdapter(
        task_name,
        seed=seed,
        control_timestep=control_timestep,
        time_limit=time_limit,
    )

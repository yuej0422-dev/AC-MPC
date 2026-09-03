"""Clock-only residual-action view of the fixed-circle offline buffer."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from antmaze_ac.envs.circle_phase_feedforward import FrozenCirclePhaseFeedforward
from antmaze_ac.envs.manisoft_circle_time_residual_env import (
    FEEDFORWARD_ENV,
    RESIDUAL_LIMIT_ENV,
)

from .circle_o2o_dataset import ManiSoftCircleOfflineDataset


class ManiSoftCircleTimeResidualDataset(ManiSoftCircleOfflineDataset):
    """Keep ``physical+tau`` observations and convert absolute u to residual u."""

    def __init__(self, path: str | Path, koopman_path: str | Path, **kwargs: object) -> None:
        super().__init__(path, koopman_path, **kwargs)
        source_sha256 = self.sha256
        feedforward_path = os.environ.get(FEEDFORWARD_ENV)
        if not feedforward_path:
            raise RuntimeError(f"{FEEDFORWARD_ENV} must identify the frozen policy")
        self.feedforward = FrozenCirclePhaseFeedforward(feedforward_path)
        self.residual_limit = float(os.environ.get(RESIDUAL_LIMIT_ENV, "0.3"))
        episode_step = self.arrays["episode_step"].astype(np.int64)
        physical_action = self.arrays["action"].astype(np.float32, copy=True)
        feedforward_action = self.feedforward.action(episode_step, self.steps_per_episode)
        residual_action = physical_action - feedforward_action
        maximum = float(np.max(np.abs(residual_action)))
        if maximum > self.residual_limit + 1e-6:
            raise ValueError(
                f"Residual dataset action {maximum:.6f} exceeds {self.residual_limit:.6f}"
            )
        self._physical_actions = physical_action
        self._feedforward_actions = feedforward_action
        self.arrays["action"] = residual_action.astype(np.float32, copy=False)
        identity = json.dumps(
            {
                "kind": "manisoft_circle_time_residual_dataset_v1",
                "source_sha256": source_sha256,
                "feedforward_sha256": self.feedforward.sha256,
                "policy_observation": "physical_state_45 + tau_1",
                "action_semantics": "physical_action - u_ff(t)",
                "residual_limit": self.residual_limit,
                "reward_mode": self.reward_mode,
                "dense_reward_weight": self.dense_reward_weight,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.sha256 = hashlib.sha256(identity).hexdigest()
        self.metadata = {
            **self.metadata,
            "kind": "manisoft_circle_time_residual_dataset_v1",
            "source_sha256": source_sha256,
            "feedforward": self.feedforward.identity(),
            "policy_observation": "physical_state_45 + tau_1",
            "target_in_observation": False,
            "xref_in_observation": False,
            "feedforward_in_observation": False,
            "action_semantics": "residual_u",
            "residual_limit": self.residual_limit,
            "residual_abs_max": maximum,
            "sha256": self.sha256,
        }

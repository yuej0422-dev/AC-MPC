"""Residual-action view of the implicit-phase ManiSoft circle dataset."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from antmaze_ac.envs.circle_phase_feedforward import (
    FrozenCirclePhaseFeedforward,
)

from .circle_implicit_kmpc_dataset import (
    ManiSoftCircleImplicitKmpcDataset,
)


FEEDFORWARD_ENV = "ACMPC_MANISOFT_CIRCLE_FEEDFORWARD"
RESIDUAL_LIMIT_ENV = "ACMPC_MANISOFT_CIRCLE_RESIDUAL_LIMIT"


class ManiSoftCircleResidualDataset(ManiSoftCircleImplicitKmpcDataset):
    """Keep physical history but expose action minus fixed feedforward to RL."""

    def __init__(
        self,
        path: str | Path,
        koopman_path: str | Path,
        **kwargs: object,
    ) -> None:
        super().__init__(path, koopman_path, **kwargs)
        feedforward_path = os.environ.get(FEEDFORWARD_ENV)
        if not feedforward_path:
            raise RuntimeError(f"{FEEDFORWARD_ENV} must identify the frozen policy")
        self.feedforward = FrozenCirclePhaseFeedforward(feedforward_path)
        self.residual_limit = float(os.environ.get(RESIDUAL_LIMIT_ENV, "0.1"))
        if not np.isfinite(self.residual_limit) or not 0 < self.residual_limit <= 0.3:
            raise ValueError("Residual action limit must lie in (0, 0.3]")

        physical_action = self.arrays["action"].astype(np.float32, copy=True)
        feedforward_action = self.feedforward.action(
            self.arrays["episode_step"].astype(np.int64),
            self.steps_per_episode,
        )
        residual_action = physical_action - feedforward_action
        maximum = float(np.max(np.abs(residual_action)))
        if maximum > self.residual_limit + 1e-6:
            raise ValueError(
                f"Residual dataset action {maximum:.6f} exceeds configured "
                f"limit {self.residual_limit:.6f}"
            )
        # History contexts were already constructed by the parent from the
        # physical applied actions.  Only the action coordinate seen by the
        # learner is replaced here.
        self._physical_actions = physical_action
        self._feedforward_actions = feedforward_action
        self.arrays["action"] = residual_action.astype(np.float32, copy=False)

        identity = json.dumps(
            {
                "kind": "manisoft_circle_residual_dataset_v1",
                "parent_sha256": self.sha256,
                "feedforward_sha256": self.feedforward.sha256,
                "residual_limit": self.residual_limit,
                "action_semantics": "physical_action - frozen_phase_feedforward",
                "history_action_semantics": "physical_applied_action",
                "reward_mode": self.reward_mode,
                "sparse_reward_weight": self.sparse_reward_weight,
                "dense_reward_weight": self.dense_reward_weight,
                "dense_reward_scale_m": self.dense_reward_scale_m,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        parent_sha256 = self.sha256
        self.sha256 = hashlib.sha256(identity).hexdigest()
        self.metadata = {
            **self.metadata,
            "kind": "manisoft_circle_residual_dataset_v1",
            "parent_sha256": parent_sha256,
            "feedforward": self.feedforward.identity(),
            "residual_limit": self.residual_limit,
            "residual_abs_max": maximum,
            "action_semantics": "residual_u",
            "physical_action_in_history": True,
            "target_in_observation": False,
            "sha256": self.sha256,
        }

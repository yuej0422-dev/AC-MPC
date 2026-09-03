"""Implicit ``physical + sin/cos phase`` view of the circle O2O dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from antmaze_ac.envs.circle_task_state import implicit_task_observation

from .circle_o2o_dataset import ManiSoftCircleOfflineDataset


class ManiSoftCircleImplicitKmpcDataset(ManiSoftCircleOfflineDataset):
    """Keep rewards/history unchanged and replace only the scalar task clock."""

    def __init__(
        self,
        path: str | Path,
        koopman_path: str | Path,
        **kwargs: object,
    ) -> None:
        super().__init__(path, koopman_path, **kwargs)
        source_sha256 = self.sha256
        episode_step = self.arrays["episode_step"].astype(np.int64)
        self._policy_observations = implicit_task_observation(
            self.arrays["observation"][:, :45],
            episode_step,
            self.steps_per_episode,
        )
        self._next_policy_observations = implicit_task_observation(
            self.arrays["next_observation"][:, :45],
            episode_step + 1,
            self.steps_per_episode,
        )
        identity = json.dumps(
            {
                "kind": "manisoft_circle_implicit_kmpc_dataset_v2",
                "source_sha256": source_sha256,
                "policy_observation": "physical_state_45 + phase_sin_cos_2",
                "history_context_dim": 630,
                "reward_mode": self.reward_mode,
                "sparse_reward_weight": self.sparse_reward_weight,
                "dense_reward_weight": self.dense_reward_weight,
                "dense_reward_scale_m": self.dense_reward_scale_m,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.source_sha256 = source_sha256
        self.sha256 = hashlib.sha256(identity).hexdigest()
        self.metadata = {
            **self.metadata,
            "kind": "manisoft_circle_implicit_kmpc_dataset_v2",
            "source_sha256": source_sha256,
            "policy_observation": "physical_state_45 + phase_sin_cos_2",
            "policy_observation_dim": 47,
            "collector_observation_dim": 677,
            "target_in_observation": False,
            "sha256": self.sha256,
        }

    @property
    def policy_observations(self) -> np.ndarray:
        return self._policy_observations

    def sample(
        self,
        size: int,
        generator: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        if size < 1:
            raise ValueError("sample size must be positive")
        index = generator.integers(0, len(self), size=size)
        context = self._context_offsets[index]
        result = {key: value[index] for key, value in self.arrays.items()}
        result["reward"] = self._rewards[index]
        result["mc_return"] = self._mc_returns[index]
        result["observation"] = np.concatenate(
            (
                self._policy_observations[index],
                self.history_context[context],
            ),
            axis=1,
        ).astype(np.float32, copy=False)
        result["next_observation"] = np.concatenate(
            (
                self._next_policy_observations[index],
                self.history_context[context + 1],
            ),
            axis=1,
        ).astype(np.float32, copy=False)
        return result

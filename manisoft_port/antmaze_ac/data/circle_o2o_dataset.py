"""Frozen ManiSoft circle dataset with derived H=10 learner contexts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from antmaze_ac.envs.anchor_residual_tracking_env import NODE_POSITION_INDICES
from antmaze_ac.envs.circle_reward import circle_reward_components, validate_reward_config

NODE_REWARD_WEIGHTS = np.asarray([0.2, 0.2, 0.6], dtype=np.float32)


CANONICAL_KEYS = (
    "observation",
    "action",
    "reward",
    "discount",
    "next_observation",
    "episode_id",
    "episode_step",
    "terminated",
    "truncated",
    "mc_return",
)


class ManiSoftCircleOfflineDataset:
    """Random sampler that derives history without changing the source NPZ."""

    def __init__(
        self,
        path: str | Path,
        koopman_path: str | Path,
        *,
        reference_path: str | Path | None = None,
        reward_mode: str = "sparse",
        sparse_reward_weight: float = 1.0,
        dense_reward_weight: float = 0.0,
        dense_reward_scale_m: float = 0.01,
        reward_radius_m: float = 0.0025,
        gamma: float = 0.99,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        digest = hashlib.sha256()
        with self.path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        self.sha256 = digest.hexdigest()
        with np.load(self.path, allow_pickle=False) as archive:
            self.arrays = {key: np.asarray(archive[key]) for key in CANONICAL_KEYS}
            self.metadata = json.loads(str(archive["metadata_json"].item()))
        if self.metadata.get("kind") not in {
            "acmpc_manisoft_circle_transitions_v1",
            "acmpc_manisoft_circle_delta_transitions_v1",
            "acmpc_manisoft_circle_ff15_perturbation_transitions_v1",
            "acmpc_manisoft_circle_rebuilt_v1",
        }:
            raise ValueError("Unsupported ManiSoft circle dataset kind")
        if self.metadata.get("task") != "manisoft_fixed_circle_tracking":
            raise ValueError("Dataset task identity is not the fixed circle")
        count = len(self.arrays["reward"])
        self.episodes = int(self.metadata.get("episodes", -1))
        self.steps_per_episode = int(
            self.metadata.get("transitions_per_episode", -1)
        )
        declared_transitions = int(self.metadata.get("transitions", -1))
        if (
            self.episodes < 1
            or self.steps_per_episode != 1000
            or declared_transitions != count
            or count != self.episodes * self.steps_per_episode
        ):
            raise ValueError(
                "ManiSoft circle metadata does not match complete 1000-step episodes"
            )
        expected = {
            "observation": (count, 46),
            "action": (count, 18),
            "reward": (count,),
            "discount": (count,),
            "next_observation": (count, 46),
        }
        for key, shape in expected.items():
            if self.arrays[key].shape != shape or not np.isfinite(self.arrays[key]).all():
                raise ValueError(f"Dataset {key} violates shape/finite contract")
        self.absolute_action_limit = float(
            self.metadata.get("absolute_action_limit", 0.3)
        )
        if (
            not np.isfinite(self.absolute_action_limit)
            or not 0 < self.absolute_action_limit <= 0.5
        ):
            raise ValueError(
                "Dataset absolute_action_limit metadata must lie in (0, 0.5]"
            )
        if (
            np.max(np.abs(self.arrays["action"]))
            > self.absolute_action_limit + 1e-6
        ):
            raise ValueError("Dataset contains non-absolute or out-of-range actions")
        if not np.array_equal(
            np.unique(self.arrays["reward"]), np.asarray([0.0, 1.0], np.float32)
        ):
            raise ValueError("Dataset reward must be binary")
        validate_reward_config(
            reward_mode,
            sparse_weight=sparse_reward_weight,
            dense_weight=dense_reward_weight,
            dense_scale_m=dense_reward_scale_m,
            radius_m=reward_radius_m,
        )
        if reward_mode == "sparse":
            dense_reward_weight = 0.0
        if reward_mode in {"hybrid", "dense_xref", "dense_joint"} and reference_path is None:
            raise ValueError(f"{reward_mode} dataset rewards require reference_path")
        self.reward_mode = str(reward_mode)
        self.sparse_reward_weight = float(sparse_reward_weight)
        self.dense_reward_weight = float(dense_reward_weight)
        self.dense_reward_scale_m = float(dense_reward_scale_m)
        self.reward_radius_m = float(reward_radius_m)
        self.reward_gamma = float(gamma)
        if not np.isfinite(self.reward_gamma) or not 0 < self.reward_gamma <= 1:
            raise ValueError("gamma must lie in (0, 1]")
        self._rewards = self.arrays["reward"].astype(np.float32, copy=True)
        if self.reward_mode in {"hybrid", "dense_xref", "dense_joint"}:
            with np.load(Path(reference_path).expanduser().resolve(), allow_pickle=False) as archive:
                targets = np.asarray(archive["target_positions"], dtype=np.float32)
                xref = np.asarray(archive["xref"], dtype=np.float32)
            if targets.shape != (self.steps_per_episode + 1, 3, 3):
                raise ValueError("reference target_positions shape does not match dataset")
            if xref.shape != (self.steps_per_episode + 1, 45):
                raise ValueError("reference xref shape does not match dataset")
            next_states = self.arrays["next_observation"][:, :45]
            following_step = self.arrays["episode_step"].astype(np.int64) + 1
            if self.reward_mode == "dense_xref":
                reward_errors = np.sqrt(
                    np.mean((next_states - xref[following_step]) ** 2, axis=1)
                )
                sparse_weight = 0.0
            else:
                actual = next_states[:, NODE_POSITION_INDICES].reshape(-1, 3, 3)
                target = targets[following_step]
                errors = np.linalg.norm(actual - target, axis=2)
                reward_errors = np.sqrt(np.sum(NODE_REWARD_WEIGHTS[None, :] * errors**2, axis=1))
                sparse_weight = self.sparse_reward_weight
            self._rewards = np.asarray(
                [
                    circle_reward_components(
                        float(error),
                        sparse_weight=sparse_weight,
                        dense_weight=self.dense_reward_weight,
                        dense_scale_m=self.dense_reward_scale_m,
                        radius_m=self.reward_radius_m,
                    )[0]
                    for error in reward_errors
                ],
                dtype=np.float32,
            )
        self._mc_returns = np.empty_like(self._rewards)
        for episode in range(self.episodes):
            left = episode * self.steps_per_episode
            right = left + self.steps_per_episode
            running = 0.0
            for index in range(right - 1, left - 1, -1):
                running = float(self._rewards[index]) + self.reward_gamma * running
                self._mc_returns[index] = running

        import torch

        payload = torch.load(
            Path(koopman_path).expanduser().resolve(),
            map_location="cpu",
            weights_only=False,
        )
        state = payload.get("normalizers", {}).get("state", {})
        mean = np.asarray(state.get("mean"), dtype=np.float32)
        std = np.maximum(np.asarray(state.get("std"), dtype=np.float32), 1e-6)
        if mean.shape != (45,) or std.shape != (45,):
            raise ValueError("Koopman normalizer does not match physical state")
        architecture = payload.get("architecture", {})
        configured_koopman = payload.get("config", {}).get("koopman", {})
        self.history_steps = int(
            architecture.get(
                "history_steps",
                configured_koopman.get("history_steps", 0),
            )
        )
        if self.history_steps < 0:
            raise ValueError("Koopman history_steps cannot be negative")
        self.history_context_dim = self.history_steps * (45 + 18)
        self.history_context = np.empty(
            (count + self.episodes, self.history_context_dim), dtype=np.float32
        )
        context_cursor = 0
        for episode in range(self.episodes):
            left = episode * self.steps_per_episode
            right = left + self.steps_per_episode
            states = np.concatenate(
                (
                    self.arrays["observation"][left : left + 1, :45],
                    self.arrays["next_observation"][left:right, :45],
                ),
                axis=0,
            )
            actions = self.arrays["action"][left:right]
            for step in range(self.steps_per_episode + 1):
                if self.history_steps == 0:
                    continue
                state_indices = np.maximum(
                    np.arange(
                        step - self.history_steps + 1,
                        step + 1,
                        dtype=np.int64,
                    ),
                    0,
                )
                state_history = (states[state_indices] - mean) / std
                action_history = np.zeros(
                    (self.history_steps, 18), dtype=np.float32
                )
                available = min(step, self.history_steps)
                if available:
                    action_history[-available:] = actions[step - available : step]
                self.history_context[context_cursor + step] = np.concatenate(
                    (state_history.reshape(-1), action_history.reshape(-1))
                )
            context_cursor += self.steps_per_episode + 1
        self.metadata = {
            **self.metadata,
            "koopman_history_steps": self.history_steps,
            "history_context_dim": self.history_context_dim,
        }
        self._context_offsets = (
            np.arange(count, dtype=np.int64) + self.arrays["episode_id"]
        )
        self._validate_history_continuity()

    def _validate_history_continuity(self) -> None:
        for episode in range(self.episodes):
            left = episode * self.steps_per_episode
            right = left + self.steps_per_episode
            if not np.all(self.arrays["episode_id"][left:right] == episode):
                raise ValueError("Episode ids are not contiguous from zero")
            if not np.array_equal(
                self.arrays["episode_step"][left:right],
                np.arange(
                    self.steps_per_episode,
                    dtype=self.arrays["episode_step"].dtype,
                ),
            ):
                raise ValueError("Episode steps are not contiguous")
            if not np.array_equal(
                self.arrays["next_observation"][left : right - 1],
                self.arrays["observation"][left + 1 : right],
            ):
                raise ValueError("Dataset transitions are discontinuous")

    def __len__(self) -> int:
        return len(self.arrays["reward"])

    @property
    def policy_observations(self) -> np.ndarray:
        return self.arrays["observation"]

    def sample(self, size: int, generator: np.random.Generator) -> dict[str, np.ndarray]:
        if size < 1:
            raise ValueError("sample size must be positive")
        index = generator.integers(0, len(self), size=size)
        context = self._context_offsets[index]
        result = {key: value[index] for key, value in self.arrays.items()}
        result["reward"] = self._rewards[index]
        result["mc_return"] = self._mc_returns[index]
        result["observation"] = np.concatenate(
            (result["observation"], self.history_context[context]), axis=1
        ).astype(np.float32, copy=False)
        result["next_observation"] = np.concatenate(
            (result["next_observation"], self.history_context[context + 1]), axis=1
        ).astype(np.float32, copy=False)
        return result

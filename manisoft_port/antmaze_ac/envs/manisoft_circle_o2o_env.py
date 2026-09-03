"""Canonical fixed-circle ManiSoft adapter for formal O2O experiments.

The public policy observation is always ``physical_state_45 + clock_1``.
The collector additionally carries a 10-step normalized state/absolute-action
history after those 46 entries.  Raw learners slice the first 46 entries;
history-Koopman learners consume the internal context when constructing their
lift.  The time-indexed XYZ target is never part of either policy input.
"""

from __future__ import annotations

from collections import deque
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import types
from typing import Any

import numpy as np
import torch


def _install_headless_stubs() -> None:
    if "pyvista" not in sys.modules and importlib.util.find_spec("pyvista") is None:
        module = types.ModuleType("pyvista")

        def unavailable(*_args: Any, **_kwargs: Any):
            raise RuntimeError("pyvista is unavailable in headless O2O training")

        module.read = module.Cube = module.Plotter = unavailable
        sys.modules["pyvista"] = module
    if "moviepy" not in sys.modules and importlib.util.find_spec("moviepy") is None:
        module = types.ModuleType("moviepy")
        module.ImageSequenceClip = object
        sys.modules["moviepy"] = module


_install_headless_stubs()

from .anchor_residual_tracking_env import NODE_POSITION_INDICES
from .circle_reward import circle_reward_components, validate_reward_config
from .manisoft_tracking_env import ManiSoftTipTrackingEnv


TASK_NAME = "manisoft_circle"
PHYSICAL_DIM = 45
POLICY_OBSERVATION_DIM = 46
ACTION_DIM = 18
NODE_REWARD_WEIGHTS = np.asarray([0.2, 0.2, 0.6], dtype=np.float32)
HISTORY_STEPS = 10
HISTORY_CONTEXT_DIM = HISTORY_STEPS * (PHYSICAL_DIM + ACTION_DIM)
COLLECTOR_OBSERVATION_DIM = POLICY_OBSERVATION_DIM + HISTORY_CONTEXT_DIM
ABSOLUTE_ACTION_LIMIT = 0.3
REWARD_RADIUS_M = 0.0025
DENSE_REWARD_SCALE_M = 0.01


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_path(environment_name: str) -> Path:
    raw = os.environ.get(environment_name)
    if not raw:
        raise RuntimeError(f"{environment_name} must name an existing file")
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


class ManiSoftCircleO2OAdapter:
    """DMC-vector compatible adapter with absolute 18-D actions."""

    def __init__(
        self,
        task_name: str,
        *,
        seed: int = 0,
        control_timestep: float | None = None,
        time_limit: float | None = None,
        reward_mode: str | None = None,
        sparse_reward_weight: float | None = None,
        dense_reward_weight: float | None = None,
        dense_reward_scale_m: float | None = None,
    ) -> None:
        if task_name != TASK_NAME:
            raise ValueError(f"Expected task {TASK_NAME!r}, got {task_name!r}")
        if control_timestep is not None and not np.isclose(control_timestep, 0.02):
            raise ValueError("ManiSoft circle control_timestep is frozen at 0.02 s")
        if time_limit is not None and not np.isclose(time_limit, 20.0):
            raise ValueError("ManiSoft circle time_limit is frozen at 20 s")
        self.scenario_path = _required_path("ACMPC_MANISOFT_SCENARIO")
        self.reference_path = _required_path("ACMPC_MANISOFT_CIRCLE_REFERENCE")
        self.koopman_path = _required_path("ACMPC_MANISOFT_KOOPMAN")
        # Environment variables are used by spawned vector workers, which
        # must construct the same adapter without pickling a closure. Direct
        # keyword arguments remain useful for tests and renderers.
        self.reward_mode = str(
            reward_mode
            if reward_mode is not None
            else os.environ.get("ACMPC_MANISOFT_REWARD_MODE", "sparse")
        )
        self.sparse_reward_weight = float(
            sparse_reward_weight
            if sparse_reward_weight is not None
            else os.environ.get("ACMPC_MANISOFT_SPARSE_REWARD_WEIGHT", "1.0")
        )
        self.dense_reward_weight = float(
            dense_reward_weight
            if dense_reward_weight is not None
            else os.environ.get("ACMPC_MANISOFT_DENSE_REWARD_WEIGHT", "0.0")
        )
        self.dense_reward_scale_m = float(
            dense_reward_scale_m
            if dense_reward_scale_m is not None
            else os.environ.get(
                "ACMPC_MANISOFT_DENSE_REWARD_SCALE_M", str(DENSE_REWARD_SCALE_M)
            )
        )
        validate_reward_config(
            self.reward_mode,
            sparse_weight=self.sparse_reward_weight,
            dense_weight=self.dense_reward_weight,
            dense_scale_m=self.dense_reward_scale_m,
            radius_m=REWARD_RADIUS_M,
        )
        if self.reward_mode == "sparse":
            # Explicit sparse mode is a compatibility switch: even if a
            # stale process environment contains dense knobs, no shaping is
            # silently added to the historical reward.
            self.dense_reward_weight = 0.0
        with np.load(self.reference_path, allow_pickle=False) as archive:
            self.targets = np.asarray(archive["target_positions"], dtype=np.float32)
            self.xref = np.asarray(archive["xref"], dtype=np.float32)
            self.episode_steps = int(np.asarray(archive["u_ff"]).shape[0])
        if self.targets.shape != (self.episode_steps + 1, 3, 3):
            raise ValueError("Circle target table has the wrong shape")
        if self.xref.shape != (self.episode_steps + 1, PHYSICAL_DIM):
            raise ValueError("Circle xref table has the wrong shape")
        payload = torch.load(self.koopman_path, map_location="cpu", weights_only=False)
        architecture = payload.get("architecture", {})
        configured_koopman = payload.get("config", {}).get("koopman", {})
        self.history_steps = int(
            architecture.get(
                "history_steps",
                configured_koopman.get("history_steps", 0),
            )
        )
        if (
            architecture.get("state_dim") != PHYSICAL_DIM
            or architecture.get("action_dim") != ACTION_DIM
            or self.history_steps not in (0, 10)
        ):
            raise ValueError("Expected a ManiSoft H0 or H10 Koopman checkpoint")
        self.history_context_dim = self.history_steps * (
            PHYSICAL_DIM + ACTION_DIM
        )
        state_normalizer = payload.get("normalizers", {}).get("state", {})
        self.state_mean = np.asarray(state_normalizer.get("mean"), dtype=np.float32)
        self.state_std = np.maximum(
            np.asarray(state_normalizer.get("std"), dtype=np.float32), 1e-6
        )
        if self.state_mean.shape != (PHYSICAL_DIM,) or self.state_std.shape != (
            PHYSICAL_DIM,
        ):
            raise ValueError("Koopman state normalizer has the wrong shape")
        self._seed = int(seed)
        self._env: ManiSoftTipTrackingEnv | None = None
        self._state_history: deque[np.ndarray] = deque(maxlen=self.history_steps)
        self._action_history: deque[np.ndarray] = deque(maxlen=self.history_steps)
        self._state: np.ndarray | None = None
        self._step_count = 0

    @property
    def obs_dim(self) -> int:
        return POLICY_OBSERVATION_DIM + self.history_context_dim

    @property
    def action_dim(self) -> int:
        return ACTION_DIM

    @property
    def action_low(self) -> np.ndarray:
        return np.full(ACTION_DIM, -ABSOLUTE_ACTION_LIMIT, dtype=np.float32)

    @property
    def action_high(self) -> np.ndarray:
        return np.full(ACTION_DIM, ABSOLUTE_ACTION_LIMIT, dtype=np.float32)

    def _history_context(self) -> np.ndarray:
        if self.history_steps == 0:
            return np.empty(0, dtype=np.float32)
        if len(self._state_history) != self.history_steps or len(
            self._action_history
        ) != self.history_steps:
            raise RuntimeError("History context is not initialized")
        states = np.asarray(self._state_history, dtype=np.float32)
        actions = np.asarray(self._action_history, dtype=np.float32)
        normalized_states = (states - self.state_mean) / self.state_std
        return np.concatenate(
            (normalized_states.reshape(-1), actions.reshape(-1))
        ).astype(np.float32, copy=False)

    def _observation(self) -> np.ndarray:
        if self._state is None:
            raise RuntimeError("Environment must be reset first")
        clock = np.asarray([self._step_count / self.episode_steps], dtype=np.float32)
        result = np.concatenate((self._state, clock, self._history_context()))
        if result.shape != (self.obs_dim,) or not np.isfinite(result).all():
            raise FloatingPointError("Invalid ManiSoft O2O observation")
        return result.astype(np.float32, copy=False)

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._seed = int(seed)
        if self._env is not None:
            self._env.close()
        self._env = ManiSoftTipTrackingEnv(
            self.scenario_path,
            target_offset=(0.01, 0.0, 0.0),
            episode_steps=self.episode_steps,
            success_threshold=-1.0,
            absolute_action_limit=ABSOLUTE_ACTION_LIMIT,
        )
        state, _ = self._env.reset(seed=self._seed)
        self._state = np.asarray(state, dtype=np.float32).reshape(PHYSICAL_DIM)
        self._step_count = 0
        self._state_history.clear()
        self._action_history.clear()
        for _ in range(self.history_steps):
            self._state_history.append(self._state.copy())
            self._action_history.append(np.zeros(ACTION_DIM, dtype=np.float32))
        return self._observation()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        if self._env is None or self._state is None:
            raise RuntimeError("Environment must be reset before step")
        requested = np.asarray(action, dtype=np.float32).reshape(-1)
        if requested.shape != (ACTION_DIM,) or not np.isfinite(requested).all():
            raise ValueError("Action must be finite with shape (18,)")
        applied = np.clip(requested, self.action_low, self.action_high).astype(np.float32)
        state, _dense_reward, _terminated, _truncated, base_info = self._env.step(applied)
        self._state = np.asarray(state, dtype=np.float32).reshape(PHYSICAL_DIM)
        self._step_count += 1
        self._state_history.append(self._state.copy())
        self._action_history.append(applied.copy())
        actual = self._state[NODE_POSITION_INDICES].reshape(3, 3)
        target = self.targets[self._step_count]
        node_error = np.linalg.norm(actual - target, axis=1)
        # The scalar dense-joint reward is a weighted XYZ position error:
        # the terminal tip contributes 0.6 and the two upstream nodes 0.2 each.
        # Keep the per-node errors in ``node_target_error`` for diagnostics.
        joint_error = float(np.sqrt(np.sum(NODE_REWARD_WEIGHTS * node_error**2)))
        xref_rmse = float(
            np.sqrt(np.mean((self._state - self.xref[self._step_count]) ** 2))
        )
        reward_error = xref_rmse if self.reward_mode == "dense_xref" else joint_error
        reward, sparse_component, dense_component = circle_reward_components(
            reward_error,
            sparse_weight=(
                0.0 if self.reward_mode in {"dense_xref", "dense_joint"} else self.sparse_reward_weight
            ),
            dense_weight=(
                self.dense_reward_weight
                if self.reward_mode in {"hybrid", "dense_xref", "dense_joint"}
                else 0.0
            ),
            dense_scale_m=self.dense_reward_scale_m,
            radius_m=REWARD_RADIUS_M,
        )
        truncated = self._step_count >= self.episode_steps
        info = dict(base_info)
        info.update(
            applied_action=applied.copy(),
            requested_action=requested.copy(),
            discount=1.0,
            terminated=False,
            truncated=bool(truncated),
            joint_target_error=joint_error,
            xref_state_rmse_m=xref_rmse,
            node_target_error=node_error.astype(np.float32),
            reward_node_weights=NODE_REWARD_WEIGHTS.copy(),
            target_positions=target.copy(),
            sparse_reward=float(sparse_component),
            dense_reward=float(dense_component),
            reward_mode=self.reward_mode,
        )
        return self._observation(), reward, bool(truncated), info

    def protocol_metadata(self) -> dict[str, Any]:
        return {
            "protocol_name": "manisoft_fixed_circle_o2o_v1",
            "protocol_schema_version": 1,
            "task": TASK_NAME,
            "simulator": "ManiSoft",
            "policy_observation_dim": POLICY_OBSERVATION_DIM,
            "policy_observation": "physical_state_45 + normalized_time",
            "target_in_observation": False,
            "collector_observation_dim": self.obs_dim,
            "history_steps": self.history_steps,
            "history_context_dim": self.history_context_dim,
            "history_semantics": "normalized physical states + absolute applied actions",
            "action_dim": ACTION_DIM,
            "action_semantics": "absolute_u",
            "action_low": self.action_low.tolist(),
            "action_high": self.action_high.tolist(),
            "control_dt": 0.02,
            "physics_dt": 0.0002,
            "n_substeps": 100,
            "step_limit": self.episode_steps,
            "time_limit": 0.02 * self.episode_steps,
            "reward": (
                "full-state xref RMSE exponential shaping"
                if self.reward_mode == "dense_xref"
                else "binary joint node6/node14/node20 XYZ radius"
                if self.reward_mode == "sparse"
                else "exponential joint XYZ shaping"
            ),
            "reward_radius_m": REWARD_RADIUS_M,
            "reward_node_weights": NODE_REWARD_WEIGHTS.tolist(),
            "reward_mode": self.reward_mode,
            "sparse_reward_weight": self.sparse_reward_weight,
            "dense_reward_weight": self.dense_reward_weight,
            "dense_reward_scale_m": self.dense_reward_scale_m,
            "scenario_path": str(self.scenario_path),
            "scenario_sha256": _sha256(self.scenario_path),
            "reference_path": str(self.reference_path),
            "reference_sha256": _sha256(self.reference_path),
            "history_koopman_path": str(self.koopman_path),
            "history_koopman_sha256": _sha256(self.koopman_path),
        }

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
        self._env = None
        self._state = None


def make_manisoft_circle_o2o_adapter(
    task_name: str,
    seed: int = 0,
    control_timestep: float | None = None,
    time_limit: float | None = None,
    reward_mode: str | None = None,
    sparse_reward_weight: float | None = None,
    dense_reward_weight: float | None = None,
    dense_reward_scale_m: float | None = None,
) -> ManiSoftCircleO2OAdapter:
    return ManiSoftCircleO2OAdapter(
        task_name,
        seed=seed,
        control_timestep=control_timestep,
        time_limit=time_limit,
        reward_mode=reward_mode,
        sparse_reward_weight=sparse_reward_weight,
        dense_reward_weight=dense_reward_weight,
        dense_reward_scale_m=dense_reward_scale_m,
    )

"""ManiSkill GPU vector runner implementing the DMC O2O vector contract."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from experiments.dmc.ppo.vector_env import VectorStep


ENVIRONMENT_PROTOCOL = {
    "protocol_name": "maniskill_hopper_hop_native_v1",
    "backend": "maniskill_sapien_physx_gpu",
    "environment_id": "MS-HopperHop-v1",
    "obs_mode": "state",
    "control_mode": "pd_joint_delta_pos",
    "reward_mode": "normalized_dense",
    "action_repeat": 1,
    "step_limit": 600,
    "observation_dim": 15,
    "action_dim": 4,
}


class ManiSkillHopperVectorEnv:
    """Synchronized GPU environments with explicit timeout-reset semantics."""

    def __init__(self, num_envs: int, seed: int) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401 - registers the task
        from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

        self.num_envs = int(num_envs)
        self.seed = int(seed)
        self.obs_dim = 15
        self.action_dim = 4
        self.action_low = np.full(4, -1.0, dtype=np.float32)
        self.action_high = np.full(4, 1.0, dtype=np.float32)
        self.protocol = dict(ENVIRONMENT_PROTOCOL)
        self._episode_counts = np.zeros(self.num_envs, dtype=np.int64)
        self._current_reset_seeds = np.zeros(self.num_envs, dtype=np.int64)
        self._observations: np.ndarray | None = None
        self._closed = False
        base = gym.make(
            "MS-HopperHop-v1",
            num_envs=self.num_envs,
            obs_mode="state",
            control_mode="pd_joint_delta_pos",
            reward_mode="normalized_dense",
            sim_backend="gpu" if torch.cuda.is_available() else "cpu",
            render_backend="none",
        )
        self._env = ManiSkillVectorEnv(
            base,
            self.num_envs,
            auto_reset=False,
            ignore_terminations=False,
            record_metrics=True,
        )

    @staticmethod
    def _numpy(value: Any, *, dtype: np.dtype[Any]) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy().astype(dtype, copy=False)
        return np.asarray(value, dtype=dtype)

    def _reset_seeds(self) -> np.ndarray:
        return (
            self.seed
            + np.arange(self.num_envs, dtype=np.int64)
            + self._episode_counts * self.num_envs
        )

    def reset(self) -> np.ndarray:
        if self._closed:
            raise RuntimeError("Cannot reset a closed environment")
        seeds = self._reset_seeds()
        observation, _ = self._env.reset(seed=[int(seed) for seed in seeds])
        result = self._numpy(observation, dtype=np.float32)
        if result.shape != (self.num_envs, self.obs_dim):
            raise ValueError(f"Unexpected ManiSkill observation shape {result.shape}")
        self._current_reset_seeds = seeds.copy()
        self._observations = result.copy()
        return result.copy()

    def step(self, action: np.ndarray) -> VectorStep:
        if self._closed or self._observations is None:
            raise RuntimeError("reset() must precede step()")
        applied = np.clip(
            np.asarray(action, dtype=np.float32), self.action_low, self.action_high
        )
        if applied.shape != (self.num_envs, self.action_dim):
            raise ValueError("ManiSkill action batch has the wrong shape")
        device = getattr(self._env.unwrapped, "device", torch.device("cuda"))
        result = self._env.step(torch.as_tensor(applied, device=device))
        observation, reward, terminated, truncated, info = result
        transition_observation = self._numpy(observation, dtype=np.float32)
        rewards = self._numpy(reward, dtype=np.float32).reshape(self.num_envs)
        terminated_array = self._numpy(terminated, dtype=np.bool_).reshape(self.num_envs)
        truncated_array = self._numpy(truncated, dtype=np.bool_).reshape(self.num_envs)
        boundary = terminated_array | truncated_array
        if bool(boundary.any()) != bool(boundary.all()):
            raise RuntimeError("ManiSkill Hopper episodes lost synchronized boundaries")
        discount = (~terminated_array).astype(np.float32)
        reset_seed = np.full(self.num_envs, -1, dtype=np.int64)
        policy_observation = transition_observation.copy()
        if bool(boundary.all()):
            reset_seed = self._current_reset_seeds.copy()
            self._episode_counts += 1
            policy_observation = self.reset()
        self._observations = policy_observation.copy()
        per_env_info = tuple(
            {
                "discount": float(discount[index]),
                "terminated": bool(terminated_array[index]),
                "truncated": bool(truncated_array[index]),
            }
            for index in range(self.num_envs)
        )
        return VectorStep(
            observation=policy_observation,
            transition_observation=transition_observation,
            reward=rewards,
            discount=discount,
            terminated=terminated_array,
            truncated=truncated_array,
            reset_boundary=boundary,
            reset_seed=reset_seed,
            applied_action=applied,
            info=per_env_info,
        )

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._env.close()


def make_maniskill_hopper_vector_env(
    task_name: str, num_envs: int, seed: int, *, workers: int = 1
) -> ManiSkillHopperVectorEnv:
    if task_name != "hopper_hop":
        raise ValueError("ManiSkill Hopper runner only supports hopper_hop")
    if workers < 1:
        raise ValueError("workers must be positive")
    return ManiSkillHopperVectorEnv(num_envs, seed)

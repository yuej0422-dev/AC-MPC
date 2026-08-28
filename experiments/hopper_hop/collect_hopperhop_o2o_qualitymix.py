"""Collect the immutable 200k ManiSkill HopperHop O2O quality mixture.

The four source policies are completed 100M-step PPO seeds with substantially
different closed-loop returns.  Every source contributes the same transition
quota.  With the default 50k quota and 600-step environment horizon this is
83 complete episodes plus one deterministic 200-step prefix per policy.  The
prefix endpoint is an explicit dataset truncation; no transition crosses an
environment reset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch.distributions import Normal

import mani_skill
import mani_skill.envs  # noqa: F401 - registers MS-HopperHop-v1
import sapien
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from experiments.dmc.o2o.dataset import (
    MANISKILL_HOPPER_DATASET_KIND,
    OfflineDataset,
    _mc_returns,
    validate_dataset_arrays,
)
from experiments.hopper_hop.train_hopper_hop_ppo import Actor


STATE_DIM = 15
ACTION_DIM = 4
EPISODE_LENGTH = 600
POLICY_SEEDS = (20_240_801, 20_240_802, 20_240_803, 20_240_804)


@dataclass(frozen=True)
class CollectConfig:
    checkpoint_root: Path
    output: Path
    policy_seeds: tuple[int, ...] = POLICY_SEEDS
    transitions_per_policy: int = 50_000
    exploration_scale: float = 1.0
    collection_seed: int = 20_260_901
    gamma: float = 0.99
    device: str = "cuda"

    @property
    def episodes_per_policy(self) -> int:
        return math.ceil(self.transitions_per_policy / EPISODE_LENGTH)


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _checkpoint_dir(root: Path, seed: int) -> Path:
    return root / f"ppo_seed_{seed}"


def _load_policy(
    checkpoint_path: Path, device: torch.device
) -> tuple[Actor, torch.Tensor, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if payload.get("actor_name") != "PPO":
        raise ValueError(f"{checkpoint_path} is not a HopperHop PPO checkpoint")
    if int(payload.get("obs_dim", -1)) != STATE_DIM:
        raise ValueError("PPO checkpoint observation dimension differs")
    if int(payload.get("action_dim", -1)) != ACTION_DIM:
        raise ValueError("PPO checkpoint action dimension differs")
    actor = Actor(STATE_DIM, ACTION_DIM).to(device)
    actor.load_state_dict(payload["actor_state"])
    actor.eval()
    log_std = torch.as_tensor(payload["log_std"], device=device).float()
    if log_std.shape != (ACTION_DIM,) or not torch.isfinite(log_std).all():
        raise ValueError("PPO checkpoint has an invalid log_std")
    return actor, log_std, payload


def _make_env(num_envs: int, device: torch.device) -> ManiSkillVectorEnv:
    base = gym.make(
        "MS-HopperHop-v1",
        num_envs=num_envs,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        reward_mode="normalized_dense",
        sim_backend="gpu" if device.type == "cuda" else "cpu",
        render_backend="none",
    )
    return ManiSkillVectorEnv(
        base,
        num_envs,
        auto_reset=False,
        ignore_terminations=False,
        record_metrics=True,
    )


@torch.no_grad()
def _rollout_policy(
    *,
    actor: Actor,
    log_std: torch.Tensor,
    config: CollectConfig,
    policy_index: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    num_envs = config.episodes_per_policy
    seed_base = config.collection_seed + policy_index * 10_000
    torch.manual_seed(seed_base)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_base)
    env = _make_env(num_envs, device)
    states: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    next_states: list[torch.Tensor] = []
    clipped_components = 0
    total_action_components = 0
    try:
        observation, _ = env.reset(seed=[seed_base + i for i in range(num_envs)])
        for _ in range(EPISODE_LENGTH):
            state = torch.as_tensor(observation, device=device).float()
            mean = actor(state)
            std = (log_std.exp() * config.exploration_scale).expand_as(mean)
            sampled_action = Normal(mean, std).sample()
            applied_action = sampled_action.clamp(-1.0, 1.0)
            next_observation, reward, terminated, truncated, _ = env.step(
                applied_action
            )
            if bool(torch.as_tensor(terminated).any()):
                raise RuntimeError("MS-HopperHop unexpectedly terminated early")
            states.append(state.cpu())
            actions.append(applied_action.cpu())
            rewards.append(torch.as_tensor(reward).float().reshape(-1).cpu())
            next_states.append(
                torch.as_tensor(next_observation).float().cpu()
            )
            clipped_components += int((sampled_action != applied_action).sum().item())
            total_action_components += int(sampled_action.numel())
        if not bool(torch.as_tensor(truncated).all()):
            raise RuntimeError("MS-HopperHop did not truncate every 600-step episode")
    finally:
        env.close()

    state_stack = torch.stack(states).numpy()
    action_stack = torch.stack(actions).numpy()
    reward_stack = torch.stack(rewards).numpy()
    next_stack = torch.stack(next_states).numpy()
    full_episodes, prefix_length = divmod(
        config.transitions_per_policy, EPISODE_LENGTH
    )
    episode_lengths = [EPISODE_LENGTH] * full_episodes
    if prefix_length:
        episode_lengths.append(prefix_length)
    if len(episode_lengths) != num_envs:
        raise AssertionError("Episode quota decomposition differs from vector count")

    parts: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
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
    }
    episode_returns: list[float] = []
    for local_episode, length in enumerate(episode_lengths):
        observation = state_stack[:length, local_episode].astype(np.float32)
        action = action_stack[:length, local_episode].astype(np.float32)
        reward = reward_stack[:length, local_episode].astype(np.float32)
        next_observation = next_stack[:length, local_episode].astype(np.float32)
        discount = np.ones(length, dtype=np.float32)
        terminated = np.zeros(length, dtype=np.bool_)
        truncated = np.zeros(length, dtype=np.bool_)
        truncated[-1] = True
        parts["observation"].append(observation)
        parts["action"].append(action)
        parts["reward"].append(reward)
        parts["discount"].append(discount)
        parts["next_observation"].append(next_observation)
        parts["episode_id"].append(
            np.full(length, local_episode, dtype=np.int64)
        )
        parts["episode_step"].append(np.arange(length, dtype=np.int32))
        parts["terminated"].append(terminated)
        parts["truncated"].append(truncated)
        parts["mc_return"].append(_mc_returns(reward, discount, config.gamma))
        episode_returns.append(float(reward.sum(dtype=np.float64)))
    arrays = {key: np.concatenate(value) for key, value in parts.items()}
    diagnostics = {
        "rollout_seed_base": seed_base,
        "episodes": len(episode_lengths),
        "complete_episodes": full_episodes,
        "prefix_length": prefix_length,
        "transitions": int(arrays["reward"].shape[0]),
        "stored_segment_return_mean": float(np.mean(episode_returns)),
        "stored_segment_return_min": float(np.min(episode_returns)),
        "stored_segment_return_max": float(np.max(episode_returns)),
        "sampled_action_clip_fraction": clipped_components
        / total_action_components,
    }
    return arrays, diagnostics


def collect(config: CollectConfig) -> dict[str, Any]:
    if len(config.policy_seeds) < 2 or len(set(config.policy_seeds)) != len(
        config.policy_seeds
    ):
        raise ValueError("Policy seeds must be distinct and include a mixture")
    if config.transitions_per_policy < 1:
        raise ValueError("transitions_per_policy must be positive")
    if not math.isfinite(config.exploration_scale) or config.exploration_scale < 0:
        raise ValueError("exploration_scale must be finite and non-negative")
    if not 0 < config.gamma <= 1:
        raise ValueError("gamma must lie in (0, 1]")
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA collection requested but CUDA is unavailable")
    random.seed(config.collection_seed)
    np.random.seed(config.collection_seed)

    source_arrays: list[dict[str, np.ndarray]] = []
    sources: list[dict[str, Any]] = []
    for policy_index, policy_seed in enumerate(config.policy_seeds):
        source_dir = _checkpoint_dir(config.checkpoint_root, policy_seed)
        checkpoint_path = (source_dir / "latest.pt").resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing PPO checkpoint: {checkpoint_path}")
        actor, log_std, payload = _load_policy(checkpoint_path, device)
        print(f"collecting seed={policy_seed} from {checkpoint_path}", flush=True)
        arrays, diagnostics = _rollout_policy(
            actor=actor,
            log_std=log_std,
            config=config,
            policy_index=policy_index,
            device=device,
        )
        source_arrays.append(arrays)
        final_path = source_dir / "final.json"
        final = (
            json.loads(final_path.read_text(encoding="utf-8"))
            if final_path.is_file()
            else {}
        )
        sources.append(
            {
                "policy_seed": policy_seed,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": _sha256_file(checkpoint_path),
                "checkpoint_global_step": int(payload["global_step"]),
                "checkpoint_update": int(payload["update"]),
                "saved_log_std": [float(value) for value in log_std.cpu()],
                "recorded_final_deterministic_return": final.get(
                    "evaluation", {}
                ).get("mean_return"),
                **diagnostics,
            }
        )

    combined: dict[str, list[np.ndarray]] = {
        key: [] for key in source_arrays[0]
    }
    episode_offset = 0
    for arrays in source_arrays:
        local = dict(arrays)
        local["episode_id"] = local["episode_id"] + episode_offset
        episode_offset = int(local["episode_id"][-1]) + 1
        for key, value in local.items():
            combined[key].append(value)
    arrays = {key: np.concatenate(value) for key, value in combined.items()}
    total_transitions = config.transitions_per_policy * len(config.policy_seeds)
    if arrays["reward"].shape[0] != total_transitions:
        raise AssertionError("Combined transition count differs from quota")
    validate_dataset_arrays(
        arrays,
        gamma_for_mc_return=config.gamma,
        expected_episode_count=episode_offset,
        task="hopper_hop",
        reward_source="recorded",
    )

    metadata: dict[str, Any] = {
        "kind": MANISKILL_HOPPER_DATASET_KIND,
        "task": "hopper_hop",
        "source": "four completed 100M PPO policies stratified by policy seed",
        "environment_backend": "maniskill_sapien_physx_gpu",
        "environment_id": "MS-HopperHop-v1",
        "environment_version": {
            "mani_skill": mani_skill.__version__,
            "sapien": sapien.__version__,
            "gymnasium": gym.__version__,
            "torch": torch.__version__,
        },
        "observation_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "transitions": total_transitions,
        "episodes": episode_offset,
        "transitions_per_episode": "83x600+1x200 per policy at the 50k default",
        "episode_horizon": EPISODE_LENGTH,
        "action_repeat": 1,
        "control_mode": "pd_joint_delta_pos",
        "reward_mode": "normalized_dense",
        "reward_source": "recorded",
        "gamma_for_mc_return": config.gamma,
        "environment_discount_values": [1.0],
        "timeout_bootstrap": "preserve_environment_discount_at_true_and_dataset_truncations",
        "boundary_semantics": (
            "no early termination; 600-step environment horizons and the fixed "
            "per-policy prefix endpoint are explicit truncations"
        ),
        "action_semantics": (
            "sample saved PPO Gaussian, clip to [-1,1] before both environment "
            "application and dataset storage"
        ),
        "selection": {
            "kind": "maniskill_hopper_hop_ppo_qualitymix_equal_v1",
            "policy_seeds": list(config.policy_seeds),
            "transitions_per_policy": config.transitions_per_policy,
            "exploration_scale": config.exploration_scale,
            "collection_seed": config.collection_seed,
            "equal_policy_weight": True,
        },
        "policy_sources": sources,
        "created_unix_seconds": time.time(),
    }
    config.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=config.output.parent,
        prefix=f".{config.output.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                **arrays,
                metadata_json=np.asarray(
                    json.dumps(metadata, sort_keys=True, separators=(",", ":"))
                ),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, config.output)
    finally:
        temporary.unlink(missing_ok=True)

    loaded = OfflineDataset.load(config.output)
    manifest = {
        "kind": "acmpc_maniskill_hopper_hop_qualitymix_manifest_v1",
        "dataset_path": str(loaded.path),
        "dataset_sha256": loaded.sha256,
        "transitions": len(loaded),
        "episodes": loaded.metadata["episodes"],
        "selection": loaded.metadata["selection"],
        "policy_sources": sources,
    }
    manifest_path = config.output.with_suffix(".manifest.json")
    _atomic_json(manifest_path, manifest)
    print(
        f"wrote {len(loaded):,} transitions to {loaded.path} "
        f"sha256={loaded.sha256}",
        flush=True,
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--policy-seeds",
        default=",".join(str(seed) for seed in POLICY_SEEDS),
    )
    parser.add_argument("--transitions-per-policy", type=int, default=50_000)
    parser.add_argument("--exploration-scale", type=float, default=1.0)
    parser.add_argument("--collection-seed", type=int, default=20_260_901)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = collect(
        CollectConfig(
            checkpoint_root=args.checkpoint_root.resolve(),
            output=args.output.resolve(),
            policy_seeds=tuple(int(value) for value in args.policy_seeds.split(",")),
            transitions_per_policy=args.transitions_per_policy,
            exploration_scale=args.exploration_scale,
            collection_seed=args.collection_seed,
            gamma=args.gamma,
            device=args.device,
        )
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

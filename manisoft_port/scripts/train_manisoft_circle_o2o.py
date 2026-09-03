#!/usr/bin/env python
"""Train one formal Walker-style O2O method on the ManiSoft fixed circle."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Spawn workers re-execute this file with ``manisoft_port/scripts`` first on
# sys.path.  Pin the repository root before importing the shared O2O package;
# otherwise the port's unrelated ``experiments`` package shadows it.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from experiments.dmc.o2o.checkpoint import atomic_torch_save, restore_rng, rng_state
from experiments.dmc.o2o.config import O2OConfig
from experiments.dmc.o2o.dataset import OnlineReplay, mark_offline, mixed_batch
from experiments.dmc.o2o.learner import O2OLearner, TensorBatch
from experiments.dmc.o2o.networks import FrozenObservationNormalizer
from experiments.dmc.ppo.vector_env import make_dmc_vector_env

# The repository and ManiSoft port both contain a top-level ``antmaze_ac``
# package, while both also contain an ``experiments`` package.  Import the
# shared O2O stack above from the repository root, then explicitly prioritize
# this subproject for the task-specific modules below.
MANISOFT_PORT_ROOT = Path(__file__).resolve().parents[1]
if str(MANISOFT_PORT_ROOT) in sys.path:
    sys.path.remove(str(MANISOFT_PORT_ROOT))
sys.path.insert(0, str(MANISOFT_PORT_ROOT))

from antmaze_ac.data.circle_o2o_dataset import ManiSoftCircleOfflineDataset
from antmaze_ac.envs.manisoft_circle_o2o_env import (
    ABSOLUTE_ACTION_LIMIT,
    COLLECTOR_OBSERVATION_DIM,
    DENSE_REWARD_SCALE_M,
    REWARD_RADIUS_M,
    TASK_NAME,
    make_manisoft_circle_o2o_adapter,
)
from antmaze_ac.koopman.o2o_history_adapter import FrozenManiSoftHistoryKoopman


FORMAL_METHODS = (
    "Cal-RLPD-KMPC",
    "Cal-RLPD",
    "Cal-RLPD-Lift",
    "Cal-QL",
    "RLPD",
    "AWAC",
    "IQL",
)
CHECKPOINT_KIND = "acmpc_manisoft_circle_o2o_checkpoint_v1"
OFFLINE_UPDATES = 50_000
ONLINE_STEPS = 20_000
OFFLINE_EVAL_INTERVAL = 5_000
ONLINE_EVAL_INTERVAL = 2_500
EVAL_EPISODES = 10
KMPC_HORIZON = 5
TRAINING_SEED = 20260851
BEST_MAX_SATURATION_FRACTION = 0.0
BEST_MAX_DELTA_P95_ABS = 0.06


def _json_safe(value: Any) -> Any:
    """Convert non-finite diagnostic scalars to JSON null.

    Offline-only batches intentionally report some online-partition metrics as
    NaN because that partition is absent.  Keep strict JSON output while
    preserving the distinction between an unavailable statistic and zero.
    """

    if isinstance(value, (float, np.floating)):
        scalar = float(value)
        return scalar if math.isfinite(scalar) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, dict):
        return {key: _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(
            _json_safe(payload),
            stream,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_state_sha256(value: Any) -> str:
    """Hash nested tensor/array state without pickle or zip metadata."""

    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if item is None:
            digest.update(b"N")
        elif isinstance(item, bool):
            digest.update(b"B1" if item else b"B0")
        elif isinstance(item, (int, np.integer)):
            digest.update(f"I{int(item)};".encode())
        elif isinstance(item, (float, np.floating)):
            digest.update(f"F{float(item).hex()};".encode())
        elif isinstance(item, str):
            encoded = item.encode("utf-8")
            digest.update(f"S{len(encoded)}:".encode())
            digest.update(encoded)
        elif isinstance(item, torch.Tensor):
            array = item.detach().cpu().contiguous().numpy()
            digest.update(f"T{array.dtype}:{array.shape}:".encode())
            digest.update(array.tobytes(order="C"))
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(f"A{array.dtype}:{array.shape}:".encode())
            digest.update(array.tobytes(order="C"))
        elif isinstance(item, dict):
            digest.update(f"D{len(item)}:".encode())
            for key in sorted(item, key=lambda candidate: repr(candidate)):
                update(key)
                update(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(f"L{len(item)}:".encode())
            for child in item:
                update(child)
        else:
            raise TypeError(f"Unsupported canonical-state value {type(item)!r}")

    update(value)
    return digest.hexdigest()


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(_json_safe(payload), sort_keys=True, allow_nan=False)
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


def _checkpoint_selection_key(evaluation: dict[str, Any]) -> tuple[float, ...]:
    """Rank deployment checkpoints by safety first, then tracking quality."""

    saturation = float(evaluation["action_saturation_fraction"])
    delta_p95 = float(evaluation["delta_action_p95_abs"])
    rmse = float(evaluation["joint_error_rmse_m"])
    reward_rate = float(evaluation["reward_rate"])
    saturation_excess = max(
        0.0, saturation - BEST_MAX_SATURATION_FRACTION
    )
    delta_excess = max(0.0, delta_p95 - BEST_MAX_DELTA_P95_ABS)
    unsafe = float(saturation_excess > 0.0 or delta_excess > 0.0)
    return (
        unsafe,
        saturation_excess,
        delta_excess,
        rmse,
        -reward_rate,
    )


def _seed_everything(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)


def _environment(
    num_envs: int, seed: int, *, workers: int | None = None
):
    return make_dmc_vector_env(
        TASK_NAME,
        num_envs,
        seed,
        workers=num_envs if workers is None else workers,
        env_factory=make_manisoft_circle_o2o_adapter,
    )


def evaluation_seed_metadata(*, seed_base: int, episodes: int) -> dict[str, Any]:
    """Return the auditable one-seed-per-episode evaluation schedule."""

    if episodes < 1:
        raise ValueError("Evaluation episodes must be positive")
    seeds = [int(seed_base) + episode for episode in range(int(episodes))]
    if len(set(seeds)) != len(seeds):
        raise RuntimeError("Evaluation episode seeds must be unique")
    return {
        "evaluation_seed_base": int(seed_base),
        "evaluation_episode_seeds": seeds,
        "evaluation_unique_seed_count": len(set(seeds)),
        "evaluation_seed_schedule": "seed_base + episode_index",
    }


class _ExpertWarmupPolicy:
    """Run a frozen high-quality PPO residual policy during online warmup.

    The expert was trained around its own absolute feed-forward table and on a
    99-D observation (physical state, target geometry, target velocity,
    previous absolute action and expert feed-forward).  The AWAC learner uses
    the current, possibly degraded, feed-forward table and receives residual
    actions.  We therefore convert the expert's absolute action back into the
    current residual coordinate before inserting the transition into replay.
    """

    def __init__(
        self,
        *,
        checkpoint: Path,
        vecnormalize: Path,
        expert_reference: Path,
        current_feedforward: Path,
        expert_residual_limit: float,
        perturbation_std: float,
        perturbation_limit: float,
        seed: int,
    ) -> None:
        from stable_baselines3 import PPO
        from antmaze_ac.envs.circle_phase_feedforward import FrozenCirclePhaseFeedforward

        if not checkpoint.is_file() or not vecnormalize.is_file():
            raise FileNotFoundError("expert PPO checkpoint/VecNormalize file missing")
        if not expert_reference.is_file():
            raise FileNotFoundError(f"expert reference missing: {expert_reference}")
        self.model = PPO.load(str(checkpoint), device="cpu")
        with vecnormalize.open("rb") as stream:
            normalizer = pickle.load(stream)
        self.obs_mean = np.asarray(normalizer.obs_rms.mean, dtype=np.float32)
        self.obs_var = np.asarray(normalizer.obs_rms.var, dtype=np.float32)
        self.obs_clip = float(normalizer.clip_obs)
        self.obs_epsilon = float(normalizer.epsilon)
        if self.obs_mean.shape != (99,) or self.obs_var.shape != (99,):
            raise ValueError(
                "expert PPO must use the 99-D anchor residual observation"
            )
        with np.load(expert_reference, allow_pickle=False) as archive:
            self.target_positions = np.asarray(archive["target_positions"], dtype=np.float32)
            self.expert_feedforward = np.asarray(archive["u_ff"], dtype=np.float32)
        if self.target_positions.ndim != 3 or self.target_positions.shape[1:] != (3, 3):
            raise ValueError("expert target_positions must have shape [T+1,3,3]")
        if self.expert_feedforward.shape != (self.target_positions.shape[0] - 1, 18):
            raise ValueError("expert u_ff shape does not match target table")
        self.current_feedforward = FrozenCirclePhaseFeedforward(current_feedforward)
        self.expert_residual_limit = float(expert_residual_limit)
        self.perturbation_std = float(perturbation_std)
        self.perturbation_limit = float(perturbation_limit)
        if self.expert_residual_limit <= 0 or self.perturbation_std < 0 or self.perturbation_limit < 0:
            raise ValueError("expert residual/noise limits must be non-negative")
        self.rng = np.random.default_rng(seed)
        self.noise_state: np.ndarray | None = None
        self.previous_absolute_action: np.ndarray | None = None

    def reset(self, num_envs: int) -> None:
        self.noise_state = np.zeros((num_envs, 18), dtype=np.float32)
        # The original anchor PPO observation intentionally starts with a zero
        # previous-action feature; keeping it zero reproduces that protocol.
        self.previous_absolute_action = np.zeros((num_envs, 18), dtype=np.float32)

    def act(self, observation: np.ndarray, phase_indices: np.ndarray) -> np.ndarray:
        if self.noise_state is None or self.previous_absolute_action is None:
            self.reset(len(observation))
        rows = []
        for index, phase in enumerate(np.asarray(phase_indices, dtype=np.int64)):
            phase = int(np.clip(phase, 0, self.expert_feedforward.shape[0] - 1))
            target = self.target_positions[phase]
            if phase + 1 < self.target_positions.shape[0]:
                velocity = self.target_positions[phase + 1] - self.target_positions[phase]
            else:
                velocity = self.target_positions[phase] - self.target_positions[phase - 1]
            rows.append(
                np.concatenate(
                    (
                        np.asarray(observation[index, :45], dtype=np.float32),
                        target.reshape(-1),
                        velocity.reshape(-1),
                        self.previous_absolute_action[index],
                        self.expert_feedforward[phase],
                    )
                )
            )
        expert_observation = np.asarray(rows, dtype=np.float32)
        expert_observation = (expert_observation - self.obs_mean) / np.sqrt(
            self.obs_var + self.obs_epsilon
        )
        expert_observation = np.clip(
            expert_observation, -self.obs_clip, self.obs_clip
        ).astype(np.float32)
        normalized_action, _ = self.model.predict(
            expert_observation, deterministic=True
        )
        normalized_action = np.asarray(normalized_action, dtype=np.float32).reshape(
            len(observation), 18
        )
        expert_absolute = self.expert_feedforward[
            np.clip(np.asarray(phase_indices, dtype=np.int64), 0, self.expert_feedforward.shape[0] - 1)
        ] + self.expert_residual_limit * np.clip(normalized_action, -1.0, 1.0)
        current_ff = np.stack(
            [self.current_feedforward.action(int(p), self.expert_feedforward.shape[0]) for p in phase_indices]
        ).astype(np.float32)
        residual = expert_absolute - current_ff
        if self.perturbation_std > 0:
            # Mildly correlated noise broadens the expert manifold without
            # turning warmup into a random-action phase.
            innovation = self.rng.normal(
                0.0, self.perturbation_std, size=residual.shape
            ).astype(np.float32)
            self.noise_state = 0.9 * self.noise_state + 0.1 * innovation
            perturbation = self.noise_state
        if self.perturbation_limit > 0:
            perturbation = np.clip(
                perturbation, -self.perturbation_limit, self.perturbation_limit
            )
            residual = residual + perturbation
        # The PPO expert was trained with the previous absolute command in
        # its observation.  Keep that feature synchronized with the command
        # actually proposed by the expert, before converting to this run's
        # residual coordinate.
        self.previous_absolute_action = expert_absolute.copy()
        return np.clip(residual, -0.5, 0.5).astype(np.float32)


@torch.no_grad()
def evaluate(learner: O2OLearner, *, seed_base: int, episodes: int) -> dict[str, Any]:
    # ManiSoft simulation instances are comparatively large.  Evaluate the
    # requested episodes sequentially in one environment so a 10-episode
    # statistic does not allocate ten simulators concurrently.
    returns = np.zeros(episodes, dtype=np.float64)
    all_errors: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_deltas: list[np.ndarray] = []
    for episode in range(episodes):
        env = _environment(1, seed_base + episode, workers=1)
        try:
            observation = env.reset()
            episode_actions: list[np.ndarray] = []
            episode_errors: list[np.ndarray] = []
            for step in range(1000):
                action = learner.act(observation, deterministic=True)
                vector_step = env.step(action)
                returns[episode] += float(vector_step.reward[0])
                episode_actions.append(
                    np.asarray(vector_step.applied_action[0], dtype=np.float64)
                )
                episode_errors.append(
                    np.asarray(
                        vector_step.info[0]["joint_target_error"],
                        dtype=np.float64,
                    )
                )
                expected_boundary = step == 999
                if bool(vector_step.reset_boundary[0]) != expected_boundary:
                    raise RuntimeError("Evaluation episode boundary drifted")
                observation = vector_step.observation
            actions = np.stack(episode_actions)
            previous = np.concatenate((np.zeros_like(actions[:1]), actions[:-1]), axis=0)
            all_actions.append(actions)
            all_deltas.append(np.abs(actions - previous))
            all_errors.append(np.stack(episode_errors))
        finally:
            env.close()
    errors = np.concatenate(all_errors, axis=0)
    action_trace = np.concatenate(all_actions, axis=0)
    absolute_delta = np.concatenate(all_deltas, axis=0)
    absolute_action = np.abs(action_trace)
    return {
        **evaluation_seed_metadata(seed_base=seed_base, episodes=episodes),
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "return_min": float(returns.min()),
        "return_max": float(returns.max()),
        "reward_rate": float(returns.mean() / 1000.0),
        "joint_error_rmse_m": float(np.sqrt(np.mean(errors**2))),
        "joint_error_p95_m": float(np.quantile(errors, 0.95)),
        "max_abs_action": float(absolute_action.max()),
        "mean_abs_action": float(absolute_action.mean()),
        "action_p95_abs": float(np.quantile(absolute_action, 0.95)),
        "action_saturation_fraction": float(
            np.mean(absolute_action >= 0.99 * ABSOLUTE_ACTION_LIMIT)
        ),
        "mean_abs_delta_action": float(absolute_delta.mean()),
        "delta_action_p95_abs": float(np.quantile(absolute_delta, 0.95)),
        "max_abs_delta_action": float(absolute_delta.max()),
        "episode_returns": returns.tolist(),
    }


def _checkpoint(
    *,
    config: O2OConfig,
    dataset: ManiSoftCircleOfflineDataset,
    koopman: FrozenManiSoftHistoryKoopman | None,
    learner: O2OLearner,
    replay: OnlineReplay,
    generator: np.random.Generator,
    environment_protocol: dict[str, Any],
    phase: str,
    offline_update: int,
    online_step: int,
    online_episode: int,
    best_return: float,
    best_evaluation: dict[str, Any] | None,
    include_replay: bool,
) -> dict[str, Any]:
    return {
        "kind": CHECKPOINT_KIND,
        "config": config.to_dict(),
        "config_fingerprint": config.fingerprint,
        "dataset": {"path": str(dataset.path), "sha256": dataset.sha256},
        "koopman": None if koopman is None else koopman.identity(),
        "environment_protocol": environment_protocol,
        "phase": phase,
        "offline_update": int(offline_update),
        "online_step": int(online_step),
        "online_episode": int(online_episode),
        "best_return": float(best_return),
        "best_evaluation": best_evaluation,
        "learner": learner.state_dict(),
        "online_replay": replay.state_dict() if include_replay else None,
        "rng": rng_state(generator),
    }


def _truncate_metrics(path: Path, *, offline_update: int, online_step: int) -> None:
    if not path.is_file():
        return
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if int(row.get("offline_update", 0)) <= offline_update and int(
            row.get("online_step", 0)
        ) <= online_step:
            rows.append(row)
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


_BOOTSTRAP_SCHEDULE_FIELDS = frozenset(
    {
        "offline_updates",
        "online_steps",
        "offline_eval_interval_updates",
        "eval_interval_online_steps",
        "eval_episodes",
        "online_utd",
        "online_warmup_steps",
        "replay_capacity",
        "num_envs",
        "env_workers",
        "checkpoint_interval_updates",
        "log_interval_updates",
        "actor_learning_rate",
        "critic_learning_rate",
        "temperature_learning_rate",
        "actor_update_interval",
        "backup_entropy",
        "q_cost_anchor_weight",
        "p_cost_anchor_weight",
        "offline_replay_ratio",
        "batch_size",
    }
)


def _bootstrap_config_compatible(
    source: dict[str, Any], target: dict[str, Any]
) -> bool:
    """Allow an explicitly requested schedule migration, and nothing else."""

    source_semantics = {
        key: value
        for key, value in source.items()
        if key not in _BOOTSTRAP_SCHEDULE_FIELDS
    }
    target_semantics = {
        key: value
        for key, value in target.items()
        if key not in _BOOTSTRAP_SCHEDULE_FIELDS
    }
    # Checkpoints created before the optional offline BC auxiliary was added
    # do not contain this field.  Its disabled default is exactly the old
    # behavior, so normalize only that one backward-compatible absence.
    source_semantics.setdefault("offline_behavior_clone_weight", 0.0)
    return source_semantics == target_semantics


def run(args: argparse.Namespace) -> None:
    if args.method not in FORMAL_METHODS:
        raise ValueError(f"Unknown formal method {args.method!r}")
    for name, value in (
        ("ACMPC_MANISOFT_SCENARIO", args.scenario),
        ("ACMPC_MANISOFT_CIRCLE_REFERENCE", args.reference),
        ("ACMPC_MANISOFT_KOOPMAN", args.koopman),
    ):
        os.environ[name] = str(Path(value).expanduser().resolve())
    # Spawned ManiSoft workers inherit these scalar reward settings.  Using
    # environment variables avoids passing a non-picklable factory closure to
    # the process vector runner.
    os.environ["ACMPC_MANISOFT_REWARD_MODE"] = str(args.reward_mode)
    os.environ["ACMPC_MANISOFT_SPARSE_REWARD_WEIGHT"] = str(args.sparse_reward_weight)
    os.environ["ACMPC_MANISOFT_DENSE_REWARD_WEIGHT"] = str(args.dense_reward_weight)
    os.environ["ACMPC_MANISOFT_DENSE_REWARD_SCALE_M"] = str(args.dense_reward_scale_m)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.jsonl"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    generator = _seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")

    config = O2OConfig(
        task=TASK_NAME,
        method=args.method,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        offline_updates=args.offline_updates,
        online_steps=args.online_steps,
        offline_eval_interval_updates=args.offline_eval_interval,
        log_interval_updates=args.log_interval_updates,
        eval_interval_online_steps=args.online_eval_interval,
        eval_episodes=args.eval_episodes,
        kmpc_horizon=args.kmpc_horizon,
        kmpc_solver_iterations=args.kmpc_solver_iterations,
        kmpc_delta_u_weight=args.kmpc_delta_u_weight,
        kmpc_delta_u_deadband=args.kmpc_delta_u_deadband,
        kmpc_delta_u_limit=args.kmpc_delta_u_limit,
        kmpc_log_std_init=args.kmpc_log_std_init,
        kmpc_log_std_max=args.kmpc_log_std_max,
        mpve_total_horizon=min(4, args.kmpc_horizon),
        replay_capacity=(
            args.online_steps
            if args.replay_capacity is None
            else args.replay_capacity
        ),
        actor_learning_rate=args.actor_learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        initial_temperature=args.initial_temperature,
        temperature_learning_rate=args.temperature_learning_rate,
        target_entropy=args.target_entropy,
        online_utd=args.online_utd,
        online_warmup_steps=args.online_warmup_steps,
        # ``cql_weight`` is a concrete scalar in O2OConfig (unlike the
        # method-resolved learning rates), so preserve the historical .01
        # default when the task-local CLI flag is omitted.
        cql_weight=0.01 if args.cql_weight is None else args.cql_weight,
        online_cql_mode=args.online_cql_mode,
        num_envs=args.num_envs,
        env_workers=args.env_workers,
        reward_mode=args.reward_mode,
        sparse_reward_weight=args.sparse_reward_weight,
        dense_reward_weight=args.dense_reward_weight,
        dense_reward_scale_m=args.dense_reward_scale_m,
        offline_replay_ratio=args.offline_replay_ratio,
        actor_update_interval=args.actor_update_interval,
        q_cost_anchor_weight=args.q_cost_anchor_weight,
        p_cost_anchor_weight=args.p_cost_anchor_weight,
        cost_anchor_gradient_diagnostics=args.anchor_gradient_diagnostics,
    )
    config.validate()
    actor_offline_replay_ratio = (
        float(args.offline_replay_ratio)
        if args.actor_offline_replay_ratio is None
        else float(args.actor_offline_replay_ratio)
    )
    # The critic's fused UTD batch must keep an exact fixed ratio.  An actor
    # batch can request fractions such as 10% of 256, which is not integral.
    # Use unbiased stochastic rounding (25 or 26 offline samples here) so the
    # long-run actor measure is exactly the requested one without changing its
    # batch size or the critic batch.
    actor_ratio_times_batch = actor_offline_replay_ratio * config.batch_size
    actor_offline_floor = int(np.floor(actor_ratio_times_batch))
    actor_offline_fraction = actor_ratio_times_batch - actor_offline_floor
    dataset = ManiSoftCircleOfflineDataset(
        args.dataset,
        args.koopman,
        reference_path=args.reference,
        reward_mode=config.reward_mode,
        sparse_reward_weight=config.sparse_reward_weight,
        dense_reward_weight=config.dense_reward_weight,
        dense_reward_scale_m=config.dense_reward_scale_m,
        reward_radius_m=REWARD_RADIUS_M,
        gamma=config.discount,
    )
    offline_residual_abs = np.abs(np.asarray(dataset.arrays["action"]))
    offline_residual_p95 = float(np.quantile(offline_residual_abs, 0.95))
    offline_residual_max = float(offline_residual_abs.max())
    episode_step_array = np.asarray(dataset.arrays["episode_step"], dtype=np.int64)
    offline_physical = np.asarray(dataset.policy_observations[:, :45], dtype=np.float64)
    support_center = np.empty((1000, 45), dtype=np.float64)
    support_scale = np.empty((1000, 45), dtype=np.float64)
    support_reference = np.empty((1000, int(len(dataset) / 1000)), dtype=np.float64)
    global_scale = np.maximum(offline_physical.std(axis=0), 1e-6)
    for phase_index in range(1000):
        phase_states = offline_physical[episode_step_array == phase_index]
        support_center[phase_index] = phase_states.mean(axis=0)
        support_scale[phase_index] = np.maximum(
            phase_states.std(axis=0), 0.05 * global_scale
        )
        phase_score = np.sqrt(
            np.mean(
                ((phase_states - support_center[phase_index])
                 / support_scale[phase_index]) ** 2,
                axis=1,
            )
        )
        support_reference[phase_index] = np.sort(phase_score)
    if config.requires_koopman:
        koopman: FrozenManiSoftHistoryKoopman | None = (
            FrozenManiSoftHistoryKoopman(args.koopman)
        )
        normalizer = None
    else:
        koopman = None
        normalizer = FrozenObservationNormalizer.from_offline_observations(
            dataset.policy_observations, dataset_sha256=dataset.sha256
        )
    learner = O2OLearner(
        config, koopman, device, observation_normalizer=normalizer
    )
    learner.actor_entropy_enabled = not args.disable_actor_entropy
    replay = OnlineReplay(
        config.replay_capacity,
        obs_dim=COLLECTOR_OBSERVATION_DIM,
        action_dim=18,
    )
    protocol_adapter = make_manisoft_circle_o2o_adapter(TASK_NAME, seed=args.seed)
    environment_protocol = protocol_adapter.protocol_metadata()
    protocol_adapter.close()

    offline_update = 0
    online_step = 0
    online_episode = 0
    best_return = float("-inf")
    best_evaluation: dict[str, Any] | None = None
    bootstrapped_online = False
    bootstrap_metadata: dict[str, Any] | None = None
    continued_online = False
    continuation_metadata: dict[str, Any] | None = None
    online_joint_errors: list[float] = []
    online_residual_abs: list[np.ndarray] = []
    online_support_scores: list[float] = []
    online_support_percentiles: list[float] = []
    warmup_buffer_episodes: list[dict[str, Any]] = []
    actor_update_statistics: list[dict[str, float]] = []
    online_evaluation_history: list[dict[str, Any]] = []
    online_start_evaluation: dict[str, Any] | None = None
    online_early_stop: dict[str, Any] | None = None
    latest_path = output / "latest.pt"
    if latest_path.is_file() and (
        args.bootstrap_checkpoint is not None
        or args.continue_online_checkpoint is not None
    ):
        raise ValueError("Cannot import a checkpoint into an output with latest.pt")
    if latest_path.is_file():
        payload = torch.load(latest_path, map_location="cpu", weights_only=False)
        expected = {
            "kind": CHECKPOINT_KIND,
            "config_fingerprint": config.fingerprint,
            "dataset_sha": dataset.sha256,
            "koopman": None if koopman is None else koopman.identity(),
            "environment_protocol": environment_protocol,
        }
        actual = {
            "kind": payload.get("kind"),
            "config_fingerprint": payload.get("config_fingerprint"),
            "dataset_sha": payload.get("dataset", {}).get("sha256"),
            "koopman": payload.get("koopman"),
            "environment_protocol": payload.get("environment_protocol"),
        }
        if actual != expected:
            raise ValueError("Resume identity differs from the formal protocol")
        learner.load_state_dict(payload["learner"])
        # An explicit offline->online fork may intentionally change optimizer
        # schedules. Optimizer state restoration otherwise overwrites the
        # target config's learning rates with source-checkpoint values.
        for optimizer, learning_rate in (
            (learner.actor_optimizer, config.actor_learning_rate),
            (learner.critic_optimizer, config.critic_learning_rate),
            (learner.temperature_optimizer, config.temperature_learning_rate),
        ):
            for group in optimizer.param_groups:
                group["lr"] = float(learning_rate)
        if payload.get("online_replay") is not None:
            replay.load_state_dict(payload["online_replay"])
        restore_rng(payload["rng"], generator)
        offline_update = int(payload["offline_update"])
        online_step = int(payload["online_step"])
        online_episode = int(payload["online_episode"])
        best_return = float(payload["best_return"])
        best_evaluation = payload.get("best_evaluation")
        _truncate_metrics(
            metrics_path, offline_update=offline_update, online_step=online_step
        )
    elif args.continue_online_checkpoint is not None:
        continuation_path = args.continue_online_checkpoint.expanduser().resolve()
        payload = torch.load(continuation_path, map_location="cpu", weights_only=False)
        expected = {
            "kind": CHECKPOINT_KIND,
            "dataset_sha": dataset.sha256,
            "koopman": None if koopman is None else koopman.identity(),
            "environment_protocol": environment_protocol,
        }
        actual = {
            "kind": payload.get("kind"),
            "dataset_sha": payload.get("dataset", {}).get("sha256"),
            "koopman": payload.get("koopman"),
            "environment_protocol": payload.get("environment_protocol"),
        }
        source_config = payload.get("config")
        if (
            actual != expected
            or not isinstance(source_config, dict)
            or not _bootstrap_config_compatible(source_config, config.to_dict())
        ):
            source_semantics = (
                {
                    key: value
                    for key, value in source_config.items()
                    if key not in _BOOTSTRAP_SCHEDULE_FIELDS
                }
                if isinstance(source_config, dict)
                else source_config
            )
            target_semantics = {
                key: value
                for key, value in config.to_dict().items()
                if key not in _BOOTSTRAP_SCHEDULE_FIELDS
            }
            raise ValueError(
                "Online continuation identity differs from the protocol: "
                f"identity_actual={actual!r}, identity_expected={expected!r}, "
                f"source_semantics={source_semantics!r}, "
                f"target_semantics={target_semantics!r}"
            )
        if payload.get("phase") != "online":
            raise ValueError("Online continuation requires an online checkpoint")
        source_online_step = int(payload.get("online_step", -1))
        if not 0 < source_online_step < args.online_steps:
            raise ValueError("Continuation step must lie inside the target schedule")
        source_replay = payload.get("online_replay")
        if not isinstance(source_replay, dict):
            raise ValueError("Online continuation checkpoint has no replay state")
        if int(source_replay.get("capacity", -1)) != config.replay_capacity:
            raise ValueError(
                "Continuation replay capacity must remain identical; pass "
                "--replay-capacity matching the source checkpoint"
            )
        learner.load_state_dict(payload["learner"])
        replay.load_state_dict(source_replay)
        restore_rng(payload["rng"], generator)
        offline_update = int(payload["offline_update"])
        online_step = source_online_step
        online_episode = int(payload["online_episode"])
        best_return = float(payload["best_return"])
        best_evaluation = payload.get("best_evaluation")
        continued_online = True
        continuation_metadata = {
            "path": str(continuation_path),
            "sha256": _sha256_file(continuation_path),
            "source_online_step": source_online_step,
            "source_replay_size": int(source_replay["size"]),
            "source_replay_capacity": int(source_replay["capacity"]),
            "target_online_steps": int(args.online_steps),
            "schedule_fields": sorted(_BOOTSTRAP_SCHEDULE_FIELDS),
        }
        restored_learner_state = learner.state_dict()
        source_learner_state = payload["learner"]
        continuation_identity = {
            "actor": (
                _canonical_state_sha256(source_learner_state["actor"]),
                _canonical_state_sha256(restored_learner_state["actor"]),
            ),
            "critic": (
                _canonical_state_sha256(source_learner_state["critic"]),
                _canonical_state_sha256(restored_learner_state["critic"]),
            ),
            "target_critic": (
                _canonical_state_sha256(source_learner_state["target_critic"]),
                _canonical_state_sha256(restored_learner_state["target_critic"]),
            ),
            "actor_optimizer": (
                _canonical_state_sha256(source_learner_state["actor_optimizer"]),
                _canonical_state_sha256(restored_learner_state["actor_optimizer"]),
            ),
            "critic_optimizer": (
                _canonical_state_sha256(source_learner_state["critic_optimizer"]),
                _canonical_state_sha256(restored_learner_state["critic_optimizer"]),
            ),
            "learner_sampling_rng": (
                _canonical_state_sha256(
                    source_learner_state["rng_substreams"]["training_sampling_state"]
                ),
                _canonical_state_sha256(
                    restored_learner_state["rng_substreams"]["training_sampling_state"]
                ),
            ),
            "online_replay": (
                _canonical_state_sha256(source_replay),
                _canonical_state_sha256(replay.state_dict()),
            ),
            "outer_rng": (
                _canonical_state_sha256(payload["rng"]),
                _canonical_state_sha256(rng_state(generator)),
            ),
        }
        continuation_metadata["component_identity"] = {
            name: {
                "source_sha256": pair[0],
                "restored_sha256": pair[1],
                "identical": pair[0] == pair[1],
            }
            for name, pair in continuation_identity.items()
        }
        if not all(pair[0] == pair[1] for pair in continuation_identity.values()):
            raise RuntimeError("Online continuation did not restore every source component")
        # The continuation identity must be checked before applying an
        # explicitly requested optimizer schedule change.  Otherwise changing
        # only actor LR (the causal variable in a screen) mutates the restored
        # optimizer param group and falsely looks like a broken checkpoint.
        for optimizer, learning_rate in (
            (learner.actor_optimizer, config.actor_learning_rate),
            (learner.critic_optimizer, config.critic_learning_rate),
            (learner.temperature_optimizer, config.temperature_learning_rate),
        ):
            for group in optimizer.param_groups:
                group["lr"] = float(learning_rate)
    elif args.bootstrap_checkpoint is not None:
        bootstrap_path = args.bootstrap_checkpoint.expanduser().resolve()
        payload = torch.load(bootstrap_path, map_location="cpu", weights_only=False)
        expected = {
            "kind": CHECKPOINT_KIND,
            "config_fingerprint": config.fingerprint,
            "dataset_sha": dataset.sha256,
            "koopman": None if koopman is None else koopman.identity(),
            "environment_protocol": environment_protocol,
        }
        actual = {
            "kind": payload.get("kind"),
            "config_fingerprint": payload.get("config_fingerprint"),
            "dataset_sha": payload.get("dataset", {}).get("sha256"),
            "koopman": payload.get("koopman"),
            "environment_protocol": payload.get("environment_protocol"),
        }
        identity_matches = actual == expected
        dataset_identity_matches = actual["dataset_sha"] == expected["dataset_sha"]
        if args.bootstrap_allow_dataset_mismatch:
            identity_matches = (
                actual["kind"] == expected["kind"]
                and actual["config_fingerprint"] == expected["config_fingerprint"]
                and actual["koopman"] == expected["koopman"]
                and actual["environment_protocol"] == expected["environment_protocol"]
            )
        if not identity_matches and args.bootstrap_actor_only:
            identity_matches = (
                actual["kind"] == expected["kind"]
                and (dataset_identity_matches or args.bootstrap_allow_dataset_mismatch)
                and actual["koopman"] == expected["koopman"]
                and actual["environment_protocol"] == expected["environment_protocol"]
            )
        if not identity_matches and args.bootstrap_allow_schedule_change:
            source_config = payload.get("config")
            identity_matches = (
                isinstance(source_config, dict)
                and actual["kind"] == expected["kind"]
                and (dataset_identity_matches or args.bootstrap_allow_dataset_mismatch)
                and actual["koopman"] == expected["koopman"]
                and actual["environment_protocol"] == expected["environment_protocol"]
                and _bootstrap_config_compatible(source_config, config.to_dict())
            )
        if not identity_matches:
            raise ValueError("Bootstrap checkpoint identity differs from the protocol")
        if payload.get("phase") != "offline" or int(payload.get("online_step", -1)) != 0:
            raise ValueError("Online bootstrap requires an offline checkpoint at online step zero")
        if args.bootstrap_actor_only:
            # Method migration used by AWAC->Cal-RLPD diagnostics: preserve
            # the structured controller exactly while keeping the freshly
            # constructed critic, target critic, temperature, optimizers,
            # replay, and RNG state.
            if args.bootstrap_preserve_actor:
                learner.actor.load_state_dict(
                    payload["learner"]["actor"], strict=True
                )
            else:
                load_offline_base = getattr(
                    learner.actor, "load_offline_base_state_dict", None
                )
                if callable(load_offline_base):
                    load_offline_base(payload["learner"]["actor"])
                else:
                    learner.actor.load_state_dict(
                        payload["learner"]["actor"], strict=True
                    )
        else:
            learner.load_state_dict(payload["learner"])
            for optimizer, learning_rate in (
                (learner.actor_optimizer, config.actor_learning_rate),
                (learner.critic_optimizer, config.critic_learning_rate),
                (learner.temperature_optimizer, config.temperature_learning_rate),
            ):
                for group in optimizer.param_groups:
                    group["lr"] = float(learning_rate)
            restore_rng(payload["rng"], generator)
        offline_update = int(payload["offline_update"])
        if offline_update < 0:
            raise ValueError("Bootstrap checkpoint has a negative offline update")
        if offline_update == 0 and not args.bootstrap_allow_schedule_change:
            raise ValueError(
                "A zero-update bootstrap requires --bootstrap-allow-schedule-change"
            )
        best_return = float(payload["best_return"])
        # Re-evaluate the imported policy at online step zero so the new run
        # owns a local best.pt and a directly comparable evaluation record.
        best_evaluation = None
        bootstrapped_online = True
        bootstrap_metadata = {
            "path": str(bootstrap_path),
            "sha256": _sha256_file(bootstrap_path),
            "source_phase": payload.get("phase"),
            "source_offline_update": offline_update,
            "source_online_step": 0,
            "source_best_evaluation": payload.get("best_evaluation"),
            "actor_only": bool(args.bootstrap_actor_only),
            "dataset_identity_matches": bool(dataset_identity_matches),
            "dataset_mismatch_allowed": bool(args.bootstrap_allow_dataset_mismatch),
            "source_dataset_sha256": payload.get("dataset", {}).get("sha256"),
            "target_dataset_sha256": dataset.sha256,
            "schedule_change_allowed": bool(args.bootstrap_allow_schedule_change),
            "schedule_fields": sorted(_BOOTSTRAP_SCHEDULE_FIELDS),
        }

    validate_bootstrap_actor = getattr(
        learner.actor, "validate_bootstrap_zero_residual", None
    )
    if (
        callable(validate_bootstrap_actor)
        and bootstrapped_online
        and not args.bootstrap_preserve_actor
    ):
        probe_indices = np.linspace(
            0, len(dataset) - 1, num=32, dtype=np.int64
        )
        probe_observations = torch.as_tensor(
            dataset.arrays["observation"][probe_indices],
            dtype=torch.float32,
            device=device,
        )
        bootstrap_zero_diagnostics = validate_bootstrap_actor(
            learner._encode(probe_observations)
        )
        _atomic_json(
            output / "bootstrap_zero_residual_sanity.json",
            bootstrap_zero_diagnostics,
        )
    elif (
        callable(validate_bootstrap_actor)
        and not continued_online
        and not bootstrapped_online
    ):
        raise ValueError(
            "Frozen-base residual actor requires an actor-only offline bootstrap"
        )
    else:
        bootstrap_zero_diagnostics = None

    if args.awac_selectivity_mode is not None:
        if not continued_online or args.continue_online_checkpoint is None:
            raise ValueError(
                "Selective AWAC is only valid for a complete online continuation"
            )
        if replay.size < 1:
            raise ValueError("Selective AWAC continuation requires a non-empty replay")
        probe_count = min(256, replay.size)
        probe_indices = np.linspace(
            0, replay.size - 1, num=probe_count, dtype=np.int64
        )
        learner.configure_awac_selectivity(
            mode=args.awac_selectivity_mode,
            reference_kl_weight=args.awac_reference_kl_weight,
            probe_observations=replay.arrays["observation"][probe_indices],
        )

    run_metadata = {
        "kind": "acmpc_manisoft_circle_formal_o2o_run_v1",
        "completed": False,
        "method": args.method,
        "training_seed": args.seed,
        "formal_methods": list(FORMAL_METHODS),
        "config": config.to_dict(),
        "config_fingerprint": config.fingerprint,
        "dataset": {"path": str(dataset.path), "sha256": dataset.sha256},
        "koopman": None if koopman is None else koopman.identity(),
        "environment_protocol": environment_protocol,
        "representation_contract": (
            "history-aware frozen Koopman lift"
            if config.requires_koopman
            else "normalized physical_state_45 + time_1 only"
        ),
        "actor_action_semantics": (
            "residual_u_in_[-0.5,0.5]; physical_u=u_ff+residual, "
            "physical_u_clipped_to_[-0.5,0.5]"
        ),
        "offline_updates": args.offline_updates,
        "online_steps": args.online_steps,
        "kmpc_horizon": args.kmpc_horizon,
        "online_bootstrap": bootstrap_metadata,
        "online_continuation": continuation_metadata,
        "online_actor_diagnostics": {
            "critic_only_steps": args.online_critic_only_steps,
            "actor_entropy_enabled": not args.disable_actor_entropy,
            "critic_offline_replay_ratio": float(config.offline_replay_ratio),
            "actor_offline_replay_ratio": actor_offline_replay_ratio,
            "actor_offline_samples_floor": actor_offline_floor,
            "actor_offline_samples_fractional_probability": actor_offline_fraction,
            "buffer_quality_steps": int(args.online_buffer_quality_steps),
            "awac_selectivity_mode": args.awac_selectivity_mode,
            "awac_reference_kl_weight": float(args.awac_reference_kl_weight),
            "awac_reference_probe_size": (
                min(256, replay.size)
                if args.awac_selectivity_mode is not None
                else 0
            ),
            "selected_episode_quality_available": False,
            "selected_episode_quality_note": (
                "Source A3 replay schema stores transitions without episode IDs; "
                "selected/rejected transition reward is logged instead."
                if args.awac_selectivity_mode is not None
                else None
            ),
            "bootstrap_zero_residual_sanity": bootstrap_zero_diagnostics,
        },
        "best_checkpoint_selection": {
            "priority": [
                "safety_constraints",
                "action_saturation_fraction",
                "delta_action_p95_abs",
                "joint_error_rmse_m",
                "reward_rate",
            ],
            "max_action_saturation_fraction": BEST_MAX_SATURATION_FRACTION,
            "max_delta_action_p95_abs": BEST_MAX_DELTA_P95_ABS,
        },
    }
    _atomic_json(output / "run.json", run_metadata)

    def save(path: Path, *, phase: str, include_replay: bool) -> None:
        atomic_torch_save(
            path,
            _checkpoint(
                config=config,
                dataset=dataset,
                koopman=koopman,
                learner=learner,
                replay=replay,
                generator=generator,
                environment_protocol=environment_protocol,
                phase=phase,
                offline_update=offline_update,
                online_step=online_step,
                online_episode=online_episode,
                best_return=best_return,
                best_evaluation=best_evaluation,
                include_replay=include_replay,
            ),
        )

    def evaluate_and_save(phase: str, counter: int) -> dict[str, Any]:
        nonlocal best_return, best_evaluation
        continuation_zero = bool(
            continued_online
            and phase == "online"
            and continuation_metadata is not None
            and counter == int(continuation_metadata["source_online_step"])
        )
        checkpoint_eligible = counter > 0 and not continuation_zero
        evaluation = evaluate(
            learner,
            seed_base=int(os.environ.get("ACMPC_EVAL_SEED_BASE", "9000000")),
            episodes=args.eval_episodes,
        )
        row = {
            "phase": f"{phase}_evaluation",
            "offline_update": offline_update,
            "online_step": online_step,
            **evaluation,
        }
        if phase == "online" and args.awac_selectivity_mode is not None:
            row.update(learner.awac_reference_probe_diagnostics())
        if phase == "online":
            row.update(
                online_replay_transitions=len(online_joint_errors),
                online_replay_error_rmse_m=(
                    float(np.sqrt(np.mean(np.square(online_joint_errors))))
                    if online_joint_errors else 0.0
                ),
                online_replay_error_p95_m=(
                    float(np.quantile(online_joint_errors, 0.95))
                    if online_joint_errors else 0.0
                ),
                online_replay_residual_p95_abs=(
                    float(np.quantile(np.concatenate(online_residual_abs), 0.95))
                    if online_residual_abs else 0.0
                ),
                online_replay_residual_max_abs=(
                    float(np.max(np.concatenate(online_residual_abs)))
                    if online_residual_abs else 0.0
                ),
                offline_dataset_residual_p95_abs=offline_residual_p95,
                offline_dataset_residual_max_abs=offline_residual_max,
                online_state_support_score_mean=(
                    float(np.mean(online_support_scores))
                    if online_support_scores else 0.0
                ),
                online_state_support_score_p95=(
                    float(np.quantile(online_support_scores, 0.95))
                    if online_support_scores else 0.0
                ),
                online_state_support_percentile_mean=(
                    float(np.mean(online_support_percentiles))
                    if online_support_percentiles else 0.0
                ),
                online_state_support_fraction_above_offline_p95=(
                    float(np.mean(np.asarray(online_support_percentiles) > 0.95))
                    if online_support_percentiles else 0.0
                ),
                online_state_support_fraction_above_offline_p99=(
                    float(np.mean(np.asarray(online_support_percentiles) > 0.99))
                    if online_support_percentiles else 0.0
                ),
            )
        is_best = checkpoint_eligible and (
            best_evaluation is None
            or _checkpoint_selection_key(evaluation)
            < _checkpoint_selection_key(best_evaluation)
        )
        best_return = max(best_return, float(evaluation["return_mean"]))
        if is_best:
            best_evaluation = dict(row)
        _append_jsonl(metrics_path, row)
        # Keep offline and online milestone cadences independent: the
        # protocol may evaluate/save offline every N updates while using a
        # shorter online cadence for closed-loop drift detection.
        checkpoint_save_interval = (
            args.offline_eval_interval
            if phase == "offline"
            else args.checkpoint_save_interval
        )
        save_milestone = (
            checkpoint_eligible
            and (
                checkpoint_save_interval is None
                or counter % checkpoint_save_interval == 0
            )
        )
        if save_milestone:
            checkpoint_path = output / f"{phase}_{counter:06d}.pt"
            save(checkpoint_path, phase=phase, include_replay=False)
        _atomic_json(
            output / f"evaluation_{phase}_{counter:06d}.json", row
        )
        if is_best:
            save(output / "best.pt", phase=phase, include_replay=False)
            _atomic_json(output / "best_evaluation.json", best_evaluation)
        synchronized_episode_span = 1000 * config.num_envs
        recovery_due = checkpoint_eligible and (
            checkpoint_save_interval is None
            or counter % checkpoint_save_interval == 0
        )
        if recovery_due and (
            phase == "offline" or counter % synchronized_episode_span == 0
        ):
            # A simulator state is intentionally not serialized.  Recovery
            # checkpoints therefore advance only at synchronized resets;
            # intermediate milestone weights remain fully evaluable.  Update
            # zero is evaluation-only, and explicit checkpoint cadence also
            # governs latest.pt so a 2k protocol never emits 0k/1k weights.
            save(latest_path, phase=phase, include_replay=True)
        print(json.dumps(row, sort_keys=True), flush=True)
        return row

    started = time.time()
    if config.uses_offline_pretraining and not bootstrapped_online:
        # Persist the initialized policy before the first offline gradient.
        # This is essential for actor-initialization studies: update zero is
        # the only measurement that separates BC transfer from offline RL.
        if offline_update == 0 and not (output / "offline_000000.pt").is_file():
            evaluate_and_save("offline", 0)
        while offline_update < args.offline_updates:
            batch_np = mark_offline(dataset.sample(config.batch_size, generator))
            metrics = learner.update(
                TensorBatch.from_numpy(batch_np, device), utd=1, phase="offline"
            )
            offline_update += 1
            if offline_update % config.log_interval_updates == 0:
                _append_jsonl(
                    metrics_path,
                    {
                        "phase": "offline_update",
                        "offline_update": offline_update,
                        "online_step": online_step,
                        **metrics,
                    },
                )
            if offline_update % args.offline_eval_interval == 0:
                evaluate_and_save("offline", offline_update)
    if args.stop_after_offline:
        run_metadata.update(
            completed=True,
            completion_scope="offline_screen",
            offline_updates_completed=offline_update,
            online_steps_completed=0,
            online_episodes_completed=0,
            best_return=best_return,
            best_evaluation=best_evaluation,
            wall_time_seconds=time.time() - started,
        )
        _atomic_json(output / "run.json", run_metadata)
        return

    continuation_evaluation_path = output / f"online_{online_step:06d}.pt"
    if continued_online and not continuation_evaluation_path.is_file():
        continuation_evaluation = evaluate_and_save("online", online_step)
        online_start_evaluation = continuation_evaluation
        online_evaluation_history.append(continuation_evaluation)
    elif not continued_online and not (output / "online_000000.pt").is_file():
        if args.skip_online_zero_eval:
            assert bootstrapped_online
            assert args.online_zero_reference_evaluation is not None
            reference_path = args.online_zero_reference_evaluation
            online_start_evaluation = json.loads(reference_path.read_text())
            online_evaluation_history.append(online_start_evaluation)
            assert bootstrap_metadata is not None
            bootstrap_metadata["online_zero_verification"] = {
                "skipped_duplicate_rollout": True,
                "reference_evaluation_path": str(reference_path),
                "reference_evaluation_sha256": _sha256_file(reference_path),
                "reference_tip_rmse_m": float(
                    online_start_evaluation["tip_rmse_m"]
                ),
                "reference_tip_p95_m": float(
                    online_start_evaluation["tip_p95_m"]
                ),
            }
            run_metadata["online_bootstrap"] = bootstrap_metadata
            _atomic_json(output / "run.json", run_metadata)
        else:
            online_zero = evaluate_and_save("online", 0)
            online_start_evaluation = online_zero
            online_evaluation_history.append(online_zero)
        if (
            not args.skip_online_zero_eval
            and bootstrapped_online
            and args.bootstrap_rmse_tolerance is not None
        ):
            assert bootstrap_metadata is not None
            source_best = bootstrap_metadata.get("source_best_evaluation")
            if not isinstance(source_best, dict):
                raise ValueError("Bootstrap checkpoint has no source best evaluation")
            source_rmse = float(source_best["joint_error_rmse_m"])
            online_zero_rmse = float(online_zero["joint_error_rmse_m"])
            relative_error = abs(online_zero_rmse - source_rmse) / source_rmse
            bootstrap_metadata["online_zero_verification"] = {
                "source_rmse_m": source_rmse,
                "online_zero_rmse_m": online_zero_rmse,
                "relative_error": relative_error,
                "tolerance": float(args.bootstrap_rmse_tolerance),
                "passed": relative_error <= args.bootstrap_rmse_tolerance,
            }
            run_metadata["online_bootstrap"] = bootstrap_metadata
            _atomic_json(output / "run.json", run_metadata)
            if relative_error > args.bootstrap_rmse_tolerance:
                raise RuntimeError(
                    "Online-zero RMSE does not reproduce the source best; "
                    "online collection is forbidden"
                )
        if (
            not args.skip_online_zero_eval
            and bootstrapped_online
            and args.bootstrap_return_target is not None
        ):
            return_error = abs(
                float(online_zero["return_mean"])
                - float(args.bootstrap_return_target)
            )
            return_check = {
                "target": float(args.bootstrap_return_target),
                "observed": float(online_zero["return_mean"]),
                "absolute_error": return_error,
                "tolerance": float(args.bootstrap_return_tolerance),
                "passed": return_error <= args.bootstrap_return_tolerance,
            }
            assert bootstrap_metadata is not None
            bootstrap_metadata["online_zero_return_verification"] = return_check
            run_metadata["online_bootstrap"] = bootstrap_metadata
            _atomic_json(output / "run.json", run_metadata)
            if not return_check["passed"]:
                raise RuntimeError(
                    "Online-zero return does not reproduce the requested source; "
                    "online collection is forbidden"
                )

    if args.stop_after_continuation_eval:
        if not continued_online:
            raise ValueError(
                "stop_after_continuation_eval requires continue_online_checkpoint"
            )
        run_metadata.update(
            completed=True,
            completion_scope="online_continuation_evaluation_only",
            offline_updates_completed=offline_update,
            online_steps_completed=online_step,
            online_episodes_completed=online_episode,
            best_return=best_return,
            best_evaluation=best_evaluation,
            wall_time_seconds=time.time() - started,
        )
        _atomic_json(output / "run.json", run_metadata)
        return

    if args.stop_after_online_zero_eval:
        if continued_online:
            raise ValueError(
                "stop_after_online_zero_eval is only valid for a fresh online run"
            )
        run_metadata.update(
            completed=True,
            completion_scope="online_zero_evaluation_only",
            offline_updates_completed=offline_update,
            online_steps_completed=0,
            online_episodes_completed=online_episode,
            best_return=best_return,
            best_evaluation=best_evaluation,
            wall_time_seconds=time.time() - started,
        )
        _atomic_json(output / "run.json", run_metadata)
        return

    if args.profile_continuation_update_memory:
        if not continued_online or device.type != "cuda":
            raise ValueError(
                "profile_continuation_update_memory requires CUDA online continuation"
            )
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        profile_batch = mixed_batch(
            dataset,
            replay,
            batch_size=config.batch_size,
            utd=config.online_utd,
            offline_ratio=config.offline_replay_ratio,
            generator=generator,
        )
        profile_metrics = learner.update(
            TensorBatch.from_numpy(profile_batch, device),
            utd=config.online_utd,
            phase="online",
        )
        torch.cuda.synchronize(device)
        memory_profile = {
            "kind": "online_continuation_single_update_cuda_memory_v1",
            "source_online_step": online_step,
            "batch_size": config.batch_size,
            "offline_replay_ratio": config.offline_replay_ratio,
            "allocated_mib": torch.cuda.memory_allocated(device) / 2**20,
            "reserved_mib": torch.cuda.memory_reserved(device) / 2**20,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
            "update_metrics": profile_metrics,
        }
        _atomic_json(output / "cuda_memory_profile.json", memory_profile)
        run_metadata.update(
            completed=True,
            completion_scope="online_continuation_single_update_memory_profile",
            offline_updates_completed=offline_update,
            online_steps_completed=online_step,
            online_episodes_completed=online_episode,
            memory_profile=memory_profile,
            wall_time_seconds=time.time() - started,
        )
        _atomic_json(output / "run.json", run_metadata)
        print(json.dumps(memory_profile, sort_keys=True), flush=True)
        return

    expert_warmup = None
    if args.expert_warmup_steps > 0:
        current_feedforward = os.environ.get("ACMPC_MANISOFT_CIRCLE_FEEDFORWARD")
        if not current_feedforward:
            raise RuntimeError(
                "expert warmup requires ACMPC_MANISOFT_CIRCLE_FEEDFORWARD"
            )
        expert_warmup = _ExpertWarmupPolicy(
            checkpoint=args.expert_warmup_checkpoint.expanduser().resolve(),
            vecnormalize=args.expert_warmup_vecnormalize.expanduser().resolve(),
            expert_reference=(
                args.expert_warmup_reference.expanduser().resolve()
                if args.expert_warmup_reference is not None
                else Path(args.reference).expanduser().resolve()
            ),
            current_feedforward=Path(current_feedforward).expanduser().resolve(),
            expert_residual_limit=args.expert_residual_limit,
            perturbation_std=args.expert_perturbation_std,
            perturbation_limit=args.expert_perturbation_limit,
            seed=args.seed + 7_000_000,
        )
        _atomic_json(
            output / "expert_warmup.json",
            {
                "checkpoint": str(args.expert_warmup_checkpoint.expanduser().resolve()),
                "vecnormalize": str(args.expert_warmup_vecnormalize.expanduser().resolve()),
                "expert_reference": str(
                    (args.expert_warmup_reference or Path(args.reference)).expanduser().resolve()
                ),
                "steps": int(args.expert_warmup_steps),
                "expert_residual_limit": float(args.expert_residual_limit),
                "perturbation_std": float(args.expert_perturbation_std),
                "perturbation_limit": float(args.expert_perturbation_limit),
                "actor_frozen_during_warmup": bool(
                    args.online_critic_only_steps >= args.expert_warmup_steps
                ),
            },
        )

    if online_step < args.online_steps:
        env = _environment(
            config.num_envs,
            args.seed + 100_000 + online_episode,
            workers=config.env_workers,
        )
        try:
            observation = env.reset()
            if expert_warmup is not None:
                expert_warmup.reset(config.num_envs)
            episode_returns = np.zeros(config.num_envs, dtype=np.float64)
            episode_lengths = np.zeros(config.num_envs, dtype=np.int64)
            episode_joint_errors: list[list[float]] = [
                [] for _ in range(config.num_envs)
            ]
            episode_tip_errors: list[list[float]] = [
                [] for _ in range(config.num_envs)
            ]
            episode_collectors: list[set[str]] = [
                set() for _ in range(config.num_envs)
            ]
            pending = (
                {key: [] for key in ("observation", "action", "reward", "discount", "next_observation")}
                if config.requires_online_mc_returns
                else None
            )

            def online_update_once() -> dict[str, float]:
                # Critic and actor intentionally draw independently.  The
                # critic keeps its fixed RLPD 50/50 distribution while an
                # AWAC screen can vary only the data distribution used by
                # weighted behaviour cloning.
                critic_batch_np = mixed_batch(
                    dataset,
                    replay,
                    batch_size=config.batch_size,
                    utd=config.online_utd,
                    offline_ratio=config.offline_replay_ratio,
                    generator=generator,
                )
                actor_offline_samples = actor_offline_floor + int(
                    generator.random() < actor_offline_fraction
                )
                actor_batch_np = mixed_batch(
                    dataset,
                    replay,
                    batch_size=config.batch_size,
                    utd=1,
                    offline_ratio=actor_offline_samples / config.batch_size,
                    generator=generator,
                )
                return learner.update(
                    TensorBatch.from_numpy(critic_batch_np, device),
                    utd=config.online_utd,
                    phase="online",
                    actor_updates_enabled=(
                        online_step > args.online_critic_only_steps
                    ),
                    actor_batch=TensorBatch.from_numpy(actor_batch_np, device),
                )

            while online_step < args.online_steps:
                random_warmup = (
                    not config.uses_offline_pretraining
                    and online_step < config.online_warmup_steps
                )
                phase_indices = episode_lengths.astype(np.int64) % 1000
                use_expert = (
                    expert_warmup is not None
                    and online_step < args.expert_warmup_steps
                )
                action = (
                    expert_warmup.act(observation, phase_indices)
                    if use_expert
                    else (
                    generator.uniform(
                        -ABSOLUTE_ACTION_LIMIT,
                        ABSOLUTE_ACTION_LIMIT,
                        size=(config.num_envs, 18),
                    ).astype(np.float32)
                    if random_warmup
                    else learner.act(observation, deterministic=False)
                    )
                )
                vector_step = env.step(action)
                for index in range(config.num_envs):
                    phase_index = int(episode_lengths[index]) % 1000
                    physical_state = np.asarray(
                        observation[index, :45], dtype=np.float64
                    )
                    support_score = float(
                        np.sqrt(
                            np.mean(
                                ((physical_state - support_center[phase_index])
                                 / support_scale[phase_index]) ** 2
                            )
                        )
                    )
                    online_support_scores.append(support_score)
                    online_support_percentiles.append(
                        float(
                            np.searchsorted(
                                support_reference[phase_index],
                                support_score,
                                side="right",
                            )
                            / support_reference.shape[1]
                        )
                    )
                    online_joint_errors.append(
                        float(vector_step.info[index]["joint_target_error"])
                    )
                    episode_joint_errors[index].append(
                        float(vector_step.info[index]["joint_target_error"])
                    )
                    episode_tip_errors[index].append(
                        float(vector_step.info[index]["node_target_error"][2])
                    )
                    episode_collectors[index].add(
                        "expert_ppo" if use_expert else "source_actor"
                    )
                    online_residual_abs.append(
                        np.abs(
                            np.asarray(
                                vector_step.applied_action[index],
                                dtype=np.float64,
                            )
                        )
                    )
                episode_returns += vector_step.reward
                episode_lengths += 1
                completed_rows = []
                for index in range(config.num_envs):
                    transition = {
                        "observation": observation[index].copy(),
                        "action": vector_step.applied_action[index].copy(),
                        "reward": float(vector_step.reward[index]),
                        "discount": float(vector_step.discount[index]),
                        "next_observation": vector_step.transition_observation[index].copy(),
                    }
                    if pending is None:
                        replay.add(**transition)
                    else:
                        for key in pending:
                            pending[key].append(transition[key])
                    online_step += 1
                    # ManiSoft's warmup is an explicit collection-only window
                    # even after offline pretraining.  The legacy default is
                    # zero, so old runs retain their exact update schedule.
                    ready = online_step >= config.online_warmup_steps
                    metrics = online_update_once() if ready and pending is None else {}
                    if "selected_sample_count" in metrics:
                        actor_update_statistics.append(
                            {
                                key: float(value)
                                for key, value in metrics.items()
                                if key.startswith(
                                    (
                                        "advantage_",
                                        "selected_",
                                        "rejected_",
                                        "actor_",
                                        "awac_",
                                        "kl_",
                                        "shared_trunk_",
                                        "q_state_head_",
                                        "action_p_head_",
                                    )
                                )
                            }
                        )
                    if vector_step.reset_boundary[index]:
                        if pending is not None:
                            transition_count = len(pending["reward"])
                            replay.add_episode(
                                np.asarray(pending["observation"]),
                                np.asarray(pending["action"]),
                                np.asarray(pending["reward"]),
                                np.asarray(pending["discount"]),
                                np.asarray(pending["next_observation"]),
                                gamma=config.discount,
                            )
                            if online_step >= config.online_warmup_steps:
                                for _ in range(transition_count):
                                    metrics = online_update_once()
                            for key in pending:
                                pending[key].clear()
                        completed_rows.append(
                            {
                                "phase": "online_episode",
                                "offline_update": offline_update,
                                "online_step": online_step,
                                "episode": online_episode,
                                "episode_return": float(episode_returns[index]),
                                "episode_length": int(episode_lengths[index]),
                                **metrics,
                            }
                        )
                        if (
                            args.online_buffer_quality_steps > 0
                            and online_step <= args.online_buffer_quality_steps
                        ):
                            warmup_buffer_episodes.append(
                                {
                                    "online_step_end": int(online_step),
                                    "collector": "+".join(sorted(episode_collectors[index])),
                                    "tip_rmse_m": float(
                                        np.sqrt(np.mean(np.square(episode_tip_errors[index])))
                                    ),
                                    "tip_p95_m": float(np.quantile(episode_tip_errors[index], 0.95)),
                                    "joint_rmse_m": float(
                                        np.sqrt(np.mean(np.square(episode_joint_errors[index])))
                                    ),
                                    "joint_p95_m": float(np.quantile(episode_joint_errors[index], 0.95)),
                                    "return": float(episode_returns[index]),
                                }
                            )
                            if online_step == args.online_buffer_quality_steps:
                                tip_rmse = np.asarray(
                                    [row["tip_rmse_m"] for row in warmup_buffer_episodes],
                                    dtype=np.float64,
                                )
                                returns = np.asarray(
                                    [row["return"] for row in warmup_buffer_episodes],
                                    dtype=np.float64,
                                )
                                top_count = max(1, int(np.ceil(0.1 * len(tip_rmse))))
                                top = np.sort(tip_rmse)[:top_count]
                                _atomic_json(
                                    output / "warmup_buffer_quality.json",
                                    {
                                        "steps": int(args.online_buffer_quality_steps),
                                        "episodes": warmup_buffer_episodes,
                                        "tip_rmse_m": {
                                            "best": float(tip_rmse.min()),
                                            "top_10pct_mean": float(top.mean()),
                                            "median": float(np.median(tip_rmse)),
                                        },
                                        "return": {
                                            "best": float(returns.max()),
                                            "top_10pct_mean": float(
                                                returns[np.argsort(tip_rmse)[:top_count]].mean()
                                            ),
                                            "median": float(np.median(returns)),
                                        },
                                    },
                                )
                        episode_joint_errors[index].clear()
                        episode_tip_errors[index].clear()
                        episode_collectors[index].clear()
                        online_episode += 1
                observation = vector_step.observation
                for row in completed_rows:
                    _append_jsonl(metrics_path, row)
                # With the default 10 parallel environments, a 5k-transition
                # warmup contains five hundred steps per environment rather
                # than a completed 1k-step circle.  Persist the exact prefix
                # quality at the requested boundary instead of silently
                # emitting no A0/A4 comparison file.
                warmup_quality_path = output / "warmup_buffer_quality.json"
                if (
                    args.online_buffer_quality_steps > 0
                    and online_step == args.online_buffer_quality_steps
                    and not warmup_quality_path.exists()
                ):
                    prefix_rows = []
                    for index in range(config.num_envs):
                        if not episode_tip_errors[index]:
                            continue
                        prefix_rows.append(
                            {
                                "environment_index": index,
                                "collector": "+".join(sorted(episode_collectors[index])),
                                "steps": len(episode_tip_errors[index]),
                                "tip_rmse_m": float(
                                    np.sqrt(np.mean(np.square(episode_tip_errors[index])))
                                ),
                                "tip_p95_m": float(np.quantile(episode_tip_errors[index], 0.95)),
                                "joint_rmse_m": float(
                                    np.sqrt(np.mean(np.square(episode_joint_errors[index])))
                                ),
                                "joint_p95_m": float(np.quantile(episode_joint_errors[index], 0.95)),
                                "return": float(episode_returns[index]),
                            }
                        )
                    if prefix_rows:
                        tip_rmse = np.asarray(
                            [row["tip_rmse_m"] for row in prefix_rows], dtype=np.float64
                        )
                        returns = np.asarray(
                            [row["return"] for row in prefix_rows], dtype=np.float64
                        )
                        top_count = max(1, int(np.ceil(0.1 * len(tip_rmse))))
                        _atomic_json(
                            warmup_quality_path,
                            {
                                "steps": int(args.online_buffer_quality_steps),
                                "trajectory_scope": "per_environment_incomplete_episode_prefix",
                                "episodes": prefix_rows,
                                "tip_rmse_m": {
                                    "best": float(tip_rmse.min()),
                                    "top_10pct_mean": float(np.sort(tip_rmse)[:top_count].mean()),
                                    "median": float(np.median(tip_rmse)),
                                },
                                "return": {
                                    "best": float(returns.max()),
                                    "top_10pct_mean": float(
                                        returns[np.argsort(tip_rmse)[:top_count]].mean()
                                    ),
                                    "median": float(np.median(returns)),
                                },
                            },
                        )
                boundary_checkpoint_due = (
                    args.checkpoint_save_interval is None
                    or online_step % args.checkpoint_save_interval == 0
                )
                if vector_step.reset_boundary.all():
                    episode_returns.fill(0.0)
                    episode_lengths.fill(0)
                    if online_step > 0 and boundary_checkpoint_due:
                        save(latest_path, phase="online", include_replay=True)
                if online_step % args.online_eval_interval == 0:
                    _append_jsonl(
                        metrics_path,
                        {
                            "phase": "online_update",
                            "offline_update": offline_update,
                            "online_step": online_step,
                            **metrics,
                        },
                    )
                    online_evaluation = evaluate_and_save("online", online_step)
                    online_evaluation_history.append(online_evaluation)
                    if args.awac_selectivity_mode is not None:
                        assert online_start_evaluation is not None
                        start_tip_rmse = float(
                            online_start_evaluation["tip_rmse_m"]
                        )
                        tip_rmse = float(online_evaluation["tip_rmse_m"])
                        reasons: list[str] = []
                        if not np.isfinite(tip_rmse):
                            reasons.append("non_finite_tip_rmse")
                        if tip_rmse > 1.5 * start_tip_rmse:
                            reasons.append("tip_rmse_above_1p5x_source")
                        if float(
                            online_evaluation["action_saturation_fraction"]
                        ) > 0.0:
                            reasons.append("nonzero_physical_action_saturation")
                        if len(online_evaluation_history) >= 4:
                            recent = [
                                float(item["tip_rmse_m"])
                                for item in online_evaluation_history[-3:]
                            ]
                            if (
                                recent[0] < recent[1] < recent[2]
                                and recent[2] > 1.25 * start_tip_rmse
                            ):
                                reasons.append("three_eval_runaway_above_1p25x")
                        if reasons:
                            online_early_stop = {
                                "online_step": online_step,
                                "reasons": reasons,
                                "online_start_tip_rmse_m": start_tip_rmse,
                                "evaluation": online_evaluation,
                            }
                            break
                    if args.online_auto_stop:
                        assert online_start_evaluation is not None
                        start_rmse = float(
                            online_start_evaluation["joint_error_rmse_m"]
                        )
                        rmse = float(online_evaluation["joint_error_rmse_m"])
                        reasons: list[str] = []
                        if rmse > 2.0 * start_rmse:
                            reasons.append("rmse_above_2x_online_zero")
                        if (
                            len(online_evaluation_history) >= 3
                            and all(
                                float(item["joint_error_rmse_m"])
                                > 1.05 * start_rmse
                                for item in online_evaluation_history[-2:]
                            )
                        ):
                            reasons.append("two_consecutive_rmse_above_online_zero_5pct")
                        if float(online_evaluation["residual_p95_abs"]) > offline_residual_p95:
                            reasons.append("eval_residual_p95_above_offline_support_p95")
                        if float(online_evaluation["action_saturation_fraction"]) > 0.0:
                            reasons.append("physical_action_saturation")
                        if len(online_evaluation_history) >= 4:
                            d_max = float(
                                getattr(learner.actor, "action_cost_center_limit", 1.0)
                            )
                            drift = [
                                float(item["q_over_q_base_std"])
                                + float(item.get("action_cost_center_p95_abs", 0.0))
                                / d_max
                                for item in online_evaluation_history[-3:]
                            ]
                            previous_growth = drift[1] - drift[0]
                            current_growth = drift[2] - drift[1]
                            if (
                                previous_growth > 0.0
                                and current_growth > 1.5 * previous_growth
                            ):
                                reasons.append("cost_map_drift_acceleration")
                        if reasons:
                            online_early_stop = {
                                "online_step": online_step,
                                "reasons": reasons,
                                "online_start_rmse_m": start_rmse,
                                "evaluation": online_evaluation,
                            }
                            break
        finally:
            env.close()

    run_metadata.update(
        completed=True,
        completion_scope=(
            "online_early_stop" if online_early_stop is not None else "offline_to_online"
        ),
        offline_updates_completed=offline_update,
        online_steps_completed=online_step,
        online_episodes_completed=online_episode,
        best_return=best_return,
        best_evaluation=best_evaluation,
        wall_time_seconds=time.time() - started,
        online_early_stop=online_early_stop,
    )
    if actor_update_statistics:
        statistic_keys = tuple(actor_update_statistics[0])
        _atomic_json(
            output / "actor_update_statistics.json",
            {
                "updates": len(actor_update_statistics),
                "per_update": actor_update_statistics,
                "summary": {
                    key: {
                        "mean": float(np.mean([row[key] for row in actor_update_statistics])),
                        "p05": float(np.quantile([row[key] for row in actor_update_statistics], 0.05)),
                        "p50": float(np.quantile([row[key] for row in actor_update_statistics], 0.50)),
                        "p95": float(np.quantile([row[key] for row in actor_update_statistics], 0.95)),
                    }
                    for key in statistic_keys
                },
            },
        )
    _atomic_json(output / "run.json", run_metadata)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=FORMAL_METHODS, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--koopman", type=Path, required=True)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-checkpoint", type=Path)
    parser.add_argument(
        "--bootstrap-actor-only",
        action="store_true",
        help=(
            "Import only the actor from an offline checkpoint and retain a "
            "fresh critic, target critic, temperature, optimizers, replay, and RNG."
        ),
    )
    parser.add_argument(
        "--bootstrap-preserve-actor",
        action="store_true",
        help=(
            "With --bootstrap-actor-only, preserve all actor parameters "
            "including learned residual heads."
        ),
    )
    parser.add_argument(
        "--continue-online-checkpoint",
        type=Path,
        help=(
            "Explicitly fork a complete online checkpoint, including replay "
            "and RNG state, into a longer schedule without overwriting its run."
        ),
    )
    parser.add_argument(
        "--bootstrap-allow-schedule-change",
        action="store_true",
        help=(
            "Allow only count/evaluation/UTD/warmup/replay/worker schedule "
            "fields to differ from a bootstrap checkpoint. This also permits "
            "an explicitly selected zero-update checkpoint."
        ),
    )
    parser.add_argument(
        "--bootstrap-allow-dataset-mismatch",
        action="store_true",
        help=(
            "Explicitly retain the bootstrap checkpoint as the canonical source "
            "when the current dataset file has a different recorded SHA256. "
            "The mismatch is recorded in run metadata and must be reviewed."
        ),
    )
    parser.add_argument("--seed", type=int, default=TRAINING_SEED)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--offline-updates", type=int, default=OFFLINE_UPDATES)
    parser.add_argument("--online-steps", type=int, default=ONLINE_STEPS)
    parser.add_argument(
        "--replay-capacity",
        type=int,
        help="Keep the source replay capacity when extending an online schedule.",
    )
    parser.add_argument("--offline-eval-interval", type=int, default=OFFLINE_EVAL_INTERVAL)
    parser.add_argument("--log-interval-updates", type=int, default=1_000)
    parser.add_argument("--online-eval-interval", type=int, default=ONLINE_EVAL_INTERVAL)
    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--kmpc-horizon", type=int, default=KMPC_HORIZON)
    parser.add_argument("--kmpc-solver-iterations", type=int, default=20)
    parser.add_argument("--kmpc-delta-u-weight", type=float, default=0.0)
    parser.add_argument("--kmpc-delta-u-deadband", type=float, default=0.0)
    parser.add_argument("--kmpc-delta-u-limit", type=float, default=0.0)
    parser.add_argument("--kmpc-log-std-init", type=float, default=0.0)
    parser.add_argument("--kmpc-log-std-max", type=float, default=2.0)
    parser.add_argument("--actor-learning-rate", type=float)
    parser.add_argument("--critic-learning-rate", type=float)
    parser.add_argument("--initial-temperature", type=float, default=1.0)
    parser.add_argument("--temperature-learning-rate", type=float)
    parser.add_argument("--target-entropy", type=float)
    parser.add_argument("--online-utd", type=int)
    parser.add_argument("--offline-replay-ratio", type=float, default=0.5)
    parser.add_argument(
        "--actor-offline-replay-ratio",
        type=float,
        default=None,
        help=(
            "Offline fraction for a separately sampled actor batch. "
            "Defaults to --offline-replay-ratio, preserving legacy runs."
        ),
    )
    parser.add_argument(
        "--awac-selectivity-mode",
        choices=("all", "positive", "positive_top50", "positive_klref"),
        default=None,
        help=(
            "Continuation-only AWAC imitation-sample mask. Enabling this also "
            "freezes the continuation source actor for probe diagnostics."
        ),
    )
    parser.add_argument(
        "--awac-reference-kl-weight",
        type=float,
        default=0.0,
        help="KL(pi||source) coefficient for positive_klref only.",
    )
    parser.add_argument(
        "--online-buffer-quality-steps",
        type=int,
        default=0,
        help="Write per-episode warmup-buffer quality through this online step.",
    )
    parser.add_argument("--actor-update-interval", type=int, default=1)
    parser.add_argument("--q-cost-anchor-weight", type=float, default=0.0)
    parser.add_argument("--p-cost-anchor-weight", type=float, default=0.0)
    parser.add_argument("--anchor-gradient-diagnostics", action="store_true")
    parser.add_argument("--bootstrap-rmse-tolerance", type=float)
    parser.add_argument("--bootstrap-return-target", type=float)
    parser.add_argument("--bootstrap-return-tolerance", type=float, default=0.1)
    parser.add_argument("--online-auto-stop", action="store_true")
    parser.add_argument("--online-warmup-steps", type=int)
    parser.add_argument(
        "--expert-warmup-checkpoint",
        type=Path,
        help="Frozen PPO expert used only to collect initial online replay.",
    )
    parser.add_argument(
        "--expert-warmup-vecnormalize",
        type=Path,
        help="VecNormalize statistics paired with --expert-warmup-checkpoint.",
    )
    parser.add_argument(
        "--expert-warmup-reference",
        type=Path,
        help="Reference table used by the PPO expert (may differ from current ff).",
    )
    parser.add_argument(
        "--expert-warmup-steps",
        type=int,
        default=0,
        help="Number of initial online transitions collected by the frozen expert.",
    )
    parser.add_argument(
        "--expert-residual-limit",
        type=float,
        default=0.02,
        help="Physical residual scale used by the PPO expert during training.",
    )
    parser.add_argument(
        "--expert-perturbation-std",
        type=float,
        default=0.001,
        help="Std. dev. of the small physical residual perturbation.",
    )
    parser.add_argument(
        "--expert-perturbation-limit",
        type=float,
        default=0.003,
        help="Absolute cap for each warmup perturbation component.",
    )
    parser.add_argument(
        "--online-critic-only-steps",
        type=int,
        default=0,
        help=(
            "Keep collecting data and updating the critic, but freeze the "
            "actor and temperature for this many initial online steps."
        ),
    )
    parser.add_argument(
        "--disable-actor-entropy",
        action="store_true",
        help=(
            "Use the deterministic -Q actor objective and freeze temperature; "
            "critic entropy backup is controlled separately."
        ),
    )
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--env-workers", type=int)
    parser.add_argument("--cql-weight", type=float)
    parser.add_argument(
        "--reward-mode", choices=("sparse", "hybrid", "dense_xref", "dense_joint"), default="sparse"
    )
    parser.add_argument("--sparse-reward-weight", type=float, default=1.0)
    parser.add_argument("--dense-reward-weight", type=float, default=0.0)
    parser.add_argument("--dense-reward-scale-m", type=float, default=DENSE_REWARD_SCALE_M)
    parser.add_argument(
        "--online-cql-mode", choices=("off", "all_valid_mc"), default=None
    )
    parser.add_argument("--stop-after-offline", action="store_true")
    parser.add_argument("--stop-after-continuation-eval", action="store_true")
    parser.add_argument("--stop-after-online-zero-eval", action="store_true")
    parser.add_argument("--skip-online-zero-eval", action="store_true")
    parser.add_argument("--online-zero-reference-evaluation", type=Path)
    parser.add_argument("--checkpoint-save-interval", type=int)
    parser.add_argument("--profile-continuation-update-memory", action="store_true")
    args = parser.parse_args()
    if (
        args.bootstrap_checkpoint is not None
        and args.continue_online_checkpoint is not None
    ):
        raise ValueError(
            "bootstrap_checkpoint and continue_online_checkpoint are mutually exclusive"
        )
    if args.continue_online_checkpoint is not None:
        args.continue_online_checkpoint = (
            args.continue_online_checkpoint.expanduser().resolve()
        )
        if not args.continue_online_checkpoint.is_file():
            raise FileNotFoundError(args.continue_online_checkpoint)
        if args.stop_after_offline:
            raise ValueError(
                "continue_online_checkpoint cannot be combined with stop_after_offline"
            )
    if args.bootstrap_checkpoint is not None:
        args.bootstrap_checkpoint = args.bootstrap_checkpoint.expanduser().resolve()
        if not args.bootstrap_checkpoint.is_file():
            raise FileNotFoundError(args.bootstrap_checkpoint)
        if args.stop_after_offline:
            raise ValueError("bootstrap_checkpoint cannot be combined with stop_after_offline")
    if args.bootstrap_actor_only and args.bootstrap_checkpoint is None:
        raise ValueError("bootstrap-actor-only requires bootstrap-checkpoint")
    if args.awac_selectivity_mode is None:
        if args.awac_reference_kl_weight != 0.0:
            raise ValueError(
                "awac-reference-kl-weight requires awac-selectivity-mode"
            )
    elif args.awac_selectivity_mode == "positive_klref":
        if args.awac_reference_kl_weight <= 0.0:
            raise ValueError("positive_klref requires a positive KL weight")
    elif args.awac_reference_kl_weight != 0.0:
        raise ValueError(
            "awac-reference-kl-weight is only valid for positive_klref"
        )
    if args.online_zero_reference_evaluation is not None:
        args.online_zero_reference_evaluation = (
            args.online_zero_reference_evaluation.expanduser().resolve()
        )
        if not args.online_zero_reference_evaluation.is_file():
            raise FileNotFoundError(args.online_zero_reference_evaluation)
    if args.skip_online_zero_eval:
        if args.bootstrap_checkpoint is None:
            raise ValueError("skip-online-zero-eval requires bootstrap-checkpoint")
        if args.online_zero_reference_evaluation is None:
            raise ValueError(
                "skip-online-zero-eval requires online-zero-reference-evaluation"
            )
        if args.stop_after_online_zero_eval:
            raise ValueError(
                "skip-online-zero-eval cannot be combined with "
                "stop-after-online-zero-eval"
            )
    if args.offline_updates < 0:
        raise ValueError("offline_updates must be non-negative")
    for name in (
        "online_steps",
        "offline_eval_interval",
        "log_interval_updates",
        "online_eval_interval",
        "eval_episodes",
        "kmpc_horizon",
    ):
        minimum = 0 if name == "online_steps" else 1
        if getattr(args, name) < minimum:
            raise ValueError(f"{name} must be positive")
    if args.checkpoint_save_interval is not None and args.checkpoint_save_interval < 1:
        raise ValueError("checkpoint-save-interval must be positive")
    if args.online_utd is not None and args.online_utd < 1:
        raise ValueError("online_utd must be positive")
    if args.replay_capacity is not None and args.replay_capacity < 1:
        raise ValueError("replay_capacity must be positive")
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.actor_update_interval < 1:
        raise ValueError("actor-update-interval must be positive")
    if not 0 <= args.offline_replay_ratio <= 1:
        raise ValueError("offline-replay-ratio must lie in [0,1]")
    if args.actor_offline_replay_ratio is not None and not 0 <= args.actor_offline_replay_ratio <= 1:
        raise ValueError("actor-offline-replay-ratio must lie in [0,1]")
    if args.online_buffer_quality_steps < 0:
        raise ValueError("online-buffer-quality-steps must be non-negative")
    for name in ("q_cost_anchor_weight", "p_cost_anchor_weight"):
        value = getattr(args, name)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if args.bootstrap_rmse_tolerance is not None and (
        not np.isfinite(args.bootstrap_rmse_tolerance)
        or args.bootstrap_rmse_tolerance < 0
    ):
        raise ValueError("bootstrap-rmse-tolerance must be finite and nonnegative")
    if args.bootstrap_return_target is not None and not np.isfinite(
        args.bootstrap_return_target
    ):
        raise ValueError("bootstrap-return-target must be finite")
    if (
        not np.isfinite(args.bootstrap_return_tolerance)
        or args.bootstrap_return_tolerance < 0
    ):
        raise ValueError("bootstrap-return-tolerance must be finite and nonnegative")
    if args.bootstrap_return_target is not None and args.bootstrap_checkpoint is None:
        raise ValueError("bootstrap-return-target requires bootstrap-checkpoint")
    if args.online_warmup_steps is not None and args.online_warmup_steps < 0:
        raise ValueError("online_warmup_steps must be non-negative")
    if args.online_critic_only_steps < 0:
        raise ValueError("online-critic-only-steps must be non-negative")
    if args.expert_warmup_steps < 0:
        raise ValueError("expert-warmup-steps must be non-negative")
    expert_fields = (
        args.expert_warmup_checkpoint,
        args.expert_warmup_vecnormalize,
    )
    if any(value is not None for value in expert_fields) and not all(
        value is not None for value in expert_fields
    ):
        raise ValueError(
            "expert-warmup-checkpoint and expert-warmup-vecnormalize must be supplied together"
        )
    if args.expert_warmup_steps and args.expert_warmup_checkpoint is None:
        raise ValueError(
            "expert-warmup-steps requires --expert-warmup-checkpoint and --expert-warmup-vecnormalize"
        )
    for name in ("num_envs", "env_workers"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"{name} must be positive")
    for name in (
        "actor_learning_rate",
        "critic_learning_rate",
        "temperature_learning_rate",
    ):
        value = getattr(args, name)
        if value is not None and (not np.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and positive")
    if args.cql_weight is not None and (
        not np.isfinite(args.cql_weight) or args.cql_weight < 0
    ):
        raise ValueError("cql_weight must be finite and non-negative")
    if not np.isfinite(args.initial_temperature) or args.initial_temperature <= 0:
        raise ValueError("initial_temperature must be finite and positive")
    if args.target_entropy is not None and not np.isfinite(args.target_entropy):
        raise ValueError("target_entropy must be finite")
    for name in ("sparse_reward_weight", "dense_reward_weight", "dense_reward_scale_m"):
        value = getattr(args, name)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if args.dense_reward_scale_m <= 0:
        raise ValueError("dense_reward_scale_m must be positive")
    if args.reward_mode == "hybrid" and (
        args.sparse_reward_weight == 0 and args.dense_reward_weight == 0
    ):
        raise ValueError("hybrid reward needs a non-zero component")
    if args.reward_mode in ("dense_xref", "dense_joint") and (
        args.sparse_reward_weight != 0 or args.dense_reward_weight <= 0
    ):
        raise ValueError(
            "dense_xref/dense_joint reward requires sparse-reward-weight=0 and "
            "dense-reward-weight>0"
        )
    if args.reward_mode == "sparse":
        args.dense_reward_weight = 0.0
    if not np.isfinite(args.kmpc_delta_u_weight) or args.kmpc_delta_u_weight < 0:
        raise ValueError("kmpc_delta_u_weight must be finite and non-negative")
    if (
        not np.isfinite(args.kmpc_delta_u_deadband)
        or not 0 <= args.kmpc_delta_u_deadband <= 0.3
        or (args.kmpc_delta_u_deadband > 0 and args.kmpc_delta_u_weight == 0)
    ):
        raise ValueError("kmpc_delta_u_deadband is invalid")
    if (
        not np.isfinite(args.kmpc_delta_u_limit)
        or not 0 <= args.kmpc_delta_u_limit <= 0.3
    ):
        raise ValueError("kmpc_delta_u_limit must lie in [0, 0.3]")
    if (
        not np.isfinite(args.kmpc_log_std_init)
        or not np.isfinite(args.kmpc_log_std_max)
        or args.kmpc_log_std_init > args.kmpc_log_std_max
    ):
        raise ValueError("KMPC log-std initialization/bounds are invalid")
    return args


if __name__ == "__main__":
    run(parse_args())

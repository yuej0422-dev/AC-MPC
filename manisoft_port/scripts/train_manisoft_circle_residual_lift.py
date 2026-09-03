#!/usr/bin/env python
"""Train residual Cal-RLPD-KMPC around a frozen coarse phase feedforward."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import train_manisoft_circle_implicit_kmpc as implicit_entry

from antmaze_ac.data.circle_residual_dataset import (
    FEEDFORWARD_ENV,
    RESIDUAL_LIMIT_ENV,
    ManiSoftCircleResidualDataset,
)
from antmaze_ac.envs.circle_phase_feedforward import (
    FrozenCirclePhaseFeedforward,
)
from antmaze_ac.envs.manisoft_circle_o2o_env import ABSOLUTE_ACTION_LIMIT
from antmaze_ac.envs.manisoft_circle_residual_env import (
    make_manisoft_circle_residual_adapter,
)
from antmaze_ac.rl.residual_kmpc_actor import (
    FeedforwardResidualKMPCTanhGaussianActor,
)

import experiments.dmc.o2o.config as config_module
import experiments.dmc.o2o.learner as learner_module


training = implicit_entry.training
_BASE_ACTION_LIMIT = getattr(config_module, "o2o_action_limit", lambda _task: 1.0)
_BASE_BUILD_ACTOR = learner_module.build_actor
_RESIDUAL_LIMIT = 0.1
_FEEDFORWARD: FrozenCirclePhaseFeedforward | None = None
_RESIDUAL_PROTOCOL: dict[str, Any] | None = None


def _residual_action_limit(task: str) -> float:
    if task == training.TASK_NAME:
        return float(_RESIDUAL_LIMIT)
    return float(_BASE_ACTION_LIMIT(task))


def _build_residual_actor(
    method: str,
    koopman: Any,
    **kwargs: Any,
):
    if method != "Cal-RLPD-KMPC" or kwargs.get("actor_kind") != "ac_kmpc":
        return _BASE_BUILD_ACTOR(method, koopman, **kwargs)
    if koopman is None or _FEEDFORWARD is None:
        raise RuntimeError("Residual KMPC actor dependencies were not installed")
    if kwargs.get("kmpc_delta_u_weight", 0.0) != 0.0:
        raise ValueError("Absolute delta-u penalty is disabled for residual KMPC")
    if kwargs.get("kmpc_delta_u_deadband", 0.0) != 0.0:
        raise ValueError("Absolute delta-u deadband is disabled for residual KMPC")
    if kwargs.get("kmpc_delta_u_limit", 0.0) != 0.0:
        raise ValueError("Absolute delta-u rate bound is disabled for residual KMPC")
    return FeedforwardResidualKMPCTanhGaussianActor(
        koopman,
        _FEEDFORWARD,
        horizon=int(kwargs["kmpc_horizon"]),
        solver_iterations=int(kwargs["kmpc_solver_iterations"]),
        hidden_dim=int(kwargs["controller_hidden_dim"]),
        hidden_layers=int(kwargs["controller_hidden_layers"]),
        residual_limit=float(_RESIDUAL_LIMIT),
        physical_action_limit=float(ABSOLUTE_ACTION_LIMIT),
        log_std_init=float(kwargs["kmpc_log_std_init"]),
        log_std_max=float(kwargs["kmpc_log_std_max"]),
    )


@np.errstate(all="raise")
def _physical_metrics(
    physical: np.ndarray,
    residual: np.ndarray,
    feedforward: np.ndarray,
) -> dict[str, float]:
    physical_previous = np.concatenate(
        (np.zeros_like(physical[:1]), physical[:-1]), axis=0
    )
    residual_previous = np.concatenate(
        (np.zeros_like(residual[:1]), residual[:-1]), axis=0
    )
    absolute_physical = np.abs(physical)
    absolute_residual = np.abs(residual)
    physical_delta = np.abs(physical - physical_previous)
    residual_delta = np.abs(residual - residual_previous)
    return {
        "max_abs_action": float(absolute_physical.max()),
        "mean_abs_action": float(absolute_physical.mean()),
        "action_p95_abs": float(np.quantile(absolute_physical, 0.95)),
        "action_saturation_fraction": float(
            np.mean(absolute_physical >= 0.99 * ABSOLUTE_ACTION_LIMIT)
        ),
        "mean_abs_delta_action": float(physical_delta.mean()),
        "delta_action_p95_abs": float(np.quantile(physical_delta, 0.95)),
        "max_abs_delta_action": float(physical_delta.max()),
        "residual_mean_abs": float(absolute_residual.mean()),
        "residual_p95_abs": float(np.quantile(absolute_residual, 0.95)),
        "residual_max_abs": float(absolute_residual.max()),
        "residual_limit_fraction": float(
            np.mean(absolute_residual >= 0.99 * _RESIDUAL_LIMIT)
        ),
        "residual_delta_p95_abs": float(np.quantile(residual_delta, 0.95)),
        "feedforward_p95_abs": float(np.quantile(np.abs(feedforward), 0.95)),
    }


def evaluate_residual(
    learner: Any,
    *,
    seed_base: int,
    episodes: int,
) -> dict[str, Any]:
    env = training._environment(episodes, seed_base, workers=episodes)
    try:
        observation = env.reset()
        returns = np.zeros(episodes, dtype=np.float64)
        joint_errors: list[np.ndarray] = []
        physical_actions: list[np.ndarray] = []
        residual_actions: list[np.ndarray] = []
        feedforward_actions: list[np.ndarray] = []
        for step in range(1000):
            residual = learner.act(observation, deterministic=True)
            vector_step = env.step(residual)
            returns += vector_step.reward
            residual_actions.append(
                np.asarray(vector_step.applied_action, dtype=np.float64)
            )
            physical_actions.append(
                np.asarray(
                    [item["physical_applied_action"] for item in vector_step.info],
                    dtype=np.float64,
                )
            )
            feedforward_actions.append(
                np.asarray(
                    [item["feedforward_action"] for item in vector_step.info],
                    dtype=np.float64,
                )
            )
            joint_errors.append(
                np.asarray(
                    [item["joint_target_error"] for item in vector_step.info],
                    dtype=np.float64,
                )
            )
            expected_boundary = step == 999
            if bool(vector_step.reset_boundary.any()) != expected_boundary:
                raise RuntimeError("Evaluation episode boundary drifted")
            observation = vector_step.observation
        errors = np.stack(joint_errors)
        physical = np.stack(physical_actions)
        residual = np.stack(residual_actions)
        feedforward = np.stack(feedforward_actions)
        return {
            "return_mean": float(returns.mean()),
            "return_std": float(returns.std()),
            "return_min": float(returns.min()),
            "return_max": float(returns.max()),
            "reward_rate": float(returns.mean() / 1000.0),
            "joint_error_rmse_m": float(np.sqrt(np.mean(errors**2))),
            "joint_error_p95_m": float(np.quantile(errors, 0.95)),
            "episode_returns": returns.tolist(),
            **_physical_metrics(physical, residual, feedforward),
        }
    finally:
        env.close()


def _parse_args() -> tuple[argparse.Namespace, Path, float, float]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--feedforward", type=Path, required=True)
    parser.add_argument(
        "--feedforward-standalone-rmse-m",
        type=float,
        default=0.011362345236579733,
    )
    parser.add_argument("--residual-limit", type=float, default=0.1)
    known, remaining = parser.parse_known_args()
    feedforward = known.feedforward.expanduser().resolve()
    if not feedforward.is_file():
        raise FileNotFoundError(feedforward)
    limit = float(known.residual_limit)
    if not math.isfinite(limit) or not 0 < limit <= 0.3:
        raise ValueError("residual-limit must lie in (0, 0.3]")
    feedforward_rmse = float(known.feedforward_standalone_rmse_m)
    if not math.isfinite(feedforward_rmse) or feedforward_rmse <= 0:
        raise ValueError("feedforward-standalone-rmse-m must be finite and positive")
    sys.argv = [sys.argv[0], *remaining]
    args = training.parse_args()
    if args.method != "Cal-RLPD-KMPC":
        raise ValueError("Expected --method Cal-RLPD-KMPC")
    if args.bootstrap_checkpoint is not None:
        raise ValueError("Residual RL forbids BC/bootstrap initialization")
    if args.reward_mode != "hybrid":
        raise ValueError("Residual protocol requires hybrid reward")
    if args.sparse_reward_weight != 1.0 or args.dense_reward_weight != 0.1:
        raise ValueError("Expected sparse weight 1 and dense weight 0.1")
    if (
        args.kmpc_delta_u_weight != 0.0
        or args.kmpc_delta_u_deadband != 0.0
        or args.kmpc_delta_u_limit != 0.0
    ):
        raise ValueError("Legacy absolute delta-u controls must remain zero")
    return args, feedforward, limit, feedforward_rmse


def main() -> None:
    global _RESIDUAL_LIMIT, _FEEDFORWARD, _RESIDUAL_PROTOCOL
    args, feedforward_path, _RESIDUAL_LIMIT, feedforward_rmse = _parse_args()
    _FEEDFORWARD = FrozenCirclePhaseFeedforward(feedforward_path)
    os.environ[FEEDFORWARD_ENV] = str(feedforward_path)
    os.environ[RESIDUAL_LIMIT_ENV] = str(_RESIDUAL_LIMIT)

    # Actor, critic, entropy and CQL all operate in residual-action coordinates.
    config_module.o2o_action_limit = _residual_action_limit
    learner_module.o2o_action_limit = _residual_action_limit
    learner_module.build_actor = _build_residual_actor
    training.ManiSoftCircleOfflineDataset = ManiSoftCircleResidualDataset
    training.make_manisoft_circle_o2o_adapter = make_manisoft_circle_residual_adapter
    training.evaluate = evaluate_residual

    _RESIDUAL_PROTOCOL = {
        "kind": "manisoft_circle_feedforward_residual_kmpc_v1",
        "method": "Cal-RLPD-KMPC",
        "policy_observation": "physical_koopman_lift + phase_sin_cos_2",
        "policy_lifted_observation_dim": 79,
        "target_in_policy_observation": False,
        "feedforward": _FEEDFORWARD.identity(),
        "feedforward_standalone_rmse_m": feedforward_rmse,
        "residual_limit": _RESIDUAL_LIMIT,
        "qp_variable": "V where U=U_ff(phase)+V",
        "qp_cost": "learned cost on residual-induced state response and V",
        "zero_cost_head_semantics": "V=0 exactly",
        "actor_initialization": "standard random trunk + zero cost head",
        "critic_initialization": "standard random",
        "bc_initialization": False,
        "legacy_absolute_delta_u_penalty": False,
        "reward": {
            "mode": "hybrid",
            "sparse_weight": 1.0,
            "dense_weight": 0.1,
            "dense_scale_m": args.dense_reward_scale_m,
        },
        "episode_steps": 1000,
    }
    original_checkpoint = training._checkpoint

    def checkpoint_with_residual_protocol(**kwargs: Any) -> dict[str, Any]:
        result = original_checkpoint(**kwargs)
        result["residual_control"] = _RESIDUAL_PROTOCOL
        return result

    training._checkpoint = checkpoint_with_residual_protocol
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "residual_protocol.json").write_text(
        json.dumps(_RESIDUAL_PROTOCOL, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    training.run(args)
    run_path = output / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["residual_control"] = _RESIDUAL_PROTOCOL
    temporary = run_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, run_path)


if __name__ == "__main__":
    main()

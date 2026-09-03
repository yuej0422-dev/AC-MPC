#!/usr/bin/env python
"""Direct-online ManiSoft Q,p-KMPC control-structure causal screen."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import train_manisoft_circle_implicit_kmpc as implicit_entry
import train_manisoft_circle_residual_lift as residual_entry

from antmaze_ac.data.circle_time_residual_dataset import (
    ManiSoftCircleTimeResidualDataset,
)
from antmaze_ac.envs.circle_phase_feedforward import FrozenCirclePhaseFeedforward
from antmaze_ac.envs.manisoft_circle_o2o_env import ABSOLUTE_ACTION_LIMIT
from antmaze_ac.envs.manisoft_circle_time_residual_env import (
    FEEDFORWARD_ENV,
    RESIDUAL_LIMIT_ENV,
    make_manisoft_circle_time_residual_adapter,
)
from antmaze_ac.koopman.o2o_time_adapter import FrozenManiSoftTimeKoopman
from antmaze_ac.rl.time_structured_qp_kmpc_actor import (
    STRUCTURES,
    TimeStructuredQpKMPCTanhGaussianActor,
)

import experiments.dmc.o2o.config as config_module
import experiments.dmc.o2o.learner as learner_module
import antmaze_ac.envs.manisoft_circle_o2o_env as circle_env_module


training = implicit_entry.training
_BASE_ACTION_LIMIT = getattr(config_module, "o2o_action_limit", lambda _task: 1.0)
_BASE_BUILD_ACTOR = learner_module.build_actor
_FEEDFORWARD: FrozenCirclePhaseFeedforward | None = None
_XREF: np.ndarray | None = None
_STRUCTURE = "k0"
_RESIDUAL_LIMIT = 0.3
_STATE_COST_SCALE = 1.0
_RESIDUAL_COST_SCALE = 10_000.0
_TERMINAL_MULTIPLIER = 1.0
_STATE_COST_GATE_INIT = 1e-4
_NOMINAL_OBSERVATIONS: np.ndarray | None = None
_NOMINAL_PHYSICAL_STATES: np.ndarray | None = None
_NOMINAL_NORMALIZED_STATES: np.ndarray | None = None
_WARM_INIT_RIDGE = 1e-4
_WARM_INIT_DIAGNOSTICS: dict[str, float] | None = None
_BASE_O2O_LEARNER = training.O2OLearner


def _residual_action_limit(task: str) -> float:
    return _RESIDUAL_LIMIT if task == training.TASK_NAME else _BASE_ACTION_LIMIT(task)


def _build_actor(method: str, koopman: Any, **kwargs: Any):
    if method != "Cal-RLPD-KMPC" or kwargs.get("actor_kind") != "ac_kmpc":
        return _BASE_BUILD_ACTOR(method, koopman, **kwargs)
    if koopman is None or _FEEDFORWARD is None or _XREF is None:
        raise RuntimeError("Control-screen actor dependencies are missing")
    return TimeStructuredQpKMPCTanhGaussianActor(
        koopman,
        _FEEDFORWARD,
        _XREF,
        structure=_STRUCTURE,
        horizon=int(kwargs["kmpc_horizon"]),
        solver_iterations=int(kwargs["kmpc_solver_iterations"]),
        hidden_dim=int(kwargs["controller_hidden_dim"]),
        hidden_layers=int(kwargs["controller_hidden_layers"]),
        residual_limit=_RESIDUAL_LIMIT,
        physical_action_limit=circle_env_module.ABSOLUTE_ACTION_LIMIT,
        state_cost_scale=_STATE_COST_SCALE,
        residual_cost_scale=_RESIDUAL_COST_SCALE,
        action_cost_center_limit=0.005,
        state_cost_gate_init=_STATE_COST_GATE_INIT,
        terminal_multiplier=_TERMINAL_MULTIPLIER,
        log_std_init=float(kwargs["kmpc_log_std_init"]),
        log_std_max=float(kwargs["kmpc_log_std_max"]),
    )


class _NominalWarmInitializedLearner(_BASE_O2O_LEARNER):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        global _WARM_INIT_DIAGNOSTICS, _NOMINAL_NORMALIZED_STATES
        super().__init__(*args, **kwargs)
        if _STRUCTURE not in {"k10v2", "k10p"}:
            return
        if _NOMINAL_OBSERVATIONS is None or _NOMINAL_PHYSICAL_STATES is None:
            raise RuntimeError("K10-v2/P requires a nominal FF observation trajectory")
        observation = torch.as_tensor(
            _NOMINAL_OBSERVATIONS[:-1], dtype=torch.float32, device=self.device
        )
        target_physical = torch.as_tensor(
            _NOMINAL_PHYSICAL_STATES[:-1], dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            lifted = self._encode(observation)
            assert self.koopman is not None
            target_normalized = (
                target_physical - self.koopman.physical_center
            ) / self.koopman.physical_scale
            _NOMINAL_NORMALIZED_STATES = target_normalized.detach().cpu().numpy()
        _WARM_INIT_DIAGNOSTICS = self.actor.warm_initialize_state_center(
            lifted, target_normalized, ridge=_WARM_INIT_RIDGE
        )


def _stats(prefix: str, value: np.ndarray) -> dict[str, float]:
    flat = np.asarray(value, dtype=np.float64).reshape(-1)
    return {
        f"{prefix}_mean": float(flat.mean()),
        f"{prefix}_std": float(flat.std()),
        f"{prefix}_p95_abs": float(np.quantile(np.abs(flat), 0.95)),
        f"{prefix}_min": float(flat.min()),
        f"{prefix}_max": float(flat.max()),
    }


def evaluate_control_structure(
    learner: Any, *, seed_base: int, episodes: int
) -> dict[str, Any]:
    # Plain AWAC uses the historical direct MLP action policy rather than the
    # KMPC cost-map actor.  Keep the same control/error metrics while leaving
    # KMPC-only diagnostics at neutral values so the shared training loop can
    # evaluate both arms.
    if not hasattr(learner.actor, "cost_terms"):
        return _evaluate_plain_actor(learner, seed_base=seed_base, episodes=episodes)
    base_integrity = {}
    assert_frozen_base = getattr(
        learner.actor, "assert_base_actor_unchanged", None
    )
    if callable(assert_frozen_base):
        base_integrity = assert_frozen_base()
    env = training._environment(episodes, seed_base, workers=episodes)
    try:
        observation = env.reset()
        trajectory_output_raw = os.environ.get("ACMPC_EVAL_TRAJECTORY_OUTPUT")
        trajectory_observations: list[np.ndarray] = []
        trajectory_rewards: list[np.ndarray] = []
        trajectory_requested_residuals: list[np.ndarray] = []
        trajectory_targets: list[np.ndarray] = []
        trajectory_node_errors: list[np.ndarray] = []
        if trajectory_output_raw:
            trajectory_observations.append(np.asarray(observation, dtype=np.float32).copy())
        returns = np.zeros(episodes, dtype=np.float64)
        sparse_rewards: list[np.ndarray] = []
        dense_rewards: list[np.ndarray] = []
        errors: list[np.ndarray] = []
        tip_errors: list[np.ndarray] = []
        physical_actions: list[np.ndarray] = []
        residual_actions: list[np.ndarray] = []
        feedforward_actions: list[np.ndarray] = []
        q_states: list[np.ndarray] = []
        q_actions: list[np.ndarray] = []
        d_actions: list[np.ndarray] = []
        d_states: list[np.ndarray] = []
        state_gates: list[np.ndarray] = []
        implicit_xyz_rows: dict[str, list[np.ndarray]] = {}
        headroom_rows: dict[str, list[np.ndarray]] = {}
        solver_rows: dict[str, list[np.ndarray]] = {
            key: []
            for key in (
                "objective",
                "terminal_contribution",
                "active_constraint_fraction",
                "projected_gradient_relative",
                "solver_converged",
            )
        }
        sensitivities: list[np.ndarray] = []
        sensitivity_phases: list[int] = []
        for step in range(1000):
            with torch.no_grad():
                tensor = torch.as_tensor(observation, dtype=torch.float32, device=learner.device)
                lifted = learner._encode(tensor)
                q_state, q_action, p_state, p_action = learner.actor.cost_terms(lifted)
                diagnostic = learner.actor.plan_diagnostics(lifted)
                location, _, _ = learner.actor.distribution(lifted)
                feedforward = learner.actor.reference_plans(lifted)[0][..., 0, :]
                lower, upper = learner.actor.residual_bounds(feedforward)
                residual = 0.5 * (lower + upper) + 0.5 * (upper - lower) * torch.tanh(location)
                q_states.append(q_state.cpu().numpy())
                q_actions.append(q_action.cpu().numpy())
                d_actions.append((-p_action / q_action).cpu().numpy())
                implicit_diagnostics = getattr(
                    learner.actor, "implicit_xyz_diagnostics", None
                )
                if callable(implicit_diagnostics):
                    implicit_values = implicit_diagnostics(lifted)
                    d_states.append(
                        implicit_values["d_state_full"].cpu().numpy()
                    )
                    for key, value in implicit_values.items():
                        if key != "d_state_full":
                            implicit_xyz_rows.setdefault(key, []).append(
                                value.cpu().numpy()
                            )
                else:
                    d_states.append((-p_state / q_state).cpu().numpy())
                state_gates.append(learner.actor.state_cost_gate(lifted).cpu().numpy())
                headroom_diagnostics = getattr(
                    learner.actor, "action_headroom_diagnostics", None
                )
                if callable(headroom_diagnostics):
                    for key, value in headroom_diagnostics(lifted).items():
                        headroom_rows.setdefault(key, []).append(value.cpu().numpy())
                for key, value in diagnostic.items():
                    solver_rows[key].append(value.cpu().numpy())
            if step % 50 == 0:
                with torch.enable_grad():
                    sensitivity = learner.actor.directional_sensitivity(lifted[:1].detach())
                sensitivities.append(sensitivity.detach().cpu().numpy())
                sensitivity_phases.append(step)
            vector_step = env.step(residual.cpu().numpy())
            if trajectory_output_raw:
                trajectory_observations.append(
                    np.asarray(vector_step.observation, dtype=np.float32).copy()
                )
                trajectory_rewards.append(
                    np.asarray(vector_step.reward, dtype=np.float32).copy()
                )
                trajectory_requested_residuals.append(
                    residual.detach().cpu().numpy().astype(np.float32, copy=True)
                )
                trajectory_targets.append(
                    np.asarray(
                        [item["target_positions"] for item in vector_step.info],
                        dtype=np.float32,
                    )
                )
                trajectory_node_errors.append(
                    np.asarray(
                        [item["node_target_error"] for item in vector_step.info],
                        dtype=np.float32,
                    )
                )
            returns += vector_step.reward
            sparse_rewards.append(
                np.asarray(
                    [item.get("sparse_reward", 0.0) for item in vector_step.info],
                    dtype=np.float64,
                )
            )
            dense_rewards.append(
                np.asarray(
                    [item.get("dense_reward", 0.0) for item in vector_step.info],
                    dtype=np.float64,
                )
            )
            errors.append(
                np.asarray(
                    [item["joint_target_error"] for item in vector_step.info],
                    dtype=np.float64,
                )
            )
            tip_errors.append(
                np.asarray(
                    [item["node_target_error"][2] for item in vector_step.info],
                    dtype=np.float64,
                )
            )
            residual_actions.append(np.asarray(vector_step.applied_action, dtype=np.float64))
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
            observation = vector_step.observation
        error_array = np.stack(errors)
        tip_error_array = np.stack(tip_errors)
        physical = np.stack(physical_actions)
        residual_array = np.stack(residual_actions)
        feedforward_array = np.stack(feedforward_actions)
        q_state_array = np.stack(q_states)
        q_action_array = np.stack(q_actions)
        d_action_array = np.stack(d_actions)
        d_state_array = np.stack(d_states)
        state_gate_array = np.stack(state_gates)
        sparse_reward_array = np.stack(sparse_rewards)
        dense_reward_array = np.stack(dense_rewards)
        q_state_ratio = q_state_array / float(_STATE_COST_SCALE)
        q_action_ratio = q_action_array / float(_RESIDUAL_COST_SCALE)
        result = {
            **training.evaluation_seed_metadata(
                seed_base=seed_base, episodes=episodes
            ),
            "return_mean": float(returns.mean()),
            "return_std": float(returns.std()),
            "return_min": float(returns.min()),
            "return_max": float(returns.max()),
            "reward_rate": float(returns.mean() / 1000.0),
            "joint_error_rmse_m": float(np.sqrt(np.mean(error_array**2))),
            "joint_error_p95_m": float(np.quantile(error_array, 0.95)),
            "success_rate_2p5mm": float(np.mean(error_array <= 0.0025)),
            "tip_rmse_m": float(np.sqrt(np.mean(tip_error_array**2))),
            "tip_p95_m": float(np.quantile(tip_error_array, 0.95)),
            "tip_success_rate_2p5mm": float(np.mean(tip_error_array <= 0.0025)),
            "joint_error_phase_mean_m": error_array.mean(axis=1).tolist(),
            "tip_error_phase_mean_m": tip_error_array.mean(axis=1).tolist(),
            "residual_action_phase_mean": residual_array.mean(axis=1).tolist(),
            "d_action_phase_mean": d_action_array[..., 0, :].mean(axis=1).tolist(),
            "d_state_phase_mean": d_state_array[..., 0, :].mean(axis=1).tolist(),
            "sensitivity_phase_index": sensitivity_phases,
            "first_action_cost_map_directional_sensitivity": np.concatenate(
                sensitivities
            ).tolist(),
            "episode_returns": returns.tolist(),
            "sparse_reward_episode_mean": float(sparse_reward_array.sum(axis=0).mean()),
            "sparse_reward_episode_std": float(sparse_reward_array.sum(axis=0).std()),
            "sparse_reward_step_mean": float(sparse_reward_array.mean()),
            "dense_reward_episode_mean": float(dense_reward_array.sum(axis=0).mean()),
            "dense_reward_episode_std": float(dense_reward_array.sum(axis=0).std()),
            "dense_reward_step_mean": float(dense_reward_array.mean()),
            "reward_component_sum_episode_mean": float(
                (sparse_reward_array + dense_reward_array).sum(axis=0).mean()
            ),
            **residual_entry._physical_metrics(physical, residual_array, feedforward_array),
            **_stats("q_state_over_base", q_state_ratio),
            **_stats("r_over_base", q_action_ratio),
            **_stats("d_action", d_action_array),
            **_stats("d_state", d_state_array),
            **_stats("state_cost_gate", state_gate_array),
            **base_integrity,
        }
        residual_update_diagnostics = getattr(
            learner.actor, "residual_parameter_update_diagnostics", None
        )
        if callable(residual_update_diagnostics):
            result.update(residual_update_diagnostics())
        absolute_d_action = np.abs(d_action_array)
        for quantile, label in ((0.50, "p50"), (0.90, "p90"), (0.95, "p95"), (0.99, "p99")):
            result[f"d_action_abs_{label}"] = float(
                np.quantile(absolute_d_action, quantile)
            )
        result["d_action_abs_max"] = float(absolute_d_action.max())
        for threshold, label in (
            (0.0090, "0p0090"),
            (0.0095, "0p0095"),
            (0.0099, "0p0099"),
        ):
            exceeded = absolute_d_action > threshold
            result[f"d_action_fraction_above_{label}"] = float(exceeded.mean())
            result[f"d_action_fraction_above_{label}_by_actuator"] = exceeded.mean(
                axis=(0, 1, 2)
            ).tolist()
            result[f"d_action_fraction_above_{label}_by_stage"] = exceeded.mean(
                axis=(0, 1, 3)
            ).tolist()
            result[f"d_action_fraction_above_{label}_by_phase"] = exceeded.mean(
                axis=(1, 2, 3)
            ).tolist()
        configured_action_limit = getattr(learner.actor, "action_headroom_limit", None)
        if configured_action_limit is None:
            configured_action_limit = getattr(
                learner.actor, "action_cost_center_limit", None
            )
        if configured_action_limit is not None:
            configured_action_limit = float(configured_action_limit)
            if np.isfinite(configured_action_limit) and configured_action_limit > 0:
                result["d_action_configured_limit"] = configured_action_limit
                result["d_action_limit_utilization_p50"] = float(
                    np.quantile(absolute_d_action, 0.50) / configured_action_limit
                )
                result["d_action_limit_utilization_p95"] = float(
                    np.quantile(absolute_d_action, 0.95) / configured_action_limit
                )
                result["d_action_limit_utilization_max"] = float(
                    absolute_d_action.max() / configured_action_limit
                )
                result["d_action_cost_center_saturation_fraction"] = float(
                    np.mean(absolute_d_action >= 0.99 * configured_action_limit)
                )
        result["d_action_abs_phase_mean"] = absolute_d_action.mean(
            axis=(1, 2, 3)
        ).tolist()
        if implicit_xyz_rows:
            implicit_arrays = {
                key: np.stack(values)
                for key, values in implicit_xyz_rows.items()
            }
            for key, value in implicit_arrays.items():
                result.update(_stats(f"implicit_{key}", value))
            for key in (
                "d_nom_xyz",
                "delta_d_xyz",
                "d_xyz",
                "delta_d_xyz_utilization",
                "q_xyz_over_base",
                "d_action",
                "predicted_xyz_error_to_d",
            ):
                value = implicit_arrays[key]
                result[f"implicit_{key}_phase_mean"] = value.mean(
                    axis=(1, 2)
                ).tolist()
                result[f"implicit_{key}_p95_abs_by_coordinate"] = np.quantile(
                    np.abs(value), 0.95, axis=(0, 1, 2)
                ).tolist()
            velocity_cost = implicit_arrays["velocity_cost_contribution"]
            result["implicit_velocity_cost_contribution_phase_mean"] = (
                velocity_cost.mean(axis=(1, 2)).tolist()
            )
            result["implicit_current_xyz_phase_mean"] = implicit_arrays[
                "current_xyz"
            ].mean(axis=1).tolist()

            def grouped_fraction(
                prefix: str, mask: np.ndarray
            ) -> None:
                # mask layout is [phase, episode, horizon, node*xyz].
                shaped = mask.reshape(*mask.shape[:-1], 3, 3)
                result[f"{prefix}_all"] = float(mask.mean())
                result[f"{prefix}_by_coordinate"] = mask.mean(
                    axis=(0, 1, 2)
                ).tolist()
                result[f"{prefix}_by_node"] = shaped.mean(
                    axis=(0, 1, 2, 4)
                ).tolist()
                result[f"{prefix}_by_axis"] = shaped.mean(
                    axis=(0, 1, 2, 3)
                ).tolist()
                result[f"{prefix}_by_stage"] = mask.mean(
                    axis=(0, 1, 3)
                ).tolist()
                result[f"{prefix}_by_phase"] = mask.mean(
                    axis=(1, 2, 3)
                ).tolist()

            d_utilization = np.abs(
                implicit_arrays["delta_d_xyz_utilization"]
            )
            for quantile, label in (
                (0.50, "p50"),
                (0.90, "p90"),
                (0.95, "p95"),
                (0.99, "p99"),
                (1.00, "max"),
            ):
                result[f"implicit_delta_d_utilization_{label}"] = float(
                    np.quantile(d_utilization, quantile)
                )
            for threshold, label in ((0.90, "gt90"), (0.95, "gt95"), (0.99, "gt99")):
                grouped_fraction(
                    f"implicit_delta_d_utilization_fraction_{label}",
                    d_utilization > threshold,
                )
            physical_delta = np.abs(
                implicit_arrays["delta_d_xyz_physical_m"]
            )
            for quantile, label in ((0.50, "p50"), (0.95, "p95"), (1.00, "max")):
                result[f"implicit_delta_d_physical_{label}_m"] = float(
                    np.quantile(physical_delta, quantile)
                )

            delta_q = implicit_arrays["delta_q_xyz"]
            q_upper = float(
                getattr(
                    learner.actor,
                    "final_q_log_upper_bound",
                    getattr(learner.actor, "q_log_upper_bound"),
                )
            )
            q_lower = float(
                getattr(
                    learner.actor,
                    "final_q_log_lower_bound",
                    getattr(learner.actor, "q_log_lower_bound"),
                )
            )
            q_span = q_upper - q_lower
            q_ratio = implicit_arrays["q_xyz_over_base"]
            for quantile, label in (
                (0.50, "p50"),
                (0.90, "p90"),
                (0.95, "p95"),
                (0.99, "p99"),
                (1.00, "max"),
            ):
                result[f"implicit_q_over_base_{label}"] = float(
                    np.quantile(q_ratio, quantile)
                )
                result[f"implicit_delta_q_upper_margin_{label}"] = float(
                    np.quantile(q_upper - delta_q, quantile)
                )
            for top_fraction, label in ((0.10, "top10"), (0.05, "top05"), (0.01, "top01")):
                grouped_fraction(
                    f"implicit_delta_q_fraction_{label}",
                    delta_q >= q_upper - top_fraction * q_span,
                )
            if "online_delta_D_utilization" in implicit_arrays:
                online_d_utilization = np.abs(
                    implicit_arrays["online_delta_D_utilization"]
                )
                for quantile, label in (
                    (0.50, "p50"),
                    (0.90, "p90"),
                    (0.95, "p95"),
                    (0.99, "p99"),
                    (1.00, "max"),
                ):
                    result[f"online_residual_D_utilization_{label}"] = float(
                        np.quantile(online_d_utilization, quantile)
                    )
                for threshold, label in (
                    (0.90, "gt90"),
                    (0.95, "gt95"),
                    (0.99, "gt99"),
                ):
                    grouped_fraction(
                        f"online_residual_D_utilization_fraction_{label}",
                        online_d_utilization > threshold,
                    )
                online_d_physical = np.abs(
                    implicit_arrays["online_delta_D_xyz"]
                    * learner.actor.koopman.physical_scale[
                        learner.actor.xyz_indices
                    ].detach().cpu().numpy()
                )
                for quantile, label in (
                    (0.50, "p50"),
                    (0.95, "p95"),
                    (1.00, "max"),
                ):
                    result[f"online_residual_D_physical_{label}_m"] = float(
                        np.quantile(online_d_physical, quantile)
                    )
            if "online_delta_q_xyz" in implicit_arrays:
                online_q = np.abs(implicit_arrays["online_delta_q_xyz"])
                for quantile, label in (
                    (0.50, "p50"),
                    (0.95, "p95"),
                    (1.00, "max"),
                ):
                    result[f"online_residual_delta_q_abs_{label}"] = float(
                        np.quantile(online_q, quantile)
                    )
                for threshold, label in (
                    (0.18, "gt0p18"),
                    (0.19, "gt0p19"),
                    (0.198, "gt0p198"),
                ):
                    result[f"online_residual_delta_q_fraction_{label}"] = float(
                        np.mean(online_q > threshold)
                    )
            if "online_delta_d_action" in implicit_arrays:
                online_action = np.abs(
                    implicit_arrays["online_delta_d_action"]
                )
                for quantile, label in (
                    (0.50, "p50"),
                    (0.95, "p95"),
                    (1.00, "max"),
                ):
                    result[f"online_residual_delta_d_abs_{label}"] = float(
                        np.quantile(online_action, quantile)
                    )
                for threshold, label in (
                    (0.0009, "gt0p0009"),
                    (0.00095, "gt0p00095"),
                    (0.00099, "gt0p00099"),
                ):
                    result[f"online_residual_delta_d_fraction_{label}"] = float(
                        np.mean(online_action > threshold)
                    )
        for threshold in (0.010, 0.012, 0.015, 0.020):
            label = f"{threshold:.3f}".replace(".", "p")
            exceeded = absolute_d_action > threshold
            result[f"d_action_fraction_above_{label}"] = float(exceeded.mean())
            result[f"d_action_fraction_above_{label}_by_actuator"] = exceeded.mean(
                axis=(0, 1, 2)
            ).tolist()
            result[f"d_action_fraction_above_{label}_by_phase"] = exceeded.mean(
                axis=(1, 2, 3)
            ).tolist()
            result[
                f"d_action_fraction_above_{label}_by_phase_actuator"
            ] = exceeded.mean(axis=(1, 2)).tolist()
        if headroom_rows:
            stacked_headroom = {
                key: np.stack(rows) for key, rows in headroom_rows.items()
            }
            delta_r = stacked_headroom["delta_r"]
            delta_from_source = (
                stacked_headroom["d_action_new"]
                - stacked_headroom["d_action_source"]
            )

            def absolute_quantiles(prefix: str, value: np.ndarray) -> dict[str, float]:
                absolute = np.abs(value).reshape(-1)
                return {
                    f"{prefix}_mean": float(value.mean()),
                    f"{prefix}_std": float(value.std()),
                    f"{prefix}_abs_p50": float(np.quantile(absolute, 0.50)),
                    f"{prefix}_abs_p95": float(np.quantile(absolute, 0.95)),
                    f"{prefix}_abs_max": float(absolute.max()),
                }

            result.update(absolute_quantiles("delta_r", delta_r))
            result.update(
                absolute_quantiles(
                    "delta_d_action_from_source", delta_from_source
                )
            )
            result.update(
                absolute_quantiles(
                    "headroom_adapter_derivative",
                    stacked_headroom["adapter_derivative"],
                )
            )
            result.update(
                absolute_quantiles(
                    "old_effective_tanh_derivative",
                    stacked_headroom["old_effective_derivative"],
                )
            )
            result["delta_r_phase_abs_mean"] = np.abs(delta_r).mean(
                axis=(1, 2, 3)
            ).tolist()
            result["delta_d_action_from_source_phase_mean"] = delta_from_source.mean(
                axis=(1, 2)
            ).tolist()
        if _STRUCTURE in {"k10v2", "k10p"}:
            if _NOMINAL_NORMALIZED_STATES is None:
                raise RuntimeError("Nominal normalized trajectory diagnostics are missing")
            learned_center = d_state_array[..., 0, :].mean(axis=1)
            nominal_center = _NOMINAL_NORMALIZED_STATES[:1000]
            xref_center = learner.actor.xref_table[:1000].detach().cpu().numpy()

            def center_metrics(prefix: str, target: np.ndarray) -> dict[str, float]:
                difference = learned_center - target
                learned_flat = learned_center.reshape(-1)
                target_flat = target.reshape(-1)
                correlation = float(np.corrcoef(learned_flat, target_flat)[0, 1])
                return {
                    f"d_state_{prefix}_mse_normalized": float(np.mean(difference**2)),
                    f"d_state_{prefix}_rmse_normalized": float(np.sqrt(np.mean(difference**2))),
                    f"d_state_{prefix}_correlation": correlation,
                }

            result.update(center_metrics("vs_xff", nominal_center))
            result.update(center_metrics("vs_xref", xref_center))
            result["d_state_correction_from_xff_phase_mean"] = (
                learned_center - nominal_center
            ).tolist()
        for key, rows in solver_rows.items():
            values = np.concatenate(rows)
            result[f"kmpc_{key}_mean"] = float(values.mean())
            result[f"kmpc_{key}_p95"] = float(np.quantile(values, 0.95))
            result[f"kmpc_{key}_max"] = float(values.max())
        result["kmpc_terminal_fraction_mean"] = float(
            result["kmpc_terminal_contribution_mean"]
            / max(abs(result["kmpc_objective_mean"]), 1e-12)
        )
        if trajectory_output_raw:
            trajectory_output = Path(trajectory_output_raw).expanduser().resolve()
            trajectory_output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                trajectory_output,
                observations=np.stack(trajectory_observations),
                physical_states=np.stack(trajectory_observations)[..., :45],
                rewards=np.stack(trajectory_rewards),
                requested_residual_actions=np.stack(trajectory_requested_residuals),
                applied_residual_actions=residual_array,
                physical_applied_actions=physical,
                feedforward_actions=feedforward_array,
                target_positions=np.stack(trajectory_targets),
                node_errors=np.stack(trajectory_node_errors),
                joint_errors=error_array,
                tip_errors=tip_error_array,
                q_state=q_state_array,
                q_action=q_action_array,
                d_state=d_state_array,
                d_action=d_action_array,
                state_cost_gate=state_gate_array,
                **(
                    {f"implicit_{key}": value for key, value in implicit_arrays.items()}
                    if implicit_xyz_rows
                    else {}
                ),
            )
        return result
    finally:
        env.close()


def _evaluate_plain_actor(learner: Any, *, seed_base: int, episodes: int) -> dict[str, Any]:
    env = training._environment(episodes, seed_base, workers=episodes)
    try:
        observation = env.reset()
        returns = np.zeros(episodes, dtype=np.float64)
        errors: list[np.ndarray] = []
        tip_errors: list[np.ndarray] = []
        physical_actions: list[np.ndarray] = []
        residual_actions: list[np.ndarray] = []
        feedforward_actions: list[np.ndarray] = []
        for _ in range(1000):
            residual = learner.act(observation, deterministic=True)
            vector_step = env.step(residual)
            returns += vector_step.reward
            errors.append(np.asarray([item["joint_target_error"] for item in vector_step.info], dtype=np.float64))
            tip_errors.append(np.asarray([item["node_target_error"][2] for item in vector_step.info], dtype=np.float64))
            residual_actions.append(np.asarray(vector_step.applied_action, dtype=np.float64))
            physical_actions.append(np.asarray([item["physical_applied_action"] for item in vector_step.info], dtype=np.float64))
            feedforward_actions.append(np.asarray([item["feedforward_action"] for item in vector_step.info], dtype=np.float64))
            observation = vector_step.observation
        error_array = np.stack(errors)
        tip_error_array = np.stack(tip_errors)
        physical = np.stack(physical_actions)
        residual_array = np.stack(residual_actions)
        feedforward_array = np.stack(feedforward_actions)
        zeros = np.zeros((error_array.shape[0], episodes, 1), dtype=np.float64)
        return {
            **training.evaluation_seed_metadata(
                seed_base=seed_base, episodes=episodes
            ),
            "return_mean": float(returns.mean()), "return_std": float(returns.std()),
            "return_min": float(returns.min()), "return_max": float(returns.max()),
            "reward_rate": float(returns.mean() / 1000.0),
            "joint_error_rmse_m": float(np.sqrt(np.mean(error_array ** 2))),
            "joint_error_p95_m": float(np.quantile(error_array, 0.95)),
            "success_rate_2p5mm": float(np.mean(error_array <= 0.0025)),
            "tip_rmse_m": float(np.sqrt(np.mean(tip_error_array ** 2))),
            "tip_p95_m": float(np.quantile(tip_error_array, 0.95)),
            "tip_success_rate_2p5mm": float(np.mean(tip_error_array <= 0.0025)),
            "joint_error_phase_mean_m": error_array.mean(axis=1).tolist(),
            "residual_action_phase_mean": residual_array.mean(axis=1).tolist(),
            "d_action_phase_mean": zeros[..., 0, :].tolist(),
            "d_state_phase_mean": zeros[..., 0, :].tolist(),
            "sensitivity_phase_index": [], "first_action_cost_map_directional_sensitivity": [],
            "episode_returns": returns.tolist(),
            **residual_entry._physical_metrics(physical, residual_array, feedforward_array),
            **_stats("q_state_over_base", zeros), **_stats("r_over_base", zeros),
            **_stats("d_action", zeros), **_stats("d_state", zeros),
            **_stats("state_cost_gate", zeros),
            "kmpc_terminal_fraction_mean": 0.0,
            "kmpc_objective_mean": 0.0, "kmpc_objective_p95": 0.0, "kmpc_objective_max": 0.0,
            "kmpc_terminal_contribution_mean": 0.0, "kmpc_terminal_contribution_p95": 0.0,
            "kmpc_terminal_contribution_max": 0.0,
            "kmpc_active_constraint_fraction_mean": 0.0, "kmpc_active_constraint_fraction_p95": 0.0,
            "kmpc_active_constraint_fraction_max": 0.0,
            "kmpc_projected_gradient_relative_mean": 0.0, "kmpc_projected_gradient_relative_p95": 0.0,
            "kmpc_projected_gradient_relative_max": 0.0,
            "kmpc_solver_converged_mean": 1.0, "kmpc_solver_converged_p95": 1.0,
            "kmpc_solver_converged_max": 1.0,
        }
    finally:
        env.close()


def parse() -> tuple[argparse.Namespace, argparse.Namespace]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--control-structure", choices=STRUCTURES, required=True)
    parser.add_argument("--feedforward", type=Path, required=True)
    parser.add_argument("--residual-limit", type=float, default=0.3)
    parser.add_argument("--state-cost-scale", type=float, default=1.0)
    parser.add_argument("--residual-cost-scale", type=float, default=10_000.0)
    parser.add_argument("--terminal-multiplier", type=float, default=1.0)
    parser.add_argument("--state-cost-gate-init", type=float, default=1e-4)
    parser.add_argument(
        "--selected-continuation",
        action="store_true",
        help="Allow the post-screen selected-method continuation schedule.",
    )
    parser.add_argument(
        "--k10-experiment",
        action="store_true",
        help="Enable the K10 FF-preserving no-xref experiment schedule.",
    )
    parser.add_argument(
        "--dense-xref-screen",
        action="store_true",
        help="Enable the reward-only pure full-state xref dense screen.",
    )
    parser.add_argument(
        "--next-stage-experiment",
        action="store_true",
        help="Enable the K2/K10-v2/K10-P 20k direct-online matrix.",
    )
    parser.add_argument(
        "--time-only-o2o",
        action="store_true",
        help="Run the t-only 10k-offline then online reproduction protocol.",
    )
    parser.add_argument("--nominal-center-trajectory", type=Path)
    parser.add_argument("--warm-init-ridge", type=float, default=1e-4)
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    args = training.parse_args()
    return known, args


def main() -> None:
    global _FEEDFORWARD, _XREF, _STRUCTURE, _RESIDUAL_LIMIT
    global _STATE_COST_SCALE, _RESIDUAL_COST_SCALE, _TERMINAL_MULTIPLIER
    global _STATE_COST_GATE_INIT, _NOMINAL_OBSERVATIONS, _NOMINAL_PHYSICAL_STATES
    global _WARM_INIT_RIDGE, _WARM_INIT_DIAGNOSTICS
    known, args = parse()
    _STRUCTURE = known.control_structure
    _RESIDUAL_LIMIT = float(known.residual_limit)
    _STATE_COST_SCALE = float(known.state_cost_scale)
    _RESIDUAL_COST_SCALE = float(known.residual_cost_scale)
    _TERMINAL_MULTIPLIER = float(known.terminal_multiplier)
    _STATE_COST_GATE_INIT = float(known.state_cost_gate_init)
    _WARM_INIT_RIDGE = float(known.warm_init_ridge)
    _WARM_INIT_DIAGNOSTICS = None
    if args.method != "Cal-RLPD-KMPC":
        raise ValueError("The causal screen requires Cal-RLPD-KMPC")
    if known.k10_experiment and _STRUCTURE != "k10":
        raise ValueError("--k10-experiment requires --control-structure k10")
    if known.next_stage_experiment and _STRUCTURE not in {"k2", "k10v2", "k10p"}:
        raise ValueError("--next-stage-experiment only supports K2, K10-v2, and K10-P")
    if _STRUCTURE in {"k10v2", "k10p"}:
        if not known.next_stage_experiment or known.nominal_center_trajectory is None:
            raise ValueError("K10-v2/P requires --next-stage-experiment and --nominal-center-trajectory")
        nominal_path = known.nominal_center_trajectory.expanduser().resolve()
        with np.load(nominal_path, allow_pickle=False) as archive:
            _NOMINAL_OBSERVATIONS = np.asarray(archive["observations"], dtype=np.float32)
            _NOMINAL_PHYSICAL_STATES = np.asarray(archive["physical_states"], dtype=np.float32)
        if (
            _NOMINAL_OBSERVATIONS.shape != (1001, 46)
            or _NOMINAL_PHYSICAL_STATES.shape != (1001, 45)
            or not np.isfinite(_NOMINAL_OBSERVATIONS).all()
            or not np.isfinite(_NOMINAL_PHYSICAL_STATES).all()
        ):
            raise ValueError("Nominal center trajectory must contain finite [1001,46]/[1001,45] arrays")
    if known.time_only_o2o:
        expected = {
            "offline_updates": 10_000,
            "online_eval_interval": 5_000,
            "online_utd": 1,
            "actor_update_interval": 1,
            "offline_replay_ratio": 0.5,
            "q_cost_anchor_weight": 0.0,
            "p_cost_anchor_weight": 0.0,
        }
    else:
        expected = {
            "offline_updates": 0,
        "online_eval_interval": (
            2000
            if known.dense_xref_screen or known.k10_experiment or known.selected_continuation or known.next_stage_experiment
            else 250
        ),
        "online_utd": 1,
        "actor_update_interval": 2,
        "offline_replay_ratio": 0.5,
        "q_cost_anchor_weight": 0.0,
        "p_cost_anchor_weight": 0.0,
    }
    for name, value in expected.items():
        if getattr(args, name) != value:
            raise ValueError(f"Frozen causal-screen setting {name} must equal {value}")
    allowed_steps = (
        (20_000,)
        if known.time_only_o2o
        else
        (20000,)
        if known.next_stage_experiment
        else
        (5000, 10000)
        if known.dense_xref_screen
        else (20000,)
        if known.k10_experiment
        else (30000,)
        if known.selected_continuation
        else (5000, 20000, 25000)
    )
    if args.online_steps not in allowed_steps:
        raise ValueError(f"Unsupported total online steps {args.online_steps}; allowed={allowed_steps}")
    expected_actor_lr = (
        1e-4
        if known.time_only_o2o
        else
        args.actor_learning_rate
        if known.next_stage_experiment
        else 1e-6
        if known.selected_continuation
        or (known.dense_xref_screen and args.online_steps == 10000)
        else 5e-7
    )
    if known.next_stage_experiment and args.actor_learning_rate not in {1e-6, 1.5e-6, 2e-6}:
        raise ValueError("Next-stage actor learning rate must be 1e-6, 1.5e-6, or 2e-6")
    expected_critic_lr = 1e-4 if known.time_only_o2o else 5e-5
    if args.actor_learning_rate != expected_actor_lr or args.critic_learning_rate != expected_critic_lr:
        raise ValueError(
            f"Expected actor/critic learning rates {expected_actor_lr:g} and 5e-5"
        )
    if args.kmpc_horizon != 5:
        raise ValueError("Frozen horizon is H=5")
    if args.online_cql_mode != "off":
        raise ValueError("Direct-online RLPD phase requires online CQL off")
    if known.dense_xref_screen or known.next_stage_experiment:
        if (
            args.reward_mode != "dense_xref"
            or args.sparse_reward_weight != 0.0
            or args.dense_reward_weight != 1.0
            or args.dense_reward_scale_m != 0.01
        ):
            raise ValueError(
                "Pure dense-xref screen requires dense_xref, sparse=0, "
                "dense=1, and scale=0.01 m"
            )
    elif args.reward_mode not in {"hybrid", "dense_joint"}:
        raise ValueError("Expected hybrid or dense_joint reward")
    if args.reward_mode == "dense_joint" and (
        args.sparse_reward_weight != 0.0
        or args.dense_reward_weight != 1.0
        or args.dense_reward_scale_m != 0.01
    ):
        raise ValueError("dense_joint requires sparse=0, dense=1, scale=0.01")

    feedforward_path = known.feedforward.expanduser().resolve()
    _FEEDFORWARD = FrozenCirclePhaseFeedforward(feedforward_path)
    with np.load(args.reference.expanduser().resolve(), allow_pickle=False) as archive:
        _XREF = np.asarray(archive["xref"], dtype=np.float32)
    os.environ[FEEDFORWARD_ENV] = str(feedforward_path)
    os.environ[RESIDUAL_LIMIT_ENV] = str(_RESIDUAL_LIMIT)
    # The expanded mixed buffer contains actions up to 0.5; keep the
    # residual dataset, actor box, and physical environment in the same box.
    circle_env_module.ABSOLUTE_ACTION_LIMIT = 0.5
    training.ABSOLUTE_ACTION_LIMIT = 0.5
    residual_entry.ABSOLUTE_ACTION_LIMIT = 0.5

    source_spec = config_module.TRAIN_METHOD_SPECS["Cal-RLPD-KMPC"]
    screen_spec = replace(
        source_spec,
        backup_entropy=False,
        profile=source_spec.profile + "_direct_online_control_structure_v1",
    )
    config_module.STANDALONE_METHOD_SPECS["Cal-RLPD-KMPC"] = screen_spec
    config_module.TRAIN_METHOD_SPECS["Cal-RLPD-KMPC"] = screen_spec
    config_module.o2o_action_limit = _residual_action_limit
    learner_module.o2o_action_limit = _residual_action_limit
    learner_module.build_actor = _build_actor
    training.O2OLearner = _NominalWarmInitializedLearner
    training.ManiSoftCircleOfflineDataset = ManiSoftCircleTimeResidualDataset
    training.FrozenManiSoftHistoryKoopman = FrozenManiSoftTimeKoopman
    training.make_manisoft_circle_o2o_adapter = make_manisoft_circle_time_residual_adapter
    koopman_payload = torch.load(args.koopman, map_location="cpu", weights_only=False)
    history_steps = int(koopman_payload.get("architecture", {}).get("history_steps", 0))
    training.COLLECTOR_OBSERVATION_DIM = 46 + history_steps * (45 + 18)
    residual_entry._RESIDUAL_LIMIT = _RESIDUAL_LIMIT
    training.evaluate = evaluate_control_structure

    normalized_reference = (
        _XREF - np.asarray(koopman_payload["normalizers"]["state"]["mean"])
    ) / np.maximum(
        np.asarray(koopman_payload["normalizers"]["state"]["std"]), 1e-6
    )
    d_state_limit = np.maximum(np.max(np.abs(normalized_reference), axis=0), 1e-3)
    protocol = {
        "kind": "manisoft_direct_online_control_structure_screen_v1",
        "structure": _STRUCTURE,
        "actor_input": "[Koopman lifted body state z_t, tau=t/(T-1)]",
        "actor_xref_input": False,
        "actor_feedforward_input": False,
        "kmpc_xref_lookup": _STRUCTURE not in {"k9", "k10", "k10v2", "k10p"},
        "kmpc_feedforward_lookup": True,
        "koopman_action": "absolute u = u_ff(t) + residual",
        "horizon": 5,
        "shared_horizon": _STRUCTURE in {"k3", "k4", "k5", "k6", "k8", "k9", "k10"},
        "terminal_cost": _STRUCTURE in {"k5", "k6", "k8"},
        "terminal_P": f"{_TERMINAL_MULTIPLIER} * Qbase_state",
        "state_q_adaptive": _STRUCTURE not in {"k0", "k10p"},
        "action_r_adaptive": _STRUCTURE in {"k7", "k8"},
        "action_p": _STRUCTURE in {"k2", "k4", "k6", "k7", "k8", "k9", "k10", "k10v2", "k10p"},
        "state_p": _STRUCTURE in {"k9", "k10", "k10v2", "k10p"},
        "d_action_max": 0.005,
        "d_state_max_basis": "per-dimension max(abs(normalized xref)); floor=1e-3",
        "d_state_max": d_state_limit.tolist() if _STRUCTURE in {"k9", "k10", "k10v2", "k10p"} else None,
        "state_cost_gate": _STATE_COST_GATE_INIT if _STRUCTURE == "k10" else None,
        "q_r_normalization": (
            "legacy joint 63-D centering" if _STRUCTURE == "k7"
            else "independent within state-Q and action-R groups"
        ),
        "zero_head": (
            "Q=Qbase,d_state=nominal_ff,d_action=0"
            if _STRUCTURE in {"k10v2", "k10p"}
            else "Q=Qbase,p=0"
        ),
        "nominal_center_trajectory": (
            str(known.nominal_center_trajectory.expanduser().resolve())
            if known.nominal_center_trajectory is not None
            else None
        ),
        "warm_init_ridge": float(_WARM_INIT_RIDGE) if _STRUCTURE in {"k10v2", "k10p"} else None,
        "actor_frozen": _STRUCTURE == "k0",
        "offline_gradient_pretraining": 0,
        "online_steps": int(args.online_steps),
        "offline_online_replay_ratio": [0.5, 0.5],
        "actor_lr": float(args.actor_learning_rate),
        "critic_lr": float(args.critic_learning_rate),
        "actor_update_interval": 2,
        "critic_utd": 1,
        "backup_entropy": False,
        "reward": {
            "mode": args.reward_mode,
            "sparse_weight": float(args.sparse_reward_weight),
            "dense_weight": float(args.dense_reward_weight),
            "dense_scale_m": float(args.dense_reward_scale_m),
            "xref_lookup": "full physical xref at t+1"
            if args.reward_mode == "dense_xref"
            else "selected-node target positions at t+1",
            "xref_in_actor_observation": False,
        },
    }
    original_checkpoint = training._checkpoint

    def checkpoint_with_protocol(**kwargs: Any) -> dict[str, Any]:
        result = original_checkpoint(**kwargs)
        result["control_structure_screen"] = protocol
        return result

    training._checkpoint = checkpoint_with_protocol
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "control_structure_protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    training.run(args)
    run_path = output / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if _WARM_INIT_DIAGNOSTICS is not None:
        protocol["warm_initialization"] = _WARM_INIT_DIAGNOSTICS
        (output / "control_structure_protocol.json").write_text(
            json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    run["control_structure_screen"] = protocol
    temporary = run_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, run_path)


if __name__ == "__main__":
    main()

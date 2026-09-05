#!/usr/bin/env python
"""ManiSoft Circle formal offline/online reproduction entry point.

This is the reproduction driver referenced as ``$TRAINER`` by
``manisoft_port/protocols/MANISOFT_CIRCLE_FORMAL_REPRODUCTION.md`` (archive
copy: ``runs/o2o/formal/.../REPRODUCTION.md``).  It trains the time-residual
ManiSoft Circle task, where the physical action is a frozen Koopman
feedforward plus a learned residual:

    physical_u = clip(u_ff(t) + residual_u, -0.5, 0.5)

The file owns no training loop of its own.  It imports the shared circle O2O
loop (``train_manisoft_circle_o2o``, re-exported by ``implicit_kmpc``) and
monkey-patches the method specs, actor factory, dataset and task adapter so
that every formal method (AWAC / IQL / Cal-QL / RLPD / AWAC-KMPC /
AWAC-lift / AWAC-raw) is selected purely through CLI flags.  ``main()`` also
locks the offline/online budget to a fixed protocol grid (formal archive:
offline 10k updates, online 15k steps, evaluation/checkpoint every 2.5k)
and writes ``time_only_protocol.json`` as an audit record.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import train_manisoft_circle_control_structure as control_entry
import train_manisoft_circle_implicit_kmpc as implicit_entry
import train_manisoft_circle_o2o as base_training
from experiments.dmc.o2o.dataset import mark_offline
from antmaze_ac.data.circle_time_residual_dataset import ManiSoftCircleTimeResidualDataset
from antmaze_ac.envs.circle_phase_feedforward import FrozenCirclePhaseFeedforward
from antmaze_ac.envs.manisoft_circle_time_residual_env import (
    FEEDFORWARD_ENV,
    RESIDUAL_LIMIT_ENV,
    make_manisoft_circle_time_residual_adapter,
)
from antmaze_ac.koopman.o2o_time_adapter import FrozenManiSoftTimeKoopman
from antmaze_ac.rl.time_structured_qp_kmpc_actor import TimeStructuredQpKMPCTanhGaussianActor
from antmaze_ac.rl.time_implicit_xyz_kmpc_actor import (
    TimeImplicitXyzQpKMPCTanhGaussianActor,
    VELOCITY_INDICES,
    XYZ_INDICES,
)
from antmaze_ac.rl.frozen_base_full_residual_actor import (
    FrozenBaseFullResidualImplicitXyzActor,
)
import experiments.dmc.o2o.config as config_module
import experiments.dmc.o2o.learner as learner_module
import antmaze_ac.envs.manisoft_circle_o2o_env as circle_env_module

# Module-level actor/controller options, filled from the CLI in ``main()``.
# ``training`` is the shared circle O2O loop object that this file patches.
training = implicit_entry.training
# Snapshot the untouched base implementations so main() can override them.
_BASE_BUILD_ACTOR = learner_module.build_actor
_BASE_GET_TASK_SPEC = learner_module.get_task_spec
_BASE_ACTION_LIMIT = config_module.o2o_action_limit
_FEEDFORWARD: FrozenCirclePhaseFeedforward | None = None
_XREF: np.ndarray | None = None
_RESIDUAL_LIMIT = 0.5
_ACTION_COST_CENTER_LIMIT = 0.005
_ACTION_HEADROOM_LIMIT: float | None = None
_ACTION_HEADROOM_ADAPTER_ONLY = False
_STRUCTURE = "k6"
_ACTIVE_VARIANT = "AWAC-KMPC"
_IMPLICIT_XYZ_NO_XREF = False
_IMPLICIT_XYZ_D_SCALE_M: np.ndarray | None = None
_IMPLICIT_XYZ_VELOCITY_COST_SCALE = 0.05
_IMPLICIT_XYZ_Q_LOG_UPPER = 1.5
_IMPLICIT_XYZ_SANITY_OBSERVATIONS: np.ndarray | None = None
_IMPLICIT_XYZ_OUTPUT: Path | None = None
_FULL_CAPACITY_ONLINE_RESIDUAL = False
_FULL_RESIDUAL_CHANNELS = "DQ"
_BASE_O2O_LEARNER = training.O2OLearner


class _ImplicitXyzSanityLearner(_BASE_O2O_LEARNER):
    """Block training unless the fresh no-xref actor preserves pure FF."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if not _IMPLICIT_XYZ_NO_XREF or _FULL_CAPACITY_ONLINE_RESIDUAL:
            return
        if _IMPLICIT_XYZ_SANITY_OBSERVATIONS is None or _IMPLICIT_XYZ_OUTPUT is None:
            raise RuntimeError("implicit-xyz sanity inputs are missing")
        observation = torch.as_tensor(
            _IMPLICIT_XYZ_SANITY_OBSERVATIONS,
            dtype=torch.float32,
            device=self.device,
        )
        lifted = self._encode(observation)
        diagnostics = self.actor.zero_update_sanity(lifted)
        (_IMPLICIT_XYZ_OUTPUT / "zero_update_sanity.json").write_text(
            json.dumps(diagnostics, indent=2, sort_keys=True) + "\n"
        )
        print("IMPLICIT_XYZ_ZERO_SANITY " + json.dumps(diagnostics, sort_keys=True))


def _action_limit(task: str) -> float:
    return _RESIDUAL_LIMIT if task == training.TASK_NAME else float(_BASE_ACTION_LIMIT(task))


def _learner_task_spec(task: str) -> Any:
    if task == training.TASK_NAME:
        return SimpleNamespace(action_dim=18)
    return _BASE_GET_TASK_SPEC(task)


# --- Actor factory -------------------------------------------------------
# Plain MLP actor for the official baselines; Koopman-lifted structured
# QP-KMPC tanh-Gaussian actor for the ``*-KMPC`` arms.  The
# ``--implicit-xyz-no-xref`` variant drops the global xref look-up and, with
# ``--full-capacity-online-residual`` (C3 recipe), freezes the bootstrapped
# actor and trains a same-capacity residual actor online.
def _build_actor(method: str, koopman: Any, **kwargs: Any):
    # Non-KMPC methods keep the untouched base MLP actor construction.
    if method not in {"AWAC-KMPC", "Cal-RLPD-KMPC"} or kwargs.get("actor_kind") != "ac_kmpc":
        return _BASE_BUILD_ACTOR(method, koopman, **kwargs)
    if koopman is None or _FEEDFORWARD is None or (
        _XREF is None and not _IMPLICIT_XYZ_NO_XREF
    ):
        raise RuntimeError("time-only AWAC-KMPC dependencies are missing")
    if _IMPLICIT_XYZ_NO_XREF:
        if _IMPLICIT_XYZ_D_SCALE_M is None:
            raise RuntimeError("implicit-xyz D scale is missing")
        actor_class = (
            FrozenBaseFullResidualImplicitXyzActor
            if _FULL_CAPACITY_ONLINE_RESIDUAL
            else TimeImplicitXyzQpKMPCTanhGaussianActor
        )
        extra_options = (
            {"residual_channels": _FULL_RESIDUAL_CHANNELS}
            if _FULL_CAPACITY_ONLINE_RESIDUAL
            else {}
        )
        return actor_class(
            koopman,
            _FEEDFORWARD,
            horizon=int(kwargs["kmpc_horizon"]),
            solver_iterations=int(kwargs["kmpc_solver_iterations"]),
            hidden_dim=int(kwargs["controller_hidden_dim"]),
            hidden_layers=int(kwargs["controller_hidden_layers"]),
            residual_limit=_RESIDUAL_LIMIT,
            physical_action_limit=0.5,
            xyz_cost_scale=1.0,
            velocity_cost_scale=_IMPLICIT_XYZ_VELOCITY_COST_SCALE,
            residual_cost_scale=10_000.0,
            q_log_upper_bound=_IMPLICIT_XYZ_Q_LOG_UPPER,
            action_cost_center_limit=_ACTION_COST_CENTER_LIMIT,
            d_xyz_scale_m=_IMPLICIT_XYZ_D_SCALE_M,
            log_std_init=float(kwargs["kmpc_log_std_init"]),
            log_std_max=float(kwargs["kmpc_log_std_max"]),
            **extra_options,
        )
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
        physical_action_limit=0.5,
        state_cost_scale=1.0,
        residual_cost_scale=10_000.0,
        action_cost_center_limit=_ACTION_COST_CENTER_LIMIT,
        action_headroom_limit=_ACTION_HEADROOM_LIMIT,
        action_headroom_adapter_only=_ACTION_HEADROOM_ADAPTER_ONLY,
        log_std_init=float(kwargs["kmpc_log_std_init"]),
        log_std_max=float(kwargs["kmpc_log_std_max"]),
    )


# --- Method registry -----------------------------------------------------
# Materialize the REPRODUCTION.md method matrix from the standalone specs.
# AWAC-KMPC is the Koopman-lifted KMPC arm (10-head LayerNorm critics,
# UTD=20); AWAC-lift/AWAC-raw are its representation ablations; AWAC/IQL/
# Cal-QL/RLPD stay close to their published baselines.
def _register_method() -> None:
    # All arms use the same time-residual ManiSoft adapter and degraded ff15.
    # The official baselines retain their algorithmic update rules while the
    # ablations only change representation/controller structure.
    official = config_module.STANDALONE_METHOD_SPECS
    awac = replace(
        official["AWAC"],
        network_profile="plain",
        critic_ensemble_size=2,
        target_critic_subset=2,
        online_utd=1,
        online_warmup_steps=0,
        profile="awac_original_raw_mlp_v1",
    )
    aliases: dict[str, Any] = {"AWAC": awac}
    # Standard IQL/Cal-QL use the raw normalized physical observation.  RLPD
    # keeps its ten LayerNorm critics and UTD=20; ``offline_pretraining`` is
    # enabled for matrix arms that request an explicit offline phase (e.g.
    # the 30k/10k weighted-matrix point).  In the formal online-only campaign
    # RLPD is launched with --offline-updates 0, so its offline row in
    # REPRODUCTION.md is N/A.
    iql = replace(official["IQL"], network_profile="plain", profile="iql_official_raw_mlp_v2")
    calql = replace(official["Cal-QL"], profile="calql_official_raw_mlp_v2")
    rlpd = replace(official["RLPD"], offline_pretraining=True, profile="rlpd_official_raw_mlp_offline_screen_v2")
    aliases.update({"IQL": iql, "Cal-QL": calql, "RLPD": rlpd})
    spec = replace(
        awac,
        name="AWAC-KMPC",
        representation="koopman_lifted",
        actor="ac_kmpc",
        profile="awac_kmpc_a1_rlpd_ensemble_mean_v2",
        controller_hidden_dim=256,
        controller_hidden_layers=2,
        critic_ensemble_size=10,
        target_critic_subset=2,
        network_profile="rlpd",
        actor_q_reduction="mean",
        online_utd=20,
        online_warmup_steps=0,
    )
    aliases["AWAC-KMPC"] = spec
    # Cal-RLPD-KMPC uses the calibrated offline critic phase and RLPD online
    # phase, while reusing the exact same structured AWAC-KMPC actor
    # initialization and controller dimensions.
    cal_spec = replace(
        official["Cal-RLPD-KMPC"],
        name="Cal-RLPD-KMPC",
        controller_hidden_dim=spec.controller_hidden_dim,
        controller_hidden_layers=spec.controller_hidden_layers,
        actor_q_reduction="mean",
        profile="cal_rlpd_kmpc_awac_actor_init_v1",
    )
    aliases["Cal-RLPD-KMPC"] = cal_spec
    # Representation ablations retain the KMPC arm's ten-head LayerNorm
    # ensemble and UTD=20; only the actor/control representation is removed.
    # This keeps AWAC-lift/raw distinct from the official two-critic AWAC arm.
    aliases["AWAC-lift"] = replace(
        spec, name="AWAC-lift", representation="koopman_lifted", actor="mlp",
        profile="awac_lifted_mlp_ensemble_ablation_v2",
    )
    aliases["AWAC-raw"] = replace(
        spec, name="AWAC-raw", representation="raw", actor="mlp",
        profile="awac_raw_mlp_ensemble_ablation_v2",
    )
    config_module.TRAIN_METHOD_SPECS.clear()
    config_module.TRAIN_METHOD_SPECS.update(aliases)
    config_module.STANDALONE_METHOD_SPECS.clear()
    config_module.STANDALONE_METHOD_SPECS.update(aliases)
    config_module.TRAIN_METHODS = tuple(aliases)
    training.FORMAL_METHODS = tuple(aliases)


def _single_replay_batch(offline, online, *, batch_size, utd, offline_ratio, generator):
    """Approximate original AWAC's single replay buffer.

    Offline and online transitions are sampled in proportion to the number of
    transitions available in the conceptual concatenated buffer, rather than
    forcing RLPD's fixed 50/50 split.
    """
    del offline_ratio
    if online.size < 1:
        raise RuntimeError("Online replay is empty")
    total = len(offline) + online.size
    offline_n = int(round(batch_size * len(offline) / total))
    offline_n = min(max(offline_n, 0), batch_size)
    online_n = batch_size - offline_n
    fused = {}
    for _ in range(utd):
        pieces = []
        if offline_n:
            pieces.append(mark_offline(offline.sample(offline_n, generator)))
        if online_n:
            pieces.append(online.sample(online_n, generator))
        keys = tuple(pieces[0])
        batch = {key: np.concatenate([p[key] for p in pieces], axis=0) for key in keys}
        perm = generator.permutation(batch_size)
        for key, value in batch.items():
            fused.setdefault(key, []).append(value[perm])
    return {key: np.concatenate(values, axis=0) for key, values in fused.items()}


def main() -> None:
    # This script parses its own small flag set; every remaining argument is
    # forwarded to the shared O2O parser (--method, --offline-updates,
    # --online-steps, learning rates, replay size, reward mode, ...).  The
    # bash templates in REPRODUCTION.md combine both flag sets.
    global _FEEDFORWARD, _XREF, _RESIDUAL_LIMIT, _ACTION_COST_CENTER_LIMIT
    global _ACTION_HEADROOM_LIMIT, _ACTION_HEADROOM_ADAPTER_ONLY, _ACTIVE_VARIANT
    global _IMPLICIT_XYZ_NO_XREF, _IMPLICIT_XYZ_D_SCALE_M
    global _IMPLICIT_XYZ_VELOCITY_COST_SCALE, _IMPLICIT_XYZ_SANITY_OBSERVATIONS
    global _IMPLICIT_XYZ_OUTPUT, _IMPLICIT_XYZ_Q_LOG_UPPER
    global _FULL_CAPACITY_ONLINE_RESIDUAL, _FULL_RESIDUAL_CHANNELS
    _register_method()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--feedforward", type=Path, required=True)
    parser.add_argument("--no-feedforward", action="store_true")
    parser.add_argument("--disable-backup-entropy", action="store_true")
    parser.add_argument(
        "--action-cost-center-limit",
        type=float,
        default=0.005,
        help="Maximum action-side affine correction d_action for KMPC.",
    )
    parser.add_argument(
        "--policy-preserving-action-headroom-limit",
        type=float,
        default=None,
        help="Expanded action-p center limit using a zero-init preserving adapter.",
    )
    parser.add_argument(
        "--action-headroom-adapter-only",
        action="store_true",
        help="Freeze the inherited actor and train only the preserving adapter.",
    )
    parser.add_argument(
        "--implicit-xyz-no-xref",
        action="store_true",
        help="Use stage-wise implicit xyz centres around the FF Koopman rollout.",
    )
    parser.add_argument(
        "--implicit-xyz-velocity-cost-scale",
        type=float,
        default=0.05,
        help="Fixed normalized linear/angular velocity damping weight.",
    )
    parser.add_argument(
        "--implicit-xyz-d-scale-ratio",
        type=float,
        default=1.0,
        help="Multiply the canonical per-node/per-axis implicit xyz bounds.",
    )
    parser.add_argument(
        "--implicit-xyz-q-log-upper",
        type=float,
        default=1.5,
        help="Positive Delta-q ceiling; the negative ceiling remains -1.5.",
    )
    parser.add_argument(
        "--full-capacity-online-residual",
        action="store_true",
        help=(
            "Freeze a bootstrapped E7 actor and train a same-capacity "
            "decoded cost-map residual actor."
        ),
    )
    parser.add_argument(
        "--full-residual-channels",
        choices=("none", "D", "DQ", "DQa"),
        default="DQ",
    )
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    args = training.parse_args()
    if args.method not in {"AWAC", "IQL", "Cal-QL", "RLPD", "AWAC-KMPC", "Cal-RLPD-KMPC", "AWAC-lift", "AWAC-raw"}:
        raise ValueError(f"Unsupported method {args.method}")
    if known.disable_backup_entropy:
        config_module.TRAIN_METHOD_SPECS[args.method] = replace(
            config_module.TRAIN_METHOD_SPECS[args.method], backup_entropy=False
        )
    _ACTIVE_VARIANT = args.method
    _IMPLICIT_XYZ_NO_XREF = bool(known.implicit_xyz_no_xref)
    _IMPLICIT_XYZ_VELOCITY_COST_SCALE = float(
        known.implicit_xyz_velocity_cost_scale
    )
    d_scale_ratio = float(known.implicit_xyz_d_scale_ratio)
    _IMPLICIT_XYZ_Q_LOG_UPPER = float(known.implicit_xyz_q_log_upper)
    _FULL_CAPACITY_ONLINE_RESIDUAL = bool(
        known.full_capacity_online_residual
    )
    _FULL_RESIDUAL_CHANNELS = str(known.full_residual_channels)
    if _FULL_CAPACITY_ONLINE_RESIDUAL and not _IMPLICIT_XYZ_NO_XREF:
        raise ValueError(
            "full-capacity online residual requires implicit-xyz no-xref"
        )
    if _IMPLICIT_XYZ_NO_XREF and args.method != "AWAC-KMPC":
        raise ValueError("implicit-xyz no-xref currently requires AWAC-KMPC")
    if not np.isfinite(_IMPLICIT_XYZ_VELOCITY_COST_SCALE) or not (
        0.0 < _IMPLICIT_XYZ_VELOCITY_COST_SCALE < 1.0
    ):
        raise ValueError("implicit-xyz velocity cost scale must lie in (0,1)")
    if not np.isfinite(d_scale_ratio) or d_scale_ratio not in {
        1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0
    }:
        raise ValueError(
            "implicit-xyz D scale ratio must be one of "
            "1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0"
        )
    if not np.isfinite(_IMPLICIT_XYZ_Q_LOG_UPPER) or _IMPLICIT_XYZ_Q_LOG_UPPER not in {
        1.5, 1.8, 2.0
    }:
        raise ValueError("implicit-xyz Q log upper must be one of 1.5, 1.8, 2.0")
    if not np.isfinite(known.action_cost_center_limit) or known.action_cost_center_limit <= 0:
        raise ValueError("action-cost-center-limit must be finite and positive")
    _ACTION_COST_CENTER_LIMIT = float(known.action_cost_center_limit)
    if known.policy_preserving_action_headroom_limit is not None:
        expanded = float(known.policy_preserving_action_headroom_limit)
        if not np.isfinite(expanded) or expanded <= _ACTION_COST_CENTER_LIMIT:
            raise ValueError(
                "policy-preserving headroom limit must exceed the old center limit"
            )
        _ACTION_HEADROOM_LIMIT = expanded
    else:
        _ACTION_HEADROOM_LIMIT = None
    _ACTION_HEADROOM_ADAPTER_ONLY = bool(known.action_headroom_adapter_only)
    if _ACTION_HEADROOM_ADAPTER_ONLY and _ACTION_HEADROOM_LIMIT is None:
        raise ValueError("adapter-only mode requires policy-preserving headroom")
    # --- Protocol budget guard -------------------------------------------
    # (offline_updates, online_steps) is restricted to a fixed grid so a
    # campaign cannot drift out of protocol.  The tuples include the formal
    # REPRODUCTION.md budgets -- offline@10k => (10_000, 0), online@15k =>
    # (0, 15_000) for the RLPD arm and (10_000, 15_000) for the arms
    # bootstrapped from offline -- next to the earlier weighted-matrix points.
    if (args.offline_updates, args.online_steps) not in {
        (0, 0),
        (0, 5_000),
        (0, 7_500),
        (0, 10_000),
        (0, 12_500),
        (0, 14_000),
        (0, 15_000),
        (0, 20_000),
        (10_000, 0),
        (10_000, 15_000),
        (20_000, 0),
        (20_000, 10_000),
        (30_000, 10_000),
    }:
        raise ValueError(
            "protocol is fixed to offline=10k, 20k or 30k/online=0/10k or "
            "online=5k, 7.5k, 10k, 14k, 15k or 20k"
        )
    if args.kmpc_horizon != 5:
        raise ValueError("AWAC-KMPC freezes horizon at H=5")
    ff_path = known.feedforward.expanduser().resolve()
    _FEEDFORWARD = FrozenCirclePhaseFeedforward(ff_path)
    with np.load(args.reference.expanduser().resolve(), allow_pickle=False) as archive:
        _XREF = np.asarray(archive["xref"], dtype=np.float32)
    # The no-feedforward arm still needs a learnable physical action box;
    # the supplied zero table makes its nominal action exactly zero.
    _RESIDUAL_LIMIT = 0.5
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if _IMPLICIT_XYZ_NO_XREF:
        with np.load(args.dataset.expanduser().resolve(), allow_pickle=False) as archive:
            observations = np.asarray(archive["observation"], dtype=np.float32)
        if observations.ndim != 2 or observations.shape[1] < 46:
            raise ValueError("implicit-xyz requires the canonical 46-D time dataset")
        xyz = observations[:, np.asarray(XYZ_INDICES, dtype=np.int64)]
        robust_span = np.quantile(xyz, 0.99, axis=0) - np.quantile(
            xyz, 0.01, axis=0
        )
        # One quarter of the robust offline workspace span, bounded to a
        # physically meaningful 5--30 mm correction per coordinate.
        canonical_d_scale = np.clip(
            0.25 * robust_span, 0.005, 0.030
        ).astype(np.float32)
        _IMPLICIT_XYZ_D_SCALE_M = canonical_d_scale * d_scale_ratio
        sanity_indices = np.linspace(
            0, observations.shape[0] - 1, 32, dtype=np.int64
        )
        _IMPLICIT_XYZ_SANITY_OBSERVATIONS = observations[sanity_indices, :46]
        _IMPLICIT_XYZ_OUTPUT = output
    os.environ[FEEDFORWARD_ENV] = str(ff_path)
    os.environ[RESIDUAL_LIMIT_ENV] = str(_RESIDUAL_LIMIT if _RESIDUAL_LIMIT else 0.5)
    os.environ["ACMPC_MANISOFT_CIRCLE_REFERENCE"] = str(args.reference.expanduser().resolve())
    os.environ["ACMPC_MANISOFT_SCENARIO"] = str(args.scenario.expanduser().resolve())
    os.environ["ACMPC_MANISOFT_KOOPMAN"] = str(args.koopman.expanduser().resolve())
    # Preserve the caller-selected reward for strict offline/online causal
    # probes.  Earlier formal matrix launches hard-coded dense_joint here,
    # which silently changed the reward when bootstrapping a hybrid source.
    os.environ["ACMPC_MANISOFT_REWARD_MODE"] = str(args.reward_mode)
    os.environ["ACMPC_MANISOFT_SPARSE_REWARD_WEIGHT"] = str(args.sparse_reward_weight)
    os.environ["ACMPC_MANISOFT_DENSE_REWARD_WEIGHT"] = str(args.dense_reward_weight)
    os.environ["ACMPC_MANISOFT_DENSE_REWARD_SCALE_M"] = str(args.dense_reward_scale_m)
    # Install the circle-specific hooks onto the shared O2O loop/learner:
    # residual action limit & task spec, actor factory, circle dataset,
    # frozen Koopman wrapper, and the time-residual env adapter.
    config_module.o2o_action_limit = _action_limit
    learner_module.o2o_action_limit = _action_limit
    learner_module.build_actor = _build_actor
    learner_module.get_task_spec = _learner_task_spec
    circle_env_module.ABSOLUTE_ACTION_LIMIT = 0.5
    training.ABSOLUTE_ACTION_LIMIT = 0.5
    training.ManiSoftCircleOfflineDataset = ManiSoftCircleTimeResidualDataset
    training.FrozenManiSoftHistoryKoopman = FrozenManiSoftTimeKoopman
    training.make_manisoft_circle_o2o_adapter = make_manisoft_circle_time_residual_adapter
    koopman_payload = torch.load(args.koopman, map_location="cpu", weights_only=False)
    history_steps = int(koopman_payload.get("architecture", {}).get("history_steps", 0))
    training.COLLECTOR_OBSERVATION_DIM = 46 + history_steps * (45 + 18)
    # Original AWAC uses one conceptual replay buffer; the KMPC arm uses
    # RLPD symmetric sampling with ten LayerNorm critics.
    # Only the official AWAC baseline keeps its original single replay.
    # AWAC-raw/lift are representation ablations of AWAC-KMPC and therefore
    # retain its RLPD-style symmetric offline/online sampling online.
    training.mixed_batch = (
        _single_replay_batch if args.method == "AWAC" else base_training.mixed_batch
    )
    if _IMPLICIT_XYZ_NO_XREF:
        training.O2OLearner = _ImplicitXyzSanityLearner
    control_entry.training = training
    training.evaluate = control_entry.evaluate_control_structure
    # Persist an audit record of the resolved protocol/actor options next to
    # the run output (``time_only_protocol.json``, plus an implicit-xyz JSON
    # for no-xref arms) so archived runs can be audited later.
    protocol = {
        "kind": "manisoft_circle_weighted_offline_online_matrix_v1",
        "actor_input": (
            "Koopman lifted body state + tau=t/(T-1)"
            if args.method in {"AWAC-KMPC", "Cal-RLPD-KMPC", "AWAC-lift"}
            else "normalized physical state + tau=t/(T-1)"
        ),
        "xref_in_actor_observation": False,
        "feedforward_in_actor_observation": False,
        "xref_lookup_in_kmpc": not _IMPLICIT_XYZ_NO_XREF,
        "feedforward_lookup_in_kmpc": True,
        "structure": _STRUCTURE if args.method in {"AWAC-KMPC", "Cal-RLPD-KMPC"} else None,
        "reward": "weighted dense_joint: sqrt(0.2*node0_xyz^2+0.2*node1_xyz^2+0.6*tip_xyz^2)",
        "offline_updates": args.offline_updates,
        "online_steps": args.online_steps,
        "offline_evaluation_interval": args.offline_eval_interval,
        "online_evaluation_interval": args.online_eval_interval,
        "checkpoint_save_interval": args.checkpoint_save_interval,
        "kmpc_solver_iterations": args.kmpc_solver_iterations,
        "method": args.method,
        "awac_update": "original AWAC actor/critic update",
        "critic_ensemble": int(config_module.TRAIN_METHOD_SPECS[args.method].critic_ensemble_size),
        "critic_network": config_module.TRAIN_METHOD_SPECS[args.method].network_profile,
        "actor_q_reduction": config_module.TRAIN_METHOD_SPECS[args.method].actor_q_reduction,
        "online_replay": "RLPD symmetric 50/50" if args.method in {"RLPD", "AWAC-KMPC", "Cal-RLPD-KMPC"} else "single conceptual replay buffer",
        "online_critic_utd": int(config_module.TRAIN_METHOD_SPECS[args.method].online_utd),
        "reward_node_weights": [0.2, 0.2, 0.6],
        "action_limit": 0.5,
        "feedforward": _FEEDFORWARD.identity(),
        "no_feedforward": bool(known.no_feedforward),
        "backup_entropy": not bool(known.disable_backup_entropy),
        "online_critic_only_steps": args.online_critic_only_steps,
        "actor_entropy_enabled": not args.disable_actor_entropy,
        "action_cost_center_limit": _ACTION_COST_CENTER_LIMIT,
        "policy_preserving_action_headroom_limit": _ACTION_HEADROOM_LIMIT,
        "action_headroom_adapter_only": _ACTION_HEADROOM_ADAPTER_ONLY,
        "implicit_xyz_no_xref": _IMPLICIT_XYZ_NO_XREF,
        "full_capacity_online_residual": _FULL_CAPACITY_ONLINE_RESIDUAL,
        "full_residual_channels": (
            _FULL_RESIDUAL_CHANNELS
            if _FULL_CAPACITY_ONLINE_RESIDUAL
            else None
        ),
    }
    (output / "time_only_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    if _IMPLICIT_XYZ_NO_XREF:
        implicit_protocol = {
            "kind": "manisoft_circle_no_xref_implicit_xyz_awac_kmpc_v1",
            "actor_input": "[frozen Koopman lifted physical state, tau]",
            "actor_output": "stage-wise delta_q_xyz[5,9], delta_D_xyz[5,9], raw_d_action[5,18]",
            "xyz_indices": list(XYZ_INDICES),
            "velocity_indices": list(VELOCITY_INDICES),
            "d_xyz_scale_rule": "clip(0.25*(dataset xyz P99-P01), 0.005m, 0.030m)",
            "d_xyz_scale_ratio": d_scale_ratio,
            "d_xyz_scale_m": _IMPLICIT_XYZ_D_SCALE_M.tolist(),
            "q_log_lower_bound": -1.5,
            "q_log_upper_bound": _IMPLICIT_XYZ_Q_LOG_UPPER,
            "q_over_base_upper_bound": float(np.exp(_IMPLICIT_XYZ_Q_LOG_UPPER)),
            "q_xyz_base": 1.0,
            "q_velocity_fixed": _IMPLICIT_XYZ_VELOCITY_COST_SCALE,
            "q_action_fixed": 10000.0,
            "d_action_max": _ACTION_COST_CENTER_LIMIT,
            "xref_in_actor": False,
            "xref_in_kmpc": False,
            "nominal_center": "frozen Koopman rollout from current z under horizon u_ff with delta_u=0",
            "latent_cost": "Qz=Cxyz^T Qxyz Cxyz + Cv^T Qv Cv; pz=-Cxyz^T Qxyz Dxyz-Cv^T Qv Dnom_v",
            "frozen_base_full_capacity_residual": _FULL_CAPACITY_ONLINE_RESIDUAL,
            "online_residual_channels": (
                _FULL_RESIDUAL_CHANNELS
                if _FULL_CAPACITY_ONLINE_RESIDUAL
                else None
            ),
            "online_residual_architecture": (
                "78->Linear(256)->GELU->Linear(256)->GELU->Linear(180)"
                if _FULL_CAPACITY_ONLINE_RESIDUAL
                else None
            ),
            "online_residual_decoded_authority": (
                {
                    "delta_D_ratio": 0.20,
                    "delta_q_log": 0.20,
                    "delta_d_action": 0.001,
                }
                if _FULL_CAPACITY_ONLINE_RESIDUAL
                else None
            ),
        }
        (output / "implicit_xyz_protocol.json").write_text(
            json.dumps(implicit_protocol, indent=2, sort_keys=True) + "\n"
        )
    if _FULL_CAPACITY_ONLINE_RESIDUAL:
        # This experiment selects deployment checkpoints solely by the
        # deterministic return requested in its causal protocol.
        training._checkpoint_selection_key = lambda evaluation: (
            -float(evaluation["return_mean"]),
        )
    # Delegate to the shared O2O loop with all hooks patched above.
    training.run(args)


if __name__ == "__main__":
    main()

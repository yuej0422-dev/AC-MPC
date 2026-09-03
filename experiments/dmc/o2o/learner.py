"""Cal-QL/RLPD-style learner with MLP or AC-KMPC stochastic actors."""

from __future__ import annotations

import copy
import hashlib
import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.dmc.o2o.config import O2OConfig
from experiments.dmc.o2o.koopman import FrozenKoopman
from experiments.dmc.tasks.registry import get_task_spec
from experiments.dmc.o2o.networks import (
    FrozenObservationNormalizer,
    ValueNetwork,
    atanh_clipped,
    build_actor,
    build_critic,
)
from experiments.dmc.reward_oracle import official_reward_for_task


RNG_SUBSTREAM_VERSION = "o2o_torch_rng_substreams_v1"
_RNG_SUBSTREAM_NAMES = ("actor_init", "critic_init", "training_sampling")


def _substream_seed(base_seed: int, name: str) -> int:
    """Derive a stable, method-independent Torch seed for one RNG purpose."""

    if name not in _RNG_SUBSTREAM_NAMES:
        raise ValueError(f"Unknown RNG substream {name!r}")
    payload = f"{RNG_SUBSTREAM_VERSION}:{int(base_seed)}:{name}".encode("utf-8")
    # Torch accepts a signed-64-bit seed.  A cryptographic derivation avoids
    # accidental overlap without relying on Python's process-randomized hash.
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63)


@contextmanager
def _cpu_initialization_stream(seed: int) -> Iterator[None]:
    """Temporarily select a CPU initialization stream without global leakage.

    Modules are constructed on CPU and moved to the requested learner device
    afterwards, so only the CPU default generator is involved in initialization.
    Restoring it here means actor architecture cannot perturb critic weights or
    the caller's global Torch RNG state.
    """

    previous = torch.random.get_rng_state()
    try:
        torch.random.default_generator.manual_seed(seed)
        yield
    finally:
        torch.random.set_rng_state(previous)


def _optimizer(parameters: Any, learning_rate: float, clip_norm: float):
    # Adam is deliberately exposed through a tiny helper so checkpoint loading
    # never silently changes its hyperparameters.
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    optimizer._acmpc_clip_norm = float(clip_norm)  # type: ignore[attr-defined]
    return optimizer


def _clip(parameters: Any, optimizer: torch.optim.Optimizer) -> float:
    value = nn.utils.clip_grad_norm_(
        parameters, float(optimizer._acmpc_clip_norm)  # type: ignore[attr-defined]
    )
    return float(value.detach())


def _component_gradient_norm(
    loss: torch.Tensor, parameters: tuple[nn.Parameter, ...], *, retain_graph: bool
) -> float:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    squared = torch.zeros((), device=loss.device)
    for gradient in gradients:
        if gradient is not None:
            squared = squared + gradient.detach().square().sum()
    return float(squared.sqrt().cpu())


def _finite_subset_statistics(
    prefix: str, values: torch.Tensor, mask: torch.Tensor
) -> dict[str, float]:
    subset = values[mask]
    if subset.numel() == 0:
        return {
            f"{prefix}_count": 0.0,
            f"{prefix}_mean": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_p90": 0.0,
            f"{prefix}_p95": 0.0,
            f"{prefix}_p99": 0.0,
            f"{prefix}_max": 0.0,
        }
    return {
        f"{prefix}_count": float(subset.numel()),
        f"{prefix}_mean": float(subset.mean()),
        f"{prefix}_p50": float(torch.quantile(subset, 0.50)),
        f"{prefix}_p90": float(torch.quantile(subset, 0.90)),
        f"{prefix}_p95": float(torch.quantile(subset, 0.95)),
        f"{prefix}_p99": float(torch.quantile(subset, 0.99)),
        f"{prefix}_max": float(subset.max()),
    }


def _optimizer_to(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    """Move restored Adam moments to the learner device.

    Checkpoints are deliberately loaded through CPU for portability.  Module
    ``load_state_dict`` handles the model device, whereas optimizer state needs
    this explicit migration before a CUDA resume can take its next step.
    """

    if any(
        parameter.device != device
        for group in optimizer.param_groups
        for parameter in group["params"]
    ):
        raise ValueError("Optimizer parameters do not live on the learner device")
    for group in optimizer.param_groups:
        scalar_step_on_parameter = bool(
            group.get("capturable", False) or group.get("fused", False)
        )
        for parameter in group["params"]:
            for key, value in optimizer.state.get(parameter, {}).items():
                if not isinstance(value, torch.Tensor):
                    continue
                # Adam deliberately keeps its scalar step counter on CPU when
                # it is neither fused nor capturable.  Moving that tensor to
                # CUDA would undo PyTorch's checkpoint-loading policy.
                target_device = (
                    parameter.device
                    if key != "step" or scalar_step_on_parameter
                    else torch.device("cpu")
                )
                optimizer.state[parameter][key] = value.to(device=target_device)


@contextmanager
def _frozen_parameters(module: nn.Module) -> Iterator[None]:
    """Freeze module weights while preserving gradients through its inputs."""

    parameters = tuple(module.parameters())
    requires_grad = tuple(parameter.requires_grad for parameter in parameters)
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, enabled in zip(parameters, requires_grad, strict=True):
            parameter.requires_grad_(enabled)


@dataclass
class TensorBatch:
    observation: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    discount: torch.Tensor
    next_observation: torch.Tensor
    mc_return: torch.Tensor
    offline_mask: torch.Tensor
    previous_action: torch.Tensor | None = None
    next_previous_action: torch.Tensor | None = None
    behavior_clone_mask: torch.Tensor | None = None

    @classmethod
    def from_numpy(cls, batch: dict[str, np.ndarray], device: torch.device):
        values = {}
        for key in cls.__dataclass_fields__:
            if key in batch:
                values[key] = torch.as_tensor(batch[key], dtype=torch.float32, device=device)
        return cls(**values)

    def slice(self, index: slice) -> "TensorBatch":
        values = {}
        for key in self.__dataclass_fields__:
            value = getattr(self, key)
            values[key] = None if value is None else value[index]
        return TensorBatch(**values)


@dataclass(frozen=True)
class CriticProposalCache:
    """Frozen actor proposals for one fused UTD batch.

    Only states, actions, and action log-probabilities are cached.  Target-Q
    and current-Q values must be evaluated again after every critic update.
    """

    state: torch.Tensor
    next_state: torch.Tensor
    target_next_action: torch.Tensor
    target_next_log_prob: torch.Tensor
    cql_current_actions: torch.Tensor | None = None
    cql_current_log_prob: torch.Tensor | None = None
    cql_next_actions: torch.Tensor | None = None
    cql_next_log_prob: torch.Tensor | None = None

    def slice(self, index: slice) -> "CriticProposalCache":
        def slice_samples(value: torch.Tensor | None) -> torch.Tensor | None:
            return None if value is None else value[:, index]

        target_action = (
            self.target_next_action[:, index]
            if self.target_next_action.ndim == 3
            else self.target_next_action[index]
        )
        target_log_prob = (
            self.target_next_log_prob[:, index]
            if self.target_next_log_prob.ndim == 2
            else self.target_next_log_prob[index]
        )

        return CriticProposalCache(
            state=self.state[index],
            next_state=self.next_state[index],
            target_next_action=target_action,
            target_next_log_prob=target_log_prob,
            cql_current_actions=slice_samples(self.cql_current_actions),
            cql_current_log_prob=slice_samples(self.cql_current_log_prob),
            cql_next_actions=slice_samples(self.cql_next_actions),
            cql_next_log_prob=slice_samples(self.cql_next_log_prob),
        )


class O2OLearner:
    """Shared off-policy Q core; only the actor and optional MPVE term differ."""

    def __init__(
        self,
        config: O2OConfig,
        koopman: FrozenKoopman | None,
        device: torch.device,
        *,
        observation_normalizer: FrozenObservationNormalizer | None = None,
    ) -> None:
        config.validate()
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            # Keep optimizer-resume validation and the private CUDA generator
            # on the same explicit device as parameters created via ``.to``.
            device = torch.device("cuda", torch.cuda.current_device())
        self.config = config
        self.device = device
        if config.requires_koopman:
            if koopman is None:
                raise ValueError(f"{config.method} requires a Koopman model")
            if observation_normalizer is not None:
                raise ValueError("Structured methods use Koopman normalization only")
            self.koopman: FrozenKoopman | None = koopman.to(device).eval()
            self.observation_normalizer: FrozenObservationNormalizer | None = None
            self.state_dim = koopman.lifted_dim
            self.action_dim = koopman.action_dim
        else:
            if koopman is not None:
                raise ValueError(f"{config.method} is raw-only and forbids Koopman")
            if observation_normalizer is None:
                raise ValueError("Raw methods require an offline-dataset normalizer")
            self.koopman = None
            self.observation_normalizer = observation_normalizer.to(device).eval()
            self.state_dim = observation_normalizer.observation_dim
            self.action_dim = get_task_spec(config.task).action_dim
        self.rng_substream_seeds = {
            name: _substream_seed(config.seed, name)
            for name in _RNG_SUBSTREAM_NAMES
        }
        generator_device = device if device.type == "cuda" else torch.device("cpu")
        self.training_generator = torch.Generator(device=generator_device)
        self.training_generator.manual_seed(
            self.rng_substream_seeds["training_sampling"]
        )

        # Actor and critic initialization use independent, method-invariant
        # streams.  In particular, changing from MLP to KMPC cannot change the
        # shared critic ensemble merely by consuming a different number of
        # default-generator draws while constructing the actor.
        with _cpu_initialization_stream(self.rng_substream_seeds["actor_init"]):
            actor_options: dict[str, Any] = {
                "actor_kind": config.method_spec.actor,
                "network_profile": config.network_profile,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "hidden_dim": config.hidden_dim,
                "controller_hidden_dim": config.controller_hidden_dim,
                "kmpc_horizon": config.kmpc_horizon,
                "kmpc_solver_iterations": config.kmpc_solver_iterations,
                "controller_hidden_layers": config.controller_hidden_layers,
            }
            # ManiSoft's extended config adds physical-action and exploration
            # controls. Keep this learner source compatible with legacy DMC
            # configs while forwarding every field when the task-local config
            # exposes it (including monkeypatched diagnostic actors).
            if hasattr(config, "kmpc_log_std_init"):
                actor_options.update(
                    # ManiSoft baseline actors operate directly in the
                    # physical residual coordinate.  The environment and
                    # replay buffer use the agreed ±0.5 residual box.
                    action_limit=(0.5 if config.task == "manisoft_circle" else 1.0),
                    kmpc_delta_u_weight=config.kmpc_delta_u_weight,
                    kmpc_delta_u_deadband=config.kmpc_delta_u_deadband,
                    kmpc_delta_u_limit=config.kmpc_delta_u_limit,
                    kmpc_log_std_init=config.kmpc_log_std_init,
                    kmpc_log_std_max=config.kmpc_log_std_max,
                )
            self.actor = build_actor(
                config.method,
                self.koopman,
                **actor_options,
            ).to(device)
        # Opt-in online ablation: the real environment and Bellman target can
        # remain rate-feasible while the reparameterized actor-loss sample is
        # evaluated before the engineering adjacent-action projection.
        self.actor_update_unbounded_action_rate = False
        with _cpu_initialization_stream(self.rng_substream_seeds["critic_init"]):
            self.critic = build_critic(
                network_profile=config.network_profile,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                ensemble_size=config.critic_ensemble_size,
                hidden_dim=config.hidden_dim,
                hidden_layers=config.critic_hidden_layers,
            ).to(device)
        self.target_critic = copy.deepcopy(self.critic).to(device).eval()
        for parameter in self.target_critic.parameters():
            parameter.requires_grad_(False)
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(config.initial_temperature), device=device)
        )
        actor_parameter_provider = getattr(
            self.actor, "online_residual_parameters", None
        )
        actor_parameters = tuple(
            actor_parameter_provider()
            if callable(actor_parameter_provider)
            else (
                parameter
                for parameter in self.actor.parameters()
                if parameter.requires_grad
            )
        )
        if not actor_parameters:
            raise ValueError("Actor optimizer has no trainable parameters")
        self.actor_optimizer = _optimizer(
            actor_parameters,
            config.actor_learning_rate,
            config.gradient_clip_norm,
        )
        self.critic_optimizer = _optimizer(
            self.critic.parameters(),
            config.critic_learning_rate,
            config.gradient_clip_norm,
        )
        self.temperature_optimizer = _optimizer(
            [self.log_temperature],
            config.temperature_learning_rate,
            config.gradient_clip_norm,
        )
        self.value: ValueNetwork | None = None
        self.value_optimizer: torch.optim.Optimizer | None = None
        if config.learner_family == "iql":
            with _cpu_initialization_stream(
                self.rng_substream_seeds["critic_init"] + 1
            ):
                self.value = ValueNetwork(self.state_dim, config.hidden_dim).to(device)
            self.value_optimizer = _optimizer(
                self.value.parameters(),
                config.critic_learning_rate,
                config.gradient_clip_norm,
            )
        self.gradient_updates = 0
        # Count logical learner calls separately from critic gradient updates.
        # With UTD>1, gradient_updates advances by UTD each call; scheduling
        # actor updates from it would make intervals dividing UTD ineffective.
        self.logical_updates = 0
        self.actor_updates = 0
        self.awac_selectivity_mode: str | None = None
        self.awac_reference_kl_weight = 0.0
        self.awac_reference_actor: nn.Module | None = None
        self.awac_reference_probe_observations: torch.Tensor | None = None

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def representation_identity(self) -> dict[str, Any]:
        if self.config.requires_koopman:
            assert self.koopman is not None
            return {
                "kind": "koopman_lifted_state_v1",
                "state_dim": self.koopman.state_dim,
                "lift_dim": self.koopman.lift_dim,
                "input_dim": self.koopman.lifted_dim,
                "koopman_sha256": self.koopman.sha256,
            }
        assert self.observation_normalizer is not None
        return {
            "kind": "normalized_raw_observation_v1",
            "input_dim": self.observation_normalizer.observation_dim,
            "normalizer": self.observation_normalizer.identity(),
        }

    def _encode(self, observation: torch.Tensor) -> torch.Tensor:
        if self.config.requires_koopman:
            assert self.koopman is not None
            return self.koopman.lift(observation)
        assert self.observation_normalizer is not None
        return self.observation_normalizer(observation)

    @torch.no_grad()
    def act(self, observation: np.ndarray, deterministic: bool) -> np.ndarray:
        tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        state = self._encode(tensor)
        action, _, _ = self.actor.sample(
            state,
            deterministic=deterministic,
            generator=self.training_generator,
        )
        return action.cpu().numpy()

    def _sample_actor_with_context(
        self,
        state: torch.Tensor,
        *,
        previous_action: torch.Tensor | None = None,
        **kwargs: Any,
    ):
        """Sample an actor, optionally passing hidden action-rate context."""
        if previous_action is None:
            return self.actor.sample(state, **kwargs)
        return self.actor.sample(state, previous_action=previous_action, **kwargs)

    @torch.no_grad()
    def _prepare_critic_cache(
        self,
        batch: TensorBatch,
        *,
        phase: Literal["offline", "online"] = "offline",
    ) -> CriticProposalCache:
        """Sample fixed proposals once while the actor is fixed across UTD."""

        state = self._encode(batch.observation)
        next_state = self._encode(batch.next_observation)
        if self.config.uses_calql_in_phase(phase):
            sample_count = self.config.cql_actions
            current_actions, current_log_prob, _ = self._sample_actor_with_context(
                state,
                previous_action=batch.previous_action,
                samples=sample_count,
                generator=self.training_generator,
            )
            # Preserve independent target-vs-CQL proposal samples while sharing
            # the expensive KMPC plan.  The task-matched ExORL-DMC profile uses
            # one target action; optional Cal-QL max backup would use K.
            target_sample_count = (
                sample_count
                if self.config.uses_calql_max_target_backup_in_phase(phase)
                else 1
            )
            next_actions, next_log_prob, _ = self._sample_actor_with_context(
                next_state,
                previous_action=batch.next_previous_action,
                samples=target_sample_count + sample_count,
                generator=self.training_generator,
            )
            # ``actor.sample(samples=1)`` intentionally returns the ordinary
            # [B,A]/[B] shape.  Canonicalize it so cache slicing is uniform.
            if sample_count == 1:
                current_actions = current_actions.unsqueeze(0)
                current_log_prob = current_log_prob.unsqueeze(0)
            target_next_actions = next_actions[:target_sample_count]
            target_next_log_prob = next_log_prob[:target_sample_count]
            if target_sample_count == 1:
                target_next_actions = target_next_actions[0]
                target_next_log_prob = target_next_log_prob[0]
            return CriticProposalCache(
                state=state.detach(),
                next_state=next_state.detach(),
                target_next_action=target_next_actions.detach(),
                target_next_log_prob=target_next_log_prob.detach(),
                cql_current_actions=current_actions.detach(),
                cql_current_log_prob=current_log_prob.detach(),
                cql_next_actions=next_actions[target_sample_count:].detach(),
                cql_next_log_prob=next_log_prob[target_sample_count:].detach(),
            )

        next_action, next_log_prob, _ = self._sample_actor_with_context(
            next_state,
            previous_action=batch.next_previous_action,
            generator=self.training_generator,
        )
        return CriticProposalCache(
            state=state.detach(),
            next_state=next_state.detach(),
            target_next_action=next_action.detach(),
            target_next_log_prob=next_log_prob.detach(),
        )

    def _reduce_critic_objective(self, per_head_per_row: torch.Tensor) -> torch.Tensor:
        """Apply the immutable method-specific reduction over Q heads.

        Cal-QL/ExORL optimizes Q1 and Q2 losses separately and adds them.  The
        vectorized equivalent is a row mean within each head followed by a head
        sum.  RLPD instead averages its ensemble objective.
        """

        if (
            per_head_per_row.ndim != 2
            or per_head_per_row.shape[0] != self.config.critic_ensemble_size
        ):
            raise ValueError("Critic objective must have shape [Q_heads, batch]")
        per_head = per_head_per_row.mean(dim=1)
        if self.config.critic_head_reduction == "sum":
            return per_head.sum()
        if self.config.critic_head_reduction == "mean":
            return per_head.mean()
        raise RuntimeError("Unknown critic-head reduction")

    def _reduce_actor_q(self, q_heads: torch.Tensor) -> torch.Tensor:
        if q_heads.ndim != 2:
            raise ValueError("Actor Q values must have shape [Q_heads, batch]")
        if self.config.actor_q_reduction == "min":
            return q_heads.amin(dim=0)
        if self.config.actor_q_reduction == "mean":
            return q_heads.mean(dim=0)
        raise RuntimeError("Unknown actor-Q reduction")

    def _temperature_loss(self, log_prob: torch.Tensor) -> torch.Tensor:
        target_entropy = float(self.config.target_entropy)
        if self.config.temperature_objective == "calql_log_alpha":
            # Official Cal-QL/JaxCQL SAC objective.  Only log(alpha) receives
            # this gradient; policy gradients come from the actor objective.
            return -(
                self.log_temperature
                * (log_prob + target_entropy).detach()
            ).mean()
        if self.config.temperature_objective == "rlpd":
            # Match RLPD's Temperature module: optimize alpha=exp(log_alpha)
            # against entropy-target_entropy.  This differs from the log-alpha
            # loss whenever alpha is not one.
            entropy = (-log_prob).detach()
            return (
                self.temperature * (entropy - target_entropy)
            ).mean()
        raise RuntimeError("Unknown temperature objective")

    def _target_q(
        self,
        batch: TensorBatch,
        cache: CriticProposalCache,
        *,
        phase: Literal["offline", "online"] = "offline",
    ) -> torch.Tensor:
        with torch.no_grad():
            # Target critic parameters evolve after every REDQ update, so only
            # the fixed actor proposal is reused here; Q is always recomputed.
            max_target_backup = (
                self.config.uses_calql_max_target_backup_in_phase(phase)
            )
            if max_target_backup:
                if (
                    cache.target_next_action.ndim != 3
                    or cache.target_next_log_prob.ndim != 2
                ):
                    raise RuntimeError("Cal-QL max target cache has invalid shapes")
                samples, batch_size, action_dim = cache.target_next_action.shape
                expanded_state = cache.next_state.unsqueeze(0).expand(
                    samples, -1, -1
                )
                candidate_q = self._minimum_target_q(
                    expanded_state.reshape(samples * batch_size, -1),
                    cache.target_next_action.reshape(
                        samples * batch_size, action_dim
                    ),
                ).reshape(samples, batch_size)
                next_q, choice = candidate_q.max(dim=0)
                if self.config.backup_entropy:
                    chosen_log_prob = cache.target_next_log_prob.gather(
                        0, choice.unsqueeze(0)
                    ).squeeze(0)
                    next_q = (
                        next_q - self.temperature.detach() * chosen_log_prob
                    )
            else:
                if (
                    cache.target_next_action.ndim != 2
                    or cache.target_next_log_prob.ndim != 1
                ):
                    raise RuntimeError("Single-action target cache has invalid shapes")
                next_q = self._minimum_target_q(
                    cache.next_state, cache.target_next_action
                )
                if self.config.backup_entropy:
                    next_q = (
                        next_q
                        - self.temperature.detach() * cache.target_next_log_prob
                    )
            return batch.reward + self.config.discount * batch.discount * next_q

    def _minimum_target_q(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """REDQ target: minimum over one freshly sampled critic subset."""

        target_heads = self.target_critic(state, action)
        if self.config.target_critic_subset < target_heads.shape[0]:
            choice = torch.randperm(
                target_heads.shape[0],
                device=target_heads.device,
                generator=self.training_generator,
            )[: self.config.target_critic_subset]
            target_heads = target_heads[choice]
        return target_heads.amin(dim=0)

    def _cql_calibrated_penalty(
        self,
        batch: TensorBatch,
        cache: CriticProposalCache,
        data_q: torch.Tensor,
        *,
        phase: Literal["offline", "online"] = "offline",
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Cal-RLPD uses calibrated conservative regularization only for its
        # offline pretraining.  Its online phase is pure RLPD, rather than the
        # previous bespoke penalty on only the offline half of a mixed batch.
        if not self.config.uses_calql_in_phase(phase):
            zero = data_q.sum() * 0.0
            return zero, {"cql_penalty": 0.0, "calibration_bound_rate": 0.0}
        calibrated = (
            torch.ones_like(batch.offline_mask, dtype=torch.bool)
            if phase == "online"
            else batch.offline_mask > 0.5
        )
        if not torch.any(calibrated):
            zero = data_q.sum() * 0.0
            return zero, {"cql_penalty": 0.0, "calibration_bound_rate": 0.0}
        calibrated_state = cache.state[calibrated]
        batch_size = calibrated_state.shape[0]
        sample_count = self.config.cql_actions
        expanded_state = calibrated_state.unsqueeze(0).expand(sample_count, -1, -1)

        physical_box_sampler = getattr(
            self.actor, "sample_uniform_actions", None
        )
        if callable(physical_box_sampler):
            # Structured physical-residual actors may have state-dependent
            # feasible boxes.  CQL proposals and their density must use the
            # same action coordinate/measure as policy and replay actions.
            with torch.no_grad():
                random_actions, random_log_density = physical_box_sampler(
                    calibrated_state,
                    samples=sample_count,
                    generator=self.training_generator,
                )
            expected_action_shape = (
                sample_count,
                batch_size,
                self.action_dim,
            )
            expected_density_shape = (sample_count, batch_size)
            if random_actions.shape != expected_action_shape:
                raise RuntimeError(
                    "Physical-box CQL sampler returned actions with shape "
                    f"{tuple(random_actions.shape)}, expected {expected_action_shape}"
                )
            if random_log_density.shape != expected_density_shape:
                raise RuntimeError(
                    "Physical-box CQL sampler returned log density with shape "
                    f"{tuple(random_log_density.shape)}, expected "
                    f"{expected_density_shape}"
                )
            uses_physical_box = True
        else:
            random_actions = torch.empty(
                sample_count, batch_size, self.action_dim, device=self.device
            ).uniform_(-1.0, 1.0, generator=self.training_generator)
            random_log_density = torch.full(
                (sample_count, batch_size),
                -self.action_dim * math.log(2.0),
                dtype=random_actions.dtype,
                device=random_actions.device,
            )
            uses_physical_box = False
        proposals = (
            cache.cql_current_actions,
            cache.cql_current_log_prob,
            cache.cql_next_actions,
            cache.cql_next_log_prob,
        )
        if any(value is None for value in proposals):
            raise RuntimeError("Cal-QL critic cache is missing policy proposals")
        current_actions = cache.cql_current_actions[:, calibrated]
        current_log_prob = cache.cql_current_log_prob[:, calibrated]
        next_actions = cache.cql_next_actions[:, calibrated]
        next_log_prob = cache.cql_next_log_prob[:, calibrated]

        def evaluate(actions: torch.Tensor) -> torch.Tensor:
            flat_state = expanded_state.reshape(-1, cache.state.shape[-1])
            flat_action = actions.reshape(-1, actions.shape[-1])
            values = self.critic(flat_state, flat_action)
            return values.reshape(values.shape[0], sample_count, batch_size)

        q_random = evaluate(random_actions)
        q_current = evaluate(current_actions)
        q_next = evaluate(next_actions)
        lower_bound = batch.mc_return[calibrated].view(1, 1, batch_size)
        bound_rate = 0.5 * (
            (q_current < lower_bound).float().mean()
            + (q_next < lower_bound).float().mean()
        )
        # Cal-QL modifies only the OOD/policy push-down side of CQL.  Q(data)
        # and the Bellman target are never clamped.
        q_current = torch.maximum(q_current, lower_bound)
        q_next = torch.maximum(q_next, lower_bound)
        candidates = torch.cat(
            (
                q_random - random_log_density.unsqueeze(0),
                q_current - current_log_prob.unsqueeze(0),
                q_next - next_log_prob.unsqueeze(0),
            ),
            dim=1,
        )
        ood = torch.logsumexp(
            candidates / self.config.cql_temperature, dim=1
        ) * self.config.cql_temperature
        penalty = self._reduce_critic_objective(ood - data_q[:, calibrated])
        return penalty, {
            "cql_penalty": float(penalty.detach()),
            "calibration_bound_rate": float(bound_rate.detach()),
            "cql_random_log_density_mean": float(
                random_log_density.detach().mean()
            ),
            "cql_random_action_abs_max": float(
                random_actions.detach().abs().max()
            ),
            "cql_uses_physical_box": float(uses_physical_box),
        }

    def _mpve_target(
        self,
        batch: TensorBatch,
        real_target: torch.Tensor,
        *,
        next_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Total H target: one real transition followed by H-1 model steps."""

        if not self.config.uses_mpve:
            return real_target.detach()
        if self.koopman is None:
            raise RuntimeError("MPVE requires a Koopman model")
        with torch.no_grad():
            current = (
                self.koopman.lift(batch.next_observation)
                if next_state is None
                else next_state
            )
            total = batch.reward.clone()
            continuation = self.config.discount * batch.discount
            # The real transition above is step one.  Add H-1 imagined rewards.
            for _ in range(1, self.config.mpve_total_horizon):
                action, log_prob, _ = self.actor.sample(
                    current,
                    generator=self.training_generator,
                )
                following = self.koopman.step(current, action)
                normalized_following = self.koopman.reconstruct_normalized(following)
                reward_oracle = official_reward_for_task(self.config.task)
                reward = reward_oracle(
                    self.koopman.denormalize(normalized_following), action
                )
                total = total + continuation * (
                    reward - self.temperature.detach() * log_prob
                )
                continuation = continuation * self.config.discount
                current = following
            terminal_action, terminal_log_prob, _ = self.actor.sample(
                current,
                generator=self.training_generator,
            )
            terminal_q = self._minimum_target_q(current, terminal_action)
            terminal_q = terminal_q - self.temperature.detach() * terminal_log_prob
            return (total + continuation * terminal_q).detach()

    def update_critic(
        self,
        batch: TensorBatch,
        *,
        apply_mpve: bool,
        cache: CriticProposalCache | None = None,
        phase: Literal["offline", "online"] = "offline",
    ) -> dict[str, float]:
        if apply_mpve and not self.config.uses_mpve:
            raise ValueError("MPVE auxiliary requested for a non-MPVE method")
        if cache is None:
            cache = self._prepare_critic_cache(batch, phase=phase)
        state = cache.state
        target = self._target_q(batch, cache, phase=phase)
        q = self.critic(state, batch.action)
        bellman_loss = self._reduce_critic_objective(
            (q - target.unsqueeze(0)).square()
        )
        cql_penalty, cql_metrics = self._cql_calibrated_penalty(
            batch, cache, q, phase=phase
        )
        loss = bellman_loss + self.config.cql_weight * cql_penalty
        mpve_loss = q.sum() * 0.0
        mpve_target_mean = 0.0
        if apply_mpve:
            model_target = self._mpve_target(
                batch,
                target,
                next_state=cache.next_state,
            )
            mpve_loss = self._reduce_critic_objective(
                (q - model_target.unsqueeze(0)).square()
            )
            loss = loss + self.config.mpve_loss_weight * mpve_loss
            mpve_target_mean = float(model_target.mean())
        self.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = _clip(self.critic.parameters(), self.critic_optimizer)
        self.critic_optimizer.step()
        with torch.no_grad():
            for target_parameter, parameter in zip(
                self.target_critic.parameters(), self.critic.parameters(), strict=True
            ):
                target_parameter.lerp_(parameter, self.config.target_tau)
        self.gradient_updates += 1
        offline_mask = batch.offline_mask.detach().reshape(-1) > 0.5
        online_mask = ~offline_mask

        def subset_mean(value: torch.Tensor, mask: torch.Tensor) -> float:
            if not bool(mask.any()):
                return float("nan")
            if value.ndim == 2:
                return float(value.detach()[:, mask].mean())
            return float(value.detach()[mask].mean())

        return {
            "critic_loss": float(loss.detach()),
            "bellman_loss": float(bellman_loss.detach()),
            "critic_grad_norm": grad_norm,
            "reward_mean": float(batch.reward.detach().mean()),
            "q_mean": float(q.detach().mean()),
            "target_q_mean": float(target.mean()),
            "offline_q_mean": subset_mean(q, offline_mask),
            "online_q_mean": subset_mean(q, online_mask),
            "offline_target_q_mean": subset_mean(target, offline_mask),
            "online_target_q_mean": subset_mean(target, online_mask),
            "target_log_pi_mean": float(
                cache.target_next_log_prob.detach().mean()
            ),
            "target_alpha_log_pi_mean": float(
                (
                    self.temperature.detach()
                    * cache.target_next_log_prob.detach()
                ).mean()
            ),
            "target_entropy_backup_mean": float(
                (
                    -self.temperature.detach()
                    * cache.target_next_log_prob.detach()
                ).mean()
                if self.config.backup_entropy
                else 0.0
            ),
            "backup_entropy_applied": float(self.config.backup_entropy),
            "mpve_applied": float(apply_mpve),
            "mpve_loss": float(mpve_loss.detach()),
            "mpve_target_mean": mpve_target_mean,
            **cql_metrics,
        }

    def update_actor_and_temperature(
        self,
        batch: TensorBatch,
        *,
        state: torch.Tensor | None = None,
    ) -> dict[str, float]:
        state = (
            self._encode(batch.observation).detach()
            if state is None
            else state
        )
        if bool(getattr(self.actor, "frozen_cost_map", False)):
            with torch.no_grad():
                action, log_prob, _ = self._sample_actor_with_context(
                    state,
                    previous_action=batch.previous_action,
                    generator=self.training_generator,
                )
                q = self._reduce_actor_q(self.critic(state, action))
            return {
                "actor_loss": 0.0,
                "actor_rl_loss": float((-q).mean()),
                "q_cost_anchor": 0.0,
                "p_cost_anchor": 0.0,
                "q_cost_anchor_term": 0.0,
                "p_cost_anchor_term": 0.0,
                "actor_update_applied": 0.0,
                "actor_frozen": 1.0,
                "actor_grad_norm": 0.0,
                "entropy": float((-log_prob).mean()),
                "actor_log_pi_mean": float(log_prob.mean()),
                "actor_alpha_log_pi_mean": float(
                    (self.temperature.detach() * log_prob).mean()
                ),
                "temperature": float(self.temperature.detach()),
                "temperature_loss": 0.0,
            }
        rate_setter = getattr(self.actor, "set_max_action_delta", None)
        saved_max_delta = getattr(self.actor, "max_action_delta", None)
        unbounded_actor_sample = bool(
            getattr(self, "actor_update_unbounded_action_rate", False)
            and callable(rate_setter)
        )
        if unbounded_actor_sample:
            rate_setter(None)
        try:
            action, log_prob, _ = self._sample_actor_with_context(
                state,
                previous_action=batch.previous_action,
                generator=self.training_generator,
            )
        finally:
            if unbounded_actor_sample:
                rate_setter(saved_max_delta)
        # The SAC policy needs dQ/da, but the actor step must neither calculate
        # nor retain gradients for critic parameters.
        self.critic_optimizer.zero_grad(set_to_none=True)
        self.actor_optimizer.zero_grad(set_to_none=True)
        with _frozen_parameters(self.critic):
            q = self._reduce_actor_q(self.critic(state, action))
            actor_entropy_enabled = bool(
                getattr(self, "actor_entropy_enabled", True)
            )
            actor_rl_loss = (
                (
                    self.temperature.detach() * log_prob
                    if actor_entropy_enabled
                    else torch.zeros_like(log_prob)
                )
                - q
            ).mean()
            anchor = getattr(self.actor, "cost_map_anchor", None)
            if callable(anchor):
                q_anchor, p_anchor = anchor(state)
            else:
                q_anchor = actor_rl_loss * 0.0
                p_anchor = actor_rl_loss * 0.0
            q_anchor_term = self.config.q_cost_anchor_weight * q_anchor
            p_anchor_term = self.config.p_cost_anchor_weight * p_anchor
            action_trust = actor_rl_loss * 0.0
            action_trust_fn = getattr(self.actor, "action_trust_anchor", None)
            if callable(action_trust_fn):
                action_trust = action_trust_fn(state)
            action_trust_weight = float(
                getattr(
                    self.actor,
                    "action_trust_anchor_weight",
                    self.config.action_trust_anchor_weight,
                )
            )
            action_trust_term = action_trust_weight * action_trust
            behavior_clone_loss = actor_rl_loss * 0.0
            behavior_clone = getattr(self.actor, "behavior_action", None)
            if self.config.offline_behavior_clone_weight > 0.0:
                if not callable(behavior_clone):
                    raise RuntimeError(
                        "offline_behavior_clone_weight requires an actor "
                        "behavior_action(state) implementation"
                    )
                behavior_action = behavior_clone(state)
                if behavior_action.shape != batch.action.shape:
                    raise RuntimeError(
                        "behavior_action and replay action shapes differ: "
                        f"{behavior_action.shape} != {batch.action.shape}"
                    )
                offline = batch.offline_mask.to(
                    dtype=behavior_action.dtype,
                    device=behavior_action.device,
                )
                selected = offline
                if batch.behavior_clone_mask is not None:
                    selected = selected * batch.behavior_clone_mask.to(
                        dtype=behavior_action.dtype,
                        device=behavior_action.device,
                    )
                per_row = (behavior_action - batch.action).square().sum(dim=-1)
                behavior_clone_loss = (
                    (per_row * selected).sum() / selected.sum().clamp_min(1.0)
                )
            behavior_clone_term = (
                self.config.offline_behavior_clone_weight * behavior_clone_loss
            )
            actor_loss = (
                actor_rl_loss
                + q_anchor_term
                + p_anchor_term
                + action_trust_term
                + behavior_clone_term
            )
            anchor_gradient_metrics: dict[str, float] = {}
            if self.config.cost_anchor_gradient_diagnostics:
                parameters = tuple(
                    parameter
                    for parameter in self.actor.parameters()
                    if parameter.requires_grad
                )

                def component_grad_norm(loss: torch.Tensor) -> float:
                    gradients = torch.autograd.grad(
                        loss,
                        parameters,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    squared = sum(
                        gradient.detach().square().sum()
                        for gradient in gradients
                        if gradient is not None
                    )
                    return float(torch.sqrt(squared))

                anchor_gradient_metrics = {
                    "actor_rl_grad_norm_unclipped": component_grad_norm(
                        actor_rl_loss
                    ),
                    "q_anchor_grad_norm_unweighted": component_grad_norm(
                        q_anchor
                    ),
                    "p_anchor_grad_norm_unweighted": component_grad_norm(
                        p_anchor
                    ),
                }
            actor_loss.backward()
        actor_grad_norm = _clip(self.actor.parameters(), self.actor_optimizer)
        self.actor_optimizer.step()

        if actor_entropy_enabled:
            temperature_loss = self._temperature_loss(log_prob)
            self.temperature_optimizer.zero_grad(set_to_none=True)
            temperature_loss.backward()
            self.temperature_optimizer.step()
        else:
            temperature_loss = log_prob.detach().sum() * 0.0
        self.actor_updates += 1
        return {
            "actor_loss": float(actor_loss.detach()),
            "actor_rl_loss": float(actor_rl_loss.detach()),
            "q_cost_anchor": float(q_anchor.detach()),
            "p_cost_anchor": float(p_anchor.detach()),
            "q_cost_anchor_term": float(q_anchor_term.detach()),
            "p_cost_anchor_term": float(p_anchor_term.detach()),
            "action_trust_anchor": float(action_trust.detach()),
            "action_trust_anchor_weight": action_trust_weight,
            "action_trust_anchor_term": float(action_trust_term.detach()),
            "offline_behavior_clone_loss": float(behavior_clone_loss.detach()),
            "offline_behavior_clone_term": float(behavior_clone_term.detach()),
            "offline_behavior_clone_weight": float(
                self.config.offline_behavior_clone_weight
            ),
            "behavior_clone_selected_fraction": float(
                (
                    batch.offline_mask
                    if batch.behavior_clone_mask is None
                    else batch.offline_mask * batch.behavior_clone_mask
                ).detach().mean()
            ),
            "actor_update_applied": 1.0,
            **anchor_gradient_metrics,
            "actor_grad_norm": actor_grad_norm,
            "entropy": float((-log_prob).detach().mean()),
            "actor_log_pi_mean": float(log_prob.detach().mean()),
            "actor_alpha_log_pi_mean": float(
                (self.temperature.detach() * log_prob.detach()).mean()
            ),
            "temperature": float(self.temperature.detach()),
            "temperature_loss": float(temperature_loss.detach()),
            "actor_entropy_enabled": float(actor_entropy_enabled),
            "actor_update_unbounded_action_rate": float(unbounded_actor_sample),
        }

    def _data_action_log_prob(
        self, state: torch.Tensor, action: torch.Tensor,
        previous_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Log probability of replay actions under a tanh-Gaussian actor."""

        actor_log_prob = getattr(self.actor, "data_action_log_prob", None)
        if callable(actor_log_prob):
            if previous_action is None:
                return actor_log_prob(state, action)
            return actor_log_prob(state, action, previous_action=previous_action)

        location, log_std = (
            self.actor.distribution(state, previous_action)
            if previous_action is not None
            else self.actor.distribution(state)
        )
        pre_tanh = atanh_clipped(action)
        normal_log_prob = -0.5 * (
            ((pre_tanh - location) / log_std.exp()).square()
            + 2.0 * log_std
            + math.log(2.0 * math.pi)
        )
        correction = 2.0 * (
            math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh)
        )
        return (normal_log_prob - correction).sum(dim=-1)

    @staticmethod
    def _distribution_parameters(
        actor: nn.Module,
        state: torch.Tensor,
        previous_action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        result = (
            actor.distribution(state, previous_action)
            if previous_action is not None
            else actor.distribution(state)
        )
        if not isinstance(result, tuple) or len(result) < 2:
            raise TypeError("Actor distribution must return location and log_std")
        return result[0], result[1]

    def configure_awac_selectivity(
        self,
        *,
        mode: str,
        reference_kl_weight: float,
        probe_observations: np.ndarray,
    ) -> None:
        """Configure a continuation-only selective AWAC actor objective."""

        allowed = {"all", "positive", "positive_top50", "positive_klref"}
        if self.config.learner_family != "awac":
            raise ValueError("AWAC selectivity requires an AWAC learner")
        if mode not in allowed:
            raise ValueError(f"Unknown AWAC selectivity mode {mode!r}")
        if not math.isfinite(reference_kl_weight) or reference_kl_weight < 0:
            raise ValueError("AWAC reference KL weight must be finite and non-negative")
        if mode == "positive_klref":
            if reference_kl_weight <= 0:
                raise ValueError("positive_klref requires a positive KL weight")
        elif reference_kl_weight != 0:
            raise ValueError("Reference KL weight is only valid for positive_klref")
        probe = np.asarray(probe_observations, dtype=np.float32)
        if probe.ndim != 2 or probe.shape[0] < 1 or not np.isfinite(probe).all():
            raise ValueError("AWAC reference probe observations must be a finite matrix")

        self.awac_selectivity_mode = mode
        self.awac_reference_kl_weight = float(reference_kl_weight)
        self.awac_reference_actor = copy.deepcopy(self.actor).to(self.device).eval()
        for parameter in self.awac_reference_actor.parameters():
            parameter.requires_grad_(False)
        self.awac_reference_probe_observations = torch.as_tensor(
            probe, dtype=torch.float32, device=self.device
        )

    def _awac_reference_kl(
        self,
        state: torch.Tensor,
        previous_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.awac_reference_actor is None:
            raise RuntimeError("AWAC reference actor has not been configured")
        location, log_std = self._distribution_parameters(
            self.actor, state, previous_action
        )
        with torch.no_grad():
            reference_location, reference_log_std = self._distribution_parameters(
                self.awac_reference_actor, state, previous_action
            )
        variance_ratio = torch.exp(2.0 * (log_std - reference_log_std))
        mean_term = (location - reference_location).square() * torch.exp(
            -2.0 * reference_log_std
        )
        return (
            reference_log_std
            - log_std
            + 0.5 * (variance_ratio + mean_term - 1.0)
        ).sum(dim=-1)

    @torch.no_grad()
    def awac_reference_probe_diagnostics(self) -> dict[str, float]:
        if (
            self.awac_reference_actor is None
            or self.awac_reference_probe_observations is None
        ):
            return {}
        state = self._encode(self.awac_reference_probe_observations).detach()
        result = {
            "policy_kl_to_a3_reference": float(self._awac_reference_kl(state).mean()),
        }
        current_terms = getattr(self.actor, "cost_terms", None)
        reference_terms = getattr(self.awac_reference_actor, "cost_terms", None)
        if not callable(current_terms) or not callable(reference_terms):
            return result
        q_state, q_action, p_state, p_action = current_terms(state)
        ref_q_state, ref_q_action, ref_p_state, ref_p_action = reference_terms(state)
        d_action = -p_action / q_action
        ref_d_action = -ref_p_action / ref_q_action

        def delta_statistics(prefix: str, value: torch.Tensor) -> dict[str, float]:
            absolute = value.abs().reshape(-1)
            return {
                f"{prefix}_p50": float(torch.quantile(absolute, 0.50)),
                f"{prefix}_p95": float(torch.quantile(absolute, 0.95)),
                f"{prefix}_max": float(absolute.max()),
            }

        drift = {
            **result,
            **delta_statistics("delta_q_state", q_state - ref_q_state),
            **delta_statistics("delta_p_state", p_state - ref_p_state),
            **delta_statistics("delta_d_action", d_action - ref_d_action),
        }
        current_implicit = getattr(self.actor, "implicit_xyz_diagnostics", None)
        reference_implicit = getattr(
            self.awac_reference_actor, "implicit_xyz_diagnostics", None
        )
        if callable(current_implicit) and callable(reference_implicit):
            current_values = current_implicit(state)
            reference_values = reference_implicit(state)
            for output_key, metric_key in (
                ("delta_d_xyz", "delta_D_output"),
                ("delta_d_xyz_physical_m", "delta_D_physical_m"),
                ("delta_d_xyz_utilization", "delta_D_utilization"),
                ("q_xyz_over_base", "delta_Q_over_base"),
                ("d_action", "delta_d_action_implicit"),
            ):
                drift.update(
                    delta_statistics(
                        metric_key,
                        current_values[output_key] - reference_values[output_key],
                    )
                )
        drift["policy_kl_to_continuation_reference"] = drift[
            "policy_kl_to_a3_reference"
        ]
        return drift

    def _update_awac_actor(self, batch: TensorBatch) -> dict[str, float]:
        state = self._encode(batch.observation).detach()
        with torch.no_grad():
            data_q = self._reduce_actor_q(self.critic(state, batch.action))
            policy_action, _, _ = self._sample_actor_with_context(
                state, generator=self.training_generator
                , previous_action=batch.previous_action
            )
            policy_q = self._reduce_actor_q(self.critic(state, policy_action))
            advantage = data_q - policy_q
            weights = torch.exp(
                advantage / self.config.method_spec.advantage_temperature
            ).clamp(max=self.config.method_spec.advantage_weight_max)
            median = torch.quantile(advantage, 0.50)
            positive_mask = advantage > 0
            strong_mask = positive_mask & (advantage >= median)
            mode = self.awac_selectivity_mode or "all"
            if mode == "all":
                selected = torch.ones_like(positive_mask)
            elif mode in {"positive", "positive_klref"}:
                selected = positive_mask
            elif mode == "positive_top50":
                selected = strong_mask
            else:
                raise RuntimeError(f"Unsupported configured AWAC selectivity {mode!r}")
        log_prob = self._data_action_log_prob(state, batch.action, batch.previous_action)
        selected_count = int(selected.sum().item())
        rejected = ~selected
        weight_max = float(self.config.method_spec.advantage_weight_max)

        quantiles = {
            f"advantage_p{int(100*q):02d}": float(torch.quantile(advantage, q))
            for q in (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
        }
        weight_quantiles = {
            f"advantage_weight_p{int(100*q):02d}": float(torch.quantile(weights, q))
            for q in (0.50, 0.90, 0.95, 0.99)
        }
        selected_weight_statistics = _finite_subset_statistics(
            "selected_weight", weights, selected
        )
        subset_statistics: dict[str, float] = {}
        for subset_name, subset_mask in (("selected", selected), ("rejected", rejected)):
            for value_name, metric_value in (
                ("data_q", data_q),
                ("policy_q", policy_q),
                ("advantage", advantage),
                ("weight", weights),
                ("reward", batch.reward),
            ):
                subset_statistics.update(
                    _finite_subset_statistics(
                        f"{subset_name}_{value_name}", metric_value, subset_mask
                    )
                )

        common_metrics = {
            "advantage_mean": float(advantage.mean()),
            "advantage_std": float(advantage.std(unbiased=False)),
            **quantiles,
            "advantage_positive_fraction": float(positive_mask.float().mean()),
            "advantage_positive_top50_fraction": float(strong_mask.float().mean()),
            "selected_fraction": float(selected.float().mean()),
            "selected_sample_count": float(selected_count),
            "advantage_weight_mean": float(weights.mean()),
            **weight_quantiles,
            "advantage_weight_max": float(weights.max()),
            "advantage_weight_max_hit_rate": float(
                (weights >= weight_max * (1.0 - 1e-6)).float().mean()
            ),
            **selected_weight_statistics,
            **subset_statistics,
        }
        if selected_count == 0:
            return {
                "actor_loss": 0.0,
                "actor_awac_loss": 0.0,
                "actor_reference_kl_loss": 0.0,
                "actor_reference_kl_value": 0.0,
                "actor_update_applied": 0.0,
                "actor_selectivity_empty_batch": 1.0,
                "actor_grad_norm": 0.0,
                "awac_grad_norm": 0.0,
                "kl_grad_norm": 0.0,
                "shared_trunk_grad_norm": 0.0,
                "q_state_head_grad_norm": 0.0,
                "action_p_head_grad_norm": 0.0,
                "entropy": float((-log_prob).detach().mean()),
                "temperature": 0.0,
                "temperature_loss": 0.0,
                **common_metrics,
            }

        awac_loss = -(weights[selected] * log_prob[selected]).sum() / selected.sum()
        reference_kl_value = log_prob.sum() * 0.0
        reference_kl_loss = log_prob.sum() * 0.0
        if mode == "positive_klref":
            reference_kl_value = self._awac_reference_kl(
                state, batch.previous_action
            ).mean()
            reference_kl_loss = self.awac_reference_kl_weight * reference_kl_value
        actor_loss = awac_loss + reference_kl_loss
        self.actor_optimizer.zero_grad(set_to_none=True)
        parameters = tuple(
            parameter for parameter in self.actor.parameters() if parameter.requires_grad
        )
        awac_grad_norm = _component_gradient_norm(
            awac_loss, parameters, retain_graph=True
        )
        kl_grad_norm = (
            _component_gradient_norm(reference_kl_loss, parameters, retain_graph=True)
            if mode == "positive_klref"
            else 0.0
        )
        actor_loss.backward()
        headroom_gradient_metrics: dict[str, float] = {}
        gradient_diagnostics = getattr(
            self.actor, "action_headroom_gradient_diagnostics", None
        )
        if callable(gradient_diagnostics):
            headroom_gradient_metrics = gradient_diagnostics()
        selective_gradient_metrics: dict[str, float] = {}
        selective_gradient_diagnostics = getattr(
            self.actor, "selective_awac_gradient_diagnostics", None
        )
        if callable(selective_gradient_diagnostics):
            selective_gradient_metrics = selective_gradient_diagnostics()
        actor_grad_norm = _clip(self.actor.parameters(), self.actor_optimizer)
        self.actor_optimizer.step()
        self.actor_updates += 1
        return {
            "actor_loss": float(actor_loss.detach()),
            "actor_awac_loss": float(awac_loss.detach()),
            "actor_reference_kl_loss": float(reference_kl_loss.detach()),
            "actor_reference_kl_value": float(reference_kl_value.detach()),
            "actor_selectivity_empty_batch": 0.0,
            "actor_update_applied": 1.0,
            "actor_grad_norm": actor_grad_norm,
            "awac_grad_norm": awac_grad_norm,
            "kl_grad_norm": kl_grad_norm,
            "entropy": float((-log_prob).detach().mean()),
            "temperature": 0.0,
            "temperature_loss": 0.0,
            **common_metrics,
            **headroom_gradient_metrics,
            **selective_gradient_metrics,
        }

    def _update_iql_once(self, batch: TensorBatch) -> dict[str, float]:
        if self.value is None or self.value_optimizer is None:
            raise RuntimeError("IQL value network is not initialized")
        state = self._encode(batch.observation).detach()
        next_state = self._encode(batch.next_observation).detach()
        with torch.no_grad():
            target_q = self.target_critic(state, batch.action).amin(dim=0)
        value = self.value(state)
        residual = target_q - value
        expectile = self.config.method_spec.expectile
        value_weight = torch.where(residual > 0, expectile, 1.0 - expectile)
        value_loss = (value_weight * residual.square()).mean()
        self.value_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        value_grad_norm = _clip(self.value.parameters(), self.value_optimizer)
        self.value_optimizer.step()

        with torch.no_grad():
            q_target = (
                batch.reward
                + self.config.discount * batch.discount * self.value(next_state)
            )
        q = self.critic(state, batch.action)
        critic_loss = self._reduce_critic_objective(
            (q - q_target.unsqueeze(0)).square()
        )
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad_norm = _clip(self.critic.parameters(), self.critic_optimizer)
        self.critic_optimizer.step()
        with torch.no_grad():
            for target_parameter, parameter in zip(
                self.target_critic.parameters(), self.critic.parameters(), strict=True
            ):
                target_parameter.lerp_(parameter, self.config.target_tau)
            advantage = self.target_critic(state, batch.action).amin(dim=0) - self.value(state)
            weights = torch.exp(
                self.config.method_spec.advantage_temperature * advantage
            ).clamp(max=self.config.method_spec.advantage_weight_max)
        log_prob = self._data_action_log_prob(state, batch.action)
        actor_loss = -(weights * log_prob).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad_norm = _clip(self.actor.parameters(), self.actor_optimizer)
        self.actor_optimizer.step()
        self.gradient_updates += 1
        self.actor_updates += 1
        return {
            "critic_loss": float(critic_loss.detach()),
            "bellman_loss": float(critic_loss.detach()),
            "critic_grad_norm": critic_grad_norm,
            "q_mean": float(q.detach().mean()),
            "target_q_mean": float(q_target.mean()),
            "value_loss": float(value_loss.detach()),
            "value_grad_norm": value_grad_norm,
            "value_mean": float(value.detach().mean()),
            "actor_loss": float(actor_loss.detach()),
            "actor_grad_norm": actor_grad_norm,
            "entropy": float((-log_prob).detach().mean()),
            "temperature": 0.0,
            "temperature_loss": 0.0,
            "advantage_mean": float(advantage.mean()),
            "advantage_weight_mean": float(weights.mean()),
            "cql_penalty": 0.0,
            "calibration_bound_rate": 0.0,
            "mpve_applied": 0.0,
            "mpve_loss": 0.0,
            "mpve_target_mean": 0.0,
        }

    def update(
        self,
        batch: TensorBatch,
        utd: int,
        *,
        phase: Literal["offline", "online"],
        actor_updates_enabled: bool = True,
        actor_batch: TensorBatch | None = None,
    ) -> dict[str, float]:
        """Apply critic updates from ``batch`` and an optional separate actor batch.

        RLPD's critic sampling ratio and AWAC's behaviour-cloning target
        distribution answer different questions.  Keeping the two tensors
        separate makes an actor-replay composition screen causal while
        preserving the critic's UTD/fused batch exactly.
        """
        if phase not in ("offline", "online"):
            raise ValueError("phase must be exactly 'offline' or 'online'")
        if batch.reward.shape[0] % utd:
            raise ValueError("Fused batch must divide evenly by UTD")
        size = batch.reward.shape[0] // utd
        if self.config.learner_family == "iql":
            metrics: dict[str, float] = {}
            for index in range(utd):
                metrics = self._update_iql_once(
                    batch.slice(slice(index * size, (index + 1) * size))
                )
            return metrics
        cache = self._prepare_critic_cache(batch, phase=phase)
        metrics: dict[str, float] = {}
        mini_batch = batch
        for index in range(utd):
            batch_slice = slice(index * size, (index + 1) * size)
            mini_batch = batch.slice(batch_slice)
            mini_cache = cache.slice(batch_slice)
            # MPVE runs once per logical update, independent of REDQ's critic
            # UTD.  Its scope is part of the immutable method identity:
            # Offline-MPVE applies it during offline pretraining, while the
            # original MPVE ablation applies it only online.
            apply_mpve = (
                (
                    phase == "offline" and self.config.uses_offline_mpve
                    or phase == "online" and self.config.uses_online_mpve
                )
                and index + 1 == utd
            )
            metrics = self.update_critic(
                mini_batch,
                apply_mpve=apply_mpve,
                cache=mini_cache,
                phase=phase,
            )
        self.logical_updates += 1
        metrics["offline_batch_fraction"] = float(
            mini_batch.offline_mask.detach().mean()
        )
        metrics["online_batch_fraction"] = float(
            1.0 - mini_batch.offline_mask.detach().mean()
        )
        actor_mini_batch = mini_batch if actor_batch is None else actor_batch
        metrics["actor_offline_batch_fraction"] = float(
            actor_mini_batch.offline_mask.detach().mean()
        )
        metrics["actor_online_batch_fraction"] = float(
            1.0 - actor_mini_batch.offline_mask.detach().mean()
        )
        if not actor_updates_enabled:
            metrics.update(
                {
                    "actor_update_applied": 0.0,
                    "actor_frozen": 1.0,
                    "actor_grad_norm": 0.0,
                    "temperature_loss": 0.0,
                }
            )
        elif self.logical_updates % self.config.actor_update_interval != 0:
            metrics.update(
                {
                    "actor_update_applied": 0.0,
                    "actor_update_interval": float(
                        self.config.actor_update_interval
                    ),
                }
            )
        elif self.config.learner_family == "awac":
            # AWAC used to bypass actor_update_interval entirely, making the
            # advertised slow-actor online schedule ineffective.  Interval 1
            # preserves the original/offline algorithm exactly; larger
            # values now provide the same explicit scheduling contract used
            # by the SAC-family learners.
            actor_metrics = self._update_awac_actor(actor_mini_batch)
            metrics.update(actor_metrics)
            metrics.setdefault("actor_update_applied", 1.0)
        else:
            actor_state = (
                mini_cache.state
                if actor_batch is None
                else self._encode(actor_mini_batch.observation).detach()
            )
            metrics.update(
                self.update_actor_and_temperature(
                    actor_mini_batch,
                    state=actor_state,
                )
            )
        metrics["actor_update_interval"] = float(
            self.config.actor_update_interval
        )
        return metrics

    def state_dict(self) -> dict[str, Any]:
        state = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "log_temperature": self.log_temperature.detach().cpu(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "temperature_optimizer": self.temperature_optimizer.state_dict(),
            "gradient_updates": self.gradient_updates,
            "logical_updates": self.logical_updates,
            "actor_updates": self.actor_updates,
            "representation": self.representation_identity(),
            # This is part of the scientific resume contract: initialization
            # seeds document cross-method pairing, while the private sampling
            # state makes the next stochastic update exactly reproducible.
            "rng_substreams": {
                "version": RNG_SUBSTREAM_VERSION,
                "base_seed": int(self.config.seed),
                "seeds": dict(self.rng_substream_seeds),
                "training_sampling_device": str(self.device),
                "training_sampling_state": self.training_generator.get_state().cpu(),
            },
        }
        if self.value is not None:
            assert self.value_optimizer is not None
            state["value"] = self.value.state_dict()
            state["value_optimizer"] = self.value_optimizer.state_dict()
        return state

    def load_state_dict(
        self,
        state: dict[str, Any],
        *,
        restore_sampling_rng: bool = True,
    ) -> None:
        """Restore learner state.

        Training resume/fork keeps the default and restores the private
        generator exactly on the configured device.  Deterministic evaluation
        may explicitly skip that device-specific CUDA/CPU generator state;
        it never draws policy noise, while model parameters remain identical.
        """
        rng_state = state.get("rng_substreams")
        if not isinstance(rng_state, dict):
            raise ValueError("Learner checkpoint is missing RNG substream state")
        expected_identity = {
            "version": RNG_SUBSTREAM_VERSION,
            "base_seed": int(self.config.seed),
            "seeds": self.rng_substream_seeds,
        }
        actual_identity = {
            key: rng_state.get(key) for key in expected_identity
        }
        if actual_identity != expected_identity:
            raise ValueError(
                "Learner checkpoint RNG substreams do not match this configuration"
            )
        if state.get("representation") != self.representation_identity():
            raise ValueError(
                "Learner checkpoint representation/normalizer identity differs"
            )
        training_sampling_state = rng_state.get("training_sampling_state")
        if not isinstance(training_sampling_state, torch.Tensor):
            raise ValueError("Learner checkpoint has no Torch sampling-generator state")
        sampling_device = rng_state.get("training_sampling_device")
        if not isinstance(sampling_device, str) or not sampling_device:
            raise ValueError("Learner checkpoint has no sampling-generator device")
        if restore_sampling_rng and sampling_device != str(self.device):
            raise ValueError(
                "Learner sampling RNG device differs; training resume must use "
                "the checkpoint device"
            )
        actor_state = state["actor"]
        augment_actor_state = getattr(
            self.actor, "augment_policy_preserving_checkpoint_actor_state", None
        )
        actor_optimizer_state = state["actor_optimizer"]
        if callable(augment_actor_state):
            actor_state = augment_actor_state(actor_state)
            # The inherited parameters keep their Adam moments exactly.  The
            # appended zero adapter and frozen diagnostic source copy start
            # without moments, which is Adam's canonical state for new
            # parameters.  Remap by parameter position and retain the target
            # optimizer's full parameter list.
            source_optimizer = copy.deepcopy(actor_optimizer_state)
            target_optimizer = self.actor_optimizer.state_dict()
            if len(source_optimizer["param_groups"]) != len(
                target_optimizer["param_groups"]
            ):
                raise ValueError("Policy-preserving optimizer group count differs")
            identifier_map: dict[int, int] = {}
            for source_group, target_group in zip(
                source_optimizer["param_groups"],
                target_optimizer["param_groups"],
            ):
                source_ids = list(source_group["params"])
                target_ids = list(target_group["params"])
                if len(source_ids) > len(target_ids):
                    raise ValueError(
                        "Policy-preserving actor has fewer parameters than its source"
                    )
                identifier_map.update(zip(source_ids, target_ids[: len(source_ids)]))
                source_group["params"] = target_ids
            source_optimizer["state"] = {
                identifier_map[source_id]: value
                for source_id, value in source_optimizer["state"].items()
                if source_id in identifier_map
            }
            actor_optimizer_state = source_optimizer
        self.actor.load_state_dict(actor_state, strict=True)
        self.critic.load_state_dict(state["critic"], strict=True)
        self.target_critic.load_state_dict(state["target_critic"], strict=True)
        with torch.no_grad():
            self.log_temperature.copy_(state["log_temperature"].to(self.device))
        self.actor_optimizer.load_state_dict(actor_optimizer_state)
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.temperature_optimizer.load_state_dict(state["temperature_optimizer"])
        if self.value is not None:
            if self.value_optimizer is None or "value" not in state:
                raise ValueError("IQL checkpoint is missing value-network state")
            self.value.load_state_dict(state["value"], strict=True)
            self.value_optimizer.load_state_dict(state["value_optimizer"])
        _optimizer_to(self.actor_optimizer, self.device)
        if self.value_optimizer is not None:
            _optimizer_to(self.value_optimizer, self.device)
        _optimizer_to(self.critic_optimizer, self.device)
        _optimizer_to(self.temperature_optimizer, self.device)
        self.target_critic.eval()
        for parameter in self.target_critic.parameters():
            parameter.requires_grad_(False)
        self.gradient_updates = int(state["gradient_updates"])
        self.logical_updates = int(state.get("logical_updates", 0))
        self.actor_updates = int(state["actor_updates"])
        if restore_sampling_rng:
            try:
                self.training_generator.set_state(training_sampling_state.cpu())
            except RuntimeError as exc:
                raise ValueError(
                    "Learner sampling RNG is incompatible with the restore device; "
                    "training resume must use the checkpoint device"
                ) from exc

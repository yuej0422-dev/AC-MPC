"""Clock-conditioned Q,p-KMPC structures for the direct-online causal screen."""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.dmc.o2o.networks import LOG_STD_MAX, LOG_STD_MIN, atanh_clipped

from antmaze_ac.envs.circle_phase_feedforward import FrozenCirclePhaseFeedforward


STRUCTURES = (*tuple(f"k{index}" for index in range(11)), "k10v2", "k10p")


class TimeStructuredQpKMPCTanhGaussianActor(nn.Module):
    """Solve residual KMPC while exposing only ``[body lift, tau]`` to Actor."""

    def __init__(
        self,
        koopman: nn.Module,
        feedforward: FrozenCirclePhaseFeedforward,
        xref: np.ndarray,
        *,
        structure: str,
        horizon: int = 5,
        solver_iterations: int = 20,
        hidden_dim: int = 256,
        hidden_layers: int = 2,
        residual_limit: float = 0.3,
        physical_action_limit: float = 0.3,
        state_cost_scale: float = 1.0,
        residual_cost_scale: float = 10_000.0,
        quadratic_log_scale: float = 1.5,
        action_cost_center_limit: float = 0.005,
        action_headroom_limit: float | None = None,
        action_headroom_adapter_only: bool = False,
        action_headroom_eps: float = 1e-6,
        state_cost_gate_init: float = 1e-4,
        state_cost_gate_min: float = 1e-6,
        terminal_multiplier: float = 1.0,
        log_std_min: float = LOG_STD_MIN,
        log_std_init: float = -3.5,
        log_std_max: float = -3.0,
    ) -> None:
        super().__init__()
        self.koopman = koopman
        self.structure = structure.lower()
        self.horizon = int(horizon)
        self.solver_iterations = int(solver_iterations)
        self.hidden_layers = int(hidden_layers)
        self.action_limit = float(residual_limit)
        self.physical_action_limit = float(physical_action_limit)
        self.quadratic_log_scale = float(quadratic_log_scale)
        self.action_cost_center_limit = float(action_cost_center_limit)
        self.action_headroom_limit = (
            None if action_headroom_limit is None else float(action_headroom_limit)
        )
        self.action_headroom_adapter_only = bool(action_headroom_adapter_only)
        self.action_headroom_eps = float(action_headroom_eps)
        self.state_cost_gate_init = float(state_cost_gate_init)
        self.state_cost_gate_min = float(state_cost_gate_min)
        self.terminal_multiplier = float(terminal_multiplier)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        if self.structure not in STRUCTURES:
            raise ValueError(f"Unknown causal-screen structure {structure!r}")
        if self.horizon != 5:
            raise ValueError("The causal screen freezes KMPC horizon at H=5")
        if min(self.solver_iterations, self.hidden_layers) < 1:
            raise ValueError("KMPC dimensions must be positive")
        if not 0 < self.action_limit <= self.physical_action_limit:
            raise ValueError("Residual limit must not exceed physical limit")
        if state_cost_scale <= 0 or residual_cost_scale <= 0:
            raise ValueError("Base Q/R scales must be positive")
        if not self.log_std_min <= log_std_init <= self.log_std_max <= LOG_STD_MAX:
            raise ValueError("Invalid log-std limits")
        if self.structure == "k10" and not (
            0.0 < self.state_cost_gate_min < self.state_cost_gate_init < 1.0
        ):
            raise ValueError("K10 state-cost gate must satisfy 0 < min < init < 1")
        if koopman.state_dim != 46 or koopman.action_dim != 18:
            raise ValueError("Expected body-state Koopman plus scalar tau")

        self.explicit_xref = self.structure not in {"k9", "k10", "k10v2", "k10p"}
        self.shared_horizon = self.structure in {"k3", "k4", "k5", "k6", "k8", "k9", "k10"}
        self.terminal_enabled = self.structure in {"k5", "k6", "k8"}
        self.action_p_enabled = self.structure in {"k2", "k4", "k6", "k7", "k8", "k9", "k10", "k10v2", "k10p"}
        if self.action_headroom_limit is not None:
            if not self.action_p_enabled:
                raise ValueError("Action headroom requires an action-p structure")
            if not self.action_headroom_limit > self.action_cost_center_limit:
                raise ValueError("Expanded action headroom must exceed the old limit")
            if not 0.0 < self.action_headroom_eps < 0.1:
                raise ValueError("Action headroom epsilon must lie in (0, 0.1)")
        self.state_p_enabled = self.structure in {"k9", "k10", "k10v2", "k10p"}
        self.adaptive_state_q = self.structure not in {"k0", "k10p"}
        self.adaptive_action_q = self.structure in {"k7", "k8"}
        self.legacy_joint_q_centering = self.structure == "k7"
        self.frozen_cost_map = self.structure == "k0"
        self.state_cost_gate_enabled = self.structure == "k10"
        groups = 1 if self.shared_horizon else self.horizon
        self.output_groups = groups
        output_dim = 0
        if self.adaptive_state_q:
            output_dim += groups * 45
        if self.adaptive_action_q:
            output_dim += groups * 18
        if self.action_p_enabled:
            output_dim += groups * 18
        if self.state_p_enabled:
            output_dim += groups * 45
        if self.state_cost_gate_enabled:
            output_dim += groups
        output_dim = max(output_dim, 1)
        state_p_offset = 0
        if self.adaptive_state_q:
            state_p_offset += groups * 45
        if self.adaptive_action_q:
            state_p_offset += groups * 18
        if self.action_p_enabled:
            self.action_p_output_slice = slice(
                state_p_offset, state_p_offset + groups * 18
            )
            state_p_offset += groups * 18
        else:
            self.action_p_output_slice = None
        self.state_q_output_slice = (
            slice(0, groups * 45) if self.adaptive_state_q else None
        )
        self.state_p_output_slice = (
            slice(state_p_offset, state_p_offset + groups * 45)
            if self.state_p_enabled
            else None
        )
        layers: list[nn.Module] = []
        input_dim = int(koopman.lifted_dim)
        for _ in range(self.hidden_layers):
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.GELU()))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, output_dim))
        self.controller = nn.Sequential(*layers)
        nn.init.zeros_(self.controller[-1].weight)
        nn.init.zeros_(self.controller[-1].bias)
        self._state_gate_offset = None
        if self.state_cost_gate_enabled:
            self._state_gate_offset = output_dim - groups
            gate_probability = min(
                max(
                    (self.state_cost_gate_init - self.state_cost_gate_min)
                    / (1.0 - self.state_cost_gate_min),
                    1e-8,
                ),
                1.0 - 1e-8,
            )
            self.controller[-1].bias.data[self._state_gate_offset :] = math.log(
                gate_probability / (1.0 - gate_probability)
            )
        self.log_std = nn.Parameter(torch.full((18,), float(log_std_init)))
        self.action_headroom_adapter: nn.Linear | None = None
        self.source_controller: nn.Sequential | None = None
        if self.action_headroom_limit is not None:
            # The adapter consumes the existing shared feature and is exactly
            # zero at construction.  The frozen source copy is diagnostic
            # only: it measures effective d_action change from A3@9k on the
            # same visited states without entering the optimizer gradient.
            self.action_headroom_adapter = nn.Linear(input_dim, groups * 18)
            nn.init.zeros_(self.action_headroom_adapter.weight)
            nn.init.zeros_(self.action_headroom_adapter.bias)
            self.source_controller = copy.deepcopy(self.controller).eval()
            for parameter in self.source_controller.parameters():
                parameter.requires_grad_(False)
            if self.action_headroom_adapter_only:
                self.log_std.requires_grad_(False)
                for parameter in self.controller.parameters():
                    parameter.requires_grad_(False)

        device = koopman.A.device
        reference = np.asarray(xref, dtype=np.float32)
        if reference.shape != (1001, 45) or not np.isfinite(reference).all():
            raise ValueError("xref must have shape [1001,45]")
        reference_normalized = (
            torch.as_tensor(reference, device=device) - koopman.physical_center
        ) / koopman.physical_scale
        self.register_buffer("xref_table", reference_normalized)
        self.register_buffer(
            "feedforward_table",
            torch.as_tensor(feedforward.action(np.arange(1000), 1000), device=device),
        )
        self.register_buffer(
            "base_state_q", torch.full((45,), float(state_cost_scale), device=device)
        )
        self.register_buffer(
            "base_action_q", torch.full((18,), float(residual_cost_scale), device=device)
        )
        # K9's moving center can span exactly the normalized reference envelope.
        # A tiny numerical floor only prevents dead dimensions from becoming
        # permanently unlearnable; it is not a hand-selected large authority.
        d_state_limit = reference_normalized.abs().amax(dim=0).clamp_min(1e-3)
        self.register_buffer("state_cost_center_limit", d_state_limit)
        self.register_buffer(
            "terminal_q", self.base_state_q * float(self.terminal_multiplier)
        )

        lifted_power = torch.eye(koopman.lifted_dim, device=device)
        lifted_action = torch.zeros(
            koopman.lifted_dim, self.horizon * 18, device=device
        )
        state_rows: list[torch.Tensor] = []
        action_rows: list[torch.Tensor] = []
        physical_output = koopman.C[:45]
        for step in range(self.horizon):
            lifted_power = koopman.A @ lifted_power
            lifted_action = koopman.A @ lifted_action
            lifted_action[:, step * 18 : (step + 1) * 18] += koopman.B
            state_rows.append(physical_output @ lifted_power)
            action_rows.append((physical_output @ lifted_action).clone())
        self.register_buffer("state_map", torch.cat(state_rows, dim=0))
        self.register_buffer("action_map", torch.cat(action_rows, dim=0))

    def warm_initialize_state_center(
        self,
        lifted_state: torch.Tensor,
        target_normalized_state: torch.Tensor,
        *,
        ridge: float = 1e-4,
    ) -> dict[str, float]:
        """Validate the exact model-consistent pure-FF center initialization.

        K10-v2/P use a zero-head correction around the frozen Koopman free
        rollout, so no approximate network fit is needed. The recorded x_ff
        trajectory is used only to size correction authority and report the
        initialization discrepancy; it is not retained as an Actor input.
        """
        if self.structure not in {"k10v2", "k10p"} or self.state_p_output_slice is None:
            raise ValueError("Nominal state-center warm initialization requires K10-v2/P")
        if lifted_state.ndim != 2 or target_normalized_state.shape != (lifted_state.shape[0], 45):
            raise ValueError("Warm initialization expects [N,lifted] inputs and [N,45] targets")
        if not math.isfinite(ridge) or ridge <= 0:
            raise ValueError("Warm initialization ridge must be positive")
        with torch.no_grad():
            lifted_state = lifted_state.detach()
            nominal_normalized_state = target_normalized_state.detach()
            target_limit = nominal_normalized_state.abs().amax(dim=0) * 1.25
            self.state_cost_center_limit.copy_(
                torch.maximum(self.state_cost_center_limit, target_limit.clamp_min(1e-3))
            )
            q_state, _, p_state, _ = self.cost_terms(lifted_state)
            fitted = -p_state / q_state
            nominal_error = fitted[..., 0, :] - nominal_normalized_state
            centered_fitted = fitted.reshape(-1) - fitted.mean()
            feedforward = self.reference_plans(lifted_state)[0]
            ff_flat = feedforward.reshape(lifted_state.shape[0], self.horizon * 18)
            free_state = (
                F.linear(lifted_state, self.state_map)
                + F.linear(ff_flat, self.action_map)
            ).reshape(lifted_state.shape[0], self.horizon, 45)
            error = fitted - free_state
            centered_target = free_state.reshape(-1) - free_state.mean()
            correlation = (centered_fitted * centered_target).sum() / (
                centered_fitted.norm() * centered_target.norm()
            ).clamp_min(1e-12)
            first_residual = self.plan(lifted_state)[..., 0, :]
        return {
            "samples": float(lifted_state.shape[0]),
            "ridge": float(ridge),
            "refinement_steps": 0.0,
            "mse_normalized": float(error.square().mean().cpu()),
            "rmse_normalized": float(error.square().mean().sqrt().cpu()),
            "correlation": float(correlation.cpu()),
            "max_abs_error_normalized": float(error.abs().amax().cpu()),
            "rmse_to_nominal_xff_normalized": float(
                nominal_error.square().mean().sqrt().cpu()
            ),
            "d_state_limit_max": float(self.state_cost_center_limit.max().cpu()),
            "first_residual_p95_abs": float(
                torch.quantile(first_residual.abs().reshape(-1), 0.95).cpu()
            ),
            "first_residual_max_abs": float(first_residual.abs().amax().cpu()),
        }

    def _time_index(self, lifted_state: torch.Tensor) -> torch.Tensor:
        return torch.round(lifted_state[..., -1].clamp(0.0, 1.0) * 1000.0).long()

    def reference_plans(self, lifted_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        index = self._time_index(lifted_state)
        offset = torch.arange(self.horizon, device=index.device)
        action_index = (index.unsqueeze(-1) + offset).clamp_max(999)
        state_index = (index.unsqueeze(-1) + offset + 1).clamp_max(1000)
        return self.feedforward_table[action_index], self.xref_table[state_index]

    def residual_bounds(self, feedforward_plan: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lower = torch.maximum(
            torch.full_like(feedforward_plan, -self.action_limit),
            -self.physical_action_limit - feedforward_plan,
        )
        upper = torch.minimum(
            torch.full_like(feedforward_plan, self.action_limit),
            self.physical_action_limit - feedforward_plan,
        )
        if torch.any(lower >= upper):
            raise RuntimeError("Feedforward leaves no feasible residual interval")
        return lower, upper

    def _expand(self, value: torch.Tensor) -> torch.Tensor:
        if self.shared_horizon:
            if value.shape[-2] != 1:
                raise ValueError("Shared-horizon cost must have exactly one output group")
            return value.expand(*value.shape[:-2], self.horizon, value.shape[-1])
        return value

    def _controller_features(self, lifted_state: torch.Tensor) -> torch.Tensor:
        return self.controller[:-1](lifted_state)

    def _raw_action_p(self, raw: torch.Tensor) -> torch.Tensor | None:
        if self.action_p_output_slice is None:
            return None
        batch = raw.shape[:-1]
        return raw[..., self.action_p_output_slice].reshape(
            *batch, self.output_groups, 18
        )

    def _headroom_terms(
        self,
        lifted_state: torch.Tensor,
        raw_action_p: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return old center, expanded center, adapter, and adapter derivative."""
        if self.action_headroom_limit is None or self.action_headroom_adapter is None:
            raise RuntimeError("Action headroom is not enabled")
        features = self._controller_features(lifted_state)
        delta_r = self.action_headroom_adapter(features).reshape(
            *lifted_state.shape[:-1], self.output_groups, 18
        )
        d_old = self.action_cost_center_limit * torch.tanh(raw_action_p)
        ratio = (d_old / self.action_headroom_limit).clamp(
            -1.0 + self.action_headroom_eps,
            1.0 - self.action_headroom_eps,
        )
        r_base = torch.atanh(ratio)
        d_new = self.action_headroom_limit * torch.tanh(r_base + delta_r)
        adapter_derivative = self.action_headroom_limit * (
            1.0 - torch.tanh(r_base + delta_r).square()
        )
        return d_old, d_new, delta_r, adapter_derivative

    def action_headroom_diagnostics(
        self, lifted_state: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if self.action_headroom_limit is None:
            return {}
        raw = self.controller(lifted_state)
        raw_action_p = self._raw_action_p(raw)
        assert raw_action_p is not None
        d_old, d_new, delta_r, adapter_derivative = self._headroom_terms(
            lifted_state, raw_action_p
        )
        assert self.source_controller is not None
        source_raw_action_p = self._raw_action_p(self.source_controller(lifted_state))
        assert source_raw_action_p is not None
        d_source = self.action_cost_center_limit * torch.tanh(source_raw_action_p)
        old_derivative = self.action_cost_center_limit * (
            1.0 - torch.tanh(raw_action_p).square()
        )
        return {
            "d_action_old_branch": self._expand(d_old),
            "d_action_new": self._expand(d_new),
            "d_action_source": self._expand(d_source),
            "delta_r": self._expand(delta_r),
            "adapter_derivative": self._expand(adapter_derivative),
            "old_effective_derivative": self._expand(old_derivative),
        }

    @staticmethod
    def _gradient_norm(parameters) -> float:
        squared = None
        for parameter in parameters:
            if parameter.grad is None:
                continue
            value = parameter.grad.detach().square().sum()
            squared = value if squared is None else squared + value
        return 0.0 if squared is None else float(torch.sqrt(squared))

    def action_headroom_gradient_diagnostics(self) -> dict[str, float]:
        if self.action_headroom_adapter is None:
            return {}
        final = self.controller[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("Controller output head must be linear")

        def sliced_norm(output_slice: slice | None) -> float:
            if output_slice is None or final.weight.grad is None:
                return 0.0
            pieces = [final.weight.grad[output_slice]]
            if final.bias.grad is not None:
                pieces.append(final.bias.grad[output_slice])
            return float(torch.sqrt(sum(piece.detach().square().sum() for piece in pieces)))

        return {
            "adapter_grad_norm": self._gradient_norm(
                self.action_headroom_adapter.parameters()
            ),
            "old_action_p_head_grad_norm": sliced_norm(self.action_p_output_slice),
            "q_head_grad_norm": sliced_norm(self.state_q_output_slice),
        }

    def selective_awac_gradient_diagnostics(self) -> dict[str, float]:
        """Split AWAC actor gradients into trunk, Q-state, and action-p heads."""

        final = self.controller[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("Controller output head must be linear")

        def parameter_norm(parameters) -> float:
            squared = torch.zeros((), device=final.weight.device)
            for parameter in parameters:
                if parameter.grad is not None:
                    squared = squared + parameter.grad.detach().square().sum()
            return float(squared.sqrt().cpu())

        def sliced_head_norm(output_slice: slice | None) -> float:
            if output_slice is None:
                return 0.0
            squared = torch.zeros((), device=final.weight.device)
            if final.weight.grad is not None:
                squared = squared + final.weight.grad[output_slice].detach().square().sum()
            if final.bias.grad is not None:
                squared = squared + final.bias.grad[output_slice].detach().square().sum()
            return float(squared.sqrt().cpu())

        return {
            "shared_trunk_grad_norm": parameter_norm(self.controller[:-1].parameters()),
            "q_state_head_grad_norm": sliced_head_norm(self.state_q_output_slice),
            "action_p_head_grad_norm": sliced_head_norm(self.action_p_output_slice),
        }

    def augment_policy_preserving_checkpoint_actor_state(
        self, source_state: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Add zero adapter/frozen-source tensors to a legacy A3 actor state."""
        if self.action_headroom_adapter is None or self.source_controller is None:
            return source_state
        augmented = dict(source_state)
        current = self.state_dict()
        for key in current:
            if key.startswith("action_headroom_adapter."):
                augmented[key] = current[key]
            elif key.startswith("source_controller."):
                inherited_key = key.removeprefix("source_")
                if inherited_key not in source_state:
                    raise KeyError(
                        f"Source actor has no inherited controller tensor {inherited_key!r}"
                    )
                augmented[key] = source_state[inherited_key]
        return augmented

    def cost_terms(
        self, lifted_state: torch.Tensor, *, raw_override: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.controller(lifted_state) if raw_override is None else raw_override
        batch = lifted_state.shape[:-1]
        groups = self.output_groups
        cursor = 0

        raw_state_q: torch.Tensor | None = None
        raw_action_q: torch.Tensor | None = None
        if self.adaptive_state_q:
            count = groups * 45
            raw_state_q = raw[..., cursor : cursor + count].reshape(*batch, groups, 45)
            cursor += count
        if self.adaptive_action_q:
            count = groups * 18
            raw_action_q = raw[..., cursor : cursor + count].reshape(*batch, groups, 18)
            cursor += count
        raw_action_p: torch.Tensor | None = None
        if self.action_p_enabled:
            count = groups * 18
            raw_action_p = raw[..., cursor : cursor + count].reshape(*batch, groups, 18)
            cursor += count
        raw_state_p: torch.Tensor | None = None
        if self.state_p_enabled:
            count = groups * 45
            raw_state_p = raw[..., cursor : cursor + count].reshape(*batch, groups, 45)
            cursor += count
        raw_gate: torch.Tensor | None = None
        if self.state_cost_gate_enabled:
            count = groups
            raw_gate = raw[..., cursor : cursor + count].reshape(*batch, groups, 1)

        if self.legacy_joint_q_centering:
            assert raw_state_q is not None and raw_action_q is not None
            joint = torch.cat((torch.tanh(raw_state_q), torch.tanh(raw_action_q)), dim=-1)
            joint = joint - joint.mean(dim=-1, keepdim=True)
            q_state = torch.exp(self.quadratic_log_scale * joint[..., :45]) * self.base_state_q
            q_action = torch.exp(self.quadratic_log_scale * joint[..., 45:]) * self.base_action_q
        else:
            if raw_state_q is None:
                q_state = self.base_state_q.expand(*batch, groups, 45)
            else:
                state_log = torch.tanh(raw_state_q)
                state_log = state_log - state_log.mean(dim=-1, keepdim=True)
                q_state = torch.exp(self.quadratic_log_scale * state_log) * self.base_state_q
            if raw_action_q is None:
                q_action = self.base_action_q.expand(*batch, groups, 18)
            else:
                action_log = torch.tanh(raw_action_q)
                action_log = action_log - action_log.mean(dim=-1, keepdim=True)
                q_action = torch.exp(self.quadratic_log_scale * action_log) * self.base_action_q
        if self.state_cost_gate_enabled:
            assert raw_gate is not None
            gate = self.state_cost_gate_min + (
                1.0 - self.state_cost_gate_min
            ) * torch.sigmoid(raw_gate)
            q_state = q_state * gate
        q_state = self._expand(q_state)
        q_action = self._expand(q_action)

        if raw_action_p is None:
            d_action = torch.zeros_like(q_action)
        else:
            if self.action_headroom_limit is None:
                d_action = self.action_cost_center_limit * torch.tanh(raw_action_p)
            else:
                _, d_action, _, _ = self._headroom_terms(
                    lifted_state, raw_action_p
                )
            d_action = self._expand(d_action)
        p_action = -q_action * d_action
        if raw_state_p is None:
            d_state = torch.zeros_like(q_state)
        else:
            d_state = torch.tanh(raw_state_p) * self.state_cost_center_limit
            d_state = self._expand(d_state)
            if self.structure in {"k10v2", "k10p"}:
                feedforward = self.reference_plans(lifted_state)[0]
                ff_flat = feedforward.reshape(*batch, self.horizon * 18)
                free_state = F.linear(lifted_state, self.state_map) + F.linear(
                    ff_flat, self.action_map
                )
                d_state = d_state + free_state.reshape(*batch, self.horizon, 45)
        p_state = -q_state * d_state
        return q_state, q_action, p_state, p_action

    def state_cost_gate(self, lifted_state: torch.Tensor) -> torch.Tensor:
        """Return the K10 state-cost gate before horizon expansion."""
        batch = lifted_state.shape[:-1]
        if not self.state_cost_gate_enabled:
            return torch.ones(
                *batch, self.output_groups, 1,
                device=lifted_state.device,
                dtype=lifted_state.dtype,
            )
        assert self._state_gate_offset is not None
        raw_gate = self.controller(lifted_state)[
            ..., self._state_gate_offset : self._state_gate_offset + self.output_groups
        ].reshape(*batch, self.output_groups, 1)
        return self.state_cost_gate_min + (
            1.0 - self.state_cost_gate_min
        ) * torch.sigmoid(raw_gate)

    def sample_uniform_actions(
        self,
        lifted_state: torch.Tensor,
        *,
        samples: int,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feedforward = self.reference_plans(lifted_state)[0][..., 0, :]
        lower, upper = self.residual_bounds(feedforward)
        width = upper - lower
        uniform = torch.rand(
            (samples, *lower.shape), device=lower.device, dtype=lower.dtype, generator=generator
        )
        action = lower.unsqueeze(0) + width.unsqueeze(0) * uniform
        log_density = -width.log().sum(dim=-1)
        return action, log_density.unsqueeze(0).expand(samples, *log_density.shape)

    def cost_map_anchor(self, lifted_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        zero = self.controller(lifted_state).sum() * 0.0
        return zero, zero

    def _qp_components(self, lifted_state: torch.Tensor, *, raw_override: torch.Tensor | None = None):
        batch = lifted_state.shape[:-1]
        q_state, q_action, p_state, p_action = self.cost_terms(
            lifted_state, raw_override=raw_override
        )
        feedforward, reference = self.reference_plans(lifted_state)
        ff_flat = feedforward.reshape(*batch, self.horizon * 18)
        free_state = F.linear(lifted_state, self.state_map)
        free_state = free_state + F.linear(ff_flat, self.action_map)
        if self.explicit_xref:
            free_state = free_state - reference.reshape(*batch, self.horizon * 45)
        q_state_flat = q_state.reshape(*batch, self.horizon * 45)
        q_action_flat = q_action.reshape(*batch, self.horizon * 18)
        p_state_flat = p_state.reshape(*batch, self.horizon * 45)
        p_action_flat = p_action.reshape(*batch, self.horizon * 18)
        weighted_map = self.action_map * q_state_flat.unsqueeze(-1)
        hessian = self.action_map.T @ weighted_map + torch.diag_embed(q_action_flat)
        linear = torch.einsum(
            "...p,pi->...i", q_state_flat * free_state + p_state_flat, self.action_map
        ) + p_action_flat
        if self.terminal_enabled:
            terminal_map = self.action_map[-45:]
            terminal_free = free_state[..., -45:]
            terminal_weighted = terminal_map * self.terminal_q.unsqueeze(-1)
            hessian = hessian + terminal_map.T @ terminal_weighted
            linear = linear + torch.einsum(
                "...p,pi->...i", self.terminal_q * terminal_free, terminal_map
            )
        lower, upper = self.residual_bounds(feedforward)
        return (
            hessian,
            linear,
            lower.reshape(*batch, self.horizon * 18),
            upper.reshape(*batch, self.horizon * 18),
            free_state,
            q_state_flat,
            q_action_flat,
            p_state_flat,
            p_action_flat,
        )

    def _solve(self, hessian, linear, lower, upper):
        lipschitz = hessian.abs().sum(dim=-1).amax(dim=-1)
        step_size = 0.95 / (lipschitz + 1e-6)
        current = torch.zeros_like(linear)
        extrapolated = current.clone()
        momentum = 1.0
        for _ in range(self.solver_iterations):
            gradient = torch.einsum("...ij,...j->...i", hessian, extrapolated) + linear
            following = torch.maximum(
                torch.minimum(extrapolated - step_size.unsqueeze(-1) * gradient, upper), lower
            )
            next_momentum = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * momentum * momentum))
            extrapolated = following + ((momentum - 1.0) / next_momentum) * (
                following - current
            )
            current = following
            momentum = next_momentum
        return current

    def plan(
        self,
        lifted_state: torch.Tensor,
        previous_action: torch.Tensor | None = None,
        *,
        raw_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del previous_action
        components = self._qp_components(lifted_state, raw_override=raw_override)
        current = self._solve(*components[:4])
        return current.reshape(*lifted_state.shape[:-1], self.horizon, 18)

    def plan_diagnostics(self, lifted_state: torch.Tensor) -> dict[str, torch.Tensor]:
        hessian, linear, lower, upper, free_state, qs, qa, ps, pa = self._qp_components(
            lifted_state
        )
        solution = self._solve(hessian, linear, lower, upper)
        gradient = torch.einsum("...ij,...j->...i", hessian, solution) + linear
        projected = solution - torch.maximum(torch.minimum(solution - gradient, upper), lower)
        projected_relative = projected.abs().amax(dim=-1) / (
            1.0 + gradient.abs().amax(dim=-1)
        )
        state = free_state + F.linear(solution, self.action_map)
        objective = (
            0.5 * (qs * state.square()).sum(dim=-1)
            + (ps * state).sum(dim=-1)
            + 0.5 * (qa * solution.square()).sum(dim=-1)
            + (pa * solution).sum(dim=-1)
        )
        terminal = torch.zeros_like(objective)
        if self.terminal_enabled:
            terminal = 0.5 * (self.terminal_q * state[..., -45:].square()).sum(dim=-1)
            objective = objective + terminal
        active = ((solution - lower).abs() < 1e-5) | ((solution - upper).abs() < 1e-5)
        return {
            "objective": objective,
            "terminal_contribution": terminal,
            "active_constraint_fraction": active.float().mean(dim=-1),
            "projected_gradient_relative": projected_relative,
            "solver_converged": (projected_relative < 1e-4).float(),
        }

    def directional_sensitivity(self, lifted_state: torch.Tensor) -> torch.Tensor:
        """Finite-difference ||d(first action)/d(raw cost map)|| in a fixed direction."""
        raw = self.controller(lifted_state)
        if self.frozen_cost_map:
            return torch.zeros(raw.shape[:-1], device=raw.device)
        direction = torch.sin(
            torch.arange(1, raw.shape[-1] + 1, device=raw.device, dtype=raw.dtype)
        )
        direction = direction / direction.norm().clamp_min(1e-12)
        epsilon = 1e-3
        plus = self.plan(lifted_state, raw_override=raw + epsilon * direction)[..., 0, :]
        minus = self.plan(lifted_state, raw_override=raw - epsilon * direction)[..., 0, :]
        return ((plus - minus) / (2.0 * epsilon)).norm(dim=-1)

    def distribution(self, lifted_state: torch.Tensor, previous_action=None):
        plan = self.plan(lifted_state, previous_action=previous_action)
        feedforward = self.reference_plans(lifted_state)[0][..., 0, :]
        lower, upper = self.residual_bounds(feedforward)
        midpoint = 0.5 * (lower + upper)
        half_range = 0.5 * (upper - lower)
        first = plan[..., 0, :].clamp(lower, upper)
        location = atanh_clipped((first - midpoint) / half_range)
        plan = plan.clone()
        plan[..., 0, :] = midpoint + half_range * torch.tanh(location)
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max).expand_as(location)
        return location, log_std, plan

    def sample(
        self,
        lifted_state: torch.Tensor,
        *,
        deterministic: bool = False,
        samples: int = 1,
        return_plan: bool = False,
        generator: torch.Generator | None = None,
        previous_action: torch.Tensor | None = None,
    ):
        location, log_std, plan = self.distribution(lifted_state, previous_action)
        feedforward = self.reference_plans(lifted_state)[0][..., 0, :]
        lower, upper = self.residual_bounds(feedforward)
        midpoint = 0.5 * (lower + upper)
        half_range = 0.5 * (upper - lower)
        sample_shape = () if samples == 1 else (samples,)
        expanded_location = location.expand(*sample_shape, *location.shape)
        expanded_log_std = log_std.expand_as(expanded_location)
        if deterministic:
            pre_tanh = expanded_location
        else:
            noise = torch.randn(
                expanded_location.shape,
                device=expanded_location.device,
                dtype=expanded_location.dtype,
                generator=generator,
            )
            pre_tanh = expanded_location + expanded_log_std.exp() * noise
        action = midpoint * torch.ones_like(pre_tanh) + half_range * torch.tanh(pre_tanh)
        normal_log_prob = -0.5 * (
            ((pre_tanh - expanded_location) / expanded_log_std.exp()).square()
            + 2.0 * expanded_log_std
            + math.log(2.0 * math.pi)
        )
        correction = 2.0 * (
            math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh)
        )
        log_prob = (normal_log_prob - correction - half_range.log()).sum(dim=-1)
        return action, log_prob, plan if return_plan else None

    def data_action_log_prob(
        self,
        lifted_state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Density of replay residuals in the actor's physical feasible box.

        AWAC evaluates dataset actions rather than newly sampled actions.  The
        KMPC actor's tanh transform is centered and scaled by the residual box
        left after adding ``u_ff(t)``; using the ordinary fixed ``[-1, 1]``
        Jacobian would therefore optimize a different action measure.
        """

        location, log_std, _ = self.distribution(lifted_state)
        feedforward = self.reference_plans(lifted_state)[0][..., 0, :]
        lower, upper = self.residual_bounds(feedforward)
        midpoint = 0.5 * (lower + upper)
        half_range = 0.5 * (upper - lower)
        normalized = (action - midpoint) / half_range
        pre_tanh = atanh_clipped(normalized)
        normal_log_prob = -0.5 * (
            ((pre_tanh - location) / log_std.exp()).square()
            + 2.0 * log_std
            + math.log(2.0 * math.pi)
        )
        correction = 2.0 * (
            math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh)
        )
        return (normal_log_prob - correction - half_range.log()).sum(dim=-1)

"""Differentiable KMPC that optimizes feedback around a frozen feedforward."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.dmc.o2o.networks import LOG_STD_MAX, LOG_STD_MIN, atanh_clipped

from antmaze_ac.envs.circle_phase_feedforward import (
    FrozenCirclePhaseFeedforward,
)


def _condense_dynamics(
    koopman: nn.Module,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    lifted_dim, action_dim = koopman.B.shape
    lifted_power = torch.eye(lifted_dim, device=koopman.A.device)
    lifted_action = torch.zeros(
        lifted_dim, horizon * action_dim, device=koopman.A.device
    )
    state_rows = []
    action_rows = []
    for step in range(horizon):
        lifted_power = koopman.A @ lifted_power
        lifted_action = koopman.A @ lifted_action
        lifted_action[:, step * action_dim : (step + 1) * action_dim] += koopman.B
        state_rows.append(koopman.C @ lifted_power)
        action_rows.append(koopman.C @ lifted_action)
    return torch.cat(state_rows, dim=0), torch.cat(action_rows, dim=0)


class FeedforwardResidualKMPCTanhGaussianActor(nn.Module):
    """Optimize V in U=U_ff(phase)+V using a residual-response QP cost."""

    def __init__(
        self,
        koopman: nn.Module,
        feedforward: FrozenCirclePhaseFeedforward,
        *,
        horizon: int = 5,
        solver_iterations: int = 20,
        hidden_dim: int = 256,
        hidden_layers: int = 2,
        residual_limit: float = 0.1,
        physical_action_limit: float = 0.3,
        log_std_min: float = LOG_STD_MIN,
        log_std_init: float = -3.5,
        log_std_max: float = -3.0,
    ) -> None:
        super().__init__()
        self.koopman = koopman
        self.horizon = int(horizon)
        self.solver_iterations = int(solver_iterations)
        self.hidden_layers = int(hidden_layers)
        self.action_limit = float(residual_limit)
        self.physical_action_limit = float(physical_action_limit)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        if self.horizon < 1 or self.solver_iterations < 1 or self.hidden_layers < 1:
            raise ValueError("Residual KMPC dimensions must be positive")
        if not 0 < self.action_limit <= self.physical_action_limit:
            raise ValueError("Residual limit must not exceed the physical limit")
        if not self.log_std_min <= log_std_init <= self.log_std_max <= LOG_STD_MAX:
            raise ValueError("Residual KMPC log-std bounds are invalid")
        if koopman.state_dim != 47 or koopman.action_dim != 18:
            raise ValueError("Residual KMPC requires the implicit ManiSoft model")

        physical_dim = int(koopman.state_dim)
        action_dim = int(koopman.action_dim)
        output_dim = 2 * self.horizon * (physical_dim + action_dim)
        layers: list[nn.Module] = []
        input_dim = int(koopman.lifted_dim)
        for _ in range(self.hidden_layers):
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.GELU()))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, output_dim))
        self.controller = nn.Sequential(*layers)
        nn.init.zeros_(self.controller[-1].weight)
        nn.init.zeros_(self.controller[-1].bias)
        self.log_std = nn.Parameter(
            torch.full((action_dim,), float(log_std_init))
        )

        state_map, action_map = _condense_dynamics(koopman, self.horizon)
        self.register_buffer("state_map", state_map)
        self.register_buffer("action_map", action_map)
        weight = feedforward.weight
        gain = weight[:-1] / feedforward.feature_scale[:, None]
        bias = weight[-1] - (feedforward.feature_mean / feedforward.feature_scale) @ weight[:-1]
        self.register_buffer("feedforward_gain", torch.as_tensor(gain, dtype=torch.float32))
        self.register_buffer("feedforward_bias", torch.as_tensor(bias, dtype=torch.float32))
        angle = 2.0 * math.pi / 1000.0
        self.register_buffer(
            "phase_rotation",
            torch.tensor(
                [[math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle)]],
                dtype=torch.float32,
            ),
        )

    def _phase8(self, fundamental: torch.Tensor) -> torch.Tensor:
        sine = fundamental[..., 0]
        cosine = fundamental[..., 1]
        current_sine = sine
        current_cosine = cosine
        components = []
        for harmonic in range(1, 9):
            components.extend((current_sine, current_cosine))
            if harmonic < 8:
                following_sine = current_sine * cosine + current_cosine * sine
                following_cosine = current_cosine * cosine - current_sine * sine
                current_sine, current_cosine = following_sine, following_cosine
        return torch.stack(components, dim=-1)

    def feedforward_plan(self, lifted_state: torch.Tensor) -> torch.Tensor:
        # The implicit-phase Koopman adapter appends [sin(phi), cos(phi)] as
        # the last two lifted coordinates and advances them with an exact rotation.
        phase = lifted_state[..., -2:]
        result = []
        for _ in range(self.horizon):
            features = self._phase8(phase)
            result.append(F.linear(features, self.feedforward_gain.T, self.feedforward_bias))
            phase = F.linear(phase, self.phase_rotation)
        return torch.stack(result, dim=-2)

    def residual_bounds(
        self, feedforward_plan: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

    def plan(
        self,
        lifted_state: torch.Tensor,
        previous_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del previous_action
        batch_shape = lifted_state.shape[:-1]
        physical_dim = int(self.koopman.state_dim)
        action_dim = int(self.koopman.action_dim)
        augmented_dim = physical_dim + action_dim
        raw = self.controller(lifted_state).reshape(
            *batch_shape, 2, self.horizon, augmented_dim
        )
        raw_quadratic = torch.tanh(raw[..., 0, :, :])
        centered = raw_quadratic - raw_quadratic.mean(dim=-1, keepdim=True)
        quadratic = torch.exp(1.5 * centered)
        linear = 10.0 * torch.tanh(raw[..., 1, :, :])
        q_state = quadratic[..., :physical_dim].reshape(
            *batch_shape, self.horizon * physical_dim
        )
        q_residual = quadratic[..., physical_dim:].reshape(
            *batch_shape, self.horizon * action_dim
        )
        p_state = linear[..., :physical_dim].reshape(
            *batch_shape, self.horizon * physical_dim
        )
        p_residual = linear[..., physical_dim:].reshape(
            *batch_shape, self.horizon * action_dim
        )

        # X-X_ff = action_map @ V. The absolute-state free response and the
        # feedforward forcing cancel from the learned residual-response cost.
        weighted_map = self.action_map * q_state.unsqueeze(-1)
        hessian = self.action_map.T @ weighted_map
        hessian = hessian + torch.diag_embed(q_residual)
        qp_linear = torch.einsum(
            "...p,pi->...i", p_state, self.action_map
        ) + p_residual
        lipschitz = hessian.abs().sum(dim=-1).amax(dim=-1)
        step_size = 0.95 / (lipschitz + 1e-6)

        feedforward = self.feedforward_plan(lifted_state)
        lower, upper = self.residual_bounds(feedforward)
        lower_flat = lower.reshape(*batch_shape, self.horizon * action_dim)
        upper_flat = upper.reshape(*batch_shape, self.horizon * action_dim)
        current = torch.zeros_like(qp_linear)
        extrapolated = current.clone()
        momentum = 1.0
        for _ in range(self.solver_iterations):
            gradient = torch.einsum("...ij,...j->...i", hessian, extrapolated)
            following = torch.maximum(
                torch.minimum(
                    extrapolated
                    - step_size.unsqueeze(-1) * (gradient + qp_linear),
                    upper_flat,
                ),
                lower_flat,
            )
            next_momentum = 0.5 * (
                1.0 + math.sqrt(1.0 + 4.0 * momentum * momentum)
            )
            extrapolated = following + ((momentum - 1.0) / next_momentum) * (
                following - current
            )
            current = following
            momentum = next_momentum
        return current.reshape(*batch_shape, self.horizon, action_dim)

    def distribution(
        self,
        lifted_state: torch.Tensor,
        previous_action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        plan = self.plan(lifted_state, previous_action=previous_action)
        feedforward = self.feedforward_plan(lifted_state)[..., 0, :]
        lower, upper = self.residual_bounds(feedforward)
        midpoint = 0.5 * (lower + upper)
        half_range = 0.5 * (upper - lower)
        first = plan[..., 0, :].clamp(lower, upper)
        location = atanh_clipped((first - midpoint) / half_range)
        first = midpoint + half_range * torch.tanh(location)
        plan = plan.clone()
        plan[..., 0, :] = first
        log_std = self.log_std.clamp(
            self.log_std_min, self.log_std_max
        ).expand_as(location)
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        location, log_std, plan = self.distribution(
            lifted_state, previous_action=previous_action
        )
        feedforward = self.feedforward_plan(lifted_state)[..., 0, :]
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
                dtype=expanded_location.dtype,
                device=expanded_location.device,
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
        log_prob = (
            normal_log_prob - correction - torch.log(half_range)
        ).sum(dim=-1)
    return action, log_prob, plan if return_plan else None

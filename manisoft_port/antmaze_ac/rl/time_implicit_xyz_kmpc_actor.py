"""No-xref, nominal-feedforward-centred xyz Q,p-KMPC actor.

Only the three tracked node positions receive learned state-cost weights and
centres.  Linear/angular velocities retain a fixed, nominal-rollout-centred
regularizer; all remaining physical coordinates have zero state cost.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .time_structured_qp_kmpc_actor import TimeStructuredQpKMPCTanhGaussianActor


XYZ_INDICES = (0, 1, 2, 15, 16, 17, 30, 31, 32)
VELOCITY_INDICES = (
    9, 10, 11, 12, 13, 14,
    24, 25, 26, 27, 28, 29,
    39, 40, 41, 42, 43, 44,
)


class TimeImplicitXyzQpKMPCTanhGaussianActor(
    TimeStructuredQpKMPCTanhGaussianActor
):
    """Stage-wise ``[z,tau] -> Q_xyz,D_xyz,d_action -> KMPC`` actor."""

    def __init__(
        self,
        koopman: nn.Module,
        feedforward,
        *,
        horizon: int = 5,
        solver_iterations: int = 5,
        hidden_dim: int = 256,
        hidden_layers: int = 2,
        residual_limit: float = 0.5,
        physical_action_limit: float = 0.5,
        xyz_cost_scale: float = 1.0,
        velocity_cost_scale: float = 0.05,
        residual_cost_scale: float = 10_000.0,
        quadratic_log_scale: float = 1.5,
        q_log_upper_bound: float = 1.5,
        action_cost_center_limit: float = 0.01,
        d_xyz_scale_m: np.ndarray,
        log_std_init: float = -3.5,
        log_std_max: float = -3.0,
    ) -> None:
        # The parent supplies the validated residual box, differentiable QP
        # solver and frozen Koopman condensed maps.  Its dummy zero reference
        # is immediately removed and is never used by this objective.
        super().__init__(
            koopman,
            feedforward,
            np.zeros((1001, 45), dtype=np.float32),
            structure="k10v2",
            horizon=horizon,
            solver_iterations=solver_iterations,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            residual_limit=residual_limit,
            physical_action_limit=physical_action_limit,
            state_cost_scale=xyz_cost_scale,
            residual_cost_scale=residual_cost_scale,
            quadratic_log_scale=quadratic_log_scale,
            action_cost_center_limit=action_cost_center_limit,
            log_std_init=log_std_init,
            log_std_max=log_std_max,
        )
        del self._buffers["xref_table"]
        self.structure = "implicit_xyz_nominal_ff"
        self.explicit_xref = False
        self.shared_horizon = False
        self.output_groups = self.horizon
        self.terminal_enabled = False
        self.state_cost_gate_enabled = False
        self.state_p_enabled = True
        self.action_p_enabled = True
        self.adaptive_state_q = True
        self.adaptive_action_q = False
        self.frozen_cost_map = False

        self.xyz_dim = len(XYZ_INDICES)
        self.velocity_dim = len(VELOCITY_INDICES)
        self.q_log_lower_bound = -float(quadratic_log_scale)
        self.q_log_upper_bound = float(q_log_upper_bound)
        if self.q_log_upper_bound < float(quadratic_log_scale):
            raise ValueError("q_log_upper_bound must be at least the baseline 1.5")
        q_count = self.horizon * self.xyz_dim
        d_count = self.horizon * self.xyz_dim
        action_count = self.horizon * 18
        output_dim = q_count + d_count + action_count
        layers: list[nn.Module] = []
        input_dim = int(koopman.lifted_dim)
        for _ in range(self.hidden_layers):
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.GELU()))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, output_dim))
        self.controller = nn.Sequential(*layers)
        nn.init.zeros_(self.controller[-1].weight)
        nn.init.zeros_(self.controller[-1].bias)
        self.state_q_output_slice = slice(0, q_count)
        self.state_p_output_slice = slice(q_count, q_count + d_count)
        self.action_p_output_slice = slice(q_count + d_count, output_dim)

        device = koopman.A.device
        self.register_buffer(
            "xyz_indices", torch.as_tensor(XYZ_INDICES, dtype=torch.long, device=device)
        )
        self.register_buffer(
            "velocity_indices",
            torch.as_tensor(VELOCITY_INDICES, dtype=torch.long, device=device),
        )
        physical_scale = koopman.physical_scale.index_select(0, self.xyz_indices)
        scale_m = torch.as_tensor(d_xyz_scale_m, dtype=torch.float32, device=device)
        if scale_m.shape != (self.xyz_dim,) or not torch.isfinite(scale_m).all():
            raise ValueError("d_xyz_scale_m must contain nine finite values")
        if torch.any(scale_m <= 0):
            raise ValueError("d_xyz_scale_m must be strictly positive")
        self.register_buffer("d_xyz_scale_m", scale_m)
        self.register_buffer("d_xyz_scale_normalized", scale_m / physical_scale)
        self.register_buffer(
            "base_xyz_q", torch.full((self.xyz_dim,), float(xyz_cost_scale), device=device)
        )
        self.register_buffer(
            "fixed_velocity_q",
            torch.full((self.velocity_dim,), float(velocity_cost_scale), device=device),
        )

    def _raw_terms(self, lifted_state: torch.Tensor, raw_override=None):
        raw = self.controller(lifted_state) if raw_override is None else raw_override
        batch = lifted_state.shape[:-1]
        q_right = self.horizon * self.xyz_dim
        d_right = q_right + self.horizon * self.xyz_dim
        raw_q = raw[..., :q_right].reshape(*batch, self.horizon, self.xyz_dim)
        raw_d = raw[..., q_right:d_right].reshape(*batch, self.horizon, self.xyz_dim)
        raw_action = raw[..., d_right:].reshape(*batch, self.horizon, 18)
        return raw_q, raw_d, raw_action

    def _delta_q(self, raw_q: torch.Tensor) -> torch.Tensor:
        """Asymmetric smooth bound with unchanged -1.5 side and zero slope.

        At the baseline upper bound this is exactly ``1.5*tanh(raw)``.  For a
        wider positive side the derivative at zero remains 1.5, isolating the
        authority ceiling from local learning-rate/gradient-scale changes.
        """
        baseline = -self.q_log_lower_bound
        positive = self.q_log_upper_bound * torch.tanh(
            (baseline / self.q_log_upper_bound) * raw_q
        )
        negative = baseline * torch.tanh(raw_q)
        return torch.where(raw_q >= 0.0, positive, negative)

    def feedforward_plan(self, lifted_state: torch.Tensor) -> torch.Tensor:
        index = self._time_index(lifted_state)
        offset = torch.arange(self.horizon, device=index.device)
        action_index = (index.unsqueeze(-1) + offset).clamp_max(999)
        return self.feedforward_table[action_index]

    def nominal_rollout(self, lifted_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feedforward = self.feedforward_plan(lifted_state)
        batch = lifted_state.shape[:-1]
        ff_flat = feedforward.reshape(*batch, self.horizon * 18)
        nominal = F.linear(lifted_state, self.state_map) + F.linear(
            ff_flat, self.action_map
        )
        return feedforward, nominal.reshape(*batch, self.horizon, 45)

    def reference_plans(self, lifted_state: torch.Tensor):
        # Compatibility contract: callers only use item 0 for the FF box.
        # Item 1 is the model-consistent nominal rollout, never an x_ref table.
        return self.nominal_rollout(lifted_state)

    def cost_terms(self, lifted_state: torch.Tensor, *, raw_override=None):
        raw_q, raw_d, raw_action = self._raw_terms(lifted_state, raw_override)
        batch = lifted_state.shape[:-1]
        q_xyz = self.base_xyz_q * torch.exp(self._delta_q(raw_q))
        delta_d_xyz = self.d_xyz_scale_normalized * torch.tanh(raw_d)
        d_action = self.action_cost_center_limit * torch.tanh(raw_action)
        _, nominal = self.nominal_rollout(lifted_state)

        q_state = torch.zeros(
            *batch, self.horizon, 45, device=lifted_state.device, dtype=lifted_state.dtype
        )
        q_state[..., self.xyz_indices] = q_xyz
        q_state[..., self.velocity_indices] = self.fixed_velocity_q
        center = nominal.clone()
        center[..., self.xyz_indices] = (
            nominal[..., self.xyz_indices] + delta_d_xyz
        )
        p_state = -q_state * center
        q_action = self.base_action_q.expand(*batch, self.horizon, 18)
        p_action = -q_action * d_action
        return q_state, q_action, p_state, p_action

    def state_cost_gate(self, lifted_state: torch.Tensor) -> torch.Tensor:
        return torch.ones(
            *lifted_state.shape[:-1], self.horizon, 1,
            device=lifted_state.device, dtype=lifted_state.dtype,
        )

    def implicit_xyz_diagnostics(self, lifted_state: torch.Tensor) -> dict[str, torch.Tensor]:
        raw_q, raw_d, raw_action = self._raw_terms(lifted_state)
        delta_q = self._delta_q(raw_q)
        q_xyz = self.base_xyz_q * torch.exp(delta_q)
        delta = self.d_xyz_scale_normalized * torch.tanh(raw_d)
        delta_physical = delta * self.koopman.physical_scale[self.xyz_indices]
        d_action = self.action_cost_center_limit * torch.tanh(raw_action)
        _, nominal = self.nominal_rollout(lifted_state)
        d_nom_xyz = nominal[..., self.xyz_indices]
        d_xyz = d_nom_xyz + delta
        components = self._qp_components(lifted_state)
        solution = self._solve(*components[:4])
        predicted = nominal.reshape(*lifted_state.shape[:-1], self.horizon * 45)
        predicted = predicted + F.linear(solution, self.action_map)
        predicted = predicted.reshape(*lifted_state.shape[:-1], self.horizon, 45)
        velocity_error = predicted[..., self.velocity_indices] - nominal[..., self.velocity_indices]
        velocity_cost = 0.5 * (self.fixed_velocity_q * velocity_error.square()).sum(dim=-1)
        xyz_error = predicted[..., self.xyz_indices] - d_xyz
        xyz_cost = 0.5 * (q_xyz * xyz_error.square()).sum(dim=-1)
        action_error = solution.reshape(
            *lifted_state.shape[:-1], self.horizon, 18
        ) - d_action
        action_cost = 0.5 * (self.base_action_q * action_error.square()).sum(dim=-1)
        full_center = torch.zeros_like(nominal)
        full_center[..., self.xyz_indices] = d_xyz
        full_center[..., self.velocity_indices] = nominal[..., self.velocity_indices]
        current_normalized = F.linear(lifted_state, self.koopman.C[:45])
        return {
            "d_nom_xyz": d_nom_xyz,
            "delta_d_xyz": delta,
            "delta_d_xyz_physical_m": delta_physical,
            "d_xyz": d_xyz,
            "delta_d_xyz_utilization": delta / self.d_xyz_scale_normalized,
            "q_xyz_over_base": q_xyz / self.base_xyz_q,
            "delta_q_xyz": delta_q,
            "delta_q_upper_margin": self.q_log_upper_bound - delta_q,
            "d_action": d_action,
            "current_xyz": current_normalized[..., self.xyz_indices],
            "predicted_xyz_error_to_d": predicted[..., self.xyz_indices] - d_xyz,
            "xyz_cost_contribution": xyz_cost,
            "action_cost_contribution": action_cost,
            "velocity_error_to_nominal": velocity_error,
            "velocity_cost_contribution": velocity_cost,
            "d_state_full": full_center,
        }

    def zero_update_sanity(self, lifted_state: torch.Tensor) -> dict[str, object]:
        if lifted_state.ndim != 2 or lifted_state.shape[0] < 32:
            raise ValueError("zero-update sanity requires at least 32 replay states")
        lifted_state = lifted_state[:32]
        with torch.no_grad():
            plan = self.plan(lifted_state)
            hessian = self._qp_components(lifted_state)[0]
            eigenvalues = torch.linalg.eigvalsh(hessian)
            condition = eigenvalues[..., -1] / eigenvalues[..., 0].clamp_min(1e-12)
        sample = lifted_state[:1].detach()
        raw0 = self.controller(sample).detach().requires_grad_(True)

        def first_action(raw: torch.Tensor) -> torch.Tensor:
            return self.plan(sample, raw_override=raw)[0, 0]

        jacobian = torch.autograd.functional.jacobian(first_action, raw0)[..., 0, :]
        d_slice = self.state_p_output_slice
        a_slice = self.action_p_output_slice
        assert d_slice is not None and a_slice is not None
        d_gradient = jacobian[:, d_slice].norm()
        action_gradient = jacobian[:, a_slice].norm()
        result = {
            "samples": 32,
            "first_residual_p95_abs": float(torch.quantile(plan[..., 0, :].abs().reshape(-1), 0.95).cpu()),
            "first_residual_max_abs": float(plan[..., 0, :].abs().amax().cpu()),
            "full_delta_u_norm_max": float(plan.reshape(32, -1).norm(dim=-1).amax().cpu()),
            "gradient_first_action_wrt_delta_d_xyz_fro": float(d_gradient.cpu()),
            "gradient_first_action_wrt_d_action_fro": float(action_gradient.cpu()),
            "qp_min_eigenvalue": float(eigenvalues.min().cpu()),
            "qp_condition_p95": float(torch.quantile(condition, 0.95).cpu()),
            "qp_condition_max": float(condition.max().cpu()),
            "xyz_indices": list(XYZ_INDICES),
            "velocity_indices": list(VELOCITY_INDICES),
            "d_xyz_scale_m": self.d_xyz_scale_m.detach().cpu().tolist(),
            "fixed_velocity_q": self.fixed_velocity_q.detach().cpu().tolist(),
            "actor_output_dim": int(self.controller[-1].out_features),
            "q_log_lower_bound": self.q_log_lower_bound,
            "q_log_upper_bound": self.q_log_upper_bound,
        }
        if result["first_residual_max_abs"] > 1e-7 or result["full_delta_u_norm_max"] > 1e-6:
            raise RuntimeError(f"zero-head does not preserve feedforward: {result}")
        if result["gradient_first_action_wrt_delta_d_xyz_fro"] <= 1e-12:
            raise RuntimeError("Delta_D_xyz has zero first-action gradient")
        if result["gradient_first_action_wrt_d_action_fro"] <= 1e-12:
            raise RuntimeError("d_action has zero first-action gradient")
        return result

    def selective_awac_gradient_diagnostics(self) -> dict[str, float]:
        final = self.controller[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("Controller output head must be linear")

        def norm(parameters) -> float:
            total = torch.zeros((), device=final.weight.device)
            for parameter in parameters:
                if parameter.grad is not None:
                    total = total + parameter.grad.detach().square().sum()
            return float(total.sqrt().cpu())

        def sliced(output_slice: slice) -> float:
            total = torch.zeros((), device=final.weight.device)
            if final.weight.grad is not None:
                total = total + final.weight.grad[output_slice].detach().square().sum()
            if final.bias.grad is not None:
                total = total + final.bias.grad[output_slice].detach().square().sum()
            return float(total.sqrt().cpu())

        assert self.state_q_output_slice is not None
        assert self.state_p_output_slice is not None
        assert self.action_p_output_slice is not None
        return {
            "shared_trunk_grad_norm": norm(self.controller[:-1].parameters()),
            "q_state_head_grad_norm": sliced(self.state_q_output_slice),
            "d_xyz_head_grad_norm": sliced(self.state_p_output_slice),
            "action_p_head_grad_norm": sliced(self.action_p_output_slice),
        }

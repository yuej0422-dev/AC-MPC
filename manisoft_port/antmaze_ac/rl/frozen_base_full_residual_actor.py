"""Frozen E7 cost map plus a full-capacity online residual cost-map actor."""

from __future__ import annotations

import hashlib
import copy
from collections.abc import Iterable, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .time_implicit_xyz_kmpc_actor import (
    TimeImplicitXyzQpKMPCTanhGaussianActor,
)


def _state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        encoded = key.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


class FrozenBaseFullResidualImplicitXyzActor(
    TimeImplicitXyzQpKMPCTanhGaussianActor
):
    """Compose a frozen E7 actor with a same-capacity residual actor.

    The inherited ``controller`` is the online residual network.  A separate
    frozen ``base_controller`` owns the source E7 cost map.  Residuals are
    composed after decoding the source Q/D/action-centre values, so source
    tanh saturation does not suppress the online gradient.
    """

    CHANNELS = {"none", "D", "DQ", "DQa"}

    def __init__(
        self,
        *args,
        residual_channels: str = "DQ",
        d_residual_ratio: float = 0.20,
        q_log_residual_limit: float = 0.20,
        action_center_residual_limit: float = 0.001,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if residual_channels not in self.CHANNELS:
            raise ValueError(
                f"residual_channels must be one of {sorted(self.CHANNELS)}"
            )
        if not 0.0 < d_residual_ratio <= 1.0:
            raise ValueError("D residual ratio must lie in (0,1]")
        if not 0.0 < q_log_residual_limit <= 1.0:
            raise ValueError("Q log residual limit must lie in (0,1]")
        if not 0.0 < action_center_residual_limit < self.action_cost_center_limit:
            raise ValueError("Action residual authority must be below base d_action")

        # Capture the exact legacy actor state schema before registering the
        # frozen source controller and experiment-only integrity buffers.
        self._source_actor_keys = tuple(nn.Module.state_dict(self).keys())
        # deepcopy retains the exact 78->256->256->180 architecture and is
        # much less brittle than reconstructing Sequential by index.
        self.base_controller = copy.deepcopy(self.controller).eval()
        for parameter in self.base_controller.parameters():
            parameter.requires_grad_(False)

        self.log_std.requires_grad_(False)
        self.residual_channels = residual_channels
        self.d_residual_ratio = float(d_residual_ratio)
        self.q_log_residual_limit = float(q_log_residual_limit)
        self.action_center_residual_limit = float(action_center_residual_limit)
        self.final_delta_d_limit_ratio = 1.20
        self.final_q_log_lower_bound = -1.70
        self.final_q_log_upper_bound = 2.00
        self.final_action_center_limit = (
            self.action_cost_center_limit + self.action_center_residual_limit
        )
        self.parameterization = "frozen_E7_plus_full_capacity_decoded_residual_v1"
        self.register_buffer(
            "base_source_sha256_bytes", torch.zeros(32, dtype=torch.uint8)
        )
        self.register_buffer("base_source_loaded", torch.tensor(False))
        self._residual_reference: dict[str, torch.Tensor] = {}
        object.__setattr__(self, "_reference_controller", None)
        self._zero_residual_head()
        self._capture_residual_reference()

    def _zero_residual_head(self) -> None:
        final = self.controller[-1]
        if not isinstance(final, nn.Linear) or final.out_features != 180:
            raise TypeError("Full residual actor requires a 256->180 linear head")
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def online_residual_parameters(self) -> Iterable[nn.Parameter]:
        """The only parameters authorized for the online actor optimizer."""
        return self.controller.parameters()

    def _source_style_state(self) -> dict[str, torch.Tensor]:
        current = nn.Module.state_dict(self)
        result: dict[str, torch.Tensor] = {}
        for key in self._source_actor_keys:
            if key.startswith("controller."):
                source_key = "base_controller." + key.removeprefix("controller.")
                result[key] = current[source_key]
            else:
                result[key] = current[key]
        return result

    def base_actor_sha256(self) -> str:
        return _state_sha256(self._source_style_state())

    def expected_base_actor_sha256(self) -> str:
        return bytes(self.base_source_sha256_bytes.cpu().tolist()).hex()

    def assert_base_actor_unchanged(self) -> dict[str, object]:
        if not bool(self.base_source_loaded.item()):
            raise RuntimeError("Frozen E7 source actor has not been loaded")
        actual = self.base_actor_sha256()
        expected = self.expected_base_actor_sha256()
        if actual != expected:
            raise RuntimeError(
                f"Frozen base actor changed: expected {expected}, observed {actual}"
            )
        return {
            "base_actor_parameter_sha256": actual,
            "base_actor_source_sha256": expected,
            "base_actor_parameter_change": 0.0,
        }

    def load_offline_base_state_dict(
        self, source_state: Mapping[str, torch.Tensor]
    ) -> None:
        """Load a legacy E7 actor into the frozen branch only.

        The residual trunk copies E7's hidden layers while its output head is
        reset to exact zero.  Static Koopman/cost buffers and frozen log_std
        are copied from the source state, making the initial policy identical.
        """
        source_keys = tuple(source_state.keys())
        if set(source_keys) != set(self._source_actor_keys):
            missing = sorted(set(self._source_actor_keys) - set(source_keys))
            extra = sorted(set(source_keys) - set(self._source_actor_keys))
            raise ValueError(
                f"E7 actor state schema differs; missing={missing}, extra={extra}"
            )
        target = nn.Module.state_dict(self)
        with torch.no_grad():
            for key in self._source_actor_keys:
                source = source_state[key]
                destination_key = key
                if key.startswith("controller."):
                    destination_key = (
                        "base_controller." + key.removeprefix("controller.")
                    )
                destination = target[destination_key]
                if destination.shape != source.shape:
                    raise ValueError(
                        f"E7 actor tensor shape differs for {key}: "
                        f"{tuple(source.shape)} != {tuple(destination.shape)}"
                    )
                destination.copy_(source.to(destination.device, destination.dtype))

            # Copy only hidden layers into the online branch.  The output head
            # remains exactly zero, which makes decoded residuals exactly zero.
            for index in (0, 2):
                for suffix in ("weight", "bias"):
                    key = f"controller.{index}.{suffix}"
                    target[key].copy_(
                        source_state[key].to(target[key].device, target[key].dtype)
                    )
            self._zero_residual_head()
            source_hash = bytes.fromhex(_state_sha256(source_state))
            self.base_source_sha256_bytes.copy_(
                torch.tensor(list(source_hash), dtype=torch.uint8, device=self.base_source_sha256_bytes.device)
            )
            self.base_source_loaded.fill_(True)
        self.assert_base_actor_unchanged()
        self._capture_residual_reference()

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        if bool(self.base_source_loaded.item()):
            self.assert_base_actor_unchanged()
        self._capture_residual_reference()
        return result

    def _capture_residual_reference(self) -> None:
        self._residual_reference = {
            key: value.detach().cpu().clone()
            for key, value in self.controller.state_dict().items()
        }
        reference_controller = copy.deepcopy(self.controller).eval()
        for parameter in reference_controller.parameters():
            parameter.requires_grad_(False)
        # Keep the causal-reference network outside Module registration: it is
        # diagnostic-only, must not enter checkpoints, and must never enter
        # the actor optimizer. A continuation load recreates it exactly from
        # that continuation source before any new update.
        object.__setattr__(self, "_reference_controller", reference_controller)

    def residual_parameter_update_diagnostics(self) -> dict[str, float]:
        current = self.controller.state_dict()

        def norm(keys: list[str]) -> float:
            total = 0.0
            for key in keys:
                reference = self._residual_reference[key].to(
                    current[key].device, current[key].dtype
                )
                total += float((current[key] - reference).square().sum().cpu())
            return total**0.5

        trunk_keys = [key for key in current if not key.startswith("4.")]
        head_keys = [key for key in current if key.startswith("4.")]
        final = self.controller[-1]
        assert isinstance(final, nn.Linear)

        def head_slice_norm(output_slice: slice) -> float:
            weight_ref = self._residual_reference["4.weight"].to(final.weight.device)
            bias_ref = self._residual_reference["4.bias"].to(final.bias.device)
            return float(
                (
                    (final.weight[output_slice] - weight_ref[output_slice]).square().sum()
                    + (final.bias[output_slice] - bias_ref[output_slice]).square().sum()
                ).sqrt().cpu()
            )

        return {
            "online_residual_trunk_parameter_update_norm": norm(trunk_keys),
            "online_residual_head_parameter_update_norm": norm(head_keys),
            "online_residual_D_head_parameter_update_norm": head_slice_norm(
                self.state_p_output_slice
            ),
            "online_residual_Q_head_parameter_update_norm": head_slice_norm(
                self.state_q_output_slice
            ),
            "online_residual_action_head_parameter_update_norm": head_slice_norm(
                self.action_p_output_slice
            ),
        }

    @property
    def _enable_D(self) -> bool:
        return self.residual_channels in {"D", "DQ", "DQa"}

    @property
    def _enable_Q(self) -> bool:
        return self.residual_channels in {"DQ", "DQa"}

    @property
    def _enable_action(self) -> bool:
        return self.residual_channels == "DQa"

    def _decoded_components(
        self, lifted_state: torch.Tensor, *, raw_override: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        base_q_raw, base_d_raw, base_action_raw = self._raw_terms_from_controller(
            self.base_controller, lifted_state
        )
        online_raw = self.controller(lifted_state) if raw_override is None else raw_override
        residual_q_raw, residual_d_raw, residual_action_raw = self._split_raw(
            lifted_state, online_raw
        )

        q_off = self.base_xyz_q * torch.exp(self._delta_q(base_q_raw))
        delta_D_off = self.d_xyz_scale_normalized * torch.tanh(base_d_raw)
        d_action_off = self.action_cost_center_limit * torch.tanh(base_action_raw)

        delta_q = self.q_log_residual_limit * torch.tanh(residual_q_raw)
        delta_D = (
            self.d_residual_ratio
            * self.d_xyz_scale_normalized
            * torch.tanh(residual_d_raw)
        )
        delta_d_action = self.action_center_residual_limit * torch.tanh(
            residual_action_raw
        )
        if not self._enable_Q:
            delta_q = delta_q * 0.0
        if not self._enable_D:
            delta_D = delta_D * 0.0
        if not self._enable_action:
            delta_d_action = delta_d_action * 0.0

        q_final = q_off * torch.exp(delta_q)
        q_final = q_final.clamp(
            min=float(torch.exp(torch.tensor(self.final_q_log_lower_bound))),
            max=float(torch.exp(torch.tensor(self.final_q_log_upper_bound))),
        )
        delta_D_final = (delta_D_off + delta_D).clamp(
            -self.final_delta_d_limit_ratio * self.d_xyz_scale_normalized,
            self.final_delta_d_limit_ratio * self.d_xyz_scale_normalized,
        )
        d_action_final = (d_action_off + delta_d_action).clamp(
            -self.final_action_center_limit, self.final_action_center_limit
        )
        return {
            "q_off": q_off,
            "delta_D_off": delta_D_off,
            "d_action_off": d_action_off,
            "delta_q_online": delta_q,
            "delta_D_online": delta_D,
            "delta_d_action_online": delta_d_action,
            "q_final": q_final,
            "delta_D_final": delta_D_final,
            "d_action_final": d_action_final,
            "residual_q_raw": residual_q_raw,
            "residual_d_raw": residual_d_raw,
            "residual_action_raw": residual_action_raw,
        }

    def _split_raw(
        self, lifted_state: torch.Tensor, raw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = lifted_state.shape[:-1]
        q_right = self.horizon * self.xyz_dim
        d_right = q_right + self.horizon * self.xyz_dim
        return (
            raw[..., :q_right].reshape(*batch, self.horizon, self.xyz_dim),
            raw[..., q_right:d_right].reshape(*batch, self.horizon, self.xyz_dim),
            raw[..., d_right:].reshape(*batch, self.horizon, 18),
        )

    def _raw_terms_from_controller(
        self, controller: nn.Module, lifted_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._split_raw(lifted_state, controller(lifted_state))

    def _terms_from_decoded(
        self, lifted_state: torch.Tensor, decoded: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = lifted_state.shape[:-1]
        _, nominal = self.nominal_rollout(lifted_state)
        q_state = torch.zeros(
            *batch,
            self.horizon,
            45,
            device=lifted_state.device,
            dtype=lifted_state.dtype,
        )
        q_state[..., self.xyz_indices] = decoded["q_final"]
        q_state[..., self.velocity_indices] = self.fixed_velocity_q
        center = nominal.clone()
        center[..., self.xyz_indices] = (
            nominal[..., self.xyz_indices] + decoded["delta_D_final"]
        )
        p_state = -q_state * center
        q_action = self.base_action_q.expand(*batch, self.horizon, 18)
        p_action = -q_action * decoded["d_action_final"]
        return q_state, q_action, p_state, p_action

    def cost_terms(self, lifted_state: torch.Tensor, *, raw_override=None):
        return self._terms_from_decoded(
            lifted_state,
            self._decoded_components(lifted_state, raw_override=raw_override),
        )

    def base_cost_terms(self, lifted_state: torch.Tensor):
        decoded = self._decoded_components(lifted_state)
        decoded = dict(decoded)
        decoded["q_final"] = decoded["q_off"]
        decoded["delta_D_final"] = decoded["delta_D_off"]
        decoded["d_action_final"] = decoded["d_action_off"]
        return self._terms_from_decoded(lifted_state, decoded)

    def _plan_from_terms(self, lifted_state: torch.Tensor, terms) -> torch.Tensor:
        batch = lifted_state.shape[:-1]
        q_state, q_action, p_state, p_action = terms
        feedforward, nominal = self.nominal_rollout(lifted_state)
        free_state = nominal.reshape(*batch, self.horizon * 45)
        qs = q_state.reshape(*batch, self.horizon * 45)
        qa = q_action.reshape(*batch, self.horizon * 18)
        ps = p_state.reshape(*batch, self.horizon * 45)
        pa = p_action.reshape(*batch, self.horizon * 18)
        weighted_map = self.action_map * qs.unsqueeze(-1)
        hessian = self.action_map.T @ weighted_map + torch.diag_embed(qa)
        linear = torch.einsum(
            "...p,pi->...i", qs * free_state + ps, self.action_map
        ) + pa
        lower, upper = self.residual_bounds(feedforward)
        solution = self._solve(
            hessian,
            linear,
            lower.reshape(*batch, self.horizon * 18),
            upper.reshape(*batch, self.horizon * 18),
        )
        return solution.reshape(*batch, self.horizon, 18)

    def base_plan(self, lifted_state: torch.Tensor) -> torch.Tensor:
        return self._plan_from_terms(lifted_state, self.base_cost_terms(lifted_state))

    def validate_bootstrap_zero_residual(
        self, lifted_state: torch.Tensor
    ) -> dict[str, object]:
        if lifted_state.ndim != 2 or lifted_state.shape[0] < 32:
            raise ValueError("Full-residual sanity requires at least 32 states")
        lifted_state = lifted_state[:32]
        with torch.no_grad():
            raw = self.controller(lifted_state)
            decoded = self._decoded_components(lifted_state)
            final_terms = self.cost_terms(lifted_state)
            base_terms = self.base_cost_terms(lifted_state)
            final_plan = self.plan(lifted_state)
            base_plan = self.base_plan(lifted_state)
        term_difference = max(
            float((final - base).abs().max().cpu())
            for final, base in zip(final_terms, base_terms)
        )
        result: dict[str, object] = {
            "samples": 32,
            "residual_channels": self.residual_channels,
            "max_abs_raw_residual": float(raw.abs().max().cpu()),
            "max_abs_delta_D": float(decoded["delta_D_online"].abs().max().cpu()),
            "max_abs_delta_q": float(decoded["delta_q_online"].abs().max().cpu()),
            "max_abs_delta_d_action": float(
                decoded["delta_d_action_online"].abs().max().cpu()
            ),
            "max_abs_final_cost_term_difference": term_difference,
            "max_abs_delta_u_difference": float(
                (final_plan - base_plan).abs().max().cpu()
            ),
            **self.assert_base_actor_unchanged(),
        }
        if max(
            float(result["max_abs_raw_residual"]),
            float(result["max_abs_delta_D"]),
            float(result["max_abs_delta_q"]),
            float(result["max_abs_delta_d_action"]),
            float(result["max_abs_final_cost_term_difference"]),
            float(result["max_abs_delta_u_difference"]),
        ) > 1e-7:
            raise RuntimeError(f"Full residual actor does not preserve E7: {result}")
        return result

    def implicit_xyz_diagnostics(
        self, lifted_state: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        decoded = self._decoded_components(lifted_state)
        q_state, q_action, _, _ = self._terms_from_decoded(lifted_state, decoded)
        _, nominal = self.nominal_rollout(lifted_state)
        d_xyz = nominal[..., self.xyz_indices] + decoded["delta_D_final"]
        solution = self.plan(lifted_state)
        predicted = nominal.reshape(*lifted_state.shape[:-1], self.horizon * 45)
        predicted = predicted + F.linear(
            solution.reshape(*lifted_state.shape[:-1], self.horizon * 18),
            self.action_map,
        )
        predicted = predicted.reshape(*lifted_state.shape[:-1], self.horizon, 45)
        velocity_error = (
            predicted[..., self.velocity_indices]
            - nominal[..., self.velocity_indices]
        )
        xyz_error = predicted[..., self.xyz_indices] - d_xyz
        action_error = solution - decoded["d_action_final"]
        full_center = torch.zeros_like(nominal)
        full_center[..., self.xyz_indices] = d_xyz
        full_center[..., self.velocity_indices] = nominal[..., self.velocity_indices]

        reference_controller = self._reference_controller
        if reference_controller is None:
            raise RuntimeError("Residual reference controller is unavailable")
        reference_raw = reference_controller(lifted_state)
        reference_decoded = self._decoded_components(
            lifted_state, raw_override=reference_raw
        )
        reference_solution = self.plan(
            lifted_state, raw_override=reference_raw
        )
        feedforward = self.feedforward_plan(lifted_state)[..., 0, :]
        lower, upper = self.residual_bounds(feedforward)
        midpoint = 0.5 * (lower + upper)
        half_range = 0.5 * (upper - lower)
        final_location = torch.atanh(
            ((solution[..., 0, :] - midpoint) / half_range).clamp(-0.999, 0.999)
        )
        reference_location = torch.atanh(
            ((reference_solution[..., 0, :] - midpoint) / half_range).clamp(-0.999, 0.999)
        )
        std = self.log_std.clamp(self.log_std_min, self.log_std_max).exp()
        policy_kl = 0.5 * (
            (final_location - reference_location) / std
        ).square().sum(-1)
        return {
            "d_nom_xyz": nominal[..., self.xyz_indices],
            "delta_d_xyz": decoded["delta_D_final"],
            "delta_d_xyz_physical_m": (
                decoded["delta_D_final"]
                * self.koopman.physical_scale[self.xyz_indices]
            ),
            "d_xyz": d_xyz,
            "delta_d_xyz_utilization": (
                decoded["delta_D_final"] / self.d_xyz_scale_normalized
            ),
            "q_xyz_over_base": decoded["q_final"] / self.base_xyz_q,
            "delta_q_xyz": torch.log(decoded["q_final"] / self.base_xyz_q),
            "delta_q_upper_margin": (
                self.final_q_log_upper_bound
                - torch.log(decoded["q_final"] / self.base_xyz_q)
            ),
            "d_action": decoded["d_action_final"],
            "current_xyz": F.linear(lifted_state, self.koopman.C[:45])[
                ..., self.xyz_indices
            ],
            "predicted_xyz_error_to_d": xyz_error,
            "xyz_cost_contribution": 0.5
            * (decoded["q_final"] * xyz_error.square()).sum(dim=-1),
            "action_cost_contribution": 0.5
            * (q_action * action_error.square()).sum(dim=-1),
            "velocity_error_to_nominal": velocity_error,
            "velocity_cost_contribution": 0.5
            * (self.fixed_velocity_q * velocity_error.square()).sum(dim=-1),
            "d_state_full": full_center,
            "base_delta_D_xyz": decoded["delta_D_off"],
            "online_delta_D_xyz": decoded["delta_D_online"],
            "online_delta_D_utilization": decoded["delta_D_online"]
            / (self.d_residual_ratio * self.d_xyz_scale_normalized),
            "base_q_xyz_over_base": decoded["q_off"] / self.base_xyz_q,
            "online_delta_q_xyz": decoded["delta_q_online"],
            "q_final_over_q_off": decoded["q_final"] / decoded["q_off"],
            "base_d_action": decoded["d_action_off"],
            "online_delta_d_action": decoded["delta_d_action_online"],
            "online_delta_D_change_from_reference": (
                decoded["delta_D_online"]
                - reference_decoded["delta_D_online"]
            ),
            "online_delta_q_change_from_reference": (
                decoded["delta_q_online"]
                - reference_decoded["delta_q_online"]
            ),
            "q_final_change_ratio_from_reference": (
                decoded["q_final"] / reference_decoded["q_final"]
            ),
            "first_mpc_action_change_from_reference": (
                solution[..., 0, :] - reference_solution[..., 0, :]
            ),
            "policy_location_delta_from_shared": (
                final_location - reference_location
            ),
            "policy_kl_from_shared": policy_kl,
        }

    def selective_awac_gradient_diagnostics(self) -> dict[str, float]:
        final = self.controller[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("Residual output head must be linear")

        def norm(parameters) -> float:
            total = torch.zeros((), device=final.weight.device)
            for parameter in parameters:
                if parameter.grad is not None:
                    total = total + parameter.grad.detach().square().sum()
            return float(total.sqrt().cpu())

        def sliced(output_slice: slice) -> float:
            total = torch.zeros((), device=final.weight.device)
            if final.weight.grad is not None:
                total += final.weight.grad[output_slice].detach().square().sum()
            if final.bias.grad is not None:
                total += final.bias.grad[output_slice].detach().square().sum()
            return float(total.sqrt().cpu())

        return {
            "online_residual_shared_trunk_grad_norm": norm(
                self.controller[:-1].parameters()
            ),
            "online_residual_Q_head_grad_norm": sliced(self.state_q_output_slice),
            "online_residual_D_head_grad_norm": sliced(self.state_p_output_slice),
            "online_residual_action_head_grad_norm": sliced(
                self.action_p_output_slice
            ),
        }

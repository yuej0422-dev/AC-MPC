from __future__ import annotations

import numpy as np
import torch
from torch import nn

from antmaze_ac.rl.frozen_base_full_residual_actor import (
    FrozenBaseFullResidualImplicitXyzActor,
)
from antmaze_ac.rl.time_implicit_xyz_kmpc_actor import (
    TimeImplicitXyzQpKMPCTanhGaussianActor,
)


class DummyKoopman(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.state_dim = 46
        self.action_dim = 18
        self.lifted_dim = 46
        self.register_buffer("A", torch.eye(46))
        self.register_buffer("B", 0.01 * torch.randn(46, 18))
        self.register_buffer("C", torch.eye(46))
        self.register_buffer("physical_center", torch.zeros(45))
        self.register_buffer("physical_scale", torch.ones(45))


class DummyFeedforward:
    def action(self, phase, episode_steps):
        del episode_steps
        phase = np.asarray(phase)
        return np.zeros((*phase.shape, 18), dtype=np.float32)


def make_actor(cls, **kwargs):
    return cls(
        DummyKoopman(),
        DummyFeedforward(),
        hidden_dim=32,
        hidden_layers=2,
        solver_iterations=2,
        d_xyz_scale_m=np.full(9, 0.025, dtype=np.float32),
        q_log_upper_bound=1.8,
        action_cost_center_limit=0.01,
        **kwargs,
    )


def test_zero_residual_strictly_preserves_source_actor() -> None:
    torch.manual_seed(17)
    source = make_actor(TimeImplicitXyzQpKMPCTanhGaussianActor)
    with torch.no_grad():
        source.controller[-1].weight.normal_(std=0.02)
        source.controller[-1].bias.normal_(std=0.02)
    target = make_actor(
        FrozenBaseFullResidualImplicitXyzActor,
        residual_channels="DQa",
    )
    target.load_offline_base_state_dict(source.state_dict())
    state = torch.randn(32, 46)
    source_plan = source.plan(state)
    target_plan = target.plan(state)
    torch.testing.assert_close(target_plan, source_plan, atol=0.0, rtol=0.0)
    sanity = target.validate_bootstrap_zero_residual(state)
    assert sanity["max_abs_raw_residual"] == 0.0
    assert sanity["max_abs_delta_u_difference"] == 0.0
    assert sanity["base_actor_parameter_change"] == 0.0


def test_optimizer_contract_exposes_only_online_controller() -> None:
    target = make_actor(
        FrozenBaseFullResidualImplicitXyzActor,
        residual_channels="DQ",
    )
    online = {id(parameter) for parameter in target.online_residual_parameters()}
    base = {id(parameter) for parameter in target.base_controller.parameters()}
    assert online
    assert online.isdisjoint(base)
    assert id(target.log_std) not in online
    assert all(parameter.requires_grad for parameter in target.controller.parameters())
    assert not any(
        parameter.requires_grad for parameter in target.base_controller.parameters()
    )
    assert not target.log_std.requires_grad


def test_disabled_channels_have_exactly_zero_decoded_residual() -> None:
    source = make_actor(TimeImplicitXyzQpKMPCTanhGaussianActor)
    target = make_actor(
        FrozenBaseFullResidualImplicitXyzActor,
        residual_channels="D",
    )
    target.load_offline_base_state_dict(source.state_dict())
    with torch.no_grad():
        target.controller[-1].weight.normal_(std=0.1)
        target.controller[-1].bias.normal_(std=0.1)
    decoded = target._decoded_components(torch.randn(4, 46))
    assert torch.count_nonzero(decoded["delta_D_online"]) > 0
    assert torch.count_nonzero(decoded["delta_q_online"]) == 0
    assert torch.count_nonzero(decoded["delta_d_action_online"]) == 0

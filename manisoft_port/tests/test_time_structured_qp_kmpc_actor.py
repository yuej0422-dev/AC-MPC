from __future__ import annotations

import numpy as np
import torch
from torch import nn

from antmaze_ac.rl.time_structured_qp_kmpc_actor import (
    TimeStructuredQpKMPCTanhGaussianActor,
)


class DummyKoopman(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.state_dim = 46
        self.action_dim = 18
        self.lifted_dim = 46
        self.register_buffer("A", torch.eye(46))
        self.register_buffer("B", torch.zeros(46, 18))
        self.register_buffer("C", torch.eye(46))
        self.register_buffer("physical_center", torch.zeros(45))
        self.register_buffer("physical_scale", torch.ones(45))


class DummyFeedforward:
    def action(self, phase, episode_steps):
        del episode_steps
        phase_array = np.asarray(phase)
        return np.zeros((*phase_array.shape, 18), dtype=np.float32)


def actor(structure: str, **kwargs) -> TimeStructuredQpKMPCTanhGaussianActor:
    xref = np.linspace(-0.2, 0.3, 1001, dtype=np.float32)[:, None]
    xref = np.repeat(xref, 45, axis=1)
    return TimeStructuredQpKMPCTanhGaussianActor(
        DummyKoopman(),
        DummyFeedforward(),
        xref,
        structure=structure,
        hidden_dim=16,
        hidden_layers=1,
        solver_iterations=2,
        **kwargs,
    )


def test_all_structures_have_neutral_zero_head() -> None:
    state = torch.zeros(2, 46)
    for index in range(10):
        model = actor(f"k{index}")
        q_state, q_action, p_state, p_action = model.cost_terms(state)
        torch.testing.assert_close(q_state, torch.ones_like(q_state))
        torch.testing.assert_close(q_action, torch.full_like(q_action, 10_000.0))
        torch.testing.assert_close(p_state, torch.zeros_like(p_state))
        torch.testing.assert_close(p_action, torch.zeros_like(p_action))


def test_shared_and_stagewise_shapes_are_identical_at_solver_boundary() -> None:
    state = torch.zeros(3, 46)
    for structure in (
        "k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8", "k9", "k10v2", "k10p"
    ):
        model = actor(structure)
        plan = model.plan(state)
        assert plan.shape == (3, 5, 18)
        assert torch.isfinite(plan).all()
        q_state, q_action, p_state, p_action = model.cost_terms(state)
        assert q_state.shape == p_state.shape == (3, 5, 45)
        assert q_action.shape == p_action.shape == (3, 5, 18)


def test_k10v2_and_k10p_use_stagewise_implicit_center_plans() -> None:
    state = torch.randn(4, 46)
    for structure in ("k10v2", "k10p"):
        model = actor(structure)
        assert not model.explicit_xref
        assert not model.shared_horizon
        assert model.output_groups == 5
        torch.testing.assert_close(model.plan(state), torch.zeros(4, 5, 18))


def test_k8_q_and_r_are_independently_centered() -> None:
    model = actor("k8")
    state = torch.zeros(1, 46)
    raw = torch.zeros(1, model.controller[-1].out_features)
    raw[..., :45] = 0.4
    raw[..., 45:63] = -0.7
    q_state, q_action, _, _ = model.cost_terms(state, raw_override=raw)
    torch.testing.assert_close(q_state, torch.ones_like(q_state))
    torch.testing.assert_close(q_action, torch.full_like(q_action, 10_000.0))


def test_k9_state_center_bounds_come_from_reference_envelope() -> None:
    model = actor("k9")
    torch.testing.assert_close(
        model.state_cost_center_limit,
        torch.full((45,), 0.3),
        atol=1e-6,
        rtol=0.0,
    )


def test_k10p_keeps_q_fixed_and_warm_initializes_only_state_center() -> None:
    torch.manual_seed(7)
    model = actor("k10p")
    state = torch.randn(64, 46)
    target = 0.1 * torch.tanh(state[:, :45])
    before_q, before_r, _, before_action_p = model.cost_terms(state)
    diagnostics = model.warm_initialize_state_center(state, target, ridge=1e-3)
    after_q, after_r, after_state_p, after_action_p = model.cost_terms(state)

    torch.testing.assert_close(after_q, before_q)
    torch.testing.assert_close(after_r, before_r)
    torch.testing.assert_close(after_action_p, before_action_p)
    assert torch.count_nonzero(after_state_p) > 0
    assert diagnostics["mse_normalized"] < float(target.square().mean())


def test_k10v2_warm_init_leaves_adaptive_q_at_base() -> None:
    model = actor("k10v2")
    state = torch.randn(32, 46)
    target = 0.05 * torch.tanh(state[:, :45])
    model.warm_initialize_state_center(state, target)
    q_state, q_action, _, p_action = model.cost_terms(state)
    torch.testing.assert_close(q_state, torch.ones_like(q_state))
    torch.testing.assert_close(q_action, torch.full_like(q_action, 10_000.0))
    torch.testing.assert_close(p_action, torch.zeros_like(p_action))


def test_policy_preserving_action_headroom_keeps_forward_and_old_gradient() -> None:
    torch.manual_seed(11)
    baseline = actor("k6", action_cost_center_limit=0.01)
    expanded = actor(
        "k6",
        action_cost_center_limit=0.01,
        action_headroom_limit=0.02,
    )
    source = baseline.state_dict()
    expanded_state = expanded.state_dict()
    for key, value in source.items():
        if key in expanded_state:
            expanded_state[key] = value
    for key in tuple(expanded_state):
        if key.startswith("source_controller."):
            expanded_state[key] = source[key.removeprefix("source_")]
    expanded.load_state_dict(expanded_state)

    state = torch.randn(8, 46)
    baseline_p = baseline.cost_terms(state)[3]
    expanded_p = expanded.cost_terms(state)[3]
    torch.testing.assert_close(expanded_p, baseline_p, atol=2e-6, rtol=2e-6)

    baseline.zero_grad(set_to_none=True)
    expanded.zero_grad(set_to_none=True)
    baseline_p.sum().backward()
    expanded_p.sum().backward()
    torch.testing.assert_close(
        expanded.controller[-1].weight.grad,
        baseline.controller[-1].weight.grad,
        atol=2e-4,
        rtol=2e-4,
    )
    assert expanded.action_headroom_adapter.weight.grad is not None
    assert torch.isfinite(expanded.action_headroom_adapter.weight.grad).all()
    assert expanded.action_headroom_adapter.weight.grad.norm() > 0


def test_adapter_only_freezes_inherited_actor() -> None:
    model = actor(
        "k6",
        action_cost_center_limit=0.01,
        action_headroom_limit=0.02,
        action_headroom_adapter_only=True,
    )
    assert model.action_headroom_adapter is not None
    assert all(parameter.requires_grad for parameter in model.action_headroom_adapter.parameters())
    assert not model.log_std.requires_grad
    assert not any(parameter.requires_grad for parameter in model.controller.parameters())
    assert not any(parameter.requires_grad for parameter in model.source_controller.parameters())

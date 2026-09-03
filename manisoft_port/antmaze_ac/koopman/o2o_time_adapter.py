"""Frozen absolute-action body Koopman lift augmented by the exact scalar clock."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .checkpoint import load_checkpoint
from .history_model import HistoryDeepKoopman


class FrozenManiSoftTimeKoopman(nn.Module):
    POLICY_OBSERVATION_DIM = 46

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self.path = Path(path).expanduser().resolve()
        model, payload = load_checkpoint(self.path, map_location="cpu")
        history_steps = int(getattr(model, "history_steps", 0))
        if (model.state_dim, model.action_dim) != (45, 18) or history_steps not in (0, 10):
            raise ValueError("Expected ManiSoft H0 or H10 Koopman")
        self.physical_model = model.float().eval().requires_grad_(False)
        state = payload.get("normalizers", {}).get("state", {})
        center = torch.as_tensor(state.get("mean"), dtype=torch.float32)
        scale = torch.as_tensor(state.get("std"), dtype=torch.float32).clamp_min(1e-6)
        if center.shape != (45,) or scale.shape != (45,):
            raise ValueError("Invalid physical-state normalizer")
        self.history_steps = history_steps
        self.context_dim = history_steps * (45 + 18)
        self.HISTORY_CONTEXT_DIM = self.context_dim
        self.INPUT_OBSERVATION_DIM = 46 + self.context_dim
        self.state_dim = 46
        self.action_dim = 18
        self.lift_dim = int(model.lift_dim)
        physical_lifted_dim = int(model.lifted_dim)
        self.physical_lifted_dim = physical_lifted_dim
        self.lifted_dim = physical_lifted_dim + 1
        self.register_buffer("physical_center", center)
        self.register_buffer("physical_scale", scale)
        self.register_buffer("center", torch.cat((center, torch.zeros(1))))
        self.register_buffer("scale", torch.cat((scale, torch.ones(1))))
        A = torch.zeros(self.lifted_dim, self.lifted_dim, dtype=torch.float32)
        A[:physical_lifted_dim, :physical_lifted_dim] = model.A.detach()
        A[-1, -1] = 1.0
        B = torch.zeros(self.lifted_dim, 18, dtype=torch.float32)
        B[:physical_lifted_dim] = model.B.detach()
        C = torch.zeros(46, self.lifted_dim, dtype=torch.float32)
        C[:45, :physical_lifted_dim] = model.C.detach()
        C[45, -1] = 1.0
        self.register_buffer("A", A)
        self.register_buffer("B", B)
        self.register_buffer("C", C)
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.sha256 = digest
        self.metadata = {
            "kind": "manisoft_koopman_time_adapter_v1",
            "architecture": dict(payload["architecture"]),
            "clock": "tau=t/(T-1), held in model state; horizon lookup advances explicitly",
        }
        self.requires_grad_(False)

    def normalize(self, observation: torch.Tensor) -> torch.Tensor:
        physical = (observation[..., :45] - self.physical_center) / self.physical_scale
        return torch.cat((physical, observation[..., 45:46]), dim=-1)

    def denormalize(self, normalized_state: torch.Tensor) -> torch.Tensor:
        physical = normalized_state[..., :45] * self.physical_scale + self.physical_center
        return torch.cat((physical, normalized_state[..., 45:46]), dim=-1)

    def lift(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.shape[-1] != self.INPUT_OBSERVATION_DIM:
            raise ValueError("Time Koopman received an observation with the wrong dimension")
        policy = observation[..., :46]
        context = observation[..., 46:]
        physical = (policy[..., :45] - self.physical_center) / self.physical_scale
        if isinstance(self.physical_model, HistoryDeepKoopman):
            body = self.physical_model.lift(physical, context)
        else:
            if context.shape[-1] != 0:
                raise ValueError("H0 Koopman received history")
            body = self.physical_model.lift(physical)
        return torch.cat((body, policy[..., 45:46]), dim=-1)

    def step(self, lifted_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return F.linear(lifted_state, self.A) + F.linear(action, self.B)

    def reconstruct_normalized(self, lifted_state: torch.Tensor) -> torch.Tensor:
        return F.linear(lifted_state, self.C)

    def reconstruct(self, lifted_state: torch.Tensor) -> torch.Tensor:
        return self.denormalize(self.reconstruct_normalized(lifted_state))

    def identity(self) -> dict[str, Any]:
        return {
            "kind": self.metadata["kind"],
            "path": str(self.path),
            "sha256": self.sha256,
            "architecture": self.metadata["architecture"],
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "lift_dim": self.lift_dim,
            "history_steps": self.history_steps,
            "clock": self.metadata["clock"],
        }

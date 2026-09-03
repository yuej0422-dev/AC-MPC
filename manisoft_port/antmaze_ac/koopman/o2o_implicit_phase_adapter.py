"""Frozen history Koopman model with exact periodic phase dynamics."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from antmaze_ac.envs.circle_task_state import PHASE_DIM, phase_transition_matrix

from .checkpoint import load_checkpoint
from .history_model import HistoryDeepKoopman


class FrozenManiSoftImplicitPhaseKoopman(nn.Module):
    POLICY_OBSERVATION_DIM = 45 + PHASE_DIM
    HISTORY_CONTEXT_DIM = 10 * (45 + 18)
    INPUT_OBSERVATION_DIM = POLICY_OBSERVATION_DIM + HISTORY_CONTEXT_DIM

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self.path = Path(path).expanduser().resolve()
        model, payload = load_checkpoint(self.path, map_location="cpu")
        if not isinstance(model, HistoryDeepKoopman):
            raise TypeError("ManiSoft O2O requires HistoryDeepKoopman")
        if (model.state_dim, model.action_dim, model.history_steps) != (45, 18, 10):
            raise ValueError("Unexpected ManiSoft history Koopman architecture")
        self.history_model = model.float().eval().requires_grad_(False)
        state = payload.get("normalizers", {}).get("state", {})
        physical_center = torch.as_tensor(state.get("mean"), dtype=torch.float32)
        physical_scale = torch.as_tensor(
            state.get("std"), dtype=torch.float32
        ).clamp_min(1e-6)
        if physical_center.shape != (45,) or physical_scale.shape != (45,):
            raise ValueError("Koopman checkpoint has an invalid state normalizer")

        self.state_dim = self.POLICY_OBSERVATION_DIM
        self.action_dim = 18
        self.lift_dim = int(model.lift_dim)
        self.lifted_dim = self.state_dim + self.lift_dim
        self.history_steps = 10
        self.context_dim = self.HISTORY_CONTEXT_DIM
        self.register_buffer(
            "center",
            torch.cat((physical_center, torch.zeros(PHASE_DIM))),
        )
        self.register_buffer(
            "scale",
            torch.cat((physical_scale, torch.ones(PHASE_DIM))),
        )

        physical_lifted_dim = int(model.lifted_dim)
        phase_left = physical_lifted_dim
        phase_right = phase_left + PHASE_DIM
        A = torch.zeros(self.lifted_dim, self.lifted_dim, dtype=torch.float32)
        A[:physical_lifted_dim, :physical_lifted_dim] = model.A.detach()
        A[phase_left:phase_right, phase_left:phase_right] = torch.from_numpy(
            phase_transition_matrix(1000)
        )
        B = torch.zeros(self.lifted_dim, self.action_dim, dtype=torch.float32)
        B[:physical_lifted_dim] = model.B.detach()
        C = torch.zeros(self.state_dim, self.lifted_dim, dtype=torch.float32)
        C[:45, :physical_lifted_dim] = model.C.detach()
        C[45:, phase_left:phase_right] = torch.eye(PHASE_DIM)
        self.register_buffer("A", A)
        self.register_buffer("B", B)
        self.register_buffer("C", C)

        digest = hashlib.sha256()
        with self.path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        self.sha256 = digest.hexdigest()
        self.metadata = {
            "kind": "manisoft_history_koopman_implicit_phase_adapter_v2",
            "architecture": dict(payload["architecture"]),
            "best_validation": float(payload.get("best_validation", float("nan"))),
            "epoch": int(payload.get("epoch", -1)),
            "phase_dynamics": "exact sin/cos rotation for a 1000-step period",
        }
        self.requires_grad_(False)

    def normalize(self, observation: torch.Tensor) -> torch.Tensor:
        return (observation[..., : self.state_dim] - self.center) / self.scale

    def denormalize(self, normalized_state: torch.Tensor) -> torch.Tensor:
        return normalized_state * self.scale + self.center

    def lift(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.shape[-1] != self.INPUT_OBSERVATION_DIM:
            raise ValueError(
                f"History lift expects {self.INPUT_OBSERVATION_DIM} inputs, "
                f"got {observation.shape[-1]}"
            )
        policy_observation = observation[..., : self.POLICY_OBSERVATION_DIM]
        context = observation[..., self.POLICY_OBSERVATION_DIM :]
        normalized = self.normalize(policy_observation)
        physical_lift = self.history_model.lift(normalized[..., :45], context)
        return torch.cat((physical_lift, normalized[..., 45:]), dim=-1)

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
            "phase_dynamics": self.metadata["phase_dynamics"],
        }

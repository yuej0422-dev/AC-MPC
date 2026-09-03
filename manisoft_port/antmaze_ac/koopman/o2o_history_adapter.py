"""Adapt ManiSoft's H=10 PyTorch Koopman checkpoint to the O2O learner API."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .checkpoint import load_checkpoint
from .history_model import HistoryDeepKoopman


class FrozenManiSoftHistoryKoopman(nn.Module):
    """History-aware 46-D task lift with frozen absolute-action dynamics.

    The trained model evolves a 45-D physical state plus 32 learned lift
    coordinates.  The fixed-circle clock is appended as one normalized state
    coordinate and held constant inside a short KMPC rollout.  This makes the
    controller time-aware without placing the time-indexed XYZ target in its
    observation or pretending the history encoder is memoryless.
    """

    POLICY_OBSERVATION_DIM = 46
    HISTORY_CONTEXT_DIM = 10 * (45 + 18)
    INPUT_OBSERVATION_DIM = POLICY_OBSERVATION_DIM + HISTORY_CONTEXT_DIM

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        model, payload = load_checkpoint(self.path, map_location="cpu")
        history_steps = int(getattr(model, "history_steps", 0))
        if (model.state_dim, model.action_dim) != (45, 18) or history_steps not in (
            0,
            10,
        ):
            raise ValueError("Expected a ManiSoft H0 or H10 Koopman checkpoint")
        self.physical_model = model.float().eval().requires_grad_(False)
        state = payload.get("normalizers", {}).get("state", {})
        physical_center = torch.as_tensor(state.get("mean"), dtype=torch.float32)
        physical_scale = torch.as_tensor(state.get("std"), dtype=torch.float32)
        if physical_center.shape != (45,) or physical_scale.shape != (45,):
            raise ValueError("Koopman checkpoint has an invalid state normalizer")
        physical_scale = physical_scale.clamp_min(1e-6)

        self.state_dim = 46
        self.action_dim = 18
        self.lift_dim = int(model.lift_dim)
        self.lifted_dim = self.state_dim + self.lift_dim
        self.history_steps = history_steps
        self.context_dim = self.history_steps * (45 + 18)
        self.HISTORY_CONTEXT_DIM = self.context_dim
        self.INPUT_OBSERVATION_DIM = self.POLICY_OBSERVATION_DIM + self.context_dim
        self.register_buffer(
            "center",
            torch.cat(
                (physical_center, torch.tensor([0.5], dtype=torch.float32))
            ),
        )
        self.register_buffer(
            "scale",
            torch.cat(
                (
                    physical_scale,
                    torch.tensor(
                        [1.0 / np.sqrt(12.0)], dtype=torch.float32
                    ),
                )
            ),
        )

        original_lifted_dim = int(model.lifted_dim)
        A = torch.zeros(self.lifted_dim, self.lifted_dim, dtype=torch.float32)
        A[:original_lifted_dim, :original_lifted_dim] = model.A.detach()
        A[-1, -1] = 1.0
        B = torch.zeros(self.lifted_dim, self.action_dim, dtype=torch.float32)
        B[:original_lifted_dim] = model.B.detach()
        C = torch.zeros(self.state_dim, self.lifted_dim, dtype=torch.float32)
        C[:45, :original_lifted_dim] = model.C.detach()
        C[45, -1] = 1.0
        self.register_buffer("A", A)
        self.register_buffer("B", B)
        self.register_buffer("C", C)

        digest = hashlib.sha256()
        with self.path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        self.sha256 = digest.hexdigest()
        self.metadata = {
            "kind": "manisoft_koopman_o2o_adapter_v2",
            "architecture": dict(payload["architecture"]),
            "best_validation": float(payload.get("best_validation", float("nan"))),
            "epoch": int(payload.get("epoch", -1)),
            "clock_dynamics": "normalized clock held constant within KMPC H=5",
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
        if isinstance(self.physical_model, HistoryDeepKoopman):
            physical_lift = self.physical_model.lift(normalized[..., :45], context)
        else:
            if context.shape[-1] != 0:
                raise ValueError("H0 Koopman received a non-empty history context")
            physical_lift = self.physical_model.lift(normalized[..., :45])
        return torch.cat((physical_lift, normalized[..., 45:46]), dim=-1)

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
            "clock_dynamics": self.metadata["clock_dynamics"],
        }

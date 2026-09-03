"""Frozen periodic feedforward action used by residual circle control."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np


ACTION_DIM = 18
HARMONICS = 8
FEATURE_DIM = 2 * HARMONICS


def phase_fourier_features(
    step: np.ndarray | int,
    episode_steps: int,
) -> np.ndarray:
    """Return sin/cos features without exposing them as learned task state."""

    phase = np.asarray(step, dtype=np.float64) / float(episode_steps)
    angle = 2.0 * np.pi * phase
    result = np.stack(
        [
            component
            for harmonic in range(1, HARMONICS + 1)
            for component in (
                np.sin(harmonic * angle),
                np.cos(harmonic * angle),
            )
        ],
        axis=-1,
    )
    return result.astype(np.float64, copy=False)


class FrozenCirclePhaseFeedforward:
    """Read-only phase-to-action map fitted before residual RL begins."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        with np.load(self.path, allow_pickle=False) as archive:
            self.feature_mean = np.asarray(
                archive["feature_mean"], dtype=np.float64
            )
            self.feature_scale = np.asarray(
                archive["feature_scale"], dtype=np.float64
            )
            self.weight = np.asarray(archive["weight"], dtype=np.float64)
        if self.feature_mean.shape != (FEATURE_DIM,):
            raise ValueError("Feedforward mean has the wrong shape")
        if self.feature_scale.shape != (FEATURE_DIM,):
            raise ValueError("Feedforward scale has the wrong shape")
        if self.weight.shape != (FEATURE_DIM + 1, ACTION_DIM):
            raise ValueError("Feedforward weight has the wrong shape")
        if (
            not np.isfinite(self.feature_mean).all()
            or not np.isfinite(self.feature_scale).all()
            or not np.isfinite(self.weight).all()
            or np.any(self.feature_scale <= 0)
        ):
            raise ValueError("Feedforward artifact is invalid")
        digest = hashlib.sha256()
        with self.path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        self.sha256 = digest.hexdigest()

    def action(
        self,
        step: np.ndarray | int,
        episode_steps: int,
    ) -> np.ndarray:
        features = phase_fourier_features(step, episode_steps)
        normalized = (features - self.feature_mean) / self.feature_scale
        ones = np.ones((*normalized.shape[:-1], 1), dtype=np.float64)
        design = np.concatenate((normalized, ones), axis=-1)
        return (design @ self.weight).astype(np.float32)

    def identity(self) -> dict[str, Any]:
        return {
            "kind": "manisoft_circle_phase8_ridge_feedforward_v1",
            "path": str(self.path),
            "sha256": self.sha256,
            "input": "clock-derived Fourier harmonics 1..8",
            "target_in_input": False,
            "action_dim": ACTION_DIM,
        }

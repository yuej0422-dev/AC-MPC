"""Reward utilities for the fixed ManiSoft circle task."""

from __future__ import annotations

import math


REWARD_MODES = ("sparse", "hybrid", "dense_xref", "dense_joint")


def validate_reward_config(
    mode: str,
    *,
    sparse_weight: float,
    dense_weight: float,
    dense_scale_m: float,
    radius_m: float,
) -> None:
    if mode not in REWARD_MODES:
        raise ValueError(f"reward mode must be one of {REWARD_MODES}, got {mode!r}")
    for name, value in (
        ("sparse_weight", sparse_weight),
        ("dense_weight", dense_weight),
        ("dense_scale_m", dense_scale_m),
        ("radius_m", radius_m),
    ):
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
        if name in ("dense_scale_m", "radius_m") and float(value) <= 0:
            raise ValueError(f"{name} must be positive")
        if name in ("sparse_weight", "dense_weight") and float(value) < 0:
            raise ValueError(f"{name} must be non-negative")
    if mode == "hybrid" and sparse_weight == 0 and dense_weight == 0:
        raise ValueError("hybrid reward needs a non-zero sparse or dense weight")
    if mode == "dense_xref" and (sparse_weight != 0 or dense_weight <= 0):
        raise ValueError("dense_xref reward requires sparse_weight=0 and dense_weight>0")
    if mode == "dense_joint" and (sparse_weight != 0 or dense_weight <= 0):
        raise ValueError("dense_joint reward requires sparse_weight=0 and dense_weight>0")


def circle_reward_components(
    error_m: float,
    *,
    sparse_weight: float = 1.0,
    dense_weight: float = 0.0,
    dense_scale_m: float = 0.01,
    radius_m: float = 0.0025,
) -> tuple[float, float, float]:
    """Return ``(total, sparse_component, dense_component)``.

    ``error_m`` may be the historical three-node joint error or the full-state
    xref RMSE.  The caller chooses that geometry explicitly.  The dense term
    is bounded in ``(0, dense_weight]`` and therefore keeps critic targets on
    the same order as the historical unit sparse reward.
    """

    if not math.isfinite(float(error_m)) or error_m < 0:
        raise ValueError("error_m must be finite and non-negative")
    sparse = float(sparse_weight) * float(error_m <= radius_m)
    dense = float(dense_weight) * math.exp(-float(error_m) / float(dense_scale_m))
    return sparse + dense, sparse, dense

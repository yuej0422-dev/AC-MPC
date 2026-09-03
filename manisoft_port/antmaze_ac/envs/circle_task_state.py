"""Task-state and reward protocol for fixed-circle ManiSoft comparisons."""

from __future__ import annotations

import math

import numpy as np


PHYSICAL_DIM = 45
PHASE_HARMONICS = 1
PHASE_DIM = 2 * PHASE_HARMONICS
TARGET_POSITION_DIM = 9
IMPLICIT_OBSERVATION_DIM = PHYSICAL_DIM + PHASE_DIM
EXPLICIT_OBSERVATION_DIM = PHYSICAL_DIM + TARGET_POSITION_DIM
OBSERVATION_MODES = ("implicit_phase", "explicit_target")
DEFAULT_NODE_WEIGHTS = np.asarray((1.0, 1.0, 2.0), dtype=np.float32)
DEFAULT_DISTANCE_SCALE_M = 0.05
DEFAULT_SUCCESS_THRESHOLD_M = 0.0025
DEFAULT_SUCCESS_BONUS = 1.0


def phase_features(
    step: np.ndarray | int,
    episode_steps: int,
    *,
    harmonics: int = PHASE_HARMONICS,
) -> np.ndarray:
    """Return periodic ``sin/cos`` features without exposing a target state."""

    if episode_steps < 1 or harmonics < 1:
        raise ValueError("episode_steps and harmonics must be positive")
    phase = np.asarray(step, dtype=np.float64) / float(episode_steps)
    angle = 2.0 * np.pi * phase
    result = np.stack(
        [
            component
            for harmonic in range(1, harmonics + 1)
            for component in (
                np.sin(harmonic * angle),
                np.cos(harmonic * angle),
            )
        ],
        axis=-1,
    )
    return result.astype(np.float32)


def phase_transition_matrix(
    episode_steps: int,
    *,
    harmonics: int = PHASE_HARMONICS,
) -> np.ndarray:
    """Exact one-control-step linear dynamics for :func:`phase_features`."""

    if episode_steps < 1 or harmonics < 1:
        raise ValueError("episode_steps and harmonics must be positive")
    result = np.zeros((2 * harmonics, 2 * harmonics), dtype=np.float32)
    base = 2.0 * np.pi / float(episode_steps)
    for index in range(harmonics):
        angle = (index + 1) * base
        cosine = math.cos(angle)
        sine = math.sin(angle)
        left = 2 * index
        # Feature ordering is [sin(theta), cos(theta)].
        result[left : left + 2, left : left + 2] = (
            (cosine, sine),
            (-sine, cosine),
        )
    return result


def implicit_task_observation(
    physical_state: np.ndarray,
    step: np.ndarray | int,
    episode_steps: int,
) -> np.ndarray:
    physical = np.asarray(physical_state, dtype=np.float32)
    phase = phase_features(step, episode_steps)
    result = np.concatenate((physical, phase), axis=-1)
    if result.shape[-1] != IMPLICIT_OBSERVATION_DIM or not np.isfinite(result).all():
        raise ValueError("Invalid implicit ManiSoft circle observation")
    return result.astype(np.float32, copy=False)


def explicit_task_observation(
    physical_state: np.ndarray,
    target_positions: np.ndarray,
) -> np.ndarray:
    physical = np.asarray(physical_state, dtype=np.float32)
    target = np.asarray(target_positions, dtype=np.float32)
    if target.shape[-2:] != (3, 3):
        raise ValueError("Target positions must end in shape (3, 3)")
    result = np.concatenate(
        (physical, target.reshape(*target.shape[:-2], 9)),
        axis=-1,
    )
    if result.shape[-1] != EXPLICIT_OBSERVATION_DIM or not np.isfinite(result).all():
        raise ValueError("Invalid explicit ManiSoft circle observation")
    return result.astype(np.float32, copy=False)


def circle_distance_bonus_reward(
    actual_positions: np.ndarray,
    target_positions: np.ndarray,
    *,
    node_weights: np.ndarray = DEFAULT_NODE_WEIGHTS,
    distance_scale_m: float = DEFAULT_DISTANCE_SCALE_M,
    success_threshold_m: float = DEFAULT_SUCCESS_THRESHOLD_M,
    success_bonus: float = DEFAULT_SUCCESS_BONUS,
) -> tuple[float, float, float, np.ndarray, float]:
    """Return total reward, distance penalty, bonus, node errors and joint error.

    The dense component is a non-saturating linear distance penalty.  Unlike an
    exponential shaping reward, it retains a useful gradient far from the
    circle.  The positive bonus uses the historical joint three-node tube.
    """

    actual = np.asarray(actual_positions, dtype=np.float32)
    target = np.asarray(target_positions, dtype=np.float32)
    weights = np.asarray(node_weights, dtype=np.float32)
    if actual.shape != (3, 3) or target.shape != (3, 3):
        raise ValueError("actual_positions and target_positions must be (3, 3)")
    if weights.shape != (3,) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("node_weights must contain three finite positive values")
    for name, value in (
        ("distance_scale_m", distance_scale_m),
        ("success_threshold_m", success_threshold_m),
        ("success_bonus", success_bonus),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    node_error = np.linalg.norm(actual - target, axis=1)
    joint_error = float(np.linalg.norm(node_error))
    weighted_distance = float(np.dot(weights, node_error) / weights.sum())
    distance_penalty = -weighted_distance / float(distance_scale_m)
    bonus = float(success_bonus) * float(joint_error <= success_threshold_m)
    return (
        float(distance_penalty + bonus),
        float(distance_penalty),
        bonus,
        node_error.astype(np.float32),
        joint_error,
    )

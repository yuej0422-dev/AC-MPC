from __future__ import annotations

import copy
from pathlib import Path

import pytest

from experiments.dmc.config import load_experiment_config, validate_config


CONFIG = Path(
    "experiments/dmc/campaigns/hopper_hop_native1000_ppo50m.yaml"
)


def test_native1000_ppo50m_campaign_contract() -> None:
    config = load_experiment_config(CONFIG)
    protocol = config.raw["protocol"]
    profile = config.raw["profiles"]["development"]

    assert config.task == "hopper_hop"
    assert protocol == {
        "name": "dmc_native_v1",
        "control_timestep": "native",
        "action_repeat": 1,
        "time_limit_seconds": 20.0,
        "episode_steps": 1000,
        "score": "sum_official_reward",
    }
    assert profile["num_envs"] == 256
    assert profile["rollout_steps"] == 8
    assert profile["total_timesteps"] == 49_999_872
    assert profile["total_timesteps"] % (
        profile["num_envs"] * profile["rollout_steps"]
    ) == 0
    assert config.raw["evaluation"]["diagnostic_every_steps"] == 500_000


def test_diagnostic_checkpoint_period_remains_on_50k_grid() -> None:
    raw = copy.deepcopy(load_experiment_config(CONFIG).raw)
    raw["evaluation"]["diagnostic_every_steps"] = 75_000
    with pytest.raises(ValueError, match="multiple of 50000"):
        validate_config(raw)

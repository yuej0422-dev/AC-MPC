"""Formal seven-method O2O campaign for native ManiSkill HopperHop."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from experiments.dmc.o2o.config import O2OConfig
from experiments.dmc.o2o.dataset import (
    MANISKILL_HOPPER_DATASET_KIND,
    OfflineDataset,
)
from experiments.dmc.o2o.formal_hopper import (
    FORMAL_DIAGNOSTIC_EPISODES,
    FORMAL_KMPC_HORIZON,
    FORMAL_METHODS,
    FORMAL_MPVE_HORIZON,
    FORMAL_OFFLINE_EVAL_INTERVAL,
    FORMAL_OFFLINE_TRANSITIONS,
    FORMAL_OFFLINE_UPDATES,
    FORMAL_ONLINE_EVAL_INTERVAL,
    FORMAL_ONLINE_TRANSITIONS,
    FORMAL_TRAINING_SEEDS,
)
from experiments.dmc.o2o.koopman import FrozenKoopman, file_sha256


TASK = "hopper_hop"
BACKEND = "maniskill_hopper_hop"
EPISODE_HORIZON = 600
STRUCTURED_METHODS = frozenset({"Cal-RLPD-KMPC", "Cal-RLPD-Lift"})
BC_METHODS = frozenset({"AWAC", "IQL"})
STRUCTURED_OFFLINE_LEARNING_RATE = 1e-4


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_dataset(path: Path) -> OfflineDataset:
    dataset = OfflineDataset.load(path.resolve())
    selection = dataset.metadata.get("selection", {})
    expected = {
        "kind": "maniskill_hopper_hop_ppo_qualitymix_equal_v1",
        "policy_seeds": [20_240_801, 20_240_802, 20_240_803, 20_240_804],
        "transitions_per_policy": 50_000,
        "equal_policy_weight": True,
    }
    mismatches = {
        key: {"dataset": selection.get(key), "expected": value}
        for key, value in expected.items()
        if selection.get(key) != value
    }
    if dataset.metadata.get("kind") != MANISKILL_HOPPER_DATASET_KIND:
        raise ValueError("Formal dataset is not a ManiSkill HopperHop archive")
    if dataset.metadata.get("task") != TASK or len(dataset) != FORMAL_OFFLINE_TRANSITIONS:
        raise ValueError("Formal ManiSkill HopperHop dataset must contain exactly 200k transitions")
    protocol = {
        "environment_id": "MS-HopperHop-v1",
        "episode_horizon": EPISODE_HORIZON,
        "action_repeat": 1,
        "control_mode": "pd_joint_delta_pos",
        "reward_mode": "normalized_dense",
        "reward_source": "recorded",
    }
    mismatches.update(
        {
            key: {"dataset": dataset.metadata.get(key), "expected": value}
            for key, value in protocol.items()
            if dataset.metadata.get(key) != value
        }
    )
    if mismatches:
        raise ValueError(f"Formal ManiSkill dataset contract differs: {mismatches}")
    return dataset


def validate_koopman(path: Path) -> FrozenKoopman:
    model = FrozenKoopman(path.resolve())
    if (model.state_dim, model.action_dim, model.lift_dim) != (15, 4, 48):
        raise ValueError("ManiSkill Hopper Koopman dimensions must be 15/4/48")
    metadata = model.metadata
    if metadata.get("task") != TASK or metadata.get("state_kind") != "hopperhop":
        raise ValueError("Koopman export is not the ManiSkill HopperHop model")
    if metadata.get("k_step") != 20:
        raise ValueError("Formal Koopman model must use K=20")
    if metadata.get("reward_layer_count") != 0:
        raise ValueError("Formal reused Koopman model must be reward-free")
    source_path = Path(str(metadata.get("source_path", "")))
    if not source_path.is_file() or file_sha256(source_path) != metadata.get("source_sha256"):
        raise ValueError("Frozen Koopman export is not bound to its source checkpoint")
    return model


def training_command(
    *, method: str, seed: int, dataset: OfflineDataset, koopman: FrozenKoopman,
    output: Path, device: str, non_bc_offline_learning_rate: float | None = None,
) -> list[str]:
    config = O2OConfig(
        task=TASK, method=method, environment_backend=BACKEND, seed=seed
    )
    command = [
        sys.executable,
        "-m", "experiments.dmc.o2o.train",
        "--task", TASK,
        "--environment-backend", BACKEND,
        "--method", method,
        "--dataset", str(dataset.path),
        "--output-dir", str(output.resolve()),
        "--seed", str(seed),
        "--device", device,
        "--kmpc-horizon", str(FORMAL_KMPC_HORIZON),
        "--mpve-total-horizon", str(FORMAL_MPVE_HORIZON),
        "--offline-updates", str(FORMAL_OFFLINE_UPDATES),
        "--offline-eval-interval-updates", str(FORMAL_OFFLINE_EVAL_INTERVAL),
        "--online-steps", str(FORMAL_ONLINE_TRANSITIONS),
        "--eval-interval-online-steps", str(FORMAL_ONLINE_EVAL_INTERVAL),
        "--eval-episodes", str(FORMAL_DIAGNOSTIC_EPISODES),
    ]
    if config.requires_koopman:
        command.extend(("--koopman", str(koopman.path)))
    offline_learning_rate = None
    if (
        non_bc_offline_learning_rate is not None
        and method not in BC_METHODS
        and config.method_spec.offline_pretraining
    ):
        offline_learning_rate = non_bc_offline_learning_rate
    elif method in STRUCTURED_METHODS:
        offline_learning_rate = STRUCTURED_OFFLINE_LEARNING_RATE
    if offline_learning_rate is not None:
        command.extend(
            (
                "--offline-actor-learning-rate",
                str(offline_learning_rate),
                "--offline-critic-learning-rate",
                str(offline_learning_rate),
            )
        )
    return command


def session_name(seed: int, method: str, tag: str | None = None) -> str:
    suffix = f"_{tag}" if tag else ""
    return f"ms_hop_{seed}_{method}{suffix}".lower().replace("-", "_")


def gpu_memory_used_mib() -> int:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    values = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    if not values:
        raise RuntimeError("nvidia-smi returned no GPU memory values")
    return max(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-seed", type=int, choices=FORMAL_TRAINING_SEEDS, required=True
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--koopman", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--non-bc-offline-learning-rate", type=float)
    parser.add_argument("--session-tag")
    parser.add_argument("--launch-max-gpu-memory-mib", type=int)
    parser.add_argument("--launch-stagger-seconds", type=int, default=0)
    parser.add_argument("--resource-poll-seconds", type=int, default=60)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if (
        args.non_bc_offline_learning_rate is not None
        and args.non_bc_offline_learning_rate <= 0
    ):
        parser.error("--non-bc-offline-learning-rate must be positive")
    if (
        args.launch_max_gpu_memory_mib is not None
        and args.launch_max_gpu_memory_mib <= 0
    ):
        parser.error("--launch-max-gpu-memory-mib must be positive")
    if args.launch_stagger_seconds < 0 or args.resource_poll_seconds <= 0:
        parser.error("launch stagger must be nonnegative and resource polling must be positive")

    dataset = validate_dataset(args.dataset)
    koopman = validate_koopman(args.koopman)
    seed_root = (args.run_root / f"seed_{args.training_seed}").resolve()
    commands = {
        method: training_command(
            method=method,
            seed=args.training_seed,
            dataset=dataset,
            koopman=koopman,
            output=seed_root / method,
            device=args.device,
            non_bc_offline_learning_rate=args.non_bc_offline_learning_rate,
        )
        for method in FORMAL_METHODS
    }
    effective_offline_learning_rates = {}
    for method in FORMAL_METHODS:
        config = O2OConfig(
            task=TASK, method=method, environment_backend=BACKEND, seed=args.training_seed
        )
        if not config.method_spec.offline_pretraining:
            effective_offline_learning_rates[method] = None
        elif args.non_bc_offline_learning_rate is not None and method not in BC_METHODS:
            effective_offline_learning_rates[method] = args.non_bc_offline_learning_rate
        elif method in STRUCTURED_METHODS:
            effective_offline_learning_rates[method] = STRUCTURED_OFFLINE_LEARNING_RATE
        else:
            effective_offline_learning_rates[method] = config.actor_learning_rate
    plan = {
        "kind": "acmpc_maniskill_hopper_hop_formal_seed_v1",
        "task": TASK,
        "environment_backend": BACKEND,
        "environment_protocol": {
            "environment_id": "MS-HopperHop-v1",
            "episode_horizon": EPISODE_HORIZON,
            "action_repeat": 1,
            "control_mode": "pd_joint_delta_pos",
            "reward_mode": "normalized_dense",
            "sim_backend": "gpu",
        },
        "training_seed": args.training_seed,
        "formal_training_seeds": list(FORMAL_TRAINING_SEEDS),
        "methods": list(FORMAL_METHODS),
        "dataset": {
            "path": str(dataset.path), "sha256": dataset.sha256,
            "transitions": len(dataset), "selection": dataset.metadata["selection"],
        },
        "koopman": {"path": str(koopman.path), "sha256": koopman.sha256},
        "training": {
            "offline_updates": FORMAL_OFFLINE_UPDATES,
            "online_transitions": FORMAL_ONLINE_TRANSITIONS,
            "offline_eval_interval_updates": FORMAL_OFFLINE_EVAL_INTERVAL,
            "online_eval_interval_transitions": FORMAL_ONLINE_EVAL_INTERVAL,
            "diagnostic_episodes_vectorized_on_gpu": FORMAL_DIAGNOSTIC_EPISODES,
            "kmpc_horizon": FORMAL_KMPC_HORIZON,
            "mpve_horizon": FORMAL_MPVE_HORIZON,
            "default_structured_offline_actor_learning_rate": (
                STRUCTURED_OFFLINE_LEARNING_RATE
            ),
            "default_structured_offline_critic_learning_rate": (
                STRUCTURED_OFFLINE_LEARNING_RATE
            ),
            "online_actor_learning_rate": 3e-4,
            "online_critic_learning_rate": 3e-4,
            "non_bc_offline_learning_rate_override": args.non_bc_offline_learning_rate,
            "bc_methods_unchanged": sorted(BC_METHODS),
            "effective_offline_actor_critic_learning_rates": effective_offline_learning_rates,
            "rlpd_offline_learning_rate": None,
            "rlpd_note": "No offline pretraining; online learning rates remain unchanged.",
        },
        "launcher": {
            "session_tag": args.session_tag,
            "max_gpu_memory_mib_before_launch": args.launch_max_gpu_memory_mib,
            "stagger_seconds": args.launch_stagger_seconds,
            "resource_poll_seconds": args.resource_poll_seconds,
        },
        "commands": {method: shlex.join(command) for method, command in commands.items()},
    }
    _atomic_json(seed_root / "seed_plan.json", plan)
    for method, command in commands.items():
        print(f"{method}: {shlex.join(command)}", flush=True)
        if not args.launch:
            continue
        output = seed_root / method
        output.mkdir(parents=True, exist_ok=True)
        name = session_name(args.training_seed, method, args.session_tag)
        exists = subprocess.run(
            ["tmux", "has-session", "-t", f"={name}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode == 0
        if exists:
            print(f"existing tmux session retained: {name}", flush=True)
            continue
        if args.launch_max_gpu_memory_mib is not None:
            while True:
                used_mib = gpu_memory_used_mib()
                if used_mib <= args.launch_max_gpu_memory_mib:
                    break
                print(
                    f"waiting to launch {method}: GPU memory {used_mib} MiB exceeds "
                    f"{args.launch_max_gpu_memory_mib} MiB",
                    flush=True,
                )
                time.sleep(args.resource_poll_seconds)
        log_path = output / "training.log"
        shell_command = (
            f"cd {shlex.quote(str(Path.cwd()))} && "
            f"{shlex.join(command)} >> {shlex.quote(str(log_path))} 2>&1"
        )
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", name, shell_command],
            check=True,
        )
        print(f"launched tmux={name} log={log_path}", flush=True)
        if args.launch_stagger_seconds:
            time.sleep(args.launch_stagger_seconds)


if __name__ == "__main__":
    main()

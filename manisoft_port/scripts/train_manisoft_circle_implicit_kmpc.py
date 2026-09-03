#!/usr/bin/env python
"""Train O2O KMPC while replacing only the implicit circle task state."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


def _install_numpy_checkpoint_compatibility() -> None:
    """Let NumPy 1 load checkpoints serialized with NumPy 2 module names."""

    if int(np.__version__.split(".", 1)[0]) >= 2:
        return
    import numpy.core as core

    sys.modules.setdefault("numpy._core", core)
    for suffix in ("multiarray", "numeric", "_multiarray_umath", "umath"):
        legacy_name = f"numpy.core.{suffix}"
        __import__(legacy_name)
        sys.modules.setdefault(f"numpy._core.{suffix}", sys.modules[legacy_name])


_install_numpy_checkpoint_compatibility()


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))
MANISOFT_PORT_ROOT = Path(__file__).resolve().parents[1]
if str(MANISOFT_PORT_ROOT) in sys.path:
    sys.path.remove(str(MANISOFT_PORT_ROOT))
sys.path.insert(0, str(MANISOFT_PORT_ROOT))

from antmaze_ac.data.circle_implicit_kmpc_dataset import (
    ManiSoftCircleImplicitKmpcDataset,
)
from antmaze_ac.envs.manisoft_circle_implicit_kmpc_env import (
    COLLECTOR_OBSERVATION_DIM,
    make_manisoft_circle_implicit_kmpc_adapter,
)
from antmaze_ac.koopman.o2o_implicit_phase_adapter import (
    FrozenManiSoftImplicitPhaseKoopman,
)

import train_manisoft_circle_o2o as training


training.ManiSoftCircleOfflineDataset = ManiSoftCircleImplicitKmpcDataset
training.FrozenManiSoftHistoryKoopman = FrozenManiSoftImplicitPhaseKoopman
training.make_manisoft_circle_o2o_adapter = (
    make_manisoft_circle_implicit_kmpc_adapter
)
training.COLLECTOR_OBSERVATION_DIM = COLLECTOR_OBSERVATION_DIM


if __name__ == "__main__":
    training.run(training.parse_args())

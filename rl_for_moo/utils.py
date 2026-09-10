"""Small, dependency-light utilities shared by training entry points."""

from __future__ import annotations

import os
import random
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float = 0.005) -> None:
    """Apply a Polyak update to target-network parameters."""

    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            tau * source_param.data + (1.0 - tau) * target_param.data
        )


def elite_list_to_dataframe(
    elite_list: Iterable[tuple[Any, Any]], obj_dim: int, x_dim: int
) -> pd.DataFrame:
    """Convert the environment's Pareto archive to a portable CSV table."""

    elites = list(elite_list)
    data = {
        f"objective_value_{index}": [elite[0][index - 1] for elite in elites]
        for index in range(1, obj_dim + 1)
    }
    data.update(
        {
            f"x_{index}": [elite[1][index - 1] for elite in elites]
            for index in range(1, x_dim + 1)
        }
    )
    return pd.DataFrame(data)


class ExperimentLogger:
    """Optional W&B adapter; training remains usable without W&B."""

    def __init__(self, enabled: bool, project: str, name: str | None = None) -> None:
        self._run = None
        if not enabled:
            return
        try:
            import wandb
        except ImportError as exc:  # pragma: no cover - depends on local install
            raise RuntimeError("Install wandb or omit --wandb.") from exc
        self._run = wandb.init(
            project=project,
            name=name,
            mode=os.getenv("WANDB_MODE", "online"),
        )

    def log(self, values: dict[str, Any]) -> None:
        if self._run is not None:
            self._run.log(values)

    def close(self) -> None:
        if self._run is not None:
            self._run.finish()


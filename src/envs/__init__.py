"""Env builders: pure per-manager ``cfg -> manager dict`` functions (rewards/obs/commands/...);
``create_envs`` assembles the Isaac train/eval envs from a Hydra cfg."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# The vendored mjlab-compat subset must be importable before the builder imports.
_VENDOR = str(Path(__file__).resolve().parents[1] / "simulator" / "isaacsim" / "_vendor")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

from .actions import build_actions  # noqa: E402
from .commands import build_commands  # noqa: E402
from .curriculum import build_curriculum  # noqa: E402
from .events import build_events  # noqa: E402
from .observations import build_observations  # noqa: E402
from .rewards import build_rewards  # noqa: E402
from .terminations import build_terminations  # noqa: E402

__all__ = [
    "build_actions", "build_commands", "build_curriculum", "build_events",
    "build_observations", "build_rewards", "build_terminations", "create_envs",
]


def create_envs(cfg: Any, device: str = "cuda"):
    """(train_env, eval_env) on the Isaac backend (simulator/isaacsim/create.py)."""
    from simulator.isaacsim.create import create_envs_isaac

    return create_envs_isaac(cfg, device=device)

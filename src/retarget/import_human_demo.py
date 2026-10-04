from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
from omegaconf import DictConfig

from retarget.human_demo.sources.base import import_demos


@hydra.main(config_path="../../config/retarget", config_name="import_human_demo", version_base=None)
def main(cfg: DictConfig) -> None:
    mano_dir = Path(cfg.mano_dir or Path(__file__).resolve().parents[2] / "assets" / "mano")
    import_demos(cfg, mano_dir, cfg.workers or max(1, (os.cpu_count() or 2) // 2))


if __name__ == "__main__":
    main()

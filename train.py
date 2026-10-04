from __future__ import annotations

import sys
from pathlib import Path

# Make ``from envs import ...`` (and siblings under src/) resolve when the
# script is run directly. Must happen before the package imports below.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import argparse
import os

import hydra
from omegaconf import OmegaConf


def run(args: argparse.Namespace) -> None:
    # ─── Hydra compose ────────────────────────────────────────────────────
    OmegaConf.register_new_resolver("eval", lambda s: eval(s), replace=True)
    # initialize_config_dir takes an absolute path → CWD-independent.
    config_dir = (Path(__file__).resolve().parent / args.config_path).resolve()
    with hydra.initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = hydra.compose(config_name=args.config_name, overrides=args.overrides)
    OmegaConf.resolve(cfg)

    # ─── Isaac Sim app (boots before torch-heavy imports) ─────────────────
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "1")
    from isaaclab.app import AppLauncher

    headless = bool(cfg.get("headless", True))
    AppLauncher({"headless": headless})

    from common.logger import setup_logger
    from common.utils import configure_torch_backends, set_seed
    from envs import create_envs
    from runner import create_runner

    set_seed(cfg.seed)
    configure_torch_backends()

    # ─── Envs / runner / training ─────────────────────────────────────────
    train_env, eval_env = create_envs(cfg, device=cfg.device)
    setup_logger(cfg.logger, cfg.runner)
    runner = create_runner(cfg.runner, cfg.eval, train_env, eval_env=eval_env, headless=headless)
    runner.learn(num_learning_iterations=int(cfg.runner.max_iterations), init_at_random_ep_len=True)

    train_env.close()
    eval_env.close()
    # isaacsim 5.1 segfaults in close(); flush wandb and hard-exit instead.
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config_path", type=str, default="config")
    parser.add_argument("--config_name", type=str, default="flashdexretarget")
    parser.add_argument("--overrides", action="append", default=[])
    parser.add_argument("--gui", action="store_true", help="isaacsim viewport mode")
    args = parser.parse_args()
    if args.gui:
        args.overrides = list(args.overrides) + ["headless=false"]
    run(args)

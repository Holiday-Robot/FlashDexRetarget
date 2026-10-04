from __future__ import annotations

import json
import os
import shutil
import signal
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tqdm
import wandb
from mjlab.rl import RslRlVecEnvWrapper
from omegaconf import DictConfig, OmegaConf

from agents import build_agent
from common.average_meters import AverageMeter
from common.logger import log_metrics
from envs.vec_env import FlatObsVecEnv
from evaluation.evaluate import evaluate

# simulation/: anchor for the relative checkpoint / load paths in the cfg.
_BASE_DIR = Path(__file__).resolve().parents[2]


class OffPolicyRunner:
    """Off-policy training loop: steps the train envs, feeds the agent's replay buffer
    (``runner.agent.class_name``), and evaluates and checkpoints on absolute env-step grids."""

    def __init__(
        self,
        env: RslRlVecEnvWrapper,
        runner_cfg: DictConfig,
        eval_cfg: DictConfig,
        eval_env: RslRlVecEnvWrapper,
        headless: bool = True,
    ) -> None:
        # ── config / loop schedule ───────────────────────────────────────────────────
        self._cfg = runner_cfg
        self._eval_cfg = eval_cfg
        self._device = str(runner_cfg.device)
        self._n_envs = int(runner_cfg.num_train_envs)
        self._step_offset = int(runner_cfg.get("step_offset", 0))  # env steps done before a resume
        self._updates_per_step = float(runner_cfg.updates_per_interaction_step)
        self._log_every = int(runner_cfg.logging_per_interaction_step) or 0
        # Absolute env-step grids, so a resume does not shift the eval / checkpoint points.
        every = int(eval_cfg.get("interval") or 0)  # env steps
        self._eval_stride = max(1, every // self._n_envs) * self._n_envs if every else 0
        self._save_stride = (int(runner_cfg.save_checkpoint_per_interaction_step) or 0) * self._n_envs
        share_obs = bool(runner_cfg.get("share_actor_critic_obs", False))

        # ── envs ─────────────────────────────────────────────────────────────────────
        heads_cfg = runner_cfg.agent.get("reward_heads", None)
        self._train_env = FlatObsVecEnv.from_env(
            env.unwrapped,
            share_obs=share_obs,
            reward_heads=OmegaConf.to_container(heads_cfg, resolve=True) if heads_cfg is not None else None,
        )
        self._eval_env = eval_env
        # The TRAIN env's command term holds the curriculum state that resume_state.json records.
        try:
            self._train_cmd = env.unwrapped.command_manager.get_term(
                eval_cfg.command_name
            )
        except Exception:  # noqa: BLE001 - no command manager (non-tracking task)
            self._train_cmd = None
        self._observations, env_infos = self._train_env.reset()
        self._just_reset = True  # no policy obs yet: the next step acts randomly
        self._nonfinite_steps = 0

        # ── agent ────────────────────────────────────────────────────────────────────
        self._agent = build_agent(runner_cfg.agent, self._train_env, env_infos, eval_cfg.command_name)
        self._obs_aux = self._train_env.current_aux()  # set when the agent registered an aux provider
        # Record the runner cfg as the agent adjusted it; eval-only entrypoints have no wandb run.
        if wandb.run is not None:
            wandb.config.update(OmegaConf.to_container(runner_cfg, resolve=True), allow_val_change=True)

        # ── run dir / eval exports ───────────────────────────────────────────────────
        # Stamped here (not in learn()) so the exports and the checkpoints share one run dir.
        self._save_path_base = _BASE_DIR / runner_cfg.save_path.replace(
            "TIMESTAMP", datetime.now().strftime("%m%d-%H%M%S")
        )
        succ_dir = eval_cfg.get("success_output_dir", None)
        self._success_dir = Path(succ_dir) if succ_dir else self._save_path_base / "success"
        # Per-trajectory "ever succeeded" vectors, kept across the evals of this run.
        self._eval_ever_state: dict = {}

        # ── resume ───────────────────────────────────────────────────────────────────
        if runner_cfg.agent_load_path is not None:
            self._agent.load(str((_BASE_DIR / runner_cfg.agent_load_path).resolve()))
        if runner_cfg.buffer_load_path is not None:
            self._agent.load_buffer(str((_BASE_DIR / runner_cfg.buffer_load_path).resolve()))
        # Ever-success vectors ride in resume_state.json, so a resume keeps the cumulative curve.
        ever_src = runner_cfg.get("ever_state_path", None)
        if ever_src is None and runner_cfg.agent_load_path is not None:
            ever_src = (_BASE_DIR / runner_cfg.agent_load_path).resolve() / "resume_state.json"
        if ever_src is not None and Path(ever_src).is_file():
            try:
                ever = json.loads(Path(ever_src).read_text()).get("ever") or {}
                for k, v in ever.items():
                    self._eval_ever_state[k] = torch.tensor(v, dtype=torch.bool)
                print(f"[eval] ever-success state restored from {ever_src}: "
                      + ", ".join(f"{k}={int(t.reshape(t.shape[0], -1)[:, -1].sum())}"
                                  for k, t in self._eval_ever_state.items()), flush=True)
            except Exception as exc:  # noqa: BLE001 - a missing curve must not stop training
                print(f"[eval] ever-success state not restored from {ever_src}: {exc!r}", flush=True)

    # ─── training loop ───────────────────────────────────────────────────────────

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        del init_at_random_ep_len
        agent, n_envs = self._agent, self._n_envs
        n_steps = int(num_learning_iterations)

        self._install_stop_signal()
        self._start_clock()
        # FDR_ITER_TIMING_FILE=<csv>: collect+update wall time per iteration for the GPU
        # packing probes, like the on-policy runner.
        timing_path = os.environ.get("FDR_ITER_TIMING_FILE")
        timing_f = open(timing_path, "w") if timing_path else None
        if timing_f:
            timing_f.write("interaction_step,env_step,iter_s,training\n")
        meter = AverageMeter()
        update_budget = 0.0
        solved_stop = False

        for step in tqdm.tqdm(range(1, n_steps + 1), smoothing=0.1, mininterval=0.5):
            env_step = self._step_offset + step * n_envs
            if timing_f:
                torch.cuda.synchronize()
                iter_t0 = time.perf_counter()

            self._collect(step, meter)
            if not agent.ready():
                continue

            update_budget += self._updates_per_step
            while update_budget >= 1:
                train_info = agent.update()
                meter.update(train_info)
                update_budget -= 1
            if timing_f:
                torch.cuda.synchronize()
                timing_f.write(f"{step},{env_step},{time.perf_counter() - iter_t0:.6f},1\n")
                timing_f.flush()

            if self._crossed(env_step, self._eval_stride) and self._eval_supported():
                eval_metrics = self._run_eval(env_step)
                log_metrics({**eval_metrics, **self._cost_metrics(env_step)}, step=env_step)
                if getattr(self._eval_env, "shares_train_env", False):
                    self._reset_train_env()  # eval drove the train env: restart its episodes
                if self._all_solved(eval_metrics, env_step):
                    solved_stop = True
                    break

            if self._log_every and step % self._log_every == 0:
                log_metrics({**meter.mean(), **self._cost_metrics(env_step)}, step=env_step)
                meter.reset()

            if self._crossed(env_step, self._save_stride):
                self._save_checkpoint(env_step)

            if self._stop_requested:
                self._graceful_stop(env_step)

        final_step = self._step_offset + n_steps * n_envs
        if timing_f:
            timing_f.close()
        # An all-solved stop has just evaluated and exported, so it skips the final sweep.
        if self._eval_supported() and not solved_stop:
            log_metrics({**self._run_eval(final_step), **self._cost_metrics(final_step)}, step=final_step)
        self._finish()

    # ─── loop stages ─────────────────────────────────────────────────────────────

    def _crossed(self, env_step: int, stride_env: int) -> bool:
        """True on the iteration that crosses a multiple of ``stride_env`` env steps.
        Absolute in env steps, so a resume (``step_offset``) cannot shift the grid."""
        if stride_env <= 0:
            return False
        return (env_step // stride_env) != ((env_step - self._n_envs) // stride_env)

    def _collect(self, step: int, meter: AverageMeter) -> None:
        """Step every train env once and hand the transition to the agent."""
        agent, env = self._agent, self._train_env
        if agent.ready() and not self._just_reset:
            actions = agent.act(self._observations)
        else:
            actions = env.action_space.sample()
        actions = np.array(actions)

        next_obs, rewards, terminateds, truncateds, infos = env.step(actions)

        # Truncated envs bootstrap on their true final obs, not the post-reset one.
        buffer_next_obs = next_obs.copy()
        for env_idx in range(self._n_envs):
            if truncateds[env_idx]:
                buffer_next_obs[env_idx] = infos["final_obs"][env_idx]
        self._clamp_nonfinite(step, rewards, buffer_next_obs, next_obs)

        if "episode_info" in infos:
            meter.update(infos["episode_info"])

        transition = {
            "observation": self._observations,
            "action": actions,
            "reward": rewards,
            "terminated": terminateds,
            "truncated": truncateds,
            "next_observation": buffer_next_obs,
        }
        if self._obs_aux is not None:
            next_aux = infos["obs_aux"].copy()
            next_aux[truncateds] = infos["final_obs_aux"][truncateds]
            transition["observation_aux"] = self._obs_aux
            transition["next_observation_aux"] = next_aux
            self._obs_aux = infos["obs_aux"]
        agent.observe(transition)
        self._observations = next_obs
        self._just_reset = False

    def _clamp_nonfinite(self, step: int, rewards, buffer_next_obs, next_obs) -> None:
        """One NaN/Inf in the buffer resurfaces in later samples and crashes the categorical
        critic's target projection, so the transition is sanitised in place."""
        bad = 0
        for bit, arr in ((1, rewards), (2, buffer_next_obs), (4, next_obs)):
            if not np.isfinite(arr).all():
                bad |= bit
                np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        if bad:
            self._nonfinite_steps += 1
            if self._nonfinite_steps == 1 or self._nonfinite_steps % 500 == 0:
                print(
                    f"[runner] clamped non-finite transition "
                    f"(mask {bad}, hit #{self._nonfinite_steps}) at interaction_step={step}"
                )

    def _reset_train_env(self) -> None:
        self._observations, reset_infos = self._train_env.reset()
        self._obs_aux = reset_infos.get("obs_aux", self._obs_aux)
        self._just_reset = True

    # ─── eval / logging ──────────────────────────────────────────────────────────

    def _eval_supported(self) -> bool:
        # False for isaac per-env-object mode, which cannot honor the sweep's motion->env map.
        return getattr(self._eval_env, "eval_supported", True)

    def _run_eval(self, env_step: int) -> dict[str, float]:
        return evaluate(
            self._agent, self._eval_env, self._eval_cfg, self._device,
            ever_state=self._eval_ever_state, success_dir=self._success_dir,
            env_step=env_step,
        )

    def _all_solved(self, eval_metrics: dict[str, float], env_step: int) -> bool:
        """eval.stop_when_all_solved: the success archive covers every clip."""
        if not bool(self._eval_cfg.get("stop_when_all_solved", True)):
            return False
        archived_frac = eval_metrics.get("eval/archived_frac", eval_metrics.get("archived_frac", 0.0))
        if archived_frac < 1.0:
            return False
        n = int(eval_metrics.get("eval/archived_motions", eval_metrics.get("archived_motions", 0)))
        print(
            f"[runner] all {n} motion(s) archived (eval.success_criterion); "
            f"stopping at env_step={env_step} (eval.stop_when_all_solved)",
            flush=True,
        )
        wandb.run.summary["stopped_all_solved"] = True
        wandb.run.summary["stopped_env_step"] = int(env_step)
        return True

    def _start_clock(self) -> None:
        # A resumed wandb run continues its logged wall clock.
        self._wall_offset = 0.0
        if getattr(wandb.run, "resumed", False):
            self._wall_offset = float(wandb.run.summary.get("time/wall_clock_s", 0.0) or 0.0)
        self._loop_start = time.perf_counter()

    def _cost_metrics(self, step: int) -> dict[str, float]:
        elapsed = time.perf_counter() - self._loop_start
        return {
            "time/env_step": float(step),
            "time/wall_clock_s": self._wall_offset + elapsed,
            "time/env_steps_per_s": step / max(elapsed, 1e-9),
        }

    # ─── checkpoints / stopping ──────────────────────────────────────────────────

    def _save_checkpoint(self, env_step: int) -> None:
        """Written to .tmp and renamed, so a failed write (e.g. full GPFS) leaves no corrupt
        step dir; a failure is logged and training continues."""
        base = self._save_path_base
        tmp, ckpt = base / f"step{env_step}.tmp", base / f"step{env_step}"
        save_buffer = bool(self._cfg.get("save_replay_buffer", False))
        try:
            self._agent.save(str(tmp))
            if save_buffer:
                self._agent.save_buffer(str(tmp))
            os.replace(tmp, ckpt)
            self._log_resume_state(ckpt, env_step)
            if bool(self._cfg.get("keep_last_checkpoint_only", False)):
                self._prune_checkpoints(keep=ckpt.name)
            elif save_buffer:
                self._prune_buffers(keep=ckpt.name)
        except Exception as exc:
            shutil.rmtree(tmp, ignore_errors=True)
            print(f"[runner] checkpoint save failed, continuing: {exc}")

    def _prune_checkpoints(self, keep: str) -> None:
        """runner.keep_last_checkpoint_only: drop every step dir but ``keep`` (nets + buffer)."""
        for old in self._save_path_base.glob("step*"):
            if old.is_dir() and old.name != keep:
                shutil.rmtree(old, ignore_errors=True)

    def _prune_buffers(self, keep: str) -> None:
        """Drop replay_buffer.pt from every checkpoint dir but ``keep`` (they are tens of GB)."""
        for old in self._save_path_base.glob("step*/replay_buffer.pt"):
            if old.parent.name != keep:
                try:
                    old.unlink()
                except OSError as exc:
                    print(f"[runner] could not drop {old}: {exc}")

    def _install_stop_signal(self) -> None:
        """`kill -USR1 <pid>`: finish the current iteration, checkpoint WITH the replay
        buffer and exit, so a resume does not refill the buffer."""
        self._stop_requested = False

        def _request_stop(*_args) -> None:
            self._stop_requested = True
            print("[runner] SIGUSR1: stopping after this iteration (saving buffer)", flush=True)

        try:
            signal.signal(signal.SIGUSR1, _request_stop)
        except ValueError:  # not the main thread
            pass

    def _graceful_stop(self, env_step: int) -> None:
        ckpt = self._save_path_base / f"stop{env_step}"
        self._agent.save(str(ckpt))
        self._agent.save_buffer(str(ckpt))
        self._log_resume_state(ckpt, env_step)
        print(f"[runner] graceful stop at env_step={env_step}: {ckpt}", flush=True)
        try:
            wandb.finish()
        except Exception:  # noqa: BLE001
            pass
        os._exit(0)

    def _finish(self) -> None:
        try:
            self._agent.save(str(self._save_path_base / "final"))
        except Exception as exc:
            print(f"[runner] final checkpoint save failed: {exc}")
        # The buffer only serves a resume; a run that ends normally has nothing to resume.
        if bool(self._cfg.get("delete_replay_buffer_on_finish", False)):
            self._prune_buffers(keep="")
            print(f"[runner] finished: dropped replay_buffer.pt under {self._save_path_base}", flush=True)

    # ─── resume state ────────────────────────────────────────────────────

    def _curriculum_state(self) -> dict[str, Any]:
        """Assist gains, as resume overrides read them."""
        xf = getattr(self._train_cmd, "_xfrc_curr_ctrl", None)
        return {
            "xfrc": None if xf is None else {
                "kp": float(xf.curr_gains["kp"]), "kv": float(xf.curr_gains["kv"]),
                "fr": float(xf.curr_gains["fr"]),
                "kp_lower": float(xf.curr_gains_lower["kp"]),
                "kv_lower": float(xf.curr_gains_lower["kv"]),
                "epochs_since_decay": int(xf.num_epoch_since_last_decay),
            },
        }

    def _log_resume_state(self, ckpt: Path, env_step: int) -> None:
        """Write the overrides a resume needs (ckpt, step, wandb id, curriculum state) next
        to the checkpoint."""
        state = {
            "env_step": int(env_step),
            "ckpt": str(ckpt),
            "wandb_id": str(wandb.run.id) if wandb.run is not None else None,
            **self._curriculum_state(),
            "ever": {k: v.to(torch.uint8).cpu().tolist() for k, v in self._eval_ever_state.items()
                     if torch.is_tensor(v)},
        }
        try:
            (ckpt / "resume_state.json").write_text(json.dumps(state))
        except Exception as exc:  # noqa: BLE001
            print(f"[runner] resume-state dump failed: {exc}")

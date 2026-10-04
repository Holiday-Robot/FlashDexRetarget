from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from mjlab.rl import RslRlVecEnvWrapper
from omegaconf import DictConfig

from agents import Agent
from evaluation.callbacks.success_export import export_success


def evaluate(
    agent: Agent,
    eval_env: RslRlVecEnvWrapper,
    eval_cfg: DictConfig,
    device: str,
    ever_state: dict | None = None,
    success_dir: Path | None = None,
    env_step: int = 0,
) -> dict[str, float]:
    """Roll the deterministic policy through every eval batch, then turn the callback
    states into metrics."""
    raw_eval = eval_env.unwrapped
    # RolloutRecord only feeds the success archive; skip its capture when not saving.
    save_success = bool(eval_cfg.get("save_success_data", True))
    callbacks = [
        instantiate(cb) for cb in eval_cfg.callbacks
        if save_success or "rollout_record" not in str(cb.get("_target_", ""))
    ]

    states = _rollout(agent, eval_env, callbacks, device)

    metrics = reduce_metrics(callbacks, states)
    metrics.update(tracking_success_metrics(
        raw_eval, eval_cfg, callbacks, states, ever_state,
        success_dir if save_success else None, env_step,
    ))
    return {f"eval/{k}": float(v) for k, v in metrics.items() if isinstance(v, (int, float))}


# ─── rollout ───────────────────────────────────────────────────────────────────
def _rollout(
    agent: Agent, eval_env: RslRlVecEnvWrapper, callbacks: list, device: str
) -> dict[int, list]:
    """Reset + deterministic inference over every eval batch;
    returns each metric callback's per-batch states, keyed by id(callback)."""
    raw_eval = eval_env.unwrapped
    setup_cb = next(cb for cb in callbacks if hasattr(cb, "set_batch"))
    metric_cbs = [cb for cb in callbacks if hasattr(cb, "reduce_state")]
    eval_k, num_motions = raw_eval.eval_k, raw_eval.eval_num_motions
    states: dict[int, list] = {id(cb): [] for cb in metric_cbs}
    with torch.no_grad():
        for b in range(raw_eval.eval_num_batches):
            setup_cb.set_batch(b, eval_k, num_motions)
            for cb in callbacks:
                if hasattr(cb, "on_eval_setup"):
                    cb.on_eval_setup(raw_eval)
            obs, _ = eval_env.reset()
            for cb in callbacks:
                cb.on_start(raw_eval)
            enc_keys = sorted(key for key in obs.keys() if key not in ("actor", "critic"))
            n_steps = max(1, max((int(getattr(cb, "rollout_steps", 0)) for cb in callbacks), default=1))
            active_mask = setup_cb.active_mask
            for _ in range(n_steps):
                actions = agent.act(_flat_actor_obs(obs, enc_keys), deterministic=True)
                obs, _, _, _ = eval_env.step(torch.as_tensor(np.asarray(actions), device=device))
                for cb in callbacks:
                    cb.on_step(raw_eval)
            for cb in metric_cbs:
                states[id(cb)].append(cb.collect_state(active_mask))
    if hasattr(eval_env, "exit_eval"):
        eval_env.exit_eval()
    return states


def _flat_actor_obs(obs: dict, enc_keys: list[str]) -> torch.Tensor:
    """Actor group + encoded (window) groups, appended exactly like FlatObsVecEnv._flatten_obs."""
    if not enc_keys:
        return obs["actor"]
    return torch.cat([obs["actor"]] + [obs[k] for k in enc_keys], dim=-1)


# ─── metrics ───────────────────────────────────────────────────────────────────
def reduce_metrics(callbacks: list, states: dict[int, list]) -> dict[str, float]:
    """reduce_state of every metric callback, then get_metrics of the rest."""
    metrics: dict[str, float] = {}
    for cb in callbacks:
        if id(cb) in states:
            metrics.update(cb.reduce_state(states[id(cb)]))
    for cb in callbacks:
        if id(cb) not in states and hasattr(cb, "get_metrics"):
            metrics.update(cb.get_metrics())
    return metrics


def tracking_success_metrics(
    raw_eval: Any,
    eval_cfg: DictConfig,
    callbacks: list,
    states: dict[int, list],
    ever_state: dict | None,
    success_dir: Path | None,
    env_step: int,
) -> dict[str, float]:
    """The success-clip archive and the cumulative ever-success curves."""
    # per_env_fail_obj alone would also match GtMocapPerformance, which has no per_env_fail.
    tp_cb = _metric_cb(callbacks, states, "per_env_fail")
    if tp_cb is None:
        return {}
    st = states[id(tp_cb)]
    fail = tp_cb.per_env_fail(st)  # (E, K), motion-major
    fail_obj = tp_cb.per_env_fail_obj(st)
    scored = tp_cb.per_env_scored(st)
    metrics: dict[str, float] = {}

    rec_cb = next(
        (cb for cb in callbacks if hasattr(cb, "rollout_arrays") and cb is not tp_cb), None
    )
    if success_dir is not None and rec_cb is not None:
        metrics.update(_archive_success(
            raw_eval, eval_cfg, tp_cb, rec_cb, states, fail, fail_obj, scored, success_dir, env_step
        ))
    if ever_state is not None:
        metrics.update(_ever_success(eval_cfg, tp_cb, fail, fail_obj, scored, ever_state))
    metrics.update(_spider_ever(raw_eval, tp_cb, st, scored, ever_state))
    return metrics


# ─── tracking success parts ────────────────────────────────────────────────────
def _archive_success(
    raw_eval: Any,
    eval_cfg: DictConfig,
    tp_cb: Any,
    rec_cb: Any,
    states: dict[int, list],
    fail: torch.Tensor,
    fail_obj: torch.Tensor,
    scored: torch.Tensor,
    success_dir: Path,
    env_step: int,
) -> dict[str, float]:
    """Archive the clips that pass SR@1.0 (success_export.py). A side artefact: it must never
    take a multi-day run down."""
    out: dict[str, float] = {}
    try:
        lib = raw_eval.command_manager.get_term(eval_cfg.command_name).motion_lib
        # "full" also demands the finger criterion; "obj" is object-only.
        crit = str(eval_cfg.get("success_criterion", "obj"))
        failed = (fail if crit == "full" else fail_obj)[:, tp_cb.k1_idx]
        export_success(
            success_dir,
            rec_cb,
            states[id(rec_cb)],
            ok=(~failed) & scored,
            motion_ids=torch.cat([s["motion_ids"] for s in states[id(rec_cb)]]),
            env_step=env_step,
            criterion=f"{crit} SR@1.0",
            motion_names=_motion_names(lib),
            motion_file=getattr(lib, "motion_file", None),
        )
        out.update(_archive_coverage(success_dir, int(lib.num_trajectories)))
    except Exception as exc:  # noqa: BLE001 - artefact must not kill training
        print(f"[success] export failed, continuing: {exc!r}", flush=True)
        traceback.print_exc()
    return out


def _archive_coverage(success_dir: Path, total: int) -> dict[str, float]:
    """Share of motions the success archive holds (drives eval.stop_when_all_solved)."""
    manifest = Path(success_dir) / "manifest.json"
    n = len(json.loads(manifest.read_text()).get("saved", [])) if manifest.exists() else 0
    return {"archived_motions": float(n), "archived_frac": float(n) / max(1, total)}


def _ever_success(
    eval_cfg: DictConfig,
    tp_cb: Any,
    fail: torch.Tensor,
    fail_obj: torch.Tensor,
    scored: torch.Tensor,
    ever_state: dict,
) -> dict[str, float]:
    """Cumulative per-trajectory "ever succeeded": a trajectory counts once ANY replica in
    ANY eval so far succeeded (full / object-only)."""
    npm = max(1, int(eval_cfg.num_per_motion))
    n_traj = fail.shape[0] // npm
    if n_traj * npm != fail.shape[0]:
        print(f"[eval] ever-success skipped: {fail.shape[0]} envs % npm {npm} != 0")
        return {}

    def traj_any(f: torch.Tensor) -> torch.Tensor:
        return ((~f) & scored[:, None]).reshape(n_traj, npm, -1).any(dim=1)  # (M, K)

    def accumulate(key: str, succ: torch.Tensor) -> torch.Tensor:
        prev = ever_state.get(key)
        if prev is None or tuple(prev.shape) != tuple(succ.shape):
            if prev is not None:
                print(f"[eval] ever '{key}' shape {tuple(prev.shape)} != {tuple(succ.shape)}: reset", flush=True)
            prev = torch.zeros_like(succ)
        ever_state[key] = prev.to(succ.device) | succ
        return ever_state[key]

    full = accumulate("full", traj_any(fail))
    obj = accumulate("obj", traj_any(fail_obj))

    metrics: dict[str, float] = {}
    for i, kk in enumerate(tp_cb.threshold_ks):
        metrics[f"success_ever_{kk:.1f}"] = float(full[:, i].float().mean().item())
        metrics[f"success_ever_obj_{kk:.1f}"] = float(obj[:, i].float().mean().item())
    print(
        f"[eval] success_ever@1.0 full={metrics['success_ever_1.0']:.4f} "
        f"obj={metrics['success_ever_obj_1.0']:.4f} "
        f"({int(ever_state['full'][:, tp_cb.k1_idx].sum())}/{n_traj} trajs)",
        flush=True,
    )
    return metrics


def _spider_ever(
    raw_eval: Any, tp_cb: Any, st: list, scored: torch.Tensor, ever_state: dict | None
) -> dict[str, float]:
    """SPIDER ever, scattered by motion id: Isaac's per-env-object eval plan is not
    motion-major (env index != motion index), so the reshape in _ever_success does not apply."""
    sp_ok = tp_cb.per_env_spider_ok(st) if hasattr(tp_cb, "per_env_spider_ok") else None
    if ever_state is None or sp_ok is None:
        return {}
    mids = tp_cb.per_env_motion_ids(st)
    succ = torch.zeros(int(raw_eval.eval_num_motions), dtype=torch.bool, device=sp_ok.device)
    succ[mids[sp_ok & scored]] = True
    prev = ever_state.get("spider")
    if prev is None or tuple(prev.shape) != tuple(succ.shape):
        prev = torch.zeros_like(succ)
    ever_state["spider"] = prev.to(succ.device) | succ
    return {"success_ever_spider": float(ever_state["spider"].float().mean().item())}


# ─── helpers ───────────────────────────────────────────────────────────────────
def _metric_cb(callbacks: list, states: dict[int, list], attr: str) -> Any | None:
    """First metric callback (one with collected states) that exposes ``attr``."""
    return next((cb for cb in callbacks if id(cb) in states and hasattr(cb, attr)), None)


def _motion_names(lib: Any) -> list[str] | None:
    return [f.rsplit("#", 1)[-1] for f in getattr(lib, "motion_files", [])] or None

"""Isaac eval on the train env: SharedEvalEnv, and the object-aware sweep for per-env-object
mode (each motion's replicas run only on envs that physically host that motion's object)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from evaluation.callbacks.motion_tracking_eval_setup import MotionTrackingEvalSetup

if TYPE_CHECKING:
    from simulator.isaacsim.env import IsaacManagerBasedRlEnv


def build_eval_plan(
    assigned_slot: torch.Tensor,
    traj_obj_slot: torch.Tensor,
    num_per_motion: int,
) -> list[dict[str, torch.Tensor]]:
    """Batches of {motion_ids (B,), active (B,)} covering every motion
    num_per_motion times (capped by each object's env count)."""
    device = assigned_slot.device
    B = assigned_slot.shape[0]
    S = int(traj_obj_slot.max().item()) + 1
    env_of = [torch.where(assigned_slot == s)[0] for s in range(S)]
    mot_of = [torch.where(traj_obj_slot.to(device) == s)[0] for s in range(S)]

    n_batches = 0
    reps, caps = [], []
    for s in range(S):
        n_env = max(1, env_of[s].numel())
        rep = min(num_per_motion, n_env)
        cap = max(1, n_env // rep)
        reps.append(rep)
        caps.append(cap)
        if mot_of[s].numel():
            n_batches = max(
                n_batches, (mot_of[s].numel() + cap - 1) // cap
            )

    plan = []
    for b in range(n_batches):
        motion_ids = torch.zeros(B, dtype=torch.long, device=device)
        active = torch.zeros(B, dtype=torch.bool, device=device)
        for s in range(S):
            m = mot_of[s]
            if not m.numel() or not env_of[s].numel():
                continue
            chunk = m[b * caps[s] : (b + 1) * caps[s]]
            if not chunk.numel():
                continue
            slots = env_of[s][: chunk.numel() * reps[s]]
            motion_ids[slots] = chunk.repeat_interleave(reps[s])[: slots.numel()]
            active[slots] = True
        plan.append({"motion_ids": motion_ids, "active": active})
    return plan


class IsaacMotionTrackingEvalSetup(MotionTrackingEvalSetup):
    """Reads the precomputed object-aware plan off the env (per-env mode)."""

    def on_eval_setup(self, env: Any) -> None:
        plan = getattr(env, "_isaac_eval_plan", None)
        if plan is None:
            return super().on_eval_setup(env)
        cmd = env.command_manager.get_term(self.command_name)
        batch = plan[self.batch_idx % len(plan)]
        cmd.motion_ids[:] = batch["motion_ids"]
        self.active_mask = batch["active"]
        active_ids = cmd.motion_ids[self.active_mask]
        max_frames = int(cmd.motion_lib._motion_num_frames[active_ids].max().item())
        self.rollout_steps = max(1, max_frames - 1 - self.start_frame)


class _NullTerminationManager:
    """Eval-time stand-in: no terminations, so rollouts run uninterrupted."""

    def __init__(self, num_envs: int, device: str) -> None:
        self._zeros = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.active_terms: list = []

    def compute(self) -> torch.Tensor:
        return self._zeros

    @property
    def terminated(self) -> torch.Tensor:
        return self._zeros

    @property
    def time_outs(self) -> torch.Tensor:
        return self._zeros

    def reset(self, env_ids=None) -> dict:
        return {}


class SharedEvalEnv:
    """Eval facade over the single train env: enter_eval() pauses training
    semantics (terminations/pin/curriculum), exit_eval() restores them."""

    def __init__(self, env: IsaacManagerBasedRlEnv, eval_cfg) -> None:
        self.env = env
        self._eval_cfg = eval_cfg
        self._in_eval = False
        self._stash: dict = {}
        self.shares_train_env = True

        cmd = env.command_manager.get_term(eval_cfg.command_name)
        num_motions = int(cmd.motion_lib.num_trajectories)
        num_per_motion = int(eval_cfg.num_per_motion)
        plan = getattr(env, "_isaac_eval_plan", None)
        if plan is not None:
            env.eval_k = 0
            env.eval_num_batches = len(plan)
        else:
            k = max(1, min(num_motions, env.num_envs // max(1, num_per_motion)))
            env.eval_k = k
            env.eval_num_batches = (num_motions + k - 1) // k
        env.eval_num_motions = num_motions

    @property
    def unwrapped(self) -> IsaacManagerBasedRlEnv:
        return self.env

    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    @property
    def device(self) -> str:
        return self.env.device

    def enter_eval(self) -> None:
        if self._in_eval:
            return
        env = self.env
        cmd = env.command_manager.get_term(self._eval_cfg.command_name)
        cmd.set_eval_mode(
            sampling_mode=self._eval_cfg.command.sampling_mode,
            noise_to_initial_level=self._eval_cfg.command.noise_to_initial_level,
            start_frame=int(self._eval_cfg.command.start_frame),
        )
        self._stash = {
            "term": env.termination_manager,
            "curr": env.curriculum_manager,
            "ctrl": cmd._xfrc_curr_ctrl,
            "pin": cmd.cfg.object.pin_objects,
        }
        env.termination_manager = _NullTerminationManager(env.num_envs, env.device)
        from simulator.isaacsim.env import _NullManager

        env.curriculum_manager = _NullManager()
        cmd._xfrc_curr_ctrl = None
        cmd.cfg.object.pin_objects = False
        # External wrenches persist in IsaacLab: clear any stale pin wrench so
        # eval rollouts are strictly assist-free.
        for side in cmd._side_list:
            for name in cmd.obj_entity_names(side):
                obj = env.scene[name]
                z = torch.zeros(env.num_envs, 1, 3, device=env.device)
                obj.write_external_wrench_to_sim(forces=z, torques=z.clone())
        self._in_eval = True

    def exit_eval(self) -> None:
        if not self._in_eval:
            return
        env = self.env
        cmd = env.command_manager.get_term(self._eval_cfg.command_name)
        env.termination_manager = self._stash["term"]
        env.curriculum_manager = self._stash["curr"]
        cmd._xfrc_curr_ctrl = self._stash["ctrl"]
        cmd.cfg.object.pin_objects = self._stash["pin"]
        cmd.set_train_mode()
        self._in_eval = False
        env.reset()  # reseed fresh TRAIN episodes for the resumed loop

    def reset(self):
        self.enter_eval()
        return self.env.reset()

    def step(self, actions: torch.Tensor):
        obs, rew, terminated, truncated, extras = self.env.step(actions)
        dones = (terminated | truncated).to(dtype=torch.long)
        extras["time_outs"] = truncated
        return obs, rew, dones, extras

    def close(self) -> None:
        pass

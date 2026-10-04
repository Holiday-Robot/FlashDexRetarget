from __future__ import annotations

import math
from typing import Any, cast

import torch

from envs.commands.motion_tracking import MotionTrackingCommand


class TrackingPerformance:
    """Per-step tracking errors and SR@k threshold scoring."""

    def __init__(
        self,
        grace_steps: int,
        threshold_ks: list[float],
        thresholds: dict[str, float],
        command_name: str,
        obj_criteria: list[list[float]] | None = None,
        spider_pos_m: float = 0.10,
        spider_rot_rad: float = 0.5,
    ) -> None:
        self.grace_steps = grace_steps
        # SPIDER / do-as-i-do object criterion (postprocess/get_success_rate.py), see
        # Whole clip, mean-centred position, sides averaged.
        self.spider_pos_m = float(spider_pos_m)
        self.spider_rot_rad = float(spider_rot_rad)
        ks = list(threshold_ks)
        if 1.0 not in ks:
            ks.append(1.0)
        self.threshold_ks = sorted(set(ks))
        self.k1_idx = self.threshold_ks.index(1.0)
        self.thresholds = dict(thresholds)
        self.command_name = command_name
        # Object-only criteria at ABSOLUTE (trans_m, rot_deg) pairs, scored next to the
        # k-scaled ones -- k scales translation and rotation together, so a loose-position
        # criterion at the same 30 deg needs its own entry.
        self.obj_criteria = [
            (float(t), float(r))
            for t, r in (obj_criteria if obj_criteria is not None else [(0.10, 30.0)])
        ]

    def on_start(self, env: Any) -> None:
        self.cmd = cast(
            MotionTrackingCommand, env.command_manager.get_term(self.command_name)
        )
        self.device = env.device
        self.num_envs = env.num_envs
        self.side_prefixes = tuple(s[0] for s in self.cmd._side_list)
        self.num_sides = len(self.side_prefixes)
        self.finger_names = tuple(self.cmd.finger_names)
        self.num_fingers = len(self.finger_names)
        self.max_motion_frames = int(
            self.cmd.motion_lib._motion_num_frames.max().item()
        )
        self.valid_steps_per_env = (
            self.cmd.motion_num_frames - self.cmd.motion_steps
        ).clamp(min=0)
        self.rollout_steps = max(1, int(self.valid_steps_per_env.max().item()))
        self.eval_end = max(self.grace_steps + 1, self.rollout_steps)
        self.threshold_ks_tensor = torch.tensor(self.threshold_ks, device=self.device)

        # Auto-discover per-side error keys from cmd.metrics: per-side scalars end
        # with `_{p}`; per-finger `error_tip_trans_{p}_{finger}` tracked separately.
        sample_p = self.side_prefixes[0]
        self.per_side_keys: list[str] = []
        has_tip = False
        for key in self.cmd.metrics.keys():
            if key.startswith(f"error_tip_trans_{sample_p}_"):
                has_tip = True
                continue
            if key.endswith(f"_{sample_p}"):
                self.per_side_keys.append(key[: -(len(sample_p) + 1)])

        self.metric_sums: dict[str, torch.Tensor] = {
            base: torch.zeros(self.num_envs, self.num_sides, device=self.device)
            for base in self.per_side_keys
        }
        self.tip_trans_sums = (
            torch.zeros(
                self.num_envs, self.num_sides, self.num_fingers, device=self.device
            )
            if has_tip
            else None
        )
        self.action_rate_sum = torch.zeros(self.num_envs, device=self.device)
        self.count_per_env = torch.zeros(self.num_envs, device=self.device)
        self.t = 0

        # SPIDER needs the per-step object offset series (its position error removes each
        # trajectory's own time-mean, which no running sum can give); rot/drift are sums.
        self.has_obj = f"error_obj_trans_{sample_p}" in self.cmd.metrics
        if self.has_obj:
            self.spider_diff = torch.zeros(
                self.rollout_steps, self.num_envs, self.num_sides, 3, device=self.device
            )
            self.spider_rot_sum = torch.zeros(
                self.num_envs, self.num_sides, device=self.device
            )
            self.spider_drift_sum = torch.zeros_like(self.spider_rot_sum)
            self.spider_count = torch.zeros(self.num_envs, device=self.device)
            # cmd.metrics compare the object with the frame seen at the PREVIOUS callback
            # (the reset frame at t=0): they are computed before motion_steps advances.
            self.spider_prev_ref = self.cmd.ref_obj_trans_w.clone()
            self.spider_ref0 = self.spider_prev_ref.clone()

    def on_step(self, env: Any) -> None:
        in_window = (
            (self.t < self.valid_steps_per_env)
            & (self.t >= self.grace_steps)
            & (self.t < self.eval_end)
        )
        if self.has_obj and self.t < self.spider_diff.shape[0]:
            self._spider_step(self.t)
        self.t += 1
        if not torch.any(in_window):
            return

        mask = in_window.to(torch.float32)
        cmd = self.cmd
        for si, p in enumerate(self.side_prefixes):
            for base in self.per_side_keys:
                key = f"{base}_{p}"
                if key in cmd.metrics:
                    self.metric_sums[base][:, si] += cmd.metrics[key] * mask
            if self.tip_trans_sums is not None:
                for fi, finger in enumerate(self.finger_names):
                    key = f"error_tip_trans_{p}_{finger}"
                    if key in cmd.metrics:
                        self.tip_trans_sums[:, si, fi] += cmd.metrics[key] * mask

        am = env.action_manager
        self.action_rate_sum += (
            torch.sum(torch.square(am.action - am.prev_action), dim=1) * mask
        )
        self.count_per_env += mask

    def _spider_step(self, t: int) -> None:
        """Whole-clip (no grace) accumulation for the SPIDER criterion at step t."""
        cmd = self.cmd
        m_all = (t < self.valid_steps_per_env).to(torch.float32)
        self.spider_diff[t] = (cmd.sim_obj_trans_w - self.spider_prev_ref) * m_all[:, None, None]
        self.spider_count += m_all
        rot = torch.stack(
            [cmd.metrics[f"error_obj_rot_{p}"] for p in self.side_prefixes], dim=-1
        )
        self.spider_rot_sum += rot * m_all[:, None]
        self.spider_drift_sum += (
            (self.spider_prev_ref - self.spider_ref0).norm(dim=-1) * m_all[:, None]
        )
        self.spider_prev_ref = cmd.ref_obj_trans_w.clone()

    def _spider_errors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-env SPIDER errors (pos m, rot rad), each (num_envs,): per-side position error
        after removing that trajectory's own time-mean offset; a side whose reference stays
        put (< 1 mm mean drift) is dropped, the remaining sides are averaged."""
        T = self.spider_diff.shape[0]
        valid = (
            torch.arange(T, device=self.device)[:, None] < self.valid_steps_per_env[None, :]
        ).to(torch.float32)  # (T, E)
        n = self.spider_count.clamp(min=1.0)
        dbar = self.spider_diff.sum(dim=0) / n[:, None, None]  # (E, S, 3)
        perr = ((self.spider_diff - dbar[None]).norm(dim=-1) * valid[:, :, None]).sum(
            dim=0
        ) / n[:, None]  # (E, S)
        rerr = self.spider_rot_sum / n[:, None]
        if self.num_sides == 1:
            return perr[:, 0], rerr[:, 0]
        static = (self.spider_drift_sum / n[:, None]) < 1e-3
        r = self.side_prefixes.index("r") if "r" in self.side_prefixes else 0
        l = 1 - r
        pos = torch.where(
            static[:, l], perr[:, r],
            torch.where(static[:, r], perr[:, l], 0.5 * (perr[:, r] + perr[:, l])),
        )
        rot = torch.where(
            static[:, l], rerr[:, r],
            torch.where(static[:, r], rerr[:, l], 0.5 * (rerr[:, r] + rerr[:, l])),
        )
        return pos, rot

    def collect_state(
        self, active_mask: torch.Tensor | None = None
    ) -> dict[str, Any]:
        """Per-env accumulators for the active envs, for cross-batch reduction.
        ``active_mask`` drops ragged-batch overflow envs; ``None`` keeps all (single-pass)."""
        sel = slice(None) if active_mask is None else active_mask
        sp_pos, sp_rot = self._spider_errors() if self.has_obj else (None, None)
        return {
            "spider_pos": sp_pos[sel] if sp_pos is not None else None,
            "spider_rot": sp_rot[sel] if sp_rot is not None else None,
            "motion_ids": self.cmd.motion_ids[sel].clone(),
            "metric_sums": {b: self.metric_sums[b][sel] for b in self.per_side_keys},
            "tip_trans_sums": (
                self.tip_trans_sums[sel] if self.tip_trans_sums is not None else None
            ),
            "action_rate_sum": self.action_rate_sum[sel],
            "count_per_env": self.count_per_env[sel],
            "valid_steps_per_env": self.valid_steps_per_env[sel],
        }

    def _concat(self, states: list[dict[str, Any]]) -> dict[str, Any]:
        """Concatenate per-batch per-env state along the env axis (motion order)."""
        has_tip = states[0]["tip_trans_sums"] is not None
        has_sp = states[0].get("spider_pos") is not None
        return {
            "spider_pos": (
                torch.cat([s["spider_pos"] for s in states], dim=0) if has_sp else None
            ),
            "spider_rot": (
                torch.cat([s["spider_rot"] for s in states], dim=0) if has_sp else None
            ),
            "motion_ids": torch.cat([s["motion_ids"] for s in states], dim=0),
            "metric_sums": {
                base: torch.cat([s["metric_sums"][base] for s in states], dim=0)
                for base in self.per_side_keys
            },
            "tip_trans_sums": (
                torch.cat([s["tip_trans_sums"] for s in states], dim=0)
                if has_tip
                else None
            ),
            "action_rate_sum": torch.cat(
                [s["action_rate_sum"] for s in states], dim=0
            ),
            "count_per_env": torch.cat([s["count_per_env"] for s in states], dim=0),
            "valid_steps_per_env": torch.cat(
                [s["valid_steps_per_env"] for s in states], dim=0
            ),
        }

    @staticmethod
    def obj_criterion_key(trans_m: float, rot_deg: float) -> str:
        """Metric name for an absolute object criterion, e.g. 10 cm / 30 deg."""
        return f"success_rate_obj_{trans_m * 100:g}cm_{rot_deg:g}deg"

    def _side_errors(
        self, cat: dict[str, Any], count_safe: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Per-env, PER-SIDE mean errors (num_envs, num_sides) feeding the SR criteria."""
        metric_sums = cat["metric_sums"]
        tip_trans_sums = cat["tip_trans_sums"]
        return {
            "obj_trans": metric_sums["error_obj_trans"] / count_safe[:, None],
            "obj_rot_deg": metric_sums["error_obj_rot"]
            * (180.0 / math.pi)
            / count_safe[:, None],
            "tip_trans": (
                tip_trans_sums.mean(dim=2) / count_safe[:, None]
                if tip_trans_sums is not None
                else torch.zeros(
                    int(count_safe.shape[0]), self.num_sides, device=self.device
                )
            ),
            "joint_trans": (
                # proximal + distal on every hand: the distal level is level3 when the mapping has one
                metric_sums["error_level1"] + metric_sums.get("error_level3", metric_sums["error_level2"])
            )
            * 0.5
            / count_safe[:, None],
        }

    def _fail_from(self, e: dict[str, torch.Tensor]) -> torch.Tensor:
        """SR@k failure flags (num_envs, num_k) from per-env error vectors (num_envs,)."""
        th = self.thresholds
        ks_t = self.threshold_ks_tensor
        return (
            (e["obj_trans"].unsqueeze(-1) > ks_t * th["obj_trans"])
            | (e["tip_trans"].unsqueeze(-1) > ks_t * th["tip_trans"])
            | (e["obj_rot_deg"].unsqueeze(-1) > ks_t * th["obj_rot_deg"])
            | (e["joint_trans"].unsqueeze(-1) > ks_t * th["joint_trans"])
        )

    def _fail_obj_from(self, e: dict[str, torch.Tensor]) -> torch.Tensor:
        """Object-only failure flags (obj trans+rot criterion), (num_envs, num_k)."""
        th = self.thresholds
        ks_t = self.threshold_ks_tensor
        return (e["obj_trans"].unsqueeze(-1) > ks_t * th["obj_trans"]) | (
            e["obj_rot_deg"].unsqueeze(-1) > ks_t * th["obj_rot_deg"]
        )

    def _per_env_fail(
        self, cat: dict[str, Any], count_safe: torch.Tensor
    ) -> torch.Tensor:
        """Per-env SR@k failure flags, shape (num_envs, num_k). Worst side scores —
        with two hands this is the BOTH-hands-succeed criterion."""
        se = self._side_errors(cat, count_safe)
        return self._fail_from({k: v.max(dim=-1).values for k, v in se.items()})

    def _per_env_fail_obj(
        self, cat: dict[str, Any], count_safe: torch.Tensor
    ) -> torch.Tensor:
        """Object-only SR@k failure flags (obj trans+rot criterion), (num_envs, num_k)."""
        se = self._side_errors(cat, count_safe)
        return self._fail_obj_from({k: v.max(dim=-1).values for k, v in se.items()})

    def per_env_fail(self, states: list[dict[str, Any]]) -> torch.Tensor:
        """Per-env SR@k fail flags over the concat (no scored filter), (num_envs, num_k).
        Motion-major: reshape to (M, num_per_motion, num_k) for per-trajectory SR."""
        cat = self._concat(states)
        count_safe = cat["count_per_env"].clamp(min=1.0)
        return self._per_env_fail(cat, count_safe)

    def per_env_fail_obj(self, states: list[dict[str, Any]]) -> torch.Tensor:
        """Object-only per-env fail flags, same layout as ``per_env_fail``."""
        cat = self._concat(states)
        count_safe = cat["count_per_env"].clamp(min=1.0)
        return self._per_env_fail_obj(cat, count_safe)

    def per_env_scored(self, states: list[dict[str, Any]]) -> torch.Tensor:
        """Envs that accumulated at least one in-window step, (num_envs,) bool."""
        return self._concat(states)["count_per_env"] > 0

    def _spider_ok(self, cat: dict[str, Any]) -> torch.Tensor | None:
        if cat["spider_pos"] is None:
            return None
        return (cat["spider_pos"] <= self.spider_pos_m) & (
            cat["spider_rot"] <= self.spider_rot_rad
        )

    def per_env_spider_ok(self, states: list[dict[str, Any]]) -> torch.Tensor | None:
        """Per-env SPIDER pass flags (num_envs,) bool, or None without objects."""
        return self._spider_ok(self._concat(states))

    def per_env_motion_ids(self, states: list[dict[str, Any]]) -> torch.Tensor:
        """Motion id per env in the same order as the per-env flags."""
        return self._concat(states)["motion_ids"]

    def reduce_state(self, states: list[dict[str, Any]]) -> dict[str, float]:
        """SR@k + per-side error metrics over the concatenation of per-batch states.
        Scored mask + fallback are recomputed over the concat (matches a single pass)."""
        cat = self._concat(states)
        metric_sums = cat["metric_sums"]
        tip_trans_sums = cat["tip_trans_sums"]
        action_rate_sum = cat["action_rate_sum"]
        count_per_env = cat["count_per_env"]
        valid_steps_per_env = cat["valid_steps_per_env"]
        num_envs = int(count_per_env.shape[0])

        scored = count_per_env > 0
        if not torch.any(scored):
            scored = torch.ones(num_envs, device=self.device, dtype=torch.bool)
        count_safe = count_per_env.clamp(min=1.0)

        fail = self._per_env_fail(cat, count_safe)
        sr = (~fail[scored]).float().mean(dim=0)

        # Object-only success: only the object's translation AND rotation within
        # k x threshold — the task-relevant criterion when finger pose isn't the goal.
        fail_obj = self._per_env_fail_obj(cat, count_safe)
        sr_obj = (~fail_obj[scored]).float().mean(dim=0)

        out: dict[str, float] = {}
        for i, k in enumerate(self.threshold_ks):
            out[f"success_rate_{k:.1f}"] = float(sr[i].item())
            out[f"success_rate_obj_{k:.1f}"] = float(sr_obj[i].item())

        # Absolute object criteria (e.g. 10 cm / 30 deg): same worst-side time-mean errors
        # as success_rate_obj_1.0, only the thresholds differ.
        se_abs = self._side_errors(cat, count_safe)
        for trans_m, rot_deg in self.obj_criteria:
            key = self.obj_criterion_key(trans_m, rot_deg)
            ok = (se_abs["obj_trans"].max(dim=-1).values <= trans_m) & (
                se_abs["obj_rot_deg"].max(dim=-1).values <= rot_deg
            )
            out[key] = float(ok[scored].float().mean().item())
            if self.num_sides > 1:
                for si, p in enumerate(self.side_prefixes):
                    ok_s = (se_abs["obj_trans"][:, si] <= trans_m) & (
                        se_abs["obj_rot_deg"][:, si] <= rot_deg
                    )
                    out[f"{key}_{p}"] = float(ok_s[scored].float().mean().item())

        # SPIDER (10 cm / 0.5 rad on the mean-centred, side-averaged whole-clip errors).
        sp_ok = self._spider_ok(cat)
        if sp_ok is not None:
            out["success_rate_spider"] = float(sp_ok[scored].float().mean().item())
            out["error_spider_pos"] = float(cat["spider_pos"][scored].mean().item())
            out["error_spider_rot"] = float(cat["spider_rot"][scored].mean().item())

        # Bimanual: also score each hand alone (the unsuffixed SR above is the
        # worst-side = both-hands-succeed criterion).
        if self.num_sides > 1:
            se = self._side_errors(cat, count_safe)
            for si, p in enumerate(self.side_prefixes):
                e_side = {k: v[:, si] for k, v in se.items()}
                sr_s = (~self._fail_from(e_side)[scored]).float().mean(dim=0)
                sr_obj_s = (~self._fail_obj_from(e_side)[scored]).float().mean(dim=0)
                for i, k in enumerate(self.threshold_ks):
                    out[f"success_rate_{k:.1f}_{p}"] = float(sr_s[i].item())
                    out[f"success_rate_obj_{k:.1f}_{p}"] = float(sr_obj_s[i].item())

        for base, sums in metric_sums.items():
            per_side_avg = (sums / count_safe[:, None])[scored].mean(dim=0)
            for si, p in enumerate(self.side_prefixes):
                out[f"{base}_{p}"] = float(per_side_avg[si].item())

        if tip_trans_sums is not None:
            per_finger_tip_avg = (tip_trans_sums / count_safe[:, None, None])[
                scored
            ].mean(dim=0)
            for si, p in enumerate(self.side_prefixes):
                for fi, finger in enumerate(self.finger_names):
                    out[f"error_tip_trans_{p}_{finger}"] = float(
                        per_finger_tip_avg[si, fi].item()
                    )

        out["action_rate_l2"] = float(
            (action_rate_sum[scored] / count_safe[scored]).mean().item()
        )
        out["num_envs"] = num_envs
        out["num_scored_envs"] = int(scored.sum().item())
        out["mean_valid_steps"] = float(valid_steps_per_env.float().mean().item())
        return out

    def get_metrics(self) -> dict[str, float]:
        return self.reduce_state([self.collect_state()])

    def on_end(self) -> None:
        m = self.get_metrics()
        print("-" * 72)
        for k in self.threshold_ks:
            flag = "  <- paper strict" if k == 1.0 else ""
            print(f"  SR @ k={k:.1f}: {m[f'success_rate_{k:.1f}'] * 100:6.2f}%{flag}")
        print(f"  action_rate_l2: {m['action_rate_l2']:.4f}")

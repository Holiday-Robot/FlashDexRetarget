"""Replay-buffer obs compaction: drop obs columns that are pure functions of (other stored
columns, motion id, motion step) and rebuild them on sample."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from mjlab.utils.lab_api.math import matrix_from_quat, normalize, quat_apply_inverse, quat_from_matrix

AUX_DIM = 2  # (motion_id, motion_step) per row, int32


@dataclass(frozen=True)
class TermSpan:
    name: str
    group: str
    start: int  # column offset in the FULL flat obs
    end: int
    params: dict[str, Any]
    scale: torch.Tensor | None  # broadcastable over the term's columns
    clip: tuple[float, float] | None
    has_noise: bool


def flat_obs_layout(env: Any, group_order: list[str]) -> list[TermSpan]:
    """Per-term column spans of the flat obs = concat of ``group_order`` groups in order."""
    om = env.observation_manager
    spans: list[TermSpan] = []
    off = 0
    for g in group_order:
        names = om.active_terms[g]
        dims = om.group_obs_term_dim[g]
        for name, dim in zip(names, dims, strict=True):
            cfg = om.get_term_cfg(g, name)
            n = 1
            for d in dim:
                n *= int(d)
            scale = cfg.scale
            if scale is not None:
                scale = torch.as_tensor(scale, dtype=torch.float32).reshape(-1)
                if scale.numel() not in (1, n):
                    raise ValueError(f"obs term {name!r}: scale has {scale.numel()} entries for {n} columns.")
            spans.append(
                TermSpan(
                    name=name,
                    group=g,
                    start=off,
                    end=off + n,
                    params={k: v for k, v in cfg.params.items() if k not in ("command_name", "entity_name", "asset_cfg")},
                    scale=scale,
                    clip=None if cfg.clip is None else (float(cfg.clip[0]), float(cfg.clip[1])),
                    has_noise=cfg.noise is not None,
                )
            )
            off += n
    return spans


class ObsCompactor:
    """Maps full flat obs <-> compact obs (reconstructible term spans removed) + int aux."""

    # term -> alternatives of source terms it is rebuilt from (besides the motion-lib lookups keyed
    # by aux); the first alternative whose sources are all stored wins. Object pose comes either
    # from obj_trans/quat_wrist or from the cube keypoints (obj_keypoint_trans_wrist, n_points >= 4).
    RECON_SOURCES: dict[str, tuple[tuple[str, ...], ...]] = {
        "obj_point_cloud_wrist": (("obj_trans_wrist", "obj_quat_wrist"), ("obj_keypoint_trans_wrist",)),
        "ref_future_obj_point_cloud_delta_wrist": (
            ("robot_wrist_pose_w", "obj_trans_wrist", "obj_quat_wrist"),
            ("robot_wrist_pose_w", "obj_keypoint_trans_wrist"),
        ),
        "ref_future_obj_traj_state_rel_wrist": ((
            "robot_wrist_pose_w",
            "obj_trans_wrist",
            "obj_quat_wrist",
            "obj_lin_vel_wrist",
            "obj_ang_vel_wrist",
        ),),
        "ref_future_obj_keypoint_traj_wrist": ((
            "robot_wrist_pose_w",
            "obj_keypoint_trans_wrist",
            "obj_keypoint_lin_vel_wrist",
        ),),
        "ref_future_mano_wrist_tips_traj_rel_wrist": (("robot_wrist_pose_w", "robot_tip_trans_wrist"),),
        # narrowed to what the listed components read by _recon_alternatives
        "ref_future_traj_wrist": ((
            "robot_wrist_pose_w",
            "obj_trans_wrist",
            "obj_quat_wrist",
            "obj_lin_vel_wrist",
            "obj_ang_vel_wrist",
            "robot_tip_trans_wrist",
        ),),
    }

    def __init__(
        self,
        command: Any,
        spans: list[TermSpan],
        full_dim: int,
        device: torch.device | str,
        terms: tuple[str, ...] | None = None,
    ) -> None:
        self._cmd = command
        self._device = torch.device(device)
        self._full_dim = int(full_dim)
        self._sides = list(command._side_list)
        self._obj_sides = [s for s in self._sides if s in command.motion_lib.obj_trans]
        by_name: dict[str, TermSpan] = {}
        for sp in spans:
            by_name.setdefault(sp.name, sp)  # first (actor) occurrence = source columns
        wanted = set(terms) if terms is not None else set(self.RECON_SOURCES)
        self._recon: list[TermSpan] = []
        self._recon_srcs: dict[str, tuple[str, ...]] = {}
        kp_span = by_name.get("obj_keypoint_trans_wrist")
        self._kp_params = {"cube_side": 0.2, "n_points": 4, **(kp_span.params if kp_span else {})}
        skipped: list[str] = []
        for sp in spans:
            if sp.name not in wanted or sp.name not in self.RECON_SOURCES:
                continue
            chosen, why = None, []
            for srcs in self._recon_alternatives(sp):
                missing = [s for s in srcs if s not in by_name]
                noisy = [s for s in srcs if s in by_name and by_name[s].has_noise]
                if "obj_keypoint_trans_wrist" in srcs and int(self._kp_params["n_points"]) < 4:
                    missing.append("obj_keypoint_trans_wrist(n_points<4)")
                if not missing and not noisy:
                    chosen = srcs
                    break
                why.append(f"missing={missing}, noisy={noisy}")
            if chosen is None or sp.has_noise:
                skipped.append(f"{sp.name}@{sp.group} ({'; '.join(why)}{'; noisy target' if sp.has_noise else ''})")
                continue
            self._recon.append(sp)
            self._recon_srcs[sp.name] = chosen
        if skipped:
            print(f"[obs_compact] NOT compacting: {skipped}")
        if not self._recon:
            raise ValueError("obs compaction enabled but no reconstructible term found in the obs layout.")

        drop = torch.zeros(self._full_dim, dtype=torch.bool)
        for sp in self._recon:
            drop[sp.start : sp.end] = True
        self._keep_idx = (~drop).nonzero().squeeze(-1).to(self._device)
        self._compact_dim = int(self._keep_idx.numel())
        # full column -> compact column (only valid for kept columns)
        full_to_compact = torch.full((self._full_dim,), -1, dtype=torch.long)
        full_to_compact[self._keep_idx.cpu()] = torch.arange(self._compact_dim)
        self._src: dict[str, tuple[torch.Tensor, torch.Tensor | None]] = {}
        for sp in self._recon:
            for s in self._recon_srcs[sp.name]:
                if s in self._src:
                    continue
                ssp = by_name[s]
                cols = full_to_compact[ssp.start : ssp.end]
                if (cols < 0).any():
                    raise ValueError(f"source term {s!r} was itself dropped; cannot reconstruct.")
                scale = None if ssp.scale is None else ssp.scale.to(self._device)
                self._src[s] = (cols.to(self._device), scale)
        self._recon_scale = [None if sp.scale is None else sp.scale.to(self._device) for sp in self._recon]
        self._builders: dict[str, Callable[..., torch.Tensor]] = {
            "obj_point_cloud_wrist": self._build_obj_point_cloud_wrist,
            "ref_future_obj_point_cloud_delta_wrist": self._build_future_obj_point_cloud_delta_wrist,
            "ref_future_obj_traj_state_rel_wrist": self._build_future_obj_traj_state_rel,
            "ref_future_obj_keypoint_traj_wrist": self._build_future_obj_keypoint_traj_wrist,
            "ref_future_mano_wrist_tips_traj_rel_wrist": self._build_future_mano_wrist_tips_traj_rel,
            "ref_future_traj_wrist": self._build_future_traj_rel,
        }
        names = [f"{sp.name}[{sp.end - sp.start}]" for sp in self._recon]
        print(f"[obs_compact] full {self._full_dim} -> compact {self._compact_dim} (+{AUX_DIM} int aux); rebuilt: {names}")

    # ── public API ───────────────────────────────────────────────────────────

    @property
    def full_dim(self) -> int:
        return self._full_dim

    @property
    def compact_dim(self) -> int:
        return self._compact_dim

    @property
    def aux_dim(self) -> int:
        return AUX_DIM

    def aux_from_command(self) -> torch.Tensor:
        """Current per-env (motion_id, motion_step) as (B, 2) int32 on the command device."""
        return torch.stack([self._cmd.motion_ids, self._cmd.motion_steps], dim=-1).to(torch.int32)

    def compact(self, full: torch.Tensor) -> torch.Tensor:
        return full.index_select(-1, self._keep_idx)

    @torch.no_grad()
    def expand(self, compact: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        """compact (N, Dc) float32 + aux (N, 2) int -> full (N, D) float32."""
        N = compact.shape[0]
        full = compact.new_zeros((N, self._full_dim))
        full[:, self._keep_idx] = compact
        mid = aux[:, 0].long()
        step = aux[:, 1].long()
        cache: dict[str, torch.Tensor] = {}
        for sp, scale in zip(self._recon, self._recon_scale, strict=True):
            ck = repr((sp.name, tuple(sorted(sp.params.items()))))
            if ck not in cache:
                cache[ck] = self._builders[sp.name](compact, mid, step, **sp.params)
            val = cache[ck]
            if sp.clip is not None:
                val = val.clamp(sp.clip[0], sp.clip[1])
            if scale is not None:
                val = val * scale
            full[:, sp.start : sp.end] = val
        return full

    def _recon_alternatives(self, sp: TermSpan) -> tuple[tuple[str, ...], ...]:
        """RECON_SOURCES[sp.name]; ref_future_traj_wrist keeps only the sources its components read."""
        alts = self.RECON_SOURCES[sp.name]
        if sp.name != "ref_future_traj_wrist":
            return alts
        comps = {str(c) for c in sp.params["components"]}
        need = {"robot_wrist_pose_w"}
        if comps & {"obj_trans_wrist", "obj_rot6d_wrist"}:
            need |= {"obj_trans_wrist", "obj_quat_wrist"}
        need |= comps & {"obj_lin_vel_wrist", "obj_ang_vel_wrist"}
        if "mano_tip_trans_wrist" in comps:
            need.add("robot_tip_trans_wrist")
        return tuple(tuple(s for s in srcs if s in need) for srcs in alts)

    # ── shared pieces ────────────────────────────────────────────────────────

    def _get(self, compact: torch.Tensor, name: str) -> torch.Tensor:
        cols, scale = self._src[name]
        v = compact.index_select(-1, cols)
        return v / scale if scale is not None else v

    def _wrist(self, compact: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-side workspace palm pos (N,S,3) + unit quat (N,S,4)."""
        wp = self._get(compact, "robot_wrist_pose_w").view(-1, len(self._sides), 7)
        return wp[..., :3], normalize(wp[..., 3:7])

    def _obj_in_wrist(self, compact: torch.Tensor, mid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-side object pose in wrist axes (N,S,3)+(N,S,4): stored, or recovered from the stored cube
        keypoints by Procrustes (exact for >= 4 non-coplanar points)."""
        S = len(self._sides)
        if "obj_trans_wrist" in self._src:
            ot = self._get(compact, "obj_trans_wrist").view(-1, S, 3)
            oq = normalize(self._get(compact, "obj_quat_wrist").view(-1, S, 4))
            return ot, oq
        P = int(self._kp_params["n_points"])
        kp = self._get(compact, "obj_keypoint_trans_wrist").view(-1, S, P, 3)
        loc = torch.stack([self._keypoints_local(s, mid) for s in self._sides], dim=1)  # (N,S,P,3)
        lc, kc = loc.mean(-2, keepdim=True), kp.mean(-2, keepdim=True)
        H = torch.einsum("nspi,nspj->nsij", loc - lc, kp - kc)
        U, _, Vh = torch.linalg.svd(H)
        V = Vh.transpose(-1, -2)
        D = torch.ones_like(H[..., 0, :])
        D[..., 2] = torch.sign(torch.det(V @ U.transpose(-1, -2)))
        R = (V * D[..., None, :]) @ U.transpose(-1, -2)
        ot = kc.squeeze(-2) - torch.einsum("nsij,nsj->nsi", R, lc.squeeze(-2))
        return ot, quat_from_matrix(R)

    def _keypoints_local(self, side: str, mid: torch.Tensor, cube_side=None, n_points=None) -> torch.Tensor:
        """(N, P, 3) object-local cube keypoints for each row's active object (shared env cache)."""
        from envs.object_points import obj_keypoints

        cs = float(self._kp_params["cube_side"] if cube_side is None else cube_side)
        npt = int(self._kp_params["n_points"] if n_points is None else n_points)
        kp = obj_keypoints(self._cmd, self._device, cs, npt)[side]
        if kp.dim() == 2:
            return kp[None].expand(mid.shape[0], -1, -1)
        return kp[self._cmd.motion_lib.traj_obj_slot[side][mid]]

    def _flat_ids(self, mid: torch.Tensor, step: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """(N, K) motion-lib flat frame ids, clamped to the clip end (= command._future_motion_flat_ids)."""
        ml = self._cmd.motion_lib
        cap = ml._motion_num_frames[mid] - 1
        t = torch.minimum(step[:, None] + offsets[None, :].to(step), cap[:, None])
        return ml.length_starts[mid][:, None] + t

    def _offsets(self, n_future: int, max_future_steps: int) -> torch.Tensor:
        return torch.linspace(1.0, float(max_future_steps), n_future, device=self._device).round().long()

    def _surface_pts(self, side: str, mid: torch.Tensor, n_points: int, n_obj_verts: int) -> torch.Tensor:
        """(N, P, 3) object-local FPS keypoints for each row's active object (shared env cache)."""
        from envs.object_points import obj_surface_keypoints

        surf = obj_surface_keypoints(self._cmd, self._device, n_points, n_obj_verts)[side]
        if surf.dim() == 2:
            return surf[None].expand(mid.shape[0], -1, -1)
        return surf[self._cmd.motion_lib.traj_obj_slot[side][mid]]

    # ── term builders (mirror envs.observations.motion_tracking) ─────────────

    def _build_obj_point_cloud_wrist(self, compact, mid, step, n_points=128, n_obj_verts=2048):
        ot, oq = self._obj_in_wrist(compact, mid)
        parts = []
        for si, side in enumerate(self._sides):
            sp = self._surface_pts(side, mid, n_points, n_obj_verts)  # (N,P,3)
            R = matrix_from_quat(oq[:, si])  # (N,3,3)
            parts.append(ot[:, si, None, :] + torch.einsum("bij,bpj->bpi", R, sp))
        return torch.stack(parts, dim=1).reshape(compact.shape[0], -1)

    def _build_future_obj_point_cloud_delta_wrist(self, compact, mid, step, n_points=128, n_obj_verts=2048):
        ml = self._cmd.motion_lib
        w_t, w_q = self._wrist(compact)
        ot, oq = self._obj_in_wrist(compact, mid)
        nfi = self._flat_ids(mid, step, torch.ones(1, dtype=torch.long, device=self._device))[:, 0]
        parts = []
        for si, side in enumerate(self._sides):
            sp = self._surface_pts(side, mid, n_points, n_obj_verts)
            nxt_t = ml.obj_trans[side][nfi]  # (N,3) workspace frame
            nxt_R = ml.obj_rot[side][nfi]  # (N,3,3)
            cur_R_w = matrix_from_quat(oq[:, si])  # object rot expressed in wrist axes
            # delta_wrist = R_w^T (nxt_t - cur_t) + (R_w^T nxt_R - R_ow) sp, cur_t = w_t + R_w ot
            d_t = quat_apply_inverse(w_q[:, si], nxt_t - w_t[:, si]) - ot[:, si]
            R_w = matrix_from_quat(w_q[:, si])
            nxt_R_w = R_w.transpose(-1, -2) @ nxt_R
            d = d_t[:, None, :] + torch.einsum("bij,bpj->bpi", nxt_R_w - cur_R_w, sp)
            parts.append(d)
        return torch.stack(parts, dim=1).reshape(compact.shape[0], -1)

    def _build_future_traj_rel(self, compact, mid, step, components, n_future=10, stride=1):
        offsets = torch.arange(1, n_future + 1, device=self._device) * int(stride)
        return self._future_traj_window(compact, mid, step, offsets, [str(c) for c in components])

    def _future_traj_window(self, compact, mid, step, offsets, comps):
        """Mirror of the env's _future_traj_window: named pieces, or a prefix for a whole group in order."""
        prefix = comps if isinstance(comps, str) else None
        want = (lambda n: n.startswith(prefix)) if prefix else (lambda n: n in comps)
        ml = self._cmd.motion_lib
        N = compact.shape[0]
        S = len(self._sides)
        w_t, w_q = self._wrist(compact)
        if want("obj_trans_wrist") or want("obj_rot6d_wrist"):
            ot, oq = self._obj_in_wrist(compact, mid)
        if want("obj_lin_vel_wrist"):
            olv = self._get(compact, "obj_lin_vel_wrist").view(N, S, 3)
        if want("obj_ang_vel_wrist"):
            oav = self._get(compact, "obj_ang_vel_wrist").view(N, S, 3)
        if want("mano_tip_trans_wrist"):
            tips = self._get(compact, "robot_tip_trans_wrist").view(N, S, 5, 3)
        ffi = self._flat_ids(mid, step, offsets)  # (N,K)
        K = ffi.shape[1]
        feats = []
        for si, side in enumerate(self._sides):
            wq = w_q[:, si, None].expand(N, K, 4)
            R_w = matrix_from_quat(wq)  # (N,K,3,3)
            p: dict[str, torch.Tensor] = {}
            if want("obj_trans_wrist"):
                p["obj_trans_wrist"] = quat_apply_inverse(wq, ml.obj_trans[side][ffi] - w_t[:, si, None]) - ot[:, si, None]
            if want("obj_rot6d_wrist"):
                R_ow = matrix_from_quat(oq[:, si, None].expand(N, K, 4))
                # R_w^T R_fut R_sim^T R_w with R_sim = R_w R_ow  ->  R_w^T R_fut R_ow^T
                R_d = R_w.transpose(-1, -2) @ ml.obj_rot[side][ffi] @ R_ow.transpose(-1, -2)
                p["obj_rot6d_wrist"] = R_d[..., :2].reshape(N, K, 6)  # first two columns, as the env term
            if want("obj_lin_vel_wrist"):
                p["obj_lin_vel_wrist"] = quat_apply_inverse(wq, ml.obj_lin_vel[side][ffi]) - olv[:, si, None]
            if want("obj_ang_vel_wrist"):
                p["obj_ang_vel_wrist"] = quat_apply_inverse(wq, ml.obj_ang_vel[side][ffi]) - oav[:, si, None]
            if want("mano_wrist_trans_wrist"):
                p["mano_wrist_trans_wrist"] = quat_apply_inverse(wq, ml.mano_wrist_trans[side][ffi] - w_t[:, si, None])
            if want("mano_wrist_rot6d_wrist"):
                R_d = R_w.transpose(-1, -2) @ ml.mano_wrist_rot[side][ffi]
                p["mano_wrist_rot6d_wrist"] = R_d[..., :2].reshape(N, K, 6)
            if want("mano_tip_trans_wrist"):
                fut_tip = ml.mano_joint_pos[side][ffi][:, :, ml.tip_ids[side]]  # (N,K,5,3)
                tq = wq[:, :, None].expand(N, K, 5, 4)
                d_tip = quat_apply_inverse(tq, fut_tip - w_t[:, si, None, None]) - tips[:, si, None]
                p["mano_tip_trans_wrist"] = d_tip.reshape(N, K, 15)
            feats.append(torch.cat([p[c] for c in (list(p) if prefix else comps)], dim=-1))  # (N,K,C), env term order
        return torch.stack(feats, dim=2).reshape(N, -1)  # (N, K*S*C)

    def _build_future_obj_traj_state_rel(self, compact, mid, step, n_future=10, max_future_steps=30):
        return self._future_traj_window(compact, mid, step, self._offsets(n_future, max_future_steps), "obj_")

    def _build_future_obj_keypoint_traj_wrist(self, compact, mid, step, n_future=10, max_future_steps=30,
                                              cube_side=0.2, n_points=4):
        ml = self._cmd.motion_lib
        N = compact.shape[0]
        S = len(self._sides)
        P = int(n_points)
        if P != int(self._kp_params["n_points"]) or float(cube_side) != float(self._kp_params["cube_side"]):
            raise ValueError("ref_future_obj_keypoint_traj_wrist must use the cube of obj_keypoint_trans_wrist.")
        w_t, w_q = self._wrist(compact)
        k_sim = self._get(compact, "obj_keypoint_trans_wrist").view(N, S, P, 3)
        v_sim = self._get(compact, "obj_keypoint_lin_vel_wrist").view(N, S, P, 3)
        ffi = self._flat_ids(mid, step, self._offsets(n_future, max_future_steps))  # (N,K)
        K = ffi.shape[1]
        feats = []
        for side in self._obj_sides:
            si = self._sides.index(side)
            lp = self._keypoints_local(side, mid, cube_side, n_points)  # (N,P,3)
            fut_t = ml.obj_trans[side][ffi]  # (N,K,3)
            R_fut = ml.obj_rot[side][ffi]  # (N,K,3,3)
            arm = torch.einsum("nkij,npj->nkpi", R_fut, lp)  # (N,K,P,3)
            fut_kp = fut_t[:, :, None] + arm
            fut_v = ml.obj_lin_vel[side][ffi][:, :, None] + torch.cross(
                ml.obj_ang_vel[side][ffi][:, :, None].expand_as(arm), arm, dim=-1)
            wq = w_q[:, si, None, None].expand(N, K, P, 4)
            d_kp = quat_apply_inverse(wq, fut_kp - w_t[:, si, None, None]) - k_sim[:, si, None]
            d_v = quat_apply_inverse(wq, fut_v) - v_sim[:, si, None]
            feats.append(torch.cat([d_kp.reshape(N, K, P * 3), d_v.reshape(N, K, P * 3)], dim=-1))  # (N,K,6P)
        return torch.stack(feats, dim=2).reshape(N, -1)  # (N, K*S_obj*6P)

    def _build_future_mano_wrist_tips_traj_rel(self, compact, mid, step, n_future=10, max_future_steps=30):
        return self._future_traj_window(compact, mid, step, self._offsets(n_future, max_future_steps), "mano_")


def build_obs_compactor(vec_env: Any, command_name: str, device: torch.device | str, terms=None) -> ObsCompactor:
    """Compactor for a FlatObsVecEnv's flat layout ([actor | critic?] + sorted encoded groups)."""
    env = vec_env.unwrapped
    groups = ["actor"] if vec_env._share_obs else ["actor", "critic"]
    groups += list(vec_env._encoded_obs_keys)
    spans = flat_obs_layout(env, groups)
    full_dim = int(vec_env.single_observation_space.shape[0])
    if spans[-1].end != full_dim:
        raise ValueError(f"obs layout mismatch: spans end at {spans[-1].end}, flat dim {full_dim}.")
    cmd = env.command_manager.get_term(command_name)
    return ObsCompactor(cmd, spans, full_dim, device, terms=terms)

from __future__ import annotations

import zlib
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from retarget.human_demo import clips as C
from retarget.robot_hand import scene as S

TIPS = ("thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip")
TARGETS = [f"{s}_{b}" for s in C.SIDES for b in ("palm",) + TIPS] + ["right_object", "left_object"]


def _constraint(name: str, cfg) -> dict:
    if "palm" in name or "object" in name:
        c = cfg.palm if "palm" in name else cfg.object
        return dict(type=mujoco.mjtEq.mjEQ_WELD, torque=c.torque, solref=list(c.solref), solimp=list(c.solimp))
    solimp = list(cfg.tip.solimp)
    solimp[2] *= 1.0 if any(f in name for f in ("thumb", "index", "middle")) else cfg.tip_width_ring_pinky
    return dict(type=mujoco.mjtEq.mjEQ_CONNECT, torque=cfg.tip.torque, solref=list(cfg.tip.solref),
                solimp=solimp)


def reference(clip: dict) -> np.ndarray:
    """(T, 14, 7) target pos + quat (wxyz) in TARGETS order; absent sides are parked far away."""
    T = C.num_frames(clip)
    park = np.tile([10.0, 10.0, 10.0, 1.0, 0.0, 0.0, 0.0], (T, 1))
    out = []
    for s in C.SIDES:
        if s not in C.sides(clip):
            out += [park] * 6
            continue
        q = Rotation.from_matrix(clip[f"mano_{s}_wrist_rot"]).as_quat()[:, [3, 0, 1, 2]]
        out.append(np.hstack([clip[f"mano_{s}_wrist_pos"], q]))
        for tip in C.tips(clip, s).transpose(1, 0, 2):
            out.append(np.hstack([tip, np.tile([1.0, 0.0, 0.0, 0.0], (T, 1))]))
    for s in C.SIDES:
        if s not in C.sides(clip):
            out.append(park)
            continue
        pose = C.obj_pose(clip, s)
        q = Rotation.from_matrix(pose[:, :3, :3]).as_quat()[:, [3, 0, 1, 2]]
        out.append(np.hstack([pose[:, :3, 3], q]))
    return np.stack(out, axis=1)


def _ik_model(spec: mujoco.MjSpec, names: list[str], cfg) -> mujoco.MjModel:
    for n in names:
        body = spec.worldbody.add_body(name=f"target_{n}", mocap=True)
        body.add_site(name=f"target_{n}", size=[0.01, 0.01, 0.01], group=1)
        c = _constraint(n, cfg.constraints)
        data = np.zeros(11)
        data[10] = c["torque"]
        eq = spec.add_equality(name=f"eq_{n}", type=c["type"], name1=n, name2=f"target_{n}",
                               objtype=mujoco.mjtObj.mjOBJ_SITE, data=data)
        eq.solref, eq.solimp = c["solref"], c["solimp"]
    m = spec.compile()
    m.opt.timestep = cfg.sim_dt
    m.opt.iterations, m.opt.ls_iterations = cfg.solver_iterations, cfg.solver_ls_iterations
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    m.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    m.opt.impratio = cfg.contact.impratio
    m.opt.o_solref[:] = list(cfg.contact.o_solref)
    m.opt.o_solimp[:] = list(cfg.contact.o_solimp)
    m.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_OVERRIDE
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION
    return m


class _Solver:
    def __init__(self, m: mujoco.MjModel, ref: np.ndarray, names: list[str], sides, rng: np.random.Generator,
                 cfg, fps: float) -> None:
        self.m, self.d, self.ref, self.nq_obj, self.rng = m, mujoco.MjData(m), ref, 7 * len(sides), rng
        self.objects = [(m.nq - self.nq_obj + 7 * i, TARGETS.index(f"{s}_object")) for i, s in enumerate(sides)]
        self.n_guess, self.guess_steps, self.jitter = cfg.n_guess, cfg.guess_steps, cfg.guess_jitter
        self.sim_steps = max(1, int(1.0 / fps / cfg.sim_dt))
        self.settle_steps = cfg.settle_steps
        body = lambda n: m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"target_{n}")]  # noqa: E731
        self.targets = [(body(n), TARGETS.index(n), "tip" in n) for n in names]
        self.tips = [(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n), TARGETS.index(n))
                     for n in names if "tip" in n]
        self.palms = []  # forearm joints: pos x y z, rot z x y
        for s, pre in (("right", "R"), ("left", "L")):
            if f"{s}_palm" not in names:
                continue
            jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{pre}_forearm_{a}_joint")
                    for a in ("pos_x", "pos_y", "pos_z", "rot_z", "rot_x", "rot_y")]
            self.palms.append((mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{s}_palm"),
                               m.jnt_qposadr[jids], m.jnt_dofadr[jids], TARGETS.index(f"{s}_palm")))
        self.tip_eq = [i for i in range(m.neq) if "tip" in (m.equality(i).name or "")]
        self.act_qadr = m.jnt_qposadr[m.actuator_trnid[:, 0]]
        wrist = np.array(["forearm" in m.joint(j).name for j in m.actuator_trnid[:, 0]])
        self.finger_act, self.wrist_act = np.flatnonzero(~wrist), np.flatnonzero(wrist)
        jid = m.actuator_trnid[self.finger_act, 0]
        self.open_q = np.clip(0.0, m.jnt_range[jid, 0], m.jnt_range[jid, 1])  # 0 = straight fingers
        self.approach_cfg = cfg.approach

    def set_targets(self, t: int, jitter: bool = False) -> None:
        for mid, col, is_tip in self.targets:
            self.d.mocap_pos[mid] = self.ref[t, col, :3]
            if jitter and is_tip:
                self.d.mocap_pos[mid] += self.rng.standard_normal(3) * self.jitter
            self.d.mocap_quat[mid] = self.ref[t, col, 3:]

    def place_objects(self, t: int) -> None:
        for adr, col in self.objects:
            self.d.qpos[adr: adr + 7] = self.ref[t, col]

    def tip_error(self, t: int) -> np.ndarray:
        return np.array([np.linalg.norm(self.d.site_xpos[sid] - self.ref[t, col, :3]) for sid, col in self.tips])

    def _seat_palms(self, t: int) -> None:
        """Slide each forearm so its palm site starts on the frame-t wrist target."""
        jac = np.zeros((3, self.m.nv))
        for sid, qadr, dadr, col in self.palms:
            for _ in range(3):
                mujoco.mj_forward(self.m, self.d)
                mujoco.mj_jacSite(self.m, self.d, jac, None, sid)
                err = self.ref[t, col, :3] - self.d.site_xpos[sid]
                if np.linalg.norm(err) < 1e-4:
                    break
                self.d.qpos[qadr[:3]] += np.linalg.lstsq(jac[:, dadr[:3]], err, rcond=None)[0]

    def _seat_palm_poses(self, t: int, backoff: float) -> None:
        """Forearm joints so each palm site has its frame-t wrist orientation, backoff m behind the wrist target
        (+z of the wrist frame: the back of the hand)."""
        jacp, jacr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        quat, dq = np.zeros(4), np.zeros(3)
        for sid, qadr, dadr, col in self.palms:
            pos = self.ref[t, col, :3] + self._back(t, col) * backoff
            for _ in range(50):
                mujoco.mj_kinematics(self.m, self.d)
                mujoco.mj_comPos(self.m, self.d)
                mujoco.mju_mat2Quat(quat, self.d.site_xmat[sid])
                mujoco.mju_subQuat(dq, self.ref[t, col, 3:], quat)
                err = np.concatenate([pos - self.d.site_xpos[sid], self.d.site_xmat[sid].reshape(3, 3) @ dq])
                if np.linalg.norm(err) < 1e-6:
                    break
                mujoco.mj_jacSite(self.m, self.d, jacp, jacr, sid)
                self.d.qpos[qadr] += np.linalg.lstsq(np.vstack([jacp[:, dadr], jacr[:, dadr]]), err, rcond=None)[0]
            self.d.qpos[qadr[3:]] = np.mod(self.d.qpos[qadr[3:]] + np.pi, 2 * np.pi) - np.pi  # within the limits

    def _back(self, t: int, col: int) -> np.ndarray:
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, self.ref[t, col, 3:])
        return R.reshape(3, 3)[:, 2]

    def _preshape(self, t: int, backoff: float) -> np.ndarray:
        """Finger joints of the best of n_guess random starts, IK'd with contacts off onto the frame-t palm and
        fingertip targets moved backoff behind the wrists."""
        m, d = self.m, self.d
        shift = {}
        for _, _, _, col in self.palms:
            side = TARGETS[col].split("_")[0]
            shift.update({k: self._back(t, col) * backoff for k, n in enumerate(TARGETS)
                          if n.startswith(side) and "object" not in n})
        contact = int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        m.opt.disableflags |= contact
        best, best_err = None, np.inf
        for _ in range(self.n_guess):
            d.qpos[:] = self.rng.random(m.nq)
            self.place_objects(t)
            d.qvel[:] = 0
            self._seat_palm_poses(t, backoff)
            self.set_targets(t, jitter=True)
            for mid, col, _ in self.targets:
                d.mocap_pos[mid] += shift.get(col, 0.0)
            for _ in range(self.guess_steps):
                mujoco.mj_step(m, d)
            err = np.mean([np.linalg.norm(d.site_xpos[sid] - self.ref[t, col, :3] - shift[col]) for sid, col in self.tips])
            if err < best_err:
                best, best_err = d.qpos[self.act_qadr[self.finger_act]].copy(), err
        m.opt.disableflags &= ~contact
        return best

    def _hand_in_object(self) -> bool:
        mujoco.mj_forward(self.m, self.d)
        for c in self.d.contact[: self.d.ncon]:
            a, b = self.m.geom(c.geom1).name or "", self.m.geom(c.geom2).name or ""
            if c.dist < -0.002 and "collision_hand_" in a + b and ("_object_" in a) != ("_object_" in b):
                return True
        return False

    def approach(self, t: int) -> tuple[float, float]:
        """Start from outside: the hand, shaped by a contact-free start search backoff behind the frame-t wrist
        targets (opened until it clears the objects), is PD-driven onto them with contacts on, then closed by
        moving the fingertip targets from where they are to frame t's. Returns the error and the shape fraction."""
        m, d, c = self.m, self.d, self.approach_cfg
        lo, hi = m.jnt_range[m.actuator_trnid[self.finger_act, 0]].T
        pre = self._preshape(t, c.backoff) if c.preshape > 0 else self.open_q
        for frac in np.arange(c.preshape, -1e-9, -0.25):
            shape = np.clip(frac * pre, lo, hi) if frac > 0 else self.open_q
            d.qpos[:] = 0
            d.qpos[self.act_qadr[self.finger_act]] = shape
            self.place_objects(t)
            self._seat_palm_poses(t, 0.0)
            q_end = d.qpos[self.act_qadr[self.wrist_act]].copy()
            self._seat_palm_poses(t, c.backoff)
            q_start = d.qpos[self.act_qadr[self.wrist_act]].copy()
            if frac <= 0 or not self._hand_in_object():
                break
        d.qvel[:] = 0
        act = int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
        m.opt.disableflags &= ~act
        d.eq_active[self.tip_eq] = 0
        d.ctrl[self.finger_act] = shape
        palms = [(mid, col) for mid, col, is_tip in self.targets if "palm" in TARGETS[col]]
        for i in range(c.slide_frames):
            a = (i + 1) / c.slide_frames
            self.set_targets(t)
            for mid, col in palms:
                d.mocap_pos[mid] += self._back(t, col) * c.backoff * (1 - a)
            d.ctrl[self.wrist_act] = (1 - a) * q_start + a * q_end
            for _ in range(self.sim_steps):
                mujoco.mj_step(m, d)
        m.opt.disableflags |= act
        d.eq_active[self.tip_eq] = 1
        open_tips = {col: d.site_xpos[sid].copy() for sid, col in self.tips}
        for i in range(c.close_frames):
            a = (i + 1) / c.close_frames
            self.step(t, {col: (1 - a) * p + a * self.ref[t, col, :3] for col, p in open_tips.items()})
        return (float(self.tip_error(t).mean()) if self.tips else 0.0), float(max(frac, 0.0))

    def start_search(self, t: int) -> tuple[np.ndarray, float]:
        """Random restarts at frame t; keep the one with the lowest mean fingertip error."""
        best, best_err = None, np.inf
        for _ in range(self.n_guess):
            self.d.qpos[:] = self.rng.random(self.m.nq)
            self.place_objects(t)  # a random object pose blows up against its stiff weld
            self.d.qvel[:] = 0
            self._seat_palms(t)
            self.d.ctrl[:] = self.rng.random(self.m.nu)
            self.set_targets(t, jitter=True)
            for _ in range(self.guess_steps):
                self.d.ctrl[:] = self.d.qpos[: self.m.nq - self.nq_obj]
                mujoco.mj_step(self.m, self.d)
            err = float(self.tip_error(t).mean()) if self.tips else 0.0
            if err < best_err:
                best, best_err = self.d.qpos.copy(), err
        return best, best_err

    def restart(self, qpos: np.ndarray) -> None:
        self.d.qpos[:] = qpos
        self.d.qvel[:] = 0
        self.d.ctrl[:] = qpos[: self.m.nq - self.nq_obj]

    def step(self, t: int, tips: dict | None = None) -> None:
        """tips: fingertip target column -> position, in place of frame t's."""
        self.set_targets(t)
        for mid, col, is_tip in self.targets if tips else ():
            if is_tip:
                self.d.mocap_pos[mid] = tips[col]
        for _ in range(self.sim_steps):
            mujoco.mj_step(self.m, self.d)
        # Settle: fingertip links free, joints PD-held at the IK angles, contacts push them out.
        act = int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
        self.m.opt.disableflags &= ~act
        self.d.eq_active[self.tip_eq] = 0
        self.d.ctrl[:] = self.d.qpos[self.act_qadr]
        for _ in range(self.settle_steps):
            mujoco.mj_step(self.m, self.d)
        self.m.opt.disableflags |= act
        self.d.eq_active[self.tip_eq] = 1


def _track(m: mujoco.MjModel, ref: np.ndarray, names: list[str], sides, cfg, fps: float,
           seed: int, approach: bool, log) -> tuple[np.ndarray, list[float]]:
    """One IK run over the clip: (T, nq) qpos and the per-frame mean fingertip error."""
    sol = _Solver(m, ref, names, sides, np.random.default_rng(seed), cfg, fps)
    sol.place_objects(0)
    if approach:
        err0, frac = sol.approach(0)
        log(f"start: approach (hand shape x{frac:.2f}), fingertip error {err0 * 1e3:.1f} mm")
    else:
        q0, err0 = sol.start_search(0)
        log(f"start: fingertip error {err0 * 1e3:.1f} mm")
        sol.restart(q0)
        mujoco.mj_step(m, sol.d)
    for _ in range(cfg.start_settle):  # the start pose converges on frame 0 before the clip runs
        sol.step(0)

    window, cooldown = cfg.recover_window, cfg.recover_cooldown
    qpos, errs, last_restart = [], [], -(10**9)
    for t in range(len(ref)):
        if t - last_restart > cooldown and len(errs) >= window:
            recent = float(np.mean(errs[-window:]))
            if recent > cfg.recover_tip_err:
                q, err = sol.start_search(t)
                log(f"frame {t}: recent fingertip error {recent * 1e3:.1f} mm -> restart {err * 1e3:.1f} mm")
                sol.restart(q)
                last_restart = t
        sol.step(t)
        errs.append(float(sol.tip_error(t).mean()) if sol.tips else 0.0)
        qpos.append(sol.d.qpos.copy())
    return np.asarray(qpos), errs


def attempt(clip: dict, robot: str, pool: Path, cfg, fps: float, seed: int | str, k: int) -> tuple:
    """IK run k of a clip (k = 0: seed, else a seed derived from it; k = 0 starts with the approach if enabled):
    robot qpos (T, nq of the robot MJCF), the per-frame mean fingertip error and the run's log lines."""
    sides = C.sides(clip)
    objects = {s: (C.obj_id(clip, s), C.obj_scale(clip, s)) for s in sides}
    spec = S.build(robot, sides, objects, pool, cfg.scene, clip.get("support_disks"), shared=C.shared_object(clip))
    names = [n for n in TARGETS if any(n.startswith(s) for s in sides)]
    m = _ik_model(spec, names, cfg)
    nq_obj = 7 * len(sides)
    if isinstance(seed, str):
        seed = zlib.crc32(seed.encode())
    lines: list[str] = []
    qpos, errs = _track(m, reference(clip), names, sides, cfg, fps,
                        seed if k == 0 else zlib.crc32(f"{seed}#{k}".encode()), k == 0 and cfg.approach.enable,
                        lines.append)
    return qpos[:, : m.nq - nq_obj], errs, lines


def best_run(runs: list[tuple], cfg) -> dict:
    """The run with the lowest mean fingertip error + cfg.pen_weight * its max hand-object penetration (an optional
    4th element): one run can settle in a wrong basin for a whole clip below recover_tip_err, and the lowest error
    can come from fingers pushed into the object. Smoothed: qpos (T - cfg.smooth, ...) aligned to the clip's last
    T - cfg.smooth frames."""
    qpos, errs, lines = min(runs, key=lambda r: np.mean(r[1]) + (cfg.pen_weight * r[3] if len(r) > 3 else 0.0))[:3]
    kernel = np.ones(cfg.smooth) / cfg.smooth
    smooth = np.stack([np.convolve(qpos[:, i], kernel, mode="valid") for i in range(qpos.shape[1])], 1)[1:]
    return {"qpos": smooth.astype(np.float32), "tip_error_mean_m": float(np.mean(errs)),
            "crop": len(qpos) - len(smooth), "log": lines}


def retarget(clip: dict, robot: str, pool: Path, cfg, fps: float, seed: int | str = 0, log=print) -> dict:
    """best_run of cfg.attempts runs, one after another (hand_retarget.py spreads them over workers)."""
    out = best_run([attempt(clip, robot, pool, cfg, fps, seed, k) for k in range(cfg.attempts)], cfg)
    for line in out["log"]:
        log(line)
    return out

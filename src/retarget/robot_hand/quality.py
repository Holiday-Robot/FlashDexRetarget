from __future__ import annotations

import mujoco
import numpy as np

from retarget.human_demo import clips as C
from retarget.robot_hand import scene as S
from retarget.robot_hand.ik import TIPS


def flags(clip: dict, robot: str, pool, cfg, fps: float) -> dict:
    """clip carries qpos (T, nq of the robot MJCF) aligned with its other arrays; cfg: retarget."""
    th = cfg.quality
    sides = C.sides(clip)
    objects = {s: (C.obj_id(clip, s), C.obj_scale(clip, s)) for s in sides}
    disks = np.asarray(clip.get("support_disks", np.zeros((0, 5)))).reshape(-1, 5)
    m = S.build(robot, sides, objects, pool, cfg.scene, disks, shared=C.shared_object(clip)).compile()
    d = mujoco.MjData(m)
    qpos = clip["qpos"]
    T, nq_robot = len(qpos), qpos.shape[1]
    z_desk = float(disks[:, 2].min()) if len(disks) else 0.0
    rng = np.random.default_rng(0)

    hand = {s: [] for s in sides}
    for g in range(m.ngeom):
        name = m.geom(g).name
        if name.startswith("collision_hand_") and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            s = "left" if "left" in name else "right"
            if s in hand:
                mid = m.geom_dataid[g]
                v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
                hand[s].append((g, v[rng.choice(len(v), min(60, len(v)), replace=False)]))
    hand_ids = {g for g in range(m.ngeom) if m.geom(g).name.startswith("collision_hand_")}
    desk_ids = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"support_{k}") for k in range(len(disks))}
    obj_ids = {g for g in range(m.ngeom) if "_object_" in m.geom(g).name}
    tip_sites = {s: [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{s}_{t}") for t in TIPS] for s in sides}
    mano_tips = {s: C.tips(clip, s) for s in sides}

    palm_body = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s[0].upper()}_forearm_rot_y_link") for s in sides}
    palm = {s: np.zeros((T, 3)) for s in sides}
    tip_err, below_out, desk_pen, obj_pen = [], np.zeros(T), np.zeros(T), np.zeros(T)
    for t in range(T):
        d.qpos[:nq_robot] = qpos[t]
        for i, s in enumerate(sides):
            pose = C.obj_pose(clip, s)[t]
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, pose[:3, :3].reshape(-1))
            d.qpos[nq_robot + 7 * i: nq_robot + 7 * i + 7] = np.concatenate([pose[:3, 3], q])
        mujoco.mj_forward(m, d)
        for s in sides:
            palm[s][t] = d.xpos[palm_body[s]]
        tip_err.append(np.concatenate([np.linalg.norm(d.site_xpos[tip_sites[s]] - mano_tips[s][t], axis=1)
                                       for s in sides]))
        for c in d.contact[: d.ncon]:
            g1, g2, depth = c.geom1, c.geom2, max(0.0, -float(c.dist))
            pair = {g1, g2}
            if pair & hand_ids and pair & obj_ids:
                obj_pen[t] = max(obj_pen[t], depth)
            elif pair & hand_ids and pair & desk_ids:
                desk_pen[t] = max(desk_pen[t], depth)
        if len(disks):
            pts = np.concatenate([v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g] for s in sides for g, v in hand[s]])
            on = np.zeros(len(pts), dtype=bool)
            for x, y, _, r, _ in disks:
                on |= np.hypot(pts[:, 0] - x, pts[:, 1] - y) < r + 0.02
            if (~on).any():
                below_out[t] = max(0.0, float((z_desk - pts[~on, 2]).max()))
    tip_err = np.asarray(tip_err)
    acc = [0.0]
    for s in sides:
        v = np.linalg.norm(np.diff(palm[s], axis=0), axis=1) * fps
        if len(v) > 1:
            acc.append(float(np.abs(np.diff(v)).max() * fps))
    out = {
        "palm_acc_max": max(acc),
        "tip_err_median_m": float(np.median(tip_err)),
        "tip_far_frac": float((tip_err > th.tip_tracking.max_dist).mean()),
        "hand_below_desk_outside_max_m": float(below_out.max()),
        "hand_obj_pen_max_m": float(obj_pen.max()),
        "hand_desk_pen_frames": int((desk_pen[th.hand_desk_penetration.after:]
                                     > th.hand_desk_penetration.max_depth).sum()),
    }
    flagged = {
        "palm_acc_spike": out["palm_acc_max"] > th.palm_acc_spike.max_acc,
        "hand_far_below_desk": out["hand_below_desk_outside_max_m"] > th.hand_far_below_desk.max_depth,
        "tip_tracking": out["tip_far_frac"] > th.tip_tracking.max_frac,
        "hand_obj_penetration": out["hand_obj_pen_max_m"] > th.hand_obj_penetration.max_depth,
        "hand_desk_penetration": out["hand_desk_pen_frames"] >= th.hand_desk_penetration.frames,
    }
    out["flags"] = ";".join(name for name, bad in flagged.items() if bad)
    return out

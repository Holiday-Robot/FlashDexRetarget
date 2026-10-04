from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np


def robot_xml(robot: str, sides: tuple[str, ...]) -> Path:
    name = "bimanual.xml" if len(sides) == 2 else f"{sides[0]}.xml"
    path = Path(__file__).resolve().parents[3] / "assets" / "robot" / robot / name
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _ik_dynamics(spec: mujoco.MjSpec, cfg) -> None:
    """Heavy joints, no damping, critically damped stiff PD: kv = 2 sqrt(kp M_ii) at qpos0."""
    for j in spec.joints:
        j.armature = cfg.armature
        j.damping = np.zeros_like(j.damping)  # (3,) polynomial damping in MuJoCo >= 3.10
        j.actfrclimited = 0
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    M = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, M)
    for a, act in enumerate(spec.actuators):
        jid = model.actuator_trnid[a, 0]
        kp = cfg.kp_wrist if "forearm" in model.joint(jid).name else cfg.kp_finger
        kv = 2.0 * np.sqrt(kp * M[model.jnt_dofadr[jid], model.jnt_dofadr[jid]])
        act.gainprm[0] = kp
        act.biasprm[0], act.biasprm[1], act.biasprm[2] = 0.0, -kp, -kv
        act.forcelimited = 0


def build(robot: str, sides: tuple[str, ...], objects: dict[str, tuple[str, float]], pool: Path, cfg,
          support_disks: np.ndarray | None = None, shared: bool = False) -> mujoco.MjSpec:
    """objects: side -> (object id, mesh scale); pool: <pool>/objects/<id>/convex/<i>.obj."""
    xml = robot_xml(robot, sides)
    spec = mujoco.MjSpec.from_file(str(xml))
    meshdir = xml.parent / (spec.meshdir or "")
    for mesh in spec.meshes:
        if not Path(mesh.file).is_absolute():
            mesh.file = str((meshdir / mesh.file).resolve())
    spec.meshdir = ""
    _ik_dynamics(spec, cfg)

    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05])
    statics = ["floor"]
    for k, (x, y, z_top, r, h) in enumerate(np.asarray(support_disks if support_disks is not None
                                                       else np.zeros((0, 5))).reshape(-1, 5)):
        h = max(float(h), cfg.desk_depth)
        spec.worldbody.add_geom(name=f"support_{k}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                                size=[r, h / 2, 0], pos=[x, y, z_top - h / 2])
        statics.append(f"support_{k}")

    obj_geoms: dict[str, list[str]] = {}
    for side in sides:
        oid, scale = objects[side]
        convex = Path(pool) / "objects" / oid / "convex"
        parts = sorted((p for p in convex.glob("*.obj") if p.stem.isdigit()), key=lambda p: int(p.stem))
        if not parts:
            raise FileNotFoundError(f"no convex parts in {convex} (run src/retarget/object/obj_convex_decompose.py)")
        body = spec.worldbody.add_body(name=f"{side}_object")
        body.add_joint(name=f"{side}_object_joint", type=mujoco.mjtJoint.mjJNT_FREE,
                       armature=cfg.object_armature, frictionloss=cfg.object_frictionloss)
        obj_geoms[side] = []
        for p in parts:
            spec.add_mesh(name=f"{side}_{p.stem}", file=str(p.resolve()), scale=[scale] * 3)
            body.add_geom(name=f"{side}_object_{p.stem}", type=mujoco.mjtGeom.mjGEOM_MESH,
                          meshname=f"{side}_{p.stem}", contype=0, conaffinity=0, density=cfg.object_density)
            obj_geoms[side].append(f"{side}_object_{p.stem}")
        body.add_site(name=f"{side}_object", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.01, 0.02, 0.03])

    solref, friction = list(cfg.pair_solref), list(cfg.pair_friction)

    def pair(a: str, b: str, friction=friction) -> None:
        spec.add_pair(name=f"{a}_{b}", geomname1=a, geomname2=b, solref=solref,
                      friction=friction, condim=3)

    hand = [g.name for g in spec.geoms if g.name.startswith("collision_hand_")]
    for h in hand:
        for s in statics:
            pair(s, h)
    for geoms in obj_geoms.values():
        for g in geoms:
            for h in hand + statics:
                pair(h, g)
    if len(sides) == 2 and not shared:
        for a in obj_geoms["right"]:
            for b in obj_geoms["left"]:
                pair(a, b, list(cfg.object_pair_friction))
    # Fingertip-link self-collision ("_0" links): across the hands and within each hand.
    done: set[tuple[str, str]] = set()
    for a in (h for h in hand if "0" in h):
        side = "right" if "right" in a else "left"
        for b in hand:
            other = "left" if side == "right" else "right"
            cross = len(sides) == 2 and other in b and ("0" in b or "1" in b)
            same = side in b and "0" in b
            if b != a and (cross or same) and (a, b) not in done and (b, a) not in done:
                pair(a, b)
                done.add((a, b))
    return spec

"""Isaac Lab scene construction + mjlab-Scene-compatible adapter (DirectRLEnv
build order). Import only AFTER the Isaac Sim app is running."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import trimesh

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg

from envs.commands.motion_tracking.reference.hand import ALLLINK_BODY_NAMES_BY_SIDE
from envs.scene._object_setup import object_park_position

from .convert_assets import COLVIS_ROBOT_DIRNAME, colvis_enabled
from .entity import ArticulationAdapter, RigidObjectAdapter, _FreshnessClock
from .mjcf_meta import RobotMjcfMeta
from .sensors import (
    AlllinkContactAdapter,
    AlllinkContactPosAdapter,
    FingertipContactAdapter,
    FingertipPenetrationAdapter,
    LinkContactBank,
)
from .sim import IsaacSimAdapter, principal_inertia

_SIM_DIR = Path(__file__).resolve().parents[3]  # simulation/


def _abl(name: str) -> bool:
    """Physics-ablation probe switch (FDR_ABL_*), all default off.
    Used by scripts/run_physics_ablation_b200.sh to attribute PhysX cost."""
    return os.environ.get(f"FDR_ABL_{name}", "0") == "1"


def _alllink_bodies_by_side(cfg) -> dict[str, list[str]]:
    """{side: all-link contact bodies} from the scene's {r,l}_alllink_contact sensors (the mujoco
    source of truth, so other hands work); the xhand constant is the fallback."""
    out: dict[str, list[str]] = {}
    for side in ("right", "left"):
        pat = None
        for sn in (cfg.scene.get("sensors", None) or []):
            if str(sn.get("name", "")) == f"{side[0]}_alllink_contact":
                pat = [str(b) for b in sn["primary"]["pattern"]]
                break
        out[side] = pat or list(ALLLINK_BODY_NAMES_BY_SIDE[side])
    return out


def _robot_usd_path(meta: RobotMjcfMeta) -> Path:
    """Converted robot USD for the configured MJCF (xhand_right / xhand_bimanual).
    FDR_COLVIS: debug robot whose visuals ARE the collided convex hulls."""
    override = os.environ.get("FDR_ROBOT_USD")
    if override:
        return Path(override)
    stem = Path(meta.xml_path).stem
    variant = next((v for v in ("bimanual", "left") if v in stem), "right")
    if Path(meta.xml_path).parent.name == "sharpa":
        return _SIM_DIR / "assets/robot/sharpa/urdf/converted" / f"sharpa_{variant}.usd"
    return _SIM_DIR / "assets/robot/xhand/urdf" / (
        COLVIS_ROBOT_DIRNAME if colvis_enabled() else "converted"
    ) / f"xhand_{variant}.usd"


# MJCF -> URDF/USD link renames (see assets fork report).
_URDF_LINK_ALIAS = {
    "R_forearm_rot_y_link": "right_hand_link",
    "L_forearm_rot_y_link": "left_hand_link",
}


class IsaacSceneAdapter:
    """Duck-typed mjlab ``Scene``: name lookup, origins, step-loop hooks."""

    def __init__(self, scene: InteractiveScene, device: str) -> None:
        self._scene = scene
        self.device = device
        self.entities: dict[str, object] = {}
        self.sensors: dict[str, object] = {}
        self.sensor_context = None

    @property
    def num_envs(self) -> int:
        return self._scene.cfg.num_envs

    @property
    def env_origins(self) -> torch.Tensor:
        return self._scene.env_origins

    def __getitem__(self, name: str):
        if name in self.entities:
            return self.entities[name]
        if name in self.sensors:
            return self.sensors[name]
        raise KeyError(name)

    def __contains__(self, name: str) -> bool:
        return name in self.entities or name in self.sensors

    def keys(self):
        return list(self.entities.keys()) + list(self.sensors.keys())

    def write_data_to_sim(self) -> None:
        self._scene.write_data_to_sim()

    def update(self, dt: float) -> None:
        self._scene.update(dt)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        for ent in self.entities.values():
            ent.reset(env_ids)
        self._scene.reset(env_ids)


def _robot_articulation_cfg(
    meta: RobotMjcfMeta, self_collision: bool, pos_iters: int, vel_iters: int,
    contact_offset: float = 0.002, rest_offset: float = 0.0,
    max_depen: float = 5.0, max_contact_impulse: float | None = None,
) -> ArticulationCfg:
    """xhand articulation: converted USD + MJCF-derived implicit PD gains."""
    robot_usd = _robot_usd_path(meta)
    if not robot_usd.exists():
        raise FileNotFoundError(
            f"{robot_usd} missing — run convert_assets.py --robot"
            + (" --collision-visual" if colvis_enabled() else "")
        )

    def _gains(joints: list[str]) -> tuple[dict, dict]:
        kp = {}
        kv = {}
        for j in joints:
            p, v = meta.kp_kv_for_joint(j)
            kp[j] = p
            # MJCF total joint damping = servo kv + passive joint damping.
            kv[j] = v + float(meta.joint_damping[meta.joint_names.index(j)])
        return kp, kv

    kp, kv = _gains(meta.joint_names)
    actuators = {
        "all": ImplicitActuatorCfg(
            joint_names_expr=list(meta.joint_names),
            stiffness=kp,
            damping=kv,
            armature={j: float(meta.joint_armature[i]) for i, j in enumerate(meta.joint_names)},
            friction={
                j: float(meta.joint_frictionloss[i])
                for i, j in enumerate(meta.joint_names)
            },
            velocity_limit_sim=100.0,
            effort_limit_sim={
                j: float(meta.actuator_forcerange[meta.actuator_names.index(a)][1])
                for a, j in meta.actuator_joint.items()
            },
        )
    }
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(robot_usd),
            activate_contact_sensors=not _abl("NOREPORT"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=max_depen,
                **({} if max_contact_impulse is None
                   else {"max_contact_impulse": max_contact_impulse}),
                linear_damping=0.0,
                angular_damping=0.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=contact_offset, rest_offset=rest_offset
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=self_collision,
                solver_position_iteration_count=pos_iters,
                solver_velocity_iteration_count=vel_iters,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={j: 0.0 for j in meta.joint_names},
        ),
        actuators=actuators,
    )


def _usd_scale(scale: float) -> tuple[float, float, float] | None:
    """Uniform prim scale for a slot, or None at 1.0 so no xformOp:scale is authored."""
    return None if float(scale) == 1.0 else (float(scale),) * 3


def _object_rigid_cfg(
    entity_name: str, usd_path: str, park: tuple, pos_iters: int, vel_iters: int,
    scale: float = 1.0, contact_offset: float = 0.002, rest_offset: float = 0.0,
    max_depen: float = 5.0, max_contact_impulse: float | None = None,
) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/{entity_name}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            scale=_usd_scale(scale),
            activate_contact_sensors=not _abl("NOREPORT"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=max_depen,
                **({} if max_contact_impulse is None
                   else {"max_contact_impulse": max_contact_impulse}),
                linear_damping=0.0,
                angular_damping=0.0,
                max_angular_velocity=200.0,
                solver_position_iteration_count=pos_iters,
                solver_velocity_iteration_count=vel_iters,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=not _abl("NOOBJCOL"),
                contact_offset=contact_offset, rest_offset=rest_offset
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=park),
    )



def _spawn_support_disks(cfg, stage, num_envs: int, coff: float, roff: float, mat_path: str) -> dict:
    """CHORD support disks: count x radii kinematic cylinders per env (parked), one RigidObject
    view per (k, radius bucket). Same grid as envs/scene/_support_disk_setup.py (mujoco)."""
    sd = cfg.commands.object.support_disks
    if not bool(sd.enable):
        return {}
    from omegaconf import open_dict

    from pxr import Sdf, UsdPhysics

    from envs.scene._support_disk_setup import support_disk_name, support_park_position

    filter_robot = not bool(sd.get("robot_collision", True))
    views: dict[str, RigidObject] = {}
    names: list[list[str]] = []
    for k in range(int(sd.count)):
        row = []
        for j, r in enumerate(sd.radii):
            name = support_disk_name(k, j)
            c = sim_utils.CylinderCfg(
                radius=float(r),
                height=float(sd.height),
                axis="Z",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True, disable_gravity=True
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True, contact_offset=coff, rest_offset=roff
                ),
                # isaaclab's spawn_preview_surface re-attaches the stage per prim (O(stage));
                # per-env mode spawns count x radii x num_envs disks, so headless skips the material.
                visual_material=(
                    sim_utils.PreviewSurfaceCfg(
                        diffuse_color=tuple(float(v) for v in sd.rgba[:3])
                    )
                    if os.environ.get("FDR_DISK_VISMAT") == "1"
                    else None
                ),
            )
            park = support_park_position(sd, k, j)
            for i in range(num_envs):
                path = f"/World/envs/env_{i}/{name}"
                # Reference clones (copy_from_source=False) already show env_0's disk in every env.
                if stage.GetPrimAtPath(path).IsValid():
                    continue
                c.func(path, c, translation=park)
                _apply_contact_material(stage, path, mat_path)
                if filter_robot:
                    # CHORD's hand task masks robot<->support-surface pairs; nested prims are
                    # included and the clone arc remaps the target onto each env's own Robot.
                    UsdPhysics.FilteredPairsAPI.Apply(
                        stage.GetPrimAtPath(path)
                    ).CreateFilteredPairsRel().AddTarget(
                        Sdf.Path(f"/World/envs/env_{i}/Robot")
                    )
            views[name] = RigidObject(
                RigidObjectCfg(
                    prim_path=f"/World/envs/env_.*/{name}",
                    spawn=None,
                    init_state=RigidObjectCfg.InitialStateCfg(pos=park),
                )
            )
            row.append(name)
        names.append(row)
    with open_dict(sd):
        sd.entity_names = names
    if filter_robot:
        # Reference clones carry env_0's mask through the arc: check both ends of the env range,
        # every disk, or some envs would silently keep hitting the desk.
        for i in sorted({0, num_envs - 1}):
            for n in (n for row in names for n in row):
                path = f"/World/envs/env_{i}/{n}"
                rel = UsdPhysics.FilteredPairsAPI(stage.GetPrimAtPath(path)).GetFilteredPairsRel()
                targets = [str(t) for t in (rel.GetTargets() if rel else [])]
                if targets != [f"/World/envs/env_{i}/Robot"]:
                    raise RuntimeError(f"support-disk robot mask missing on {path}: {targets}")
        print(
            f"[isaac] support disks: robot contact pairs masked on {len(views)} disks x {num_envs} "
            f"envs (support_disks.robot_collision=false)"
        )
    return views


class SupportDiskAdapter:
    """Kinematic disk view with the mjlab mocap write API the command uses."""

    def __init__(self, obj: RigidObject, device: str) -> None:
        self._obj = obj
        self.device = device

    def write_mocap_pose_to_sim(self, pose: torch.Tensor, env_ids: torch.Tensor) -> None:
        self._obj.write_root_link_pose_to_sim(pose, env_ids=env_ids)

    def reset(self, env_ids=None) -> None:
        self._obj.reset(env_ids)


def _apply_contact_material(stage, prim_path: str, mat_path: str) -> None:
    """De-instance the subtree (holosoma recipe) then force-bind the material;
    converted USDs hide colliders behind instanced references otherwise."""
    from pxr import Usd

    prim = stage.GetPrimAtPath(prim_path)
    prim.SetInstanceable(False)
    paths = [
        pr.GetPath()
        for pr in Usd.PrimRange(prim, Usd.TraverseInstanceProxies())
        if pr.IsInstanceable()
    ]
    for path in paths:
        sub = stage.GetPrimAtPath(path)
        if sub:
            sub.SetInstanceable(False)
    from isaaclab.sim.utils import bind_physics_material

    bind_physics_material(prim_path, mat_path)


def build_isaac_scene(
    cfg,
    sim_ctx,
    device: str,
    meta: RobotMjcfMeta,
    object_slots: list[dict],
    assigned_obj_slot: torch.Tensor | None = None,
):
    """Author + clone the scene, boot physics, wrap in adapters. assigned_obj_slot
    (B,) selects PER-ENV-OBJECT mode; None = legacy parked-slot mode."""
    num_envs = int(cfg.scene.num_envs)
    per_env_mode = assigned_obj_slot is not None
    scene = InteractiveScene(
        InteractiveSceneCfg(
            num_envs=num_envs,
            env_spacing=float(cfg.scene.env_spacing),
            replicate_physics=not (per_env_mode or _abl("NOREPLICATE")),
        )
    )

    # ── static scene ──────────────────────────────────────────────────────
    # A second Kit process on the node loses the shared kvdb asset cache, so the nucleus
    # ground plane stops resolving; ISAAC_ASSET_ROOT points at an on-disk mirror instead
    # (the cfg default is baked at import, hence the explicit override here).
    if not _abl("NOGROUND"):
        ground = sim_utils.GroundPlaneCfg()
        if os.environ.get("ISAAC_ASSET_ROOT"):
            ground.usd_path = (
                f"{os.environ['ISAAC_ASSET_ROOT']}/Isaac/Environments/Grid/default_environment.usd"
            )
        ground.func("/World/ground", ground)
    sim_utils.DomeLightCfg(intensity=2000.0).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=2000.0)
    )

    # Contact material (friction parity + solimp-like compliance), bound per
    # asset because instanced references shadow scene-default materials.
    import isaacsim.core.utils.stage as stage_utils

    stage = stage_utils.get_current_stage()
    _MAT = "/World/Materials/handContact"
    _MAT_OBJ = "/World/Materials/objContact"
    _px = cfg.sim.physx
    # object_friction=null keeps objects on the hand's material, as before.
    _hand_fric = float(_px.get("friction", 1.0))
    _obj_fric_raw = _px.get("object_friction", None)
    _obj_fric = _hand_fric if _obj_fric_raw is None else float(_obj_fric_raw)

    # "max" on the non-object materials pins hand/disk-object friction at the hand value
    # (PhysX: MAX beats AVERAGE) so object_friction then only changes object-object pairs.
    _combine = str(_px.get("nonobject_friction_combine_mode", "average"))

    def _mat(path: str, fric: float, combine: str = "average"):
        c = sim_utils.RigidBodyMaterialCfg(
            static_friction=fric,
            dynamic_friction=fric,
            restitution=0.0,
            friction_combine_mode=combine,
            compliant_contact_stiffness=float(
                _px.get("compliant_contact_stiffness", 0.0)
            ),
            compliant_contact_damping=float(_px.get("compliant_contact_damping", 0.0)),
        )
        c.func(path, c)

    _mat(_MAT, _hand_fric, _combine)
    _mat(_MAT_OBJ, _obj_fric)
    # disk_friction=null keeps the support disks on the object material, as before.
    _MAT_DISK = _MAT_OBJ
    if _px.get("disk_friction", None) is not None:
        _MAT_DISK = "/World/Materials/diskContact"
        _mat(_MAT_DISK, float(_px.get("disk_friction")), _combine)

    # ── env_0 assets ──────────────────────────────────────────────────────
    px = cfg.sim.physx
    pos_iters = int(px.solver_position_iteration_count)
    vel_iters = int(px.solver_velocity_iteration_count)
    coff = float(px.get("contact_offset", 0.002))
    roff = float(px.get("rest_offset", 0.0))
    mdep = float(px.get("max_depenetration_velocity", 5.0))
    mimp = px.get("max_contact_impulse", None)
    mimp = None if mimp is None else float(mimp)
    robot = Articulation(
        _robot_articulation_cfg(
            meta, bool(cfg.robot.get("self_collision", False)), pos_iters, vel_iters,
            coff, roff, mdep, mimp,
        )
    )
    _apply_contact_material(stage, "/World/envs/env_0/Robot", _MAT)
    objects: dict[str, RigidObject] = {}
    if not per_env_mode:
        for slot, spec in enumerate(object_slots):
            park = object_park_position(slot)
            objects[spec["entity_name"]] = RigidObject(
                _object_rigid_cfg(
                    spec["entity_name"], spec["usd_path"], park, pos_iters, vel_iters,
                    float(spec.get("scale", 1.0)), coff, roff, mdep, mimp,
                )
            )
            _apply_contact_material(
                stage, f"/World/envs/env_0/{spec['entity_name']}", _MAT_OBJ
            )


    # per-env mode: independent copies — reference-clones would compose any
    # post-clone env_0 edit (the per-env object spawns) into EVERY env.
    scene.clone_environments(copy_from_source=per_env_mode)

    # Bimanual pair-slot specs store per-side sub-dicts; single-side specs are
    # scalars whose side is baked into the entity name (object_<side>[_<slot>]).
    def _slot_sides(spec: dict) -> list[str]:
        if isinstance(spec["usd_path"], dict):
            return list(spec["usd_path"])
        return ["left"] if "object_left" in spec.get("entity_name", "") else ["right"]

    def _slot_val(spec: dict, key: str, side: str):
        v = spec[key]
        return v[side] if isinstance(v, dict) else v

    obj_sides = sorted({s for sp in object_slots for s in _slot_sides(sp)}) or ["right"]
    # Index into the per-side gid/hull lists: equals slot_idx when every slot has both
    # sides (per-env mode), and counts within the side for one-sided legacy slots.
    _side_slot: dict[tuple[int, str], int] = {}

    if per_env_mode:
        # Post-clone per-env USD references (heterogeneous objects), then one
        # RigidObject view PER SIDE over the regex path (cfg.spawn=None).
        obj_spawn_cfg = sim_utils.UsdFileCfg(
            usd_path=_slot_val(object_slots[0], "usd_path", obj_sides[0]),
            activate_contact_sensors=not _abl("NOREPORT"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=mdep,
                **({} if mimp is None else {"max_contact_impulse": mimp}),
                linear_damping=0.0,
                angular_damping=0.0,
                max_angular_velocity=200.0,
                solver_position_iteration_count=pos_iters,
                solver_velocity_iteration_count=vel_iters,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=not _abl("NOOBJCOL"),
                contact_offset=coff, rest_offset=roff
            ),
        )
        # Mixed pair pt: a shared slot's left prim is an inert ghost (kinematic, no gravity,
        # parked under the ground) so the per-side view stays uniform; the command aliases it.
        ghost_spawn_cfg = obj_spawn_cfg.replace(
            rigid_props=obj_spawn_cfg.rigid_props.replace(
                disable_gravity=True, kinematic_enabled=True
            )
        )
        slots_np = assigned_obj_slot.cpu().numpy()
        for i in range(num_envs):
            spec = object_slots[int(slots_np[i])]
            for side in obj_sides:
                ghost = side in spec.get("ghost", ())
                c = (ghost_spawn_cfg if ghost else obj_spawn_cfg).replace(
                    usd_path=_slot_val(spec, "usd_path", side),
                    scale=_usd_scale(
                        _slot_val(spec, "scale", side) if "scale" in spec else 1.0
                    ),
                )
                c.func(
                    f"/World/envs/env_{i}/object_{side}",
                    c,
                    translation=(0.0, 0.0, -3.0)
                    if ghost
                    else (0.0, 0.15 if side == "left" else 0.0, 0.2),
                )
                _apply_contact_material(
                    stage, f"/World/envs/env_{i}/object_{side}", _MAT_OBJ
                )
        # Verify per-env spawns landed: a reference clone would compose env_0's
        # object everywhere and the spawner would skip silently.
        check = {i for i in (0, 1, num_envs // 2, num_envs - 1) if i < num_envs}
        for i in check:
            for side in obj_sides:
                prim = stage.GetPrimAtPath(f"/World/envs/env_{i}/object_{side}")
                refs = (prim.GetMetadata("references") or None) if prim else None
                items = list(refs.prependedItems) if refs else []
                want = _slot_val(object_slots[int(slots_np[i])], "usd_path", side)
                got = items[0].assetPath if items else "<none>"
                if got != want:
                    raise RuntimeError(
                        f"per-env object spawn mismatch at env_{i}/{side}: "
                        f"{got} != {want}"
                    )
                # Scale-augmented pts: the prim MUST carry the slot's scale, else
                # PhysX simulates the original geometry against scaled refs/inertia.
                spec_i = object_slots[int(slots_np[i])]
                want_s = float(
                    _slot_val(spec_i, "scale", side) if "scale" in spec_i else 1.0
                )
                attr = prim.GetAttribute("xformOp:scale")
                got_s = float(attr.Get()[0]) if attr and attr.Get() is not None else 1.0
                if abs(got_s - want_s) > 1e-6:
                    raise RuntimeError(
                        f"per-env object scale mismatch at env_{i}/{side}: "
                        f"{got_s} != {want_s}"
                    )

        for side in obj_sides:
            objects[f"object_{side}"] = RigidObject(
                RigidObjectCfg(
                    prim_path=f"/World/envs/env_.*/object_{side}",
                    spawn=None,
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.2)),
                )
            )

    support_views = _spawn_support_disks(cfg, stage, num_envs, coff, roff, _MAT_DISK)
    for name, obj in support_views.items():
        objects[name] = obj

    scene.filter_collisions(
        global_prim_paths=[] if _abl("NOGROUND") else ["/World/ground"]
    )

    # One ContactSensor per alllink body, net force only: PhysX GPU cannot
    # filter pairs vs COACD multi-shape colliders (attribution is SDF-side).
    alllink = _alllink_bodies_by_side(cfg)
    robot_sides = [
        s for s in ("right", "left")
        if alllink[s][0] in meta.body_names
    ]
    link_sensors: dict[str, list[ContactSensor]] = {}
    for side in robot_sides:
        link_sensors[side] = []
        for body in [] if _abl("NOREPORT") else alllink[side]:
            urdf_body = body  # xhand URDF renames the palms; MJCF-named USDs keep the name
            if not stage.GetPrimAtPath(f"/World/envs/env_0/Robot/{body}").IsValid():
                urdf_body = _URDF_LINK_ALIAS.get(body, body)
            s = ContactSensor(
                ContactSensorCfg(
                    prim_path=f"/World/envs/env_.*/Robot/{urdf_body}",
                    update_period=0.0,
                )
            )
            link_sensors[side].append(s)

    # register into the InteractiveScene so update/write/reset drive them
    scene.articulations["robot"] = robot
    for name, obj in objects.items():
        scene.rigid_objects[name] = obj
    for side in robot_sides:
        for body, s in zip(alllink[side], link_sensors[side]):
            scene.sensors[f"_link_contact_{body}"] = s

    # ── boot physics (binds views, applies actuator cfgs) ─────────────────
    sim_ctx.reset()
    scene.update(0.0)

    if not bool(cfg.get("headless", True)):
        # GUI sanity-check framing: look at env_0's hand/object workspace.
        sim_ctx.set_camera_view(eye=(0.9, -0.9, 0.7), target=(0.0, 0.0, 0.15))

    # ── adapters ──────────────────────────────────────────────────────────
    clock = _FreshnessClock()
    sim_adapter = IsaacSimAdapter(
        sim_ctx, device, render=not bool(cfg.get("headless", True))
    )
    sim_adapter._clock = clock

    robot_adapter = ArticulationAdapter(robot, meta, clock, device)
    # 0.0 matches every run before the fix (resets cleared it); 0.8 is the MJCF gravcomp.
    robot_adapter.init_gravcomp(float(cfg.sim.get("robot_gravcomp", 0.0)))

    scene_adapter = IsaacSceneAdapter(scene, device)
    scene_adapter._clock = clock
    scene_adapter.entities[cfg.robot.entity_name] = robot_adapter
    for name in support_views:
        scene_adapter.entities[name] = SupportDiskAdapter(objects[name], device)

    # Per-slot mass properties from the COACD hulls (mjwarp compile parity);
    # written into PhysX below so pin feedforward == true sim inertia.
    from pathlib import Path as _P

    from .sim import hull_mass_props

    hull_mass, hull_com, hull_I = {s: [] for s in obj_sides}, {s: [] for s in obj_sides}, {s: [] for s in obj_sides}
    gids = {s: [] for s in obj_sides}
    for _si, spec in enumerate(object_slots):
        for side in _slot_sides(spec):
            _side_slot[(_si, side)] = len(gids[side])
            m, c, I = hull_mass_props(
                _P(_slot_val(spec, "mesh_path", side)).parent,
                _slot_val(spec, "scale", side) if "scale" in spec else 1.0,
                800.0,
            )
            # Measured mass wins over the density estimate, clamp included.
            _tgt = float(_slot_val(spec, "mass", side)) if "mass" in spec else 0.0
            if _tgt > 0.0 and m > 0.0:
                I, m = I * (_tgt / m), _tgt
            hull_mass[side].append(m)
            hull_com[side].append(c)
            hull_I[side].append(I)
            evals, evecs = np.linalg.eigh(I)
            if np.linalg.det(evecs) < 0:
                evecs[:, 0] = -evecs[:, 0]
            import trimesh.transformations as _tt

            T = np.eye(4)
            T[:3, :3] = evecs
            gids[side].append(
                sim_adapter.model.register_body(
                    m, np.maximum(evals, 1e-7), _tt.quaternion_from_matrix(T)
                )
            )
    import os as _os0

    if _os0.environ.get("FDR_HULL_PROPS", "0") == "1":
        scene_adapter.obj_inertia_stack = {
            s: torch.tensor(np.stack(hull_I[s]), dtype=torch.float32, device=device)
            for s in obj_sides
        }  # side -> (S, 3, 3)
    else:
        # USD-authored (visual-mesh) inertia — what PhysX actually simulates.
        from mjlab.utils.lab_api.math import matrix_from_quat as _mfq

        stacks = {}
        for side in obj_sides:
            mats = []
            for spec in object_slots:
                mesh = trimesh.load(_slot_val(spec, "mesh_path", side), force="mesh")
                scale = _slot_val(spec, "scale", side) if "scale" in spec else 1.0
                if scale != 1.0:
                    mesh = mesh.copy()
                    mesh.apply_scale(float(scale))
                _m, _diag, _iq = principal_inertia(mesh, density=800.0)
                _t = float(_slot_val(spec, "mass", side)) if "mass" in spec else 0.0
                if _t > 0.0 and _m > 0.0:
                    _diag = _diag * (_t / _m)
                R = _mfq(torch.tensor(_iq, dtype=torch.float32).unsqueeze(0))[0]
                mats.append(
                    R @ torch.diag(torch.tensor(_diag, dtype=torch.float32)) @ R.T
                )
            stacks[side] = torch.stack(mats).to(device)
        scene_adapter.obj_inertia_stack = stacks

    def _write_hull_props(view, slot_of_row: np.ndarray, side: str) -> None:
        """Overwrite PhysX mass/com/inertia with the hull-parity values (CPU)."""
        n = len(slot_of_row)
        view.set_masses(
            torch.tensor([hull_mass[side][s] for s in slot_of_row]), torch.arange(n)
        )
        coms = view.get_coms().clone().reshape(n, 7)
        coms[:, :3] = torch.tensor(
            np.stack([hull_com[side][s] for s in slot_of_row]), dtype=coms.dtype
        )
        view.set_coms(coms.reshape(*view.get_coms().shape), torch.arange(n))
        view.set_inertias(
            torch.tensor(
                np.stack([hull_I[side][s].reshape(9) for s in slot_of_row]),
                dtype=torch.float32,
            ),
            torch.arange(n),
        )

    import os as _os

    _hull_props = _os.environ.get("FDR_HULL_PROPS", "0") == "1"
    if per_env_mode:
        scene_adapter.assigned_obj_slot = assigned_obj_slot.to(device)
        for side in obj_sides:
            name = f"object_{side}"
            if _hull_props:
                _write_hull_props(
                    objects[name].root_physx_view,
                    assigned_obj_slot.cpu().numpy(),
                    side,
                )
            default_state = torch.zeros(num_envs, 13, device=device)
            default_state[:, 0:3] = scene.env_origins + torch.tensor(
                (0.0, 0.0, 0.2), dtype=torch.float32, device=device
            )
            default_state[:, 3] = 1.0
            scene_adapter.entities[name] = RigidObjectAdapter(
                objects[name], gids[side][0], clock, device, default_state
            )
    else:
        for slot_idx, spec in enumerate(object_slots):
            if _hull_props:
                _write_hull_props(
                    objects[spec["entity_name"]].root_physx_view,
                    np.full(num_envs, slot_idx),
                    _slot_sides(spec)[0],
                )
            park = object_park_position(slot_idx)
            default_state = torch.zeros(num_envs, 13, device=device)
            default_state[:, 0:3] = scene.env_origins + torch.tensor(
                park, dtype=torch.float32, device=device
            )
            default_state[:, 3] = 1.0
            scene_adapter.entities[spec["entity_name"]] = RigidObjectAdapter(
                objects[spec["entity_name"]],
                gids[_slot_sides(spec)[0]][_side_slot[(slot_idx, _slot_sides(spec)[0])]],
                clock, device,
                default_state,
            )

    # sensor adapters (command callback bound after command-term creation);
    # one bank per hand side, each SDF-gated against its own side's object.
    banks: dict[str, LinkContactBank] = {}
    for side in robot_sides:
        names = alllink[side]
        bank = LinkContactBank(link_sensors[side], clock, device)
        bank.side = side
        bank.margins = torch.tensor(
            [meta.body_bound.get(b, 0.03) + 0.015 for b in names],
            dtype=torch.float32, device=device,
        )
        # fingertip rows = the links carrying contact_<side>_<finger>_tip (xhand: 3,6,8,10,12; sharpa: 3,7,11,15,20)
        tip_sites = [meta.sites.get(f"contact_{side}_{f}_tip") for f in ("thumb", "index", "middle", "ring", "pinky")]
        if all(t is not None and t.body_name in names for t in tip_sites):
            bank.tip_rows = tuple(names.index(t.body_name) for t in tip_sites)
        ids, _ = robot_adapter.find_bodies(names, preserve_order=True)
        bank.link_pos_fn = (
            lambda ids=tuple(ids): robot_adapter.data.body_link_pos_w[:, list(ids)]
        )
        p = side[0]
        scene_adapter.sensors[f"{p}_alllink_contact"] = AlllinkContactAdapter(bank)
        scene_adapter.sensors[f"{p}_alllink_contact_pos"] = AlllinkContactPosAdapter(bank)
        scene_adapter.sensors[f"{p}_fingertip_contact"] = FingertipContactAdapter(bank)
        scene_adapter.sensors[f"{p}_fingertip_penetration"] = FingertipPenetrationAdapter(
            bank, side=side
        )
        banks[side] = bank

    def _bind(slot_fn, cmd_fn) -> None:
        del slot_fn  # slot attribution is SDF-side now
        for bank in banks.values():
            bank.command_fn = cmd_fn

    scene_adapter._bind_sensor_callbacks = _bind

    return scene_adapter, sim_adapter, robot_adapter

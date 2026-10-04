"""Isaac twin of envs.create_envs: same manager builders, Isaac scene/sim.
Single scene per process -> no dedicated eval env."""

from __future__ import annotations

from pathlib import Path

import torch

from mjlab.rl import RslRlVecEnvWrapper

from envs import (  # builders only; mujoco-specific side effects are gated
    build_actions,
    build_commands,
    build_curriculum,
    build_events,
    build_observations,
    build_rewards,
    build_terminations,
)
from envs.scene._object_setup import object_pool_dir

from .convert_assets import COLVIS_OBJ_DIRNAME, colvis_enabled
from .env import IsaacEnvCfg, IsaacManagerBasedRlEnv
from .mjcf_meta import load_robot_mjcf_meta

_SIM_DIR = Path(__file__).resolve().parents[3]  # simulation/


def _discover_object_slots(cfg, per_env_mode: bool) -> tuple[list[dict], torch.Tensor | None]:
    """Object slot specs (+ per-env assignment) from the packed motion .pt.
    Per-env mode: single entity, slot-indexed SDF, motion-proportional envs."""
    motion_cfg = cfg.commands
    packed = torch.load(str(motion_cfg.motion_file), weights_only=False)
    pool_dir = object_pool_dir(motion_cfg)

    # FDR_COLVIS: debug assets whose visuals ARE the collision hulls.
    colvis = colvis_enabled()
    usd_dirname = COLVIS_OBJ_DIRNAME if colvis else ".isaac_usd"

    if not packed.get("right_object_mesh_dirs") and not packed.get("left_object_mesh_dirs"):
        # Singular pt (one object per side, incl. bimanual pairs): one entity per
        # side, mirroring the mujoco single-object discovery. No per-env mode.
        return _discover_single_object_sides(
            motion_cfg, packed, pool_dir, usd_dirname, colvis
        ), None
    if packed.get("right_object_mesh_dirs") and packed.get("left_object_mesh_dirs"):
        # Bimanual pair-slot pt: one PAIR per slot, both sides spawned per env.
        assert per_env_mode, "bimanual multi pt requires per-env-object mode"
        return _discover_bimanual_pair_slots(
            cfg, motion_cfg, packed, pool_dir, usd_dirname, colvis
        )
    # Single-side multi pt (right- or left-hand-only training).
    side = "right" if packed.get("right_object_mesh_dirs") else "left"
    dirs = list(packed[f"{side}_object_mesh_dirs"])
    scales = list(packed.get(f"{side}_object_mesh_scales") or [1.0] * len(dirs))

    slots: list[dict] = []
    names: list[str] = []
    mesh_paths: list[str] = []
    for slot, rel in enumerate(dirs):
        obj_dir = pool_dir / rel
        usd = obj_dir / usd_dirname / "object.usd"
        if not usd.exists():
            raise FileNotFoundError(
                f"{usd} missing — run: python src/simulator/isaacsim/convert_assets.py"
                + (" --collision-visual" if colvis else "")
            )
        entity_name = f"object_{side}_{slot}"
        slots.append(
            {
                "entity_name": entity_name,
                "usd_path": str(usd),
                "mesh_path": str(obj_dir / "visual.obj"),
                "scale": float(scales[slot]),
            }
        )
        names.append(entity_name)
        mesh_paths.append(str(obj_dir / "visual.obj"))

    motion_cfg.object.mesh_paths = {side: mesh_paths}
    motion_cfg.object.mesh_scales = {side: [float(s) for s in scales]}

    if not per_env_mode:
        motion_cfg.object.entity_names = {side: names}
        return slots, None

    motion_cfg.object.entity_names = {side: f"object_{side}"}

    # Proportional env allotment (largest remainder), every object >= 1 env.
    traj_slot = packed[f"motion_object_slot_{side}"].long()
    return slots, _alloc_by_motion_count(cfg, traj_slot, len(dirs))


def _alloc_by_motion_count(cfg, traj_slot: torch.Tensor, S: int) -> torch.Tensor:
    """(num_envs,) slot per env, proportional to each slot's motion count (largest remainder,
    every slot with motions >= 1 env)."""
    num_envs = int(cfg.scene.num_envs)
    counts = torch.bincount(traj_slot, minlength=S).float()
    n_used = int((counts > 0).sum())
    if num_envs < n_used:
        raise ValueError(
            f"per-env-object mode needs num_envs >= object slots ({num_envs} < {n_used})"
        )
    quota = counts / counts.sum() * num_envs
    alloc = quota.floor().long()
    alloc[counts > 0] = alloc[counts > 0].clamp(min=1)
    rem = num_envs - int(alloc.sum())
    if rem > 0:
        order = torch.argsort(quota - quota.floor(), descending=True)
        order = order[counts[order] > 0]
        for j in range(rem):
            alloc[order[j % order.numel()]] += 1
    elif rem < 0:
        order = torch.argsort(alloc, descending=True)
        j = 0
        while rem < 0:
            s = order[j % S]
            if alloc[s] > 1:
                alloc[s] -= 1
                rem += 1
            j += 1
    return torch.repeat_interleave(torch.arange(S), alloc)[:num_envs]


def _alloc_envs_to_slots(cfg, packed, S: int) -> torch.Tensor:
    return _alloc_by_motion_count(cfg, packed["motion_object_slot_right"].long(), S)


def _discover_bimanual_pair_slots(
    cfg, motion_cfg, packed, pool_dir: Path, usd_dirname: str, colvis: bool
) -> tuple[list[dict], torch.Tensor]:
    """Slot specs for a bimanual pair-slot pt: each slot carries BOTH sides'
    assets ({"usd_path": {side: ...}, ...}); one object entity per side."""
    sides = ("right", "left")
    dirs = {s: list(packed[f"{s}_object_mesh_dirs"]) for s in sides}
    scales = {
        s: list(packed.get(f"{s}_object_mesh_scales") or [1.0] * len(dirs[s]))
        for s in sides
    }
    # Optional measured masses (kg, 0 = derive from density); bypasses the hull clamp.
    masses = {
        s: list(packed.get(f"{s}_object_masses") or [0.0] * len(dirs[s]))
        for s in sides
    }
    S = len(dirs["right"])
    assert len(dirs["left"]) == S, "pair pt must have equal-length side slot lists"

    # Shared-object pts (every slot's two sides = same mesh) spawn ONE entity per env; the
    # left side aliases it. Mixed pts spawn both; a shared slot's left is an inert ghost.
    shared_slots = [dirs["right"][i] == dirs["left"][i] for i in range(S)]
    shared = all(shared_slots)
    mixed = any(shared_slots) and not shared
    spawn_sides = ("right",) if shared else sides
    if mixed:
        print(
            f"[isaac] mixed pair pt: {sum(shared_slots)}/{S} shared slots -> "
            "left ghost + per-env alias to object_right"
        )

    slots: list[dict] = []
    for slot in range(S):
        spec = {"usd_path": {}, "mesh_path": {}, "scale": {}, "mass": {}}
        if mixed and shared_slots[slot]:
            spec["ghost"] = ("left",)
        for s in spawn_sides:
            obj_dir = pool_dir / str(dirs[s][slot])
            usd = obj_dir / usd_dirname / "object.usd"
            if not usd.exists():
                raise FileNotFoundError(
                    f"{usd} missing — run: python src/simulator/isaacsim/convert_assets.py"
                    + (" --collision-visual" if colvis else "")
                )
            spec["usd_path"][s] = str(usd)
            spec["mesh_path"][s] = str(obj_dir / "visual.obj")
            spec["scale"][s] = float(scales[s][slot])
            spec["mass"][s] = float(masses[s][slot])
        slots.append(spec)

    motion_cfg.object.entity_names = {
        s: ("object_right" if shared else f"object_{s}") for s in sides
    }
    motion_cfg.object.shared_slots = shared_slots if mixed else None
    motion_cfg.object.mesh_paths = {
        s: [str(pool_dir / str(r) / "visual.obj") for r in dirs[s]] for s in sides
    }
    motion_cfg.object.mesh_scales = {
        s: [float(x) for x in scales[s]] for s in sides
    }
    return slots, _alloc_envs_to_slots(cfg, packed, S)


def _discover_single_object_sides(
    motion_cfg, packed, pool_dir: Path, usd_dirname: str, colvis: bool
) -> list[dict]:
    """Slot specs for a singular pt: entity object_<side> per populated side
    (mirrors envs.scene._object_setup's single-object discovery)."""
    sides = [
        s for s in ("right", "left") if packed.get(f"{s}_object_mesh_dir") is not None
    ]
    assert sides, "no *_object_mesh_dir(s) in motion .pt"
    # Shared object (both slots -> same mesh): ONE physical entity, the left
    # side aliases it (DexMachina-style single-object bimanual).
    shared = (
        len(sides) == 2
        and packed["right_object_mesh_dir"] == packed["left_object_mesh_dir"]
    )
    slots: list[dict] = []
    entity_names: dict[str, str] = {}
    mesh_paths: dict[str, str] = {}
    mesh_scales: dict[str, float] = {}
    for side in sides:
        if shared and side == "left":
            entity_names["left"] = "object_right"
            mesh_paths["left"] = mesh_paths["right"]
            mesh_scales["left"] = mesh_scales["right"]
            continue
        obj_dir = pool_dir / str(packed[f"{side}_object_mesh_dir"])
        usd = obj_dir / usd_dirname / "object.usd"
        if not usd.exists():
            raise FileNotFoundError(
                f"{usd} missing — run: python src/simulator/isaacsim/convert_assets.py"
                + (" --collision-visual" if colvis else "")
            )
        scale = float(packed.get(f"{side}_object_mesh_scale", 1.0))
        entity_name = f"object_{side}"
        slots.append(
            {
                "entity_name": entity_name,
                "usd_path": str(usd),
                "mesh_path": str(obj_dir / "visual.obj"),
                "scale": scale,
            }
        )
        entity_names[side] = entity_name
        mesh_paths[side] = str(obj_dir / "visual.obj")
        mesh_scales[side] = scale
    motion_cfg.object.entity_names = entity_names
    motion_cfg.object.mesh_paths = mesh_paths
    motion_cfg.object.mesh_scales = mesh_scales
    return slots


def _make_manager_cfgs(cfg) -> dict:
    entity_name = cfg.robot.entity_name
    command_name = cfg.commands.name
    # Route the command through the Isaac subclass (sensor-wiring delta only).
    cfg.commands._target_ = (
        "simulator.isaacsim.commands_isaac.IsaacMotionTrackingCommandCfg"
    )
    return dict(
        observations=build_observations(
            cfg.obs, command_name=command_name, entity_name=entity_name
        ),
        rewards=build_rewards(
            cfg.rewards, command_name=command_name, entity_name=entity_name
        ),
        commands=build_commands(cfg.commands),
        actions=build_actions(cfg.actions),
        terminations=build_terminations(
            cfg.terminations, command_name=command_name, entity_name=entity_name
        ),
        curriculum=build_curriculum(
            cfg.curriculum, command_name=command_name, entity_name=entity_name
        ),
        events=build_events(cfg.events),
    )


def create_envs_isaac(cfg, device: str = "cuda"):
    """Build the Isaac train env (+ shared-eval facade), wrapper-compatible
    with the mujoco ``create_envs`` return signature."""
    from isaaclab.sim import PhysxCfg, SimulationCfg, SimulationContext

    from .scene import build_isaac_scene

    import os

    per_env_mode = os.environ.get("FDR_SLOTS", "0") != "1"
    object_slots, assigned = _discover_object_slots(cfg, per_env_mode)
    # Singular (per-pair) pts have no slot table — legacy replicated scene.
    per_env_mode = per_env_mode and assigned is not None

    xml_path = _SIM_DIR / str(cfg.robot.xml_path)
    meta = load_robot_mjcf_meta(xml_path)

    px = cfg.sim.physx
    timestep = float(cfg.sim.dt)
    decimation = int(cfg.get("decimation", 6))
    # Default scene material: converted USDs hide colliders behind instanced
    # references (per-prim binding fails), so friction/compliance go here.
    import isaaclab.sim as sim_utils

    sim_ctx = SimulationContext(
        SimulationCfg(
            dt=timestep,
            render_interval=decimation,
            device=device,
            # Viewer sessions need USD transforms synced (fabric leaves link
            # xforms stale -> "exploded" hands in the viewport); headless keeps
            # the fast fabric path.
            use_fabric=bool(cfg.get("headless", True)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=float(px.get("friction", 1.0)),
                dynamic_friction=float(px.get("friction", 1.0)),
                restitution=0.0,
                friction_combine_mode=str(px.get("nonobject_friction_combine_mode", "average")),
                compliant_contact_stiffness=float(
                    px.get("compliant_contact_stiffness", 0.0)
                ),
                compliant_contact_damping=float(
                    px.get("compliant_contact_damping", 0.0)
                ),
            ),
            physx=PhysxCfg(
                bounce_threshold_velocity=float(px.bounce_threshold_velocity),
                # REGRIND-style rigid contacts need a high global position-iter
                # cap to resolve accurately; default None keeps IsaacLab's.
                **(
                    {"max_position_iteration_count": int(px.max_position_iteration_count)}
                    if px.get("max_position_iteration_count", None) is not None
                    else {}
                ),
                gpu_max_num_partitions=int(px.gpu_max_num_partitions),
                gpu_found_lost_pairs_capacity=int(px.gpu_found_lost_pairs_capacity),
                gpu_total_aggregate_pairs_capacity=int(
                    px.gpu_total_aggregate_pairs_capacity
                ),
                gpu_max_rigid_patch_count=int(px.gpu_max_rigid_patch_count),
            ),
        )
    )

    scene_adapter, sim_adapter, _robot_adapter = build_isaac_scene(
        cfg, sim_ctx, device, meta, object_slots, assigned_obj_slot=assigned
    )

    managers = _make_manager_cfgs(cfg)
    env_cfg = IsaacEnvCfg(
        decimation=decimation,
        physics_dt=timestep,
        episode_length_s=float(cfg.get("episode_length_s", 20.0)),
        seed=cfg.get("seed", None),
        num_envs=int(cfg.scene.num_envs),
        **managers,
    )
    env = IsaacManagerBasedRlEnv(env_cfg, scene_adapter, sim_adapter, device=device)

    if per_env_mode:
        # Object-aware eval: each motion's replicas run on envs hosting its
        # object; the sweep plan + isaac setup callback replace the defaults.
        from evaluation.isaac import build_eval_plan

        cmd = env.command_manager.get_term(cfg.eval.command_name)
        tos = cmd.motion_lib.traj_obj_slot
        env._isaac_eval_plan = build_eval_plan(
            scene_adapter.assigned_obj_slot,
            tos["right"] if "right" in tos else tos["left"],
            int(cfg.eval.num_per_motion),
        )
        cfg.eval.callbacks[0]._target_ = (
            "evaluation.isaac.IsaacMotionTrackingEvalSetup"
        )
        # Shared-env eval pauses training for one pass per batch; the configured
        # cadence stands either way, so just report the sweep cost.
        n_batches = max(1, len(env._isaac_eval_plan))
        every = cfg.eval.get("interval", "?")
        print(f"[isaac] eval sweep: {n_batches} batches, every {every} steps")

    from evaluation.isaac import SharedEvalEnv

    train_env = RslRlVecEnvWrapper(env)
    # SharedEvalEnv already speaks the rsl_rl 4-tuple surface evaluation.evaluate needs;
    # wrapping it again would double-convert the step return.
    eval_env = SharedEvalEnv(env, cfg.eval)
    return train_env, eval_env


__all__ = ["create_envs_isaac"]

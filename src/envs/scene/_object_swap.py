"""Slot-free multi-object: ONE entity, per-world batched model rows (geom_dataid
etc.) select each env's object hulls."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import mujoco
import numpy as np
from mjlab.entity import EntityCfg

from ._object_entity import get_object_spec

# Per-world model fields (all batched `(*, ...)` in MJWarp); body/dof inertia
# must follow geom_dataid or the swapped shape keeps the template's mass matrix.
SWAP_MODEL_FIELDS = (
    "geom_dataid",
    "geom_pos",
    "geom_quat",
    "geom_size",
    "geom_rbound",
    "geom_aabb",
    "body_mass",
    "body_inertia",
    "body_ipos",
    "body_iquat",
    "body_invweight0",
    "body_subtreemass",
    "dof_invweight0",
)

# 1mm tet for DISABLED padding geoms: zeroed rbound/aabb is NOT enough (plane
# pairs bypass sphere culling and would hit the full hull behind geom_dataid).
NULL_MESH_OBJ = b"""v 0 0 0
v 0.001 0 0
v 0 0.001 0
v 0 0 0.001
f 1 2 3
f 1 4 2
f 1 3 4
f 2 4 3
"""


def _hull_files(obj_dir: str | Path) -> list[Path]:
    return sorted(Path(obj_dir, "convex").glob("*.obj"), key=lambda p: int(p.stem))


def _decimate_hull_obj(src: Path, dst: Path, maxhullvert: int) -> None:
    """Write a <=maxhullvert-vertex convex copy of one hull piece via a throwaway
    compile; verts mapped canonical -> file frame."""
    from scipy.spatial import ConvexHull

    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="o")
    mesh = spec.add_mesh()
    mesh.name = "h"
    mesh.file = "h.obj"
    mesh.maxhullvert = maxhullvert
    spec.assets["h.obj"] = src.read_bytes()
    body.add_geom(
        name="g", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="h",
        contype=2, conaffinity=3, density=500.0,
    )
    m = spec.compile()
    i = m.mesh("h").id
    ga, va = m.mesh_graphadr[i], m.mesh_vertadr[i]
    if ga < 0:  # small mesh, no hull graph built: already cheap, copy as-is
        dst.write_bytes(src.read_bytes())
        return
    numvert = m.mesh_graph[ga]
    globalid = m.mesh_graph[ga + 2 + numvert : ga + 2 + 2 * numvert]
    verts = m.mesh_vert[va + globalid]
    # canonical -> file frame (empirically validated: bbox error 0 vs source)
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, m.mesh_quat[i])
    verts = verts @ rot.reshape(3, 3).T + m.mesh_pos[i]
    hull = ConvexHull(verts)
    tmp = dst.with_suffix(f".tmp{os.getpid()}")  # per-process: parallel-safe
    with open(tmp, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        # qhull simplices are unordered; irrelevant, the compiler re-hulls.
        for s in hull.simplices:
            f.write(f"f {s[0] + 1} {s[1] + 1} {s[2] + 1}\n")
    tmp.replace(dst)


def maybe_decimated_dirs(obj_dirs: list[str]) -> list[str]:
    """Mirror obj dirs with decimated hulls when FDR_SWAP_MAXHULLVERT=<K>
    (0 = off; optional since prebuilt graph injection). visual.obj = symlink."""
    mhv = int(os.environ.get("FDR_SWAP_MAXHULLVERT", "0") or 0)
    if mhv <= 0:
        return obj_dirs
    # v2: file-frame output (v1 caches hold canonical-frame collapsed hulls)
    root = Path(
        os.environ.get(
            "FDR_SWAP_DECIM_CACHE",
            Path.home() / ".cache" / "dexmanip" / "swap_decim",
        )
    ) / f"mhv{mhv}-v2"
    out = []
    for d in obj_dirs:
        src = Path(d).resolve()
        mdir = root / hashlib.sha1(str(src).encode()).hexdigest()[:16]
        (mdir / "convex").mkdir(parents=True, exist_ok=True)
        for f in _hull_files(src):
            dst = mdir / "convex" / f.name
            if not dst.exists():
                _decimate_hull_obj(f, dst, mhv)
        vis = mdir / "visual.obj"
        if not vis.exists():
            vis.symlink_to(src / "visual.obj")
        out.append(str(mdir))
    return out


def swap_mesh_name(body_name: str, obj_idx: int, hull: int) -> str:
    return f"{body_name}_swap_o{obj_idx}_h{hull}"


def swap_visual_mesh_name(body_name: str, obj_idx: int) -> str:
    return f"{body_name}_swap_vis_{obj_idx}"


def get_swap_object_cfg(
    obj_dirs: list[str],
    scales: list[float],
    name: str,
    density: float,
    p_max: int,
) -> EntityCfg:
    """Single swap entity: template geometry = object 0, null-tet padding up to
    ``p_max`` collision geoms, all objects' hulls attached as orphan assets."""

    def spec_fn(
        dirs=tuple(str(d) for d in obj_dirs),
        sc=tuple(float(s) for s in scales),
        n=name,
        dn=float(density),
        p=int(p_max),
    ) -> mujoco.MjSpec:
        spec = get_object_spec(dirs[0], n, dn, sc[0])
        body = next(b for b in spec.bodies if b.name == n)

        null_mesh = spec.add_mesh()
        null_mesh.name = f"{n}_null_tet"
        null_mesh.file = f"{n}_null_tet.obj"
        spec.assets[f"{n}_null_tet.obj"] = NULL_MESH_OBJ

        n_template = len(_hull_files(dirs[0]))
        for k in range(n_template, p):
            body.add_geom(
                name=f"{n}_col_{k}",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=f"{n}_null_tet",
                contype=2,
                conaffinity=3,
                condim=3,
                friction=(2.0, 0.05, 0.05),
                rgba=(0, 0, 0, 0),
                group=3,
                density=dn,
            )

        for i, d in enumerate(dirs):
            for f in _hull_files(d):
                mesh_name = swap_mesh_name(n, i, int(f.stem))
                mesh = spec.add_mesh()
                mesh.name = mesh_name
                mesh.file = f"{mesh_name}.obj"
                mesh.scale = (sc[i], sc[i], sc[i])
                spec.assets[f"{mesh_name}.obj"] = f.read_bytes()

        # Render-only opt-in: ship every visual.obj as an orphan asset for
        # offline render retargeting; never set in training (model bloat).
        if os.environ.get("FDR_SWAP_VISUAL_ASSETS"):
            for i, d in enumerate(dirs):
                mesh_name = swap_visual_mesh_name(n, i)
                mesh = spec.add_mesh()
                mesh.name = mesh_name
                mesh.file = f"{mesh_name}.obj"
                mesh.scale = (sc[i], sc[i], sc[i])
                # visual.obj in a decimated mirror dir is a symlink to the
                # original (maybe_decimated_dirs), so this reads full-res.
                spec.assets[f"{mesh_name}.obj"] = (Path(d) / "visual.obj").read_bytes()
        return spec

    return EntityCfg(
        spec_fn=spec_fn,
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


def harvest_swap_row_table(
    obj_dirs: list[str],
    scales: list[float],
    density: float,
    p_max: int,
    mesh_blob_fn=None,
) -> tuple[dict[str, np.ndarray], list[list] | None]:
    """Per-object model rows from standalone compiles (= single-object ground
    truth); mesh_blob_fn optionally harvests hull graph/poly blobs (see docs)."""
    n_obj = len(obj_dirs)
    tbl = {
        "nhull": np.zeros((n_obj,), np.int64),
        "gpos": np.zeros((n_obj, p_max, 3), np.float32),
        "gquat": np.zeros((n_obj, p_max, 4), np.float32),
        "gsize": np.full((n_obj, p_max, 3), 1e-3, np.float32),
        "grbound": np.full((n_obj, p_max), 2e-3, np.float32),
        "gaabb": np.zeros((n_obj, p_max, 2, 3), np.float32),
        "bmass": np.zeros((n_obj,), np.float32),
        "binertia": np.zeros((n_obj, 3), np.float32),
        "bipos": np.zeros((n_obj, 3), np.float32),
        "biquat": np.zeros((n_obj, 4), np.float32),
        "binvw": np.zeros((n_obj, 2), np.float32),
        "bsubm": np.zeros((n_obj,), np.float32),
        "dinvw": np.zeros((n_obj, 6), np.float32),
    }
    tbl["gquat"][:, :, 0] = 1.0
    tbl["gaabb"][:, :, 1, :] = 2e-3
    hulls: list[list] | None = [] if mesh_blob_fn is not None else None
    for i, obj_dir in enumerate(obj_dirs):
        spec = get_object_spec(obj_dir, "obj", float(density), float(scales[i]))
        m = spec.compile()
        b = m.body("obj").id
        nh = len(_hull_files(obj_dir))
        if nh > p_max:
            raise ValueError(f"object {obj_dir}: {nh} hulls > p_max {p_max}")
        tbl["nhull"][i] = nh
        blobs = []
        for k in range(nh):
            g = m.geom(f"obj_col_{k}").id
            tbl["gpos"][i, k] = m.geom_pos[g]
            tbl["gquat"][i, k] = m.geom_quat[g]
            tbl["gsize"][i, k] = m.geom_size[g]
            tbl["grbound"][i, k] = m.geom_rbound[g]
            tbl["gaabb"][i, k] = m.geom_aabb[g].reshape(2, 3)
            if mesh_blob_fn is not None:
                blobs.append(mesh_blob_fn(m, m.geom_dataid[g]))
        if hulls is not None:
            hulls.append(blobs)
        tbl["bmass"][i] = m.body_mass[b]
        tbl["binertia"][i] = m.body_inertia[b]
        tbl["bipos"][i] = m.body_ipos[b]
        tbl["biquat"][i] = m.body_iquat[b]
        tbl["binvw"][i] = m.body_invweight0[b]
        tbl["bsubm"][i] = m.body_subtreemass[b]
        d0 = m.body("obj").dofadr[0]
        tbl["dinvw"][i] = m.dof_invweight0[d0 : d0 + 6]
    return tbl, hulls

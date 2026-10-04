"""MJCF -> URDF generator for the xhand assets (right / left / bimanual).
Reproduces the conventions of the checked-in xhand_right.urdf; validate with
--check-right which numerically diffs a regenerated right URDF against it."""

import argparse
import os

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

SIM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
XHAND_DIR = os.path.join(SIM_DIR, "assets", "robot", "xhand")

# MJCF palm bodies carry the hand mesh; URDF uses the mesh-style link name.
MJCF_TO_URDF_LINK = {
    "R_forearm_rot_y_link": "right_hand_link",
    "L_forearm_rot_y_link": "left_hand_link",
}


def _fmt(x, digits=10):
    s = f"{x:.{digits}g}"
    return "0" if s in ("-0", "0.0", "-0.0") else s


def _vec(v, digits=10):
    return " ".join(_fmt(x, digits) for x in v)


def _quat_to_rpy(quat_wxyz):
    q = np.asarray(quat_wxyz, dtype=np.float64)
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz")


def _link_name(mjcf_body_name):
    return MJCF_TO_URDF_LINK.get(mjcf_body_name, mjcf_body_name)


def _inertial_xml(model, bid, indent="    "):
    ipos = model.body_ipos[bid]
    iq = model.body_iquat[bid]
    diag = model.body_inertia[bid]
    mass = model.body_mass[bid]
    R = Rotation.from_quat([iq[1], iq[2], iq[3], iq[0]]).as_matrix()
    I = R @ np.diag(diag) @ R.T
    return (
        f"{indent}<inertial>\n"
        f"{indent}  <origin xyz=\"{_vec(ipos)}\" rpy=\"0 0 0\"/>\n"
        f"{indent}  <mass value=\"{_fmt(mass)}\"/>\n"
        f"{indent}  <inertia ixx=\"{_fmt(I[0, 0])}\" ixy=\"{_fmt(I[0, 1])}\" "
        f"ixz=\"{_fmt(I[0, 2])}\" iyy=\"{_fmt(I[1, 1])}\" iyz=\"{_fmt(I[1, 2])}\" "
        f"izz=\"{_fmt(I[2, 2])}\"/>\n"
        f"{indent}</inertial>\n"
    )


def _geoms_xml(model, spec_geoms, bid, mesh_files=None):
    """Visual (group 1, named *_visual) + collision (collision_*) geom entries.
    Geom origins MUST come from the spec: the compiled model re-centers mesh
    assets (CoM/principal frame) and bakes the compensation into geom_pos/quat,
    which is wrong for a URDF that references the RAW mesh files."""
    out = []
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) != bid:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        mesh_id = int(model.geom_dataid[g])
        if mesh_id < 0:
            continue  # marker primitives are not exported
        mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
        pos, quat = spec_geoms[name]
        rpy = _quat_to_rpy(quat)
        origin = f"<origin xyz=\"{_vec(pos)}\" rpy=\"{_vec(rpy, 6)}\"/>"
        # Use the asset's actual file, not its name: an MJCF may point a mesh
        # at a path whose basename differs from the asset name.
        rel = (mesh_files or {}).get(mesh_name, f"{mesh_name}.STL")
        rel = os.path.normpath(os.path.join("../meshes", rel))
        mesh = f"<geometry><mesh filename=\"{rel}\"/></geometry>"
        if name.startswith("collision"):
            out.append(
                f"    <collision name=\"{name}\">\n"
                f"      {origin}\n      {mesh}\n    </collision>\n"
            )
        else:
            rgba = model.geom_rgba[g]
            out.append(
                f"    <visual>\n      {origin}\n      {mesh}\n"
                f"      <material name=\"mat_{mesh_name}\">\n"
                f"        <color rgba=\"{_vec(rgba, 6)}\"/>\n      </material>\n"
                f"    </visual>\n"
            )
    return "".join(out)


TRACK_LINK = (
    "  <link name=\"{name}\">\n"
    "    <inertial>\n"
    "      <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>\n"
    "      <mass value=\"1e-06\"/>\n"
    "      <inertia ixx=\"1e-09\" ixy=\"0\" ixz=\"0\" iyy=\"1e-09\" iyz=\"0\" izz=\"1e-09\"/>\n"
    "    </inertial>\n"
    "  </link>\n"
)


def _track_sites_xml(model, bid, link):
    """track_hand_* sites become massless fixed links (isaac tip-pose reads)."""
    out = []
    for s in range(model.nsite):
        if int(model.site_bodyid[s]) != bid:
            continue
        sname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, s) or ""
        if not sname.startswith("track_hand_"):
            continue
        rpy = _quat_to_rpy(model.site_quat[s])
        out.append(
            TRACK_LINK.format(name=sname)
            + f"  <joint name=\"{sname}_fixed\" type=\"fixed\">\n"
            f"    <origin xyz=\"{_vec(model.site_pos[s])}\" rpy=\"{_vec(rpy, 6)}\"/>\n"
            f"    <parent link=\"{link}\"/>\n"
            f"    <child link=\"{sname}\"/>\n"
            f"  </joint>\n"
        )
    return "".join(out)


def _chain_xml(xml_path):
    """URDF links+joints for one hand chain, rooted at (and excluding) 'world'."""
    model = mujoco.MjModel.from_xml_path(xml_path)
    spec = mujoco.MjSpec.from_file(xml_path)
    mesh_files = {m.name: m.file for m in spec.meshes if m.name}
    spec_geoms = {
        g.name: (np.array(g.pos, dtype=np.float64), np.array(g.quat, dtype=np.float64))
        for b in spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_BODY)
        for g in b.geoms
        if g.name
    }
    parts = []
    for bid in range(1, model.nbody):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        link = _link_name(bname)
        parts.append(
            f"  <link name=\"{link}\">\n"
            + _inertial_xml(model, bid)
            + _geoms_xml(model, spec_geoms, bid, mesh_files)
            + "  </link>\n"
        )

        assert model.body_jntnum[bid] == 1, f"{bname}: expected exactly 1 joint"
        j = int(model.body_jntadr[bid])
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        jtype = int(model.jnt_type[j])
        assert np.allclose(model.jnt_pos[j], 0.0), f"{jname}: joint offset unsupported"
        urdf_type = {
            int(mujoco.mjtJoint.mjJNT_SLIDE): "prismatic",
            int(mujoco.mjtJoint.mjJNT_HINGE): "revolute",
        }[jtype]
        vel = 5 if urdf_type == "prismatic" else 10
        parent_bid = int(model.body_parentid[bid])
        parent = (
            "world" if parent_bid == 0
            else _link_name(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_bid))
        )
        rpy = _quat_to_rpy(model.body_quat[bid])
        lo, hi = model.jnt_range[j]
        parts.append(
            f"  <joint name=\"{jname}\" type=\"{urdf_type}\">\n"
            f"    <origin xyz=\"{_vec(model.body_pos[bid])}\" rpy=\"{_vec(rpy, 10)}\"/>\n"
            f"    <parent link=\"{parent}\"/>\n"
            f"    <child link=\"{link}\"/>\n"
            f"    <axis xyz=\"{_vec(model.jnt_axis[j])}\"/>\n"
            f"    <limit lower=\"{_fmt(lo)}\" upper=\"{_fmt(hi)}\" "
            f"effort=\"100\" velocity=\"{vel}\"/>\n"
            f"  </joint>\n"
        )
        parts.append(_track_sites_xml(model, bid, link))
    return "".join(parts)


WORLD_LINK = (
    "  <link name=\"world\">\n"
    "    <inertial>\n"
    "      <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>\n"
    "      <mass value=\"0.001\"/>\n"
    "      <inertia ixx=\"1e-06\" ixy=\"0\" ixz=\"0\" iyy=\"1e-06\" iyz=\"0\" izz=\"1e-06\"/>\n"
    "    </inertial>\n"
    "  </link>\n"
)


def generate(variant, out_path, srcs=None):
    # srcs overrides the variant's stock MJCF(s).
    srcs = srcs or {
        "right": ["right.xml"],
        "left": ["left.xml"],
        "bimanual": ["right.xml", "left.xml"],
    }[variant]
    dof = 18 * len(srcs)
    body = "".join(_chain_xml(os.path.join(XHAND_DIR, x)) for x in srcs)
    urdf = (
        "<?xml version=\"1.0\"?>\n"
        f"<!-- xhand {variant} dexterous hand. Auto-generated by mjcf_to_urdf.py from "
        f"{' + '.join(srcs)}. {dof} actuated DoF: per hand a 6-DoF wrist serial chain "
        "(3 prismatic + 3 revolute) + 12 finger joints. -->\n"
        f"<robot name=\"xhand_{variant}\">\n" + WORLD_LINK + body + "</robot>\n"
    )
    with open(out_path, "w") as f:
        f.write(urdf)
    print(f"wrote {out_path}")


def check_right(tmp_path):
    """Regenerate the right URDF and numerically diff it against the checked-in
    xhand_right.urdf (origins, axes, limits, inertials, mesh files)."""
    import xml.etree.ElementTree as ET

    generate("right", tmp_path)
    ref = ET.parse(os.path.join(XHAND_DIR, "urdf", "xhand_right.urdf")).getroot()
    new = ET.parse(tmp_path).getroot()

    def _f(s):
        return np.array([float(x) for x in s.split()])

    # Geometry origins: every visual/collision origin+mesh must match the
    # reference (catches compiled-vs-spec mesh-frame contamination).
    def _geom_entries(root_el):
        out = {}
        for link in root_el.findall("link"):
            for i, col in enumerate(link.findall("collision")):
                out[col.get("name") or f"{link.get('name')}_col{i}"] = col
            for i, vis in enumerate(link.findall("visual")):
                mesh = vis.find("geometry/mesh")
                out[f"{link.get('name')}_vis_{mesh.get('filename')}"] = vis
        return out

    ref_g, new_g = _geom_entries(ref), _geom_entries(new)
    assert set(ref_g) == set(new_g), f"geom sets differ: {set(ref_g) ^ set(new_g)}"
    worst_g = 0.0
    for name, rg in ref_g.items():
        ro, no = rg.find("origin"), new_g[name].find("origin")
        for attr in ("xyz", "rpy"):
            d = np.abs(_f(ro.get(attr)) - _f(no.get(attr))).max()
            worst_g = max(worst_g, d)
            assert d < 2e-6, f"geom {name} origin@{attr}: {ro.get(attr)} != {no.get(attr)}"
    print(f"check-right geom origins OK: worst diff = {worst_g:.2e}")

    worst = 0.0
    for tag, key in [("link", "name"), ("joint", "name")]:
        ref_items = {e.get(key): e for e in ref.findall(tag)}
        new_items = {e.get(key): e for e in new.findall(tag)}
        assert set(ref_items) == set(new_items), (
            f"{tag} sets differ: {set(ref_items) ^ set(new_items)}"
        )
        for name, re_ in ref_items.items():
            ne = new_items[name]
            for path in ["inertial/origin", "inertial/mass", "inertial/inertia",
                         "origin", "axis", "limit"]:
                r, n = re_.find(path), ne.find(path)
                if r is None and n is None:
                    continue
                assert (r is None) == (n is None), f"{tag} {name}: {path} presence"
                # Checked-in URDF predates the tipfix site move; track-link
                # origins may differ by up to the tipfix delta (~2 cm).
                tol = 0.03 if name.startswith("track_hand_") and path == "origin" else 2e-6
                for attr, rv in r.attrib.items():
                    nv = n.get(attr)
                    try:
                        d = np.abs(_f(rv) - _f(nv)).max()
                    except ValueError:
                        assert rv == nv, f"{tag} {name} {path}@{attr}: {rv} != {nv}"
                        continue
                    if tol == 0.03 and d > 2e-6:
                        print(f"  tipfix delta on {name}: {rv} -> {nv}")
                    worst = max(worst, d if tol < 0.03 else 0.0)
                    assert d < tol, f"{tag} {name} {path}@{attr}: {rv} != {nv} (d={d})"
    print(f"check-right OK: worst numeric diff vs checked-in URDF = {worst:.2e} "
          "(track-site tipfix deltas exempt)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=["right", "left", "bimanual"], default="bimanual")
    p.add_argument("--out", default=None)
    p.add_argument("--check-right", action="store_true")
    p.add_argument("--src", nargs="+", default=None,
                   help="MJCF basename(s) under assets/robot/xhand to convert instead "
                        "of the variant's stock file(s).")
    args = p.parse_args()
    if args.check_right:
        check_right("/tmp/xhand_right_regen.urdf")
    out = args.out or os.path.join(XHAND_DIR, "urdf", f"xhand_{args.variant}.urdf")
    generate(args.variant, out, args.src)

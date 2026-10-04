from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import trimesh
import warp as wp


def _cache_path(mesh_path: str, extent: float, N: int, mesh_scale: float) -> Path:
    mesh_path_p = Path(mesh_path)
    key = hashlib.md5(
        f"{mesh_path_p.name}|method=v2_analytic|e={extent}|n={N}|s={mesh_scale}".encode()
    ).hexdigest()[:12]
    return mesh_path_p.parent / ".sdf_cache" / f"sdf_v2_e{extent}_n{N}_s{mesh_scale}_{key}.pt"


def _preprocess_mesh(mesh_path: str, mesh_scale: float, max_faces: int = 16000) -> trimesh.Trimesh:
    """Load, scale and decimate to max_faces (detail below a voxel only slows closest_point)."""
    mesh = trimesh.load(Path(mesh_path), process=False, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"SDF bake expected a single trimesh.Trimesh, got {type(mesh)} from {mesh_path!r}")
    if mesh_scale != 1.0:
        mesh.apply_scale(mesh_scale)
    if len(mesh.faces) > max_faces:
        decim = mesh.simplify_quadric_decimation(face_count=max_faces)
        if isinstance(decim, trimesh.Trimesh) and len(decim.faces) > 0:
            mesh = decim
    return mesh


@wp.kernel
def _closest_point(mesh: wp.uint64, pts: wp.array(dtype=wp.vec3), max_dist: float,
                   closest: wp.array(dtype=wp.vec3), sign: wp.array(dtype=float)):
    i = wp.tid()
    q = wp.mesh_query_point_sign_winding_number(mesh, pts[i], max_dist)
    closest[i] = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
    sign[i] = q.sign


def _build_object_sdf(mesh: trimesh.Trimesh, extent: float, N: int, device: str = "cpu") -> np.ndarray:
    """BVH closest-point queries (warp): seconds and < 1 GB for 128^3, where trimesh.proximity took hours
    and ~17 GB (it expands every candidate face of every grid point far from the mesh)."""
    xs = np.linspace(-extent, extent, N, dtype=np.float32)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float32)

    wp.init()
    wmesh = wp.Mesh(points=wp.array(mesh.vertices.astype(np.float32), dtype=wp.vec3, device=device),
                    indices=wp.array(mesh.faces.astype(np.int32).reshape(-1), dtype=wp.int32, device=device),
                    support_winding_number=True)
    closest_wp = wp.empty(len(pts), dtype=wp.vec3, device=device)
    sign_wp = wp.empty(len(pts), dtype=float, device=device)
    wp.launch(_closest_point, dim=len(pts), device=device,
              inputs=[wmesh.id, wp.array(pts, dtype=wp.vec3, device=device), 1.0e6, closest_wp, sign_wp])
    closest, sign = closest_wp.numpy(), sign_wp.numpy().astype(np.float32)  # sign: -1 inside (winding number)

    disp = (pts - closest).astype(np.float32)
    dist = np.linalg.norm(disp, axis=-1)
    sdf = (sign * dist).astype(np.float32).reshape(N, N, N)
    unit = disp / np.maximum(dist[:, None], 1e-8)
    grad = (sign[:, None] * unit).astype(np.float32).reshape(N, N, N, 3).transpose(3, 0, 1, 2)
    return np.concatenate([sdf[None], grad], axis=0)


def bake_object_sdf_grid(mesh_path: str, mesh_scale: float, extent: float, N: int, device: str,
                         bake_device: str | None = None) -> torch.Tensor:
    """(4, N, N, N) grid over [-extent, extent]^3 in the object frame: signed distance (negative inside)
    and its gradient, cached in <mesh dir>/.sdf_cache/; built on bake_device (default: device)."""
    cache_file = _cache_path(mesh_path, extent, N, mesh_scale)
    if cache_file.exists():
        return torch.load(cache_file, map_location=device, weights_only=True).to(device=device, dtype=torch.float32)
    grid = _build_object_sdf(_preprocess_mesh(mesh_path, mesh_scale), extent, N, str(bake_device or device))
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.from_numpy(grid), cache_file)
    return torch.from_numpy(grid).to(device=device, dtype=torch.float32)

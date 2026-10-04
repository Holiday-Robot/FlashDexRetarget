from __future__ import annotations

import atexit
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import trimesh
from omegaconf import OmegaConf

_CUACD = None


def _done(convex_dir: Path) -> bool:
    return convex_dir.is_dir() and any(p.suffix == ".obj" and p.stem.isdigit() for p in convex_dir.iterdir())


def _load(obj: Path, decimate_faces: int) -> trimesh.Trimesh:
    mesh = trimesh.load(obj / "visual.obj", process=False, skip_materials=True, force="mesh")
    if decimate_faces and len(mesh.faces) > decimate_faces:  # dense scans make CoACD very slow
        mesh = mesh.simplify_quadric_decimation(face_count=decimate_faces)
    return mesh


def _write(obj: Path, parts: list[trimesh.Trimesh]) -> None:
    out = obj / "convex"
    out.mkdir(exist_ok=True)
    for old in out.glob("*.obj"):
        old.unlink()
    for i, part in enumerate(parts):
        part.export(out / f"{i}.obj")


def decompose(obj_dir: str, coacd_kwargs: dict, decimate_faces: int = 0, seed: int = 1) -> tuple[str, int, float]:
    import coacd

    t0 = time.time()
    obj = Path(obj_dir)
    mesh = _load(obj, decimate_faces)
    coacd.set_log_level("warn")
    parts = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), seed=seed, **coacd_kwargs)
    _write(obj, [trimesh.Trimesh(vs, np.asarray(fs, dtype=int)) for vs, fs in parts])
    return obj.name, len(parts), time.time() - t0


def _cuacd_init(counter, gpus: list[str]) -> None:
    global _CUACD
    with counter.get_lock():
        i = counter.value
        counter.value += 1
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus[i % len(gpus)]
    _CUACD = _cuacd_context()


def _cuacd_context():
    import cuacd

    ctx = cuacd.Context()
    atexit.register(ctx.close)  # before interpreter teardown, where its __del__ fails
    return ctx


def _cuacd_hulls(mesh: trimesh.Trimesh, cfg) -> list[np.ndarray]:
    verts, tris = mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32)
    if _CUACD.check_mesh(verts, tris)["needs_remesh"]:
        verts, tris = _CUACD.preprocess(verts, tris, resolution=cfg.preprocess_resolution)
    lo, hi = verts.min(0), verts.max(0)
    c, s = (lo + hi) / 2, float((hi - lo).max()) / 2  # CuACD works in a ~2-unit box
    out = _CUACD.lookahead_decompose(((verts - c) / s).astype(np.float32), tris, threshold=cfg.threshold,
                                     depth=cfg.depth, decompose_components=True, merge_hulls=True)
    return [hv * s + c for _, _, hv, _ in out if len(hv) >= 4]


def decompose_cuacd(obj_dir: str, cfg, decimate_faces: int = 0) -> tuple[str, int, float]:
    """CuACD on this worker's GPU, then CoACD's merge (convex_merge.py)."""
    global _CUACD
    from retarget.object.convex_merge import merge_parts

    t0 = time.time()
    obj = Path(obj_dir)
    mesh = _load(obj, decimate_faces)
    try:
        hulls = _cuacd_hulls(mesh, cfg)
    except RuntimeError:  # a GPU error leaves the context unusable: retry once on a fresh one, as cuacd's CLI
        _CUACD.close()
        _CUACD = _cuacd_context()
        hulls = _cuacd_hulls(mesh, cfg)
    parts = merge_parts(hulls, cfg.threshold, cfg.max_convex_hull)
    _write(obj, [trimesh.convex.convex_hull(p) for p in parts])
    return obj.name, len(parts), time.time() - t0


def decompose_objects(pool: str | Path, ids: list[str], cfg, workers: int) -> None:
    """cfg: convex of config/retarget/kinematic_retargeting.yaml + config/retarget/convex/<name>.yaml."""
    root = Path(pool) / "objects"
    todo = [root / i for i in ids if cfg.force or not _done(root / i / "convex")]
    missing = [str(p) for p in todo if not (p / "visual.obj").exists()]
    if missing:
        raise FileNotFoundError(f"visual.obj missing for {len(missing)} objects, e.g. {missing[0]}")
    if cfg.name == "coacd":
        where = f"{workers} CPU workers"
    elif cfg.name == "cuacd":
        from retarget.object.obj_sdf_bake import visible_gpus

        visible = visible_gpus()
        gpus = [visible[g] for g in cfg.gpus] if cfg.gpus is not None else visible
        if not gpus:
            raise RuntimeError("convex=cuacd but no GPU is visible; run with convex=coacd")
        workers, where = len(gpus), f"GPU {','.join(gpus)}"
    else:
        raise ValueError(f"convex.name must be coacd or cuacd, got {cfg.name!r}")
    print(f"[obj_convex_decompose] {len(todo)} of {len(ids)} objects need convex parts ({cfg.name}, {where})",
          flush=True)
    if not todo:
        return
    if cfg.name == "coacd":
        kwargs = OmegaConf.to_container(cfg.coacd)
        ex = ProcessPoolExecutor(max_workers=min(workers, len(todo)))
        futures = [ex.submit(decompose, str(p), kwargs, cfg.decimate_faces, cfg.seed) for p in todo]
    else:
        ctx = mp.get_context("spawn")
        ex = ProcessPoolExecutor(max_workers=min(workers, len(todo)), mp_context=ctx, initializer=_cuacd_init,
                                 initargs=(ctx.Value("i", 0), gpus))
        futures = [ex.submit(decompose_cuacd, str(p), cfg, cfg.decimate_faces) for p in todo]
    with ex:
        for i, fut in enumerate(as_completed(futures), 1):
            name, n, sec = fut.result()
            print(f"[obj_convex_decompose] {i}/{len(todo)} {name}: {n} parts, {sec:.0f} s", flush=True)

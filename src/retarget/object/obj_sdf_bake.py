from __future__ import annotations

import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from retarget.build_motion_pt import object_meshes

_DEVICE = "cpu"


def visible_gpus() -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    return visible.split(",") if visible else [str(g) for g in range(torch.cuda.device_count())]


def _init_worker(counter, gpus: list[str], device: str) -> None:
    global _DEVICE
    with counter.get_lock():
        i = counter.value
        counter.value += 1
    # one visible GPU per worker: warp init over every visible GPU takes ~20 s with 8 workers
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus[i % len(gpus)] if gpus else ""
    _DEVICE = "cuda:0" if device == "cuda" else "cpu"
    torch.set_num_threads(1)


def _bake(mesh: str, scale: float, extent: float, n: int) -> tuple[str, float]:
    from utils.sdf import bake_object_sdf_grid

    t0 = time.time()
    bake_object_sdf_grid(mesh, scale, extent, n, device="cpu", bake_device=_DEVICE)
    return Path(mesh).parent.name, time.time() - t0


def bake(motion: Path, cfg, workers: int) -> None:
    """cfg: sdf of config/retarget/kinematic_retargeting.yaml."""
    from utils.sdf import _cache_path

    packed = torch.load(motion, map_location="cpu", weights_only=False)
    meshes = object_meshes(packed, Path(motion).parent)
    missing = [str(m) for m, _ in meshes if not m.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} meshes missing, e.g. {missing[0]}")
    todo = [(m, s) for m, s in meshes if not _cache_path(str(m), cfg.extent, cfg.grid_n, s).exists()]
    visible = visible_gpus()
    if cfg.device == "cuda":
        gpus = [visible[g] for g in cfg.gpus] if cfg.gpus is not None else visible
        if not gpus:
            raise RuntimeError("sdf.device=cuda but no GPU is visible; run with sdf.device=cpu")
        workers = cfg.workers_per_gpu * len(gpus)
    elif cfg.device == "cpu":
        gpus = visible[:1]  # warp still initializes CUDA
    else:
        raise ValueError(f"sdf.device must be cuda or cpu, got {cfg.device!r}")
    print(f"[obj_sdf_bake] {len(todo)} of {len(meshes)} meshes need a grid, {cfg.grid_n}^3 over +-{cfg.extent} m, "
          f"{workers} workers on {'GPU ' + ','.join(gpus) if cfg.device == 'cuda' else 'CPU'}", flush=True)
    if not todo:
        return
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(workers, len(todo)), mp_context=ctx, initializer=_init_worker,
                             initargs=(ctx.Value("i", 0), gpus, cfg.device)) as ex:
        futures = [ex.submit(_bake, str(m), s, cfg.extent, cfg.grid_n) for m, s in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            name, sec = fut.result()
            print(f"[obj_sdf_bake] {i}/{len(todo)} {name} {sec:.1f} s", flush=True)

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
import torch
from omegaconf import DictConfig

from retarget.build_motion_pt import build, object_meshes
from retarget.human_demo.clips import clip_fps, list_clips, object_ids
from retarget.human_demo.preprocess import preprocess_clips
from retarget.object.obj_convex_decompose import decompose_objects
from retarget.object.obj_sdf_bake import bake
from retarget.robot_hand.hand_retarget import retarget_clips


def _log(msg: str) -> None:
    print(f"[kinematic_retargeting] {msg}", flush=True)


def _write_report(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["clip"]))


@hydra.main(config_path="../../config/retarget", config_name="kinematic_retargeting", version_base=None)
def main(cfg: DictConfig) -> None:
    workers = cfg.workers or max(1, (os.cpu_count() or 2) // 2)
    steps = cfg.steps
    dataset = Path(cfg.dataset)
    demo, retargeted = dataset / "human_demo", dataset / "retargeted"
    motion, report = dataset / "motion.pt", dataset / "report.csv"
    from_clips = demo.is_dir()
    if from_clips:
        fps = clip_fps(list_clips(demo))
        _log(f"preprocess: {demo} ({fps:g} Hz)")
        rows, kept = preprocess_clips(demo, dataset, cfg, fps, workers)
        _write_report(report, rows)
        if not kept:
            raise RuntimeError(f"every clip was rejected; see {report}")
        ids = object_ids(kept)
    elif motion.exists():
        _log(f"no {demo}: preparing the objects of {motion}")
        packed = torch.load(motion, map_location="cpu", weights_only=False)
        ids = sorted({mesh.parent.name for mesh, _ in object_meshes(packed, dataset)})
    else:
        raise FileNotFoundError(f"{dataset} has neither human_demo/ nor motion.pt")

    if steps.convex:
        _log(f"convex: {len(ids)} objects (<object>/convex/, {cfg.convex.name})")
        decompose_objects(dataset, ids, cfg.convex, workers)
    if from_clips:
        if steps.retarget:
            _log(f"retarget: {len(kept)} clips -> {retargeted} ({cfg.robot})")
            ik = {r["clip"]: r for r in retarget_clips(list(kept.items()), dataset, retargeted, cfg.robot,
                                                       cfg.retarget, fps, workers)}
            for r in rows:
                r.update(ik.get(r["clip"], {}))
            _write_report(report, rows)
            if all(r.get("error") for r in ik.values()):
                raise RuntimeError(f"every clip failed to retarget; see {report}")
        if steps.pack:
            _log(f"pack: {retargeted} -> {motion}")
            build(retargeted, dataset, fps, cfg.pack, names={p.stem for p in kept})
    if steps.usd:
        _log("usd: <object>/.isaac_usd/object.usd")
        # One Kit process on one GPU: the conversion is CPU work, and Kit otherwise spans every visible GPU.
        env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="1", MUJOCO_GL=os.environ.get("MUJOCO_GL", "disabled"),
                   CUDA_VISIBLE_DEVICES=str(cfg.usd_gpu))
        subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "simulator/isaacsim/convert_assets.py"),
                        "--data-root", str(dataset.resolve()), "--objects", *ids], env=env, check=True)
    if steps.sdf and not motion.exists():
        _log(f"sdf: skipped, no {motion} yet")
    elif steps.sdf:
        _log(f"sdf: <object>/.sdf_cache/, {cfg.sdf.grid_n}^3")
        bake(motion, cfg.sdf, workers)
    _log(f"done: train with MOTION={motion}" if motion.exists() else "done")


if __name__ == "__main__":
    main()

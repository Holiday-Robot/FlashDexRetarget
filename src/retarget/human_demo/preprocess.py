from __future__ import annotations

import functools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from retarget.human_demo import clips as C
from retarget.human_demo.desk import support_disks
from retarget.human_demo.filters import features, reject_reasons


@functools.cache
def _meshes(pool: str) -> C.ObjectMeshes:
    return C.ObjectMeshes(pool)


def _process(path: str, pool: str, cfg, fps: float) -> tuple[dict, np.ndarray | None]:
    meshes = _meshes(pool)
    path = Path(path)
    row: dict = {"clip": path.stem}
    disks = None
    try:
        clip = C.load_clip(path)
        desk, filt = cfg.desk.mode, cfg.filters.enable
        if desk == "fit" or (desk == "keep" and "support_disks" not in clip):
            clip["support_disks"] = support_disks(clip, meshes, cfg.desk, fps)
        if filt:
            row.update(features(clip, meshes, cfg.filters, fps))
        reasons = reject_reasons(row, cfg.filters) if filt else []
        row["disks"] = len(np.asarray(clip.get("support_disks", np.zeros((0, 5)))).reshape(-1, 5))
        row["reject"] = ";".join(reasons)
        disks = clip.get("support_disks")
    except Exception as e:  # noqa: BLE001 - one bad clip must not stop the batch
        row["reject"] = f"error: {e!r}"
    return row, disks


def preprocess_clips(clips: Path, pool: Path, cfg, fps: float, workers: int
                     ) -> tuple[list[dict], dict[Path, np.ndarray | None]]:
    """Report rows of every clip, and the kept clips with their support disks (cfg: desk, filters)."""
    if cfg.desk.mode not in ("fit", "keep", "none"):
        raise ValueError(f"desk.mode must be fit, keep or none, not {cfg.desk.mode!r}")
    paths = C.list_clips(clips)
    print(f"[preprocess] {len(paths)} clips, desk={cfg.desk.mode}, filter={cfg.filters.enable}", flush=True)
    jobs = [(str(p), str(pool), cfg, fps) for p in paths]
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(jobs)))) as ex:
        results = list(ex.map(_process, *zip(*jobs))) if jobs else []

    rows = [r for r, _ in results]
    kept = {p: disks for p, (r, disks) in zip(paths, results) if not r["reject"]}
    counts: dict[str, int] = {}
    for r in rows:
        for reason in filter(None, r["reject"].split(";")):
            key = "error" if reason.startswith("error") else reason
            counts[key] = counts.get(key, 0) + 1
    print(f"[preprocess] kept {len(kept)} / {len(rows)}  {counts}")
    return rows, kept

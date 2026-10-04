from __future__ import annotations

import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np

from retarget.human_demo import clips as C
from retarget.robot_hand.ik import attempt, best_run
from retarget.robot_hand.quality import flags


def _crop(clip: dict, n: int) -> dict:
    """Drop the first n frames of every per-frame array."""
    T = C.num_frames(clip)
    out = {}
    for k, v in clip.items():
        per_frame = isinstance(v, np.ndarray) and v.ndim > 0 and len(v) == T
        out[k] = v[n:] if per_frame and not k.endswith(("support_disks", "_joint_names")) else v
    return out


def _load(path: str, disks) -> dict:
    clip = C.load_clip(path)
    if disks is not None:
        clip["support_disks"] = disks
    return clip


def _needs_ik(disks, out: Path, cfg) -> bool:
    """No cached result for these support disks."""
    return not (out.exists() and not cfg.force and np.array_equal(C.load_clip(out).get("support_disks"), disks))


def _penetration(clip: dict, run: tuple, robot: str, pool: Path, cfg, fps: float) -> float:
    """hand_obj_pen_max_m of one IK run, as report.csv scores it."""
    res = best_run([run], cfg)
    clip = _crop(clip, res["crop"])
    clip["qpos"] = res["qpos"]
    return flags(clip, robot, pool, cfg, fps)["hand_obj_pen_max_m"]


def _attempt(path: str, disks, pool: str, robot: str, cfg, fps: float, k: int):
    t0 = time.time()
    try:
        clip = _load(path, disks)
        run = attempt(clip, robot, Path(pool), cfg, fps, Path(path).stem, k)
        return (*run, _penetration(clip, run, robot, Path(pool), cfg, fps)), time.time() - t0
    except Exception as e:  # noqa: BLE001 - one bad clip must not stop the batch
        return repr(e), time.time() - t0


def _finish(path: str, disks, pool: str, out_dir: str, robot: str, cfg, fps: float, runs, seconds: float) -> dict:
    """Save the best IK run (runs None: a cached clip) and replay it for the quality flags."""
    path, out = Path(path), Path(out_dir) / Path(path).name
    row: dict = {"clip": path.stem}
    t0 = time.time()
    try:
        if runs is None:
            clip, row["source"] = C.load_clip(out), "cached"
        else:
            clip = _load(str(path), disks)
            res = best_run(runs, cfg)
            clip = _crop(clip, res["crop"])
            clip["qpos"] = res["qpos"]
            clip["ik_tip_err_mean_m"] = np.float64(res["tip_error_mean_m"])
            clip["ik_restarts"] = np.int64(len(res["log"]) - 1)
            row["source"] = "ik"
            C.save_clip(out, clip)
        if "ik_restarts" in clip:
            row.update(ik_tip_err_mean_m=float(clip["ik_tip_err_mean_m"]), ik_restarts=int(clip["ik_restarts"]))
        row.update(flags(clip, robot, Path(pool), cfg, fps))
    except Exception as e:  # noqa: BLE001
        row["error"] = repr(e)
    row["retarget_seconds"] = round(seconds + time.time() - t0, 1)
    return row


def retarget_clips(clips: list[tuple[Path, np.ndarray | None]], pool: Path, out: Path, robot: str, cfg,
                   fps: float, workers: int) -> list[dict]:
    """IK of (clip file, support disks) pairs into out, a cache: a clip whose desk is unchanged is not redone
    unless cfg.force (cfg: retarget). Every IK run of every clip (cfg.attempts per clip) is a job of its own, so
    a long clip takes one run's time. Returns the report rows; retarget_seconds sums a clip's runs."""
    out.mkdir(parents=True, exist_ok=True)
    print(f"[hand_retarget] {len(clips)} clips to {robot}", flush=True)
    args = (str(pool), robot, cfg, fps)
    rows, runs, spent, jobs = [], {}, {}, {}
    with ProcessPoolExecutor(max_workers=max(1, workers)) as ex:
        for p, d in clips:
            if _needs_ik(d, out / p.name, cfg):
                runs[p], spent[p] = [None] * cfg.attempts, 0.0
                for k in range(cfg.attempts):
                    jobs[ex.submit(_attempt, str(p), d, *args, k)] = (p, d, k)
            else:
                jobs[ex.submit(_finish, str(p), d, str(pool), str(out), robot, cfg, fps, None, 0.0)] = (p, d, None)
        while jobs:
            done, _ = wait(jobs, return_when=FIRST_COMPLETED)
            for fut in done:
                p, d, k = jobs.pop(fut)
                if k is None:
                    r = fut.result()
                    rows.append(r)
                    msg = r.get("error") or (f"{r['source']}, tip {r.get('ik_tip_err_mean_m', float('nan')) * 1e3:.1f} mm "
                                             f"{r.get('flags') or 'ok'}")
                    print(f"[hand_retarget] {len(rows)}/{len(clips)} {r['clip']} ({r['retarget_seconds']} s): {msg}",
                          flush=True)
                    continue
                runs[p][k], seconds = fut.result()
                spent[p] += seconds
                if any(r is None for r in runs[p]):
                    continue
                finished = runs.pop(p)
                errors = [r for r in finished if isinstance(r, str)]
                if errors:
                    rows.append({"clip": p.stem, "error": errors[0], "retarget_seconds": round(spent[p], 1)})
                    print(f"[hand_retarget] {len(rows)}/{len(clips)} {p.stem}: {errors[0]}", flush=True)
                    continue
                jobs[ex.submit(_finish, str(p), d, str(pool), str(out), robot, cfg, fps, finished,
                               spent[p])] = (p, d, None)
    errors = sum(1 for r in rows if r.get("error"))
    flagged = sum(1 for r in rows if r.get("flags"))
    print(f"[hand_retarget] {len(rows) - errors} clips in {out} ({errors} errors, {flagged} flagged)")
    return rows

"""Archive the eval clips that pass the success criterion: one npz rollout per motion plus
manifest.json. A motion keeps the first clip archived for it."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomic rewrite, so a killed job cannot leave a truncated manifest."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def export_success(
    out_dir: str | Path,
    rec_cb: Any,
    rec_states: list[dict[str, Any]],
    ok: torch.Tensor,
    motion_ids: torch.Tensor,
    env_step: int,
    criterion: str,
    motion_names: list[str] | None = None,
    motion_file: str | None = None,
) -> int:
    """Archive the first passing env row of each motion not archived yet; returns the count.
    ``ok`` and ``motion_ids`` are per env over the concatenated eval sweep."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    entries: dict[str, dict[str, Any]] = {}
    if manifest_path.exists():
        try:
            prev = json.loads(manifest_path.read_text())
            entries = {str(e["motion_id"]): e for e in prev.get("saved", [])}
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            print(f"[success] unreadable manifest, starting fresh: {exc}")

    written = 0
    for row in torch.nonzero(ok, as_tuple=False).flatten().tolist():
        mid = int(motion_ids[row].item())
        if str(mid) in entries:
            continue
        arrays = rec_cb.rollout_arrays(rec_states, row)
        if arrays is None:
            continue
        rel = f"rollouts/m{mid:05d}.npz"
        (out_dir / "rollouts").mkdir(exist_ok=True)
        np.savez_compressed(out_dir / rel, **{k: np.asarray(v) for k, v in arrays.items()})
        entries[str(mid)] = {
            "motion_id": mid,
            "name": motion_names[mid] if motion_names is not None and mid < len(motion_names) else None,
            "frames": int(arrays["joint_pos"].shape[0]),
            "rollout": rel,
            "first_saved_env_step": int(env_step),
        }
        written += 1
    if not written:
        return 0

    saved = sorted(entries.values(), key=lambda e: int(e["motion_id"]))
    _write_json(
        manifest_path,
        {
            "criterion": criterion,
            "motion_file": motion_file,
            "num_saved": len(saved),
            "updated_at_env_step": int(env_step),
            "coordinate_frame": "env-local (scene env_origins subtracted)",
            "fields": "joint_pos, joint_vel, actions, root_pos, root_quat, obj_pos, obj_quat",
            "saved": saved,
        },
    )
    print(f"[success] archived {written} new clip(s); {len(saved)} motions in {out_dir}", flush=True)
    return written

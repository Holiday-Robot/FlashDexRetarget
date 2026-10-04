from __future__ import annotations

import ast
import json
import pickle

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation

from retarget.human_demo.sources.base import YUP_TO_ZUP, Demo, Hand, Source as Base, _forward, _mano


class Source(Base):
    """OakInk2: anno_preview/<seq>.pkl (raw_mano and obj_transf per mocap frame, 120 Hz, y-up) split into the
    stages of program/program_info/<seq>.json (clip <seq>__stNN); meshes object_repair/align_ds/<id>/model.obj."""

    mano = {"use_pca": False, "flat_hand_mean": True}

    def __init__(self, raw, cfg, mano_dir: str | None = None) -> None:
        super().__init__(raw, cfg, mano_dir)
        self.scales = {str(k): float(v) for k, v in (cfg.get("scales") or {}).items()}

    def units(self) -> list[str]:
        programs = self.raw / "program" / "program_info"
        return sorted(p.stem for p in (self.raw / "anno_preview").glob("*.pkl") if (programs / f"{p.stem}.json").exists())

    def demos(self, seq: str) -> list[Demo | str]:
        anno = pickle.load(open(self.raw / "anno_preview" / f"{seq}.pkl", "rb"))
        program = json.load(open(self.raw / "program" / "program_info" / f"{seq}.json"))
        out: list[Demo | str] = []
        # The desk is fitted over the whole recording, as CHORD does: every object with a mesh, every frame.
        context = {o: np.stack([anno["obj_transf"][o][f] for f in anno["mocap_frame_id_list"]]).astype(np.float64)
                   for o in anno["obj_transf"] if self._mesh_path(o).exists()}
        for i, (key, stage) in enumerate(program.items()):
            name = f"{seq}__st{i:02d}"
            spans = [s for s in ast.literal_eval(key) if s is not None]
            lists = {"right": stage.get("obj_list_rh") or [], "left": stage.get("obj_list_lh") or []}
            if not spans:
                out.append(f"{name}: no stage interval")
                continue
            if any(len(v) > 1 for v in lists.values()):
                out.append(f"{name}: several objects in one hand")
                continue
            start, end = min(s[0] for s in spans), max(s[1] for s in spans)
            frames = [f for f in anno["mocap_frame_id_list"] if start <= f < end]
            hands = {side: self._hand(anno, frames, pre) for side, pre in (("right", "rh"), ("left", "lh"))}
            objects = self._held(hands, anno, frames, context, lists)
            poses = {o: np.stack([anno["obj_transf"][o][f] for f in frames]).astype(np.float64)
                     for o in set(objects.values())}
            out.append(Demo(name, 120.0, hands, objects, poses, to_world=YUP_TO_ZUP,
                            scales={o: self.scales[o] for o in {*poses, *context} if o in self.scales}, context=context))
        return out

    def _held(self, hands: dict[str, Hand], anno: dict, frames: list[int], candidates, lists: dict) -> dict[str, str]:
        """Hand side -> the object its 16 MANO links touch (a vertex within source.contact) the most link-frames,
        as CHORD assigns them (the program annotation can name the other hand's object); the annotation's when
        it touches none. Counted on the unscaled meshes, so source.scales never changes which object a hand holds."""
        out = {}
        for side, hand in hands.items():
            model = _mano(self.mano_dir, side, tuple(sorted(self.mano.items())))
            _, verts = _forward(model, hand)
            link_of = model.lbs_weights.numpy().argmax(1)
            best, most = (lists[side] or [None])[0], 0
            for o in candidates:
                pose = np.stack([anno["obj_transf"][o][f] for f in frames])
                local = np.matmul(verts - pose[:, None, :3, 3], pose[:, :3, :3])
                near = self.near(o, local.reshape(-1, 3), self.cfg.contact)
                near = near.reshape(verts.shape[:2]) < self.cfg.contact
                n = sum(int(near[:, link_of == k].any(1).sum()) for k in range(16))
                if n > most:
                    best, most = o, n
            if best is not None:
                out[side] = best
        return out

    @staticmethod
    def _hand(anno: dict, frames: list[int], pre: str) -> Hand:
        mano = [anno["raw_mano"][f] for f in frames]
        quat = torch.cat([m[f"{pre}__pose_coeffs"] for m in mano]).numpy().astype(np.float64)
        rotvec = Rotation.from_quat(quat.reshape(-1, 4)[:, [1, 2, 3, 0]]).as_rotvec().reshape(len(frames), 16, 3)
        return Hand(rotvec[:, 0], rotvec[:, 1:].reshape(len(frames), 45),
                    torch.cat([m[f"{pre}__betas"] for m in mano]).numpy().astype(np.float64),
                    wrist=torch.cat([m[f"{pre}__tsl"] for m in mano]).numpy().astype(np.float64))

    def _mesh_path(self, obj_id: str):
        return self.raw / "object_repair" / "align_ds" / obj_id / "model.obj"

    def load_mesh(self, obj_id: str) -> trimesh.Trimesh:
        return trimesh.load(self._mesh_path(obj_id), force="mesh", process=False, skip_materials=True)

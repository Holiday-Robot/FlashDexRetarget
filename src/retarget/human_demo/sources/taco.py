from __future__ import annotations

import pickle

import numpy as np
import trimesh

from retarget.human_demo.sources.base import Demo, Hand, Source as Base, assign


class Source(Base):
    """TACO: Hand_Poses/<(action, tool, target)>/<seq>/{side}_hand{,_shape}.pkl and Object_Poses/.../{tool,target}_<id>.npy
    at 30 Hz, z-up; each hand takes the object it stays nearest to; meshes object_models_released/<id>_cm.obj."""

    mano = {"use_pca": False, "flat_hand_mean": True}

    def units(self) -> list[str]:
        root = self.raw / "Hand_Poses"
        return sorted(f"{d.parent.name}/{d.name}" for d in root.glob("*/*") if d.is_dir())

    def demos(self, unit: str) -> list[Demo | str]:
        triplet, seq = unit.split("/")
        name = "taco_" + "__".join(w.replace(" ", "_") for w in triplet.strip("()").split(", ")) + f"_{seq}"
        hand_dir, pose_dir = self.raw / "Hand_Poses" / unit, self.raw / "Object_Poses" / unit
        poses = {}
        for kind in ("tool", "target"):
            files = sorted(pose_dir.glob(f"{kind}_*.npy"))
            if len(files) == 1:
                poses[files[0].stem.split("_", 1)[1]] = np.load(files[0]).astype(np.float64)
        hands = {}
        for side in ("right", "left"):
            if (hand_dir / f"{side}_hand.pkl").exists() and (hand_dir / f"{side}_hand_shape.pkl").exists():
                frames = pickle.load(open(hand_dir / f"{side}_hand.pkl", "rb"))
                keys = sorted(frames)
                pose = np.stack([frames[k]["hand_pose"].numpy() for k in keys]).astype(np.float64)
                betas = pickle.load(open(hand_dir / f"{side}_hand_shape.pkl", "rb"))["hand_shape"].numpy()
                hands[side] = Hand(pose[:, :3], pose[:, 3:], betas.astype(np.float64),
                                   wrist=np.stack([frames[k]["hand_trans"].numpy() for k in keys]).astype(np.float64))
        if len({len(p) for p in poses.values()} | {len(h.wrist) for h in hands.values()}) > 1:
            return [f"{name}: frame counts differ"]
        centres = {o: p[:, :3, :3] @ self.mesh(o).vertices.mean(0) + p[:, :3, 3] for o, p in poses.items()}
        objects = assign({s: h.wrist for s, h in hands.items()}, centres)
        return [Demo(name, 30.0, hands, objects, poses, to_world=np.diag([-1.0, -1.0, 1.0]))]

    def load_mesh(self, obj_id: str) -> trimesh.Trimesh:
        mesh = trimesh.load(self.raw / "object_models_released" / f"{obj_id}_cm.obj", force="mesh", process=False,
                            skip_materials=True)
        mesh.apply_scale(0.01)
        return mesh

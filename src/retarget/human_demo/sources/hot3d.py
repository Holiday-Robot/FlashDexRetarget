from __future__ import annotations

import csv
import json
from collections import Counter

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from retarget.human_demo import clips as C
from retarget.human_demo.sources.base import Demo, Hand, Source as Base, _forward, _mano


class Source(Base):
    """HOT3D recordings: dataset/<recording>/{mano_hand_pose_trajectory.jsonl, dynamic_objects.csv,
    headset_trajectory.csv, metadata.json}, meshes _assets/assets/<uid>.glb. Read as CHORD's hot3d_loader.py
    (one timestamp per 10 ms, 30 Hz; frames with both hands and every object; Quest3 y-up turned z-up; yaw so
    the workspace lies along +y from the headset) and cut into hand-object segments as its segment_to_atomic.py
    (clip <recording>_seg<NNN>, its numbering); segments of source.segment.modes are kept."""

    mano = {"use_pca": True, "num_pca_comps": 15, "flat_hand_mean": False}

    def units(self) -> list[str]:
        root = self.raw / "dataset"
        return sorted(p.name for p in root.iterdir() if (p / "mano_hand_pose_trajectory.jsonl").exists()
                      and json.load(open(p / "metadata.json")).get("have_hand_object_pose_gt"))

    def demos(self, unit: str) -> list[Demo | str]:
        rec = self.raw / "dataset" / unit
        meta = json.load(open(rec / "metadata.json"))
        uids = [str(u) for u in meta["object_uids"]]
        hand_rows, obj_rows = self._hand_rows(rec), self._object_rows(rec)
        stamps, last = [], None
        for t in sorted(hand_rows):  # the camera streams' timestamps of one frame are < 10 ms apart
            if last is None or t - last >= 10_000_000:
                last = t
                if {"0", "1"} <= set(hand_rows[t]) and all(u in obj_rows.get(t, {}) for u in uids):
                    stamps.append(t)
        if not stamps:
            return [f"{unit}: no frame with both hands and every object"]
        hands = {side: self._hand([hand_rows[t][key] for t in stamps]) for side, key in (("right", "1"), ("left", "0"))}
        poses = {u: np.stack([obj_rows[t][u] for t in stamps]) for u in uids}
        up = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64) if meta["headset"] == "Quest3" else np.eye(3)
        to_world = self._yaw(rec, up, poses, hands) @ up

        cfg = self.cfg.segment
        out: list[Demo | str] = []
        for i, (a, b, mode, obj) in enumerate(self._segments(hands, poses, uids)):
            name = f"{unit}_seg{i:03d}"
            if mode not in cfg.modes:
                out.append(f"{name}: mode {mode}")
                continue
            if obj is None:
                out.append(f"{name}: no object in contact")
                continue
            idx = np.r_[[a] * cfg.pad_frames, np.arange(a, b), [b - 1] * cfg.pad_frames]  # still frames at both ends
            seg_hands = {s: Hand(h.global_orient[idx], h.hand_pose[idx], h.betas, transl=h.transl[idx])
                         for s, h in hands.items()}
            seg_poses = {u: p[idx] for u, p in poses.items()}
            out.append(Demo(name, 30.0, seg_hands, {s: obj for s in seg_hands}, {obj: seg_poses[obj]},
                            to_world=to_world, scene={u: p for u, p in seg_poses.items() if u != obj},
                            context=poses))
        return out

    def _segments(self, hands: dict[str, Hand], poses: dict[str, np.ndarray], uids: list[str]) -> list[tuple]:
        """(start, end, mode, object) per segment: a hand is active while a fingertip is within tip_contact of an
        object (gaps < gap_frames bridged); windows of activity split where the touched object changes; segments
        shorter than min_frames dropped. Mode A: one hand active, B: both on one object, C: on two objects."""
        cfg = self.cfg.segment
        active, links = {}, {}
        for side, hand in hands.items():
            model = _mano(self.mano_dir, side, tuple(sorted(self.mano.items())))
            _, verts = _forward(model, hand)
            link_of = model.lbs_weights.numpy().argmax(1)  # (778,) the MANO link each vertex follows
            tips = verts[:, list(C.TIP_VERTEX.values())]
            tip_dist = np.full(tips.shape[:2], np.inf)
            link_body = np.zeros((len(verts), 16), dtype=int)
            link_dist = np.full((len(verts), 16), np.inf)
            for k, u in enumerate(uids, 1):
                rot, pos = poses[u][:, :3, :3], poses[u][:, :3, 3]
                local = lambda x: np.matmul(x - pos[:, None], rot).reshape(-1, 3)  # noqa: E731
                tip_dist = np.minimum(tip_dist, self.near(u, local(tips), cfg.tip_contact).reshape(tip_dist.shape))
                d = self.near(u, local(verts), cfg.link_contact).reshape(verts.shape[:2])
                for link in range(16):
                    dl = d[:, link_of == link].min(1)
                    closer = dl < link_dist[:, link]
                    link_dist[closer, link], link_body[closer, link] = dl[closer], k
            active[side] = _fill_gaps(tip_dist.min(1) < cfg.tip_contact, cfg.gap_frames)
            links[side] = np.where(np.isfinite(link_dist), link_body, 0)

        both = np.concatenate([links["right"], links["left"]], axis=1)
        per_frame = np.array([Counter(r[r != 0].tolist()).most_common(1)[0][0] if (r != 0).any() else 0 for r in both])
        segments = []
        for start, end in _runs(active["right"] | active["left"]):
            for a, b in _split_by_object(start, end, per_frame, cfg.gap_frames):
                if b - a < cfg.min_frames:
                    continue
                body = {s: _majority(links[s][a:b][active[s][a:b]]) for s in hands}
                on = {s: active[s][a:b].any() for s in hands}
                if not (on["right"] and on["left"]):
                    mode = "A"
                elif None in body.values() or body["right"] == body["left"]:
                    mode = "B"
                else:
                    mode = "C"
                k = body["right"] or body["left"]
                segments.append((a, b, mode, uids[k - 1] if k else None))
        return segments

    @staticmethod
    def _yaw(rec, up: np.ndarray, poses: dict[str, np.ndarray], hands: dict[str, Hand]) -> np.ndarray:
        """Rotation about z putting the workspace (mean of every object and wrist position) along +y from the
        headset's first position, then turned 90 deg further (CHORD's convention)."""
        pts = [p[:, :3, 3] for p in poses.values()] + [h.transl for h in hands.values()]
        centre = (np.concatenate(pts) @ up.T).mean(0)[:2]
        with open(rec / "headset_trajectory.csv") as f:
            row = next(csv.DictReader(f))
        head = up @ np.array([float(row[f"t_wo_{a}[m]"]) for a in "xyz"])
        d = centre - head[:2]
        return Rotation.from_euler("z", np.pi - np.arctan2(d[1], d[0])).as_matrix()

    @staticmethod
    def _hand_rows(rec) -> dict[int, dict]:
        rows = {}
        with open(rec / "mano_hand_pose_trajectory.jsonl") as f:
            for line in f:
                e = json.loads(line)
                rows.setdefault(int(e["timestamp_ns"]), e["hand_poses"])
        return rows

    @staticmethod
    def _object_rows(rec) -> dict[int, dict[str, np.ndarray]]:
        rows: dict[int, dict[str, np.ndarray]] = {}
        with open(rec / "dynamic_objects.csv") as f:
            for r in csv.DictReader(f):
                pose = np.eye(4)
                pose[:3, :3] = Rotation.from_quat([float(r[f"q_wo_{a}"]) for a in "xyzw"]).as_matrix()
                pose[:3, 3] = [float(r[f"t_wo_{a}[m]"]) for a in "xyz"]
                rows.setdefault(int(r["timestamp[ns]"]), {})[r["object_uid"]] = pose
        return rows

    @staticmethod
    def _hand(frames: list[dict]) -> Hand:
        """MANO as 15 PCA components, the wrist rotation and smplx transl, the first frame's betas."""
        q = np.array([h["wrist_xform"]["q_wxyz"] for h in frames], dtype=np.float64)
        return Hand(Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_rotvec(),
                    np.array([h["pose"] for h in frames], dtype=np.float64),
                    np.array(frames[0]["betas"], dtype=np.float64)[:10],
                    transl=np.array([h["wrist_xform"]["t_xyz"] for h in frames], dtype=np.float64))

    def load_mesh(self, obj_id: str) -> trimesh.Trimesh:
        return trimesh.load(self.raw / "_assets" / "assets" / f"{obj_id}.glb", force="mesh", process=False)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]))


def _fill_gaps(mask: np.ndarray, gap: int) -> np.ndarray:
    """Bridge runs of False shorter than gap between two True frames."""
    out = mask.copy()
    for a, b in _runs(~mask):
        if b - a < gap and a > 0 and b < len(mask):
            out[a:b] = True
    return out


def _split_by_object(start: int, end: int, body: np.ndarray, gap: int) -> list[tuple[int, int]]:
    """[start, end) cut where the frame's touched object changes; no-contact gaps < gap inside one object kept."""
    w = body[start:end]
    out, i = [], 0
    while i < len(w):
        if w[i] == 0:
            i += 1
            continue
        cur, j = w[i], i + 1
        while j < len(w):
            if w[j] == cur:
                j += 1
            elif w[j] == 0:
                k = j
                while k < len(w) and w[k] == 0:
                    k += 1
                if k < len(w) and w[k] == cur and k - j < gap:
                    j = k + 1
                else:
                    break
            else:
                break
        out.append((start + i, start + j))
        i = j
    return out or [(start, end)]


def _majority(ids: np.ndarray) -> int | None:
    ids = ids[ids != 0]
    return int(Counter(ids.tolist()).most_common(1)[0][0]) if len(ids) else None
